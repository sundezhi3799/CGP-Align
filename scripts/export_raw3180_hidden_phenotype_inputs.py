from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import torch

import train_cgp_align_replicate as cgp
from cgp_align.checkpoint_paths import add_data_arguments, configure_inference
from cgp_align.checkpoint_io import load_checkpoint
from eval_cgp_align_crossmodal_bridge import namespace_from_config


DEFAULT_CHECKPOINT = Path(
    "output/cgp_align/ablation/raw3180_four_branch_cwcl_local_modality_seed41/"
    "checkpoints/raw3180_4branch_cwcl_sep_k32_w100_seed41/best_model.pt"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export final raw3180 held-out embeddings and entity-mean Cell Painting "
            "profiles for hidden phenotype-neighbour recovery."
        )
    )
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "output/cgp_align/paper/manuscript_cgp_align_profile/"
            "hidden_phenotype_neighbour_recovery_20260908/final_raw3180_inputs"
        ),
    )
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--max-compounds", type=int, default=0)
    p.add_argument("--max-genes", type=int, default=0)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=20260908)
    p.add_argument("--device", default="auto")
    add_data_arguments(p)
    return p.parse_args()


def json_default(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    raise TypeError(f"Unsupported JSON type: {type(x)!r}")


def select_rows(rows: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if int(max_rows) <= 0 or len(rows) <= int(max_rows):
        return rows
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(rows, size=int(max_rows), replace=False)).astype(np.int64)


def collect_replicate_rows(entity_to_reps: Sequence[np.ndarray], entity_rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    feature_rows: list[int] = []
    owner_pos: list[int] = []
    for pos, entity_idx in enumerate(np.asarray(entity_rows, dtype=np.int64).tolist()):
        reps = np.asarray(entity_to_reps[int(entity_idx)], dtype=np.int64)
        if reps.size == 0:
            continue
        feature_rows.extend([int(x) for x in reps.tolist()])
        owner_pos.extend([int(pos)] * int(reps.size))
    return np.asarray(feature_rows, dtype=np.int64), np.asarray(owner_pos, dtype=np.int64)


def entity_mean_profiles(
    features: np.ndarray,
    entity_to_reps: Sequence[np.ndarray],
    entity_rows: np.ndarray,
    norm: Dict[str, Any],
    batch_size: int,
) -> np.ndarray:
    feature_rows, owner_pos = collect_replicate_rows(entity_to_reps, entity_rows)
    dim = int(features.shape[1])
    sums = np.zeros((len(entity_rows), dim), dtype=np.float32)
    counts = np.zeros(len(entity_rows), dtype=np.float32)
    for start in range(0, len(feature_rows), max(1, int(batch_size))):
        rows = feature_rows[start : start + max(1, int(batch_size))]
        owners = owner_pos[start : start + len(rows)]
        x = cgp.transform_rows(features, rows, norm).astype(np.float32)
        np.add.at(sums, owners, x)
        counts += np.bincount(owners, minlength=len(entity_rows)).astype(np.float32)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0)[:10].tolist()
        raise RuntimeError(f"Entities without replicate profiles after filtering: {missing}")
    return sums / counts[:, None]


def entity_mean_profile_latents(
    model: cgp.ReplicateCGPAlign,
    features: np.ndarray,
    entity_to_reps: Sequence[np.ndarray],
    entity_rows: np.ndarray,
    source_ids_by_entity: np.ndarray,
    norm: Dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    feature_rows, owner_pos = collect_replicate_rows(entity_to_reps, entity_rows)
    source_ids = source_ids_by_entity[owner_pos].astype(np.int64)
    z = cgp.encode_profiles(model, features, feature_rows, source_ids, norm, device, batch_size)
    sums = np.zeros((len(entity_rows), int(z.shape[1])), dtype=np.float32)
    counts = np.zeros(len(entity_rows), dtype=np.float32)
    np.add.at(sums, owner_pos, z.astype(np.float32))
    counts += np.bincount(owner_pos, minlength=len(entity_rows)).astype(np.float32)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0)[:10].tolist()
        raise RuntimeError(f"Entities without profile latents after filtering: {missing}")
    return sums / counts[:, None]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = cgp.select_device(str(args.device))

    payload = load_checkpoint(args.checkpoint)
    model_args = namespace_from_config(payload.get("config", {}))
    configure_inference(model_args, args)
    model_args.eval_batch_size = int(args.eval_batch_size)
    model_args.eval_max_replicates_per_entity = int(getattr(model_args, "eval_max_replicates_per_entity", 0))
    model_args.smoke_test = False

    data = cgp.build_data(model_args)
    model, _ = cgp.build_model(model_args, data, device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    data["compound"]["norm"] = payload.get("compound_profile_normalizer", data["compound"]["norm"])
    data["gene"]["norm"] = payload.get("gene_profile_normalizer", data["gene"]["norm"])
    model.eval()

    c_rows = select_rows(np.asarray(data["compound"][f"{args.split}_entities"], dtype=np.int64), args.max_compounds, args.seed + 11)
    g_rows = select_rows(np.asarray(data["gene"][f"{args.split}_entities"], dtype=np.int64), args.max_genes, args.seed + 23)

    z_compound = cgp.encode_compounds(model, data["compound"]["graph_store"], c_rows, device, args.eval_batch_size)
    z_gene = cgp.encode_gene_entities(
        model,
        data["gene"]["entities"],
        data["gene"]["protein_embeddings"],
        data["gene"]["modality_ids"],
        g_rows,
        device,
        args.eval_batch_size,
    )

    c_source_ids = np.zeros(len(c_rows), dtype=np.int64)
    if str(getattr(model_args, "profile_source_mode", "compound_gene_modality")) == "compound_gene":
        g_source_ids = np.ones(len(g_rows), dtype=np.int64)
    else:
        g_source_ids = 1 + data["gene"]["modality_ids"][g_rows].astype(np.int64)

    z_compound_profile = entity_mean_profile_latents(
        model,
        data["compound"]["features"],
        data["compound"]["entity_to_reps"],
        c_rows,
        c_source_ids,
        data["compound"]["norm"],
        device,
        args.eval_batch_size,
    )
    z_gene_profile = entity_mean_profile_latents(
        model,
        data["gene"]["features"],
        data["gene"]["entity_to_reps"],
        g_rows,
        g_source_ids,
        data["gene"]["norm"],
        device,
        args.eval_batch_size,
    )

    c_profile_input = entity_mean_profiles(
        data["compound"]["features"],
        data["compound"]["entity_to_reps"],
        c_rows,
        data["compound"]["norm"],
        args.eval_batch_size,
    )
    g_profile_input = entity_mean_profiles(
        data["gene"]["features"],
        data["gene"]["entity_to_reps"],
        g_rows,
        data["gene"]["norm"],
        args.eval_batch_size,
    )

    c_entities = data["compound"]["entities"].iloc[c_rows].reset_index(drop=True)
    g_entities = data["gene"]["entities"].iloc[g_rows].reset_index(drop=True)
    compound_ids = c_entities["entity_id"].astype(str).to_numpy()
    gene_entity_ids = g_entities["entity_id"].astype(str).to_numpy()
    gene_symbols = g_entities["gene_symbol"].astype(str).to_numpy()
    gene_modalities = g_entities["perturbation_modality"].astype(str).str.lower().to_numpy()
    gene_source_indices = g_entities["source_gene_index"].to_numpy(dtype=np.int64)
    compound_structure_indices = c_entities["structure_feature_index"].to_numpy(dtype=np.int64)

    embedding_path = args.output_dir / "final_raw3180_test_embeddings.npz"
    profile_path = args.output_dir / "final_raw3180_test_profile_inputs.npz"
    metadata_path = args.output_dir / "final_raw3180_hidden_input_metadata.json"

    np.savez_compressed(
        embedding_path,
        compound_indices=compound_structure_indices,
        compound_entity_indices=c_rows.astype(np.int64),
        compound_ids=compound_ids,
        z_compound=z_compound.astype(np.float32),
        z_compound_profile=z_compound_profile.astype(np.float32),
        gene_indices=gene_source_indices,
        gene_entity_indices=g_rows.astype(np.int64),
        gene_entity_ids=gene_entity_ids,
        gene_symbols=gene_symbols,
        gene_modalities=gene_modalities,
        z_protein=z_gene.astype(np.float32),
        z_gene_profile=z_gene_profile.astype(np.float32),
    )
    np.savez_compressed(
        profile_path,
        ids=np.concatenate([compound_ids, gene_entity_ids]),
        gene_symbols=np.concatenate([np.repeat("", len(compound_ids)), gene_symbols]),
        profile_type=np.concatenate([np.repeat("compound", len(compound_ids)), gene_modalities]),
        source_label=np.concatenate([np.repeat("compound", len(compound_ids)), gene_modalities]),
        x=np.vstack([c_profile_input, g_profile_input]).astype(np.float32),
    )

    metadata = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_best_score": payload.get("best_score"),
        "split": args.split,
        "profile_dim": int(data["profile_dim"]),
        "compound_entities": int(len(c_rows)),
        "gene_entities": int(len(g_rows)),
        "orf_entities": int((gene_modalities == "orf").sum()),
        "crispr_entities": int((gene_modalities == "crispr").sum()),
        "compound_data_dir": str(model_args.compound_data_dir),
        "gene_data_dir": str(model_args.gene_data_dir),
        "profile_source_mode": str(getattr(model_args, "profile_source_mode", "")),
        "profile_input_definition": "entity mean of normalized 3180-dimensional replicate-level CellProfiler profiles",
        "profile_latent_definition": "entity mean of replicate-level profile-encoder embeddings",
        "embedding_npz": str(embedding_path),
        "profile_input_npz": str(profile_path),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
