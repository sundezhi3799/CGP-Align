from __future__ import annotations

import argparse
import json
import os
import pathlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from cgp_common import (
    DATA_DIR,
    INTRINSIC_DIR,
    PROTEIN_EMB_DIR,
    CGPBundle,
    metrics_from_ranks,
    retrieval_ranks_single_positive,
    tensor_from_numpy,
    write_dataframe,
    write_json,
)
from cgp_gnn import GraphStore
from train_gene_mocop import (
    GENE_MOCOP_DATA_DIR,
    MODALITY_TO_ID,
    build_entity_replicate_index,
    eligible_entities,
    full_gallery_metrics,
    gene_codes,
    load_gene_mocop_data,
    modality_ids,
    normalizer_summary,
    sampled_metrics,
)
from train_tri_intrinsic_gnn_esm650 import (
    DEFAULT_GENE_CKPT_DIR,
    DEFAULT_GENE_PROTEIN_EMB_DIR,
    DEFAULT_INTRINSIC_SPLIT,
    TriIntrinsicGNNESM,
    compound_profile_batch,
    entity_mean_profiles,
    gene_profile_source_ids,
    gene_protein_tensor,
    parse_hidden_dims,
    quick_val_gene,
    smiles_list,
    source_ids,
)


DEFAULT_OUTPUT_DIR = Path("output/cgp_align/tri_intrinsic/eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tri-modal intrinsic GNN/ESM650 alignment.")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--cgp_data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--cgp_protein_embedding_dir", type=Path, default=PROTEIN_EMB_DIR)
    parser.add_argument("--intrinsic_split_path", type=Path, default=DEFAULT_INTRINSIC_SPLIT)
    parser.add_argument("--gene_mocop_data_dir", type=Path, default=GENE_MOCOP_DATA_DIR)
    parser.add_argument("--gene_protein_embedding_dir", type=Path, default=DEFAULT_GENE_PROTEIN_EMB_DIR)
    parser.add_argument("--checkpoint_dir", type=Path, default=DEFAULT_GENE_CKPT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint_name", default="best_model.pt")
    parser.add_argument("--query_split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--compound_encode_batch_size", type=int, default=512)
    parser.add_argument("--gene_encode_batch_size", type=int, default=2048)
    parser.add_argument("--negative_ratios", default="100,1000")
    parser.add_argument("--negative_pool", default="query_split", choices=["query_split", "full_gallery"])
    parser.add_argument("--num_repeats", type=int, default=5)
    parser.add_argument("--sample_batch_size", type=int, default=256)
    parser.add_argument("--modalities", default="orf,crispr")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compound_only", action="store_true")
    return parser.parse_args()


def parse_ratios(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def load_gene_data(args: argparse.Namespace) -> Tuple[Any, Any, np.ndarray, np.ndarray, Dict[str, Any]]:
    shim = argparse.Namespace(
        data_dir=args.gene_mocop_data_dir,
        protein_embedding_dir=args.gene_protein_embedding_dir,
        modalities=args.modalities,
        smoke_test=False,
    )
    return load_gene_mocop_data(shim)


def load_model(args: argparse.Namespace, cgp_bundle: CGPBundle, gene_features: np.ndarray, gene_protein_embeddings: np.ndarray, device: torch.device):
    ckpt_dir = args.checkpoint_dir / args.run_name
    config = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
    model = TriIntrinsicGNNESM(
        compound_profile_dim=cgp_bundle.profile_dim,
        gene_profile_dim=int(gene_features.shape[1]),
        protein_dim=int(gene_protein_embeddings.shape[1]),
        embed_dim=int(config.get("embed_dim", 128)),
        gnn_hidden_dim=int(config.get("gnn_hidden_dim", 256)),
        gnn_layers=int(config.get("gnn_layers", 6)),
        protein_hidden_dims=parse_hidden_dims(config.get("protein_hidden_dims", "512,256")),
        profile_hidden_dims=parse_hidden_dims(config.get("profile_hidden_dims", "512,256")),
        modality_context_dim=int(config.get("modality_context_dim", 32)),
        dropout=float(config.get("dropout", 0.1)),
        profile_source_adapters=bool(config.get("profile_source_adapters", False)),
        profile_source_embedding=not bool(config.get("disable_profile_source_embedding", False)),
        num_profile_sources=int(config.get("num_profile_sources", 1 + len(MODALITY_TO_ID) if config.get("gene_profile_source_mode") == "modality" else 2)),
    ).to(device)
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    payload = torch.load(ckpt_dir / args.checkpoint_name, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, config, payload


def encode_compounds(model, bundle: CGPBundle, graph_store: GraphStore, indices: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    chunks = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            z = model.encode_compound_graphs(graph_store.get_many(batch.tolist()), device)
            chunks.append(z.detach().cpu().numpy().astype(np.float32))
    return np.vstack(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)


def encode_compound_profiles(
    model,
    bundle: CGPBundle,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    norm: Dict[str, Any],
) -> np.ndarray:
    chunks = []
    rows = bundle.profile_rows_for_compounds(indices)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            sub = rows[start : start + batch_size]
            x = compound_profile_batch(bundle, indices[start : start + batch_size], device, norm)
            z = model.encode_profile(x, source_ids(len(sub), 0, device))
            chunks.append(z.detach().cpu().numpy().astype(np.float32))
    return np.vstack(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)


def sampled_ranks(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    pos_indices: np.ndarray,
    query_ids: List[str],
    task: str,
    negative_ratio: int,
    repeat: int,
    seed: int,
    batch_size: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rng = np.random.default_rng(seed + repeat * 100003 + negative_ratio * 17)
    n_gallery = int(gallery_embeddings.shape[0])
    n_neg = min(int(negative_ratio), n_gallery - 1)
    ranks: List[int] = []
    rows: List[Dict[str, Any]] = []
    all_gallery = np.arange(n_gallery, dtype=np.int64)
    for start in range(0, len(query_embeddings), batch_size):
        end = min(start + batch_size, len(query_embeddings))
        bsz = end - start
        candidates = np.empty((bsz, n_neg + 1), dtype=np.int64)
        candidates[:, 0] = pos_indices[start:end]
        for local_i, pos in enumerate(pos_indices[start:end]):
            needed = n_neg
            negs: List[np.ndarray] = []
            while needed > 0:
                draw = rng.choice(all_gallery, size=max(needed + 4, int(needed * 1.05)), replace=False)
                draw = draw[draw != int(pos)]
                if len(draw):
                    take = draw[:needed]
                    negs.append(take)
                    needed -= len(take)
            candidates[local_i, 1:] = np.concatenate(negs)[:n_neg]
        scores = np.einsum("bd,bkd->bk", query_embeddings[start:end], gallery_embeddings[candidates], optimize=True)
        batch_ranks = 1 + np.sum(scores[:, 1:] > scores[:, :1], axis=1)
        ranks.extend(int(x) for x in batch_ranks.tolist())
        for local_i, rank in enumerate(batch_ranks.tolist()):
            rows.append(
                {
                    "task": task,
                    "query_id": query_ids[start + local_i],
                    "negative_ratio": int(negative_ratio),
                    "repeat": int(repeat),
                    "rank": int(rank),
                    "candidate_count": int(n_neg + 1),
                }
            )
    arr = np.asarray(ranks, dtype=np.int64)
    return {
        "num_queries": int(len(arr)),
        "negative_ratio": int(negative_ratio),
        "candidate_count": int(n_neg + 1),
        "Top1_accuracy": float(np.mean(arr <= 1)) if len(arr) else 0.0,
        "Top5_accuracy": float(np.mean(arr <= 5)) if len(arr) else 0.0,
        "Top10_accuracy": float(np.mean(arr <= 10)) if len(arr) else 0.0,
        "MRR": float(np.mean(1.0 / arr)) if len(arr) else 0.0,
        "mean_rank": float(np.mean(arr)) if len(arr) else 0.0,
        "median_rank": float(np.median(arr)) if len(arr) else 0.0,
    }, rows


def summarize_repeat_metrics(repeats: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {"num_repeats": len(repeats), "repeats": repeats}
    for key in ["Top1_accuracy", "Top5_accuracy", "Top10_accuracy", "MRR", "mean_rank", "median_rank"]:
        vals = np.asarray([float(r[key]) for r in repeats], dtype=np.float64)
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_std"] = float(vals.std(ddof=0))
    if repeats:
        out["num_queries"] = int(repeats[0]["num_queries"])
        out["candidate_count"] = int(repeats[0]["candidate_count"])
        out["negative_ratio"] = int(repeats[0]["negative_ratio"])
    return out


def evaluate_sampled(task: str, query: np.ndarray, gallery: np.ndarray, pos: np.ndarray, ids: List[str], ratios: List[int], args):
    metrics: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    for ratio in ratios:
        repeats = []
        for repeat in range(args.num_repeats):
            m, r = sampled_ranks(query, gallery, pos, ids, task, ratio, repeat, args.seed, args.sample_batch_size)
            repeats.append(m)
            rows.extend(r)
        metrics[f"1:{ratio}"] = summarize_repeat_metrics(repeats)
    return metrics, rows


def encode_gene_entities(model, entities, features, protein_embeddings, rows, entity_to_reps, entity_mods, norm, profile_source_mode, device, batch_size):
    z_p_chunks: List[np.ndarray] = []
    z_prof_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            mod_np = entity_mods[batch]
            mod = torch.from_numpy(mod_np.astype(np.int64)).long().to(device)
            zp = model.encode_protein(gene_protein_tensor(entities, protein_embeddings, batch, device), mod)
            prof = entity_mean_profiles(features, entity_to_reps, batch, entity_mods, norm)
            zprof = model.encode_profile(tensor_from_numpy(prof, device), gene_profile_source_ids(mod, profile_source_mode))
            z_p_chunks.append(zp.detach().cpu().numpy().astype(np.float32))
            z_prof_chunks.append(zprof.detach().cpu().numpy().astype(np.float32))
    return np.vstack(z_p_chunks), np.vstack(z_prof_chunks)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    ratios = parse_ratios(args.negative_ratios)
    cgp_bundle = CGPBundle(args.cgp_data_dir, args.cgp_protein_embedding_dir)
    split = json.loads(args.intrinsic_split_path.read_text(encoding="utf-8"))
    gene_entities, gene_replicates, gene_features, gene_protein_embeddings, gene_splits = load_gene_data(args)
    model, config, payload = load_model(args, cgp_bundle, gene_features, gene_protein_embeddings, device)
    gene_norm = payload.get("gene_profile_normalizer", {"mode": "none"})
    compound_norm = payload.get("compound_profile_normalizer", {"mode": "none"})
    graph_store = GraphStore(smiles_list(cgp_bundle))
    metrics: Dict[str, Any] = {
        "run_name": args.run_name,
        "query_split": args.query_split,
        "negative_pool": args.negative_pool,
        "negative_ratios": ratios,
        "num_repeats": int(args.num_repeats),
        "checkpoint_epoch": payload.get("epoch"),
        "best_score": payload.get("best_score"),
        "config": config,
        "compound_profile_normalizer": normalizer_summary(compound_norm),
        "gene_profile_normalizer": normalizer_summary(gene_norm),
        "uses_target_or_link_labels_for_evaluation": False,
    }
    rank_rows: List[Dict[str, Any]] = []

    compound_gallery_source = np.asarray(split["compound"][args.query_split] if args.negative_pool == "query_split" else np.arange(len(cgp_bundle.compounds)), dtype=np.int64)
    compound_gallery = compound_gallery_source[
        cgp_bundle.has_compound_profile[compound_gallery_source] & (cgp_bundle.compound_profile_row[compound_gallery_source] >= 0)
    ]
    compound_queries = np.asarray([c for c in split["compound"][args.query_split] if int(c) in set(compound_gallery.tolist())], dtype=np.int64)
    metrics["compound_gallery_size"] = int(len(compound_gallery))
    metrics["compound_query_size"] = int(len(compound_queries))
    if len(compound_queries):
        compound_gallery_map = {int(c): i for i, c in enumerate(compound_gallery.tolist())}
        pos = np.asarray([compound_gallery_map[int(c)] for c in compound_queries], dtype=np.int64)
        z_c_query = encode_compounds(model, cgp_bundle, graph_store, compound_queries, device, args.compound_encode_batch_size)
        z_c_gallery = encode_compounds(model, cgp_bundle, graph_store, compound_gallery, device, args.compound_encode_batch_size)
        z_cp_query = encode_compound_profiles(model, cgp_bundle, compound_queries, device, args.compound_encode_batch_size, compound_norm)
        z_cp_gallery = encode_compound_profiles(model, cgp_bundle, compound_gallery, device, args.compound_encode_batch_size, compound_norm)
        ranks, gaps = retrieval_ranks_single_positive(z_c_query, z_cp_gallery, pos)
        metrics["compound_structure_to_compound_profile"] = metrics_from_ranks(ranks, gaps)
        ranks, gaps = retrieval_ranks_single_positive(z_cp_query, z_c_gallery, pos)
        metrics["compound_profile_to_compound_structure"] = metrics_from_ranks(ranks, gaps)
        ids = cgp_bundle.compounds.iloc[compound_queries]["compound_id"].astype(str).tolist()
        sampled, rows = evaluate_sampled("compound_structure_to_compound_profile", z_c_query, z_cp_gallery, pos, ids, ratios, args)
        metrics["compound_structure_to_compound_profile_sampled"] = sampled
        rank_rows.extend(rows)
        sampled, rows = evaluate_sampled("compound_profile_to_compound_structure", z_cp_query, z_c_gallery, pos, ids, ratios, args)
        metrics["compound_profile_to_compound_structure_sampled"] = sampled
        rank_rows.extend(rows)

    if not args.compound_only:
        entity_to_reps = build_entity_replicate_index(gene_replicates, len(gene_entities))
        entity_mods = modality_ids(gene_entities)
        entity_gene_codes = gene_codes(gene_entities)
        test_entities = eligible_entities(gene_entities, gene_splits["cold_gene"][args.query_split], gene_protein_embeddings)
        metrics["gene_query_size"] = int(len(test_entities))
        metrics["gene_gallery_size"] = int(len(test_entities))
        z_p, z_prof = encode_gene_entities(
            model,
            gene_entities,
            gene_features,
            gene_protein_embeddings,
            test_entities,
            entity_to_reps,
            entity_mods,
            gene_norm,
            str(config.get("gene_profile_source_mode", "shared")),
            device,
            args.gene_encode_batch_size,
        )
        metrics["protein_to_gene_profile"] = full_gallery_metrics(z_p, z_prof)
        metrics["gene_profile_to_protein"] = full_gallery_metrics(z_prof, z_p)
        metrics["gene_mocop_protocol_sampled"] = sampled_metrics(
            z_p,
            z_prof,
            entity_gene_codes[test_entities],
            entity_mods[test_entities],
            ratios,
            args.num_repeats,
            args.seed,
            bool(config.get("mask_crossmod_negatives", True)),
            bool(config.get("mask_same_gene_negatives", False)),
        )

    write_json(args.output_dir / f"{args.run_name}_metrics.json", metrics)
    write_dataframe(args.output_dir / f"{args.run_name}_sampled_ranks.parquet", rank_rows)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
