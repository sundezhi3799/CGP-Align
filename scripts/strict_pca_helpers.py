"""Non-visual PCA input helpers adapted from the original multistage exporter."""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import argparse
import numpy as np
import pandas as pd
import torch
import train_cgp_align_replicate as cgp

def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(denom, eps)).astype(np.float32)



def make_source_ids(train_args: argparse.Namespace, data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    c_source = np.zeros(len(data["compound"]["entities"]), dtype=np.int64)
    if str(getattr(train_args, "profile_source_mode", "compound_gene_modality")) == "compound_gene":
        g_source = np.ones(len(data["gene"]["entities"]), dtype=np.int64)
    else:
        g_source = 1 + data["gene"]["modality_ids"].astype(np.int64)
    return c_source, g_source


def mean_profile_latents(
    model: cgp.ReplicateCGPAlign,
    features: np.ndarray,
    entity_to_reps: Sequence[np.ndarray],
    entity_rows: np.ndarray,
    source_id_by_entity: np.ndarray,
    norm: Dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    feature_rows: List[int] = []
    owner_entities: List[int] = []
    for entity_idx in np.asarray(entity_rows, dtype=np.int64).tolist():
        reps = entity_to_reps[int(entity_idx)]
        if reps.size == 0:
            continue
        feature_rows.extend([int(x) for x in reps.tolist()])
        owner_entities.extend([int(entity_idx)] * int(reps.size))
    if not feature_rows:
        return np.zeros((len(entity_rows), 0), dtype=np.float32)

    feature_rows_np = np.asarray(feature_rows, dtype=np.int64)
    owner_entities_np = np.asarray(owner_entities, dtype=np.int64)
    src = source_id_by_entity[owner_entities_np].astype(np.int64)
    z_reps = cgp.encode_profiles(model, features, feature_rows_np, src, norm, device, batch_size)

    pos = {int(entity_idx): i for i, entity_idx in enumerate(np.asarray(entity_rows, dtype=np.int64).tolist())}
    out = np.zeros((len(entity_rows), z_reps.shape[1]), dtype=np.float32)
    counts = np.zeros(len(entity_rows), dtype=np.int32)
    for i, owner in enumerate(owner_entities_np.tolist()):
        j = pos[int(owner)]
        out[j] += z_reps[i]
        counts[j] += 1
    out /= np.maximum(counts[:, None], 1)
    return l2_normalize(out)


def metadata_for_compounds(entities: pd.DataFrame, rows: np.ndarray, modality: str) -> pd.DataFrame:
    sub = entities.iloc[np.asarray(rows, dtype=np.int64)].reset_index(drop=True)
    display_group = "compound_structure" if modality == "compound_structure" else "compound_profile"
    return pd.DataFrame(
        {
            "entity_index": np.asarray(rows, dtype=np.int64),
            "entity_id": sub.get("entity_id", pd.Series([""] * len(sub))).astype(str),
            "pair_kind": "compound",
            "modality": modality,
            "display_group": display_group,
            "gene_symbol": "",
            "perturbation_modality": "",
            "label": sub.get("compound_id", sub.get("entity_id", pd.Series([""] * len(sub)))).astype(str),
        }
    )


def metadata_for_genes(entities: pd.DataFrame, rows: np.ndarray, modality: str) -> pd.DataFrame:
    sub = entities.iloc[np.asarray(rows, dtype=np.int64)].reset_index(drop=True)
    perturb_mod = sub.get("perturbation_modality", pd.Series(["gene"] * len(sub))).astype(str).str.lower()
    group_prefix = "gene_embedding" if modality == "gene_embedding" else "gene_profile"
    return pd.DataFrame(
        {
            "entity_index": np.asarray(rows, dtype=np.int64),
            "entity_id": sub.get("entity_id", pd.Series([""] * len(sub))).astype(str),
            "pair_kind": "gene",
            "modality": modality,
            "display_group": group_prefix + "::" + perturb_mod,
            "gene_symbol": sub.get("gene_symbol", pd.Series([""] * len(sub))).astype(str),
            "perturbation_modality": perturb_mod,
            "label": sub.get("gene_symbol", pd.Series([""] * len(sub))).astype(str) + "::" + perturb_mod,
        }
    )



def encode_stage(
    model: cgp.ReplicateCGPAlign,
    data: Dict[str, Any],
    train_args: argparse.Namespace,
    c_rows: np.ndarray,
    g_rows: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
    c_source, g_source = make_source_ids(train_args, data)

    z_c = l2_normalize(cgp.encode_compounds(model, data["compound"]["graph_store"], c_rows, device, batch_size))
    z_g = l2_normalize(
        cgp.encode_gene_entities(
            model,
            data["gene"]["entities"],
            data["gene"]["protein_embeddings"],
            data["gene"]["modality_ids"],
            g_rows,
            device,
            batch_size,
        )
    )
    z_cp = mean_profile_latents(
        model,
        data["compound"]["features"],
        data["compound"]["entity_to_reps"],
        c_rows,
        c_source,
        data["compound"]["norm"],
        device,
        batch_size,
    )
    z_gp = mean_profile_latents(
        model,
        data["gene"]["features"],
        data["gene"]["entity_to_reps"],
        g_rows,
        g_source,
        data["gene"]["norm"],
        device,
        batch_size,
    )

    x = np.vstack([z_c, z_cp, z_g, z_gp]).astype(np.float32)
    meta = pd.concat(
        [
            metadata_for_compounds(data["compound"]["entities"], c_rows, "compound_structure"),
            metadata_for_compounds(data["compound"]["entities"], c_rows, "compound_profile"),
            metadata_for_genes(data["gene"]["entities"], g_rows, "gene_embedding"),
            metadata_for_genes(data["gene"]["entities"], g_rows, "gene_profile"),
        ],
        ignore_index=True,
    )
    diagnostics = {
        "compound_matched_cosine_mean": float(np.mean(np.sum(z_c * z_cp, axis=1))) if len(z_c) else None,
        "gene_matched_cosine_mean": float(np.mean(np.sum(z_g * z_gp, axis=1))) if len(z_g) else None,
    }
    rng = np.random.default_rng(991)
    if len(z_c) > 1:
        diagnostics["compound_random_cosine_mean"] = float(np.mean(np.sum(z_c * z_cp[rng.permutation(len(z_cp))], axis=1)))
    if len(z_g) > 1:
        diagnostics["gene_random_cosine_mean"] = float(np.mean(np.sum(z_g * z_gp[rng.permutation(len(z_gp))], axis=1)))
    return x, meta, diagnostics


