from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from cgp_common import (
    CHECKPOINT_DIR,
    DATA_DIR,
    LOG_DIR,
    PROTEIN_EMB_DIR,
    CGPBundle,
    append_jsonl,
    bidirectional_infonce,
    long_tensor,
    metrics_from_ranks,
    retrieval_ranks_single_positive,
    set_seed,
    tensor_from_numpy,
    write_json,
)
from cgp_gnn import GGNNCompoundEncoder, GraphStore
from train_gene_mocop import (
    GENE_MOCOP_DATA_DIR,
    MODALITY_TO_ID,
    MoCoPMLP,
    augment_with_same_gene_siblings,
    build_gene_modality_entity_index,
    build_entity_replicate_index,
    eligible_entities,
    fit_profile_normalizer,
    gene_codes,
    load_gene_mocop_data,
    masked_clip_loss,
    modality_ids,
    normalizer_summary,
    parse_hidden_dims,
    profile_tensor,
    sampled_metrics,
    sample_replicate_features,
    transform_feature_rows,
    update_ema,
    weighted_multipositive_clip_loss,
)


DEFAULT_GENE_PROTEIN_EMB_DIR = Path("protein_embeddings/protein_encoder_ablation/esm650_l33_mean_cls")
DEFAULT_INTRINSIC_SPLIT = Path("data/cgp_cpg_full_motive_edges/splits_intrinsic_entity.json")
DEFAULT_GENE_CKPT_DIR = Path("output/cgp_align/tri_intrinsic/checkpoints")
DEFAULT_GENE_LOG_DIR = Path("output/cgp_align/tri_intrinsic/logs")
DEFAULT_EVAL_DIR = Path("output/cgp_align/tri_intrinsic/eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tri-modal intrinsic alignment with GNN compound encoder, frozen ESM650 protein features, and a shared profile encoder."
    )
    parser.add_argument("--cgp_data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--cgp_protein_embedding_dir", type=Path, default=PROTEIN_EMB_DIR)
    parser.add_argument("--intrinsic_split_path", type=Path, default=DEFAULT_INTRINSIC_SPLIT)
    parser.add_argument("--gene_mocop_data_dir", type=Path, default=GENE_MOCOP_DATA_DIR)
    parser.add_argument("--gene_protein_embedding_dir", type=Path, default=DEFAULT_GENE_PROTEIN_EMB_DIR)
    parser.add_argument("--checkpoint_dir", type=Path, default=DEFAULT_GENE_CKPT_DIR)
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_GENE_LOG_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--gnn_hidden_dim", type=int, default=256)
    parser.add_argument("--gnn_layers", type=int, default=6)
    parser.add_argument("--protein_hidden_dims", default="512,256")
    parser.add_argument("--profile_hidden_dims", default="512,256")
    parser.add_argument("--modality_context_dim", type=int, default=32)
    parser.add_argument("--compound_batch_size", type=int, default=256)
    parser.add_argument("--gene_batch_size", type=int, default=512)
    parser.add_argument("--steps_per_epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--init_compound_checkpoint", type=Path, default=None)
    parser.add_argument("--init_gene_checkpoint", type=Path, default=None)
    parser.add_argument("--init_tri_checkpoint", type=Path, default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--compound_temperature", type=float, default=0.07)
    parser.add_argument("--gene_temperature", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda_compound_profile", type=float, default=1.0)
    parser.add_argument("--lambda_gene_profile", type=float, default=1.0)
    parser.add_argument("--lambda_profile_bridge", type=float, default=0.0)
    parser.add_argument("--lambda_target", type=float, default=0.0)
    parser.add_argument("--lambda_typed", type=float, default=0.0)
    parser.add_argument("--lambda_direction", type=float, default=0.0)
    parser.add_argument("--lambda_anchor", type=float, default=0.0)
    parser.add_argument("--profile_bridge_temperature", type=float, default=0.1)
    parser.add_argument("--profile_bridge_target_temperature", type=float, default=0.07)
    parser.add_argument("--profile_bridge_topk", type=int, default=16)
    parser.add_argument("--target_temperature", type=float, default=0.07)
    parser.add_argument("--typed_temperature", type=float, default=0.07)
    parser.add_argument("--target_split", default="random")
    parser.add_argument("--target_edge_file", default="compound_target_edges_typed.parquet")
    parser.add_argument("--target_known_positive_file", default="compound_target_edges_unpruned.parquet")
    parser.add_argument("--target_batch_size", type=int, default=512)
    parser.add_argument("--typed_batch_size", type=int, default=256)
    parser.add_argument("--target_query_modes", default="graph")
    parser.add_argument("--typed_query_modes", default="profile")
    parser.add_argument("--direction_query_mode", choices=["graph", "profile", "fused"], default="fused")
    parser.add_argument("--direction_margin", type=float, default=0.05)
    parser.add_argument("--typed_direction_control", choices=["matched", "wrong"], default="matched")
    parser.add_argument("--val_max_target_edges", type=int, default=512)
    parser.add_argument("--profile_source_adapters", action="store_true")
    parser.add_argument("--disable_profile_source_embedding", action="store_true")
    parser.add_argument("--profile_source_dropout", type=float, default=0.0)
    parser.add_argument("--gene_profile_source_mode", choices=["shared", "modality"], default="shared")
    parser.add_argument("--compound_profile_norm", choices=["none", "train_zscore"], default="none")
    parser.add_argument("--profile_norm", choices=["none", "train_zscore", "source_train_zscore"], default="train_zscore")
    parser.add_argument("--modalities", default="orf,crispr")
    parser.add_argument("--mask_crossmod_negatives", action="store_true", default=True)
    parser.add_argument("--mask_same_gene_negatives", action="store_true")
    parser.add_argument("--same_gene_same_modality_positive_weight", type=float, default=0.0)
    parser.add_argument("--same_gene_crossmod_positive_weight", type=float, default=0.0)
    parser.add_argument("--same_gene_sibling_batch_prob", type=float, default=0.0)
    parser.add_argument("--gene_train_profile_mode", choices=["replicate", "mean"], default="replicate")
    parser.add_argument(
        "--min_train_gene_replicate_cosine",
        type=float,
        default=-2.0,
        help="Filter gene training entities by raw within-entity replicate cosine. Values below -1 disable filtering.",
    )
    parser.add_argument("--disable_mask_crossmod_negatives", action="store_true")
    parser.add_argument("--lr_scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--disable_ema", action="store_true")
    parser.add_argument("--val_every", type=int, default=20)
    parser.add_argument("--val_max_compounds", type=int, default=2048)
    parser.add_argument("--val_max_gene_entities", type=int, default=2048)
    parser.add_argument("--val_negative_ratio", type=int, default=100)
    parser.add_argument("--val_repeats", type=int, default=5)
    parser.add_argument("--selection_metric", default="val_tri_alignment_score")
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=1.0,
        help="Fraction of each training pool to use. Validation/test folds are unchanged.",
    )
    parser.add_argument("--train_max_compounds", type=int, default=0)
    parser.add_argument("--train_max_gene_entities", type=int, default=0)
    parser.add_argument("--train_max_target_edges", type=int, default=0)
    parser.add_argument("--train_max_typed_edges", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--train_only", action="store_true")
    parser.add_argument("--strict_init_shapes", action="store_true")
    parser.add_argument(
        "--allow_partial_init",
        action="store_true",
        help="Allow shape-mismatched checkpoint tensors to be skipped. Use only for exploratory warm starts, not controlled ablations.",
    )
    return parser.parse_args()


def subsample_train_pool(
    rows: np.ndarray,
    fraction: float,
    max_rows: int,
    seed: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
    rows = np.asarray(rows)
    before = int(len(rows))
    if before == 0:
        return rows, {"before": 0, "after": 0, "fraction": float(fraction), "max_rows": int(max_rows)}
    keep = before
    if 0.0 < float(fraction) < 1.0:
        keep = min(keep, max(1, int(round(before * float(fraction)))))
    if int(max_rows) > 0:
        keep = min(keep, int(max_rows))
    if keep < before:
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(np.arange(before), size=keep, replace=False))
        rows = rows[idx]
    return rows, {
        "before": before,
        "after": int(len(rows)),
        "fraction": float(fraction),
        "max_rows": int(max_rows),
    }


class SharedProfileEncoder(nn.Module):
    def __init__(
        self,
        profile_dim: int,
        hidden_dims: Sequence[int],
        embed_dim: int,
        dropout: float,
        use_source_adapters: bool = False,
        use_source_embedding: bool = True,
        source_dropout: float = 0.0,
        num_sources: int = 2,
    ):
        super().__init__()
        self.use_source_adapters = bool(use_source_adapters)
        self.source_dropout = float(source_dropout)
        self.num_sources = int(num_sources)
        self.source_embedding = nn.Embedding(self.num_sources, int(profile_dim)) if use_source_embedding else None
        if self.use_source_adapters:
            self.adapters = nn.ModuleList()
            for _ in range(self.num_sources):
                adapter = nn.Sequential(
                    nn.LayerNorm(int(profile_dim)),
                    nn.Linear(int(profile_dim), int(profile_dim)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(int(profile_dim), int(profile_dim)),
                )
                nn.init.zeros_(adapter[-1].weight)
                nn.init.zeros_(adapter[-1].bias)
                self.adapters.append(adapter)
        else:
            self.adapters = None
        self.encoder = MoCoPMLP(int(profile_dim), hidden_dims, int(embed_dim), dropout)

    def forward(self, x: torch.Tensor, source_id: torch.Tensor) -> torch.Tensor:
        h = x
        source_gate = None
        if self.training and self.source_dropout > 0:
            keep = torch.rand((x.shape[0], 1), device=x.device) >= float(self.source_dropout)
            source_gate = keep.to(dtype=x.dtype)
        if self.source_embedding is not None:
            emb = self.source_embedding(source_id)
            if source_gate is not None:
                emb = emb * source_gate
            h = h + emb
        if self.adapters is not None:
            adapted = torch.empty_like(h)
            for source in range(self.num_sources):
                mask = source_id == source
                if torch.any(mask):
                    residual = self.adapters[source](h[mask])
                    if source_gate is not None:
                        residual = residual * source_gate[mask]
                    adapted[mask] = h[mask] + residual
            h = adapted
        return F.normalize(self.encoder(h), dim=-1)


class TriIntrinsicGNNESM(nn.Module):
    def __init__(
        self,
        compound_profile_dim: int,
        gene_profile_dim: int,
        protein_dim: int,
        embed_dim: int,
        gnn_hidden_dim: int,
        gnn_layers: int,
        protein_hidden_dims: Sequence[int],
        profile_hidden_dims: Sequence[int],
        modality_context_dim: int,
        dropout: float,
        profile_source_adapters: bool = False,
        profile_source_embedding: bool = True,
        profile_source_dropout: float = 0.0,
        num_profile_sources: int = 2,
    ):
        super().__init__()
        if int(compound_profile_dim) != int(gene_profile_dim):
            raise ValueError(f"Profile dims differ: compound={compound_profile_dim}, gene={gene_profile_dim}")
        self.compound_encoder = GGNNCompoundEncoder(
            atom_dim=__import__("cgp_gnn").atom_feature_dim(),
            bond_dim=__import__("cgp_gnn").bond_feature_dim(),
            embed_dim=embed_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_layers,
            projection_hidden_dims=(512,),
            dropout=dropout,
        )
        self.modality_embedding = nn.Embedding(len(MODALITY_TO_ID), int(modality_context_dim))
        self.protein_encoder = MoCoPMLP(int(protein_dim) + int(modality_context_dim), protein_hidden_dims, embed_dim, dropout)
        self.profile_encoder = SharedProfileEncoder(
            int(compound_profile_dim),
            profile_hidden_dims,
            embed_dim,
            dropout,
            use_source_adapters=profile_source_adapters,
            use_source_embedding=profile_source_embedding,
            source_dropout=profile_source_dropout,
            num_sources=num_profile_sources,
        )

    def encode_compound_graphs(self, graphs: Sequence[Any], device: torch.device) -> torch.Tensor:
        from cgp_gnn import batch_graphs

        atom_x, edge_index, edge_features, batch_index = batch_graphs(graphs, device)
        return self.compound_encoder(atom_x, edge_index, edge_features, batch_index)

    def encode_protein(self, x: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, self.modality_embedding(modality)], dim=-1)
        return F.normalize(self.protein_encoder(x), dim=-1)

    def encode_profile(self, x: torch.Tensor, source_id: torch.Tensor) -> torch.Tensor:
        return self.profile_encoder(x, source_id)


def smiles_list(bundle: CGPBundle) -> List[str]:
    return bundle.compounds["canonical_smiles"].fillna("").astype(str).tolist()


def load_intrinsic_split(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_gene_data(args: argparse.Namespace) -> Tuple[Any, Any, np.ndarray, np.ndarray, Dict[str, Any]]:
    shim = argparse.Namespace(
        data_dir=args.gene_mocop_data_dir,
        protein_embedding_dir=args.gene_protein_embedding_dir,
        modalities=args.modalities,
        smoke_test=args.smoke_test,
    )
    return load_gene_mocop_data(shim)


def fit_compound_profile_normalizer(bundle: CGPBundle, train_compounds: np.ndarray, mode: str) -> Dict[str, Any]:
    if mode == "none":
        return {"mode": "none"}
    rows = bundle.profile_rows_for_compounds(train_compounds)
    rows = rows[rows >= 0]
    if rows.size == 0:
        raise RuntimeError("No train compound profiles available for normalization.")
    x = bundle.profile_features_for_rows(rows).astype(np.float32)
    mean = np.nanmean(x, axis=0).astype(np.float32)
    std = np.nanstd(x, axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return {"mode": mode, "mean": mean, "std": std, "num_fit_profiles": np.asarray([rows.size], dtype=np.int64)}


def transform_compound_profile_rows(bundle: CGPBundle, rows: np.ndarray, norm: Dict[str, Any]) -> np.ndarray:
    x = bundle.profile_features_for_rows(rows).astype(np.float32).copy()
    mode = norm.get("mode", "none")
    if mode == "none":
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if mode == "train_zscore":
        x = (x - norm["mean"]) / norm["std"]
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    raise ValueError(f"Unsupported compound profile normalizer mode: {mode}")


def compound_profile_batch(
    bundle: CGPBundle,
    compound_indices: np.ndarray,
    device: torch.device,
    norm: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    rows = bundle.profile_rows_for_compounds(compound_indices)
    features = transform_compound_profile_rows(bundle, rows, norm or {"mode": "none"})
    return tensor_from_numpy(features, device)


def source_ids(size: int, value: int, device: torch.device) -> torch.Tensor:
    return torch.full((int(size),), int(value), dtype=torch.long, device=device)


def parse_mode_list(text: str, allowed: Sequence[str]) -> List[str]:
    out: List[str] = []
    allowed_set = set(allowed)
    for item in str(text).split(","):
        mode = item.strip().lower()
        if not mode:
            continue
        if mode not in allowed_set:
            raise ValueError(f"Unsupported mode '{mode}', expected one of {sorted(allowed_set)}")
        if mode not in out:
            out.append(mode)
    return out or [str(allowed[0])]


def gene_profile_source_ids(modality: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "modality":
        return 1 + modality.long()
    return torch.ones_like(modality, dtype=torch.long)


def edge_records_to_index_pairs(bundle: CGPBundle, records: Sequence[Dict[str, Any]]) -> np.ndarray:
    pairs: List[Tuple[int, int]] = []
    for item in records:
        c = str(item.get("compound_id", ""))
        g = str(item.get("gene_id", ""))
        if c in bundle.compound_id_to_idx and g in bundle.gene_id_to_idx:
            pairs.append((bundle.compound_id_to_idx[c], bundle.gene_id_to_idx[g]))
    return np.asarray(pairs, dtype=np.int64) if pairs else np.zeros((0, 2), dtype=np.int64)


def load_target_split_pairs(bundle: CGPBundle, split_name: str, fold: str) -> np.ndarray:
    split = json.loads((bundle.data_dir / f"splits_{split_name}.json").read_text(encoding="utf-8"))
    return edge_records_to_index_pairs(bundle, split.get(fold, []))


def load_known_positive_pairs(bundle: CGPBundle, filename: str) -> set[Tuple[int, int]]:
    path = bundle.data_dir / filename
    if not path.exists():
        return set(bundle.known_positive_pairs)
    table = __import__("pandas").read_parquet(path)
    out: set[Tuple[int, int]] = set()
    for row in table[["compound_id", "gene_id"]].itertuples(index=False):
        c = str(row.compound_id)
        g = str(row.gene_id)
        if c in bundle.compound_id_to_idx and g in bundle.gene_id_to_idx:
            out.add((bundle.compound_id_to_idx[c], bundle.gene_id_to_idx[g]))
    return out


def load_typed_target_pairs(bundle: CGPBundle, filename: str, split_name: str, fold: str) -> np.ndarray:
    path = bundle.data_dir / filename
    if not path.exists():
        return np.zeros((0, 3), dtype=np.int64)
    table = __import__("pandas").read_parquet(path)
    split_pairs = {tuple(x) for x in load_target_split_pairs(bundle, split_name, fold).tolist()}
    if not split_pairs:
        return np.zeros((0, 3), dtype=np.int64)
    out: List[Tuple[int, int, int]] = []
    for row in table[
        ["compound_id", "gene_id", "typed_target_label", "usable_for_typed_prior"]
    ].itertuples(index=False):
        if not bool(row.usable_for_typed_prior):
            continue
        c = str(row.compound_id)
        g = str(row.gene_id)
        if c not in bundle.compound_id_to_idx or g not in bundle.gene_id_to_idx:
            continue
        c_idx = bundle.compound_id_to_idx[c]
        g_idx = bundle.gene_id_to_idx[g]
        if (c_idx, g_idx) not in split_pairs:
            continue
        label = str(row.typed_target_label)
        if label == "inhibitory":
            out.append((c_idx, g_idx, MODALITY_TO_ID["crispr"]))
        elif label == "activating":
            out.append((c_idx, g_idx, MODALITY_TO_ID["orf"]))
    return np.asarray(out, dtype=np.int64) if out else np.zeros((0, 3), dtype=np.int64)


def build_target_profile_entity_index(
    entities: Any,
    entity_to_reps: Sequence[np.ndarray],
) -> Dict[Tuple[int, int], np.ndarray]:
    entity_mods = modality_ids(entities)
    buckets: Dict[Tuple[int, int], List[int]] = {}
    for row_idx, row in entities.reset_index(drop=True).iterrows():
        source_gene_index = int(row.get("source_gene_index", -1))
        if source_gene_index < 0 or not bool(row.get("has_protein_sequence", True)):
            continue
        if entity_to_reps[int(row_idx)].size == 0:
            continue
        key = (source_gene_index, int(entity_mods[int(row_idx)]))
        buckets.setdefault(key, []).append(int(row_idx))
    return {k: np.asarray(v, dtype=np.int64) for k, v in buckets.items()}


def target_protein_embeddings(
    model: TriIntrinsicGNNESM,
    protein_embeddings: np.ndarray,
    gene_indices: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for start in range(0, len(gene_indices), batch_size):
        sub = gene_indices[start : start + batch_size]
        x = tensor_from_numpy(protein_embeddings[sub], device)
        per_mod: List[torch.Tensor] = []
        for modality_name in ["orf", "crispr"]:
            mod = torch.full((len(sub),), MODALITY_TO_ID[modality_name], dtype=torch.long, device=device)
            per_mod.append(model.encode_protein(x, mod))
        chunks.append(F.normalize(torch.stack(per_mod, dim=0).mean(dim=0), dim=-1))
    return torch.cat(chunks, dim=0) if chunks else torch.zeros((0, 0), device=device)


def positive_matrix(
    row_ids: np.ndarray,
    col_ids: np.ndarray,
    positive_pairs: np.ndarray,
    device: torch.device,
    known_positive_pairs: Optional[set[Tuple[int, int]]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    row_pos = {int(v): i for i, v in enumerate(row_ids.tolist())}
    col_pos = {int(v): i for i, v in enumerate(col_ids.tolist())}
    pos = torch.zeros((len(row_ids), len(col_ids)), dtype=torch.bool, device=device)
    for c, g in positive_pairs[:, :2].tolist():
        if int(c) in row_pos and int(g) in col_pos:
            pos[row_pos[int(c)], col_pos[int(g)]] = True
    invalid = None
    if known_positive_pairs is not None:
        invalid = torch.zeros_like(pos)
        for i, c in enumerate(row_ids.tolist()):
            for j, g in enumerate(col_ids.tolist()):
                if (int(c), int(g)) in known_positive_pairs and not bool(pos[i, j]):
                    invalid[i, j] = True
    return pos, invalid


def multi_positive_contrastive_loss(
    query: torch.Tensor,
    gallery: torch.Tensor,
    positives: torch.Tensor,
    temperature: float,
    invalid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if query.shape[0] == 0 or gallery.shape[0] == 0 or not bool(positives.any()):
        return query.new_tensor(0.0)
    logits = query @ gallery.T / float(temperature)
    if invalid_mask is not None:
        logits = logits.masked_fill(invalid_mask, -1e4)
    row_has = positives.any(dim=1)
    col_has = positives.any(dim=0)
    losses: List[torch.Tensor] = []
    if bool(row_has.any()):
        row_targets = positives[row_has].float()
        row_targets = row_targets / row_targets.sum(dim=1, keepdim=True).clamp_min(1.0)
        losses.append(-(row_targets * F.log_softmax(logits[row_has], dim=1)).sum(dim=1).mean())
    if bool(col_has.any()):
        col_targets = positives[:, col_has].T.float()
        col_targets = col_targets / col_targets.sum(dim=1, keepdim=True).clamp_min(1.0)
        col_logits = logits[:, col_has].T
        if invalid_mask is not None:
            col_logits = col_logits.masked_fill(invalid_mask[:, col_has].T, -1e4)
        losses.append(-(col_targets * F.log_softmax(col_logits, dim=1)).sum(dim=1).mean())
    return torch.stack(losses).mean() if losses else query.new_tensor(0.0)


def cosine_anchor_loss(student: Sequence[torch.Tensor], teacher: Sequence[torch.Tensor]) -> torch.Tensor:
    losses = []
    for s, t in zip(student, teacher):
        if s.shape == t.shape and s.numel():
            losses.append((1.0 - (s * t.detach()).sum(dim=-1)).mean())
    return torch.stack(losses).mean() if losses else student[0].new_tensor(0.0)


def fused_embedding(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.normalize(0.5 * (a + b), dim=-1)


def select_query_embeddings(
    mode: str,
    graph_embeddings: torch.Tensor,
    profile_embeddings: torch.Tensor,
) -> torch.Tensor:
    if mode == "graph":
        return graph_embeddings
    if mode == "profile":
        return profile_embeddings
    if mode == "fused":
        return fused_embedding(graph_embeddings, profile_embeddings)
    raise ValueError(f"Unsupported query mode: {mode}")


def pairwise_direction_margin_loss(
    query: torch.Tensor,
    matched: torch.Tensor,
    opposite: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    if query.shape[0] == 0:
        return query.new_tensor(0.0)
    matched_score = (query * matched).sum(dim=-1)
    opposite_score = (query * opposite).sum(dim=-1)
    return F.relu(float(margin) - matched_score + opposite_score).mean()


def quick_val_target_binary(
    model: TriIntrinsicGNNESM,
    bundle: CGPBundle,
    graph_store: GraphStore,
    protein_embeddings: np.ndarray,
    val_pairs: np.ndarray,
    known_positive_pairs: set[Tuple[int, int]],
    device: torch.device,
    rng: np.random.Generator,
    max_edges: int,
    negative_ratio: int,
    repeats: int,
    seed: int,
) -> Dict[str, float]:
    protein_genes = np.where(bundle.has_protein)[0].astype(np.int64)
    if val_pairs.shape[0] == 0 or protein_genes.size < 2:
        return {}
    pairs = val_pairs
    if len(pairs) > int(max_edges):
        pairs = pairs[rng.choice(len(pairs), size=int(max_edges), replace=False)]
    top10: List[float] = []
    mrr: List[float] = []
    for repeat in range(int(repeats)):
        rep_rng = np.random.default_rng(int(seed) + repeat * 100003 + int(negative_ratio) * 17)
        query_compounds = pairs[:, 0].astype(np.int64)
        pos_genes = pairs[:, 1].astype(np.int64)
        candidates = np.zeros((len(pairs), int(negative_ratio) + 1), dtype=np.int64)
        candidates[:, 0] = pos_genes
        for i, (c, g) in enumerate(pairs.tolist()):
            negs: List[int] = []
            while len(negs) < int(negative_ratio):
                draw = rep_rng.choice(protein_genes, size=int(negative_ratio) + 16, replace=False)
                for cand in draw.tolist():
                    if int(cand) == int(g) or (int(c), int(cand)) in known_positive_pairs:
                        continue
                    negs.append(int(cand))
                    if len(negs) >= int(negative_ratio):
                        break
            candidates[i, 1:] = np.asarray(negs[: int(negative_ratio)], dtype=np.int64)
        unique_compounds = np.asarray(sorted(set(query_compounds.tolist())), dtype=np.int64)
        unique_genes = np.asarray(sorted(set(candidates.reshape(-1).tolist())), dtype=np.int64)
        c_pos = {int(v): i for i, v in enumerate(unique_compounds.tolist())}
        g_pos = {int(v): i for i, v in enumerate(unique_genes.tolist())}
        with torch.no_grad():
            zc = model.encode_compound_graphs(graph_store.get_many(unique_compounds.tolist()), device)
            zg = target_protein_embeddings(model, protein_embeddings, unique_genes, device)
            score = (zc @ zg.T).detach().cpu().numpy()
        ranks = []
        for i, c in enumerate(query_compounds.tolist()):
            row = c_pos[int(c)]
            cols = [g_pos[int(x)] for x in candidates[i].tolist()]
            scores = score[row, cols]
            ranks.append(int(1 + np.sum(scores[1:] > scores[0])))
        arr = np.asarray(ranks, dtype=np.int64)
        top10.append(float(np.mean(arr <= 10)))
        mrr.append(float(np.mean(1.0 / arr)))
    return {
        "val_target_binary_Top10_1to100": float(np.mean(top10)),
        "val_target_binary_MRR_1to100": float(np.mean(mrr)),
        "val_target_binary_num_edges": int(len(pairs)),
    }


def profile_bridge_loss(
    z_compound_profile: torch.Tensor,
    z_gene_profile: torch.Tensor,
    compound_profile_x: torch.Tensor,
    gene_profile_x: torch.Tensor,
    temperature: float,
    target_temperature: float,
    topk: int,
) -> torch.Tensor:
    if z_compound_profile.shape[0] == 0 or z_gene_profile.shape[0] == 0:
        return z_compound_profile.new_tensor(0.0)
    latent_logits = z_compound_profile @ z_gene_profile.T / float(temperature)
    with torch.no_grad():
        raw_c = F.normalize(compound_profile_x.float(), dim=-1)
        raw_g = F.normalize(gene_profile_x.float(), dim=-1)
        target_logits = raw_c @ raw_g.T / float(target_temperature)
        k = int(topk)
        if k > 0 and k < target_logits.shape[1]:
            row_keep = torch.topk(target_logits, k=k, dim=1).indices
            row_mask = torch.ones_like(target_logits, dtype=torch.bool)
            row_mask.scatter_(1, row_keep, False)
            row_target_logits = target_logits.masked_fill(row_mask, -1e4)
        else:
            row_target_logits = target_logits
        if k > 0 and k < target_logits.shape[0]:
            col_keep = torch.topk(target_logits, k=k, dim=0).indices
            col_mask = torch.ones_like(target_logits, dtype=torch.bool)
            col_mask.scatter_(0, col_keep, False)
            col_target_logits = target_logits.masked_fill(col_mask, -1e4).T
        else:
            col_target_logits = target_logits.T
        row_targets = F.softmax(row_target_logits, dim=1)
        col_targets = F.softmax(col_target_logits, dim=1)
    row_loss = -(row_targets * F.log_softmax(latent_logits, dim=1)).sum(dim=1).mean()
    col_loss = -(col_targets * F.log_softmax(latent_logits.T, dim=1)).sum(dim=1).mean()
    return 0.5 * (row_loss + col_loss)


def gene_protein_tensor(entities: Any, protein_embeddings: np.ndarray, entity_rows: np.ndarray, device: torch.device) -> torch.Tensor:
    idx = entities.iloc[entity_rows]["source_gene_index"].to_numpy(dtype=np.int64)
    return tensor_from_numpy(protein_embeddings[idx], device)


def quick_val_compound(
    model: TriIntrinsicGNNESM,
    bundle: CGPBundle,
    graph_store: GraphStore,
    split: Dict[str, Any],
    norm: Dict[str, Any],
    device: torch.device,
    rng: np.random.Generator,
    max_queries: int,
    negative_ratio: int,
    repeats: int,
    seed: int,
) -> Dict[str, float]:
    c_val = np.asarray(split["compound"]["val"], dtype=np.int64)
    if len(c_val) > max_queries:
        c_val = rng.choice(c_val, size=max_queries, replace=False)
    if len(c_val) < 2:
        return {}
    model.eval()
    with torch.no_grad():
        z_c = model.encode_compound_graphs(graph_store.get_many(c_val.tolist()), device).detach().cpu().numpy().astype(np.float32)
        z_p = model.encode_profile(compound_profile_batch(bundle, c_val, device, norm), source_ids(len(c_val), 0, device)).detach().cpu().numpy().astype(np.float32)
    ranks, gaps = retrieval_ranks_single_positive(z_c, z_p, np.arange(len(c_val), dtype=np.int64))
    metrics = metrics_from_ranks(ranks, gaps)
    c2p_sampled = sampled_single_positive_metrics(z_c, z_p, negative_ratio, repeats, seed)
    p2c_sampled = sampled_single_positive_metrics(z_p, z_c, negative_ratio, repeats, seed + 3001)
    return {
        "val_compound_to_profile_Recall@10": metrics["Recall@10"],
        "val_compound_to_profile_MRR": metrics["MRR"],
        "val_compound_c2profile_Top10": c2p_sampled["Top10_accuracy_mean"],
        "val_compound_profile2c_Top10": p2c_sampled["Top10_accuracy_mean"],
        "val_compound_c2profile_MRR": c2p_sampled["MRR_mean"],
        "val_compound_profile2c_MRR": p2c_sampled["MRR_mean"],
        "val_compound_alignment_score": 0.5
        * float(c2p_sampled["Top10_accuracy_mean"] + p2c_sampled["Top10_accuracy_mean"]),
    }


def sampled_single_positive_metrics(
    query: np.ndarray,
    gallery: np.ndarray,
    negative_ratio: int,
    repeats: int,
    seed: int,
) -> Dict[str, float]:
    n = int(min(query.shape[0], gallery.shape[0]))
    if n < 2:
        return {
            "Top10_accuracy_mean": 0.0,
            "MRR_mean": 0.0,
            "candidate_count": int(n),
            "num_repeats": int(repeats),
            "num_queries": int(n),
        }
    n_neg = min(int(negative_ratio), n - 1)
    all_idx = np.arange(n, dtype=np.int64)
    top10: List[float] = []
    mrr: List[float] = []
    for repeat in range(int(repeats)):
        rng = np.random.default_rng(int(seed) + repeat * 100003 + n_neg * 17)
        ranks = np.zeros(n, dtype=np.int64)
        for i in range(n):
            if n_neg == n - 1:
                neg = all_idx[all_idx != i]
            else:
                neg = rng.choice(all_idx[all_idx != i], size=n_neg, replace=False)
            candidates = np.concatenate([np.asarray([i], dtype=np.int64), neg])
            scores = gallery[candidates] @ query[i]
            ranks[i] = int(1 + np.sum(scores[1:] > scores[0]))
        top10.append(float(np.mean(ranks <= 10)))
        mrr.append(float(np.mean(1.0 / ranks)))
    return {
        "Top10_accuracy_mean": float(np.mean(top10)),
        "Top10_accuracy_std": float(np.std(top10, ddof=0)),
        "MRR_mean": float(np.mean(mrr)),
        "MRR_std": float(np.std(mrr, ddof=0)),
        "candidate_count": int(n_neg + 1),
        "num_repeats": int(repeats),
        "num_queries": int(n),
    }


def set_epoch_lr(optimizer: torch.optim.Optimizer, base_lr: float, epoch: int, total_epochs: int, args: argparse.Namespace) -> float:
    if args.lr_scheduler == "none":
        return float(optimizer.param_groups[0]["lr"])
    if args.lr_scheduler != "cosine":
        raise ValueError(f"Unsupported lr_scheduler: {args.lr_scheduler}")
    warmup = max(0, int(args.warmup_epochs))
    if warmup > 0 and epoch <= warmup:
        lr = float(base_lr) * float(epoch) / float(warmup)
    else:
        denom = max(1, int(total_epochs) - warmup)
        progress = min(1.0, max(0.0, float(epoch - warmup) / float(denom)))
        lr = float(args.min_lr) + 0.5 * (float(base_lr) - float(args.min_lr)) * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return float(lr)


def load_partial_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    prefix_rules: Sequence[Tuple[str, str]],
) -> Dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    source_state = payload.get("model_state_dict", payload)
    target_state = model.state_dict()
    matched: Dict[str, torch.Tensor] = {}
    skipped_shape: List[Dict[str, Any]] = []
    skipped_missing = 0
    for src_key, value in source_state.items():
        dst_key = None
        for src_prefix, dst_prefix in prefix_rules:
            if src_key.startswith(src_prefix):
                dst_key = f"{dst_prefix}{src_key[len(src_prefix):]}"
                break
        if dst_key is None:
            continue
        if dst_key not in target_state:
            skipped_missing += 1
            continue
        if tuple(target_state[dst_key].shape) != tuple(value.shape):
            skipped_shape.append(
                {
                    "source_key": src_key,
                    "target_key": dst_key,
                    "source_shape": list(value.shape),
                    "target_shape": list(target_state[dst_key].shape),
                }
            )
            continue
        matched[dst_key] = value
    target_state.update(matched)
    model.load_state_dict(target_state, strict=True)
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_best_score": payload.get("best_score"),
        "loaded_tensors": int(len(matched)),
        "skipped_missing": int(skipped_missing),
        "skipped_shape": skipped_shape[:20],
        "skipped_shape_count": int(len(skipped_shape)),
    }


def pairwise_cosine_mean_np(x: np.ndarray) -> float:
    if x.shape[0] < 2:
        return 1.0
    z = x.astype(np.float32, copy=False)
    z = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-12, None)
    sim = z @ z.T
    upper = sim[np.triu_indices(sim.shape[0], k=1)]
    return float(np.nanmean(upper)) if upper.size else 1.0


def compute_entity_replicate_reliability(
    features: np.ndarray,
    entity_to_reps: Sequence[np.ndarray],
    num_entities: int,
) -> np.ndarray:
    reliability = np.full(int(num_entities), np.nan, dtype=np.float32)
    for entity_idx in range(int(num_entities)):
        rep_rows = np.asarray(entity_to_reps[entity_idx], dtype=np.int64)
        if rep_rows.size == 0:
            continue
        reliability[entity_idx] = pairwise_cosine_mean_np(np.asarray(features[rep_rows], dtype=np.float32))
    return reliability


def entity_mean_profiles(
    features: np.ndarray,
    entity_to_reps: Sequence[np.ndarray],
    entity_rows: np.ndarray,
    entity_mods: np.ndarray,
    norm: Dict[str, Any],
) -> np.ndarray:
    prof = []
    for ent in entity_rows.astype(int).tolist():
        reps = entity_to_reps[ent]
        mod = np.full(len(reps), int(entity_mods[ent]), dtype=np.int64)
        prof.append(transform_feature_rows(features, reps, mod, norm).mean(axis=0))
    return np.vstack(prof).astype(np.float32)


def encode_target_profile_keys(
    model: TriIntrinsicGNNESM,
    gene_features: np.ndarray,
    entity_to_reps: Sequence[np.ndarray],
    target_profile_entity_index: Dict[Tuple[int, int], np.ndarray],
    keys: Sequence[Tuple[int, int]],
    gene_norm: Dict[str, Any],
    profile_source_mode: str,
    rng: np.random.Generator,
    device: torch.device,
) -> torch.Tensor:
    entity_rows = np.asarray(
        [rng.choice(target_profile_entity_index[(int(g), int(m))]) for g, m in keys],
        dtype=np.int64,
    )
    modality_np = np.asarray([int(m) for _g, m in keys], dtype=np.int64)
    rep_rows = sample_replicate_features(entity_rows, entity_to_reps, rng)
    modality = torch.from_numpy(modality_np).long().to(device)
    return model.encode_profile(
        profile_tensor(gene_features, rep_rows, modality_np, gene_norm, device),
        gene_profile_source_ids(modality, profile_source_mode),
    )


def quick_val_gene(
    model: TriIntrinsicGNNESM,
    entities: Any,
    features: np.ndarray,
    protein_embeddings: np.ndarray,
    val_entities: np.ndarray,
    entity_to_reps: Sequence[np.ndarray],
    entity_mods: np.ndarray,
    entity_gene_codes: np.ndarray,
    norm: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
) -> Dict[str, float]:
    rows = np.asarray(val_entities, dtype=np.int64)
    if len(rows) > args.val_max_gene_entities:
        rows = rng.choice(rows, size=args.val_max_gene_entities, replace=False)
    if len(rows) < 2:
        return {}
    model.eval()
    with torch.no_grad():
        mod_np = entity_mods[rows]
        mod = torch.from_numpy(mod_np.astype(np.int64)).long().to(device)
        z_prot = model.encode_protein(gene_protein_tensor(entities, protein_embeddings, rows, device), mod).detach().cpu().numpy().astype(np.float32)
        prof = entity_mean_profiles(features, entity_to_reps, rows, entity_mods, norm)
        z_prof = model.encode_profile(
            tensor_from_numpy(prof, device),
            gene_profile_source_ids(mod, args.gene_profile_source_mode),
        ).detach().cpu().numpy().astype(np.float32)
    sampled = sampled_metrics(
        z_prot,
        z_prof,
        entity_gene_codes[rows],
        entity_mods[rows],
        [args.val_negative_ratio],
        args.val_repeats,
        args.seed,
        bool(args.mask_crossmod_negatives and not args.disable_mask_crossmod_negatives),
        bool(args.mask_same_gene_negatives),
    )
    p2 = sampled["protein_to_profile"][f"1:{args.val_negative_ratio}"]
    q2 = sampled["profile_to_protein"][f"1:{args.val_negative_ratio}"]
    ranks, gaps = retrieval_ranks_single_positive(z_prot, z_prof, np.arange(len(rows), dtype=np.int64))
    full = metrics_from_ranks(ranks, gaps)
    return {
        "val_gene_p2profile_Top10": p2.get("Top10_accuracy_mean"),
        "val_gene_profile2p_Top10": q2.get("Top10_accuracy_mean"),
        "val_gene_alignment_score": 0.5 * float((p2.get("Top10_accuracy_mean") or 0.0) + (q2.get("Top10_accuracy_mean") or 0.0)),
        "val_gene_full_p2profile_Recall@10": full["Recall@10"],
        "val_gene_full_p2profile_MRR": full["MRR"],
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.mask_crossmod_negatives = bool(args.mask_crossmod_negatives and not args.disable_mask_crossmod_negatives)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    if args.amp and device.type == "cuda":
        raise SystemExit("AMP is disabled for this GNN tri-modal trainer because graph message passing has dtype-sensitive index_add operations.")

    cgp_bundle = CGPBundle(args.cgp_data_dir, args.cgp_protein_embedding_dir)
    intrinsic_split = load_intrinsic_split(args.intrinsic_split_path)
    graph_store = GraphStore(smiles_list(cgp_bundle))
    gene_entities, gene_replicates, gene_features, gene_protein_embeddings, gene_splits = load_gene_data(args)
    entity_to_reps = build_entity_replicate_index(gene_replicates, len(gene_entities))
    entity_mods = modality_ids(gene_entities)
    entity_gene_codes = gene_codes(gene_entities)
    target_profile_entity_index = build_target_profile_entity_index(gene_entities, entity_to_reps)
    train_gene_entities = eligible_entities(gene_entities, gene_splits["cold_gene"]["train"], gene_protein_embeddings)
    val_gene_entities = eligible_entities(gene_entities, gene_splits["cold_gene"]["val"], gene_protein_embeddings)
    train_compounds = np.asarray(intrinsic_split["compound"]["train"], dtype=np.int64)
    train_compounds = train_compounds[
        cgp_bundle.has_structure[train_compounds]
        & cgp_bundle.has_compound_profile[train_compounds]
        & (cgp_bundle.profile_rows_for_compounds(train_compounds) >= 0)
    ]
    known_target_pairs = load_known_positive_pairs(cgp_bundle, args.target_known_positive_file)
    target_train_pairs = load_target_split_pairs(cgp_bundle, args.target_split, "train")
    target_val_pairs = load_target_split_pairs(cgp_bundle, args.target_split, "val")
    target_train_pairs = target_train_pairs[
        (target_train_pairs[:, 0] < len(cgp_bundle.compounds))
        & (target_train_pairs[:, 1] < len(cgp_bundle.genes))
        & cgp_bundle.has_structure[target_train_pairs[:, 0]]
        & cgp_bundle.has_protein[target_train_pairs[:, 1]]
        & (target_train_pairs[:, 1] < gene_protein_embeddings.shape[0])
    ] if target_train_pairs.size else target_train_pairs
    target_val_pairs = target_val_pairs[
        (target_val_pairs[:, 0] < len(cgp_bundle.compounds))
        & (target_val_pairs[:, 1] < len(cgp_bundle.genes))
        & cgp_bundle.has_structure[target_val_pairs[:, 0]]
        & cgp_bundle.has_protein[target_val_pairs[:, 1]]
        & (target_val_pairs[:, 1] < gene_protein_embeddings.shape[0])
    ] if target_val_pairs.size else target_val_pairs
    typed_train_pairs = load_typed_target_pairs(cgp_bundle, args.target_edge_file, args.target_split, "train")
    if typed_train_pairs.size:
        keep_typed = []
        for i, row in enumerate(typed_train_pairs.tolist()):
            gene_idx = int(row[1])
            mod_id = int(row[2])
            if args.typed_direction_control == "wrong":
                mod_id = MODALITY_TO_ID["orf"] if mod_id == MODALITY_TO_ID["crispr"] else MODALITY_TO_ID["crispr"]
            if (gene_idx, mod_id) in target_profile_entity_index:
                keep_typed.append(i)
        typed_train_pairs = typed_train_pairs[np.asarray(keep_typed, dtype=np.int64)] if keep_typed else np.zeros((0, 3), dtype=np.int64)
    train_scale_summary: Dict[str, Any] = {
        "requested_train_fraction": float(args.train_fraction),
        "validation_and_test_policy": "unchanged",
    }
    train_compounds, train_scale_summary["compound_train_pool"] = subsample_train_pool(
        train_compounds,
        args.train_fraction,
        args.train_max_compounds,
        int(args.seed) + 101,
    )
    train_gene_entities, train_scale_summary["gene_entity_train_pool"] = subsample_train_pool(
        train_gene_entities,
        args.train_fraction,
        args.train_max_gene_entities,
        int(args.seed) + 202,
    )
    target_train_pairs, train_scale_summary["target_edge_train_pool"] = subsample_train_pool(
        target_train_pairs,
        args.train_fraction,
        args.train_max_target_edges,
        int(args.seed) + 303,
    )
    typed_train_pairs, train_scale_summary["typed_edge_train_pool"] = subsample_train_pool(
        typed_train_pairs,
        args.train_fraction,
        args.train_max_typed_edges,
        int(args.seed) + 404,
    )
    if float(args.min_train_gene_replicate_cosine) > -1.0:
        entity_reliability = compute_entity_replicate_reliability(
            gene_features,
            entity_to_reps,
            len(gene_entities),
        )
        rel = entity_reliability[train_gene_entities.astype(np.int64)]
        keep = np.isfinite(rel) & (rel >= float(args.min_train_gene_replicate_cosine))
        before = int(len(train_gene_entities))
        train_gene_entities = train_gene_entities[keep]
        if len(train_gene_entities) == 0:
            raise SystemExit(
                f"No gene training entities remain after min_train_gene_replicate_cosine={args.min_train_gene_replicate_cosine}"
            )
        kept_rel = entity_reliability[train_gene_entities.astype(np.int64)]
        train_scale_summary["gene_replicate_reliability_filter"] = {
            "min_train_gene_replicate_cosine": float(args.min_train_gene_replicate_cosine),
            "before": before,
            "after": int(len(train_gene_entities)),
            "kept_fraction": float(len(train_gene_entities) / max(before, 1)),
            "kept_mean": float(np.nanmean(kept_rel)),
            "kept_median": float(np.nanmedian(kept_rel)),
        }
    if args.smoke_test:
        train_compounds = train_compounds[: min(len(train_compounds), 2048)]
        train_gene_entities = train_gene_entities[: min(len(train_gene_entities), 2048)]
        val_gene_entities = val_gene_entities[: min(len(val_gene_entities), 256)]
        target_train_pairs = target_train_pairs[: min(len(target_train_pairs), 2048)]
        target_val_pairs = target_val_pairs[: min(len(target_val_pairs), 256)]
        typed_train_pairs = typed_train_pairs[: min(len(typed_train_pairs), 1024)]
        args.epochs = min(args.epochs, 2)
        args.val_every = 1
        args.val_max_compounds = min(args.val_max_compounds, 256)
        args.val_max_gene_entities = min(args.val_max_gene_entities, 256)
        args.val_max_target_edges = min(args.val_max_target_edges, 128)

    if cgp_bundle.profile_dim != int(gene_features.shape[1]):
        raise SystemExit(f"Profile dim mismatch: compound={cgp_bundle.profile_dim}, gene={gene_features.shape[1]}")
    gene_norm = fit_profile_normalizer(gene_features, entity_to_reps, train_gene_entities, entity_mods, args.profile_norm)
    compound_norm = fit_compound_profile_normalizer(cgp_bundle, train_compounds, args.compound_profile_norm)

    model = TriIntrinsicGNNESM(
        compound_profile_dim=cgp_bundle.profile_dim,
        gene_profile_dim=int(gene_features.shape[1]),
        protein_dim=int(gene_protein_embeddings.shape[1]),
        embed_dim=args.embed_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_layers=args.gnn_layers,
        protein_hidden_dims=parse_hidden_dims(args.protein_hidden_dims),
        profile_hidden_dims=parse_hidden_dims(args.profile_hidden_dims),
        modality_context_dim=args.modality_context_dim,
        dropout=args.dropout,
        profile_source_adapters=bool(args.profile_source_adapters),
        profile_source_embedding=not bool(args.disable_profile_source_embedding),
        profile_source_dropout=float(args.profile_source_dropout),
        num_profile_sources=1 + len(MODALITY_TO_ID) if args.gene_profile_source_mode == "modality" else 2,
    ).to(device)
    init_summary: Dict[str, Any] = {}
    if args.init_compound_checkpoint is not None:
        init_summary["compound"] = load_partial_checkpoint(
            model,
            args.init_compound_checkpoint,
            [("compound_encoder.", "compound_encoder.")],
        )
    if args.init_gene_checkpoint is not None:
        init_summary["gene"] = load_partial_checkpoint(
            model,
            args.init_gene_checkpoint,
            [
                ("modality_embedding.", "modality_embedding."),
                ("protein_encoder.", "protein_encoder."),
                ("profile_source_embedding.", "profile_encoder.source_embedding."),
                ("profile_encoder.", "profile_encoder.encoder."),
            ],
        )
    if args.init_tri_checkpoint is not None:
        init_summary["tri"] = load_partial_checkpoint(
            model,
            args.init_tri_checkpoint,
            [("", "")],
        )
    bad_shape_init = {
        name: summary
        for name, summary in init_summary.items()
        if int(summary.get("skipped_shape_count", 0)) > 0
    }
    if bad_shape_init and not bool(args.allow_partial_init):
        raise RuntimeError(
            "Checkpoint initialization skipped shape-mismatched tensors. "
            "This is not allowed by default because it invalidates controlled initialization ablations. "
            f"Pass --allow_partial_init only for exploratory warm starts. Details: {bad_shape_init}"
        )
    teacher_model = None
    if float(args.lambda_anchor) > 0:
        teacher_model = copy.deepcopy(model).to(device).eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    ema_model = copy.deepcopy(model).to(device).eval() if not args.disable_ema else model
    if not args.disable_ema:
        for p in ema_model.parameters():
            p.requires_grad = False

    ckpt_dir = args.checkpoint_dir / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"{args.run_name}_train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    steps_per_epoch = int(args.steps_per_epoch or np.ceil(max(len(train_compounds), len(train_gene_entities)) / max(1, args.compound_batch_size)))
    config = vars(args).copy()
    config.update(
        {
            "mode": "tri_intrinsic_gnn_esm650_shared_profile",
            "compound_encoder_reference": "intrinsic_base_gnn_lr1e3_broad_1000_align architecture/hyperparameters",
            "protein_encoder_reference": "gene_mocop_esm650_l33_mean_cls_cold_gene architecture/hyperparameters",
            "from_scratch_joint_training": not bool(init_summary),
            "init_summary": init_summary,
            "uses_target_or_link_labels_for_training": bool(
                float(args.lambda_target) > 0 or float(args.lambda_typed) > 0 or float(args.lambda_direction) > 0
            ),
            "gene_loss": "weighted_multipositive_clip"
            if (args.same_gene_same_modality_positive_weight > 0 or args.same_gene_crossmod_positive_weight > 0)
            else "masked_clip",
            "compound_profile_dim": int(cgp_bundle.profile_dim),
            "gene_profile_dim": int(gene_features.shape[1]),
            "gene_protein_dim": int(gene_protein_embeddings.shape[1]),
            "num_profile_sources": int(1 + len(MODALITY_TO_ID) if args.gene_profile_source_mode == "modality" else 2),
            "num_train_compounds": int(len(train_compounds)),
            "num_train_gene_entities": int(len(train_gene_entities)),
            "num_val_gene_entities": int(len(val_gene_entities)),
            "training_scale": train_scale_summary,
            "target_prior": {
                "target_split": args.target_split,
                "target_edge_file": args.target_edge_file,
                "target_known_positive_file": args.target_known_positive_file,
                "num_target_train_edges": int(len(target_train_pairs)),
                "num_target_val_edges": int(len(target_val_pairs)),
                "num_typed_train_edges": int(len(typed_train_pairs)),
                "typed_direction_control": args.typed_direction_control,
                "target_query_modes": parse_mode_list(args.target_query_modes, ["graph", "profile", "fused"]),
                "typed_query_modes": parse_mode_list(args.typed_query_modes, ["graph", "profile", "fused"]),
                "direction_query_mode": args.direction_query_mode,
                "direction_margin": float(args.direction_margin),
                "target_profile_policy": "inhibitory->crispr, activating->orf",
                "target_protein_policy": "average ORF and CRISPR protein-context encodings",
            },
            "intrinsic_split_summary": intrinsic_split.get("summary"),
            "compound_profile_normalizer": normalizer_summary(compound_norm),
            "gene_profile_normalizer": normalizer_summary(gene_norm),
            "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "total_params": int(sum(p.numel() for p in model.parameters())),
        }
    )
    write_json(ckpt_dir / "config.json", config)

    rng = np.random.default_rng(args.seed)
    target_query_modes = parse_mode_list(args.target_query_modes, ["graph", "profile", "fused"])
    typed_query_modes = parse_mode_list(args.typed_query_modes, ["graph", "profile", "fused"])
    best_score = -float("inf")
    best_epoch = 0
    checks_without_improvement = 0
    stopped_early = False
    gene_modality_index = build_gene_modality_entity_index(train_gene_entities, entity_gene_codes, entity_mods)
    for epoch in range(1, args.epochs + 1):
        current_lr = set_epoch_lr(optimizer, args.lr, epoch, args.epochs, args)
        model.train()
        components: Dict[str, List[float]] = {
            "compound_profile": [],
            "gene_profile": [],
            "profile_bridge": [],
            "target": [],
            "typed": [],
            "direction": [],
            "anchor": [],
            "total": [],
        }
        for _ in range(steps_per_epoch):
            c_sel = rng.choice(train_compounds, size=min(args.compound_batch_size, len(train_compounds)), replace=False)
            g_sel = rng.choice(train_gene_entities, size=min(args.gene_batch_size, len(train_gene_entities)), replace=False)
            g_sel = augment_with_same_gene_siblings(
                g_sel,
                entity_gene_codes,
                entity_mods,
                gene_modality_index,
                rng,
                args.same_gene_sibling_batch_prob,
            )
            mod_np = entity_mods[g_sel]
            gene_np = entity_gene_codes[g_sel]
            mod = torch.from_numpy(mod_np.astype(np.int64)).long().to(device)
            gene_code = torch.from_numpy(gene_np.astype(np.int64)).long().to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=bool(args.amp and device.type == "cuda")):
                z_c = model.encode_compound_graphs(graph_store.get_many(c_sel.tolist()), device)
                compound_x = compound_profile_batch(cgp_bundle, c_sel, device, compound_norm)
                z_cp = model.encode_profile(compound_x, source_ids(len(c_sel), 0, device))
                compound_loss = bidirectional_infonce(z_c, z_cp, args.compound_temperature)

                z_p = model.encode_protein(gene_protein_tensor(gene_entities, gene_protein_embeddings, g_sel, device), mod)
                if args.gene_train_profile_mode == "mean":
                    gene_x = tensor_from_numpy(
                        entity_mean_profiles(gene_features, entity_to_reps, g_sel, entity_mods, gene_norm),
                        device,
                    )
                else:
                    rep_rows = sample_replicate_features(g_sel, entity_to_reps, rng)
                    gene_x = profile_tensor(gene_features, rep_rows, mod_np, gene_norm, device)
                z_gp = model.encode_profile(
                    gene_x,
                    gene_profile_source_ids(mod, args.gene_profile_source_mode),
                )
                if args.same_gene_same_modality_positive_weight > 0 or args.same_gene_crossmod_positive_weight > 0:
                    gene_loss = weighted_multipositive_clip_loss(
                        z_p,
                        z_gp,
                        gene_code,
                        mod,
                        args.gene_temperature,
                        args.mask_crossmod_negatives,
                        bool(args.mask_same_gene_negatives),
                        args.same_gene_same_modality_positive_weight,
                        args.same_gene_crossmod_positive_weight,
                    )
                else:
                    gene_loss = masked_clip_loss(
                        z_p,
                        z_gp,
                        gene_code,
                        mod,
                        args.gene_temperature,
                        args.mask_crossmod_negatives,
                        bool(args.mask_same_gene_negatives),
                    )
                if float(args.lambda_profile_bridge) > 0:
                    bridge_loss = profile_bridge_loss(
                        z_cp,
                        z_gp,
                        compound_x,
                        gene_x,
                        args.profile_bridge_temperature,
                        args.profile_bridge_target_temperature,
                        args.profile_bridge_topk,
                    )
                else:
                    bridge_loss = compound_loss.new_tensor(0.0)
                if float(args.lambda_target) > 0 and len(target_train_pairs):
                    t_sel = target_train_pairs[
                        rng.choice(
                            len(target_train_pairs),
                            size=min(int(args.target_batch_size), len(target_train_pairs)),
                            replace=False,
                        )
                    ]
                    if any(mode in {"profile", "fused"} for mode in target_query_modes):
                        keep = cgp_bundle.has_compound_profile[t_sel[:, 0]] & (cgp_bundle.compound_profile_row[t_sel[:, 0]] >= 0)
                        t_sel = t_sel[keep]
                else:
                    t_sel = np.zeros((0, 2), dtype=np.int64)
                if len(t_sel):
                    t_compounds = np.asarray(sorted(set(t_sel[:, 0].astype(int).tolist())), dtype=np.int64)
                    t_genes = np.asarray(sorted(set(t_sel[:, 1].astype(int).tolist())), dtype=np.int64)
                    z_tc_graph = None
                    z_tc_profile = None
                    if any(mode in {"graph", "fused"} for mode in target_query_modes):
                        z_tc_graph = model.encode_compound_graphs(graph_store.get_many(t_compounds.tolist()), device)
                    if any(mode in {"profile", "fused"} for mode in target_query_modes):
                        z_tc_profile = model.encode_profile(
                            compound_profile_batch(cgp_bundle, t_compounds, device, compound_norm),
                            source_ids(len(t_compounds), 0, device),
                        )
                    z_tg = target_protein_embeddings(model, gene_protein_embeddings, t_genes, device)
                    pos, invalid = positive_matrix(t_compounds, t_genes, t_sel, device, known_target_pairs)
                    t_losses = []
                    for mode in target_query_modes:
                        q = select_query_embeddings(
                            mode,
                            z_tc_graph if z_tc_graph is not None else z_tc_profile,
                            z_tc_profile if z_tc_profile is not None else z_tc_graph,
                        )
                        t_losses.append(multi_positive_contrastive_loss(q, z_tg, pos, args.target_temperature, invalid))
                    target_loss = torch.stack(t_losses).mean() if t_losses else compound_loss.new_tensor(0.0)
                else:
                    target_loss = compound_loss.new_tensor(0.0)

                direction_loss = compound_loss.new_tensor(0.0)
                if (float(args.lambda_typed) > 0 or float(args.lambda_direction) > 0) and len(typed_train_pairs):
                    y_sel = typed_train_pairs[
                        rng.choice(
                            len(typed_train_pairs),
                            size=min(int(args.typed_batch_size), len(typed_train_pairs)),
                            replace=False,
                        )
                    ].copy()
                    if args.typed_direction_control == "wrong":
                        y_sel[:, 2] = np.where(
                            y_sel[:, 2] == MODALITY_TO_ID["crispr"],
                            MODALITY_TO_ID["orf"],
                            MODALITY_TO_ID["crispr"],
                        )
                    keep = np.asarray(
                        [(int(g), int(m)) in target_profile_entity_index for _c, g, m in y_sel.tolist()],
                        dtype=bool,
                    )
                    y_sel = y_sel[keep]
                    if any(mode in {"profile", "fused"} for mode in typed_query_modes) or args.direction_query_mode in {"profile", "fused"}:
                        keep = cgp_bundle.has_compound_profile[y_sel[:, 0]] & (cgp_bundle.compound_profile_row[y_sel[:, 0]] >= 0)
                        y_sel = y_sel[keep]
                    if len(y_sel):
                        y_compounds = np.asarray(sorted(set(y_sel[:, 0].astype(int).tolist())), dtype=np.int64)
                        y_keys = sorted({(int(g), int(m)) for _c, g, m in y_sel.tolist()})
                        y_row = {int(c): i for i, c in enumerate(y_compounds.tolist())}
                        y_col = {key: i for i, key in enumerate(y_keys)}
                        y_pos = torch.zeros((len(y_compounds), len(y_keys)), dtype=torch.bool, device=device)
                        for c, g, m in y_sel.tolist():
                            y_pos[y_row[int(c)], y_col[(int(g), int(m))]] = True
                        z_yc_graph = None
                        z_yc_profile = None
                        if any(mode in {"graph", "fused"} for mode in typed_query_modes) or args.direction_query_mode in {"graph", "fused"}:
                            z_yc_graph = model.encode_compound_graphs(graph_store.get_many(y_compounds.tolist()), device)
                        if any(mode in {"profile", "fused"} for mode in typed_query_modes) or args.direction_query_mode in {"profile", "fused"}:
                            z_yc_profile = model.encode_profile(
                                compound_profile_batch(cgp_bundle, y_compounds, device, compound_norm),
                                source_ids(len(y_compounds), 0, device),
                            )
                        z_yg = encode_target_profile_keys(
                            model,
                            gene_features,
                            entity_to_reps,
                            target_profile_entity_index,
                            y_keys,
                            gene_norm,
                            args.gene_profile_source_mode,
                            rng,
                            device,
                        )
                        typed_losses = []
                        if float(args.lambda_typed) > 0:
                            for mode in typed_query_modes:
                                q = select_query_embeddings(
                                    mode,
                                    z_yc_graph if z_yc_graph is not None else z_yc_profile,
                                    z_yc_profile if z_yc_profile is not None else z_yc_graph,
                                )
                                typed_losses.append(multi_positive_contrastive_loss(q, z_yg, y_pos, args.typed_temperature))
                        typed_loss = torch.stack(typed_losses).mean() if typed_losses else compound_loss.new_tensor(0.0)
                        if float(args.lambda_direction) > 0:
                            opposite = {MODALITY_TO_ID["crispr"]: MODALITY_TO_ID["orf"], MODALITY_TO_ID["orf"]: MODALITY_TO_ID["crispr"]}
                            dir_rows = [
                                (int(c), int(g), int(m), int(opposite[int(m)]))
                                for c, g, m in y_sel.tolist()
                                if int(m) in opposite and (int(g), int(opposite[int(m)])) in target_profile_entity_index
                            ]
                            if dir_rows:
                                dir_compounds = np.asarray([x[0] for x in dir_rows], dtype=np.int64)
                                matched_keys = [(x[1], x[2]) for x in dir_rows]
                                opposite_keys = [(x[1], x[3]) for x in dir_rows]
                                z_dir_graph = None
                                z_dir_profile = None
                                if args.direction_query_mode in {"graph", "fused"}:
                                    z_dir_graph = model.encode_compound_graphs(graph_store.get_many(dir_compounds.tolist()), device)
                                if args.direction_query_mode in {"profile", "fused"}:
                                    z_dir_profile = model.encode_profile(
                                        compound_profile_batch(cgp_bundle, dir_compounds, device, compound_norm),
                                        source_ids(len(dir_compounds), 0, device),
                                    )
                                z_dir_query = select_query_embeddings(
                                    args.direction_query_mode,
                                    z_dir_graph if z_dir_graph is not None else z_dir_profile,
                                    z_dir_profile if z_dir_profile is not None else z_dir_graph,
                                )
                                z_matched = encode_target_profile_keys(
                                    model,
                                    gene_features,
                                    entity_to_reps,
                                    target_profile_entity_index,
                                    matched_keys,
                                    gene_norm,
                                    args.gene_profile_source_mode,
                                    rng,
                                    device,
                                )
                                z_opposite = encode_target_profile_keys(
                                    model,
                                    gene_features,
                                    entity_to_reps,
                                    target_profile_entity_index,
                                    opposite_keys,
                                    gene_norm,
                                    args.gene_profile_source_mode,
                                    rng,
                                    device,
                                )
                                direction_loss = pairwise_direction_margin_loss(
                                    z_dir_query,
                                    z_matched,
                                    z_opposite,
                                    args.direction_margin,
                                )
                    else:
                        typed_loss = compound_loss.new_tensor(0.0)
                else:
                    typed_loss = compound_loss.new_tensor(0.0)

                if teacher_model is not None:
                    with torch.no_grad():
                        t_z_c = teacher_model.encode_compound_graphs(graph_store.get_many(c_sel.tolist()), device)
                        t_z_cp = teacher_model.encode_profile(compound_x, source_ids(len(c_sel), 0, device))
                        t_z_p = teacher_model.encode_protein(gene_protein_tensor(gene_entities, gene_protein_embeddings, g_sel, device), mod)
                        t_z_gp = teacher_model.encode_profile(
                            gene_x,
                            gene_profile_source_ids(mod, args.gene_profile_source_mode),
                        )
                    anchor_loss = cosine_anchor_loss([z_c, z_cp, z_p, z_gp], [t_z_c, t_z_cp, t_z_p, t_z_gp])
                else:
                    anchor_loss = compound_loss.new_tensor(0.0)
                loss = (
                    args.lambda_compound_profile * compound_loss
                    + args.lambda_gene_profile * gene_loss
                    + args.lambda_profile_bridge * bridge_loss
                    + args.lambda_target * target_loss
                    + args.lambda_typed * typed_loss
                    + args.lambda_direction * direction_loss
                    + args.lambda_anchor * anchor_loss
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            if not args.disable_ema:
                update_ema(model, ema_model, args.ema_decay)
            components["compound_profile"].append(float(compound_loss.detach().cpu().item()))
            components["gene_profile"].append(float(gene_loss.detach().cpu().item()))
            components["profile_bridge"].append(float(bridge_loss.detach().cpu().item()))
            components["target"].append(float(target_loss.detach().cpu().item()))
            components["typed"].append(float(typed_loss.detach().cpu().item()))
            components["direction"].append(float(direction_loss.detach().cpu().item()))
            components["anchor"].append(float(anchor_loss.detach().cpu().item()))
            components["total"].append(float(loss.detach().cpu().item()))

        eval_model = ema_model if not args.disable_ema else model
        val: Dict[str, Any] = {}
        if epoch % args.val_every == 0 or epoch == args.epochs:
            val.update(
                quick_val_compound(
                    eval_model,
                    cgp_bundle,
                    graph_store,
                    intrinsic_split,
                    compound_norm,
                    device,
                    rng,
                    args.val_max_compounds,
                    args.val_negative_ratio,
                    args.val_repeats,
                    args.seed,
                )
            )
            val.update(
                quick_val_gene(
                    eval_model,
                    gene_entities,
                    gene_features,
                    gene_protein_embeddings,
                    val_gene_entities,
                    entity_to_reps,
                    entity_mods,
                    entity_gene_codes,
                    gene_norm,
                    args,
                    device,
                    rng,
                )
            )
            vals = [
                val.get("val_compound_alignment_score"),
                val.get("val_gene_alignment_score"),
            ]
            vals = [float(v) for v in vals if v is not None]
            if vals:
                val["val_tri_alignment_score"] = float(np.mean(vals))
            compound_score = val.get("val_compound_alignment_score")
            gene_score = val.get("val_gene_alignment_score")
            if compound_score is not None and gene_score is not None:
                c = max(float(compound_score), 0.0)
                g = max(float(gene_score), 0.0)
                val["val_tri_hmean_alignment_score"] = float(2.0 * c * g / max(c + g, 1e-12))
                val["val_tri_min_alignment_score"] = float(min(c, g))
            if float(args.lambda_target) > 0 and len(target_val_pairs):
                val.update(
                    quick_val_target_binary(
                        eval_model,
                        cgp_bundle,
                        graph_store,
                        gene_protein_embeddings,
                        target_val_pairs,
                        known_target_pairs,
                        device,
                        rng,
                        args.val_max_target_edges,
                        args.val_negative_ratio,
                        args.val_repeats,
                        args.seed,
                    )
                )
            if val.get("val_tri_hmean_alignment_score") is not None and val.get("val_target_binary_Top10_1to100") is not None:
                a = max(float(val["val_tri_hmean_alignment_score"]), 0.0)
                t = max(float(val["val_target_binary_Top10_1to100"]), 0.0)
                val["val_joint_target_alignment_score"] = float(2.0 * a * t / max(a + t, 1e-12))

        rec = {
            "epoch": int(epoch),
            "steps_per_epoch": int(steps_per_epoch),
            "lr": float(current_lr),
            "train_compound_profile": float(np.mean(components["compound_profile"])),
            "train_gene_profile": float(np.mean(components["gene_profile"])),
            "train_profile_bridge": float(np.mean(components["profile_bridge"])),
            "train_target": float(np.mean(components["target"])),
            "train_typed": float(np.mean(components["typed"])),
            "train_direction": float(np.mean(components["direction"])),
            "train_anchor": float(np.mean(components["anchor"])),
            "train_total": float(np.mean(components["total"])),
            **val,
        }
        append_jsonl(log_path, rec)
        print(json.dumps(rec, sort_keys=True), flush=True)
        score = rec.get(args.selection_metric)
        improved = score is not None and float(score) > best_score
        if improved:
            best_score = float(score)
            best_epoch = int(epoch)
            checks_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": eval_model.state_dict(),
                    "epoch": int(epoch),
                    "best_score": best_score,
                    "selection_metric": args.selection_metric,
                    "config": config,
                    "compound_profile_normalizer": compound_norm,
                    "gene_profile_normalizer": gene_norm,
                },
                ckpt_dir / "best_model.pt",
            )
        elif score is not None and args.early_stopping_patience > 0:
            checks_without_improvement += 1
            if checks_without_improvement >= int(args.early_stopping_patience):
                stopped_early = True
                append_jsonl(
                    log_path,
                    {
                        "epoch": int(epoch),
                        "event": "early_stopping",
                        "best_epoch": int(best_epoch),
                        "best_score": float(best_score),
                        "checks_without_improvement": int(checks_without_improvement),
                    },
                )
                print(
                    json.dumps(
                        {
                            "epoch": int(epoch),
                            "event": "early_stopping",
                            "best_epoch": int(best_epoch),
                            "best_score": float(best_score),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                break

    if not (ckpt_dir / "best_model.pt").exists():
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": int(args.epochs),
                "best_score": best_score,
                "selection_metric": args.selection_metric,
                "config": config,
                "compound_profile_normalizer": compound_norm,
                "gene_profile_normalizer": gene_norm,
            },
            ckpt_dir / "best_model.pt",
        )
    if not args.train_only:
        write_json(
            args.output_dir / f"{args.run_name}_training_summary.json",
            {
                "run_name": args.run_name,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "stopped_early": stopped_early,
            },
        )


if __name__ == "__main__":
    main()
