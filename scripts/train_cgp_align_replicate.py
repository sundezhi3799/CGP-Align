from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from cgp_gnn import GGNNCompoundEncoder, GraphStore, atom_feature_dim, batch_graphs, bond_feature_dim


MODALITY_TO_ID = {"orf": 0, "crispr": 1}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replicate-aware CGP-Align trainer with shared profile space and no compound-gene link supervision."
    )
    p.add_argument("--compound_data_dir", type=Path, required=True)
    p.add_argument("--gene_data_dir", type=Path, required=True)
    p.add_argument("--gene_protein_embedding_dir", type=Path, required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--checkpoint_dir", type=Path, required=True)
    p.add_argument("--log_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--compound_split_name", default="cold_compound")
    p.add_argument("--gene_split_name", default="cold_gene")
    p.add_argument("--init_compound_checkpoint", type=Path)
    p.add_argument("--init_gene_checkpoint", type=Path)
    p.add_argument("--init_orf_gene_checkpoint", type=Path)
    p.add_argument("--init_crispr_gene_checkpoint", type=Path)
    p.add_argument("--resume_checkpoint", type=Path)
    p.add_argument("--disable_staged_training", action="store_true")
    p.add_argument("--profile_source_adapters", action="store_true")
    p.add_argument("--disable_profile_source_embedding", action="store_true")
    p.add_argument("--profile_source_dropout", type=float, default=0.0)
    p.add_argument("--profile_source_mode", choices=["compound_gene_modality", "compound_gene"], default="compound_gene_modality")
    p.add_argument("--profile_norm", choices=["none", "train_zscore"], default="none")
    p.add_argument("--train_replicates_per_entity", type=int, default=2)
    p.add_argument("--train_fraction", type=float, default=1.0, help="Fraction of eligible train entities to use for both branches.")
    p.add_argument("--compound_train_fraction", type=float, default=None, help="Optional compound-specific train entity fraction.")
    p.add_argument("--gene_train_fraction", type=float, default=None, help="Optional gene-specific train entity fraction.")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--gnn_hidden_dim", type=int, default=256)
    p.add_argument("--gnn_layers", type=int, default=6)
    p.add_argument("--gene_hidden_dims", default="512,256")
    p.add_argument("--profile_hidden_dims", default="512,256")
    p.add_argument("--profile_input_layernorm", action="store_true")
    p.add_argument("--profile_mlp_norm", action="store_true")
    p.add_argument("--profile_feature_dropout", type=float, default=0.0)
    p.add_argument("--modality_context_dim", type=int, default=32)
    p.add_argument("--disable_gene_modality_context", action="store_true")
    p.add_argument("--freeze_gene_modality_embedding", action="store_true")
    p.add_argument("--gene_modality_balanced_loss", action="store_true")
    p.add_argument("--gene_modality_soft_balanced_loss", action="store_true")
    p.add_argument("--balanced_gene_batch_sampling", action="store_true")
    p.add_argument("--split_gene_modality_branches", action="store_true")
    p.add_argument("--gene_branch_loss_reduction", choices=["sum", "mean"], default="sum",
                   help="Sum ORF/CRISPR losses for the manuscript objective; mean reproduces historical training.")
    p.add_argument("--split_gene_shared_trunk", action="store_true")
    p.add_argument("--split_gene_distribution_alignment_weight", type=float, default=0.0)
    p.add_argument("--split_gene_distribution_alignment_start_epoch", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--compound_temperature", type=float, default=0.07)
    p.add_argument("--gene_temperature", type=float, default=0.07)
    p.add_argument("--compound_loss_weight", type=float, default=1.0)
    p.add_argument("--gene_loss_weight", type=float, default=1.0)
    p.add_argument("--anchor_weight_stage2", type=float, default=0.05)
    p.add_argument("--anchor_weight_stage3", type=float, default=0.01)
    p.add_argument("--cg_teacher_weight", type=float, default=0.0)
    p.add_argument("--cg_teacher_start_epoch", type=int, default=0)
    p.add_argument("--cg_teacher_warmup_epochs", type=int, default=20)
    p.add_argument("--cg_teacher_temperature", type=float, default=0.05)
    p.add_argument("--cg_student_temperature", type=float, default=0.07)
    p.add_argument("--cg_teacher_batch_topk", type=int, default=32)
    p.add_argument("--cg_teacher_profile_transform", choices=["l2", "none"], default="l2")
    p.add_argument("--cg_teacher_gene_modality_scope", choices=["mixed", "separate"], default="mixed")
    p.add_argument("--cg_teacher_mutual_topk", action="store_true")
    p.add_argument(
        "--cg_teacher_min_similarity",
        type=float,
        default=-2.0,
        help="Optional raw-profile cosine threshold for C-G soft teacher candidates. Values below -1 disable it.",
    )
    p.add_argument("--profile_structure_weight", type=float, default=0.0)
    p.add_argument("--profile_structure_start_epoch", type=int, default=0)
    p.add_argument("--profile_structure_warmup_epochs", type=int, default=20)
    p.add_argument("--profile_structure_temperature", type=float, default=0.05)
    p.add_argument("--profile_structure_student_temperature", type=float, default=0.07)
    p.add_argument("--profile_structure_batch_topk", type=int, default=32)
    p.add_argument("--profile_structure_profile_transform", choices=["l2", "none"], default="l2")
    p.add_argument("--profile_structure_components", default="cc,gg,cg")
    p.add_argument("--profile_structure_target", choices=["branch", "profile"], default="branch")
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--stage1_epochs", type=int, default=20)
    p.add_argument("--stage2_epochs", type=int, default=40)
    p.add_argument("--min_selection_epoch", type=int, default=0)
    p.add_argument("--save_candidate_epochs", default="")
    p.add_argument("--compound_batch_size", type=int, default=256)
    p.add_argument("--gene_batch_size", type=int, default=256)
    p.add_argument("--steps_per_epoch", type=int, default=0)
    p.add_argument("--profile_lr", type=float, default=5e-4)
    p.add_argument("--branch_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--val_every", type=int, default=10)
    p.add_argument("--val_max_compounds", type=int, default=2048)
    p.add_argument("--val_max_gene_entities", type=int, default=2048)
    p.add_argument("--val_negative_ratio", type=int, default=100)
    p.add_argument("--val_repeats", type=int, default=5)
    p.add_argument("--negative_ratios", default="100")
    p.add_argument("--num_repeats", type=int, default=10)
    p.add_argument("--eval_batch_size", type=int, default=1024)
    p.add_argument("--eval_max_replicates_per_entity", type=int, default=0)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--train_only", action="store_true")
    return p.parse_args()


def parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_hidden_dims(text: str) -> List[int]:
    return parse_ints(text)


def resolve_train_fraction(args: argparse.Namespace, branch: str) -> float:
    value = getattr(args, f"{branch}_train_fraction", None)
    fraction = float(getattr(args, "train_fraction", 1.0) if value is None else value)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"{branch} train fraction must be in (0, 1], got {fraction}")
    return fraction


def subsample_train_entities(rows: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if float(fraction) >= 0.999999 or len(rows) <= 1:
        return rows
    n = max(1, int(round(len(rows) * float(fraction))))
    rng = np.random.default_rng(int(seed))
    positions = np.sort(rng.choice(np.arange(len(rows)), size=n, replace=False))
    return rows[positions]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, default=json_default)
        handle.write("\n")


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=json_default) + "\n")


def json_default(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    raise TypeError(f"Unsupported JSON type: {type(x)!r}")


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def tensor_from_numpy(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device=device, dtype=torch.float32)


class MoCoPMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], embed_dim: int, dropout: float, use_norm: bool = False):
        super().__init__()
        layers: List[nn.Module] = []
        prev = int(input_dim)
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, int(hidden)))
            if bool(use_norm):
                layers.append(nn.LayerNorm(int(hidden)))
            layers.extend([nn.GELU(), nn.Dropout(float(dropout))])
            prev = int(hidden)
        layers.append(nn.Linear(prev, int(embed_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class SharedProfileEncoder(nn.Module):
    def __init__(
        self,
        profile_dim: int,
        hidden_dims: Sequence[int],
        embed_dim: int,
        dropout: float,
        use_source_adapters: bool,
        use_source_embedding: bool,
        source_dropout: float,
        num_sources: int,
        input_layernorm: bool = False,
        feature_dropout: float = 0.0,
        mlp_norm: bool = False,
    ):
        super().__init__()
        self.source_dropout = float(source_dropout)
        self.num_sources = int(num_sources)
        self.feature_dropout = nn.Dropout(float(feature_dropout)) if float(feature_dropout) > 0 else None
        self.input_norm = nn.LayerNorm(int(profile_dim)) if bool(input_layernorm) else None
        self.source_embedding = nn.Embedding(self.num_sources, int(profile_dim)) if use_source_embedding else None
        self.adapters = nn.ModuleList()
        if use_source_adapters:
            for _ in range(self.num_sources):
                adapter = nn.Sequential(
                    nn.LayerNorm(int(profile_dim)),
                    nn.Linear(int(profile_dim), int(profile_dim)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(int(profile_dim), int(profile_dim)),
                )
                nn.init.zeros_(adapter[-1].weight)
                nn.init.zeros_(adapter[-1].bias)
                self.adapters.append(adapter)
        self.encoder = MoCoPMLP(int(profile_dim), hidden_dims, int(embed_dim), float(dropout), use_norm=bool(mlp_norm))

    def forward(self, x: torch.Tensor, source_id: torch.Tensor) -> torch.Tensor:
        h = x
        if self.input_norm is not None:
            h = self.input_norm(h)
        if self.feature_dropout is not None:
            h = self.feature_dropout(h)
        source_gate = None
        if self.training and self.source_dropout > 0:
            keep = torch.rand((x.shape[0], 1), device=x.device) >= float(self.source_dropout)
            source_gate = keep.to(dtype=x.dtype)
        if self.source_embedding is not None:
            emb = self.source_embedding(source_id.long())
            h = h + (emb if source_gate is None else emb * source_gate)
        if len(self.adapters):
            adapted = torch.empty_like(h)
            for source_idx, adapter in enumerate(self.adapters):
                mask = source_id.long().eq(int(source_idx))
                if torch.any(mask):
                    residual = adapter(h[mask])
                    if source_gate is not None:
                        residual = residual * source_gate[mask]
                    adapted[mask] = h[mask] + residual
            h = adapted
        return self.encoder(h)


class ReplicateCGPAlign(nn.Module):
    def __init__(
        self,
        profile_dim: int,
        protein_dim: int,
        embed_dim: int,
        gnn_hidden_dim: int,
        gnn_layers: int,
        gene_hidden_dims: Sequence[int],
        profile_hidden_dims: Sequence[int],
        modality_context_dim: int,
        dropout: float,
        profile_source_adapters: bool,
        profile_source_embedding: bool,
        profile_source_dropout: float,
        num_profile_sources: int,
        profile_input_layernorm: bool,
        profile_feature_dropout: float,
        profile_mlp_norm: bool,
        disable_gene_modality_context: bool = False,
        split_gene_modality_branches: bool = False,
        split_gene_shared_trunk: bool = False,
    ):
        super().__init__()
        self.split_gene_modality_branches = bool(split_gene_modality_branches)
        self.split_gene_shared_trunk = bool(split_gene_shared_trunk) and self.split_gene_modality_branches
        self.compound_encoder = GGNNCompoundEncoder(
            atom_dim=atom_feature_dim(),
            bond_dim=bond_feature_dim(),
            embed_dim=int(embed_dim),
            hidden_dim=int(gnn_hidden_dim),
            num_layers=int(gnn_layers),
            projection_hidden_dims=(512,),
            dropout=float(dropout),
        )
        self.modality_context_dim = 0 if bool(disable_gene_modality_context) or self.split_gene_modality_branches else int(modality_context_dim)
        self.modality_embedding = (
            nn.Embedding(len(MODALITY_TO_ID), int(self.modality_context_dim))
            if int(self.modality_context_dim) > 0
            else None
        )
        gene_input_dim = int(protein_dim) + int(self.modality_context_dim)
        branch_input_dim = gene_input_dim
        branch_hidden_dims = tuple(gene_hidden_dims)
        self.gene_shared_trunk = None
        if self.split_gene_shared_trunk:
            if not gene_hidden_dims:
                raise ValueError("split_gene_shared_trunk requires at least one gene hidden dimension")
            first_hidden = int(gene_hidden_dims[0])
            self.gene_shared_trunk = nn.Sequential(
                nn.Linear(gene_input_dim, first_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(float(dropout)),
            )
            branch_input_dim = first_hidden
            branch_hidden_dims = tuple(gene_hidden_dims[1:])
        self.gene_encoder = MoCoPMLP(
            branch_input_dim,
            branch_hidden_dims,
            int(embed_dim),
            float(dropout),
        )
        self.crispr_gene_encoder = (
            MoCoPMLP(
                branch_input_dim,
                branch_hidden_dims,
                int(embed_dim),
                float(dropout),
            )
            if self.split_gene_modality_branches
            else None
        )
        self.profile_encoder = SharedProfileEncoder(
            int(profile_dim),
            profile_hidden_dims,
            int(embed_dim),
            float(dropout),
            bool(profile_source_adapters),
            bool(profile_source_embedding),
            float(profile_source_dropout),
            int(num_profile_sources),
            bool(profile_input_layernorm),
            float(profile_feature_dropout),
            bool(profile_mlp_norm),
        )

    def encode_compound_graphs(self, graphs: Sequence[Any], device: torch.device) -> torch.Tensor:
        atom_x, edge_index, edge_features, batch_index = batch_graphs(graphs, device)
        return self.compound_encoder(atom_x, edge_index, edge_features, batch_index)

    def encode_gene(self, protein_x: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
        if self.split_gene_modality_branches:
            branch_x = self.gene_shared_trunk(protein_x) if self.gene_shared_trunk is not None else protein_x
            out = torch.empty(
                (int(branch_x.shape[0]), int(self.gene_encoder.net[-1].out_features)),
                device=protein_x.device,
                dtype=protein_x.dtype,
            )
            orf_mask = modality.long().eq(int(MODALITY_TO_ID["orf"]))
            crispr_mask = modality.long().eq(int(MODALITY_TO_ID["crispr"]))
            if bool(orf_mask.any()):
                out[orf_mask] = self.gene_encoder(branch_x[orf_mask])
            if bool(crispr_mask.any()):
                if self.crispr_gene_encoder is None:
                    raise RuntimeError("split_gene_modality_branches requires crispr_gene_encoder")
                out[crispr_mask] = self.crispr_gene_encoder(branch_x[crispr_mask])
            other_mask = ~(orf_mask | crispr_mask)
            if bool(other_mask.any()):
                out[other_mask] = self.gene_encoder(branch_x[other_mask])
            return out
        if self.modality_embedding is None:
            x = protein_x
        else:
            x = torch.cat([protein_x, self.modality_embedding(modality.long())], dim=-1)
        return self.gene_encoder(x)

    def encode_profile(self, profile_x: torch.Tensor, source_id: torch.Tensor) -> torch.Tensor:
        return self.profile_encoder(profile_x, source_id.long())


def load_compound_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Dict[str, Any]]:
    paths = [
        data_dir / "compound_mocop_entities.parquet",
        data_dir / "compound_mocop_replicates.parquet",
        data_dir / "compound_mocop_replicate_features.npy",
        data_dir / "splits_compound_mocop.json",
    ]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit("Missing compound files:\n" + "\n".join(missing))
    return (
        pd.read_parquet(paths[0]).reset_index(drop=True),
        pd.read_parquet(paths[1]).reset_index(drop=True),
        np.load(paths[2]).astype(np.float32),
        json.loads(paths[3].read_text(encoding="utf-8")),
    )


def load_gene_data(data_dir: Path, protein_embedding_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    paths = [
        data_dir / "gene_mocop_entities.parquet",
        data_dir / "gene_mocop_replicates.parquet",
        data_dir / "gene_mocop_replicate_features.npy",
        data_dir / "splits_gene_mocop.json",
        protein_embedding_dir / "protein_sequence_embeddings.npy",
    ]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit("Missing gene files:\n" + "\n".join(missing))
    return (
        pd.read_parquet(paths[0]).reset_index(drop=True),
        pd.read_parquet(paths[1]).reset_index(drop=True),
        np.load(paths[2]).astype(np.float32),
        np.load(paths[4]).astype(np.float32),
        json.loads(paths[3].read_text(encoding="utf-8")),
    )


def build_entity_replicate_index(replicates: pd.DataFrame, num_entities: int) -> List[np.ndarray]:
    out: List[List[int]] = [[] for _ in range(num_entities)]
    for row in replicates.itertuples(index=False):
        out[int(row.entity_index)].append(int(row.feature_index))
    return [np.asarray(x, dtype=np.int64) for x in out]


def compound_entity_codes(entities: pd.DataFrame) -> np.ndarray:
    key_col = "entity_id" if "entity_id" in entities.columns else "compound_id"
    codes, _ = pd.factorize(entities[key_col].astype(str), sort=True)
    return codes.astype(np.int64)


def gene_entity_codes(entities: pd.DataFrame) -> np.ndarray:
    key_col = "entity_id" if "entity_id" in entities.columns else "gene_symbol"
    codes, _ = pd.factorize(entities[key_col].astype(str), sort=True)
    return codes.astype(np.int64)


def gene_modality_ids(entities: pd.DataFrame) -> np.ndarray:
    mapped = entities["perturbation_modality"].astype(str).str.lower().map(MODALITY_TO_ID)
    if mapped.isna().any():
        bad = sorted(entities.loc[mapped.isna(), "perturbation_modality"].astype(str).unique().tolist())
        raise ValueError(f"Unsupported gene modalities: {bad}")
    return mapped.to_numpy(dtype=np.int64)


def eligible_compounds(entities: pd.DataFrame, rows: Sequence[int], entity_to_reps: Sequence[np.ndarray]) -> np.ndarray:
    idx = np.asarray(rows, dtype=np.int64)
    if idx.size == 0:
        return idx
    has_reps = np.asarray([entity_to_reps[int(i)].size > 0 for i in idx], dtype=bool)
    has_structure = entities.iloc[idx]["has_structure"].astype(bool).to_numpy() if "has_structure" in entities.columns else np.ones(len(idx), dtype=bool)
    smiles_ok = entities.iloc[idx]["canonical_smiles"].fillna("").astype(str).str.len().to_numpy() > 0
    return idx[has_reps & has_structure & smiles_ok]


def eligible_genes(entities: pd.DataFrame, rows: Sequence[int], entity_to_reps: Sequence[np.ndarray], protein_embeddings: np.ndarray) -> np.ndarray:
    idx = np.asarray(rows, dtype=np.int64)
    if idx.size == 0:
        return idx
    source_idx = entities.iloc[idx]["source_gene_index"].fillna(-1).astype(np.int64).to_numpy()
    has_seq = entities.iloc[idx]["has_protein_sequence"].astype(bool).to_numpy()
    has_reps = np.asarray([entity_to_reps[int(i)].size > 0 for i in idx], dtype=bool)
    keep = has_seq & has_reps & (source_idx >= 0) & (source_idx < int(protein_embeddings.shape[0]))
    return idx[keep]


def fit_profile_normalizer(features: np.ndarray, entity_to_reps: Sequence[np.ndarray], train_entities: np.ndarray, mode: str) -> Dict[str, Any]:
    if mode == "none":
        return {"mode": "none", "fit_split": "none"}
    train_reps = np.concatenate([entity_to_reps[int(i)] for i in train_entities if entity_to_reps[int(i)].size])
    if train_reps.size == 0:
        raise RuntimeError("No train replicate profiles available for normalization.")
    x = features[train_reps]
    mean = np.nanmean(x, axis=0).astype(np.float32)
    std = np.nanstd(x, axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return {"mode": "train_zscore", "fit_split": "train", "mean": mean, "std": std}


def transform_rows(features: np.ndarray, rows: np.ndarray, norm: Dict[str, Any]) -> np.ndarray:
    x = features[np.asarray(rows, dtype=np.int64)].astype(np.float32).copy()
    if norm.get("mode") == "train_zscore":
        x = (x - norm["mean"]) / norm["std"]
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def l2_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(denom, eps)).astype(np.float32)


def entity_mean_profiles(
    features: np.ndarray,
    entity_to_reps: Sequence[np.ndarray],
    entity_rows: np.ndarray,
    norm: Dict[str, Any],
) -> np.ndarray:
    entity_rows = np.asarray(entity_rows, dtype=np.int64)
    out = np.zeros((len(entity_rows), int(features.shape[1])), dtype=np.float32)
    for out_idx, entity_idx in enumerate(entity_rows.tolist()):
        reps = entity_to_reps[int(entity_idx)]
        if reps.size:
            out[out_idx] = transform_rows(features, reps, norm).mean(axis=0)
    return out


def build_cg_profile_teacher(data: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if float(args.cg_teacher_weight) <= 0:
        return {"enabled": False, "reason": "cg_teacher_weight <= 0"}
    c_rows = np.asarray(data["compound"]["train_entities"], dtype=np.int64)
    g_rows = np.asarray(data["gene"]["train_entities"], dtype=np.int64)
    c_profiles = entity_mean_profiles(
        data["compound"]["features"],
        data["compound"]["entity_to_reps"],
        c_rows,
        data["compound"]["norm"],
    )
    g_profiles = entity_mean_profiles(
        data["gene"]["features"],
        data["gene"]["entity_to_reps"],
        g_rows,
        data["gene"]["norm"],
    )
    if args.cg_teacher_profile_transform == "l2":
        c_profiles = l2_normalize_np(c_profiles)
        g_profiles = l2_normalize_np(g_profiles)
    c_map = np.full(len(data["compound"]["entities"]), -1, dtype=np.int64)
    g_map = np.full(len(data["gene"]["entities"]), -1, dtype=np.int64)
    c_map[c_rows] = np.arange(len(c_rows), dtype=np.int64)
    g_map[g_rows] = np.arange(len(g_rows), dtype=np.int64)
    return {
        "enabled": True,
        "mode": "inbatch_profile_similarity_distillation",
        "profile_transform": str(args.cg_teacher_profile_transform),
        "compound_train_entities": c_rows,
        "gene_train_entities": g_rows,
        "compound_entity_to_teacher_row": c_map,
        "gene_entity_to_teacher_row": g_map,
        "compound_profiles": c_profiles.astype(np.float32),
        "gene_profiles": g_profiles.astype(np.float32),
    }


def masked_soft_targets(scores: torch.Tensor, temperature: float, topk: int) -> torch.Tensor:
    logits = scores.float() / float(temperature)
    if int(topk) > 0 and int(topk) < int(logits.shape[1]):
        k = max(1, int(topk))
        top_idx = torch.topk(logits, k=k, dim=1).indices
        masked = torch.full_like(logits, -1e9)
        masked.scatter_(1, top_idx, logits.gather(1, top_idx))
        logits = masked
    return F.softmax(logits, dim=1)


def topk_candidate_mask(scores: torch.Tensor, topk: int, dim: int) -> torch.Tensor:
    if scores.numel() == 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    size = int(scores.shape[dim])
    if int(topk) <= 0 or int(topk) >= size:
        return torch.ones_like(scores, dtype=torch.bool)
    k = max(1, int(topk))
    idx = torch.topk(scores.float(), k=k, dim=dim).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(dim, idx, True)
    return mask


def masked_soft_targets_with_valid(
    scores: torch.Tensor,
    temperature: float,
    candidate_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scores.numel() == 0:
        return scores.new_zeros(scores.shape), torch.zeros((scores.shape[0],), dtype=torch.bool, device=scores.device)
    row_valid = candidate_mask.any(dim=1)
    logits = scores.float() / float(temperature)
    logits = logits.masked_fill(~candidate_mask, -1e9)
    targets = torch.zeros_like(logits)
    if bool(row_valid.any()):
        targets[row_valid] = F.softmax(logits[row_valid], dim=1)
    return targets, row_valid


def cg_teacher_candidate_mask(scores: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    mask = topk_candidate_mask(scores, int(args.cg_teacher_batch_topk), dim=1)
    if bool(getattr(args, "cg_teacher_mutual_topk", False)):
        mask = mask & topk_candidate_mask(scores, int(args.cg_teacher_batch_topk), dim=0)
    min_similarity = float(getattr(args, "cg_teacher_min_similarity", -2.0))
    if min_similarity >= -1.0:
        mask = mask & (scores.float() >= min_similarity)
    return mask


def cg_profile_teacher_loss_one(
    zc: torch.Tensor,
    zg: torch.Tensor,
    c_prof: torch.Tensor,
    g_prof: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if zc.shape[0] == 0 or zg.shape[0] == 0:
        return zc.new_tensor(0.0)
    with torch.no_grad():
        teacher_scores = c_prof.float() @ g_prof.float().T
        candidate_mask = cg_teacher_candidate_mask(teacher_scores, args)
        target_c2g, valid_c = masked_soft_targets_with_valid(
            teacher_scores,
            float(args.cg_teacher_temperature),
            candidate_mask,
        )
        target_g2c, valid_g = masked_soft_targets_with_valid(
            teacher_scores.T,
            float(args.cg_teacher_temperature),
            candidate_mask.T,
        )
    student = zc.float() @ zg.float().T / float(args.cg_student_temperature)
    losses: List[torch.Tensor] = []
    if bool(valid_c.any()):
        losses.append(-(target_c2g[valid_c] * F.log_softmax(student[valid_c], dim=1)).sum(dim=1).mean())
    if bool(valid_g.any()):
        losses.append(-(target_g2c[valid_g] * F.log_softmax(student.T[valid_g], dim=1)).sum(dim=1).mean())
    return torch.stack(losses).mean() if losses else zc.new_tensor(0.0)


def cg_profile_teacher_loss(
    z_c: torch.Tensor,
    z_g: torch.Tensor,
    c_batch: np.ndarray,
    g_batch: np.ndarray,
    gene_modality: torch.Tensor,
    teacher: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    if not teacher.get("enabled"):
        return z_c.new_tensor(0.0)
    c_map = teacher["compound_entity_to_teacher_row"][np.asarray(c_batch, dtype=np.int64)]
    g_map = teacher["gene_entity_to_teacher_row"][np.asarray(g_batch, dtype=np.int64)]
    valid_c = c_map >= 0
    valid_g = g_map >= 0
    if not bool(valid_c.any()) or not bool(valid_g.any()):
        return z_c.new_tensor(0.0)
    zc = z_c[torch.from_numpy(valid_c).to(device=device, dtype=torch.bool)]
    zg = z_g[torch.from_numpy(valid_g).to(device=device, dtype=torch.bool)]
    mod = gene_modality[torch.from_numpy(valid_g).to(device=device, dtype=torch.bool)]
    c_prof = tensor_from_numpy(teacher["compound_profiles"][c_map[valid_c]], device)
    g_prof = tensor_from_numpy(teacher["gene_profiles"][g_map[valid_g]], device)
    if str(getattr(args, "cg_teacher_gene_modality_scope", "mixed")) == "separate":
        losses: List[torch.Tensor] = []
        for mod_id in sorted(int(x) for x in torch.unique(mod).detach().cpu().tolist()):
            keep = mod == int(mod_id)
            if bool(keep.any()):
                losses.append(cg_profile_teacher_loss_one(zc, zg[keep], c_prof, g_prof[keep], args))
        return torch.stack(losses).mean() if losses else z_c.new_tensor(0.0)
    return cg_profile_teacher_loss_one(zc, zg, c_prof, g_prof, args)


def parse_profile_structure_components(text: str) -> List[str]:
    allowed = {"cc", "gg", "cg"}
    components: List[str] = []
    for item in str(text or "").split(","):
        value = item.strip().lower()
        if not value:
            continue
        if value not in allowed:
            raise ValueError(f"Unknown profile-structure component {value!r}; expected one of {sorted(allowed)}")
        if value not in components:
            components.append(value)
    return components


def build_profile_structure_teacher(data: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if float(args.profile_structure_weight) <= 0:
        return {"enabled": False, "reason": "profile_structure_weight <= 0"}
    components = parse_profile_structure_components(args.profile_structure_components)
    if not components:
        return {"enabled": False, "reason": "profile_structure_components is empty"}
    c_rows = np.asarray(data["compound"]["train_entities"], dtype=np.int64)
    g_rows = np.asarray(data["gene"]["train_entities"], dtype=np.int64)
    c_profiles = entity_mean_profiles(
        data["compound"]["features"],
        data["compound"]["entity_to_reps"],
        c_rows,
        data["compound"]["norm"],
    )
    g_profiles = entity_mean_profiles(
        data["gene"]["features"],
        data["gene"]["entity_to_reps"],
        g_rows,
        data["gene"]["norm"],
    )
    if args.profile_structure_profile_transform == "l2":
        c_profiles = l2_normalize_np(c_profiles)
        g_profiles = l2_normalize_np(g_profiles)
    c_map = np.full(len(data["compound"]["entities"]), -1, dtype=np.int64)
    g_map = np.full(len(data["gene"]["entities"]), -1, dtype=np.int64)
    c_map[c_rows] = np.arange(len(c_rows), dtype=np.int64)
    g_map[g_rows] = np.arange(len(g_rows), dtype=np.int64)
    return {
        "enabled": True,
        "mode": "inbatch_profile_structure_distillation",
        "components": components,
        "profile_transform": str(args.profile_structure_profile_transform),
        "compound_train_entities": c_rows,
        "gene_train_entities": g_rows,
        "compound_entity_to_teacher_row": c_map,
        "gene_entity_to_teacher_row": g_map,
        "compound_profiles": c_profiles.astype(np.float32),
        "gene_profiles": g_profiles.astype(np.float32),
    }


def profile_pair_structure_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    prof_a: torch.Tensor,
    prof_b: torch.Tensor,
    args: argparse.Namespace,
    exclude_diagonal: bool,
) -> torch.Tensor:
    if z_a.shape[0] == 0 or z_b.shape[0] == 0:
        return z_a.new_tensor(0.0)
    if bool(exclude_diagonal) and int(z_a.shape[0]) <= 1:
        return z_a.new_tensor(0.0)
    with torch.no_grad():
        teacher_scores = prof_a.float() @ prof_b.float().T
        if bool(exclude_diagonal):
            diag_mask = torch.eye(int(teacher_scores.shape[0]), dtype=torch.bool, device=teacher_scores.device)
            teacher_scores = teacher_scores.masked_fill(diag_mask, -1e9)
        target_a2b = masked_soft_targets(
            teacher_scores,
            float(args.profile_structure_temperature),
            int(args.profile_structure_batch_topk),
        )
        target_b2a = masked_soft_targets(
            teacher_scores.T,
            float(args.profile_structure_temperature),
            int(args.profile_structure_batch_topk),
        )
    student = z_a.float() @ z_b.float().T / float(args.profile_structure_student_temperature)
    if bool(exclude_diagonal):
        diag_mask = torch.eye(int(student.shape[0]), dtype=torch.bool, device=student.device)
        student = student.masked_fill(diag_mask, -1e9)
    a2b = -(target_a2b * F.log_softmax(student, dim=1)).sum(dim=1).mean()
    b2a = -(target_b2a * F.log_softmax(student.T, dim=1)).sum(dim=1).mean()
    return 0.5 * (a2b + b2a)


def profile_structure_loss(
    z_c: torch.Tensor,
    z_g: torch.Tensor,
    c_batch: np.ndarray,
    g_batch: np.ndarray,
    teacher: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if not teacher.get("enabled"):
        zero = z_c.new_tensor(0.0)
        return zero, {"cc": 0.0, "gg": 0.0, "cg": 0.0}
    components = set(teacher.get("components", []))
    c_map = teacher["compound_entity_to_teacher_row"][np.asarray(c_batch, dtype=np.int64)]
    g_map = teacher["gene_entity_to_teacher_row"][np.asarray(g_batch, dtype=np.int64)]
    valid_c = c_map >= 0
    valid_g = g_map >= 0
    losses: List[torch.Tensor] = []
    parts = {"cc": 0.0, "gg": 0.0, "cg": 0.0}
    zc = z_c[torch.from_numpy(valid_c).to(device=device, dtype=torch.bool)] if bool(valid_c.any()) else z_c[:0]
    zg = z_g[torch.from_numpy(valid_g).to(device=device, dtype=torch.bool)] if bool(valid_g.any()) else z_g[:0]
    c_prof = tensor_from_numpy(teacher["compound_profiles"][c_map[valid_c]], device) if bool(valid_c.any()) else z_c.new_zeros((0, 0))
    g_prof = tensor_from_numpy(teacher["gene_profiles"][g_map[valid_g]], device) if bool(valid_g.any()) else z_g.new_zeros((0, 0))

    if "cc" in components and int(zc.shape[0]) > 1:
        loss_cc = profile_pair_structure_loss(zc, zc, c_prof, c_prof, args, exclude_diagonal=True)
        losses.append(loss_cc)
        parts["cc"] = float(loss_cc.detach().cpu().item())
    if "gg" in components and int(zg.shape[0]) > 1:
        loss_gg = profile_pair_structure_loss(zg, zg, g_prof, g_prof, args, exclude_diagonal=True)
        losses.append(loss_gg)
        parts["gg"] = float(loss_gg.detach().cpu().item())
    if "cg" in components and int(zc.shape[0]) > 0 and int(zg.shape[0]) > 0:
        loss_cg = profile_pair_structure_loss(zc, zg, c_prof, g_prof, args, exclude_diagonal=False)
        losses.append(loss_cg)
        parts["cg"] = float(loss_cg.detach().cpu().item())
    total = torch.stack(losses).mean() if losses else z_c.new_tensor(0.0)
    return total, parts


def mean_profile_latents_by_entity(
    z_reps: torch.Tensor,
    owner_entities: np.ndarray,
    batch_entities: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    owners = np.asarray(owner_entities, dtype=np.int64)
    rows: List[torch.Tensor] = []
    for entity_idx in np.asarray(batch_entities, dtype=np.int64).tolist():
        mask_np = owners == int(entity_idx)
        if not bool(mask_np.any()):
            raise RuntimeError(f"Entity {int(entity_idx)} has no profile latent in the sampled batch.")
        mask = torch.from_numpy(mask_np).to(device=device, dtype=torch.bool)
        rows.append(z_reps[mask].mean(dim=0))
    if not rows:
        return z_reps[:0]
    return F.normalize(torch.stack(rows, dim=0), dim=-1)


def sample_replicates_multi(
    entity_rows: np.ndarray,
    entity_to_reps: Sequence[np.ndarray],
    rng: np.random.Generator,
    replicates_per_entity: int,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_rows: List[int] = []
    owner_rows: List[int] = []
    r = max(1, int(replicates_per_entity))
    for entity_idx in np.asarray(entity_rows, dtype=np.int64).tolist():
        reps = entity_to_reps[int(entity_idx)]
        if reps.size == 0:
            raise RuntimeError(f"Entity {entity_idx} has no replicate profiles.")
        replace = bool(reps.size < r)
        sampled = rng.choice(reps, size=r, replace=replace)
        feature_rows.extend([int(x) for x in sampled.tolist()])
        owner_rows.extend([int(entity_idx)] * r)
    return np.asarray(feature_rows, dtype=np.int64), np.asarray(owner_rows, dtype=np.int64)


def build_replicate_gallery(
    entity_rows: np.ndarray,
    entity_to_reps: Sequence[np.ndarray],
    entity_codes: np.ndarray,
    cap: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    feature_rows: List[int] = []
    owners: List[int] = []
    codes: List[int] = []
    for entity_idx in np.asarray(entity_rows, dtype=np.int64).tolist():
        reps = entity_to_reps[int(entity_idx)]
        if reps.size == 0:
            continue
        if int(cap) > 0 and reps.size > int(cap):
            reps = np.sort(rng.choice(reps, size=int(cap), replace=False))
        feature_rows.extend([int(x) for x in reps.tolist()])
        owners.extend([int(entity_idx)] * int(reps.size))
        codes.extend([int(entity_codes[int(entity_idx)])] * int(reps.size))
    return {
        "feature_rows": np.asarray(feature_rows, dtype=np.int64),
        "owner_entities": np.asarray(owners, dtype=np.int64),
        "codes": np.asarray(codes, dtype=np.int64),
    }


def gene_protein_tensor(entities: pd.DataFrame, protein_embeddings: np.ndarray, entity_rows: np.ndarray, device: torch.device) -> torch.Tensor:
    idx = entities.iloc[np.asarray(entity_rows, dtype=np.int64)]["source_gene_index"].to_numpy(dtype=np.int64)
    return tensor_from_numpy(protein_embeddings[idx], device)


def gene_modality_tensor(modality_ids: np.ndarray, entity_rows: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(modality_ids[np.asarray(entity_rows, dtype=np.int64)].astype(np.int64)).to(device=device, dtype=torch.long)


def gene_profile_source_ids(args: argparse.Namespace, modality_ids: np.ndarray, owner_entities: np.ndarray, device: torch.device) -> torch.Tensor:
    if args.profile_source_mode == "compound_gene":
        values = np.ones(len(owner_entities), dtype=np.int64)
    else:
        values = 1 + modality_ids[np.asarray(owner_entities, dtype=np.int64)].astype(np.int64)
    return torch.from_numpy(values).to(device=device, dtype=torch.long)


def compound_profile_source_ids(n: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(int(n), dtype=torch.long, device=device)


def multipositive_contrastive_loss(
    query: torch.Tensor,
    gallery: torch.Tensor,
    query_code: torch.Tensor,
    gallery_code: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if query.shape[0] == 0 or gallery.shape[0] == 0:
        return query.new_tensor(0.0)
    logits = query @ gallery.T / float(temperature)
    positives = query_code.view(-1, 1).eq(gallery_code.view(1, -1))
    losses: List[torch.Tensor] = []
    row_has = positives.any(dim=1)
    if bool(row_has.any()):
        target = positives[row_has].float()
        target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0)
        losses.append(-(target * F.log_softmax(logits[row_has], dim=1)).sum(dim=1).mean())
    col_has = positives.any(dim=0)
    if bool(col_has.any()):
        target_t = positives[:, col_has].T.float()
        target_t = target_t / target_t.sum(dim=1, keepdim=True).clamp_min(1.0)
        logits_t = logits[:, col_has].T
        losses.append(-(target_t * F.log_softmax(logits_t, dim=1)).sum(dim=1).mean())
    return torch.stack(losses).mean() if losses else query.new_tensor(0.0)


def modality_balanced_contrastive_loss(
    query: torch.Tensor,
    gallery: torch.Tensor,
    query_code: torch.Tensor,
    gallery_code: torch.Tensor,
    query_modality: torch.Tensor,
    gallery_modality: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    for modality_id in sorted(MODALITY_TO_ID.values()):
        q_mask = query_modality.long().eq(int(modality_id))
        g_mask = gallery_modality.long().eq(int(modality_id))
        if bool(q_mask.any()) and bool(g_mask.any()):
            losses.append(
                multipositive_contrastive_loss(
                    query[q_mask],
                    gallery[g_mask],
                    query_code[q_mask],
                    gallery_code[g_mask],
                    temperature,
                )
            )
    if losses:
        return torch.stack(losses).mean()
    return multipositive_contrastive_loss(query, gallery, query_code, gallery_code, temperature)


def modality_balanced_contrastive_loss_parts(
    query: torch.Tensor,
    gallery: torch.Tensor,
    query_code: torch.Tensor,
    gallery_code: torch.Tensor,
    query_modality: torch.Tensor,
    gallery_modality: torch.Tensor,
    temperature: float,
    reduction: str = "sum",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, int]]:
    if reduction not in {"sum", "mean"}:
        raise ValueError(f"Unknown gene branch loss reduction: {reduction}")
    losses: List[torch.Tensor] = []
    parts: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    id_to_name = {int(v): str(k) for k, v in MODALITY_TO_ID.items()}
    for modality_id in sorted(MODALITY_TO_ID.values()):
        q_mask = query_modality.long().eq(int(modality_id))
        g_mask = gallery_modality.long().eq(int(modality_id))
        name = id_to_name[int(modality_id)]
        counts[f"{name}_query"] = int(q_mask.sum().detach().cpu().item())
        counts[f"{name}_profile"] = int(g_mask.sum().detach().cpu().item())
        if bool(q_mask.any()) and bool(g_mask.any()):
            cur = multipositive_contrastive_loss(
                query[q_mask],
                gallery[g_mask],
                query_code[q_mask],
                gallery_code[g_mask],
                temperature,
            )
            parts[name] = cur
            losses.append(cur)
    if losses:
        return (torch.stack(losses).sum() if reduction == "sum" else torch.stack(losses).mean()), parts, counts
    return multipositive_contrastive_loss(query, gallery, query_code, gallery_code, temperature), parts, counts


def inverse_modality_weights(modality: torch.Tensor) -> torch.Tensor:
    modality = modality.long()
    weights = torch.ones_like(modality, dtype=torch.float32)
    present = []
    for modality_id in sorted(MODALITY_TO_ID.values()):
        mask = modality.eq(int(modality_id))
        if bool(mask.any()):
            present.append(mask)
    if not present:
        return weights
    n = float(modality.numel())
    groups = float(len(present))
    for mask in present:
        weights[mask] = n / (groups * float(mask.sum().item()))
    return weights / weights.mean().clamp_min(1e-8)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def modality_soft_balanced_contrastive_loss(
    query: torch.Tensor,
    gallery: torch.Tensor,
    query_code: torch.Tensor,
    gallery_code: torch.Tensor,
    query_modality: torch.Tensor,
    gallery_modality: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if query.shape[0] == 0 or gallery.shape[0] == 0:
        return query.new_tensor(0.0)
    logits = query @ gallery.T / float(temperature)
    positives = query_code.view(-1, 1).eq(gallery_code.view(1, -1))
    q_weights = inverse_modality_weights(query_modality).to(device=query.device, dtype=query.dtype)
    g_weights = inverse_modality_weights(gallery_modality).to(device=gallery.device, dtype=gallery.dtype)
    losses: List[torch.Tensor] = []
    row_has = positives.any(dim=1)
    if bool(row_has.any()):
        target = positives[row_has].float()
        target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0)
        row_loss = -(target * F.log_softmax(logits[row_has], dim=1)).sum(dim=1)
        losses.append(weighted_mean(row_loss, q_weights[row_has]))
    col_has = positives.any(dim=0)
    if bool(col_has.any()):
        target_t = positives[:, col_has].T.float()
        target_t = target_t / target_t.sum(dim=1, keepdim=True).clamp_min(1.0)
        logits_t = logits[:, col_has].T
        col_loss = -(target_t * F.log_softmax(logits_t, dim=1)).sum(dim=1)
        losses.append(weighted_mean(col_loss, g_weights[col_has]))
    return torch.stack(losses).mean() if losses else query.new_tensor(0.0)


def cosine_anchor_loss(current: Sequence[torch.Tensor], reference: Sequence[torch.Tensor]) -> torch.Tensor:
    losses = []
    for cur, ref in zip(current, reference):
        if cur.shape == ref.shape and cur.numel():
            losses.append((1.0 - (cur.float() * ref.detach().float()).sum(dim=-1)).mean())
    return torch.stack(losses).mean() if losses else current[0].new_tensor(0.0)


def harmonic_mean(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and float(v) > 0]
    if len(vals) != len(values) or not vals:
        return None
    return float(len(vals) / sum(1.0 / v for v in vals))


def full_gallery_metrics_general(
    query: np.ndarray,
    gallery: np.ndarray,
    query_code: np.ndarray,
    gallery_code: np.ndarray,
    chunk_size: int = 512,
) -> Dict[str, Any]:
    if len(query) == 0 or len(gallery) == 0:
        return {"num_queries": int(len(query)), "num_gallery": int(len(gallery))}
    ranks: List[int] = []
    hits1 = hits5 = hits10 = 0
    gallery_code = np.asarray(gallery_code, dtype=np.int64)
    for start in range(0, len(query), max(1, int(chunk_size))):
        q = query[start : start + max(1, int(chunk_size))]
        scores = q @ gallery.T
        for local_i in range(scores.shape[0]):
            i = start + local_i
            pos = gallery_code == int(query_code[i])
            if not bool(pos.any()):
                continue
            best_pos = float(scores[local_i, pos].max())
            rank = int(1 + np.sum(scores[local_i] > best_pos))
            ranks.append(rank)
            hits1 += int(rank <= 1)
            hits5 += int(rank <= 5)
            hits10 += int(rank <= 10)
    if not ranks:
        return {"num_queries": int(len(query)), "num_gallery": int(len(gallery)), "num_evaluated": 0}
    arr = np.asarray(ranks, dtype=np.float64)
    return {
        "num_queries": int(len(query)),
        "num_gallery": int(len(gallery)),
        "num_evaluated": int(len(arr)),
        "Recall@1": float(hits1 / len(arr)),
        "Recall@5": float(hits5 / len(arr)),
        "Recall@10": float(hits10 / len(arr)),
        "MRR": float(np.mean(1.0 / arr)),
        "median_rank": float(np.median(arr)),
        "mean_rank": float(np.mean(arr)),
    }


def sampled_one_direction_general(
    query: np.ndarray,
    gallery: np.ndarray,
    query_code: np.ndarray,
    gallery_code: np.ndarray,
    ratio: int,
    repeats: int,
    seed: int,
) -> Dict[str, Any]:
    if len(query) == 0 or len(gallery) == 0:
        return {"Top1_accuracy_mean": None, "Top10_accuracy_mean": None, "num_queries": int(len(query))}
    rng = np.random.default_rng(int(seed))
    all_idx = np.arange(len(gallery), dtype=np.int64)
    top1_vals: List[float] = []
    top10_vals: List[float] = []
    pos_counts: List[int] = []
    gallery_code = np.asarray(gallery_code, dtype=np.int64)
    query_code = np.asarray(query_code, dtype=np.int64)
    for rep in range(int(repeats)):
        hits1 = hits10 = evaluated = 0
        for i in range(len(query)):
            pos_pool = all_idx[gallery_code == query_code[i]]
            neg_pool = all_idx[gallery_code != query_code[i]]
            if pos_pool.size == 0 or neg_pool.size == 0:
                continue
            n_neg = min(int(ratio), int(neg_pool.size))
            neg = rng.choice(neg_pool, size=n_neg, replace=False)
            cand = np.concatenate([pos_pool, neg])
            scores = gallery[cand] @ query[i]
            order = np.argsort(-scores)
            ranked = cand[order]
            pos_set = set(pos_pool.tolist())
            hits1 += int(int(ranked[0]) in pos_set)
            hits10 += int(any(int(x) in pos_set for x in ranked[: min(10, len(ranked))]))
            evaluated += 1
            if rep == 0:
                pos_counts.append(int(pos_pool.size))
        if evaluated:
            top1_vals.append(float(hits1 / evaluated))
            top10_vals.append(float(hits10 / evaluated))
    return {
        "Top1_accuracy_mean": float(np.mean(top1_vals)) if top1_vals else None,
        "Top1_accuracy_std": float(np.std(top1_vals)) if top1_vals else None,
        "Top10_accuracy_mean": float(np.mean(top10_vals)) if top10_vals else None,
        "Top10_accuracy_std": float(np.std(top10_vals)) if top10_vals else None,
        "num_queries": int(len(query)),
        "num_gallery": int(len(gallery)),
        "ratio": int(ratio),
        "repeats": int(repeats),
        "positive_count_mean": float(np.mean(pos_counts)) if pos_counts else None,
    }


def sampled_metrics_general(
    query: np.ndarray,
    gallery: np.ndarray,
    query_code: np.ndarray,
    gallery_code: np.ndarray,
    forward_name: str,
    reverse_name: str,
    ratios: Sequence[int],
    repeats: int,
    seed: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {forward_name: {}, reverse_name: {}, "positive_label": "entity_id"}
    for ratio in ratios:
        key = f"1:{int(ratio)}"
        out[forward_name][key] = sampled_one_direction_general(query, gallery, query_code, gallery_code, ratio, repeats, seed)
        out[reverse_name][key] = sampled_one_direction_general(gallery, query, gallery_code, query_code, ratio, repeats, seed + 17)
    return out


def encode_compounds(
    model: ReplicateCGPAlign,
    graph_store: GraphStore,
    rows: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    chunks: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), max(1, int(batch_size))):
            batch = np.asarray(rows[start : start + max(1, int(batch_size))], dtype=np.int64)
            z = model.encode_compound_graphs(graph_store.get_many(batch.tolist()), device)
            chunks.append(z.detach().cpu().numpy().astype(np.float32))
    return np.vstack(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)


def encode_gene_entities(
    model: ReplicateCGPAlign,
    entities: pd.DataFrame,
    protein_embeddings: np.ndarray,
    modality_ids: np.ndarray,
    rows: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    chunks: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), max(1, int(batch_size))):
            batch = np.asarray(rows[start : start + max(1, int(batch_size))], dtype=np.int64)
            x = gene_protein_tensor(entities, protein_embeddings, batch, device)
            mod = gene_modality_tensor(modality_ids, batch, device)
            z = model.encode_gene(x, mod)
            chunks.append(z.detach().cpu().numpy().astype(np.float32))
    return np.vstack(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)


def encode_profiles(
    model: ReplicateCGPAlign,
    features: np.ndarray,
    feature_rows: np.ndarray,
    source_ids_np: np.ndarray,
    norm: Dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    chunks: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(feature_rows), max(1, int(batch_size))):
            rows = np.asarray(feature_rows[start : start + max(1, int(batch_size))], dtype=np.int64)
            src = torch.from_numpy(source_ids_np[start : start + len(rows)].astype(np.int64)).to(device=device, dtype=torch.long)
            x = tensor_from_numpy(transform_rows(features, rows, norm), device)
            z = model.encode_profile(x, src)
            chunks.append(z.detach().cpu().numpy().astype(np.float32))
    return np.vstack(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)


def load_mapped_checkpoint(model: nn.Module, path: Optional[Path], mappings: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    if path is None:
        return {"enabled": False}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source_state = payload.get("model_state_dict", payload)
    target_state = model.state_dict()
    copied: Dict[str, torch.Tensor] = {}
    partial_copied: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for src_name, tensor in source_state.items():
        mapped_name = None
        for src_prefix, dst_prefix in mappings:
            if src_name.startswith(src_prefix):
                mapped_name = dst_prefix + src_name[len(src_prefix) :]
                break
        if mapped_name is None:
            continue
        if mapped_name in target_state and tuple(target_state[mapped_name].shape) == tuple(tensor.shape):
            copied[mapped_name] = tensor
        elif (
            mapped_name in target_state
            and hasattr(tensor, "ndim")
            and int(tensor.ndim) == 2
            and int(target_state[mapped_name].ndim) == 2
            and int(tensor.shape[0]) == int(target_state[mapped_name].shape[0])
            and int(tensor.shape[1]) >= int(target_state[mapped_name].shape[1])
        ):
            copied[mapped_name] = tensor[:, : target_state[mapped_name].shape[1]]
            partial_copied.append(
                {
                    "source": src_name,
                    "target": mapped_name,
                    "source_shape": list(tensor.shape),
                    "target_shape": list(target_state[mapped_name].shape),
                    "mode": "truncate_input_columns",
                }
            )
        else:
            skipped.append({"source": src_name, "target": mapped_name, "shape": list(tensor.shape) if hasattr(tensor, "shape") else None})
    expected = {k for k in target_state if any(k.startswith(dst) for _, dst in mappings)}
    if partial_copied or skipped or set(copied) != expected:
        raise RuntimeError(f"Initializer must exactly cover mapped branches: partial={partial_copied}, skipped={skipped}, missing={sorted(expected-set(copied))}")
    missing, unexpected = model.load_state_dict(copied, strict=False)
    return {
        "enabled": True,
        "path": str(path),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_best_score": payload.get("best_score"),
        "copied": int(len(copied)),
        "partial_copied": partial_copied[:20],
        "skipped": skipped[:20],
        "missing_after_partial_load": len(missing),
        "unexpected_after_partial_load": len(unexpected),
    }


def load_split_shared_gene_checkpoints(model: nn.Module, orf_path: Optional[Path], crispr_path: Optional[Path]) -> Dict[str, Any]:
    if orf_path is None or crispr_path is None:
        return {"enabled": False, "reason": "missing ORF or CRISPR checkpoint"}
    if getattr(model, "gene_shared_trunk", None) is None:
        return {"enabled": False, "reason": "model has no gene_shared_trunk"}
    orf_payload = torch.load(orf_path, map_location="cpu", weights_only=False)
    crispr_payload = torch.load(crispr_path, map_location="cpu", weights_only=False)
    orf_state = orf_payload.get("model_state_dict", orf_payload)
    crispr_state = crispr_payload.get("model_state_dict", crispr_payload)
    target = model.state_dict()
    copied: Dict[str, torch.Tensor] = {}
    copied_names: List[str] = []

    for suffix in ["weight", "bias"]:
        src_name = f"protein_encoder.net.0.{suffix}"
        dst_name = f"gene_shared_trunk.0.{suffix}"
        if src_name in orf_state and src_name in crispr_state and dst_name in target:
            tensor = 0.5 * (orf_state[src_name].float() + crispr_state[src_name].float())
            if tuple(tensor.shape) == tuple(target[dst_name].shape):
                copied[dst_name] = tensor.to(dtype=target[dst_name].dtype)
                copied_names.append(dst_name)

    linear_map = [(3, 0), (6, 3)]
    for src_idx, dst_idx in linear_map:
        for suffix in ["weight", "bias"]:
            src_name = f"protein_encoder.net.{src_idx}.{suffix}"
            orf_dst = f"gene_encoder.net.{dst_idx}.{suffix}"
            crispr_dst = f"crispr_gene_encoder.net.{dst_idx}.{suffix}"
            if src_name in orf_state and orf_dst in target and tuple(orf_state[src_name].shape) == tuple(target[orf_dst].shape):
                copied[orf_dst] = orf_state[src_name].to(dtype=target[orf_dst].dtype)
                copied_names.append(orf_dst)
            if src_name in crispr_state and crispr_dst in target and tuple(crispr_state[src_name].shape) == tuple(target[crispr_dst].shape):
                copied[crispr_dst] = crispr_state[src_name].to(dtype=target[crispr_dst].dtype)
                copied_names.append(crispr_dst)

    missing, unexpected = model.load_state_dict(copied, strict=False)
    return {
        "enabled": True,
        "orf_path": str(orf_path),
        "crispr_path": str(crispr_path),
        "orf_checkpoint_epoch": orf_payload.get("epoch"),
        "crispr_checkpoint_epoch": crispr_payload.get("epoch"),
        "copied": int(len(copied)),
        "copied_names": copied_names,
        "missing_after_partial_load": len(missing),
        "unexpected_after_partial_load": len(unexpected),
    }


def split_gene_distribution_alignment_loss(z: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
    orf = z[modality.long().eq(int(MODALITY_TO_ID["orf"]))]
    crispr = z[modality.long().eq(int(MODALITY_TO_ID["crispr"]))]
    if int(orf.shape[0]) < 2 or int(crispr.shape[0]) < 2:
        return z.new_tensor(0.0)
    mean_loss = (orf.mean(dim=0) - crispr.mean(dim=0)).pow(2).mean()
    var_loss = (orf.var(dim=0, unbiased=False) - crispr.var(dim=0, unbiased=False)).pow(2).mean()
    return mean_loss + var_loss


def set_branch_trainable(model: ReplicateCGPAlign, trainable: bool) -> None:
    for module in [model.compound_encoder, model.modality_embedding, model.gene_shared_trunk, model.gene_encoder, model.crispr_gene_encoder]:
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = bool(trainable)


def set_gene_modality_embedding_trainable(model: ReplicateCGPAlign, trainable: bool) -> None:
    if model.modality_embedding is None:
        return
    for param in model.modality_embedding.parameters():
        param.requires_grad = bool(trainable)


def set_profile_trainable(model: ReplicateCGPAlign, trainable: bool) -> None:
    for param in model.profile_encoder.parameters():
        param.requires_grad = bool(trainable)


def epoch_stage(args: argparse.Namespace, epoch: int, has_branch_init: bool) -> Tuple[str, bool, float]:
    if bool(args.disable_staged_training):
        return "direct_joint", True, 0.0
    if int(epoch) <= int(args.stage1_epochs):
        return "stage1_profile_calibration", not bool(has_branch_init), 0.0
    if int(epoch) <= int(args.stage1_epochs) + int(args.stage2_epochs):
        return "stage2_branch_stabilization", True, float(args.anchor_weight_stage2 if has_branch_init else 0.0)
    return "stage3_shared_space_joint", True, float(args.anchor_weight_stage3 if has_branch_init else 0.0)


def has_required_branch_initialization(args: argparse.Namespace) -> bool:
    has_compound = args.init_compound_checkpoint is not None
    if bool(getattr(args, "split_gene_modality_branches", False)):
        has_orf = (args.init_orf_gene_checkpoint is not None) or (args.init_gene_checkpoint is not None)
        has_crispr = (args.init_crispr_gene_checkpoint is not None) or (args.init_gene_checkpoint is not None)
        return bool(has_compound and has_orf and has_crispr)
    return bool(has_compound and args.init_gene_checkpoint is not None)


def cg_teacher_weight_for_epoch(args: argparse.Namespace, epoch: int) -> float:
    base = float(args.cg_teacher_weight)
    if base <= 0:
        return 0.0
    start = int(args.cg_teacher_start_epoch)
    if start <= 0:
        start = 1 if bool(args.disable_staged_training) else int(args.stage1_epochs) + int(args.stage2_epochs) + 1
    if int(epoch) < start:
        return 0.0
    warmup = max(1, int(args.cg_teacher_warmup_epochs))
    progress = min(1.0, float(int(epoch) - start + 1) / float(warmup))
    return float(base * progress)


def profile_structure_weight_for_epoch(args: argparse.Namespace, epoch: int) -> float:
    base = float(args.profile_structure_weight)
    if base <= 0:
        return 0.0
    start = int(args.profile_structure_start_epoch)
    if start <= 0:
        start = 1 if bool(args.disable_staged_training) else int(args.stage1_epochs) + 1
    if int(epoch) < start:
        return 0.0
    warmup = max(1, int(args.profile_structure_warmup_epochs))
    progress = min(1.0, float(int(epoch) - start + 1) / float(warmup))
    return float(base * progress)


def choose_steps(args: argparse.Namespace, n_compounds: int, n_genes: int) -> int:
    if int(args.steps_per_epoch) > 0:
        return int(args.steps_per_epoch)
    c_steps = math.ceil(n_compounds / max(1, int(args.compound_batch_size)))
    g_steps = math.ceil(n_genes / max(1, int(args.gene_batch_size)))
    return max(1, min(c_steps, g_steps))


def maybe_subsample(rows: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if int(max_rows) > 0 and len(rows) > int(max_rows):
        return np.sort(rng.choice(rows, size=int(max_rows), replace=False))
    return rows


def sample_gene_batch(
    train_entities: np.ndarray,
    modality_ids: np.ndarray,
    batch_size: int,
    balanced: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    train_entities = np.asarray(train_entities, dtype=np.int64)
    size = min(int(batch_size), len(train_entities))
    if size <= 0 or not bool(balanced):
        return rng.choice(train_entities, size=size, replace=False)
    selected: List[np.ndarray] = []
    remaining_mask = np.ones(len(train_entities), dtype=bool)
    base = size // len(MODALITY_TO_ID)
    remainder = size % len(MODALITY_TO_ID)
    for offset, modality_id in enumerate(sorted(MODALITY_TO_ID.values())):
        want = base + (1 if offset < remainder else 0)
        pool_pos = np.where(modality_ids[train_entities] == int(modality_id))[0]
        if len(pool_pos) == 0 or want <= 0:
            continue
        take = min(int(want), int(len(pool_pos)))
        chosen_pos = rng.choice(pool_pos, size=take, replace=False)
        remaining_mask[chosen_pos] = False
        selected.append(train_entities[chosen_pos])
    current = int(sum(len(x) for x in selected))
    if current < size:
        rest = train_entities[remaining_mask]
        if len(rest) > 0:
            take = min(size - current, len(rest))
            selected.append(rng.choice(rest, size=take, replace=False))
    if selected:
        out = np.concatenate(selected).astype(np.int64)
        rng.shuffle(out)
        return out
    return rng.choice(train_entities, size=size, replace=False)


def evaluate_compound(
    model: ReplicateCGPAlign,
    entities: pd.DataFrame,
    features: np.ndarray,
    rows: np.ndarray,
    entity_to_reps: Sequence[np.ndarray],
    codes: np.ndarray,
    norm: Dict[str, Any],
    graph_store: GraphStore,
    args: argparse.Namespace,
    device: torch.device,
    ratios: Sequence[int],
    repeats: int,
    seed: int,
) -> Dict[str, Any]:
    z_c = encode_compounds(model, graph_store, rows, device, args.eval_batch_size)
    query_codes = codes[np.asarray(rows, dtype=np.int64)]
    gallery = build_replicate_gallery(rows, entity_to_reps, codes, args.eval_max_replicates_per_entity, seed)
    src = np.zeros(len(gallery["feature_rows"]), dtype=np.int64)
    z_p = encode_profiles(model, features, gallery["feature_rows"], src, norm, device, args.eval_batch_size)
    sampled = sampled_metrics_general(
        z_c,
        z_p,
        query_codes,
        gallery["codes"],
        "compound_to_profile",
        "profile_to_compound",
        ratios,
        repeats,
        seed,
    )
    return {
        "num_entities": int(len(rows)),
        "num_profile_replicates": int(len(gallery["feature_rows"])),
        "full_gallery": {
            "compound_to_profile": full_gallery_metrics_general(z_c, z_p, query_codes, gallery["codes"]),
            "profile_to_compound": full_gallery_metrics_general(z_p, z_c, gallery["codes"], query_codes),
        },
        "mocop_protocol_sampled": sampled,
    }


def evaluate_gene(
    model: ReplicateCGPAlign,
    entities: pd.DataFrame,
    features: np.ndarray,
    protein_embeddings: np.ndarray,
    modality_ids_np: np.ndarray,
    rows: np.ndarray,
    entity_to_reps: Sequence[np.ndarray],
    codes: np.ndarray,
    norm: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    ratios: Sequence[int],
    repeats: int,
    seed: int,
) -> Dict[str, Any]:
    z_g = encode_gene_entities(model, entities, protein_embeddings, modality_ids_np, rows, device, args.eval_batch_size)
    query_codes = codes[np.asarray(rows, dtype=np.int64)]
    gallery = build_replicate_gallery(rows, entity_to_reps, codes, args.eval_max_replicates_per_entity, seed)
    if args.profile_source_mode == "compound_gene":
        src = np.ones(len(gallery["feature_rows"]), dtype=np.int64)
    else:
        src = 1 + modality_ids_np[gallery["owner_entities"]].astype(np.int64)
    z_p = encode_profiles(model, features, gallery["feature_rows"], src, norm, device, args.eval_batch_size)
    sampled = sampled_metrics_general(
        z_g,
        z_p,
        query_codes,
        gallery["codes"],
        "gene_to_profile",
        "profile_to_gene",
        ratios,
        repeats,
        seed,
    )
    modality_breakdown: Dict[str, Any] = {}
    query_modality = modality_ids_np[np.asarray(rows, dtype=np.int64)]
    gallery_modality = modality_ids_np[gallery["owner_entities"]]
    for modality_name, modality_id in MODALITY_TO_ID.items():
        q_mask = query_modality == int(modality_id)
        p_mask = gallery_modality == int(modality_id)
        if not bool(q_mask.any()) or not bool(p_mask.any()):
            continue
        q = z_g[q_mask]
        p = z_p[p_mask]
        q_codes_m = query_codes[q_mask]
        p_codes_m = gallery["codes"][p_mask]
        modality_breakdown[modality_name] = {
            "num_entities": int(q_mask.sum()),
            "num_profile_replicates": int(p_mask.sum()),
            "full_gallery": {
                "gene_to_profile": full_gallery_metrics_general(q, p, q_codes_m, p_codes_m),
                "profile_to_gene": full_gallery_metrics_general(p, q, p_codes_m, q_codes_m),
            },
            "mocop_protocol_sampled": sampled_metrics_general(
                q,
                p,
                q_codes_m,
                p_codes_m,
                f"{modality_name}_to_profile",
                f"profile_to_{modality_name}",
                ratios,
                repeats,
                seed + 1000 + int(modality_id),
            ),
        }
    return {
        "num_entities": int(len(rows)),
        "num_profile_replicates": int(len(gallery["feature_rows"])),
        "modality_counts": {str(k): int(v) for k, v in entities.iloc[np.asarray(rows, dtype=np.int64)]["perturbation_modality"].value_counts().items()},
        "modality_breakdown": modality_breakdown,
        "full_gallery": {
            "gene_to_profile": full_gallery_metrics_general(z_g, z_p, query_codes, gallery["codes"]),
            "profile_to_gene": full_gallery_metrics_general(z_p, z_g, gallery["codes"], query_codes),
        },
        "mocop_protocol_sampled": sampled,
    }


def summarize_top10(compound_metrics: Dict[str, Any], gene_metrics: Dict[str, Any], ratio: int = 100) -> Dict[str, Any]:
    key = f"1:{int(ratio)}"
    c2p = compound_metrics.get("mocop_protocol_sampled", {}).get("compound_to_profile", {}).get(key, {}).get("Top10_accuracy_mean")
    p2c = compound_metrics.get("mocop_protocol_sampled", {}).get("profile_to_compound", {}).get(key, {}).get("Top10_accuracy_mean")
    g2p = gene_metrics.get("mocop_protocol_sampled", {}).get("gene_to_profile", {}).get(key, {}).get("Top10_accuracy_mean")
    p2g = gene_metrics.get("mocop_protocol_sampled", {}).get("profile_to_gene", {}).get(key, {}).get("Top10_accuracy_mean")
    values = [c2p, p2c, g2p, p2g]
    valid = [float(v) for v in values if v is not None]
    return {
        "ratio": int(ratio),
        "compound_to_profile": c2p,
        "profile_to_compound": p2c,
        "gene_to_profile": g2p,
        "profile_to_gene": p2g,
        "mean": float(np.mean(valid)) if len(valid) == 4 else None,
        "hmean": harmonic_mean(values),
    }


def evaluate_all(
    model: ReplicateCGPAlign,
    data: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    split: str,
    ratios: Sequence[int],
    repeats: int,
    seed: int,
    max_compounds: int = 0,
    max_genes: int = 0,
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    c_rows = data["compound"][f"{split}_entities"]
    g_rows = data["gene"][f"{split}_entities"]
    c_rows = maybe_subsample(c_rows, max_compounds, rng)
    g_rows = maybe_subsample(g_rows, max_genes, rng)
    compound_metrics = evaluate_compound(
        model,
        data["compound"]["entities"],
        data["compound"]["features"],
        c_rows,
        data["compound"]["entity_to_reps"],
        data["compound"]["codes"],
        data["compound"]["norm"],
        data["compound"]["graph_store"],
        args,
        device,
        ratios,
        repeats,
        seed + 101,
    )
    gene_metrics = evaluate_gene(
        model,
        data["gene"]["entities"],
        data["gene"]["features"],
        data["gene"]["protein_embeddings"],
        data["gene"]["modality_ids"],
        g_rows,
        data["gene"]["entity_to_reps"],
        data["gene"]["codes"],
        data["gene"]["norm"],
        args,
        device,
        ratios,
        repeats,
        seed + 202,
    )
    return {
        "split": split,
        "compound": compound_metrics,
        "gene": gene_metrics,
        "top10_100": summarize_top10(compound_metrics, gene_metrics, 100),
    }


def build_data(args: argparse.Namespace) -> Dict[str, Any]:
    c_entities, c_reps, c_features, c_splits = load_compound_data(args.compound_data_dir)
    g_entities, g_reps, g_features, protein_embeddings, g_splits = load_gene_data(args.gene_data_dir, args.gene_protein_embedding_dir)
    if int(c_features.shape[1]) != int(g_features.shape[1]):
        raise SystemExit(f"Profile dim mismatch: compound={c_features.shape[1]} gene={g_features.shape[1]}")
    c_entity_to_reps = build_entity_replicate_index(c_reps, len(c_entities))
    g_entity_to_reps = build_entity_replicate_index(g_reps, len(g_entities))
    c_train = eligible_compounds(c_entities, c_splits[args.compound_split_name]["train"], c_entity_to_reps)
    c_val = eligible_compounds(c_entities, c_splits[args.compound_split_name]["val"], c_entity_to_reps)
    c_test = eligible_compounds(c_entities, c_splits[args.compound_split_name]["test"], c_entity_to_reps)
    g_train = eligible_genes(g_entities, g_splits[args.gene_split_name]["train"], g_entity_to_reps, protein_embeddings)
    g_val = eligible_genes(g_entities, g_splits[args.gene_split_name]["val"], g_entity_to_reps, protein_embeddings)
    g_test = eligible_genes(g_entities, g_splits[args.gene_split_name]["test"], g_entity_to_reps, protein_embeddings)
    c_full_train_count = int(len(c_train))
    g_full_train_count = int(len(g_train))
    c_train_fraction = resolve_train_fraction(args, "compound")
    g_train_fraction = resolve_train_fraction(args, "gene")
    c_train = subsample_train_entities(c_train, c_train_fraction, int(args.seed) + 101)
    g_train = subsample_train_entities(g_train, g_train_fraction, int(args.seed) + 211)
    if args.smoke_test:
        c_train = c_train[: min(512, len(c_train))]
        c_val = c_val[: min(128, len(c_val))]
        c_test = c_test[: min(128, len(c_test))]
        g_train = g_train[: min(512, len(g_train))]
        g_val = g_val[: min(128, len(g_val))]
        g_test = g_test[: min(128, len(g_test))]
        args.epochs = min(int(args.epochs), 2)
        args.val_every = 1
        args.val_repeats = min(int(args.val_repeats), 2)
        args.num_repeats = min(int(args.num_repeats), 2)
        args.val_max_compounds = min(int(args.val_max_compounds), 128)
        args.val_max_gene_entities = min(int(args.val_max_gene_entities), 128)
    c_norm = fit_profile_normalizer(c_features, c_entity_to_reps, c_train, args.profile_norm)
    g_norm = fit_profile_normalizer(g_features, g_entity_to_reps, g_train, args.profile_norm)
    return {
        "profile_dim": int(c_features.shape[1]),
        "compound": {
            "entities": c_entities,
            "replicates": c_reps,
            "features": c_features,
            "splits": c_splits,
            "entity_to_reps": c_entity_to_reps,
            "codes": compound_entity_codes(c_entities),
            "train_entities": c_train,
            "full_train_entities": c_full_train_count,
            "train_fraction_requested": float(c_train_fraction),
            "val_entities": c_val,
            "test_entities": c_test,
            "norm": c_norm,
            "graph_store": GraphStore(c_entities["canonical_smiles"].fillna("").astype(str).tolist()),
        },
        "gene": {
            "entities": g_entities,
            "replicates": g_reps,
            "features": g_features,
            "protein_embeddings": protein_embeddings,
            "splits": g_splits,
            "entity_to_reps": g_entity_to_reps,
            "codes": gene_entity_codes(g_entities),
            "modality_ids": gene_modality_ids(g_entities),
            "train_entities": g_train,
            "full_train_entities": g_full_train_count,
            "train_fraction_requested": float(g_train_fraction),
            "val_entities": g_val,
            "test_entities": g_test,
            "norm": g_norm,
        },
    }


def build_model(args: argparse.Namespace, data: Dict[str, Any], device: torch.device) -> Tuple[ReplicateCGPAlign, Dict[str, Any]]:
    num_sources = 2 if args.profile_source_mode == "compound_gene" else 1 + len(MODALITY_TO_ID)
    model = ReplicateCGPAlign(
        profile_dim=data["profile_dim"],
        protein_dim=int(data["gene"]["protein_embeddings"].shape[1]),
        embed_dim=args.embed_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_layers=args.gnn_layers,
        gene_hidden_dims=parse_hidden_dims(args.gene_hidden_dims),
        profile_hidden_dims=parse_hidden_dims(args.profile_hidden_dims),
        modality_context_dim=args.modality_context_dim,
        dropout=args.dropout,
        profile_source_adapters=bool(args.profile_source_adapters),
        profile_source_embedding=not bool(args.disable_profile_source_embedding),
        profile_source_dropout=float(args.profile_source_dropout),
        num_profile_sources=num_sources,
        profile_input_layernorm=bool(args.profile_input_layernorm),
        profile_feature_dropout=float(args.profile_feature_dropout),
        profile_mlp_norm=bool(args.profile_mlp_norm),
        disable_gene_modality_context=bool(getattr(args, "disable_gene_modality_context", False)),
        split_gene_modality_branches=bool(getattr(args, "split_gene_modality_branches", False)),
        split_gene_shared_trunk=bool(getattr(args, "split_gene_shared_trunk", False)),
    ).to(device)
    split_gene = bool(getattr(args, "split_gene_modality_branches", False))
    init_summary = {
        "compound": load_mapped_checkpoint(
            model,
            args.init_compound_checkpoint,
            [("compound_encoder.", "compound_encoder.")],
        ),
    }
    if split_gene:
        orf_ckpt = args.init_orf_gene_checkpoint or args.init_gene_checkpoint
        crispr_ckpt = args.init_crispr_gene_checkpoint or args.init_gene_checkpoint
        if bool(getattr(args, "split_gene_shared_trunk", False)):
            init_summary["gene_split_shared"] = load_split_shared_gene_checkpoints(model, orf_ckpt, crispr_ckpt)
        else:
            init_summary["gene_orf"] = load_mapped_checkpoint(
                model,
                orf_ckpt,
                [("protein_encoder.", "gene_encoder.")],
            )
            init_summary["gene_crispr"] = load_mapped_checkpoint(
                model,
                crispr_ckpt,
                [("protein_encoder.", "crispr_gene_encoder.")],
            )
    else:
        init_summary["gene"] = load_mapped_checkpoint(
            model,
            args.init_gene_checkpoint,
            [("modality_embedding.", "modality_embedding."), ("protein_encoder.", "gene_encoder.")],
        )
    return model, init_summary


def make_optimizer(model: ReplicateCGPAlign, args: argparse.Namespace) -> torch.optim.Optimizer:
    gene_params = list(model.gene_encoder.parameters())
    if model.crispr_gene_encoder is not None:
        gene_params += list(model.crispr_gene_encoder.parameters())
    if model.gene_shared_trunk is not None:
        gene_params = list(model.gene_shared_trunk.parameters()) + gene_params
    if model.modality_embedding is not None:
        gene_params = list(model.modality_embedding.parameters()) + gene_params
    return torch.optim.AdamW(
        [
            {"params": model.compound_encoder.parameters(), "lr": float(args.branch_lr), "name": "compound_encoder"},
            {"params": gene_params, "lr": float(args.branch_lr), "name": "gene_encoder"},
            {"params": model.profile_encoder.parameters(), "lr": float(args.profile_lr), "name": "profile_encoder"},
        ],
        weight_decay=float(args.weight_decay),
    )


def train(args: argparse.Namespace, model: ReplicateCGPAlign, data: Dict[str, Any], init_summary: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    ckpt_dir = args.checkpoint_dir / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"{args.run_name}_train_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    cg_teacher = build_cg_profile_teacher(data, args)
    profile_structure_teacher = build_profile_structure_teacher(data, args)
    cg_teacher_summary = {
        "enabled": bool(cg_teacher.get("enabled", False)),
        "mode": cg_teacher.get("mode"),
        "profile_transform": cg_teacher.get("profile_transform"),
        "compound_train_entities": int(len(cg_teacher.get("compound_train_entities", []))),
        "gene_train_entities": int(len(cg_teacher.get("gene_train_entities", []))),
        "weight": float(args.cg_teacher_weight),
        "start_epoch": int(args.cg_teacher_start_epoch),
        "warmup_epochs": int(args.cg_teacher_warmup_epochs),
        "teacher_temperature": float(args.cg_teacher_temperature),
        "student_temperature": float(args.cg_student_temperature),
        "batch_topk": int(args.cg_teacher_batch_topk),
        "gene_modality_scope": str(args.cg_teacher_gene_modality_scope),
        "mutual_topk": bool(args.cg_teacher_mutual_topk),
        "min_similarity": float(args.cg_teacher_min_similarity),
    }
    profile_structure_summary = {
        "enabled": bool(profile_structure_teacher.get("enabled", False)),
        "mode": profile_structure_teacher.get("mode"),
        "components": profile_structure_teacher.get("components"),
        "profile_transform": profile_structure_teacher.get("profile_transform"),
        "compound_train_entities": int(len(profile_structure_teacher.get("compound_train_entities", []))),
        "gene_train_entities": int(len(profile_structure_teacher.get("gene_train_entities", []))),
        "weight": float(args.profile_structure_weight),
        "start_epoch": int(args.profile_structure_start_epoch),
        "warmup_epochs": int(args.profile_structure_warmup_epochs),
        "teacher_temperature": float(args.profile_structure_temperature),
        "student_temperature": float(args.profile_structure_student_temperature),
        "batch_topk": int(args.profile_structure_batch_topk),
        "target": str(args.profile_structure_target),
    }
    config = {
        **vars(args),
        "profile_dim": int(data["profile_dim"]),
        "compound_train_entities": int(len(data["compound"]["train_entities"])),
        "compound_full_train_entities": int(data["compound"]["full_train_entities"]),
        "compound_train_fraction_requested": float(data["compound"]["train_fraction_requested"]),
        "compound_train_fraction_actual": float(len(data["compound"]["train_entities"]) / max(1, int(data["compound"]["full_train_entities"]))),
        "compound_val_entities": int(len(data["compound"]["val_entities"])),
        "compound_test_entities": int(len(data["compound"]["test_entities"])),
        "gene_train_entities": int(len(data["gene"]["train_entities"])),
        "gene_full_train_entities": int(data["gene"]["full_train_entities"]),
        "gene_train_fraction_requested": float(data["gene"]["train_fraction_requested"]),
        "gene_train_fraction_actual": float(len(data["gene"]["train_entities"]) / max(1, int(data["gene"]["full_train_entities"]))),
        "gene_val_entities": int(len(data["gene"]["val_entities"])),
        "gene_test_entities": int(len(data["gene"]["test_entities"])),
        "protein_dim": int(data["gene"]["protein_embeddings"].shape[1]),
        "num_profile_sources": int(2 if args.profile_source_mode == "compound_gene" else 1 + len(MODALITY_TO_ID)),
        "init_summary": init_summary,
        "cg_profile_teacher": cg_teacher_summary,
        "profile_structure_teacher": profile_structure_summary,
        "training_definition": "replicate-aware shared profile CGP-Align; optional train-only profile-structure soft loss; no explicit compound-gene positive pairs or hard-negative masks",
    }
    write_json(ckpt_dir / "config.json", config)
    optimizer = make_optimizer(model, args)
    scaler = GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    rng = np.random.default_rng(int(args.seed))
    has_branch_init = has_required_branch_initialization(args)
    start_epoch = 1
    resume_payload: Optional[Dict[str, Any]] = None
    if args.resume_checkpoint is not None:
        resume_payload = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(resume_payload["model_state_dict"], strict=True)
        data["compound"]["norm"] = resume_payload.get("compound_profile_normalizer", data["compound"]["norm"])
        data["gene"]["norm"] = resume_payload.get("gene_profile_normalizer", data["gene"]["norm"])
        start_epoch = int(resume_payload.get("epoch", 0)) + 1
    anchor_model: Optional[ReplicateCGPAlign] = None
    if has_branch_init and not bool(args.disable_staged_training):
        anchor_model = copy.deepcopy(model).to(device)
        anchor_model.eval()
        for p in anchor_model.parameters():
            p.requires_grad = False
    steps = choose_steps(args, len(data["compound"]["train_entities"]), len(data["gene"]["train_entities"]))
    best = float(resume_payload.get("best_score", -float("inf"))) if resume_payload is not None else -float("inf")
    best_epoch = int(resume_payload.get("epoch", 0)) if resume_payload is not None else 0
    last_stage = None
    for epoch in range(start_epoch, int(args.epochs) + 1):
        stage, branch_trainable, anchor_weight = epoch_stage(args, epoch, has_branch_init)
        cg_teacher_weight = cg_teacher_weight_for_epoch(args, epoch)
        profile_structure_weight = profile_structure_weight_for_epoch(args, epoch)
        if stage != last_stage:
            set_profile_trainable(model, True)
            set_branch_trainable(model, branch_trainable)
            if bool(args.freeze_gene_modality_embedding):
                set_gene_modality_embedding_trainable(model, False)
            last_stage = stage
        model.train()
        c_losses: List[float] = []
        g_losses: List[float] = []
        g_orf_losses: List[float] = []
        g_crispr_losses: List[float] = []
        g_orf_query_counts: List[int] = []
        g_crispr_query_counts: List[int] = []
        g_orf_profile_counts: List[int] = []
        g_crispr_profile_counts: List[int] = []
        a_losses: List[float] = []
        cg_losses: List[float] = []
        pp_losses: List[float] = []
        split_align_losses: List[float] = []
        pp_cc_losses: List[float] = []
        pp_gg_losses: List[float] = []
        pp_cg_losses: List[float] = []
        total_losses: List[float] = []
        for _ in range(steps):
            c_batch = rng.choice(
                data["compound"]["train_entities"],
                size=min(int(args.compound_batch_size), len(data["compound"]["train_entities"])),
                replace=False,
            )
            g_batch = sample_gene_batch(
                data["gene"]["train_entities"],
                data["gene"]["modality_ids"],
                int(args.gene_batch_size),
                bool(getattr(args, "balanced_gene_batch_sampling", False)),
                rng,
            )
            c_rep_rows, c_rep_owner = sample_replicates_multi(
                c_batch,
                data["compound"]["entity_to_reps"],
                rng,
                args.train_replicates_per_entity,
            )
            g_rep_rows, g_rep_owner = sample_replicates_multi(
                g_batch,
                data["gene"]["entity_to_reps"],
                rng,
                args.train_replicates_per_entity,
            )
            c_code = torch.from_numpy(data["compound"]["codes"][c_batch].astype(np.int64)).to(device=device, dtype=torch.long)
            c_prof_code = torch.from_numpy(data["compound"]["codes"][c_rep_owner].astype(np.int64)).to(device=device, dtype=torch.long)
            g_code = torch.from_numpy(data["gene"]["codes"][g_batch].astype(np.int64)).to(device=device, dtype=torch.long)
            g_prof_code = torch.from_numpy(data["gene"]["codes"][g_rep_owner].astype(np.int64)).to(device=device, dtype=torch.long)
            with autocast(enabled=bool(args.amp and device.type == "cuda")):
                z_c = model.encode_compound_graphs(data["compound"]["graph_store"].get_many(c_batch.tolist()), device)
                c_profile_x = tensor_from_numpy(transform_rows(data["compound"]["features"], c_rep_rows, data["compound"]["norm"]), device)
                z_cp = model.encode_profile(c_profile_x, compound_profile_source_ids(len(c_rep_rows), device))
                c_loss = multipositive_contrastive_loss(z_c, z_cp, c_code, c_prof_code, args.compound_temperature)

                gene_x = gene_protein_tensor(data["gene"]["entities"], data["gene"]["protein_embeddings"], g_batch, device)
                gene_mod = gene_modality_tensor(data["gene"]["modality_ids"], g_batch, device)
                z_g = model.encode_gene(gene_x, gene_mod)
                g_profile_x = tensor_from_numpy(transform_rows(data["gene"]["features"], g_rep_rows, data["gene"]["norm"]), device)
                g_profile_source = gene_profile_source_ids(args, data["gene"]["modality_ids"], g_rep_owner, device)
                z_gp = model.encode_profile(g_profile_x, g_profile_source)
                g_loss_parts: Dict[str, torch.Tensor] = {}
                g_loss_counts: Dict[str, int] = {}
                if bool(getattr(args, "split_gene_modality_branches", False)):
                    g_profile_mod = gene_modality_tensor(data["gene"]["modality_ids"], g_rep_owner, device)
                    g_loss, g_loss_parts, g_loss_counts = modality_balanced_contrastive_loss_parts(
                        z_g,
                        z_gp,
                        g_code,
                        g_prof_code,
                        gene_mod,
                        g_profile_mod,
                        args.gene_temperature,
                        reduction=args.gene_branch_loss_reduction,
                    )
                elif bool(getattr(args, "gene_modality_soft_balanced_loss", False)):
                    g_profile_mod = gene_modality_tensor(data["gene"]["modality_ids"], g_rep_owner, device)
                    g_loss = modality_soft_balanced_contrastive_loss(
                        z_g,
                        z_gp,
                        g_code,
                        g_prof_code,
                        gene_mod,
                        g_profile_mod,
                        args.gene_temperature,
                    )
                elif bool(args.gene_modality_balanced_loss):
                    g_profile_mod = gene_modality_tensor(data["gene"]["modality_ids"], g_rep_owner, device)
                    g_loss = modality_balanced_contrastive_loss(
                        z_g,
                        z_gp,
                        g_code,
                        g_prof_code,
                        gene_mod,
                        g_profile_mod,
                        args.gene_temperature,
                    )
                else:
                    g_loss = multipositive_contrastive_loss(z_g, z_gp, g_code, g_prof_code, args.gene_temperature)

                anchor_loss = c_loss.new_tensor(0.0)
                if anchor_model is not None and float(anchor_weight) > 0:
                    with torch.no_grad():
                        t_z_c = anchor_model.encode_compound_graphs(data["compound"]["graph_store"].get_many(c_batch.tolist()), device)
                        t_z_g = anchor_model.encode_gene(gene_x, gene_mod)
                    anchor_loss = cosine_anchor_loss([z_c, z_g], [t_z_c, t_z_g])
                cg_loss = c_loss.new_tensor(0.0)
                if float(cg_teacher_weight) > 0:
                    cg_loss = cg_profile_teacher_loss(z_c, z_g, c_batch, g_batch, gene_mod, cg_teacher, args, device)
                pp_loss = c_loss.new_tensor(0.0)
                pp_parts = {"cc": 0.0, "gg": 0.0, "cg": 0.0}
                if float(profile_structure_weight) > 0:
                    if args.profile_structure_target == "profile":
                        z_c_struct = mean_profile_latents_by_entity(z_cp, c_rep_owner, c_batch, device)
                        z_g_struct = mean_profile_latents_by_entity(z_gp, g_rep_owner, g_batch, device)
                    else:
                        z_c_struct = z_c
                        z_g_struct = z_g
                    pp_loss, pp_parts = profile_structure_loss(
                        z_c_struct,
                        z_g_struct,
                        c_batch,
                        g_batch,
                        profile_structure_teacher,
                        args,
                        device,
                    )
                split_align_weight = 0.0
                if (
                    bool(getattr(args, "split_gene_modality_branches", False))
                    and float(getattr(args, "split_gene_distribution_alignment_weight", 0.0)) > 0
                    and int(epoch) >= int(getattr(args, "split_gene_distribution_alignment_start_epoch", 1))
                ):
                    split_align_weight = float(getattr(args, "split_gene_distribution_alignment_weight", 0.0))
                split_align_loss = (
                    split_gene_distribution_alignment_loss(z_g, gene_mod)
                    if split_align_weight > 0
                    else c_loss.new_tensor(0.0)
                )
                loss = (
                    float(args.compound_loss_weight) * c_loss
                    + float(args.gene_loss_weight) * g_loss
                    + float(anchor_weight) * anchor_loss
                    + float(cg_teacher_weight) * cg_loss
                    + float(profile_structure_weight) * pp_loss
                    + float(split_align_weight) * split_align_loss
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch}, stage={stage}")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            c_losses.append(float(c_loss.detach().cpu().item()))
            g_losses.append(float(g_loss.detach().cpu().item()))
            if "orf" in g_loss_parts:
                g_orf_losses.append(float(g_loss_parts["orf"].detach().cpu().item()))
            if "crispr" in g_loss_parts:
                g_crispr_losses.append(float(g_loss_parts["crispr"].detach().cpu().item()))
            g_orf_query_counts.append(int(g_loss_counts.get("orf_query", 0)))
            g_crispr_query_counts.append(int(g_loss_counts.get("crispr_query", 0)))
            g_orf_profile_counts.append(int(g_loss_counts.get("orf_profile", 0)))
            g_crispr_profile_counts.append(int(g_loss_counts.get("crispr_profile", 0)))
            a_losses.append(float(anchor_loss.detach().cpu().item()))
            cg_losses.append(float(cg_loss.detach().cpu().item()))
            pp_losses.append(float(pp_loss.detach().cpu().item()))
            split_align_losses.append(float(split_align_loss.detach().cpu().item()))
            pp_cc_losses.append(float(pp_parts.get("cc", 0.0)))
            pp_gg_losses.append(float(pp_parts.get("gg", 0.0)))
            pp_cg_losses.append(float(pp_parts.get("cg", 0.0)))
            total_losses.append(float(loss.detach().cpu().item()))
        rec: Dict[str, Any] = {
            "epoch": int(epoch),
            "stage": stage,
            "branch_trainable": bool(branch_trainable),
            "anchor_weight": float(anchor_weight),
            "cg_teacher_weight": float(cg_teacher_weight),
            "profile_structure_weight": float(profile_structure_weight),
            "split_gene_distribution_alignment_weight": float(getattr(args, "split_gene_distribution_alignment_weight", 0.0)),
            "train_compound_loss": float(np.mean(c_losses)),
            "train_gene_loss": float(np.mean(g_losses)),
            "train_gene_orf_loss": float(np.mean(g_orf_losses)) if g_orf_losses else None,
            "train_gene_crispr_loss": float(np.mean(g_crispr_losses)) if g_crispr_losses else None,
            "train_gene_orf_query_count": float(np.mean(g_orf_query_counts)) if g_orf_query_counts else 0.0,
            "train_gene_crispr_query_count": float(np.mean(g_crispr_query_counts)) if g_crispr_query_counts else 0.0,
            "train_gene_orf_profile_count": float(np.mean(g_orf_profile_counts)) if g_orf_profile_counts else 0.0,
            "train_gene_crispr_profile_count": float(np.mean(g_crispr_profile_counts)) if g_crispr_profile_counts else 0.0,
            "train_anchor_loss": float(np.mean(a_losses)),
            "train_cg_teacher_loss": float(np.mean(cg_losses)),
            "train_profile_structure_loss": float(np.mean(pp_losses)),
            "train_split_gene_distribution_alignment_loss": float(np.mean(split_align_losses)),
            "train_profile_structure_cc_loss": float(np.mean(pp_cc_losses)),
            "train_profile_structure_gg_loss": float(np.mean(pp_gg_losses)),
            "train_profile_structure_cg_loss": float(np.mean(pp_cg_losses)),
            "train_total_loss": float(np.mean(total_losses)),
            "steps_per_epoch": int(steps),
        }
        if epoch % int(args.val_every) == 0 or epoch == int(args.epochs):
            val = evaluate_all(
                model,
                data,
                args,
                device,
                "val",
                [int(args.val_negative_ratio)],
                int(args.val_repeats),
                int(args.seed) + epoch,
                int(args.val_max_compounds),
                int(args.val_max_gene_entities),
            )
            top = val["top10_100"]
            rec.update({
                "val_c2p_Top10": top.get("compound_to_profile"),
                "val_p2c_Top10": top.get("profile_to_compound"),
                "val_g2p_Top10": top.get("gene_to_profile"),
                "val_p2g_Top10": top.get("profile_to_gene"),
                "val_mean_Top10": top.get("mean"),
                "val_hmean_Top10": top.get("hmean"),
                "selection_active": bool(int(epoch) >= int(args.min_selection_epoch)),
            })
            score = top.get("hmean") if top.get("hmean") is not None else top.get("mean")
            candidate_epochs = set(parse_ints(args.save_candidate_epochs)) if str(args.save_candidate_epochs).strip() else set()
            if candidate_epochs and int(epoch) in candidate_epochs:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": int(epoch),
                        "best_score": score,
                        "selection_metric": "val_hmean_Top10_snapshot",
                        "config": config,
                        "compound_profile_normalizer": data["compound"]["norm"],
                        "gene_profile_normalizer": data["gene"]["norm"],
                    },
                    ckpt_dir / f"val_epoch{int(epoch):04d}.pt",
                )
            if int(epoch) >= int(args.min_selection_epoch) and score is not None and float(score) > best:
                best = float(score)
                best_epoch = int(epoch)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": int(epoch),
                        "best_score": best,
                        "selection_metric": "val_hmean_Top10",
                        "config": config,
                        "compound_profile_normalizer": data["compound"]["norm"],
                        "gene_profile_normalizer": data["gene"]["norm"],
                    },
                    ckpt_dir / "best_model.pt",
                )
        append_jsonl(log_path, rec)
        print(json.dumps(rec, sort_keys=True), flush=True)
    if not (ckpt_dir / "best_model.pt").exists():
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": int(args.epochs),
                "best_score": best,
                "selection_metric": "val_hmean_Top10",
                "config": config,
                "compound_profile_normalizer": data["compound"]["norm"],
                "gene_profile_normalizer": data["gene"]["norm"],
            },
            ckpt_dir / "best_model.pt",
        )
    return {"best": best, "best_epoch": best_epoch}


def evaluate_checkpoint(args: argparse.Namespace, model: ReplicateCGPAlign, data: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    ckpt_path = args.checkpoint_dir / args.run_name / "best_model.pt"
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    data["compound"]["norm"] = payload.get("compound_profile_normalizer", data["compound"]["norm"])
    data["gene"]["norm"] = payload.get("gene_profile_normalizer", data["gene"]["norm"])
    metrics = evaluate_all(
        model,
        data,
        args,
        device,
        "test",
        parse_ints(args.negative_ratios),
        int(args.num_repeats),
        int(args.seed) + 999,
    )
    metrics.update({
        "run_name": args.run_name,
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": payload.get("epoch"),
        "best_score": payload.get("best_score"),
        "selection_metric": payload.get("selection_metric"),
        "compound_data_dir": str(args.compound_data_dir),
        "gene_data_dir": str(args.gene_data_dir),
        "profile_source_adapters": bool(args.profile_source_adapters),
        "staged_training": not bool(args.disable_staged_training),
        "has_branch_init": has_required_branch_initialization(args),
    })
    out_path = args.output_dir / f"{args.run_name}_test_metrics.json"
    write_json(out_path, metrics)
    print(json.dumps({"test_metrics": str(out_path), "top10_100": metrics.get("top10_100")}, sort_keys=True), flush=True)
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    data = build_data(args)
    model, init_summary = build_model(args, data, device)
    if not args.eval_only:
        train(args, model, data, init_summary, device)
    if not args.train_only:
        evaluate_checkpoint(args, model, data, device)


if __name__ == "__main__":
    main()
