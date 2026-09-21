from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from cgp_align.reliability import (
    entity_key_codes_from_entities,
    estimate_train_only_gene_reliability,
    parse_allowed_modalities,
    strict_no_leakage_assertions,
)
from cgp_common import metrics_from_ranks, retrieval_ranks_single_positive, set_seed, tensor_from_numpy, write_dataframe, write_json


GENE_MOCOP_ROOT = Path("output/cgp_align/gene_mocop")
GENE_MOCOP_DATA_DIR = GENE_MOCOP_ROOT / "data"
GENE_MOCOP_CKPT_DIR = GENE_MOCOP_ROOT / "checkpoints"
GENE_MOCOP_LOG_DIR = GENE_MOCOP_ROOT / "logs"
GENE_MOCOP_EVAL_DIR = GENE_MOCOP_ROOT / "eval"
DEFAULT_PROTEIN_EMB_DIR = Path("protein_embeddings/protein_encoder_ablation/esm2_650m")
MODALITY_TO_ID = {"orf": 0, "crispr": 1}


def modality_filter_from_arg(value: str) -> List[str]:
    return parse_allowed_modalities(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gene-MoCoP: MoCoP-style protein/gene-perturbation profile contrastive learning.")
    parser.add_argument("--data_dir", type=Path, default=GENE_MOCOP_DATA_DIR)
    parser.add_argument("--protein_embedding_dir", type=Path, default=DEFAULT_PROTEIN_EMB_DIR)
    parser.add_argument("--structure_embedding_dir", type=Path, default=None)
    parser.add_argument("--structure_embedding_file", default="protein_sequence_embeddings.npy")
    parser.add_argument(
        "--kg_graph_dir",
        type=Path,
        default=None,
        help=(
            "Optional external gene KG graph directory with edge_index.npy/edge_weight.npy. "
            "When set, a trainable GNN adapter encodes all genes from public annotations under protocol-A cold-gene evaluation."
        ),
    )
    parser.add_argument("--kg_encoder", choices=["none", "graphsage"], default="none")
    parser.add_argument("--kg_hidden_dim", type=int, default=512)
    parser.add_argument("--kg_layers", type=int, default=2)
    parser.add_argument("--kg_dropout", type=float, default=0.1)
    parser.add_argument("--kg_fusion", choices=["gated_residual", "residual", "replace"], default="gated_residual")
    parser.add_argument("--kg_zero_init_adapter", action="store_true")
    parser.add_argument("--checkpoint_dir", type=Path, default=GENE_MOCOP_CKPT_DIR)
    parser.add_argument("--log_dir", type=Path, default=GENE_MOCOP_LOG_DIR)
    parser.add_argument("--output_dir", type=Path, default=GENE_MOCOP_EVAL_DIR)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--protein_encoder_name", default="esm2_650m_precomputed")
    parser.add_argument("--structure_encoder_name", default="")
    parser.add_argument("--split_name", choices=["random_entity", "cold_gene"], default="cold_gene")
    parser.add_argument(
        "--modalities",
        default="orf,crispr",
        help="Comma/plus separated modalities, or aliases: crispr, orf, mixed, CRISPR_ORF_mixed.",
    )
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--protein_hidden_dims", default="512,256")
    parser.add_argument("--profile_hidden_dims", default="512,256")
    parser.add_argument("--fusion_dim", type=int, default=512)
    parser.add_argument(
        "--protein_fusion",
        choices=["seq_only", "struct_only", "concat", "gated", "residual"],
        default="seq_only",
        help="Optional sequence/structure fusion mode for the protein branch.",
    )
    parser.add_argument("--modality_context_dim", type=int, default=32)
    parser.add_argument(
        "--protein_conditioning",
        choices=["concat", "film", "concat_film", "separate_heads", "none"],
        default="concat",
        help="How perturbation modality/treatment type conditions the protein branch.",
    )
    parser.add_argument(
        "--entity_feature_mode",
        choices=["none", "basic", "metadata", "metadata_hash", "metadata_plate", "metadata_well"],
        default="none",
        help=(
            "Optional entity-specific inputs appended to the gene/protein embedding. "
            "basic uses replicate/source/reagent-family summaries; metadata adds ORF construct metadata; "
            "metadata_hash adds hashed reagent/broad-sample IDs for diagnostic upper-bound tests; "
            "metadata_plate additionally adds hashed plate metadata; "
            "metadata_well additionally adds hashed well-position metadata for diagnostic shortcut tests."
        ),
    )
    parser.add_argument("--entity_metadata_dir", type=Path, default=Path("raw/jump_metadata"))
    parser.add_argument("--entity_external_annotation_dir", type=Path, default=Path("raw/external_annotations/JUMP-Target-master"))
    parser.add_argument(
        "--dynamic_entity_encoding",
        choices=["none", "gene_symbol", "reagent", "gene_reagent", "gene_modality_reagent"],
        default="none",
        help=(
            "Trainable entity token/hash embedding appended to the gene branch. "
            "Uses gene/reagent/JCP/broad/modality tokens only; plate/well metadata is intentionally excluded."
        ),
    )
    parser.add_argument("--dynamic_hash_buckets", type=int, default=4096)
    parser.add_argument("--dynamic_token_dim", type=int, default=128)
    parser.add_argument("--dynamic_max_tokens", type=int, default=64)
    parser.add_argument("--dynamic_embedding_dropout", type=float, default=0.1)
    parser.add_argument("--disable_modality_context", action="store_true")
    parser.add_argument("--disable_profile_source_embedding", action="store_true")
    parser.add_argument("--mask_crossmod_negatives", action="store_true")
    parser.add_argument(
        "--mask_same_gene_negatives",
        action="store_true",
        help="Treat every non-diagonal entity with the same gene_symbol as neutral, including same-modality different JCP2022 reagents.",
    )
    parser.add_argument("--same_gene_same_modality_positive_weight", type=float, default=0.0)
    parser.add_argument("--same_gene_crossmod_positive_weight", type=float, default=0.0)
    parser.add_argument("--same_gene_sibling_batch_prob", type=float, default=0.0)
    parser.add_argument("--profile_norm", choices=["none", "train_zscore", "source_train_zscore"], default="train_zscore")
    parser.add_argument(
        "--train_profile_mode",
        choices=["random_replicate", "mean_profile"],
        default="random_replicate",
        help="Use one random replicate per entity step, or the normalized entity mean profile.",
    )
    parser.add_argument(
        "--eval_profile_mode",
        choices=["entity_mean", "replicate_gallery"],
        default="entity_mean",
        help=(
            "Evaluation profile gallery semantics. entity_mean keeps the historical one-profile-per-entity protocol; "
            "replicate_gallery keeps every replicate profile and treats any profile with the same entity_key as a positive."
        ),
    )
    parser.add_argument(
        "--eval_max_replicates_per_entity",
        type=int,
        default=0,
        help="Optional cap for replicate_gallery evaluation; <=0 uses all replicate profiles for each selected entity.",
    )
    parser.add_argument("--train_fraction", type=float, default=1.0, help="Fraction of eligible train entities to use.")
    parser.add_argument(
        "--sample_weight_mode",
        choices=["none", "replicate_count", "split_half_cosine"],
        default="none",
        help="Optional train-only entity weighting for noisy profile observations.",
    )
    parser.add_argument("--sample_weight_power", type=float, default=1.0)
    parser.add_argument("--sample_weight_min", type=float, default=0.25)
    parser.add_argument("--sample_weight_max", type=float, default=2.0)
    parser.add_argument("--sample_weight_splits", type=int, default=5)
    parser.add_argument(
        "--profile_geometry_loss_weight",
        type=float,
        default=0.0,
        help="Weakly preserve input-profile cosine geometry inside the learned profile latent space.",
    )
    parser.add_argument(
        "--gene_profile_geometry_loss_weight",
        type=float,
        default=0.0,
        help="Preserve input-profile cosine geometry inside the learned gene/protein latent space.",
    )
    parser.add_argument(
        "--profile_neighbor_distill_weight",
        type=float,
        default=0.0,
        help="Use profile-profile similarities as soft targets for cross-branch contrastive logits.",
    )
    parser.add_argument("--profile_neighbor_target_temperature", type=float, default=0.2)
    parser.add_argument(
        "--profile_replicate_contrastive_loss_weight",
        type=float,
        default=0.0,
        help="Contrast two independently sampled replicate profiles from the same entity to stabilize profile latents.",
    )
    parser.add_argument(
        "--profile_adversary_well_weight",
        type=float,
        default=0.0,
        help="Gradient-reversal loss to remove well-position information from profile latents.",
    )
    parser.add_argument(
        "--profile_adversary_plate_weight",
        type=float,
        default=0.0,
        help="Gradient-reversal loss to remove plate identity information from profile latents.",
    )
    parser.add_argument(
        "--profile_adversary_source_weight",
        type=float,
        default=0.0,
        help="Gradient-reversal loss to remove source/modality batch information from profile latents.",
    )
    parser.add_argument("--profile_adversary_hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--steps_per_epoch", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--strict_no_leakage", action="store_true")
    parser.add_argument("--gene_head", choices=["single", "multi_expert"], default="single")
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--expert_temperature_init", type=float, default=0.07)
    parser.add_argument("--expert_temperature_min", type=float, default=0.03)
    parser.add_argument("--expert_temperature_max", type=float, default=0.30)
    parser.add_argument("--gene_loss_type", choices=["original", "soft_clip"], default="original")
    parser.add_argument("--use_reliability_weight", action="store_true")
    parser.add_argument("--reliability_default_guide_consistency", type=float, default=0.5)
    parser.add_argument("--soft_clip_eta", type=float, default=0.2)
    parser.add_argument("--soft_clip_gamma", type=float, default=2.0)
    parser.add_argument("--lambda_expert_div", type=float, default=0.01)
    parser.add_argument("--lambda_expert_balance", type=float, default=0.01)
    parser.add_argument(
        "--strict_split_log_every",
        type=int,
        default=0,
        help="If >0, log train/val/test Top10@100 every N epochs using up to val_max_queries rows per split.",
    )
    parser.add_argument("--historical_mixed_baseline", type=float, default=47.91)
    parser.add_argument("--historical_crispr_only_baseline", type=float, default=37.93)
    parser.add_argument("--historical_baseline", type=float, default=None)
    parser.add_argument(
        "--clip_loss_type",
        choices=["softmax", "sigmoid"],
        default="softmax",
        help="Base cross-branch CLIP objective. softmax keeps the original InfoNCE loss; sigmoid uses independent pair labels.",
    )
    parser.add_argument(
        "--sigmoid_neg_weight",
        type=float,
        default=1.0,
        help="Relative weight for non-matching pairs when --clip_loss_type sigmoid.",
    )
    parser.add_argument(
        "--rank_loss_weight",
        type=float,
        default=0.0,
        help="Optional pairwise rank loss that pushes the diagonal match above hard negatives.",
    )
    parser.add_argument("--rank_margin", type=float, default=0.2)
    parser.add_argument(
        "--rank_hard_negatives",
        type=int,
        default=128,
        help="Use only the hardest K valid negatives per row/column for rank loss; <=0 uses all valid negatives.",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--disable_ema", action="store_true")
    parser.add_argument("--val_every", type=int, default=10)
    parser.add_argument("--val_max_queries", type=int, default=2048)
    parser.add_argument("--val_negative_ratio", type=int, default=100)
    parser.add_argument("--val_repeats", type=int, default=5)
    parser.add_argument("--eval_batch_size", type=int, default=2048)
    parser.add_argument("--negative_ratios", default="100,1000")
    parser.add_argument("--num_repeats", type=int, default=20)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--train_only", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_hidden_dims(text: str) -> List[int]:
    return parse_ints(text)


def subsample_train_entities(rows: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"train_fraction must be in (0, 1], got {fraction}")
    if fraction >= 0.999999 or len(rows) <= 1:
        return rows
    n = max(1, int(round(len(rows) * fraction)))
    rng = np.random.default_rng(int(seed))
    positions = np.sort(rng.choice(np.arange(len(rows)), size=n, replace=False))
    return rows[positions]


def remap_splits_after_entity_filter(splits: Dict[str, Any], keep_entities: np.ndarray) -> Dict[str, Any]:
    old_to_new = {int(old): i for i, old in enumerate(np.asarray(keep_entities, dtype=np.int64).tolist())}
    out = copy.deepcopy(splits)
    for split_name in ["random_entity", "cold_gene"]:
        if split_name not in out:
            continue
        for fold in ["train", "val", "test"]:
            out[split_name][fold] = [old_to_new[int(i)] for i in out[split_name].get(fold, []) if int(i) in old_to_new]
    return out


def load_gene_mocop_data(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    entity_path = args.data_dir / "gene_mocop_entities.parquet"
    replicate_path = args.data_dir / "gene_mocop_replicates.parquet"
    feature_path = args.data_dir / "gene_mocop_replicate_features.npy"
    split_path = args.data_dir / "splits_gene_mocop.json"
    protein_path = args.protein_embedding_dir / "protein_sequence_embeddings.npy"
    missing = [str(p) for p in [entity_path, replicate_path, feature_path, split_path, protein_path] if not p.exists()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(missing))

    entities = pd.read_parquet(entity_path).reset_index(drop=True)
    replicates = pd.read_parquet(replicate_path).reset_index(drop=True)
    features = np.load(feature_path).astype(np.float32)
    protein_embeddings = np.load(protein_path).astype(np.float32)
    splits = json.loads(split_path.read_text(encoding="utf-8"))

    keep_modalities = set(modality_filter_from_arg(str(args.modalities)))
    if keep_modalities and keep_modalities != set(MODALITY_TO_ID):
        keep_entities = entities.index[entities["perturbation_modality"].astype(str).str.lower().isin(keep_modalities)].to_numpy(dtype=np.int64)
        entities = entities.iloc[keep_entities].reset_index(drop=True)
        splits = remap_splits_after_entity_filter(splits, keep_entities)
        keep_set = set(keep_entities.astype(int).tolist())
        old_to_new = {int(old): i for i, old in enumerate(keep_entities.tolist())}
        replicates = replicates[replicates["entity_index"].astype(int).isin(keep_set)].copy()
        replicates["entity_index"] = replicates["entity_index"].astype(int).map(old_to_new).astype(np.int64)
        replicates = replicates.reset_index(drop=True)

    if args.smoke_test:
        keep_genes = sorted(entities["gene_symbol"].drop_duplicates().astype(str).tolist())[:512]
        keep_entities = entities.index[entities["gene_symbol"].astype(str).isin(keep_genes)].to_numpy(dtype=np.int64)
        entities = entities.iloc[keep_entities].reset_index(drop=True)
        splits = remap_splits_after_entity_filter(splits, keep_entities)
        keep_set = set(keep_entities.astype(int).tolist())
        old_to_new = {int(old): i for i, old in enumerate(keep_entities.tolist())}
        replicates = replicates[replicates["entity_index"].astype(int).isin(keep_set)].copy()
        replicates["entity_index"] = replicates["entity_index"].astype(int).map(old_to_new).astype(np.int64)
        replicates = replicates.reset_index(drop=True)

    return entities, replicates, features, protein_embeddings, splits


def load_structure_embeddings(args: argparse.Namespace, num_genes: int) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    if args.structure_embedding_dir is None:
        if args.protein_fusion != "seq_only":
            raise SystemExit("--protein_fusion requires --structure_embedding_dir unless protein_fusion=seq_only.")
        return None, {"enabled": False}
    path = args.structure_embedding_dir / args.structure_embedding_file
    if not path.exists():
        raise SystemExit(f"Missing structure embedding file: {path}")
    embeddings = np.load(path).astype(np.float32)
    if embeddings.ndim != 2:
        raise SystemExit(f"Structure embeddings must be a 2D matrix, got shape {embeddings.shape}: {path}")
    if embeddings.shape[0] != int(num_genes):
        raise SystemExit(f"Structure embedding row count {embeddings.shape[0]} does not match num_genes {num_genes}: {path}")
    summary_path = args.structure_embedding_dir / "protein_embedding_summary.json"
    summary: Dict[str, Any] = {
        "enabled": True,
        "structure_embedding_dir": str(args.structure_embedding_dir),
        "structure_embedding_file": str(args.structure_embedding_file),
        "structure_embedding_dim": int(embeddings.shape[1]),
    }
    if summary_path.exists():
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["source_summary"] = loaded
        except json.JSONDecodeError:
            summary["source_summary_error"] = f"Could not parse {summary_path}"
    return embeddings, summary


def load_kg_graph(args: argparse.Namespace, num_genes: int) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, Any]]:
    if str(getattr(args, "kg_encoder", "none") or "none") == "none" or getattr(args, "kg_graph_dir", None) is None:
        return None, {"enabled": False}
    graph_dir = Path(args.kg_graph_dir)
    edge_index_path = graph_dir / "edge_index.npy"
    edge_weight_path = graph_dir / "edge_weight.npy"
    missing = [str(p) for p in [edge_index_path, edge_weight_path] if not p.exists()]
    if missing:
        raise SystemExit("Missing KG graph files:\n" + "\n".join(missing))
    edge_index = np.load(edge_index_path).astype(np.int64)
    edge_weight = np.load(edge_weight_path).astype(np.float32)
    if edge_index.shape[0] != 2:
        raise SystemExit(f"KG edge_index must have shape [2, num_edges], got {edge_index.shape}: {edge_index_path}")
    if edge_weight.shape[0] != edge_index.shape[1]:
        raise SystemExit(f"KG edge_weight length {edge_weight.shape[0]} != edges {edge_index.shape[1]}: {edge_weight_path}")
    valid = (
        (edge_index[0] >= 0)
        & (edge_index[0] < int(num_genes))
        & (edge_index[1] >= 0)
        & (edge_index[1] < int(num_genes))
        & np.isfinite(edge_weight)
        & (edge_weight > 0)
    )
    dropped = int(np.sum(~valid))
    edge_index = edge_index[:, valid]
    edge_weight = edge_weight[valid]
    summary_path = graph_dir / "kg_graph_summary.json"
    summary: Dict[str, Any] = {
        "enabled": True,
        "kg_graph_dir": str(graph_dir),
        "encoder": str(getattr(args, "kg_encoder", "none")),
        "hidden_dim": int(getattr(args, "kg_hidden_dim", 512)),
        "layers": int(getattr(args, "kg_layers", 2)),
        "dropout": float(getattr(args, "kg_dropout", 0.1)),
        "fusion": str(getattr(args, "kg_fusion", "gated_residual")),
        "zero_init_adapter": bool(getattr(args, "kg_zero_init_adapter", False)),
        "num_edges": int(edge_index.shape[1]),
        "dropped_edges": int(dropped),
        "uses_profile_or_retrieval_labels": False,
        "cold_gene_protocol": "A: public KG annotations for val/test genes are visible; val/test profile pairs are never supervised.",
    }
    if summary_path.exists():
        try:
            summary["source_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary["source_summary_error"] = f"Could not parse {summary_path}"
    return {"edge_index": edge_index, "edge_weight": edge_weight}, summary


def _first_token(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return text.split("|")[0].strip()


def _split_tokens(value: Any) -> List[str]:
    if value is None or pd.isna(value):
        return []
    out = [x.strip() for x in str(value).split("|") if x.strip()]
    return out


def _stable_bucket(text: str, buckets: int) -> int:
    digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) % int(buckets)


def _add_hash_block(
    blocks: List[np.ndarray],
    names: List[str],
    values: Sequence[Any],
    block_name: str,
    buckets: int,
    split_pipe: bool = True,
) -> Dict[str, Any]:
    block = np.zeros((len(values), int(buckets)), dtype=np.float32)
    covered = 0
    token_count = 0
    for i, value in enumerate(values):
        tokens = _split_tokens(value) if split_pipe else [_first_token(value)]
        tokens = [t for t in tokens if t]
        if not tokens:
            continue
        covered += 1
        weight = 1.0 / np.sqrt(float(len(tokens)))
        for token in tokens:
            block[i, _stable_bucket(f"{block_name}:{token}", buckets)] += float(weight)
            token_count += 1
    blocks.append(block)
    names.extend([f"{block_name}_{i}" for i in range(int(buckets))])
    return {"buckets": int(buckets), "covered_entities": int(covered), "tokens": int(token_count)}


def _add_numeric_feature(
    blocks: List[np.ndarray],
    names: List[str],
    name: str,
    values: Sequence[Any],
    log1p: bool = False,
    add_missing: bool = False,
) -> Dict[str, Any]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float32)
    missing = ~np.isfinite(arr)
    fill = float(np.nanmedian(arr)) if np.any(~missing) else 0.0
    arr[missing] = fill
    if log1p:
        arr = np.log1p(np.maximum(arr, 0.0)).astype(np.float32)
    mean = float(np.mean(arr)) if arr.size else 0.0
    std = float(np.std(arr)) if arr.size else 1.0
    if std < 1e-6:
        scaled = np.zeros_like(arr, dtype=np.float32)
    else:
        scaled = ((arr - mean) / std).astype(np.float32)
    blocks.append(scaled[:, None])
    names.append(name)
    if add_missing:
        blocks.append(missing.astype(np.float32)[:, None])
        names.append(f"{name}_missing")
    return {"missing_entities": int(missing.sum()), "mean": mean, "std": std}


def _load_optional_table(path: Path, sep: str = ",") -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path, sep=sep)


def build_entity_features(entities: pd.DataFrame, args: argparse.Namespace) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    mode = str(getattr(args, "entity_feature_mode", "none") or "none")
    if mode == "none":
        return None, {"enabled": False, "mode": "none", "dim": 0}

    blocks: List[np.ndarray] = []
    names: List[str] = []
    summary: Dict[str, Any] = {"enabled": True, "mode": mode, "blocks": {}}

    for col in ["num_replicates", "num_jcp2022_ids", "num_plates", "num_sources"]:
        if col in entities.columns:
            summary["blocks"][col] = _add_numeric_feature(blocks, names, col, entities[col].to_numpy(), log1p=True)

    if "metadata_sources" in entities.columns:
        summary["blocks"]["metadata_sources_hash"] = _add_hash_block(
            blocks, names, entities["metadata_sources"].tolist(), "metadata_source", 16
        )

    jcp_ids = entities["jcp2022_ids"].map(_first_token) if "jcp2022_ids" in entities.columns else entities["reagent_id"].map(_first_token)
    jcp_numbers = []
    jcp_prefixes = []
    for value in jcp_ids.tolist():
        tail = value.split("_", 1)[1] if "_" in value else ""
        number = int(tail) if tail.isdigit() else 0
        jcp_numbers.append(number)
        jcp_prefixes.append(str(number // 1000) if number else "")
    summary["blocks"]["jcp_prefix_hash"] = _add_hash_block(blocks, names, jcp_prefixes, "jcp_prefix", 8, split_pipe=False)

    if mode in {"metadata", "metadata_hash", "metadata_plate", "metadata_well"}:
        metadata_dir = Path(getattr(args, "entity_metadata_dir", Path("raw/jump_metadata")))
        external_dir = Path(getattr(args, "entity_external_annotation_dir", Path("raw/external_annotations/JUMP-Target-master")))
        summary["metadata_dir"] = str(metadata_dir)
        summary["entity_external_annotation_dir"] = str(external_dir)

        orf_meta = _load_optional_table(metadata_dir / "orf.csv.gz")
        if orf_meta is not None:
            by_jcp = orf_meta.drop_duplicates("Metadata_JCP2022").set_index("Metadata_JCP2022")
            joined = by_jcp.reindex(jcp_ids.tolist())
            present = joined["Metadata_Vector"].notna().to_numpy(dtype=np.float32)
            blocks.append(present[:, None])
            names.append("orf_metadata_present")
            summary["blocks"]["orf_metadata_present"] = {"covered_entities": int(present.sum())}
            summary["blocks"]["orf_vector_hash"] = _add_hash_block(
                blocks, names, joined["Metadata_Vector"].fillna("").tolist(), "orf_vector", 8, split_pipe=False
            )
            summary["blocks"]["orf_transcript_hash"] = _add_hash_block(
                blocks, names, joined["Metadata_Transcript"].fillna("").tolist(), "orf_transcript", 64, split_pipe=False
            )
            summary["blocks"]["orf_prot_match"] = _add_numeric_feature(
                blocks, names, "orf_prot_match", joined["Metadata_Prot_Match"].to_numpy(), add_missing=True
            )
            summary["blocks"]["orf_insert_length"] = _add_numeric_feature(
                blocks, names, "orf_insert_length", joined["Metadata_Insert_Length"].to_numpy(), log1p=True, add_missing=True
            )
        else:
            summary["orf_metadata_missing"] = str(metadata_dir / "orf.csv.gz")

        target_orf = _load_optional_table(external_dir / "JUMP-Target-1_orf_metadata.tsv", sep="\t")
        if target_orf is not None and "broad_sample_ids" in entities.columns:
            by_broad = target_orf.drop_duplicates("broad_sample").set_index("broad_sample")
            broad_ids = entities["broad_sample_ids"].map(_first_token).tolist()
            joined = by_broad.reindex(broad_ids)
            seq = joined.get("target_matching_region_seq", pd.Series([""] * len(joined), index=joined.index)).fillna("").astype(str)
            lengths = seq.map(len).to_numpy(dtype=np.float32)
            gc = seq.map(lambda s: (s.upper().count("G") + s.upper().count("C")) / max(len(s), 1)).to_numpy(dtype=np.float32)
            summary["blocks"]["target1_orf_sequence_length"] = _add_numeric_feature(
                blocks, names, "target1_orf_sequence_length", lengths, log1p=True, add_missing=False
            )
            summary["blocks"]["target1_orf_sequence_gc"] = _add_numeric_feature(
                blocks, names, "target1_orf_sequence_gc", gc, add_missing=False
            )
            summary["blocks"]["target1_orf_sequence_present"] = {"covered_entities": int((lengths > 0).sum())}

        target_crispr = _load_optional_table(external_dir / "JUMP-Target-1_crispr_metadata.tsv", sep="\t")
        if target_crispr is not None and "broad_sample_ids" in entities.columns:
            by_broad = target_crispr.drop_duplicates("broad_sample").set_index("broad_sample")
            broad_ids = entities["broad_sample_ids"].map(_first_token).tolist()
            joined = by_broad.reindex(broad_ids)
            seq = joined.get("target_sequence", pd.Series([""] * len(joined), index=joined.index)).fillna("").astype(str)
            lengths = seq.map(len).to_numpy(dtype=np.float32)
            gc = seq.map(lambda s: (s.upper().count("G") + s.upper().count("C")) / max(len(s), 1)).to_numpy(dtype=np.float32)
            summary["blocks"]["target1_crispr_sequence_length"] = _add_numeric_feature(
                blocks, names, "target1_crispr_sequence_length", lengths, log1p=True, add_missing=False
            )
            summary["blocks"]["target1_crispr_sequence_gc"] = _add_numeric_feature(
                blocks, names, "target1_crispr_sequence_gc", gc, add_missing=False
            )
            summary["blocks"]["target1_crispr_sequence_present"] = {"covered_entities": int((lengths > 0).sum())}

    if mode in {"metadata_hash", "metadata_plate"}:
        if "reagent_id" in entities.columns:
            summary["blocks"]["reagent_id_hash"] = _add_hash_block(
                blocks, names, entities["reagent_id"].tolist(), "reagent_id", 256, split_pipe=False
            )
        if "broad_sample_ids" in entities.columns:
            summary["blocks"]["broad_sample_hash"] = _add_hash_block(
                blocks, names, entities["broad_sample_ids"].tolist(), "broad_sample", 256
            )
        summary["blocks"]["jcp_number"] = _add_numeric_feature(blocks, names, "jcp_number", jcp_numbers)

    if mode == "metadata_plate" and "metadata_plates" in entities.columns:
        summary["blocks"]["metadata_plates_hash"] = _add_hash_block(
            blocks, names, entities["metadata_plates"].tolist(), "metadata_plate", 128
        )
    if mode == "metadata_well" and "metadata_wells" in entities.columns:
        summary["blocks"]["metadata_wells_hash"] = _add_hash_block(
            blocks, names, entities["metadata_wells"].tolist(), "metadata_well", 128
        )

    if not blocks:
        return None, {"enabled": False, "mode": mode, "dim": 0, "warning": "No entity features were built."}

    features = np.concatenate(blocks, axis=1).astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    summary["dim"] = int(features.shape[1])
    summary["feature_names"] = names
    return features, summary


def _token_ngrams(text: str, prefix: str, n_values: Sequence[int] = (2, 3, 4)) -> List[str]:
    clean = "".join(ch for ch in str(text).upper() if ch.isalnum())
    if not clean:
        return []
    out = [f"{prefix}:raw:{clean}", f"{prefix}:len:{min(len(clean), 24)}"]
    padded = f"^{clean}$"
    for n in n_values:
        if len(padded) < n:
            continue
        out.extend(f"{prefix}:ng{n}:{padded[i:i+n]}" for i in range(len(padded) - n + 1))
    return out


def build_dynamic_entity_tokens(entities: pd.DataFrame, args: argparse.Namespace) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    mode = str(getattr(args, "dynamic_entity_encoding", "none") or "none")
    if mode == "none":
        return None, {"enabled": False, "mode": "none", "dim": 0}
    buckets = int(getattr(args, "dynamic_hash_buckets", 4096))
    max_tokens = int(getattr(args, "dynamic_max_tokens", 64))
    if buckets <= 0 or max_tokens <= 0:
        raise ValueError("--dynamic_hash_buckets and --dynamic_max_tokens must be positive when dynamic entity encoding is enabled.")

    token_ids = np.zeros((len(entities), max_tokens), dtype=np.int64)
    token_counts: List[int] = []
    truncated = 0
    token_type_counts: Dict[str, int] = {}

    include_gene = mode in {"gene_symbol", "gene_reagent", "gene_modality_reagent"}
    include_reagent = mode in {"reagent", "gene_reagent", "gene_modality_reagent"}
    include_modality = mode == "gene_modality_reagent"

    for row_idx, row in enumerate(entities.itertuples(index=False)):
        tokens: List[str] = []
        if include_gene:
            gene = str(getattr(row, "gene_symbol", "") or "")
            source_idx = str(getattr(row, "source_gene_index", "") or "")
            gene_tokens = [f"gene_symbol:{gene}", f"source_gene_index:{source_idx}"]
            gene_tokens.extend(_token_ngrams(gene, "gene_symbol"))
            tokens.extend(gene_tokens)
            token_type_counts["gene"] = token_type_counts.get("gene", 0) + len(gene_tokens)
        if include_modality:
            modality = str(getattr(row, "perturbation_modality", "") or "").lower()
            modality_tokens = [f"modality:{modality}"]
            tokens.extend(modality_tokens)
            token_type_counts["modality"] = token_type_counts.get("modality", 0) + len(modality_tokens)
        if include_reagent:
            reagent = _first_token(getattr(row, "reagent_id", ""))
            jcp = _first_token(getattr(row, "jcp2022_ids", ""))
            broad = _first_token(getattr(row, "broad_sample_ids", ""))
            reagent_tokens: List[str] = []
            for label, value in [("reagent", reagent), ("jcp", jcp), ("broad_sample", broad)]:
                if not value:
                    continue
                reagent_tokens.append(f"{label}:raw:{value}")
                reagent_tokens.extend(_token_ngrams(value, label, n_values=(2, 3)))
            if jcp and "_" in jcp:
                tail = jcp.split("_", 1)[1]
                if tail.isdigit():
                    number = int(tail)
                    reagent_tokens.extend(
                        [
                            f"jcp_bin_1k:{number // 1000}",
                            f"jcp_bin_10k:{number // 10000}",
                            f"jcp_last2:{number % 100:02d}",
                        ]
                    )
            tokens.extend(reagent_tokens)
            token_type_counts["reagent"] = token_type_counts.get("reagent", 0) + len(reagent_tokens)

        # De-duplicate while preserving order so repeated IDs do not dominate the embedding bag.
        deduped = list(dict.fromkeys(t for t in tokens if t))
        token_counts.append(len(deduped))
        if len(deduped) > max_tokens:
            truncated += 1
            deduped = deduped[:max_tokens]
        for j, token in enumerate(deduped):
            token_ids[row_idx, j] = 1 + _stable_bucket(f"dynamic_entity:{token}", buckets)

    counts = np.asarray(token_counts, dtype=np.int64)
    summary: Dict[str, Any] = {
        "enabled": True,
        "mode": mode,
        "dim": int(getattr(args, "dynamic_token_dim", 128)),
        "hash_buckets": int(buckets),
        "max_tokens": int(max_tokens),
        "dropout": float(getattr(args, "dynamic_embedding_dropout", 0.1)),
        "mean_tokens_per_entity": float(counts.mean()) if counts.size else 0.0,
        "median_tokens_per_entity": float(np.median(counts)) if counts.size else 0.0,
        "max_observed_tokens": int(counts.max()) if counts.size else 0,
        "truncated_entities": int(truncated),
        "token_type_counts": token_type_counts,
        "excluded_metadata": ["metadata_plates", "metadata_wells", "plate", "well"],
    }
    return token_ids, summary


def build_feature_label_codes(replicates: pd.DataFrame, num_features: int, column: str) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    if column not in replicates.columns:
        return None, {"enabled": False, "column": column, "reason": "missing column"}
    values = replicates[column].fillna("").astype(str)
    codes, uniques = pd.factorize(values, sort=True)
    labels = np.full(int(num_features), -1, dtype=np.int64)
    feature_rows = replicates["feature_index"].to_numpy(dtype=np.int64)
    valid = (feature_rows >= 0) & (feature_rows < int(num_features))
    labels[feature_rows[valid]] = codes.astype(np.int64)[valid]
    return labels, {
        "enabled": True,
        "column": column,
        "num_labels": int(len(uniques)),
        "covered_features": int(np.sum(labels >= 0)),
    }


def build_profile_adversary_labels(
    replicates: pd.DataFrame,
    num_features: int,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Optional[np.ndarray]], Dict[str, Any]]:
    specs = {
        "well": ("Metadata_Well", float(getattr(args, "profile_adversary_well_weight", 0.0))),
        "plate": ("Metadata_Plate", float(getattr(args, "profile_adversary_plate_weight", 0.0))),
        "source": ("Metadata_Source", float(getattr(args, "profile_adversary_source_weight", 0.0))),
    }
    labels: Dict[str, Optional[np.ndarray]] = {}
    summary: Dict[str, Any] = {"enabled": False, "heads": {}, "excluded_as_query_inputs": True}
    for name, (column, weight) in specs.items():
        if weight <= 0:
            labels[name] = None
            summary["heads"][name] = {"enabled": False, "weight": float(weight)}
            continue
        arr, head_summary = build_feature_label_codes(replicates, num_features, column)
        head_summary["weight"] = float(weight)
        labels[name] = arr
        summary["heads"][name] = head_summary
        summary["enabled"] = bool(summary["enabled"] or (arr is not None))
    return labels, summary


def modality_ids(entities: pd.DataFrame) -> np.ndarray:
    mapped = entities["perturbation_modality"].astype(str).str.lower().map(MODALITY_TO_ID)
    if mapped.isna().any():
        bad = sorted(entities.loc[mapped.isna(), "perturbation_modality"].astype(str).unique().tolist())
        raise ValueError(f"Unsupported modalities: {bad}")
    return mapped.to_numpy(dtype=np.int64)


def gene_codes(entities: pd.DataFrame) -> np.ndarray:
    codes, _ = pd.factorize(entities["gene_symbol"].astype(str), sort=True)
    return codes.astype(np.int64)


def entity_key_codes(entities: pd.DataFrame) -> np.ndarray:
    codes, _ = entity_key_codes_from_entities(entities)
    return codes.astype(np.int64)


def eligible_entities(entities: pd.DataFrame, rows: Sequence[int], protein_embeddings: np.ndarray) -> np.ndarray:
    idx = np.asarray(rows, dtype=np.int64)
    if idx.size == 0:
        return idx
    source_idx = entities.iloc[idx]["source_gene_index"].fillna(-1).astype(np.int64).to_numpy()
    has_seq = entities.iloc[idx]["has_protein_sequence"].astype(bool).to_numpy()
    keep = has_seq & (source_idx >= 0) & (source_idx < protein_embeddings.shape[0])
    return idx[keep]


def build_entity_replicate_index(replicates: pd.DataFrame, num_entities: int) -> List[np.ndarray]:
    out: List[List[int]] = [[] for _ in range(num_entities)]
    for row in replicates.itertuples(index=False):
        out[int(row.entity_index)].append(int(row.feature_index))
    return [np.asarray(x, dtype=np.int64) for x in out]


def sample_replicate_features(entity_rows: np.ndarray, entity_to_reps: Sequence[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    out = np.zeros(len(entity_rows), dtype=np.int64)
    for i, entity_idx in enumerate(entity_rows.astype(int).tolist()):
        reps = entity_to_reps[entity_idx]
        if reps.size == 0:
            raise RuntimeError(f"Entity {entity_idx} has no replicate features.")
        out[i] = int(rng.choice(reps))
    return out


def fit_profile_normalizer(
    features: np.ndarray,
    entity_to_reps: Sequence[np.ndarray],
    train_entities: np.ndarray,
    entity_mods: np.ndarray,
    mode: str,
) -> Dict[str, Any]:
    if mode == "none":
        return {"mode": "none", "fit_split": "none"}
    train_reps = np.concatenate([entity_to_reps[int(i)] for i in train_entities if entity_to_reps[int(i)].size])
    if train_reps.size == 0:
        raise RuntimeError("No train replicate profiles available for normalization.")
    if mode == "train_zscore":
        x = features[train_reps]
        mean = np.nanmean(x, axis=0).astype(np.float32)
        std = np.nanstd(x, axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        return {"mode": mode, "mean": mean, "std": std, "fit_split": "train"}
    if mode == "source_train_zscore":
        stats: Dict[int, Dict[str, np.ndarray]] = {}
        for modality_id in sorted(set(entity_mods.tolist())):
            ents = train_entities[entity_mods[train_entities] == modality_id]
            reps = np.concatenate([entity_to_reps[int(i)] for i in ents if entity_to_reps[int(i)].size]) if len(ents) else train_reps
            if reps.size == 0:
                reps = train_reps
            x = features[reps]
            mean = np.nanmean(x, axis=0).astype(np.float32)
            std = np.nanstd(x, axis=0).astype(np.float32)
            std[std < 1e-6] = 1.0
            stats[int(modality_id)] = {"mean": mean, "std": std, "num_fit_replicates": np.asarray([reps.size], dtype=np.int64)}
        return {"mode": mode, "stats": stats, "fit_split": "train"}
    raise ValueError(f"Unsupported profile_norm: {mode}")


def transform_feature_rows(features: np.ndarray, feature_rows: np.ndarray, modality: np.ndarray, norm: Dict[str, Any]) -> np.ndarray:
    x = features[feature_rows].astype(np.float32).copy()
    mode = norm.get("mode", "none")
    if mode == "none":
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "train_zscore":
        x = (x - norm["mean"]) / norm["std"]
    elif mode == "source_train_zscore":
        for modality_id, stats in norm["stats"].items():
            mask = modality == int(modality_id)
            if np.any(mask):
                x[mask] = (x[mask] - stats["mean"]) / stats["std"]
    else:
        raise ValueError(f"Unsupported normalizer mode: {mode}")
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def compute_entity_mean_profiles(
    features: np.ndarray,
    entity_to_reps: Sequence[np.ndarray],
    entity_mods: np.ndarray,
    norm: Dict[str, Any],
) -> np.ndarray:
    out = np.zeros((len(entity_to_reps), features.shape[1]), dtype=np.float32)
    for entity_idx, reps in enumerate(entity_to_reps):
        if reps.size == 0:
            continue
        rep_mod = np.full(len(reps), int(entity_mods[entity_idx]), dtype=np.int64)
        out[entity_idx] = transform_feature_rows(features, reps, rep_mod, norm).mean(axis=0)
    return out


def normalizer_summary(norm: Dict[str, Any]) -> Dict[str, Any]:
    mode = norm.get("mode", "none")
    if mode == "none":
        return {"mode": "none", "fit_split": norm.get("fit_split", "none")}
    if mode == "train_zscore":
        return {"mode": mode, "fit_split": norm.get("fit_split", "train"), "mean_shape": list(norm["mean"].shape), "std_shape": list(norm["std"].shape)}
    if mode == "source_train_zscore":
        return {
            "mode": mode,
            "fit_split": norm.get("fit_split", "train"),
            "sources": {
                str(k): {
                    "mean_shape": list(v["mean"].shape),
                    "std_shape": list(v["std"].shape),
                    "num_fit_replicates": int(v["num_fit_replicates"][0]),
                }
                for k, v in norm["stats"].items()
            },
        }
    return {"mode": str(mode)}


def build_entity_sample_weights(
    features: np.ndarray,
    entity_to_reps: Sequence[np.ndarray],
    train_entities: np.ndarray,
    entity_mods: np.ndarray,
    norm: Dict[str, Any],
    mode: str,
    power: float,
    min_weight: float,
    max_weight: float,
    num_splits: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    num_entities = len(entity_to_reps)
    mode = str(mode or "none")
    weights = np.ones(num_entities, dtype=np.float32)
    train_entities = np.asarray(train_entities, dtype=np.int64)
    if mode == "none":
        return weights, {"mode": "none"}

    if mode == "replicate_count":
        counts = np.asarray([len(reps) for reps in entity_to_reps], dtype=np.float32)
        raw = np.sqrt(np.maximum(counts, 1.0))
        weights = raw.astype(np.float32)
        summary: Dict[str, Any] = {
            "mode": mode,
            "replicate_count_min": int(np.min(counts[train_entities])) if train_entities.size else 0,
            "replicate_count_median": float(np.median(counts[train_entities])) if train_entities.size else 0.0,
            "replicate_count_max": int(np.max(counts[train_entities])) if train_entities.size else 0,
        }
    elif mode == "split_half_cosine":
        rng = np.random.default_rng(seed)
        reliability = np.full(num_entities, np.nan, dtype=np.float32)
        split_count = max(1, int(num_splits))
        for entity_idx in train_entities.astype(int).tolist():
            reps = np.asarray(entity_to_reps[entity_idx], dtype=np.int64)
            if reps.size < 2:
                continue
            vals: List[float] = []
            modality_id = int(entity_mods[entity_idx])
            for _ in range(split_count):
                shuffled = reps.copy()
                rng.shuffle(shuffled)
                cut = max(1, shuffled.size // 2)
                left = shuffled[:cut]
                right = shuffled[cut:]
                if right.size == 0:
                    right = shuffled[-1:]
                left_mod = np.full(len(left), modality_id, dtype=np.int64)
                right_mod = np.full(len(right), modality_id, dtype=np.int64)
                left_vec = transform_feature_rows(features, left, left_mod, norm).mean(axis=0)
                right_vec = transform_feature_rows(features, right, right_mod, norm).mean(axis=0)
                denom = float(np.linalg.norm(left_vec) * np.linalg.norm(right_vec))
                vals.append(float(np.dot(left_vec, right_vec) / denom) if denom > 1e-8 else 0.0)
            reliability[entity_idx] = float(np.mean(vals)) if vals else np.nan
        fallback = float(np.nanmedian(reliability[train_entities])) if np.isfinite(reliability[train_entities]).any() else 0.0
        reliability = np.where(np.isfinite(reliability), reliability, fallback).astype(np.float32)
        raw = np.clip((reliability + 1.0) * 0.5, 1e-6, None)
        weights = raw.astype(np.float32)
        summary = {
            "mode": mode,
            "num_splits": int(split_count),
            "reliability_fallback": float(fallback),
            "train_reliability_min": float(np.min(reliability[train_entities])) if train_entities.size else None,
            "train_reliability_median": float(np.median(reliability[train_entities])) if train_entities.size else None,
            "train_reliability_max": float(np.max(reliability[train_entities])) if train_entities.size else None,
        }
    else:
        raise ValueError(f"Unsupported sample_weight_mode: {mode}")

    p = float(power)
    if p != 1.0:
        weights = np.power(np.clip(weights, 1e-6, None), p).astype(np.float32)
    train_mean = float(np.mean(weights[train_entities])) if train_entities.size else 1.0
    if train_mean > 1e-8:
        weights = weights / train_mean
    lo = float(min_weight)
    hi = float(max_weight)
    if hi > 0 and hi >= lo:
        weights = np.clip(weights, lo, hi)
    train_mean = float(np.mean(weights[train_entities])) if train_entities.size else 1.0
    if train_mean > 1e-8:
        weights = weights / train_mean
    summary.update(
        {
            "power": float(power),
            "min_weight": float(min_weight),
            "max_weight": float(max_weight),
            "train_weight_min": float(np.min(weights[train_entities])) if train_entities.size else None,
            "train_weight_median": float(np.median(weights[train_entities])) if train_entities.size else None,
            "train_weight_max": float(np.max(weights[train_entities])) if train_entities.size else None,
        }
    )
    return weights.astype(np.float32), summary


def weighted_mean_loss(loss_vec: torch.Tensor, sample_weight: Optional[torch.Tensor]) -> torch.Tensor:
    if sample_weight is None:
        return loss_vec.mean()
    weight = sample_weight.to(dtype=loss_vec.dtype, device=loss_vec.device)
    return (loss_vec * weight).sum() / weight.sum().clamp_min(1e-8)


class MoCoPMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], embed_dim: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev = int(input_dim)
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, int(hidden)), nn.GELU(), nn.Dropout(dropout)])
            prev = int(hidden)
        layers.append(nn.Linear(prev, embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiExpertGeneHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_experts: int,
        temperature_init: float,
        temperature_min: float,
        temperature_max: float,
        dropout: float,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.embed_dim = int(embed_dim)
        self.temperature_min = float(temperature_min)
        self.temperature_max = float(temperature_max)
        self.expert_proj = nn.Linear(int(input_dim), int(num_experts) * int(embed_dim))
        self.gate = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(input_dim), int(num_experts)),
        )
        init = float(np.log(max(float(temperature_init), 1e-6)))
        self.log_temperature = nn.Parameter(torch.full((int(num_experts),), init, dtype=torch.float32))

    def temperatures(self) -> torch.Tensor:
        return torch.exp(self.log_temperature).clamp(float(self.temperature_min), float(self.temperature_max))

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw_experts = self.expert_proj(h).view(h.shape[0], self.num_experts, self.embed_dim)
        expert_raw_norm = raw_experts.norm(dim=-1)
        experts = F.normalize(raw_experts, dim=-1)
        gate_logits = self.gate(h)
        gate_probs = F.softmax(gate_logits, dim=-1)
        return {
            "experts": experts,
            "expert_raw_norm": expert_raw_norm,
            "gate_logits": gate_logits,
            "gate_probs": gate_probs,
            "temperatures": self.temperatures(),
        }


class GradientReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.scale * grad_output, None


def gradient_reverse(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return GradientReverseFn.apply(x, float(scale))


class ProfileAdversary(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_labels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(embed_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(num_labels)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def is_multi_expert_output(x: Any) -> bool:
    return isinstance(x, dict) and "experts" in x and "gate_probs" in x


def multi_expert_logits(gene_out: Dict[str, torch.Tensor], z_profile: torch.Tensor) -> torch.Tensor:
    experts = gene_out["experts"].float()
    profile = F.normalize(z_profile.float(), dim=-1)
    gate = gene_out["gate_probs"].float().clamp_min(1e-8)
    tau = gene_out["temperatures"].float().clamp(0.03, 0.30)
    dots = torch.einsum("bed,nd->ben", experts, profile)
    logits = dots / tau[None, :, None] + torch.log(gate)[:, :, None]
    return torch.logsumexp(logits, dim=1)


def gene_profile_logits(z_gene: Any, z_profile: torch.Tensor, temperature: float) -> torch.Tensor:
    if is_multi_expert_output(z_gene):
        return multi_expert_logits(z_gene, z_profile)
    return z_gene.float() @ z_profile.float().T / float(temperature)


def gene_output_as_single(z_gene: Any) -> torch.Tensor:
    if is_multi_expert_output(z_gene):
        pooled = torch.sum(z_gene["experts"] * z_gene["gate_probs"].unsqueeze(-1), dim=1)
        return F.normalize(pooled, dim=-1)
    return z_gene


def multi_expert_regularization(z_gene: Any) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    if not is_multi_expert_output(z_gene):
        zero = z_gene.sum() * 0.0 if isinstance(z_gene, torch.Tensor) else torch.tensor(0.0)
        return zero, zero, {"expert_diversity_loss": 0.0, "expert_load_balance_loss": 0.0, "expert_gate_entropy": 0.0, "expert_gate_usage": []}
    experts = z_gene["experts"].float()
    gate = z_gene["gate_probs"].float()
    sim = torch.einsum("bed,bfd->bef", experts, experts)
    e = int(experts.shape[1])
    eye = torch.eye(e, dtype=torch.bool, device=experts.device)
    div_loss = sim[:, ~eye].abs().mean() if e > 1 else sim.sum() * 0.0
    mean_gate = gate.mean(dim=0)
    balance_loss = torch.sum(mean_gate * mean_gate)
    entropy = -(gate * gate.clamp_min(1e-8).log()).sum(dim=1).mean()
    pairwise = sim.detach().mean(dim=0)
    raw_norm = z_gene.get("expert_raw_norm")
    raw_norm_f = raw_norm.float() if isinstance(raw_norm, torch.Tensor) else None
    return div_loss, balance_loss, {
        "expert_diversity_loss": float(div_loss.detach().cpu()),
        "expert_load_balance_loss": float(balance_loss.detach().cpu()),
        "expert_gate_entropy": float(entropy.detach().cpu()),
        "effective_num_experts": float(torch.exp(entropy).detach().cpu()),
        "expert_gate_usage": [float(x) for x in mean_gate.detach().cpu().tolist()],
        "max_gate_prob": float(mean_gate.max().detach().cpu()) if mean_gate.numel() else None,
        "min_gate_prob": float(mean_gate.min().detach().cpu()) if mean_gate.numel() else None,
        "expert_pairwise_cosine_mean": float(div_loss.detach().cpu()),
        "expert_pairwise_cosine_matrix": [[float(v) for v in row] for row in pairwise.detach().cpu().tolist()],
        "expert_norm_mean": float(raw_norm_f.detach().cpu().mean()) if raw_norm_f is not None and raw_norm_f.numel() else None,
        "expert_norm_std": float(raw_norm_f.detach().cpu().std(unbiased=False)) if raw_norm_f is not None and raw_norm_f.numel() else None,
    }


def expert_temperature_values(z_gene: Any) -> List[float]:
    if not is_multi_expert_output(z_gene):
        return []
    return [float(x) for x in z_gene["temperatures"].detach().cpu().tolist()]


def logit_scale_diagnostics(logits: torch.Tensor, positive_code: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        x = logits.detach().float()
        code = positive_code.detach().view(-1)
        pos_mask = code[:, None].eq(code[None, :])
        neg_mask = ~pos_mask
        pos = x[pos_mask]
        neg = x[neg_mask]
        pos_mean = pos.mean() if pos.numel() else x.new_tensor(float("nan"))
        neg_mean = neg.mean() if neg.numel() else x.new_tensor(float("nan"))
        return {
            "logits_mean": float(x.mean().cpu()) if x.numel() else None,
            "logits_std": float(x.std(unbiased=False).cpu()) if x.numel() else None,
            "positive_logits_mean": float(pos_mean.cpu()) if pos.numel() else None,
            "negative_logits_mean": float(neg_mean.cpu()) if neg.numel() else None,
            "positive_minus_negative_margin": float((pos_mean - neg_mean).cpu()) if pos.numel() and neg.numel() else None,
            "max_logit": float(x.max().cpu()) if x.numel() else None,
            "min_logit": float(x.min().cpu()) if x.numel() else None,
        }


def gate_usage_by_modality(z_gene: Any, modality: torch.Tensor) -> Dict[str, List[float]]:
    if not is_multi_expert_output(z_gene):
        return {}
    gate = z_gene["gate_probs"].detach().float()
    out: Dict[str, List[float]] = {}
    for name, modality_id in MODALITY_TO_ID.items():
        mask = modality.detach().eq(int(modality_id))
        if torch.any(mask):
            out[name] = [float(x) for x in gate[mask].mean(dim=0).cpu().tolist()]
    return out


def temperature_diagnostics(z_gene: Any, temperature_min: float = 0.03, temperature_max: float = 0.30) -> Dict[str, Any]:
    if not is_multi_expert_output(z_gene):
        return {}
    tau = z_gene["temperatures"].detach().float().cpu().numpy()
    lo = float(tau.min()) if tau.size else None
    hi = float(tau.max()) if tau.size else None
    boundary = (tau <= float(temperature_min) + 1e-7) | (tau >= float(temperature_max) - 1e-7)
    return {
        "temperature_per_expert": [float(x) for x in tau.tolist()],
        "temperature_min": lo,
        "temperature_max": hi,
        "temperature_mean": float(tau.mean()) if tau.size else None,
        "temperature_clamped_fraction": float(boundary.mean()) if tau.size else None,
        "temperature_any_clamped": bool(boundary.any()) if tau.size else False,
    }


def parameter_grad_norm(parameters: Sequence[torch.nn.Parameter]) -> Tuple[Optional[float], bool]:
    sq = 0.0
    seen = False
    finite = True
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad.detach()
        seen = True
        finite = finite and bool(torch.isfinite(g).all().item())
        sq += float(torch.sum(g.float() * g.float()).cpu())
    return (float(np.sqrt(sq)) if seen else None), finite


def model_gradient_diagnostics(model: nn.Module) -> Dict[str, Any]:
    gene_params: List[torch.nn.Parameter] = []
    if getattr(model, "protein_encoder", None) is not None:
        gene_params.extend(list(model.protein_encoder.parameters()))
    if getattr(model, "protein_encoders", None) is not None:
        gene_params.extend(list(model.protein_encoders.parameters()))
    for name in ["seq_fusion_proj", "structure_fusion_proj", "fusion_gate", "structure_residual", "modality_embedding", "protein_film", "dynamic_token_embedding"]:
        module = getattr(model, name, None)
        if module is not None:
            gene_params.extend(list(module.parameters()))
    moe_params = list(model.multi_expert_head.parameters()) if getattr(model, "multi_expert_head", None) is not None else []
    profile_params = list(model.profile_encoder.parameters())
    gene_norm, gene_finite = parameter_grad_norm(gene_params)
    moe_norm, moe_finite = parameter_grad_norm(moe_params)
    profile_norm, profile_finite = parameter_grad_norm(profile_params)
    return {
        "gene_encoder_grad_norm": gene_norm,
        "moe_head_grad_norm": moe_norm,
        "profile_encoder_grad_norm": profile_norm,
        "gradients_all_finite": bool(gene_finite and moe_finite and profile_finite),
    }


def has_nonfinite_tensor(x: Any) -> bool:
    if isinstance(x, torch.Tensor):
        return not bool(torch.isfinite(x.detach()).all().item())
    if isinstance(x, dict):
        return any(has_nonfinite_tensor(v) for v in x.values() if isinstance(v, torch.Tensor))
    return False


def finite_mean(values: Sequence[Any]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def mean_vector(values: Sequence[Sequence[float]]) -> Optional[List[float]]:
    arrs = [np.asarray(v, dtype=np.float64) for v in values if v is not None and len(v)]
    if not arrs:
        return None
    return [float(x) for x in np.vstack(arrs).mean(axis=0).tolist()]


def mean_matrix(values: Sequence[Sequence[Sequence[float]]]) -> Optional[List[List[float]]]:
    arrs = [np.asarray(v, dtype=np.float64) for v in values if v is not None and len(v)]
    if not arrs:
        return None
    return [[float(x) for x in row] for row in np.stack(arrs, axis=0).mean(axis=0).tolist()]


class GeneMoCoP(nn.Module):
    def __init__(
        self,
        protein_dim: int,
        profile_dim: int,
        structure_dim: Optional[int],
        protein_hidden_dims: Sequence[int],
        profile_hidden_dims: Sequence[int],
        embed_dim: int,
        fusion_dim: int,
        protein_fusion: str,
        modality_context_dim: int,
        dropout: float,
        use_modality_context: bool,
        use_profile_source_embedding: bool,
        protein_conditioning: str = "concat",
        dynamic_token_vocab_size: int = 0,
        dynamic_token_dim: int = 0,
        dynamic_embedding_dropout: float = 0.0,
        profile_adversary_hidden_dim: int = 256,
        num_profile_well_labels: int = 0,
        num_profile_plate_labels: int = 0,
        num_profile_source_labels: int = 0,
        gene_head: str = "single",
        num_experts: int = 4,
        expert_temperature_init: float = 0.07,
        expert_temperature_min: float = 0.03,
        expert_temperature_max: float = 0.30,
    ):
        super().__init__()
        if protein_fusion not in {"seq_only", "struct_only", "concat", "gated", "residual"}:
            raise ValueError(f"Unsupported protein_fusion: {protein_fusion}")
        if protein_fusion != "seq_only" and not structure_dim:
            raise ValueError(f"protein_fusion={protein_fusion} requires structure_dim.")
        self.protein_fusion = str(protein_fusion)
        self.protein_dim = int(protein_dim)
        self.structure_dim = int(structure_dim or 0)
        self.fusion_dim = int(fusion_dim)
        if self.protein_fusion == "seq_only":
            base_dim = int(protein_dim)
        elif self.protein_fusion == "struct_only":
            base_dim = int(self.structure_dim)
        elif self.protein_fusion == "concat":
            base_dim = int(protein_dim + self.structure_dim)
        elif self.protein_fusion == "gated":
            base_dim = int(fusion_dim)
            self.seq_fusion_proj = nn.Linear(protein_dim, base_dim)
            self.structure_fusion_proj = nn.Linear(self.structure_dim, base_dim)
            self.fusion_gate = nn.Linear(protein_dim + self.structure_dim, base_dim)
        elif self.protein_fusion == "residual":
            base_dim = int(protein_dim)
            self.structure_residual = nn.Linear(self.structure_dim, base_dim)
            nn.init.zeros_(self.structure_residual.weight)
            nn.init.zeros_(self.structure_residual.bias)
        else:
            raise ValueError(f"Unsupported protein_fusion: {protein_fusion}")
        self.dynamic_token_dim = int(dynamic_token_dim) if int(dynamic_token_vocab_size) > 0 else 0
        self.dynamic_token_embedding = (
            nn.Embedding(int(dynamic_token_vocab_size) + 1, self.dynamic_token_dim, padding_idx=0)
            if self.dynamic_token_dim > 0
            else None
        )
        self.dynamic_dropout = nn.Dropout(float(dynamic_embedding_dropout)) if self.dynamic_token_dim > 0 else None
        base_dim = int(base_dim + self.dynamic_token_dim)
        if not use_modality_context:
            protein_conditioning = "none"
        if protein_conditioning not in {"concat", "film", "concat_film", "separate_heads", "none"}:
            raise ValueError(f"Unsupported protein_conditioning: {protein_conditioning}")
        self.protein_conditioning = str(protein_conditioning)
        self.gene_head = str(gene_head)
        self.use_modality_context = self.protein_conditioning != "none"
        self.use_profile_source_embedding = bool(use_profile_source_embedding)
        context_dim = int(modality_context_dim) if self.use_modality_context else 0
        self.modality_embedding = nn.Embedding(len(MODALITY_TO_ID), context_dim) if context_dim > 0 else None
        self.protein_film = None
        self.profile_source_embedding = nn.Embedding(len(MODALITY_TO_ID), profile_dim) if self.use_profile_source_embedding else None
        if self.protein_conditioning in {"film", "concat_film"}:
            self.protein_film = nn.Embedding(len(MODALITY_TO_ID), base_dim * 2)
            nn.init.zeros_(self.protein_film.weight)
        if self.protein_conditioning == "separate_heads":
            self.protein_encoder = None
            self.protein_encoders = nn.ModuleList(
            [MoCoPMLP(base_dim, protein_hidden_dims, embed_dim, dropout) for _ in range(len(MODALITY_TO_ID))]
            )
        else:
            input_dim = base_dim + (context_dim if self.protein_conditioning in {"concat", "concat_film"} else 0)
            self.protein_encoder = MoCoPMLP(input_dim, protein_hidden_dims, embed_dim, dropout)
            self.protein_encoders = None
        if self.gene_head == "multi_expert":
            self.multi_expert_head = MultiExpertGeneHead(
                embed_dim,
                embed_dim,
                int(num_experts),
                float(expert_temperature_init),
                float(expert_temperature_min),
                float(expert_temperature_max),
                float(dropout),
            )
        elif self.gene_head == "single":
            self.multi_expert_head = None
        else:
            raise ValueError(f"Unsupported gene_head: {gene_head}")
        self.profile_encoder = MoCoPMLP(profile_dim, profile_hidden_dims, embed_dim, dropout)
        self.profile_well_adversary = (
            ProfileAdversary(embed_dim, profile_adversary_hidden_dim, num_profile_well_labels, dropout)
            if int(num_profile_well_labels) > 1
            else None
        )
        self.profile_plate_adversary = (
            ProfileAdversary(embed_dim, profile_adversary_hidden_dim, num_profile_plate_labels, dropout)
            if int(num_profile_plate_labels) > 1
            else None
        )
        self.profile_source_adversary = (
            ProfileAdversary(embed_dim, profile_adversary_hidden_dim, num_profile_source_labels, dropout)
            if int(num_profile_source_labels) > 1
            else None
        )

    def fuse_protein_inputs(self, x: torch.Tensor, structure: Optional[torch.Tensor]) -> torch.Tensor:
        if self.protein_fusion == "seq_only":
            return x
        if structure is None:
            raise ValueError(f"protein_fusion={self.protein_fusion} requires structure embeddings.")
        if self.protein_fusion == "struct_only":
            return structure
        if self.protein_fusion == "concat":
            return torch.cat([x, structure], dim=-1)
        if self.protein_fusion == "gated":
            z_seq = self.seq_fusion_proj(x)
            z_struct = self.structure_fusion_proj(structure)
            gate = torch.sigmoid(self.fusion_gate(torch.cat([x, structure], dim=-1)))
            return gate * z_seq + (1.0 - gate) * z_struct
        if self.protein_fusion == "residual":
            return x + self.structure_residual(structure)
        raise ValueError(f"Unsupported protein_fusion: {self.protein_fusion}")

    def encode_dynamic_tokens(self, token_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.dynamic_token_embedding is None:
            return None
        if token_ids is None:
            raise ValueError("dynamic_token_ids are required when dynamic entity encoding is enabled.")
        emb = self.dynamic_token_embedding(token_ids.long())
        mask = token_ids.ne(0).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1).to(dtype=emb.dtype)
        pooled = (emb * mask.to(dtype=emb.dtype)).sum(dim=1) / denom
        if self.dynamic_dropout is not None:
            pooled = self.dynamic_dropout(pooled)
        return pooled

    def encode_protein(
        self,
        x: torch.Tensor,
        modality: torch.Tensor,
        structure: Optional[torch.Tensor] = None,
        dynamic_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.fuse_protein_inputs(x, structure)
        dynamic = self.encode_dynamic_tokens(dynamic_token_ids)
        if dynamic is not None:
            x = torch.cat([x, dynamic], dim=-1)
        if self.protein_film is not None:
            gamma, beta = self.protein_film(modality).chunk(2, dim=-1)
            x = x * (1.0 + gamma) + beta
        if self.protein_conditioning in {"concat", "concat_film"}:
            x = torch.cat([x, self.modality_embedding(modality)], dim=-1)
            return F.normalize(self.protein_encoder(x), dim=-1)
        if self.protein_conditioning == "separate_heads":
            out = None
            for modality_id, encoder in enumerate(self.protein_encoders):
                mask = modality == int(modality_id)
                if torch.any(mask):
                    encoded = encoder(x[mask])
                    if out is None:
                        out = encoded.new_zeros((x.shape[0], encoded.shape[-1]))
                    out[mask] = encoded
            if out is None:
                out = self.protein_encoders[0](x[:0]).new_zeros((x.shape[0], self.protein_encoders[0].net[-1].out_features))
            return F.normalize(out, dim=-1)
        return F.normalize(self.protein_encoder(x), dim=-1)

    def encode_profile(self, x: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
        if self.use_profile_source_embedding:
            x = x + self.profile_source_embedding(modality)
        return F.normalize(self.profile_encoder(x), dim=-1)

    def encode_protein_multi_expert(
        self,
        x: torch.Tensor,
        modality: torch.Tensor,
        structure: Optional[torch.Tensor] = None,
        dynamic_token_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h = self.encode_protein(x, modality, structure, dynamic_token_ids)
        if self.multi_expert_head is None:
            raise RuntimeError("encode_protein_multi_expert called with gene_head=single.")
        return self.multi_expert_head(h)


class GeneKGGraphSAGE(nn.Module):
    def __init__(
        self,
        base_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(int(base_dim))
        self.input_proj = nn.Linear(int(base_dim), int(hidden_dim))
        self.self_layers = nn.ModuleList([nn.Linear(int(hidden_dim), int(hidden_dim)) for _ in range(int(layers))])
        self.neigh_layers = nn.ModuleList([nn.Linear(int(hidden_dim), int(hidden_dim), bias=False) for _ in range(int(layers))])
        self.dropout = nn.Dropout(float(dropout))
        self.register_buffer("edge_index", edge_index.long(), persistent=False)
        self.register_buffer("edge_weight", edge_weight.float(), persistent=False)

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        h = self.dropout(F.relu(self.input_proj(self.input_norm(base))))
        if self.edge_index.numel() == 0:
            return h
        src = self.edge_index[0]
        dst = self.edge_index[1]
        weight = self.edge_weight.to(dtype=h.dtype, device=h.device)
        for self_layer, neigh_layer in zip(self.self_layers, self.neigh_layers):
            msg = h[src] * weight[:, None]
            agg = h.new_zeros(h.shape)
            agg.index_add_(0, dst, msg)
            degree = h.new_zeros((h.shape[0], 1))
            degree.index_add_(0, dst, weight[:, None])
            neigh = agg / degree.clamp_min(1e-6)
            h = F.relu(self_layer(h) + neigh_layer(neigh))
            h = self.dropout(h)
        return h


class KGGeneMoCoP(nn.Module):
    def __init__(
        self,
        base_model: GeneMoCoP,
        base_gene_embeddings: np.ndarray,
        edge_index: np.ndarray,
        edge_weight: np.ndarray,
        kg_hidden_dim: int,
        kg_layers: int,
        kg_dropout: float,
        kg_fusion: str,
        zero_init_adapter: bool,
    ):
        super().__init__()
        if kg_fusion not in {"gated_residual", "residual", "replace"}:
            raise ValueError(f"Unsupported kg_fusion: {kg_fusion}")
        self.base_model = base_model
        base = torch.from_numpy(np.asarray(base_gene_embeddings, dtype=np.float32))
        self.register_buffer("base_gene_embeddings", base, persistent=False)
        edge_index_t = torch.from_numpy(np.asarray(edge_index, dtype=np.int64))
        edge_weight_t = torch.from_numpy(np.asarray(edge_weight, dtype=np.float32))
        self.kg_encoder = GeneKGGraphSAGE(
            base_dim=base.shape[1],
            hidden_dim=int(kg_hidden_dim),
            layers=int(kg_layers),
            dropout=float(kg_dropout),
            edge_index=edge_index_t,
            edge_weight=edge_weight_t,
        )
        self.kg_adapter = nn.Linear(int(kg_hidden_dim), int(base.shape[1]))
        self.kg_gate = nn.Linear(int(base.shape[1]) * 2, int(base.shape[1]))
        self.kg_fusion = str(kg_fusion)
        if zero_init_adapter:
            nn.init.zeros_(self.kg_adapter.weight)
            nn.init.zeros_(self.kg_adapter.bias)

    def fused_gene_embeddings(self) -> torch.Tensor:
        base = self.base_gene_embeddings
        kg_hidden = self.kg_encoder(base)
        kg_delta = self.kg_adapter(kg_hidden)
        if self.kg_fusion == "replace":
            return kg_delta
        if self.kg_fusion == "residual":
            return base + kg_delta
        gate = torch.sigmoid(self.kg_gate(torch.cat([base, kg_delta], dim=-1)))
        return base + gate * kg_delta

    def encode_protein_entities(
        self,
        source_gene_index: torch.Tensor,
        modality: torch.Tensor,
        entity_features: Optional[torch.Tensor] = None,
        structure: Optional[torch.Tensor] = None,
        dynamic_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.fused_gene_embeddings()[source_gene_index.long()]
        if entity_features is not None:
            x = torch.cat([x, entity_features.to(dtype=x.dtype, device=x.device)], dim=-1)
        return self.base_model.encode_protein(x, modality, structure, dynamic_token_ids)

    def encode_protein(
        self,
        x: torch.Tensor,
        modality: torch.Tensor,
        structure: Optional[torch.Tensor] = None,
        dynamic_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.base_model.encode_protein(x, modality, structure, dynamic_token_ids)

    def encode_profile(self, x: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
        return self.base_model.encode_profile(x, modality)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            base_model = super().__getattr__("base_model")
            return getattr(base_model, name)


def masked_clip_loss(
    z_protein: torch.Tensor,
    z_profile: torch.Tensor,
    gene_code: torch.Tensor,
    modality: torch.Tensor,
    temperature: float,
    mask_crossmod: bool,
    mask_same_gene: bool = False,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logits = z_protein @ z_profile.T / float(temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    if (mask_crossmod or mask_same_gene) and logits.shape[0] > 1:
        same_gene = gene_code[:, None].eq(gene_code[None, :])
        if mask_same_gene:
            neutral = same_gene
        else:
            diff_mod = modality[:, None].ne(modality[None, :])
            neutral = same_gene & diff_mod
        eye = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        neutral = neutral & ~eye
        logits_row = logits.masked_fill(neutral, -1e4)
        logits_col = logits.T.masked_fill(neutral.T, -1e4)
    else:
        logits_row = logits
        logits_col = logits.T
    row_loss = F.cross_entropy(logits_row, labels, reduction="none")
    col_loss = F.cross_entropy(logits_col, labels, reduction="none")
    return 0.5 * (weighted_mean_loss(row_loss, sample_weight) + weighted_mean_loss(col_loss, sample_weight))


def masked_clip_loss_from_logits(
    logits: torch.Tensor,
    gene_code: torch.Tensor,
    modality: torch.Tensor,
    mask_crossmod: bool,
    mask_same_gene: bool = False,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    labels = torch.arange(logits.shape[0], device=logits.device)
    neutral = neutral_pair_mask(gene_code, modality, mask_crossmod, mask_same_gene)
    logits_row = logits.masked_fill(neutral, -1e4)
    logits_col = logits.T.masked_fill(neutral.T, -1e4)
    row_loss = F.cross_entropy(logits_row, labels, reduction="none")
    col_loss = F.cross_entropy(logits_col, labels, reduction="none")
    return 0.5 * (weighted_mean_loss(row_loss, sample_weight) + weighted_mean_loss(col_loss, sample_weight))


def multipositive_clip_loss_from_logits(
    logits: torch.Tensor,
    positive_code: torch.Tensor,
    gene_code: torch.Tensor,
    modality: torch.Tensor,
    mask_crossmod: bool,
    mask_same_gene: bool = False,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    n = int(logits.shape[0])
    if n <= 1:
        labels = torch.arange(n, device=logits.device)
        row_loss = F.cross_entropy(logits, labels, reduction="none")
        col_loss = F.cross_entropy(logits.T, labels, reduction="none")
        return 0.5 * (weighted_mean_loss(row_loss, sample_weight) + weighted_mean_loss(col_loss, sample_weight))
    pos = positive_code[:, None].eq(positive_code[None, :])
    neutral = neutral_pair_mask(gene_code, modality, mask_crossmod, mask_same_gene) & (~pos)
    logits_row = logits.masked_fill(neutral, -1e4)
    logits_col = logits.T.masked_fill(neutral.T, -1e4)
    row_targets = pos.to(logits.dtype) / pos.to(logits.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
    col_targets = row_targets.T / row_targets.T.sum(dim=1, keepdim=True).clamp_min(1e-8)
    row_loss = -(row_targets * F.log_softmax(logits_row, dim=1)).sum(dim=1)
    col_loss = -(col_targets * F.log_softmax(logits_col, dim=1)).sum(dim=1)
    return 0.5 * (weighted_mean_loss(row_loss, sample_weight) + weighted_mean_loss(col_loss, sample_weight))


def soft_clip_targets(
    entity_key_code: torch.Tensor,
    theta: torch.Tensor,
    eta: float,
    gamma: float,
) -> torch.Tensor:
    n = int(entity_key_code.shape[0])
    same_entity = entity_key_code[:, None].eq(entity_key_code[None, :]).to(theta.dtype)
    theta = F.normalize(theta.float(), dim=-1)
    sim = torch.clamp(theta @ theta.T, min=0.0).pow(float(gamma)).to(theta.dtype)
    target = same_entity + float(eta) * sim
    eye = torch.eye(n, dtype=torch.bool, device=target.device)
    target[eye] = torch.clamp(target[eye], min=1.0)
    return target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)


def target_entropy(target: torch.Tensor) -> torch.Tensor:
    return -(target * target.clamp_min(1e-8).log()).sum(dim=1).mean()


def soft_clip_target_diagnostics(
    entity_key_code: torch.Tensor,
    gene_code: torch.Tensor,
    modality: torch.Tensor,
    theta: torch.Tensor,
    eta: float,
    gamma: float,
) -> Dict[str, float]:
    n = int(entity_key_code.shape[0])
    if n == 0:
        return {
            "same_entity_positive_mass": 0.0,
            "same_gene_cross_modality_soft_mass": 0.0,
            "cross_modality_offdiag_mass": 0.0,
            "fraction_cross_modality_soft_mass_gt_same_entity_mass": 0.0,
        }
    same_entity = entity_key_code[:, None].eq(entity_key_code[None, :])
    same_gene = gene_code[:, None].eq(gene_code[None, :])
    diff_mod = modality[:, None].ne(modality[None, :])
    eye = torch.eye(n, dtype=torch.bool, device=theta.device)
    theta = F.normalize(theta.float(), dim=-1)
    sim = torch.clamp(theta @ theta.T, min=0.0).pow(float(gamma))
    hard = same_entity.to(sim.dtype)
    soft = float(eta) * sim
    target = hard + soft
    target[eye] = torch.clamp(target[eye], min=1.0)
    denom = target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    hard_mass = (hard / denom).masked_fill(~same_entity, 0.0).sum(dim=1)
    same_gene_cross_soft = (soft / denom).masked_fill(~(same_gene & diff_mod & ~same_entity), 0.0).sum(dim=1)
    cross_mod_soft = (soft / denom).masked_fill(~(diff_mod & ~eye), 0.0).sum(dim=1)
    return {
        "same_entity_positive_mass": float(hard_mass.mean().detach().cpu()),
        "same_gene_cross_modality_soft_mass": float(same_gene_cross_soft.mean().detach().cpu()),
        "cross_modality_offdiag_mass": float(cross_mod_soft.mean().detach().cpu()),
        "fraction_cross_modality_soft_mass_gt_same_entity_mass": float((cross_mod_soft > hard_mass).float().mean().detach().cpu()),
    }


def soft_clip_loss_from_logits(
    logits: torch.Tensor,
    entity_key_code: torch.Tensor,
    theta: torch.Tensor,
    eta: float,
    gamma: float,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    row_targets = soft_clip_targets(entity_key_code, theta, eta, gamma)
    col_targets = row_targets.T / row_targets.T.sum(dim=1, keepdim=True).clamp_min(1e-8)
    row_loss = -(row_targets.detach() * F.log_softmax(logits, dim=1)).sum(dim=1)
    col_loss = -(col_targets.detach() * F.log_softmax(logits.T, dim=1)).sum(dim=1)
    return 0.5 * (weighted_mean_loss(row_loss, sample_weight) + weighted_mean_loss(col_loss, sample_weight))


def weighted_multipositive_clip_loss(
    z_protein: torch.Tensor,
    z_profile: torch.Tensor,
    gene_code: torch.Tensor,
    modality: torch.Tensor,
    temperature: float,
    mask_crossmod: bool,
    mask_same_gene: bool,
    same_gene_same_modality_positive_weight: float,
    same_gene_crossmod_positive_weight: float,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logits = z_protein @ z_profile.T / float(temperature)
    n = logits.shape[0]
    if n <= 1:
        labels = torch.arange(n, device=logits.device)
        row_loss = F.cross_entropy(logits, labels, reduction="none")
        col_loss = F.cross_entropy(logits.T, labels, reduction="none")
        return 0.5 * (weighted_mean_loss(row_loss, sample_weight) + weighted_mean_loss(col_loss, sample_weight))

    eye = torch.eye(n, dtype=torch.bool, device=logits.device)
    same_gene = gene_code[:, None].eq(gene_code[None, :])
    same_mod = modality[:, None].eq(modality[None, :])
    diff_mod = ~same_mod

    pos = logits.new_zeros((n, n))
    pos[eye] = 1.0
    same_mod_weight = float(same_gene_same_modality_positive_weight)
    crossmod_weight = float(same_gene_crossmod_positive_weight)
    if same_mod_weight > 0:
        pos = pos + (same_gene & same_mod & ~eye).to(logits.dtype) * same_mod_weight
    if crossmod_weight > 0:
        pos = pos + (same_gene & diff_mod & ~eye).to(logits.dtype) * crossmod_weight

    neutral = torch.zeros_like(eye)
    if mask_same_gene:
        neutral = same_gene & (pos <= 0)
    elif mask_crossmod:
        neutral = same_gene & diff_mod & (pos <= 0)
    neutral = neutral & ~eye

    logits_row = logits.masked_fill(neutral, -1e4)
    logits_col = logits.T.masked_fill(neutral.T, -1e4)
    row_targets = pos / pos.sum(dim=1, keepdim=True).clamp_min(1e-8)
    col_targets = pos.T / pos.T.sum(dim=1, keepdim=True).clamp_min(1e-8)
    row_loss = -(row_targets * F.log_softmax(logits_row, dim=1)).sum(dim=1)
    col_loss = -(col_targets * F.log_softmax(logits_col, dim=1)).sum(dim=1)
    row_loss = weighted_mean_loss(row_loss, sample_weight)
    col_loss = weighted_mean_loss(col_loss, sample_weight)
    return 0.5 * (row_loss + col_loss)


def neutral_pair_mask(
    gene_code: torch.Tensor,
    modality: torch.Tensor,
    mask_crossmod: bool,
    mask_same_gene: bool,
) -> torch.Tensor:
    n = int(gene_code.shape[0])
    same_gene = gene_code[:, None].eq(gene_code[None, :])
    if mask_same_gene:
        neutral = same_gene
    elif mask_crossmod:
        diff_mod = modality[:, None].ne(modality[None, :])
        neutral = same_gene & diff_mod
    else:
        neutral = torch.zeros((n, n), dtype=torch.bool, device=gene_code.device)
    eye = torch.eye(n, dtype=torch.bool, device=gene_code.device)
    return neutral & ~eye


def sigmoid_clip_loss(
    z_protein: torch.Tensor,
    z_profile: torch.Tensor,
    gene_code: torch.Tensor,
    modality: torch.Tensor,
    temperature: float,
    mask_crossmod: bool,
    mask_same_gene: bool,
    sample_weight: Optional[torch.Tensor] = None,
    neg_weight: float = 1.0,
) -> torch.Tensor:
    logits = z_protein @ z_profile.T / float(temperature)
    n = int(logits.shape[0])
    eye = torch.eye(n, dtype=torch.bool, device=logits.device)
    neutral = neutral_pair_mask(gene_code, modality, mask_crossmod, mask_same_gene)
    valid_neg = (~eye) & (~neutral)
    pos_loss = F.softplus(-logits.diag())
    neg_loss = F.softplus(logits[valid_neg])
    pos_term = weighted_mean_loss(pos_loss, sample_weight)
    neg_term = neg_loss.mean() if neg_loss.numel() else logits.sum() * 0.0
    return pos_term + float(neg_weight) * neg_term


def sigmoid_clip_loss_from_logits(
    logits: torch.Tensor,
    gene_code: torch.Tensor,
    modality: torch.Tensor,
    mask_crossmod: bool,
    mask_same_gene: bool,
    sample_weight: Optional[torch.Tensor] = None,
    neg_weight: float = 1.0,
) -> torch.Tensor:
    n = int(logits.shape[0])
    eye = torch.eye(n, dtype=torch.bool, device=logits.device)
    neutral = neutral_pair_mask(gene_code, modality, mask_crossmod, mask_same_gene)
    valid_neg = (~eye) & (~neutral)
    pos_loss = F.softplus(-logits.diag())
    neg_loss = F.softplus(logits[valid_neg])
    pos_term = weighted_mean_loss(pos_loss, sample_weight)
    neg_term = neg_loss.mean() if neg_loss.numel() else logits.sum() * 0.0
    return pos_term + float(neg_weight) * neg_term


def _directional_rank_loss(
    logits: torch.Tensor,
    neutral: torch.Tensor,
    margin: float,
    hard_negatives: int,
    sample_weight: Optional[torch.Tensor],
) -> torch.Tensor:
    n = int(logits.shape[0])
    if n <= 1:
        return logits.sum() * 0.0
    eye = torch.eye(n, dtype=torch.bool, device=logits.device)
    valid_neg = (~eye) & (~neutral)
    pos = logits.diag()[:, None]
    pair_loss = F.softplus(logits - pos + float(margin))
    pair_loss = pair_loss.masked_fill(~valid_neg, 0.0)
    if int(hard_negatives) > 0 and int(hard_negatives) < n - 1:
        masked_logits = logits.masked_fill(~valid_neg, -1e4)
        k = min(int(hard_negatives), max(1, n - 1))
        hard_idx = torch.topk(masked_logits, k=k, dim=1).indices
        hard_mask = torch.zeros_like(valid_neg)
        hard_mask.scatter_(1, hard_idx, True)
        valid_neg = valid_neg & hard_mask
        pair_loss = pair_loss.masked_fill(~valid_neg, 0.0)
    denom = valid_neg.sum(dim=1).clamp_min(1).to(pair_loss.dtype)
    row_loss = pair_loss.sum(dim=1) / denom
    return weighted_mean_loss(row_loss, sample_weight)


def pairwise_rank_clip_loss(
    z_protein: torch.Tensor,
    z_profile: torch.Tensor,
    gene_code: torch.Tensor,
    modality: torch.Tensor,
    temperature: float,
    mask_crossmod: bool,
    mask_same_gene: bool,
    margin: float,
    hard_negatives: int,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logits = z_protein @ z_profile.T / float(temperature)
    neutral = neutral_pair_mask(gene_code, modality, mask_crossmod, mask_same_gene)
    row_loss = _directional_rank_loss(logits, neutral, margin, hard_negatives, sample_weight)
    col_loss = _directional_rank_loss(logits.T, neutral.T, margin, hard_negatives, sample_weight)
    return 0.5 * (row_loss + col_loss)


def pairwise_rank_loss_from_logits(
    logits: torch.Tensor,
    gene_code: torch.Tensor,
    modality: torch.Tensor,
    mask_crossmod: bool,
    mask_same_gene: bool,
    margin: float,
    hard_negatives: int,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    neutral = neutral_pair_mask(gene_code, modality, mask_crossmod, mask_same_gene)
    row_loss = _directional_rank_loss(logits, neutral, margin, hard_negatives, sample_weight)
    col_loss = _directional_rank_loss(logits.T, neutral.T, margin, hard_negatives, sample_weight)
    return 0.5 * (row_loss + col_loss)


def profile_geometry_loss(profile_input: torch.Tensor, z_profile: torch.Tensor) -> torch.Tensor:
    n = int(z_profile.shape[0])
    if n <= 1:
        return z_profile.sum() * 0.0
    target = F.normalize(profile_input.float(), dim=-1) @ F.normalize(profile_input.float(), dim=-1).T
    pred = z_profile.float() @ z_profile.float().T
    eye = torch.eye(n, dtype=torch.bool, device=z_profile.device)
    return F.mse_loss(pred[~eye], target.detach()[~eye])


def gene_profile_geometry_loss(z_protein: torch.Tensor, profile_input: torch.Tensor) -> torch.Tensor:
    n = int(z_protein.shape[0])
    if n <= 1:
        return z_protein.sum() * 0.0
    target = F.normalize(profile_input.float(), dim=-1) @ F.normalize(profile_input.float(), dim=-1).T
    pred = z_protein.float() @ z_protein.float().T
    eye = torch.eye(n, dtype=torch.bool, device=z_protein.device)
    return F.mse_loss(pred[~eye], target.detach()[~eye])


def profile_neighbor_distill_loss(
    z_protein: torch.Tensor,
    z_profile: torch.Tensor,
    profile_input: torch.Tensor,
    temperature: float,
    target_temperature: float,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    n = int(z_protein.shape[0])
    if n <= 1:
        return z_protein.sum() * 0.0
    logits = z_protein.float() @ z_profile.float().T / float(temperature)
    with torch.no_grad():
        x = F.normalize(profile_input.float(), dim=-1)
        target_logits = (x @ x.T) / max(float(target_temperature), 1e-6)
        row_targets = F.softmax(target_logits, dim=1)
        col_targets = F.softmax(target_logits.T, dim=1)
    row_loss = -(row_targets * F.log_softmax(logits, dim=1)).sum(dim=1)
    col_loss = -(col_targets * F.log_softmax(logits.T, dim=1)).sum(dim=1)
    return 0.5 * (weighted_mean_loss(row_loss, sample_weight) + weighted_mean_loss(col_loss, sample_weight))


def profile_adversary_loss(
    model: GeneMoCoP,
    z_profile: torch.Tensor,
    feature_rows: np.ndarray,
    feature_label_codes: Dict[str, Optional[np.ndarray]],
    device: torch.device,
    well_weight: float,
    plate_weight: float,
    source_weight: float,
) -> torch.Tensor:
    total = z_profile.sum() * 0.0
    specs = [
        ("well", model.profile_well_adversary, float(well_weight)),
        ("plate", model.profile_plate_adversary, float(plate_weight)),
        ("source", model.profile_source_adversary, float(source_weight)),
    ]
    for name, head, weight in specs:
        labels_all = feature_label_codes.get(name)
        if head is None or labels_all is None or weight <= 0:
            continue
        labels_np = labels_all[feature_rows]
        valid = labels_np >= 0
        if not np.any(valid):
            continue
        labels = torch.from_numpy(labels_np[valid].astype(np.int64)).long().to(device)
        logits = head(gradient_reverse(z_profile[torch.from_numpy(valid).to(device=device)], 1.0))
        total = total + weight * F.cross_entropy(logits.float(), labels)
    return total


def build_gene_modality_entity_index(
    entity_rows: np.ndarray,
    gene_codes_all: np.ndarray,
    entity_mods_all: np.ndarray,
) -> Dict[Tuple[int, int], np.ndarray]:
    buckets: Dict[Tuple[int, int], List[int]] = {}
    for row in np.asarray(entity_rows, dtype=np.int64).tolist():
        key = (int(gene_codes_all[row]), int(entity_mods_all[row]))
        buckets.setdefault(key, []).append(int(row))
    return {key: np.asarray(vals, dtype=np.int64) for key, vals in buckets.items()}


def augment_with_same_gene_siblings(
    batch_entities: np.ndarray,
    gene_codes_all: np.ndarray,
    entity_mods_all: np.ndarray,
    gene_modality_index: Dict[Tuple[int, int], np.ndarray],
    rng: np.random.Generator,
    replacement_prob: float,
) -> np.ndarray:
    prob = float(replacement_prob)
    if prob <= 0 or len(batch_entities) <= 1:
        return batch_entities
    out = np.asarray(batch_entities, dtype=np.int64).copy()
    used = {int(x) for x in out.tolist()}
    protected: set[int] = set()
    order = np.arange(len(out), dtype=np.int64)
    rng.shuffle(order)
    for pos in order.tolist():
        if int(pos) in protected:
            continue
        entity = int(out[int(pos)])
        if rng.random() >= prob:
            continue
        key = (int(gene_codes_all[entity]), int(entity_mods_all[entity]))
        candidates = gene_modality_index.get(key)
        if candidates is None or len(candidates) <= 1:
            continue
        available = [int(x) for x in candidates.tolist() if int(x) != entity and int(x) not in used]
        if not available:
            continue
        replaceable = [int(i) for i in range(len(out)) if int(i) != int(pos) and int(i) not in protected]
        if not replaceable:
            break
        replace_pos = int(rng.choice(np.asarray(replaceable, dtype=np.int64)))
        old = int(out[replace_pos])
        new = int(rng.choice(np.asarray(available, dtype=np.int64)))
        out[replace_pos] = new
        used.discard(old)
        used.add(new)
        protected.add(int(pos))
        protected.add(replace_pos)
    return out


def update_ema(model: nn.Module, ema_model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
        for ema_b, b in zip(ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)


def protein_tensor(
    entities: pd.DataFrame,
    protein_embeddings: np.ndarray,
    entity_rows: np.ndarray,
    device: torch.device,
    entity_features: Optional[np.ndarray] = None,
) -> torch.Tensor:
    idx = entities.iloc[entity_rows]["source_gene_index"].to_numpy(dtype=np.int64)
    x = protein_embeddings[idx]
    if entity_features is not None:
        x = np.concatenate([x, entity_features[entity_rows]], axis=1)
    return tensor_from_numpy(x.astype(np.float32), device)


def structure_tensor(
    entities: pd.DataFrame,
    structure_embeddings: Optional[np.ndarray],
    entity_rows: np.ndarray,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if structure_embeddings is None:
        return None
    idx = entities.iloc[entity_rows]["source_gene_index"].to_numpy(dtype=np.int64)
    return tensor_from_numpy(structure_embeddings[idx], device)


def dynamic_token_tensor(dynamic_token_ids: Optional[np.ndarray], entity_rows: np.ndarray, device: torch.device) -> Optional[torch.Tensor]:
    if dynamic_token_ids is None:
        return None
    return torch.from_numpy(dynamic_token_ids[entity_rows].astype(np.int64)).long().to(device)


def entity_feature_tensor(entity_features: Optional[np.ndarray], entity_rows: np.ndarray, device: torch.device) -> Optional[torch.Tensor]:
    if entity_features is None:
        return None
    return tensor_from_numpy(entity_features[entity_rows].astype(np.float32), device)


def source_gene_tensor(entities: pd.DataFrame, entity_rows: np.ndarray, device: torch.device) -> torch.Tensor:
    idx = entities.iloc[entity_rows]["source_gene_index"].to_numpy(dtype=np.int64)
    return torch.from_numpy(idx.astype(np.int64)).long().to(device)


def encode_protein_entities(
    model: nn.Module,
    entities: pd.DataFrame,
    protein_embeddings: np.ndarray,
    entity_features: Optional[np.ndarray],
    structure_embeddings: Optional[np.ndarray],
    dynamic_token_ids: Optional[np.ndarray],
    entity_rows: np.ndarray,
    modality: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if hasattr(model, "encode_protein_entities"):
        return model.encode_protein_entities(
            source_gene_tensor(entities, entity_rows, device),
            modality,
            entity_feature_tensor(entity_features, entity_rows, device),
            structure_tensor(entities, structure_embeddings, entity_rows, device),
            dynamic_token_tensor(dynamic_token_ids, entity_rows, device),
        )
    return model.encode_protein(
        protein_tensor(entities, protein_embeddings, entity_rows, device, entity_features),
        modality,
        structure_tensor(entities, structure_embeddings, entity_rows, device),
        dynamic_token_tensor(dynamic_token_ids, entity_rows, device),
    )


def encode_gene_branch_entities(
    model: nn.Module,
    entities: pd.DataFrame,
    protein_embeddings: np.ndarray,
    entity_features: Optional[np.ndarray],
    structure_embeddings: Optional[np.ndarray],
    dynamic_token_ids: Optional[np.ndarray],
    entity_rows: np.ndarray,
    modality: torch.Tensor,
    device: torch.device,
) -> Any:
    if getattr(model, "gene_head", "single") == "multi_expert" and hasattr(model, "encode_protein_multi_expert"):
        return model.encode_protein_multi_expert(
            protein_tensor(entities, protein_embeddings, entity_rows, device, entity_features),
            modality,
            structure_tensor(entities, structure_embeddings, entity_rows, device),
            dynamic_token_tensor(dynamic_token_ids, entity_rows, device),
        )
    return encode_protein_entities(
        model,
        entities,
        protein_embeddings,
        entity_features,
        structure_embeddings,
        dynamic_token_ids,
        entity_rows,
        modality,
        device,
    )


def profile_tensor(features: np.ndarray, feature_rows: np.ndarray, modality: np.ndarray, norm: Dict[str, Any], device: torch.device) -> torch.Tensor:
    return tensor_from_numpy(transform_feature_rows(features, feature_rows, modality, norm), device)


def entity_mean_feature_rows(entity_to_reps: Sequence[np.ndarray], entity_rows: np.ndarray) -> List[np.ndarray]:
    return [entity_to_reps[int(i)] for i in entity_rows.astype(int).tolist()]


def encode_entities(
    model: GeneMoCoP,
    entities: pd.DataFrame,
    features: np.ndarray,
    protein_embeddings: np.ndarray,
    structure_embeddings: Optional[np.ndarray],
    entity_features: Optional[np.ndarray],
    dynamic_token_ids: Optional[np.ndarray],
    entity_rows: np.ndarray,
    entity_to_reps: Sequence[np.ndarray],
    entity_mods: np.ndarray,
    norm: Dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    z_p_chunks: List[Any] = []
    z_prof_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(entity_rows), batch_size):
            batch = entity_rows[start : start + batch_size]
            mod_np = entity_mods[batch]
            mod = torch.from_numpy(mod_np.astype(np.int64)).long().to(device)
            zp = encode_gene_branch_entities(
                model,
                entities,
                protein_embeddings,
                entity_features,
                structure_embeddings,
                dynamic_token_ids,
                batch,
                mod,
                device,
            )
            reps = entity_mean_feature_rows(entity_to_reps, batch)
            # For stable evaluation, encode the mean normalized replicate profile for each entity.
            prof_arr = []
            for rep_rows, modality_id in zip(reps, mod_np.tolist()):
                rep_mod = np.full(len(rep_rows), int(modality_id), dtype=np.int64)
                prof_arr.append(transform_feature_rows(features, rep_rows, rep_mod, norm).mean(axis=0))
            px = tensor_from_numpy(np.vstack(prof_arr).astype(np.float32), device)
            zprof = model.encode_profile(px, mod)
            if is_multi_expert_output(zp):
                z_p_chunks.append(
                    {
                        "experts": zp["experts"].detach().cpu().numpy().astype(np.float32),
                        "gate_probs": zp["gate_probs"].detach().cpu().numpy().astype(np.float32),
                        "temperatures": zp["temperatures"].detach().cpu().numpy().astype(np.float32),
                    }
                )
            else:
                z_p_chunks.append(zp.detach().cpu().numpy().astype(np.float32))
            z_prof_chunks.append(zprof.detach().cpu().numpy().astype(np.float32))
    if not z_p_chunks:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
    if isinstance(z_p_chunks[0], dict):
        z_gene = {
            "experts": np.vstack([x["experts"] for x in z_p_chunks]).astype(np.float32),
            "gate_probs": np.vstack([x["gate_probs"] for x in z_p_chunks]).astype(np.float32),
            "temperatures": z_p_chunks[0]["temperatures"].astype(np.float32),
        }
    else:
        z_gene = np.vstack(z_p_chunks)
    return z_gene, np.vstack(z_prof_chunks)


def build_replicate_eval_gallery(
    entity_rows: np.ndarray,
    entity_to_reps: Sequence[np.ndarray],
    entity_mods_all: np.ndarray,
    entity_key_codes_all: np.ndarray,
    gene_codes_all: np.ndarray,
    max_replicates_per_entity: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed) + 99173)
    feature_rows: List[int] = []
    owner_entities: List[int] = []
    cap = int(max_replicates_per_entity)
    for entity_idx in np.asarray(entity_rows, dtype=np.int64).tolist():
        reps = np.asarray(entity_to_reps[int(entity_idx)], dtype=np.int64)
        if reps.size == 0:
            continue
        if cap > 0 and reps.size > cap:
            reps = np.asarray(rng.choice(reps, size=cap, replace=False), dtype=np.int64)
        feature_rows.extend([int(x) for x in reps.tolist()])
        owner_entities.extend([int(entity_idx)] * int(reps.size))
    owner = np.asarray(owner_entities, dtype=np.int64)
    feat = np.asarray(feature_rows, dtype=np.int64)
    return {
        "feature_rows": feat,
        "owner_entities": owner,
        "entity_key_codes": entity_key_codes_all[owner].astype(np.int64) if owner.size else np.zeros(0, dtype=np.int64),
        "gene_codes": gene_codes_all[owner].astype(np.int64) if owner.size else np.zeros(0, dtype=np.int64),
        "modalities": entity_mods_all[owner].astype(np.int64) if owner.size else np.zeros(0, dtype=np.int64),
    }


def encode_profile_replicates(
    model: GeneMoCoP,
    features: np.ndarray,
    feature_rows: np.ndarray,
    modality: np.ndarray,
    norm: Dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(feature_rows), batch_size):
            rows = feature_rows[start : start + batch_size]
            mod_np = modality[start : start + batch_size]
            mod = torch.from_numpy(mod_np.astype(np.int64)).long().to(device)
            px = profile_tensor(features, rows, mod_np, norm, device)
            zprof = model.encode_profile(px, mod)
            chunks.append(zprof.detach().cpu().numpy().astype(np.float32))
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(chunks).astype(np.float32)


def _numpy_moe_score(gene: Dict[str, np.ndarray], profile: np.ndarray) -> np.ndarray:
    experts = gene["experts"].astype(np.float32)
    gate = np.clip(gene["gate_probs"].astype(np.float32), 1e-8, 1.0)
    tau = np.clip(gene["temperatures"].astype(np.float32), 0.03, 0.30)
    dots = np.einsum("bed,nd->ben", experts, profile.astype(np.float32), optimize=True)
    logits = dots / tau[None, :, None] + np.log(gate)[:, :, None]
    maxv = np.max(logits, axis=1, keepdims=True)
    return (np.squeeze(maxv, axis=1) + np.log(np.sum(np.exp(logits - maxv), axis=1))).astype(np.float32)


def retrieval_score_matrix(query: Any, gallery: Any) -> np.ndarray:
    if isinstance(query, dict) and not isinstance(gallery, dict):
        return _numpy_moe_score(query, gallery)
    if not isinstance(query, dict) and isinstance(gallery, dict):
        return _numpy_moe_score(gallery, query).T
    if isinstance(query, dict) and isinstance(gallery, dict):
        raise ValueError("MoE-vs-MoE retrieval is not used in gene/profile evaluation.")
    return query.astype(np.float32) @ gallery.astype(np.float32).T


def slice_encoded(encoded: Any, idx: np.ndarray) -> Any:
    if isinstance(encoded, dict):
        return {
            "experts": encoded["experts"][idx],
            "gate_probs": encoded["gate_probs"][idx],
            "temperatures": encoded["temperatures"],
        }
    return encoded[idx]


def encoded_len(encoded: Any) -> int:
    if isinstance(encoded, dict):
        return int(encoded["experts"].shape[0])
    return int(len(encoded))


def encoded_gate_summary(encoded: Any) -> Dict[str, Any]:
    if not isinstance(encoded, dict) or encoded_len(encoded) == 0:
        return {"enabled": False}
    gate = encoded["gate_probs"].astype(np.float64)
    usage = gate.mean(axis=0)
    entropy = -np.sum(gate * np.log(np.clip(gate, 1e-8, 1.0)), axis=1)
    effective = float(1.0 / np.sum(np.square(usage))) if usage.size else 0.0
    dominant = float(np.max(usage)) if usage.size else 0.0
    return {
        "enabled": True,
        "num_experts": int(gate.shape[1]),
        "mean_gate_probs": [float(x) for x in usage.tolist()],
        "gate_entropy_mean": float(np.mean(entropy)),
        "gate_entropy_std": float(np.std(entropy)),
        "effective_num_experts": effective,
        "dominant_expert_usage": dominant,
        "gate_collapse": bool(dominant > 0.80 or effective < 1.5),
    }


def full_gallery_metrics(query: Any, gallery: Any, positive_code: Optional[np.ndarray] = None) -> Dict[str, Any]:
    if encoded_len(query) == 0:
        return metrics_from_ranks([])
    n = encoded_len(query)
    if positive_code is None:
        positive_code = np.arange(n, dtype=np.int64)
    positive_code = np.asarray(positive_code, dtype=np.int64)
    scores = retrieval_score_matrix(query, gallery)
    ranks = []
    gaps = []
    eps = 1e-8
    for i in range(scores.shape[0]):
        row = scores[i]
        pos_mask = positive_code == positive_code[i]
        if not np.any(pos_mask):
            continue
        p = float(np.max(row[pos_mask]))
        ranks.append(int(1 + np.sum(row[~pos_mask] > p + eps)))
        if np.any(~pos_mask):
            gaps.append(float(p - np.max(row[~pos_mask])))
    return metrics_from_ranks(ranks, gaps)


def full_gallery_metrics_general(
    query: Any,
    gallery: Any,
    query_positive_code: np.ndarray,
    gallery_positive_code: np.ndarray,
) -> Dict[str, Any]:
    if encoded_len(query) == 0 or encoded_len(gallery) == 0:
        return metrics_from_ranks([])
    query_positive_code = np.asarray(query_positive_code, dtype=np.int64)
    gallery_positive_code = np.asarray(gallery_positive_code, dtype=np.int64)
    scores = retrieval_score_matrix(query, gallery)
    ranks = []
    gaps = []
    eps = 1e-8
    for i in range(scores.shape[0]):
        row = scores[i]
        pos_mask = gallery_positive_code == query_positive_code[i]
        if not np.any(pos_mask):
            continue
        p = float(np.max(row[pos_mask]))
        ranks.append(int(1 + np.sum(row[~pos_mask] > p + eps)))
        if np.any(~pos_mask):
            gaps.append(float(p - np.max(row[~pos_mask])))
    return metrics_from_ranks(ranks, gaps)


def sampled_top10_bidirectional(metrics: Dict[str, Any], ratio: int = 100) -> Dict[str, Any]:
    key = f"1:{int(ratio)}"
    sampled = metrics.get("mocop_protocol_sampled", {})
    p2 = sampled.get("protein_to_profile", {}).get(key, {}).get("Top10_accuracy_mean")
    q2 = sampled.get("profile_to_protein", {}).get(key, {}).get("Top10_accuracy_mean")
    vals = [float(x) for x in [p2, q2] if x is not None]
    return {
        "ratio": key,
        "gene_to_profile": p2,
        "profile_to_gene": q2,
        "bidirectional_mean": float(np.mean(vals)) if vals else None,
    }


def sampled_one_direction(
    query: Any,
    gallery: Any,
    positive_code: np.ndarray,
    gene_code: np.ndarray,
    modality: np.ndarray,
    ratio: int,
    repeats: int,
    seed: int,
    mask_crossmod: bool,
    mask_same_gene: bool = False,
) -> Dict[str, Any]:
    if encoded_len(query) <= 1:
        return {"num_repeats": int(repeats), "repeats": []}
    all_idx = np.arange(encoded_len(query), dtype=np.int64)
    positive_code = np.asarray(positive_code, dtype=np.int64)
    repeat_rows = []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat * 100003 + int(ratio) * 17)
        ranks = []
        n_neg_seen = []
        n_pos_seen = []
        for i in range(encoded_len(query)):
            pos_pool = all_idx[positive_code == positive_code[i]]
            if pos_pool.size == 0:
                continue
            neg_pool = all_idx[positive_code != positive_code[i]]
            if mask_crossmod or mask_same_gene:
                if mask_same_gene:
                    neutral = gene_code[neg_pool] == gene_code[i]
                else:
                    neutral = (gene_code[neg_pool] == gene_code[i]) & (modality[neg_pool] != modality[i])
                neg_pool = neg_pool[~neutral]
            n_neg = min(int(ratio), len(neg_pool))
            if n_neg <= 0:
                continue
            neg = rng.choice(neg_pool, size=n_neg, replace=False)
            cand = np.concatenate([pos_pool, neg])
            scores = retrieval_score_matrix(slice_encoded(query, np.asarray([i], dtype=np.int64)), slice_encoded(gallery, cand))[0]
            pos_scores = scores[: pos_pool.size]
            neg_scores = scores[pos_pool.size :]
            best_pos = float(np.max(pos_scores))
            ranks.append(int(1 + np.sum(neg_scores > best_pos)))
            n_neg_seen.append(n_neg)
            n_pos_seen.append(int(pos_pool.size))
        arr = np.asarray(ranks, dtype=np.float64)
        repeat_rows.append(
            {
                "num_queries": int(len(arr)),
                "negative_ratio": int(ratio),
                "candidate_count_mean": float(np.mean(np.asarray(n_neg_seen) + np.asarray(n_pos_seen))) if n_neg_seen else 0.0,
                "positive_count_mean": float(np.mean(n_pos_seen)) if n_pos_seen else 0.0,
                "Top1_accuracy": float(np.mean(arr <= 1)) if arr.size else None,
                "Top5_accuracy": float(np.mean(arr <= 5)) if arr.size else None,
                "Top10_accuracy": float(np.mean(arr <= 10)) if arr.size else None,
                "MRR": float(np.mean(1.0 / arr)) if arr.size else None,
                "median_rank": float(np.median(arr)) if arr.size else None,
            }
        )
    summary: Dict[str, Any] = {"num_repeats": int(repeats), "repeats": repeat_rows}
    for key in ["Top1_accuracy", "Top5_accuracy", "Top10_accuracy", "MRR", "median_rank"]:
        vals = np.asarray([m[key] for m in repeat_rows if m[key] is not None], dtype=np.float64)
        summary[f"{key}_mean"] = float(vals.mean()) if vals.size else None
        summary[f"{key}_std"] = float(vals.std(ddof=0)) if vals.size else None
    return summary


def sampled_one_direction_general(
    query: Any,
    gallery: Any,
    query_positive_code: np.ndarray,
    gallery_positive_code: np.ndarray,
    query_gene_code: np.ndarray,
    gallery_gene_code: np.ndarray,
    query_modality: np.ndarray,
    gallery_modality: np.ndarray,
    ratio: int,
    repeats: int,
    seed: int,
    mask_crossmod: bool,
    mask_same_gene: bool = False,
) -> Dict[str, Any]:
    if encoded_len(query) == 0 or encoded_len(gallery) == 0:
        return {"num_repeats": int(repeats), "repeats": []}
    gallery_idx = np.arange(encoded_len(gallery), dtype=np.int64)
    query_positive_code = np.asarray(query_positive_code, dtype=np.int64)
    gallery_positive_code = np.asarray(gallery_positive_code, dtype=np.int64)
    query_gene_code = np.asarray(query_gene_code, dtype=np.int64)
    gallery_gene_code = np.asarray(gallery_gene_code, dtype=np.int64)
    query_modality = np.asarray(query_modality, dtype=np.int64)
    gallery_modality = np.asarray(gallery_modality, dtype=np.int64)
    repeat_rows = []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat * 100003 + int(ratio) * 17)
        ranks = []
        n_neg_seen = []
        n_pos_seen = []
        for i in range(encoded_len(query)):
            pos_pool = gallery_idx[gallery_positive_code == query_positive_code[i]]
            if pos_pool.size == 0:
                continue
            neg_pool = gallery_idx[gallery_positive_code != query_positive_code[i]]
            if mask_crossmod or mask_same_gene:
                if mask_same_gene:
                    neutral = gallery_gene_code[neg_pool] == query_gene_code[i]
                else:
                    neutral = (gallery_gene_code[neg_pool] == query_gene_code[i]) & (
                        gallery_modality[neg_pool] != query_modality[i]
                    )
                neg_pool = neg_pool[~neutral]
            n_neg = min(int(ratio), len(neg_pool))
            if n_neg <= 0:
                continue
            neg = rng.choice(neg_pool, size=n_neg, replace=False)
            cand = np.concatenate([pos_pool, neg])
            scores = retrieval_score_matrix(slice_encoded(query, np.asarray([i], dtype=np.int64)), slice_encoded(gallery, cand))[0]
            pos_scores = scores[: pos_pool.size]
            neg_scores = scores[pos_pool.size :]
            best_pos = float(np.max(pos_scores))
            ranks.append(int(1 + np.sum(neg_scores > best_pos)))
            n_neg_seen.append(n_neg)
            n_pos_seen.append(int(pos_pool.size))
        arr = np.asarray(ranks, dtype=np.float64)
        repeat_rows.append(
            {
                "num_queries": int(len(arr)),
                "negative_ratio": int(ratio),
                "candidate_count_mean": float(np.mean(np.asarray(n_neg_seen) + np.asarray(n_pos_seen))) if n_neg_seen else 0.0,
                "positive_count_mean": float(np.mean(n_pos_seen)) if n_pos_seen else 0.0,
                "Top1_accuracy": float(np.mean(arr <= 1)) if arr.size else None,
                "Top5_accuracy": float(np.mean(arr <= 5)) if arr.size else None,
                "Top10_accuracy": float(np.mean(arr <= 10)) if arr.size else None,
                "MRR": float(np.mean(1.0 / arr)) if arr.size else None,
                "median_rank": float(np.median(arr)) if arr.size else None,
            }
        )
    summary: Dict[str, Any] = {"num_repeats": int(repeats), "repeats": repeat_rows}
    for key in ["Top1_accuracy", "Top5_accuracy", "Top10_accuracy", "MRR", "median_rank"]:
        vals = np.asarray([m[key] for m in repeat_rows if m[key] is not None], dtype=np.float64)
        summary[f"{key}_mean"] = float(vals.mean()) if vals.size else None
        summary[f"{key}_std"] = float(vals.std(ddof=0)) if vals.size else None
    return summary


def sampled_metrics_general(
    z_gene: Any,
    z_profile: Any,
    gene_positive_code: np.ndarray,
    profile_positive_code: np.ndarray,
    gene_code_for_gene: np.ndarray,
    gene_code_for_profile: np.ndarray,
    modality_for_gene: np.ndarray,
    modality_for_profile: np.ndarray,
    ratios: Sequence[int],
    repeats: int,
    seed: int,
    mask_crossmod: bool,
    mask_same_gene: bool = False,
    positive_label_name: str = "entity_key",
) -> Dict[str, Any]:
    out = {"protein_to_profile": {}, "profile_to_protein": {}, "positive_label": str(positive_label_name)}
    for ratio in ratios:
        out["protein_to_profile"][f"1:{ratio}"] = sampled_one_direction_general(
            z_gene,
            z_profile,
            gene_positive_code,
            profile_positive_code,
            gene_code_for_gene,
            gene_code_for_profile,
            modality_for_gene,
            modality_for_profile,
            ratio,
            repeats,
            seed,
            mask_crossmod,
            mask_same_gene,
        )
        out["profile_to_protein"][f"1:{ratio}"] = sampled_one_direction_general(
            z_profile,
            z_gene,
            profile_positive_code,
            gene_positive_code,
            gene_code_for_profile,
            gene_code_for_gene,
            modality_for_profile,
            modality_for_gene,
            ratio,
            repeats,
            seed,
            mask_crossmod,
            mask_same_gene,
        )
    return out


def sampled_metrics(
    z_protein: Any,
    z_profile: Any,
    positive_code: np.ndarray,
    gene_code: np.ndarray,
    modality: np.ndarray,
    ratios: Sequence[int],
    repeats: int,
    seed: int,
    mask_crossmod: bool,
    mask_same_gene: bool = False,
    positive_label_name: str = "entity_key",
) -> Dict[str, Any]:
    out = {"protein_to_profile": {}, "profile_to_protein": {}, "positive_label": str(positive_label_name)}
    for ratio in ratios:
        out["protein_to_profile"][f"1:{ratio}"] = sampled_one_direction(
            z_protein, z_profile, positive_code, gene_code, modality, ratio, repeats, seed, mask_crossmod, mask_same_gene
        )
        out["profile_to_protein"][f"1:{ratio}"] = sampled_one_direction(
            z_profile, z_protein, positive_code, gene_code, modality, ratio, repeats, seed, mask_crossmod, mask_same_gene
        )
    return out


def evaluate_entity_rows(
    model: GeneMoCoP,
    entities: pd.DataFrame,
    features: np.ndarray,
    protein_embeddings: np.ndarray,
    structure_embeddings: Optional[np.ndarray],
    entity_features: Optional[np.ndarray],
    dynamic_token_ids: Optional[np.ndarray],
    entity_rows: np.ndarray,
    entity_to_reps: Sequence[np.ndarray],
    entity_mods_all: np.ndarray,
    entity_key_codes_all: np.ndarray,
    gene_codes_all: np.ndarray,
    norm: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    repeats: int,
    ratios: Sequence[int],
) -> Dict[str, Any]:
    z_p, z_prof = encode_entities(
        model,
        entities,
        features,
        protein_embeddings,
        structure_embeddings,
        entity_features,
        dynamic_token_ids,
        entity_rows,
        entity_to_reps,
        entity_mods_all,
        norm,
        device,
        args.eval_batch_size,
    )
    mods = entity_mods_all[entity_rows]
    genes = gene_codes_all[entity_rows]
    entity_keys = entity_key_codes_all[entity_rows]
    profile_eval_mode = str(getattr(args, "eval_profile_mode", "entity_mean") or "entity_mean")
    if profile_eval_mode == "replicate_gallery":
        gallery = build_replicate_eval_gallery(
            entity_rows,
            entity_to_reps,
            entity_mods_all,
            entity_key_codes_all,
            gene_codes_all,
            int(getattr(args, "eval_max_replicates_per_entity", 0)),
            int(args.seed),
        )
        z_prof_reps = encode_profile_replicates(
            model,
            features,
            gallery["feature_rows"],
            gallery["modalities"],
            norm,
            device,
            args.eval_batch_size,
        )
        rep_entity_keys = gallery["entity_key_codes"]
        rep_genes = gallery["gene_codes"]
        rep_mods = gallery["modalities"]
        out = {
            "profile_eval_mode": "replicate_gallery",
            "num_entities": int(len(entity_rows)),
            "num_profile_replicates": int(len(rep_entity_keys)),
            "eval_max_replicates_per_entity": int(getattr(args, "eval_max_replicates_per_entity", 0)),
            "num_genes": int(entities.iloc[entity_rows]["gene_symbol"].nunique()) if len(entity_rows) else 0,
            "num_entity_keys": int(len(set(entity_keys.astype(int).tolist()))) if len(entity_rows) else 0,
            "modality_counts": {str(k): int(v) for k, v in entities.iloc[entity_rows]["perturbation_modality"].value_counts().items()},
            "full_gallery": {
                "protein_to_profile": full_gallery_metrics_general(z_p, z_prof_reps, entity_keys, rep_entity_keys),
                "profile_to_protein": full_gallery_metrics_general(z_prof_reps, z_p, rep_entity_keys, entity_keys),
            },
            "expert_gate_usage": encoded_gate_summary(z_p),
            "mocop_protocol_sampled": sampled_metrics_general(
                z_p,
                z_prof_reps,
                entity_keys,
                rep_entity_keys,
                genes,
                rep_genes,
                mods,
                rep_mods,
                ratios,
                repeats,
                args.seed,
                args.mask_crossmod_negatives,
                bool(getattr(args, "mask_same_gene_negatives", False)),
                "entity_key",
            ),
            "mocop_protocol_sampled_gene_only": sampled_metrics_general(
                z_p,
                z_prof_reps,
                genes,
                rep_genes,
                genes,
                rep_genes,
                mods,
                rep_mods,
                ratios,
                repeats,
                args.seed,
                args.mask_crossmod_negatives,
                False,
                "gene_id",
            ),
            "within_modality": {},
        }
        for name, modality_id in MODALITY_TO_ID.items():
            local_gene = np.where(mods == modality_id)[0]
            local_prof = np.where(rep_mods == modality_id)[0]
            if local_gene.size == 0 or local_prof.size == 0:
                out["within_modality"][name] = {"num_entities": int(local_gene.size), "num_profile_replicates": int(local_prof.size)}
                continue
            out["within_modality"][name] = {
                "profile_eval_mode": "replicate_gallery",
                "num_entities": int(local_gene.size),
                "num_profile_replicates": int(local_prof.size),
                "num_entity_keys": int(len(set(entity_keys[local_gene].astype(int).tolist()))),
                "full_gallery": {
                    "protein_to_profile": full_gallery_metrics_general(
                        slice_encoded(z_p, local_gene),
                        z_prof_reps[local_prof],
                        entity_keys[local_gene],
                        rep_entity_keys[local_prof],
                    ),
                    "profile_to_protein": full_gallery_metrics_general(
                        z_prof_reps[local_prof],
                        slice_encoded(z_p, local_gene),
                        rep_entity_keys[local_prof],
                        entity_keys[local_gene],
                    ),
                },
                "expert_gate_usage": encoded_gate_summary(slice_encoded(z_p, local_gene)),
                "mocop_protocol_sampled": sampled_metrics_general(
                    slice_encoded(z_p, local_gene),
                    z_prof_reps[local_prof],
                    entity_keys[local_gene],
                    rep_entity_keys[local_prof],
                    genes[local_gene],
                    rep_genes[local_prof],
                    mods[local_gene],
                    rep_mods[local_prof],
                    ratios,
                    repeats,
                    args.seed,
                    False,
                    bool(getattr(args, "mask_same_gene_negatives", False)),
                    "entity_key",
                ),
                "mocop_protocol_sampled_gene_only": sampled_metrics_general(
                    slice_encoded(z_p, local_gene),
                    z_prof_reps[local_prof],
                    genes[local_gene],
                    rep_genes[local_prof],
                    genes[local_gene],
                    rep_genes[local_prof],
                    mods[local_gene],
                    rep_mods[local_prof],
                    ratios,
                    repeats,
                    args.seed,
                    False,
                    False,
                    "gene_id",
                ),
            }
        out["top10_100"] = sampled_top10_bidirectional(out, ratio=100)
        out["top10_100_gene_only"] = sampled_top10_bidirectional({"mocop_protocol_sampled": out.get("mocop_protocol_sampled_gene_only", {})}, ratio=100)
        out["modality_stratified_top10_100"] = {
            "mixed": out["top10_100"],
            "crispr": sampled_top10_bidirectional(out["within_modality"].get("crispr", {}), ratio=100),
            "orf": sampled_top10_bidirectional(out["within_modality"].get("orf", {}), ratio=100),
        }
        out["modality_stratified_gene_only_top10_100"] = {
            "mixed": out["top10_100_gene_only"],
            "crispr": sampled_top10_bidirectional({"mocop_protocol_sampled": out["within_modality"].get("crispr", {}).get("mocop_protocol_sampled_gene_only", {})}, ratio=100),
            "orf": sampled_top10_bidirectional({"mocop_protocol_sampled": out["within_modality"].get("orf", {}).get("mocop_protocol_sampled_gene_only", {})}, ratio=100),
        }
        for scope_name, scope_metrics in out["modality_stratified_top10_100"].items():
            out[f"{scope_name}_entity_exact_top10_at_100"] = scope_metrics
        for scope_name, scope_metrics in out["modality_stratified_gene_only_top10_100"].items():
            out[f"{scope_name}_gene_only_top10_at_100"] = scope_metrics
        crispr_gate = out["within_modality"].get("crispr", {}).get("expert_gate_usage", {})
        orf_gate = out["within_modality"].get("orf", {}).get("expert_gate_usage", {})
        if crispr_gate.get("enabled") and orf_gate.get("enabled"):
            c = np.asarray(crispr_gate.get("mean_gate_probs", []), dtype=np.float64)
            o = np.asarray(orf_gate.get("mean_gate_probs", []), dtype=np.float64)
            if c.size and o.size and c.size == o.size:
                diff = np.abs(c - o)
                out["moe_modality_diagnostics"] = {
                    "overall_effective_num_experts": out["expert_gate_usage"].get("effective_num_experts"),
                    "crispr_effective_num_experts": crispr_gate.get("effective_num_experts"),
                    "orf_effective_num_experts": orf_gate.get("effective_num_experts"),
                    "crispr_gate_usage": crispr_gate.get("mean_gate_probs"),
                    "orf_gate_usage": orf_gate.get("mean_gate_probs"),
                    "mean_abs_crispr_orf_gate_diff": float(np.mean(diff)),
                    "max_abs_crispr_orf_gate_diff": float(np.max(diff)),
                    "possible_modality_specific_expert_routing": bool(np.max(diff) > 0.50 or int(np.argmax(c)) != int(np.argmax(o))),
                    "overall_gate_collapse": out["expert_gate_usage"].get("gate_collapse"),
                    "crispr_gate_collapse": crispr_gate.get("gate_collapse"),
                    "orf_gate_collapse": orf_gate.get("gate_collapse"),
                }
            else:
                out["moe_modality_diagnostics"] = {"available": False}
        else:
            out["moe_modality_diagnostics"] = {"available": False}
        return out
    out: Dict[str, Any] = {
        "profile_eval_mode": "entity_mean",
        "num_profile_replicates": int(len(entity_rows)),
        "num_entities": int(len(entity_rows)),
        "num_genes": int(entities.iloc[entity_rows]["gene_symbol"].nunique()) if len(entity_rows) else 0,
        "num_entity_keys": int(len(set(entity_keys.astype(int).tolist()))) if len(entity_rows) else 0,
        "modality_counts": {str(k): int(v) for k, v in entities.iloc[entity_rows]["perturbation_modality"].value_counts().items()},
        "full_gallery": {
            "protein_to_profile": full_gallery_metrics(z_p, z_prof, entity_keys),
            "profile_to_protein": full_gallery_metrics(z_prof, z_p, entity_keys),
        },
        "expert_gate_usage": encoded_gate_summary(z_p),
        "mocop_protocol_sampled": sampled_metrics(
            z_p,
            z_prof,
            entity_keys,
            genes,
            mods,
            ratios,
            repeats,
            args.seed,
            args.mask_crossmod_negatives,
            bool(getattr(args, "mask_same_gene_negatives", False)),
            "entity_key",
        ),
        "mocop_protocol_sampled_gene_only": sampled_metrics(
            z_p,
            z_prof,
            genes,
            genes,
            mods,
            ratios,
            repeats,
            args.seed,
            args.mask_crossmod_negatives,
            False,
            "gene_id",
        ),
        "within_modality": {},
    }
    for name, modality_id in MODALITY_TO_ID.items():
        local = np.where(mods == modality_id)[0]
        if local.size == 0:
            out["within_modality"][name] = {"num_entities": 0}
            continue
        out["within_modality"][name] = {
            "num_entities": int(local.size),
            "num_entity_keys": int(len(set(entity_keys[local].astype(int).tolist()))),
            "full_gallery": {
                "protein_to_profile": full_gallery_metrics(slice_encoded(z_p, local), slice_encoded(z_prof, local), entity_keys[local]),
                "profile_to_protein": full_gallery_metrics(slice_encoded(z_prof, local), slice_encoded(z_p, local), entity_keys[local]),
            },
            "expert_gate_usage": encoded_gate_summary(slice_encoded(z_p, local)),
            "mocop_protocol_sampled": sampled_metrics(
                slice_encoded(z_p, local),
                slice_encoded(z_prof, local),
                entity_keys[local],
                genes[local],
                mods[local],
                ratios,
                repeats,
                args.seed,
                False,
                bool(getattr(args, "mask_same_gene_negatives", False)),
                "entity_key",
            ),
            "mocop_protocol_sampled_gene_only": sampled_metrics(
                slice_encoded(z_p, local),
                slice_encoded(z_prof, local),
                genes[local],
                genes[local],
                mods[local],
                ratios,
                repeats,
                args.seed,
                False,
                False,
                "gene_id",
            ),
        }
    out["top10_100"] = sampled_top10_bidirectional(out, ratio=100)
    out["top10_100_gene_only"] = sampled_top10_bidirectional({"mocop_protocol_sampled": out.get("mocop_protocol_sampled_gene_only", {})}, ratio=100)
    out["modality_stratified_top10_100"] = {
        "mixed": out["top10_100"],
        "crispr": sampled_top10_bidirectional(out["within_modality"].get("crispr", {}), ratio=100),
        "orf": sampled_top10_bidirectional(out["within_modality"].get("orf", {}), ratio=100),
    }
    out["modality_stratified_gene_only_top10_100"] = {
        "mixed": out["top10_100_gene_only"],
        "crispr": sampled_top10_bidirectional({"mocop_protocol_sampled": out["within_modality"].get("crispr", {}).get("mocop_protocol_sampled_gene_only", {})}, ratio=100),
        "orf": sampled_top10_bidirectional({"mocop_protocol_sampled": out["within_modality"].get("orf", {}).get("mocop_protocol_sampled_gene_only", {})}, ratio=100),
    }
    for scope_name, scope_metrics in out["modality_stratified_top10_100"].items():
        out[f"{scope_name}_entity_exact_top10_at_100"] = scope_metrics
    for scope_name, scope_metrics in out["modality_stratified_gene_only_top10_100"].items():
        out[f"{scope_name}_gene_only_top10_at_100"] = scope_metrics
    crispr_gate = out["within_modality"].get("crispr", {}).get("expert_gate_usage", {})
    orf_gate = out["within_modality"].get("orf", {}).get("expert_gate_usage", {})
    if crispr_gate.get("enabled") and orf_gate.get("enabled"):
        c = np.asarray(crispr_gate.get("mean_gate_probs", []), dtype=np.float64)
        o = np.asarray(orf_gate.get("mean_gate_probs", []), dtype=np.float64)
        if c.size and o.size and c.size == o.size:
            diff = np.abs(c - o)
            out["moe_modality_diagnostics"] = {
                "overall_effective_num_experts": out["expert_gate_usage"].get("effective_num_experts"),
                "crispr_effective_num_experts": crispr_gate.get("effective_num_experts"),
                "orf_effective_num_experts": orf_gate.get("effective_num_experts"),
                "crispr_gate_usage": crispr_gate.get("mean_gate_probs"),
                "orf_gate_usage": orf_gate.get("mean_gate_probs"),
                "mean_abs_crispr_orf_gate_diff": float(np.mean(diff)),
                "max_abs_crispr_orf_gate_diff": float(np.max(diff)),
                "possible_modality_specific_expert_routing": bool(np.max(diff) > 0.50 or int(np.argmax(c)) != int(np.argmax(o))),
                "overall_gate_collapse": out["expert_gate_usage"].get("gate_collapse"),
                "crispr_gate_collapse": crispr_gate.get("gate_collapse"),
                "orf_gate_collapse": orf_gate.get("gate_collapse"),
            }
        else:
            out["moe_modality_diagnostics"] = {"available": False}
    else:
        out["moe_modality_diagnostics"] = {"available": False}
    return out


def select_val_rows(rows: np.ndarray, max_queries: int, rng: np.random.Generator) -> np.ndarray:
    if max_queries > 0 and len(rows) > max_queries:
        return rng.choice(rows, size=max_queries, replace=False)
    return rows


def read_last_jsonl(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    last: Optional[str] = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = line
    if last is None:
        return None
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return {"parse_error": str(path)}


def build_model(
    args: argparse.Namespace,
    protein_dim: int,
    profile_dim: int,
    structure_dim: Optional[int],
    dynamic_token_summary: Dict[str, Any],
    profile_adversary_summary: Dict[str, Any],
    device: torch.device,
    kg_graph: Optional[Dict[str, np.ndarray]] = None,
    kg_summary: Optional[Dict[str, Any]] = None,
    protein_embeddings: Optional[np.ndarray] = None,
) -> GeneMoCoP:
    adv_heads = profile_adversary_summary.get("heads", {}) if profile_adversary_summary else {}
    base_model = GeneMoCoP(
        protein_dim=protein_dim,
        profile_dim=profile_dim,
        structure_dim=structure_dim,
        protein_hidden_dims=parse_hidden_dims(args.protein_hidden_dims),
        profile_hidden_dims=parse_hidden_dims(args.profile_hidden_dims),
        embed_dim=args.embed_dim,
        fusion_dim=args.fusion_dim,
        protein_fusion=args.protein_fusion,
        modality_context_dim=args.modality_context_dim,
        dropout=args.dropout,
        use_modality_context=not args.disable_modality_context,
        use_profile_source_embedding=not args.disable_profile_source_embedding,
        protein_conditioning=args.protein_conditioning,
        dynamic_token_vocab_size=int(dynamic_token_summary.get("hash_buckets", 0)) if dynamic_token_summary.get("enabled") else 0,
        dynamic_token_dim=int(dynamic_token_summary.get("dim", 0)) if dynamic_token_summary.get("enabled") else 0,
        dynamic_embedding_dropout=float(dynamic_token_summary.get("dropout", 0.0)) if dynamic_token_summary.get("enabled") else 0.0,
        profile_adversary_hidden_dim=int(getattr(args, "profile_adversary_hidden_dim", 256)),
        num_profile_well_labels=int(adv_heads.get("well", {}).get("num_labels", 0)),
        num_profile_plate_labels=int(adv_heads.get("plate", {}).get("num_labels", 0)),
        num_profile_source_labels=int(adv_heads.get("source", {}).get("num_labels", 0)),
        gene_head=str(getattr(args, "gene_head", "single")),
        num_experts=int(getattr(args, "num_experts", 4)),
        expert_temperature_init=float(getattr(args, "expert_temperature_init", getattr(args, "temperature", 0.1))),
        expert_temperature_min=float(getattr(args, "expert_temperature_min", 0.03)),
        expert_temperature_max=float(getattr(args, "expert_temperature_max", 0.30)),
    )
    if kg_graph is not None and str(getattr(args, "kg_encoder", "none") or "none") != "none":
        if protein_embeddings is None:
            raise ValueError("protein_embeddings are required when kg_graph is enabled.")
        model = KGGeneMoCoP(
            base_model=base_model,
            base_gene_embeddings=protein_embeddings,
            edge_index=kg_graph["edge_index"],
            edge_weight=kg_graph["edge_weight"],
            kg_hidden_dim=int(getattr(args, "kg_hidden_dim", 512)),
            kg_layers=int(getattr(args, "kg_layers", 2)),
            kg_dropout=float(getattr(args, "kg_dropout", 0.1)),
            kg_fusion=str(getattr(args, "kg_fusion", "gated_residual")),
            zero_init_adapter=bool(getattr(args, "kg_zero_init_adapter", False)),
        )
        setattr(model, "kg_summary", kg_summary or {"enabled": True})
        return model.to(device)
    setattr(base_model, "kg_summary", kg_summary or {"enabled": False})
    return base_model.to(device)


def train(
    args: argparse.Namespace,
    model: GeneMoCoP,
    entities: pd.DataFrame,
    replicates: pd.DataFrame,
    features: np.ndarray,
    protein_embeddings: np.ndarray,
    structure_embeddings: Optional[np.ndarray],
    structure_summary: Dict[str, Any],
    entity_features: Optional[np.ndarray],
    entity_feature_summary: Dict[str, Any],
    dynamic_token_ids: Optional[np.ndarray],
    dynamic_token_summary: Dict[str, Any],
    profile_adversary_labels: Dict[str, Optional[np.ndarray]],
    profile_adversary_summary: Dict[str, Any],
    splits: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    entity_to_reps = build_entity_replicate_index(replicates, len(entities))
    entity_mods = modality_ids(entities)
    entity_gene_codes = gene_codes(entities)
    entity_key_codes_all = entity_key_codes(entities)
    train_entities = eligible_entities(entities, splits[args.split_name]["train"], protein_embeddings)
    val_entities = eligible_entities(entities, splits[args.split_name]["val"], protein_embeddings)
    full_train_entities = int(len(train_entities))
    train_entities = subsample_train_entities(train_entities, float(args.train_fraction), int(args.seed) + 211)
    if args.smoke_test:
        train_entities = train_entities[: min(512, len(train_entities))]
        val_entities = val_entities[: min(256, len(val_entities))]
    if len(train_entities) <= 1:
        raise RuntimeError("Not enough eligible train entities.")
    norm = fit_profile_normalizer(features, entity_to_reps, train_entities, entity_mods, args.profile_norm)
    ckpt_dir = args.checkpoint_dir / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_weights, sample_weight_summary = build_entity_sample_weights(
        features,
        entity_to_reps,
        train_entities,
        entity_mods,
        norm,
        args.sample_weight_mode,
        args.sample_weight_power,
        args.sample_weight_min,
        args.sample_weight_max,
        args.sample_weight_splits,
        args.seed,
    )
    reliability_artifacts = None
    theta_by_entity_key: Optional[np.ndarray] = None
    theta_valid: Optional[np.ndarray] = None
    leakage_guard_summary: Dict[str, Any] = {"strict_no_leakage": bool(args.strict_no_leakage), "passed": not bool(args.strict_no_leakage)}
    if bool(args.use_reliability_weight) or str(args.gene_loss_type) == "soft_clip" or bool(args.strict_no_leakage):
        selected_modality_names = modality_filter_from_arg(str(args.modalities))
        selected_modality_ids = [MODALITY_TO_ID[m] for m in selected_modality_names]
        reliability_artifacts = estimate_train_only_gene_reliability(
            entities=entities,
            replicates=replicates,
            features=features,
            entity_to_reps=entity_to_reps,
            entity_mods=entity_mods,
            gene_codes_all=entity_gene_codes,
            train_entities=train_entities,
            norm=norm,
            transform_feature_rows=transform_feature_rows,
            allowed_modality_ids=selected_modality_ids,
            modality_id_to_name={v: k for k, v in MODALITY_TO_ID.items()},
            default_guide_consistency=float(args.reliability_default_guide_consistency),
        )
        theta_by_entity_key = reliability_artifacts.theta_by_entity_key
        theta_valid = reliability_artifacts.theta_valid
        reliability_artifacts.table.to_csv(ckpt_dir / "train_only_entity_reliability.csv", index=False)
        reliability_artifacts.theta_table.to_csv(ckpt_dir / "train_only_entity_theta_table.csv", index=False)
        np.save(ckpt_dir / "train_only_entity_theta_by_entity_key.npy", theta_by_entity_key.astype(np.float32))
        np.save(ckpt_dir / "train_only_entity_theta_valid.npy", theta_valid.astype(bool))
        if bool(args.use_reliability_weight):
            sample_weights = (sample_weights * reliability_artifacts.entity_weights).astype(np.float32)
            train_mean = float(np.mean(sample_weights[train_entities])) if len(train_entities) else 1.0
            if train_mean > 1e-8:
                sample_weights = (sample_weights / train_mean).astype(np.float32)
        sample_weight_summary = {
            **sample_weight_summary,
            "reliability_weight_enabled": bool(args.use_reliability_weight),
            "train_only_entity_reliability": reliability_artifacts.summary,
        }
    if bool(args.strict_no_leakage):
        leakage_guard_summary = strict_no_leakage_assertions(
            entities=entities,
            splits=splits,
            split_name=args.split_name,
            gene_codes_all=entity_gene_codes,
            reliability_table=reliability_artifacts.table if reliability_artifacts is not None else None,
            theta_valid=theta_valid,
            config={**vars(args), "profile_normalizer_fit_split": norm.get("fit_split", "train")},
            entity_key_codes_all=entity_key_codes_all,
        )
        write_json(ckpt_dir / "strict_no_leakage_guard.json", leakage_guard_summary)
        print(json.dumps({"strict_no_leakage_guard": leakage_guard_summary}, sort_keys=True), flush=True)
    entity_mean_profiles: Optional[np.ndarray] = None
    if args.train_profile_mode == "mean_profile":
        entity_mean_profiles = compute_entity_mean_profiles(features, entity_to_reps, entity_mods, norm)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"{args.run_name}_train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    steps_per_epoch = int(args.steps_per_epoch or max(1, np.ceil(len(train_entities) / max(1, args.batch_size))))
    config = vars(args).copy()
    selected_modalities_for_config = modality_filter_from_arg(str(args.modalities))
    historical_current = (
        float(args.historical_baseline)
        if args.historical_baseline is not None
        else float(args.historical_mixed_baseline)
        if len(selected_modalities_for_config) > 1
        else float(args.historical_crispr_only_baseline)
        if selected_modalities_for_config == ["crispr"]
        else None
    )
    config.update(
        {
            "num_train_entities": int(len(train_entities)),
            "num_full_train_entities": int(full_train_entities),
            "train_fraction_requested": float(args.train_fraction),
            "train_fraction_actual": float(len(train_entities) / max(1, full_train_entities)),
            "num_val_entities": int(len(val_entities)),
            "protein_dim": int(protein_embeddings.shape[1] + (0 if entity_features is None else entity_features.shape[1])),
            "base_protein_dim": int(protein_embeddings.shape[1]),
            "entity_features": entity_feature_summary,
            "dynamic_entity_encoding": dynamic_token_summary,
            "profile_adversaries": profile_adversary_summary,
            "structure_dim": int(structure_embeddings.shape[1]) if structure_embeddings is not None else 0,
            "structure_embeddings": structure_summary,
            "kg_graph": getattr(model, "kg_summary", {"enabled": False}),
            "profile_dim": int(features.shape[1]),
            "profile_normalizer": normalizer_summary(norm),
            "selected_modalities": selected_modalities_for_config,
            "use_modality_feature": bool(not args.disable_modality_context and str(args.protein_conditioning) != "none"),
            "historical_mixed_baseline": float(args.historical_mixed_baseline),
            "historical_crispr_only_baseline": float(args.historical_crispr_only_baseline),
            "historical_baseline_for_current_modality": historical_current,
            "sample_weighting": sample_weight_summary,
            "strict_no_leakage_guard": leakage_guard_summary,
            "gene_head": str(args.gene_head),
            "num_experts": int(args.num_experts),
            "gene_loss_type": str(args.gene_loss_type),
            "soft_clip_eta": float(args.soft_clip_eta),
            "soft_clip_gamma": float(args.soft_clip_gamma),
            "lambda_expert_div": float(args.lambda_expert_div),
            "lambda_expert_balance": float(args.lambda_expert_balance),
            "profile_geometry_loss_weight": float(args.profile_geometry_loss_weight),
            "gene_profile_geometry_loss_weight": float(args.gene_profile_geometry_loss_weight),
            "profile_neighbor_distill_weight": float(args.profile_neighbor_distill_weight),
            "profile_neighbor_target_temperature": float(args.profile_neighbor_target_temperature),
            "profile_replicate_contrastive_loss_weight": float(args.profile_replicate_contrastive_loss_weight),
            "clip_loss_type": str(args.clip_loss_type),
            "sigmoid_neg_weight": float(args.sigmoid_neg_weight),
            "rank_loss_weight": float(args.rank_loss_weight),
            "rank_margin": float(args.rank_margin),
            "rank_hard_negatives": int(args.rank_hard_negatives),
            "uses_target_or_link_labels_for_training": False,
            "training_profile_sampling": args.train_profile_mode,
            "eval_profile_mode": str(args.eval_profile_mode),
            "eval_max_replicates_per_entity": int(args.eval_max_replicates_per_entity),
            "gene_loss": (
                "reliability_weighted_soft_clip_moe"
                if str(args.gene_loss_type) == "soft_clip" and str(args.gene_head) == "multi_expert"
                else "soft_clip"
                if str(args.gene_loss_type) == "soft_clip"
                else "multi_expert_original_infonce"
                if str(args.gene_head) == "multi_expert"
                else
                "weighted_multipositive_clip"
                if (args.same_gene_same_modality_positive_weight > 0 or args.same_gene_crossmod_positive_weight > 0)
                else f"{args.clip_loss_type}_clip"
            ),
            "selection_metric": "MoCoP-style sampled Top10 average",
            "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "total_params": int(sum(p.numel() for p in model.parameters())),
        }
    )
    write_json(ckpt_dir / "config.json", config)

    ema_enabled = not bool(args.disable_ema)
    ema_model = copy.deepcopy(model).to(device).eval() if ema_enabled else model
    if ema_enabled:
        for p in ema_model.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    rng = np.random.default_rng(args.seed)
    best = -float("inf")
    gene_modality_index = build_gene_modality_entity_index(train_entities, entity_gene_codes, entity_mods)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: List[float] = []
        batch_weight_means: List[float] = []
        expert_div_losses: List[float] = []
        expert_balance_losses: List[float] = []
        expert_entropies: List[float] = []
        expert_gate_usages: List[List[float]] = []
        expert_effective_nums: List[float] = []
        expert_max_gate_probs: List[float] = []
        expert_min_gate_probs: List[float] = []
        expert_pairwise_cosine_means: List[float] = []
        expert_pairwise_cosine_matrices: List[List[List[float]]] = []
        expert_norm_means: List[float] = []
        expert_norm_stds: List[float] = []
        gate_usage_by_modality_batches: Dict[str, List[List[float]]] = {"crispr": [], "orf": []}
        temperature_per_expert_batches: List[List[float]] = []
        temperature_mins: List[float] = []
        temperature_maxs: List[float] = []
        temperature_means: List[float] = []
        temperature_clamped_fractions: List[float] = []
        logit_diag_values: Dict[str, List[float]] = {
            "logits_mean": [],
            "logits_std": [],
            "positive_logits_mean": [],
            "negative_logits_mean": [],
            "positive_minus_negative_margin": [],
            "max_logit": [],
            "min_logit": [],
        }
        gene_grad_norms: List[float] = []
        moe_grad_norms: List[float] = []
        profile_grad_norms: List[float] = []
        gradient_finite_flags: List[bool] = []
        forward_finite_flags: List[bool] = []
        soft_target_entropies: List[float] = []
        soft_target_mass_stats: List[Dict[str, float]] = []
        last_temperatures: List[float] = []
        for _ in range(steps_per_epoch):
            batch_size = min(args.batch_size, len(train_entities))
            batch_entities = rng.choice(train_entities, size=batch_size, replace=False)
            batch_entities = augment_with_same_gene_siblings(
                batch_entities,
                entity_gene_codes,
                entity_mods,
                gene_modality_index,
                rng,
                args.same_gene_sibling_batch_prob,
            )
            mod_np = entity_mods[batch_entities]
            gene_np = entity_gene_codes[batch_entities]
            entity_key_np = entity_key_codes_all[batch_entities]
            mod = torch.from_numpy(mod_np.astype(np.int64)).long().to(device)
            gene = torch.from_numpy(gene_np.astype(np.int64)).long().to(device)
            entity_key = torch.from_numpy(entity_key_np.astype(np.int64)).long().to(device)
            weight = None
            if args.sample_weight_mode != "none" or bool(args.use_reliability_weight):
                weight = torch.from_numpy(sample_weights[batch_entities].astype(np.float32)).to(device)
                batch_weight_means.append(float(weight.detach().cpu().mean()))
            with autocast(enabled=bool(args.amp and device.type == "cuda")):
                z_p = encode_gene_branch_entities(
                    model,
                    entities,
                    protein_embeddings,
                    entity_features,
                    structure_embeddings,
                    dynamic_token_ids,
                    batch_entities,
                    mod,
                    device,
                )
                if entity_mean_profiles is None:
                    rep_rows = sample_replicate_features(batch_entities, entity_to_reps, rng)
                    profile_x = profile_tensor(features, rep_rows, mod_np, norm, device)
                else:
                    profile_x = torch.from_numpy(entity_mean_profiles[batch_entities]).to(device=device, dtype=torch.float32)
                z_prof = model.encode_profile(profile_x, mod)
                logits = gene_profile_logits(z_p, z_prof, args.temperature)
                logit_diag = logit_scale_diagnostics(logits, entity_key)
                for key, value in logit_diag.items():
                    if value is not None and np.isfinite(float(value)):
                        logit_diag_values[key].append(float(value))
                if str(args.gene_loss_type) == "soft_clip":
                    if theta_by_entity_key is None or theta_valid is None:
                        raise RuntimeError("soft_clip requires train-only theta_by_entity_key artifacts.")
                    if not np.all(theta_valid[entity_key_np]):
                        raise RuntimeError("soft_clip batch contains an entity_key without train-only theta; this indicates leakage or split misuse.")
                    theta = torch.from_numpy(theta_by_entity_key[entity_key_np].astype(np.float32)).to(device)
                    soft_targets = soft_clip_targets(entity_key, theta, args.soft_clip_eta, args.soft_clip_gamma)
                    soft_target_entropies.append(float(target_entropy(soft_targets).detach().cpu()))
                    soft_diag = soft_clip_target_diagnostics(entity_key, gene, mod, theta, args.soft_clip_eta, args.soft_clip_gamma)
                    soft_target_mass_stats.append(soft_diag)
                    loss = soft_clip_loss_from_logits(
                        logits,
                        entity_key,
                        theta,
                        args.soft_clip_eta,
                        args.soft_clip_gamma,
                        weight,
                    )
                elif is_multi_expert_output(z_p):
                    if args.clip_loss_type == "sigmoid":
                        loss = sigmoid_clip_loss_from_logits(
                            logits,
                            gene,
                            mod,
                            args.mask_crossmod_negatives,
                            bool(getattr(args, "mask_same_gene_negatives", False)),
                            weight,
                            args.sigmoid_neg_weight,
                        )
                    else:
                        loss = multipositive_clip_loss_from_logits(
                            logits,
                            entity_key,
                            gene,
                            mod,
                            args.mask_crossmod_negatives,
                            bool(getattr(args, "mask_same_gene_negatives", False)),
                            weight,
                        )
                elif args.same_gene_same_modality_positive_weight > 0 or args.same_gene_crossmod_positive_weight > 0:
                    loss = weighted_multipositive_clip_loss(
                        z_p,
                        z_prof,
                        gene,
                        mod,
                        args.temperature,
                        args.mask_crossmod_negatives,
                        bool(getattr(args, "mask_same_gene_negatives", False)),
                        args.same_gene_same_modality_positive_weight,
                        args.same_gene_crossmod_positive_weight,
                        weight,
                    )
                else:
                    if args.clip_loss_type == "sigmoid":
                        loss = sigmoid_clip_loss(
                            z_p,
                            z_prof,
                            gene,
                            mod,
                            args.temperature,
                            args.mask_crossmod_negatives,
                            bool(getattr(args, "mask_same_gene_negatives", False)),
                            weight,
                            args.sigmoid_neg_weight,
                        )
                    else:
                        loss = multipositive_clip_loss_from_logits(
                            logits,
                            entity_key,
                            gene,
                            mod,
                            args.mask_crossmod_negatives,
                            bool(getattr(args, "mask_same_gene_negatives", False)),
                            weight,
                        )
                if args.rank_loss_weight > 0:
                    if is_multi_expert_output(z_p):
                        loss = loss + float(args.rank_loss_weight) * pairwise_rank_loss_from_logits(
                            logits,
                            gene,
                            mod,
                            args.mask_crossmod_negatives,
                            bool(getattr(args, "mask_same_gene_negatives", False)),
                            args.rank_margin,
                            args.rank_hard_negatives,
                            weight,
                        )
                    else:
                        loss = loss + float(args.rank_loss_weight) * pairwise_rank_clip_loss(
                            z_p,
                            z_prof,
                            gene,
                            mod,
                            args.temperature,
                            args.mask_crossmod_negatives,
                            bool(getattr(args, "mask_same_gene_negatives", False)),
                            args.rank_margin,
                            args.rank_hard_negatives,
                            weight,
                        )
                if args.profile_geometry_loss_weight > 0:
                    loss = loss + float(args.profile_geometry_loss_weight) * profile_geometry_loss(profile_x, z_prof)
                if args.gene_profile_geometry_loss_weight > 0:
                    loss = loss + float(args.gene_profile_geometry_loss_weight) * gene_profile_geometry_loss(gene_output_as_single(z_p), profile_x)
                if args.profile_neighbor_distill_weight > 0:
                    loss = loss + float(args.profile_neighbor_distill_weight) * profile_neighbor_distill_loss(
                        gene_output_as_single(z_p),
                        z_prof,
                        profile_x,
                        args.temperature,
                        args.profile_neighbor_target_temperature,
                        weight,
                    )
                if args.profile_replicate_contrastive_loss_weight > 0 and entity_mean_profiles is None:
                    rep_rows_2 = sample_replicate_features(batch_entities, entity_to_reps, rng)
                    profile_x_2 = profile_tensor(features, rep_rows_2, mod_np, norm, device)
                    z_prof_2 = model.encode_profile(profile_x_2, mod)
                    loss = loss + float(args.profile_replicate_contrastive_loss_weight) * masked_clip_loss(
                        z_prof,
                        z_prof_2,
                        gene,
                        mod,
                        args.temperature,
                        False,
                        bool(getattr(args, "mask_same_gene_negatives", False)),
                        weight,
                    )
                if (
                    entity_mean_profiles is None
                    and (
                        args.profile_adversary_well_weight > 0
                        or args.profile_adversary_plate_weight > 0
                        or args.profile_adversary_source_weight > 0
                    )
                ):
                    loss = loss + profile_adversary_loss(
                        model,
                        z_prof,
                        rep_rows,
                        profile_adversary_labels,
                        device,
                        args.profile_adversary_well_weight,
                        args.profile_adversary_plate_weight,
                        args.profile_adversary_source_weight,
                    )
                if is_multi_expert_output(z_p):
                    div_loss, balance_loss, expert_stats = multi_expert_regularization(z_p)
                    loss = loss + float(args.lambda_expert_div) * div_loss + float(args.lambda_expert_balance) * balance_loss
                    expert_div_losses.append(expert_stats["expert_diversity_loss"])
                    expert_balance_losses.append(expert_stats["expert_load_balance_loss"])
                    expert_entropies.append(expert_stats["expert_gate_entropy"])
                    expert_gate_usages.append(expert_stats["expert_gate_usage"])
                    if expert_stats.get("effective_num_experts") is not None:
                        expert_effective_nums.append(float(expert_stats["effective_num_experts"]))
                    if expert_stats.get("max_gate_prob") is not None:
                        expert_max_gate_probs.append(float(expert_stats["max_gate_prob"]))
                    if expert_stats.get("min_gate_prob") is not None:
                        expert_min_gate_probs.append(float(expert_stats["min_gate_prob"]))
                    if expert_stats.get("expert_pairwise_cosine_mean") is not None:
                        expert_pairwise_cosine_means.append(float(expert_stats["expert_pairwise_cosine_mean"]))
                    if expert_stats.get("expert_pairwise_cosine_matrix") is not None:
                        expert_pairwise_cosine_matrices.append(expert_stats["expert_pairwise_cosine_matrix"])
                    if expert_stats.get("expert_norm_mean") is not None:
                        expert_norm_means.append(float(expert_stats["expert_norm_mean"]))
                    if expert_stats.get("expert_norm_std") is not None:
                        expert_norm_stds.append(float(expert_stats["expert_norm_std"]))
                    for mod_name, usage in gate_usage_by_modality(z_p, mod).items():
                        gate_usage_by_modality_batches.setdefault(mod_name, []).append(usage)
                    temp_diag = temperature_diagnostics(
                        z_p,
                        float(getattr(args, "expert_temperature_min", 0.03)),
                        float(getattr(args, "expert_temperature_max", 0.30)),
                    )
                    if temp_diag.get("temperature_per_expert"):
                        temperature_per_expert_batches.append(temp_diag["temperature_per_expert"])
                    for key, dst in [
                        ("temperature_min", temperature_mins),
                        ("temperature_max", temperature_maxs),
                        ("temperature_mean", temperature_means),
                        ("temperature_clamped_fraction", temperature_clamped_fractions),
                    ]:
                        if temp_diag.get(key) is not None:
                            dst.append(float(temp_diag[key]))
                    last_temperatures = expert_temperature_values(z_p)
                forward_ok = not (
                    has_nonfinite_tensor(z_p)
                    or has_nonfinite_tensor(z_prof)
                    or has_nonfinite_tensor(logits)
                    or not bool(torch.isfinite(loss.detach()).all().item())
                )
                forward_finite_flags.append(bool(forward_ok))
                if not forward_ok:
                    raise RuntimeError("NaN/Inf guard failed before backward in gene branch training.")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_diag = model_gradient_diagnostics(model)
            if grad_diag.get("gene_encoder_grad_norm") is not None:
                gene_grad_norms.append(float(grad_diag["gene_encoder_grad_norm"]))
            if grad_diag.get("moe_head_grad_norm") is not None:
                moe_grad_norms.append(float(grad_diag["moe_head_grad_norm"]))
            if grad_diag.get("profile_encoder_grad_norm") is not None:
                profile_grad_norms.append(float(grad_diag["profile_encoder_grad_norm"]))
            gradient_finite_flags.append(bool(grad_diag.get("gradients_all_finite", True)))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if ema_enabled:
                update_ema(model, ema_model, args.ema_decay)
            losses.append(float(loss.detach().cpu().item()))

        eval_model = ema_model if ema_enabled else model
        rec: Dict[str, Any] = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(losses)) if losses else None,
            "steps_per_epoch": int(steps_per_epoch),
            "strict_no_leakage_passed": bool(leakage_guard_summary.get("passed", False)),
            "mean_reliability_weight": float(np.mean(batch_weight_means)) if batch_weight_means else None,
            "expert_diversity_loss": float(np.mean(expert_div_losses)) if expert_div_losses else None,
            "expert_load_balance_loss": float(np.mean(expert_balance_losses)) if expert_balance_losses else None,
            "expert_gate_entropy": float(np.mean(expert_entropies)) if expert_entropies else None,
            "expert_gate_usage": [float(x) for x in np.asarray(expert_gate_usages, dtype=np.float64).mean(axis=0).tolist()] if expert_gate_usages else None,
            "expert_temperatures": last_temperatures,
            "gate_entropy": finite_mean(expert_entropies),
            "effective_num_experts": finite_mean(expert_effective_nums),
            "mean_gate_prob_per_expert": mean_vector(expert_gate_usages),
            "max_gate_prob": finite_mean(expert_max_gate_probs),
            "min_gate_prob": finite_mean(expert_min_gate_probs),
            "mean_gate_prob_per_expert_by_modality": {
                k: mean_vector(v) for k, v in gate_usage_by_modality_batches.items() if mean_vector(v) is not None
            },
            "temperature_per_expert": mean_vector(temperature_per_expert_batches) or last_temperatures,
            "temperature_min": finite_mean(temperature_mins),
            "temperature_max": finite_mean(temperature_maxs),
            "temperature_mean": finite_mean(temperature_means),
            "temperature_clamped_fraction": finite_mean(temperature_clamped_fractions),
            "temperature_any_clamped": bool(any(float(x) > 0.0 for x in temperature_clamped_fractions)) if temperature_clamped_fractions else None,
            "expert_pairwise_cosine_mean": finite_mean(expert_pairwise_cosine_means),
            "expert_pairwise_cosine_matrix": mean_matrix(expert_pairwise_cosine_matrices),
            "expert_norm_mean": finite_mean(expert_norm_means),
            "expert_norm_std": finite_mean(expert_norm_stds),
            "gene_encoder_grad_norm": finite_mean(gene_grad_norms),
            "moe_head_grad_norm": finite_mean(moe_grad_norms),
            "profile_encoder_grad_norm": finite_mean(profile_grad_norms),
            "gradients_all_finite": bool(all(gradient_finite_flags)) if gradient_finite_flags else None,
            "forward_tensors_all_finite": bool(all(forward_finite_flags)) if forward_finite_flags else None,
            "soft_target_entropy": float(np.mean(soft_target_entropies)) if soft_target_entropies else None,
        }
        for key, vals in logit_diag_values.items():
            rec[key] = finite_mean(vals)
        if soft_target_mass_stats:
            for key in [
                "same_entity_positive_mass",
                "same_gene_cross_modality_soft_mass",
                "cross_modality_offdiag_mass",
                "fraction_cross_modality_soft_mass_gt_same_entity_mass",
            ]:
                rec[key] = float(np.mean([x[key] for x in soft_target_mass_stats]))
        if reliability_artifacts is not None:
            rec["profile_activity_mean"] = reliability_artifacts.summary.get("profile_activity_mean")
            rec["replicate_consistency_mean"] = reliability_artifacts.summary.get("replicate_consistency_mean")
            rec["guide_consistency_mean"] = reliability_artifacts.summary.get("guide_consistency_mean")
            rec["reagent_consistency_mean"] = reliability_artifacts.summary.get("reagent_consistency_mean")
            rec["missing_guide_consistency_fraction"] = reliability_artifacts.summary.get("missing_guide_consistency_fraction")
        if int(args.strict_split_log_every) > 0 and (epoch % int(args.strict_split_log_every) == 0 or epoch == args.epochs):
            for split_label, split_rows_all in [
                ("train", train_entities),
                ("val", val_entities),
                ("test", eligible_entities(entities, splits[args.split_name]["test"], protein_embeddings)),
            ]:
                if len(split_rows_all) == 0:
                    continue
                split_rows = select_val_rows(np.asarray(split_rows_all, dtype=np.int64), args.val_max_queries, rng)
                split_metrics = evaluate_entity_rows(
                    eval_model,
                    entities,
                    features,
                    protein_embeddings,
                    structure_embeddings,
                    entity_features,
                    dynamic_token_ids,
                    split_rows,
                    entity_to_reps,
                    entity_mods,
                    entity_key_codes_all,
                    entity_gene_codes,
                    norm,
                    args,
                    device,
                    repeats=max(1, args.val_repeats),
                    ratios=[args.val_negative_ratio],
                )
                p2s = split_metrics["mocop_protocol_sampled"]["protein_to_profile"][f"1:{args.val_negative_ratio}"]
                q2s = split_metrics["mocop_protocol_sampled"]["profile_to_protein"][f"1:{args.val_negative_ratio}"]
                rec[f"{split_label}_mocop_p2profile_Top10"] = p2s.get("Top10_accuracy_mean")
                rec[f"{split_label}_mocop_profile2p_Top10"] = q2s.get("Top10_accuracy_mean")
                rec[f"{split_label}_mocop_alignment_score"] = 0.5 * float((p2s.get("Top10_accuracy_mean") or 0.0) + (q2s.get("Top10_accuracy_mean") or 0.0))
                for scope_name, scope_metrics in split_metrics.get("modality_stratified_top10_100", {}).items():
                    rec[f"{split_label}_{scope_name}_Top10_100_gene2profile"] = scope_metrics.get("gene_to_profile")
                    rec[f"{split_label}_{scope_name}_Top10_100_profile2gene"] = scope_metrics.get("profile_to_gene")
                    rec[f"{split_label}_{scope_name}_Top10_100_mean"] = scope_metrics.get("bidirectional_mean")
                    rec[f"{split_label}_{scope_name}_entity_exact_top10_at_100_gene_to_profile"] = scope_metrics.get("gene_to_profile")
                    rec[f"{split_label}_{scope_name}_entity_exact_top10_at_100_profile_to_gene"] = scope_metrics.get("profile_to_gene")
                    rec[f"{split_label}_{scope_name}_entity_exact_top10_at_100_mean"] = scope_metrics.get("bidirectional_mean")
                for scope_name, scope_metrics in split_metrics.get("modality_stratified_gene_only_top10_100", {}).items():
                    rec[f"{split_label}_{scope_name}_gene_only_top10_at_100_gene_to_profile"] = scope_metrics.get("gene_to_profile")
                    rec[f"{split_label}_{scope_name}_gene_only_top10_at_100_profile_to_gene"] = scope_metrics.get("profile_to_gene")
                    rec[f"{split_label}_{scope_name}_gene_only_top10_at_100_mean"] = scope_metrics.get("bidirectional_mean")
        if (epoch % args.val_every == 0 or epoch == args.epochs) and len(val_entities):
            val_rows = select_val_rows(val_entities, args.val_max_queries, rng)
            val = evaluate_entity_rows(
                eval_model,
                entities,
                features,
                protein_embeddings,
                structure_embeddings,
                entity_features,
                dynamic_token_ids,
                val_rows,
                entity_to_reps,
                entity_mods,
                entity_key_codes_all,
                entity_gene_codes,
                norm,
                args,
                device,
                repeats=max(1, args.val_repeats),
                ratios=[args.val_negative_ratio],
            )
            p2 = val["mocop_protocol_sampled"]["protein_to_profile"][f"1:{args.val_negative_ratio}"]
            q2 = val["mocop_protocol_sampled"]["profile_to_protein"][f"1:{args.val_negative_ratio}"]
            rec["val_mocop_p2profile_Top10"] = p2.get("Top10_accuracy_mean")
            rec["val_mocop_profile2p_Top10"] = q2.get("Top10_accuracy_mean")
            rec["val_mocop_alignment_score"] = 0.5 * float((p2.get("Top10_accuracy_mean") or 0.0) + (q2.get("Top10_accuracy_mean") or 0.0))
            rec["val_full_p2profile_Recall@10"] = val["full_gallery"]["protein_to_profile"].get("Recall@10")
            for scope_name, scope_metrics in val.get("modality_stratified_top10_100", {}).items():
                rec[f"val_{scope_name}_Top10_100_gene2profile"] = scope_metrics.get("gene_to_profile")
                rec[f"val_{scope_name}_Top10_100_profile2gene"] = scope_metrics.get("profile_to_gene")
                rec[f"val_{scope_name}_Top10_100_mean"] = scope_metrics.get("bidirectional_mean")
                rec[f"val_{scope_name}_entity_exact_top10_at_100_gene_to_profile"] = scope_metrics.get("gene_to_profile")
                rec[f"val_{scope_name}_entity_exact_top10_at_100_profile_to_gene"] = scope_metrics.get("profile_to_gene")
                rec[f"val_{scope_name}_entity_exact_top10_at_100_mean"] = scope_metrics.get("bidirectional_mean")
            for scope_name, scope_metrics in val.get("modality_stratified_gene_only_top10_100", {}).items():
                rec[f"val_{scope_name}_gene_only_top10_at_100_gene_to_profile"] = scope_metrics.get("gene_to_profile")
                rec[f"val_{scope_name}_gene_only_top10_at_100_profile_to_gene"] = scope_metrics.get("profile_to_gene")
                rec[f"val_{scope_name}_gene_only_top10_at_100_mean"] = scope_metrics.get("bidirectional_mean")
            if rec["val_mocop_alignment_score"] > best:
                best = float(rec["val_mocop_alignment_score"])
                torch.save(
                    {
                        "model_state_dict": eval_model.state_dict(),
                        "epoch": int(epoch),
                        "best_score": best,
                        "config": config,
                        "profile_normalizer": norm,
                        "train_only_gene_reliability_summary": reliability_artifacts.summary if reliability_artifacts is not None else None,
                        "train_only_entity_reliability_summary": reliability_artifacts.summary if reliability_artifacts is not None else None,
                        "train_only_entity_theta_by_entity_key": theta_by_entity_key,
                        "train_only_entity_theta_valid": theta_valid,
                        "strict_no_leakage_guard": leakage_guard_summary,
                    },
                    ckpt_dir / "best_model.pt",
                )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec, sort_keys=True) + "\n")
        print(json.dumps(rec, sort_keys=True), flush=True)

    if not (ckpt_dir / "best_model.pt").exists():
        torch.save(
            {
                "model_state_dict": (ema_model if ema_enabled else model).state_dict(),
                "epoch": int(args.epochs),
                "best_score": best,
                "config": config,
                "profile_normalizer": norm,
                "train_only_gene_reliability_summary": reliability_artifacts.summary if reliability_artifacts is not None else None,
                "train_only_entity_reliability_summary": reliability_artifacts.summary if reliability_artifacts is not None else None,
                "train_only_entity_theta_by_entity_key": theta_by_entity_key,
                "train_only_entity_theta_valid": theta_valid,
                "strict_no_leakage_guard": leakage_guard_summary,
            },
            ckpt_dir / "best_model.pt",
        )
    return norm


def evaluate(
    args: argparse.Namespace,
    model: GeneMoCoP,
    entities: pd.DataFrame,
    replicates: pd.DataFrame,
    features: np.ndarray,
    protein_embeddings: np.ndarray,
    structure_embeddings: Optional[np.ndarray],
    structure_summary: Dict[str, Any],
    entity_features: Optional[np.ndarray],
    entity_feature_summary: Dict[str, Any],
    dynamic_token_ids: Optional[np.ndarray],
    dynamic_token_summary: Dict[str, Any],
    profile_adversary_summary: Dict[str, Any],
    splits: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    ckpt = args.checkpoint_dir / args.run_name / "best_model.pt"
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    norm = payload.get("profile_normalizer", {"mode": "none"})
    entity_to_reps = build_entity_replicate_index(replicates, len(entities))
    entity_mods = modality_ids(entities)
    entity_gene_codes = gene_codes(entities)
    entity_key_codes_all = entity_key_codes(entities)
    test_entities = eligible_entities(entities, splits[args.split_name]["test"], protein_embeddings)
    if args.smoke_test:
        test_entities = test_entities[: min(256, len(test_entities))]
    metrics = evaluate_entity_rows(
        model,
        entities,
        features,
        protein_embeddings,
        structure_embeddings,
        entity_features,
        dynamic_token_ids,
        test_entities,
        entity_to_reps,
        entity_mods,
        entity_key_codes_all,
        entity_gene_codes,
        norm,
        args,
        device,
        repeats=args.num_repeats,
        ratios=parse_ints(args.negative_ratios),
    )
    metrics.update(
        {
            "run_name": args.run_name,
            "protein_encoder_name": args.protein_encoder_name,
            "structure_encoder_name": args.structure_encoder_name,
            "protein_fusion": args.protein_fusion,
            "split_name": args.split_name,
            "query_split": "test",
            "checkpoint_epoch": payload.get("epoch"),
            "best_score": payload.get("best_score"),
            "data_dir": str(args.data_dir),
            "protein_embedding_dir": str(args.protein_embedding_dir),
            "structure_embedding_dir": str(args.structure_embedding_dir) if args.structure_embedding_dir is not None else None,
            "structure_embeddings": structure_summary,
            "kg_graph": getattr(model, "kg_summary", {"enabled": False}),
            "entity_features": entity_feature_summary,
            "dynamic_entity_encoding": dynamic_token_summary,
            "profile_adversaries": profile_adversary_summary,
            "profile_normalizer": normalizer_summary(norm),
            "eval_profile_mode": str(args.eval_profile_mode),
            "eval_max_replicates_per_entity": int(args.eval_max_replicates_per_entity),
            "train_only_gene_reliability_summary": payload.get("train_only_gene_reliability_summary") or payload.get("train_only_entity_reliability_summary"),
            "train_only_entity_reliability_summary": payload.get("train_only_entity_reliability_summary") or payload.get("train_only_gene_reliability_summary"),
            "strict_no_leakage_guard": payload.get("strict_no_leakage_guard"),
            "phase1_training_log_last": read_last_jsonl(args.log_dir / f"{args.run_name}_train_log.jsonl"),
            "historical_mixed_baseline": float(args.historical_mixed_baseline),
            "historical_crispr_only_baseline": float(args.historical_crispr_only_baseline),
            "historical_baseline_for_current_modality": (
                float(args.historical_baseline)
                if args.historical_baseline is not None
                else float(args.historical_mixed_baseline)
                if len(modality_filter_from_arg(str(args.modalities))) > 1
                else float(args.historical_crispr_only_baseline)
                if modality_filter_from_arg(str(args.modalities)) == ["crispr"]
                else None
            ),
            "uses_target_or_link_labels_for_evaluation": False,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / f"{args.run_name}_metrics.json", metrics)

    rows = []
    for scope, payload_metrics in [("combined", metrics), *[(f"within_{k}", v) for k, v in metrics["within_modality"].items()]]:
        for direction, vals in payload_metrics.get("full_gallery", {}).items():
            if isinstance(vals, dict):
                rows.append({"scope": scope, "metric": "full_gallery", "direction": direction, **vals})
        for top_name, top in [("sampled_entity_exact_top10_100", payload_metrics.get("top10_100")), ("sampled_gene_only_top10_100", payload_metrics.get("top10_100_gene_only"))]:
            if not isinstance(top, dict):
                continue
            rows.append(
                {
                    "scope": scope,
                    "metric": top_name,
                    "direction": "gene_to_profile",
                    "Top10_accuracy_mean": top.get("gene_to_profile"),
                    "ratio": top.get("ratio"),
                }
            )
            rows.append(
                {
                    "scope": scope,
                    "metric": top_name,
                    "direction": "profile_to_gene",
                    "Top10_accuracy_mean": top.get("profile_to_gene"),
                    "ratio": top.get("ratio"),
                }
            )
            rows.append(
                {
                    "scope": scope,
                    "metric": top_name,
                    "direction": "bidirectional_mean",
                    "Top10_accuracy_mean": top.get("bidirectional_mean"),
                    "ratio": top.get("ratio"),
                }
            )
    write_dataframe(args.output_dir / f"{args.run_name}_summary.parquet", rows)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    entities, replicates, features, protein_embeddings, splits = load_gene_mocop_data(args)
    structure_embeddings, structure_summary = load_structure_embeddings(args, protein_embeddings.shape[0])
    kg_graph, kg_summary = load_kg_graph(args, protein_embeddings.shape[0])
    entity_features, entity_feature_summary = build_entity_features(entities, args)
    dynamic_token_ids, dynamic_token_summary = build_dynamic_entity_tokens(entities, args)
    profile_adversary_labels, profile_adversary_summary = build_profile_adversary_labels(replicates, features.shape[0], args)
    protein_input_dim = protein_embeddings.shape[1] + (0 if entity_features is None else entity_features.shape[1])
    model = build_model(
        args,
        protein_input_dim,
        features.shape[1],
        structure_embeddings.shape[1] if structure_embeddings is not None else None,
        dynamic_token_summary,
        profile_adversary_summary,
        device,
        kg_graph,
        kg_summary,
        protein_embeddings,
    )
    if not args.eval_only:
        train(
            args,
            model,
            entities,
            replicates,
            features,
            protein_embeddings,
            structure_embeddings,
            structure_summary,
            entity_features,
            entity_feature_summary,
            dynamic_token_ids,
            dynamic_token_summary,
            profile_adversary_labels,
            profile_adversary_summary,
            splits,
            device,
        )
    if not args.train_only:
        if args.eval_only:
            model = build_model(
                args,
                protein_input_dim,
                features.shape[1],
                structure_embeddings.shape[1] if structure_embeddings is not None else None,
                dynamic_token_summary,
                profile_adversary_summary,
                device,
                kg_graph,
                kg_summary,
                protein_embeddings,
            )
        evaluate(
            args,
            model,
            entities,
            replicates,
            features,
            protein_embeddings,
            structure_embeddings,
            structure_summary,
            entity_features,
            entity_feature_summary,
            dynamic_token_ids,
            dynamic_token_summary,
            profile_adversary_summary,
            splits,
            device,
        )


if __name__ == "__main__":
    main()
