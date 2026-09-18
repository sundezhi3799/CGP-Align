from __future__ import annotations

import argparse
import csv
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

import train_cgp_align_replicate as cgp
from cgp_gnn import CGPAlignGNNModel, GraphStore
from train_gene_mocop import GeneMoCoP


DEFAULT_ROWS = [
    ("Full CGP-Align", "full_cgp_align_seed13"),
    ("w/o branch initialization", "wo_branch_initialization_seed13"),
    ("w/o staged training", "wo_staged_training_seed13"),
    ("w/o source-specific profile adapter", "wo_source_specific_profile_adapter_seed13"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate compound-gene bridge alignment by comparing latent C-G similarity "
            "against profile-defined phenotype similarity."
        )
    )
    p.add_argument("--out_root", type=Path, required=True)
    p.add_argument("--checkpoint_dir", type=Path)
    p.add_argument("--output_dir", type=Path)
    p.add_argument(
        "--rows",
        default="",
        help="Comma-separated row_label:run_name pairs. Defaults to the original ablation rows.",
    )
    p.add_argument("--base_checkpoint", type=Path, help="Checkpoint whose config is used to load datasets.")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--top_ks", default="10,50,100")
    p.add_argument("--pair_samples", type=int, default=500000)
    p.add_argument("--eval_batch_size", type=int, default=1024)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--compound_branch_checkpoint", type=Path)
    p.add_argument("--gene_branch_checkpoint", type=Path)
    p.add_argument("--include_independent_direct", action="store_true")
    p.add_argument("--max_compounds", type=int, default=0)
    p.add_argument("--max_genes", type=int, default=0)
    return p.parse_args()


def parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_rows(text: str) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            label, run_name = item.split(":", 1)
            rows.append((label.strip(), run_name.strip()))
        else:
            rows.append((item, item))
    return rows


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


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def namespace_from_config(config: Dict[str, Any]) -> argparse.Namespace:
    cfg = dict(config)
    for key in [
        "compound_data_dir",
        "gene_data_dir",
        "gene_protein_embedding_dir",
        "checkpoint_dir",
        "log_dir",
        "output_dir",
    ]:
        if key in cfg and cfg[key] is not None:
            cfg[key] = Path(cfg[key])
    cfg["init_compound_checkpoint"] = None
    cfg["init_gene_checkpoint"] = None
    cfg.setdefault("profile_source_mode", "compound_gene_modality")
    cfg.setdefault("disable_profile_source_embedding", True)
    cfg.setdefault("profile_source_adapters", False)
    cfg.setdefault("profile_source_dropout", 0.0)
    cfg.setdefault("profile_input_layernorm", False)
    cfg.setdefault("profile_feature_dropout", 0.0)
    cfg.setdefault("profile_mlp_norm", False)
    cfg.setdefault("embed_dim", 256)
    cfg.setdefault("gnn_hidden_dim", 256)
    cfg.setdefault("gnn_layers", 6)
    cfg.setdefault("gene_hidden_dims", "512,256")
    cfg.setdefault("profile_hidden_dims", "512,256")
    cfg.setdefault("modality_context_dim", 32)
    cfg.setdefault("dropout", 0.2)
    cfg.setdefault("eval_batch_size", 1024)
    cfg.setdefault("eval_max_replicates_per_entity", 0)
    cfg.setdefault("profile_norm", "none")
    cfg.setdefault("smoke_test", False)
    return argparse.Namespace(**cfg)


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, eps)


def entity_mean_profiles(
    features: np.ndarray,
    entity_to_reps: Sequence[np.ndarray],
    rows: np.ndarray,
    norm: Dict[str, Any],
) -> np.ndarray:
    out = np.zeros((len(rows), features.shape[1]), dtype=np.float32)
    for i, entity_idx in enumerate(np.asarray(rows, dtype=np.int64).tolist()):
        reps = entity_to_reps[int(entity_idx)]
        if reps.size:
            out[i] = cgp.transform_rows(features, reps, norm).mean(axis=0)
    return out


def choose_rows(data: Dict[str, Any], split: str, max_compounds: int, max_genes: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    c_rows = np.asarray(data["compound"][f"{split}_entities"], dtype=np.int64)
    g_rows = np.asarray(data["gene"][f"{split}_entities"], dtype=np.int64)
    if int(max_compounds) > 0 and len(c_rows) > int(max_compounds):
        c_rows = np.sort(rng.choice(c_rows, size=int(max_compounds), replace=False))
    if int(max_genes) > 0 and len(g_rows) > int(max_genes):
        g_rows = np.sort(rng.choice(g_rows, size=int(max_genes), replace=False))
    return c_rows, g_rows


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(int(k), int(scores.shape[1]))
    if k <= 0:
        return np.zeros((scores.shape[0], 0), dtype=np.int64)
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    part = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-part, axis=1)
    return np.take_along_axis(idx, order, axis=1).astype(np.int64)


def mean_recall_at_k(pred_top: np.ndarray, target_top: np.ndarray, k: int) -> float:
    k = min(int(k), pred_top.shape[1], target_top.shape[1])
    if k <= 0:
        return float("nan")
    vals: List[float] = []
    for pred, target in zip(pred_top[:, :k], target_top[:, :k]):
        vals.append(len(set(pred.tolist()) & set(target.tolist())) / float(k))
    return float(np.mean(vals)) if vals else float("nan")


def ndcg_at_k(latent_top: np.ndarray, relevance: np.ndarray, ideal_top: np.ndarray, k: int) -> float:
    k = min(int(k), latent_top.shape[1], ideal_top.shape[1])
    if k <= 0:
        return float("nan")
    discounts = (1.0 / np.log2(np.arange(k, dtype=np.float64) + 2.0)).astype(np.float32)
    rel = np.maximum(relevance, 0.0).astype(np.float32)
    pred_rel = np.take_along_axis(rel, latent_top[:, :k], axis=1)
    ideal_rel = np.take_along_axis(rel, ideal_top[:, :k], axis=1)
    dcg = (pred_rel * discounts[None, :]).sum(axis=1)
    idcg = (ideal_rel * discounts[None, :]).sum(axis=1)
    valid = idcg > 1e-8
    return float(np.mean(dcg[valid] / idcg[valid])) if bool(valid.any()) else float("nan")


def topk_enrichment(latent_top: np.ndarray, profile_sim: np.ndarray, k: int) -> Dict[str, float]:
    k = min(int(k), latent_top.shape[1])
    top_vals = np.take_along_axis(profile_sim, latent_top[:, :k], axis=1)
    baseline = profile_sim.mean(axis=1)
    std = profile_sim.std(axis=1)
    diff = top_vals.mean(axis=1) - baseline
    return {
        f"top{k}_profile_sim_mean": float(top_vals.mean()),
        f"top{k}_minus_query_mean": float(diff.mean()),
        f"top{k}_z_enrichment": float(np.mean(diff / np.maximum(std, 1e-8))),
    }


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return float((x * y).sum() / denom) if denom > 0 else float("nan")


def rankdata_average(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.stats import rankdata

        return rankdata(x, method="average")
    except Exception:
        return pd.Series(np.asarray(x)).rank(method="average").to_numpy(dtype=np.float64)


def pair_correlations(latent_sim: np.ndarray, profile_sim: np.ndarray, num_samples: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(int(seed))
    n_rows, n_cols = latent_sim.shape
    total = int(n_rows * n_cols)
    n = min(int(num_samples), total)
    flat = rng.integers(0, total, size=n, endpoint=False)
    rr = flat // n_cols
    cc = flat % n_cols
    latent = latent_sim[rr, cc].astype(np.float64)
    profile = profile_sim[rr, cc].astype(np.float64)
    latent_rank = rankdata_average(latent)
    profile_rank = rankdata_average(profile)
    return {
        "pair_samples": int(n),
        "pearson": pearson_corr(latent, profile),
        "spearman": pearson_corr(latent_rank, profile_rank),
    }


def bridge_metrics(
    row_name: str,
    run_name: str,
    latent_c: np.ndarray,
    latent_g: np.ndarray,
    profile_sim: np.ndarray,
    profile_top_cg: np.ndarray,
    profile_top_gc: np.ndarray,
    top_ks: Sequence[int],
    pair_samples: int,
    seed: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    zc = l2_normalize(latent_c)
    zg = l2_normalize(latent_g)
    latent_sim = (zc @ zg.T).astype(np.float32)
    max_k_cg = min(max(top_ks), latent_sim.shape[1])
    max_k_gc = min(max(top_ks), latent_sim.shape[0])
    latent_top_cg = topk_indices(latent_sim, max_k_cg)
    latent_top_gc = topk_indices(latent_sim.T, max_k_gc)
    rel_cg = (profile_sim + 1.0).astype(np.float32)
    rel_gc = (profile_sim.T + 1.0).astype(np.float32)
    rel_cg_centered = (profile_sim - profile_sim.min(axis=1, keepdims=True)).astype(np.float32)
    rel_gc_centered = (profile_sim.T - profile_sim.T.min(axis=1, keepdims=True)).astype(np.float32)
    out: Dict[str, Any] = {
        "row": row_name,
        "run_name": run_name,
        "num_compounds": int(latent_sim.shape[0]),
        "num_genes": int(latent_sim.shape[1]),
        "correlation": pair_correlations(latent_sim, profile_sim, pair_samples, seed),
        "latent_sim_mean": float(latent_sim.mean()),
        "latent_sim_std": float(latent_sim.std()),
        "metadata": metadata or {},
    }
    c2g: Dict[str, Any] = {}
    g2c: Dict[str, Any] = {}
    for k in top_ks:
        kk_cg = min(int(k), latent_sim.shape[1])
        kk_gc = min(int(k), latent_sim.shape[0])
        c2g[f"raw_top{k}_recall@{k}"] = mean_recall_at_k(latent_top_cg, profile_top_cg, kk_cg)
        g2c[f"raw_top{k}_recall@{k}"] = mean_recall_at_k(latent_top_gc, profile_top_gc, kk_gc)
        c2g[f"ndcg@{k}"] = ndcg_at_k(latent_top_cg, rel_cg, profile_top_cg, kk_cg)
        g2c[f"ndcg@{k}"] = ndcg_at_k(latent_top_gc, rel_gc, profile_top_gc, kk_gc)
        c2g[f"centered_ndcg@{k}"] = ndcg_at_k(latent_top_cg, rel_cg_centered, profile_top_cg, kk_cg)
        g2c[f"centered_ndcg@{k}"] = ndcg_at_k(latent_top_gc, rel_gc_centered, profile_top_gc, kk_gc)
    c2g.update(topk_enrichment(latent_top_cg, profile_sim, min(10, max_k_cg)))
    g2c.update(topk_enrichment(latent_top_gc, profile_sim.T, min(10, max_k_gc)))
    out["compound_to_gene"] = c2g
    out["gene_to_compound"] = g2c
    out["mean_ndcg@50"] = float(np.mean([c2g.get("ndcg@50", np.nan), g2c.get("ndcg@50", np.nan)]))
    out["mean_centered_ndcg@50"] = float(
        np.mean([c2g.get("centered_ndcg@50", np.nan), g2c.get("centered_ndcg@50", np.nan)])
    )
    out["mean_raw_top50_recall@50"] = float(
        np.mean([c2g.get("raw_top50_recall@50", np.nan), g2c.get("raw_top50_recall@50", np.nan)])
    )
    c2g_random = 50.0 / float(latent_sim.shape[1]) if latent_sim.shape[1] else float("nan")
    g2c_random = 50.0 / float(latent_sim.shape[0]) if latent_sim.shape[0] else float("nan")
    out["c2g_raw_top50_random_recall"] = c2g_random
    out["g2c_raw_top50_random_recall"] = g2c_random
    out["c2g_raw_top50_lift"] = float(c2g.get("raw_top50_recall@50", np.nan) / c2g_random)
    out["g2c_raw_top50_lift"] = float(g2c.get("raw_top50_recall@50", np.nan) / g2c_random)
    out["mean_raw_top50_lift"] = float(np.mean([out["c2g_raw_top50_lift"], out["g2c_raw_top50_lift"]]))
    out["mean_top10_profile_sim_enrichment"] = float(
        np.mean([c2g.get("top10_minus_query_mean", np.nan), g2c.get("top10_minus_query_mean", np.nan)])
    )
    out["mean_top10_z_enrichment"] = float(
        np.mean([c2g.get("top10_z_enrichment", np.nan), g2c.get("top10_z_enrichment", np.nan)])
    )
    return out


def load_joint_embeddings(
    checkpoint_path: Path,
    base_data: Dict[str, Any],
    c_rows: np.ndarray,
    g_rows: np.ndarray,
    device: torch.device,
    eval_batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload.get("config", {})
    args = namespace_from_config(cfg)
    args.eval_batch_size = int(eval_batch_size)
    model, _ = cgp.build_model(args, base_data, device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    z_c = cgp.encode_compounds(model, base_data["compound"]["graph_store"], c_rows, device, eval_batch_size)
    z_g = cgp.encode_gene_entities(
        model,
        base_data["gene"]["entities"],
        base_data["gene"]["protein_embeddings"],
        base_data["gene"]["modality_ids"],
        g_rows,
        device,
        eval_batch_size,
    )
    return z_c, z_g, {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": payload.get("epoch"),
        "best_score": payload.get("best_score"),
        "branch_init": bool(cfg.get("init_compound_checkpoint") and cfg.get("init_gene_checkpoint")),
        "staged_training": not bool(cfg.get("disable_staged_training", False)),
        "source_adapter": bool(cfg.get("profile_source_adapters", False)),
    }


def load_independent_direct_embeddings(
    compound_checkpoint: Path,
    gene_checkpoint: Path,
    data: Dict[str, Any],
    c_rows: np.ndarray,
    g_rows: np.ndarray,
    device: torch.device,
    eval_batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    c_payload = torch.load(compound_checkpoint, map_location=device, weights_only=False)
    c_cfg = c_payload.get("config", {})
    c_model = CGPAlignGNNModel(
        protein_dim=1,
        profile_dim=int(data["profile_dim"]),
        embed_dim=int(c_cfg.get("embed_dim", 256)),
        gnn_hidden_dim=int(c_cfg.get("gnn_hidden_dim", 256)),
        gnn_layers=int(c_cfg.get("gnn_layers", 6)),
        dropout=float(c_cfg.get("dropout", 0.2)),
        feature_dim_by_source=None,
        protein_hidden_dims=(8,),
        profile_hidden_dim=int(c_cfg.get("profile_hidden_dim", 1024)),
    ).to(device)
    c_model.load_state_dict(c_payload["model_state_dict"], strict=True)
    graph_store = GraphStore(data["compound"]["entities"]["canonical_smiles"].fillna("").astype(str).tolist())
    c_chunks: List[np.ndarray] = []
    c_model.eval()
    with torch.no_grad():
        for start in range(0, len(c_rows), int(eval_batch_size)):
            batch = c_rows[start : start + int(eval_batch_size)]
            z = c_model.encode_compound_graphs(graph_store.get_many(batch.tolist()), device)
            c_chunks.append(z.detach().cpu().numpy().astype(np.float32))
    z_c = np.vstack(c_chunks)

    g_payload = torch.load(gene_checkpoint, map_location=device, weights_only=False)
    g_cfg = g_payload.get("config", {})
    g_model = GeneMoCoP(
        protein_dim=int(data["gene"]["protein_embeddings"].shape[1]),
        profile_dim=int(data["profile_dim"]),
        structure_dim=None,
        protein_hidden_dims=parse_ints(g_cfg.get("protein_hidden_dims", "512,256")),
        profile_hidden_dims=parse_ints(g_cfg.get("profile_hidden_dims", "512,256")),
        embed_dim=int(g_cfg.get("embed_dim", 256)),
        fusion_dim=int(g_cfg.get("fusion_dim", 512)),
        protein_fusion=str(g_cfg.get("protein_fusion", "seq_only")),
        modality_context_dim=int(g_cfg.get("modality_context_dim", 32)),
        dropout=float(g_cfg.get("dropout", 0.2)),
        use_modality_context=not bool(g_cfg.get("disable_modality_context", False)),
        use_profile_source_embedding=not bool(g_cfg.get("disable_profile_source_embedding", False)),
        protein_conditioning=str(g_cfg.get("protein_conditioning", "concat")),
        gene_head=str(g_cfg.get("gene_head", "single")),
    ).to(device)
    g_model.load_state_dict(g_payload["model_state_dict"], strict=True)
    g_chunks: List[np.ndarray] = []
    g_model.eval()
    with torch.no_grad():
        for start in range(0, len(g_rows), int(eval_batch_size)):
            batch = g_rows[start : start + int(eval_batch_size)]
            x = cgp.gene_protein_tensor(data["gene"]["entities"], data["gene"]["protein_embeddings"], batch, device)
            mod = cgp.gene_modality_tensor(data["gene"]["modality_ids"], batch, device)
            z = g_model.encode_protein(x, mod)
            g_chunks.append(z.detach().cpu().numpy().astype(np.float32))
    z_g = np.vstack(g_chunks)
    return z_c, z_g, {
        "compound_checkpoint": str(compound_checkpoint),
        "gene_checkpoint": str(gene_checkpoint),
        "compound_epoch": c_payload.get("epoch"),
        "gene_epoch": g_payload.get("epoch"),
        "shared_latent_training": False,
    }


def flatten_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
    c2g = metrics.get("compound_to_gene", {})
    g2c = metrics.get("gene_to_compound", {})
    corr = metrics.get("correlation", {})
    return {
        "row": metrics.get("row"),
        "run_name": metrics.get("run_name"),
        "checkpoint_epoch": metrics.get("metadata", {}).get("checkpoint_epoch"),
        "pearson": corr.get("pearson"),
        "spearman": corr.get("spearman"),
        "C2G_NDCG@50": c2g.get("ndcg@50"),
        "G2C_NDCG@50": g2c.get("ndcg@50"),
        "mean_NDCG@50": metrics.get("mean_ndcg@50"),
        "C2G_centered_NDCG@50": c2g.get("centered_ndcg@50"),
        "G2C_centered_NDCG@50": g2c.get("centered_ndcg@50"),
        "mean_centered_NDCG@50": metrics.get("mean_centered_ndcg@50"),
        "C2G_RawTop50_Recall@50": c2g.get("raw_top50_recall@50"),
        "G2C_RawTop50_Recall@50": g2c.get("raw_top50_recall@50"),
        "mean_RawTop50_Recall@50": metrics.get("mean_raw_top50_recall@50"),
        "C2G_RawTop50_lift": metrics.get("c2g_raw_top50_lift"),
        "G2C_RawTop50_lift": metrics.get("g2c_raw_top50_lift"),
        "mean_RawTop50_lift": metrics.get("mean_raw_top50_lift"),
        "C2G_Top10_enrichment": c2g.get("top10_minus_query_mean"),
        "G2C_Top10_enrichment": g2c.get("top10_minus_query_mean"),
        "mean_Top10_enrichment": metrics.get("mean_top10_profile_sim_enrichment"),
        "C2G_Top10_z_enrichment": c2g.get("top10_z_enrichment"),
        "G2C_Top10_z_enrichment": g2c.get("top10_z_enrichment"),
        "mean_Top10_z_enrichment": metrics.get("mean_top10_z_enrichment"),
        "latent_sim_mean": metrics.get("latent_sim_mean"),
        "latent_sim_std": metrics.get("latent_sim_std"),
    }


def write_summary(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "crossmodal_bridge_metrics.json", rows)
    flat = [flatten_summary(x) for x in rows]
    fields = list(flat[0].keys()) if flat else []
    with (output_dir / "crossmodal_bridge_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in flat:
            writer.writerow(row)
    lines = [
        "| Row | Spearman | Pearson | Mean centered NDCG@50 | C2G R@50 lift | G2C R@50 lift | Mean R@50 lift | C2G R@50 | G2C R@50 | Top10 z-enrich |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in flat:
        lines.append(
            "| {row} | {sp} | {pe} | {mcndcg} | {clift} | {glift} | {mlift} | {cr} | {gr} | {zen} |".format(
                row=row["row"],
                sp=fmt(row.get("spearman")),
                pe=fmt(row.get("pearson")),
                mcndcg=fmt(row.get("mean_centered_NDCG@50")),
                clift=fmt(row.get("C2G_RawTop50_lift")),
                glift=fmt(row.get("G2C_RawTop50_lift")),
                mlift=fmt(row.get("mean_RawTop50_lift")),
                cr=fmt(row.get("C2G_RawTop50_Recall@50")),
                gr=fmt(row.get("G2C_RawTop50_Recall@50")),
                zen=fmt(row.get("mean_Top10_z_enrichment")),
            )
        )
    (output_dir / "crossmodal_bridge_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def main() -> None:
    args = parse_args()
    device = cgp.select_device(args.device)
    checkpoint_dir = args.checkpoint_dir or (args.out_root / "checkpoints")
    output_dir = args.output_dir or (args.out_root / "crossmodal_bridge")
    top_ks = parse_ints(args.top_ks)
    max_k = max(top_ks)
    eval_rows = parse_rows(args.rows) if str(args.rows).strip() else list(DEFAULT_ROWS)

    base_ckpt = args.base_checkpoint
    if base_ckpt is None:
        base_ckpt = checkpoint_dir / eval_rows[0][1] / "best_model.pt"
    if not base_ckpt.exists():
        raise SystemExit(f"Missing base checkpoint for dataset config: {base_ckpt}")
    base_cfg = torch.load(base_ckpt, map_location="cpu", weights_only=False).get("config", {})
    base_args = namespace_from_config(base_cfg)
    base_args.smoke_test = False
    data = cgp.build_data(base_args)
    c_rows, g_rows = choose_rows(data, args.split, args.max_compounds, args.max_genes, args.seed)

    compound_profile = entity_mean_profiles(
        data["compound"]["features"],
        data["compound"]["entity_to_reps"],
        c_rows,
        data["compound"]["norm"],
    )
    gene_profile = entity_mean_profiles(
        data["gene"]["features"],
        data["gene"]["entity_to_reps"],
        g_rows,
        data["gene"]["norm"],
    )
    compound_profile = l2_normalize(compound_profile)
    gene_profile = l2_normalize(gene_profile)
    profile_sim = (compound_profile @ gene_profile.T).astype(np.float32)
    profile_top_cg = topk_indices(profile_sim, min(max_k, profile_sim.shape[1]))
    profile_top_gc = topk_indices(profile_sim.T, min(max_k, profile_sim.shape[0]))

    rows: List[Dict[str, Any]] = []
    for row_name, run_name in eval_rows:
        ckpt = checkpoint_dir / run_name / "best_model.pt"
        if not ckpt.exists():
            rows.append({"row": row_name, "run_name": run_name, "status": "missing", "checkpoint": str(ckpt)})
            continue
        z_c, z_g, meta = load_joint_embeddings(ckpt, data, c_rows, g_rows, device, args.eval_batch_size)
        metrics = bridge_metrics(
            row_name,
            run_name,
            z_c,
            z_g,
            profile_sim,
            profile_top_cg,
            profile_top_gc,
            top_ks,
            args.pair_samples,
            args.seed + len(rows) * 100,
            meta,
        )
        rows.append(metrics)
        write_summary(output_dir, rows)
        print(json.dumps({"completed": row_name, "spearman": metrics["correlation"]["spearman"]}, sort_keys=True), flush=True)

    if args.include_independent_direct:
        if not args.compound_branch_checkpoint or not args.gene_branch_checkpoint:
            raise SystemExit("--include_independent_direct requires both branch checkpoint paths.")
        z_c, z_g, meta = load_independent_direct_embeddings(
            args.compound_branch_checkpoint,
            args.gene_branch_checkpoint,
            data,
            c_rows,
            g_rows,
            device,
            args.eval_batch_size,
        )
        rows.append(
            bridge_metrics(
                "independent branch direct latent",
                "independent_branch_direct_seed13",
                z_c,
                z_g,
                profile_sim,
                profile_top_cg,
                profile_top_gc,
                top_ks,
                args.pair_samples,
                args.seed + 900,
                meta,
            )
        )
        print(json.dumps({"completed": "independent branch direct latent"}, sort_keys=True), flush=True)

    rows.append(
        bridge_metrics(
            "profile-only bridge oracle",
            "profile_only_bridge_oracle",
            compound_profile,
            gene_profile,
            profile_sim,
            profile_top_cg,
            profile_top_gc,
            top_ks,
            args.pair_samples,
            args.seed + 1000,
            {"shared_latent_training": False, "description": "Uses raw replicate-mean profile cosine as both query and relevance."},
        )
    )
    write_summary(output_dir, rows)
    print(json.dumps({"output_dir": str(output_dir), "rows": len(rows)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
