from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from cgp_common import append_jsonl, metrics_from_ranks, set_seed, tensor_from_numpy, write_json
from cgp_gnn import CGPAlignGNNModel, GraphStore
from cgp_align.models.modules import SharedProfileEncoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compound replicate MoCoP-style baseline with replicate-gallery evaluation.")
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--checkpoint_dir", type=Path, required=True)
    p.add_argument("--log_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--split_name", default="cold_compound", choices=["cold_compound", "random_entity"])
    p.add_argument("--profile_norm", default="none", choices=["none", "train_zscore"])
    p.add_argument("--train_profile_mode", default="random_replicate", choices=["random_replicate", "mean_profile"])
    p.add_argument("--eval_profile_mode", default="replicate_gallery", choices=["entity_mean", "replicate_gallery"])
    p.add_argument("--eval_max_replicates_per_entity", type=int, default=0)
    p.add_argument("--train_fraction", type=float, default=1.0, help="Fraction of eligible train entities to use.")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--gnn_hidden_dim", type=int, default=256)
    p.add_argument("--gnn_layers", type=int, default=6)
    p.add_argument("--profile_hidden_dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--steps_per_epoch", type=int, default=0)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--val_every", type=int, default=10)
    p.add_argument("--val_max_queries", type=int, default=2048)
    p.add_argument("--val_negative_ratio", type=int, default=100)
    p.add_argument("--val_repeats", type=int, default=5)
    p.add_argument("--negative_ratios", default="100")
    p.add_argument("--num_repeats", type=int, default=10)
    p.add_argument("--eval_batch_size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--train_only", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    return p.parse_args()


def parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


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


def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Dict[str, Any]]:
    paths = [
        data_dir / "compound_mocop_entities.parquet",
        data_dir / "compound_mocop_replicates.parquet",
        data_dir / "compound_mocop_replicate_features.npy",
        data_dir / "splits_compound_mocop.json",
    ]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit("Missing required compound replicate files:\n" + "\n".join(missing))
    entities = pd.read_parquet(paths[0]).reset_index(drop=True)
    reps = pd.read_parquet(paths[1]).reset_index(drop=True)
    features = np.load(paths[2]).astype(np.float32)
    splits = json.loads(paths[3].read_text(encoding="utf-8"))
    return entities, reps, features, splits


def build_entity_replicate_index(replicates: pd.DataFrame, num_entities: int) -> List[np.ndarray]:
    out: List[List[int]] = [[] for _ in range(num_entities)]
    for row in replicates.itertuples(index=False):
        out[int(row.entity_index)].append(int(row.feature_index))
    return [np.asarray(x, dtype=np.int64) for x in out]


def entity_key_codes(entities: pd.DataFrame) -> np.ndarray:
    key_col = "entity_id" if "entity_id" in entities.columns else "compound_id"
    codes, _ = pd.factorize(entities[key_col].astype(str), sort=True)
    return codes.astype(np.int64)


def eligible_entities(entities: pd.DataFrame, rows: Sequence[int], entity_to_reps: Sequence[np.ndarray]) -> np.ndarray:
    idx = np.asarray(rows, dtype=np.int64)
    if idx.size == 0:
        return idx
    has_structure = entities.iloc[idx]["has_structure"].astype(bool).to_numpy() if "has_structure" in entities.columns else np.ones(len(idx), dtype=bool)
    has_reps = np.asarray([entity_to_reps[int(i)].size > 0 for i in idx], dtype=bool)
    smiles_ok = entities.iloc[idx]["canonical_smiles"].fillna("").astype(str).str.len().to_numpy() > 0
    return idx[has_structure & has_reps & smiles_ok]


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
    x = features[rows].astype(np.float32).copy()
    if norm.get("mode") == "train_zscore":
        x = (x - norm["mean"]) / norm["std"]
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def sample_replicates(entity_rows: np.ndarray, entity_to_reps: Sequence[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    out = np.zeros(len(entity_rows), dtype=np.int64)
    for i, e in enumerate(entity_rows.astype(int).tolist()):
        reps = entity_to_reps[e]
        if reps.size == 0:
            raise RuntimeError(f"Entity {e} has no replicate features.")
        out[i] = int(rng.choice(reps))
    return out


def compute_entity_mean_profiles(features: np.ndarray, entity_to_reps: Sequence[np.ndarray], norm: Dict[str, Any]) -> np.ndarray:
    out = np.zeros((len(entity_to_reps), features.shape[1]), dtype=np.float32)
    for i, reps in enumerate(entity_to_reps):
        if reps.size:
            out[i] = transform_rows(features, reps, norm).mean(axis=0)
    return out


def source_ids(n: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(int(n), dtype=torch.long, device=device)


def build_model(profile_dim: int, args: argparse.Namespace, device: torch.device) -> CGPAlignGNNModel:
    model = CGPAlignGNNModel(
        protein_dim=1,
        profile_dim=profile_dim,
        embed_dim=args.embed_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_layers=args.gnn_layers,
        dropout=args.dropout,
        feature_dim_by_source=None,
        protein_hidden_dims=(8,),
        profile_hidden_dim=args.profile_hidden_dim,
    ).to(device)
    model.profile_encoder = SharedProfileEncoder(profile_dim, (512,), args.embed_dim, args.dropout, use_source_embedding=False).to(device)
    return model


def multipositive_loss(logits: torch.Tensor, positive_code: torch.Tensor) -> torch.Tensor:
    n = int(logits.shape[0])
    if n <= 1:
        labels = torch.arange(n, device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    pos = positive_code[:, None].eq(positive_code[None, :])
    target = pos.to(logits.dtype) / pos.to(logits.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
    target_t = target.T / target.T.sum(dim=1, keepdim=True).clamp_min(1e-8)
    row = -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    col = -(target_t * F.log_softmax(logits.T, dim=1)).sum(dim=1).mean()
    return 0.5 * (row + col)


def encode_compounds(model: CGPAlignGNNModel, graph_store: GraphStore, rows: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            z = model.encode_compound_graphs(graph_store.get_many(batch.tolist()), device)
            chunks.append(z.detach().cpu().numpy().astype(np.float32))
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(chunks)


def encode_profiles(model: CGPAlignGNNModel, features: np.ndarray, feature_rows: np.ndarray, norm: Dict[str, Any], device: torch.device, batch_size: int) -> np.ndarray:
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(feature_rows), batch_size):
            rows = feature_rows[start : start + batch_size]
            x = tensor_from_numpy(transform_rows(features, rows, norm), device)
            z = model.encode_profile(x, source_ids(len(rows), device))
            chunks.append(z.detach().cpu().numpy().astype(np.float32))
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(chunks)


def build_replicate_gallery(entity_rows: np.ndarray, entity_to_reps: Sequence[np.ndarray], entity_codes: np.ndarray, cap: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed) + 2203)
    owners: List[int] = []
    feat: List[int] = []
    for e in np.asarray(entity_rows, dtype=np.int64).tolist():
        reps = np.asarray(entity_to_reps[int(e)], dtype=np.int64)
        if reps.size == 0:
            continue
        if int(cap) > 0 and reps.size > int(cap):
            reps = np.asarray(rng.choice(reps, size=int(cap), replace=False), dtype=np.int64)
        owners.extend([int(e)] * int(reps.size))
        feat.extend([int(x) for x in reps.tolist()])
    owners_np = np.asarray(owners, dtype=np.int64)
    return {
        "owner_entities": owners_np,
        "feature_rows": np.asarray(feat, dtype=np.int64),
        "entity_codes": entity_codes[owners_np].astype(np.int64) if owners_np.size else np.zeros(0, dtype=np.int64),
    }


def full_gallery_metrics_general(query: np.ndarray, gallery: np.ndarray, query_code: np.ndarray, gallery_code: np.ndarray) -> Dict[str, Any]:
    if len(query) == 0 or len(gallery) == 0:
        return metrics_from_ranks([])
    scores = query.astype(np.float32) @ gallery.astype(np.float32).T
    ranks = []
    gaps = []
    eps = 1e-8
    gallery_code = np.asarray(gallery_code, dtype=np.int64)
    query_code = np.asarray(query_code, dtype=np.int64)
    for i, row in enumerate(scores):
        pos = gallery_code == query_code[i]
        if not np.any(pos):
            continue
        p = float(np.max(row[pos]))
        ranks.append(int(1 + np.sum(row[~pos] > p + eps)))
        if np.any(~pos):
            gaps.append(float(p - np.max(row[~pos])))
    return metrics_from_ranks(ranks, gaps)


def sampled_one_direction(query: np.ndarray, gallery: np.ndarray, query_code: np.ndarray, gallery_code: np.ndarray, ratio: int, repeats: int, seed: int) -> Dict[str, Any]:
    gallery_idx = np.arange(len(gallery), dtype=np.int64)
    query_code = np.asarray(query_code, dtype=np.int64)
    gallery_code = np.asarray(gallery_code, dtype=np.int64)
    reps = []
    for r in range(int(repeats)):
        rng = np.random.default_rng(int(seed) + r * 100003 + int(ratio) * 17)
        ranks = []
        n_pos = []
        n_neg = []
        for i in range(len(query)):
            pos_pool = gallery_idx[gallery_code == query_code[i]]
            if pos_pool.size == 0:
                continue
            neg_pool = gallery_idx[gallery_code != query_code[i]]
            k = min(int(ratio), len(neg_pool))
            if k <= 0:
                continue
            neg = rng.choice(neg_pool, size=k, replace=False)
            cand = np.concatenate([pos_pool, neg])
            scores = query[i : i + 1].astype(np.float32) @ gallery[cand].astype(np.float32).T
            row = scores[0]
            best_pos = float(np.max(row[: pos_pool.size]))
            ranks.append(int(1 + np.sum(row[pos_pool.size :] > best_pos)))
            n_pos.append(int(pos_pool.size))
            n_neg.append(int(k))
        arr = np.asarray(ranks, dtype=np.float64)
        reps.append({
            "num_queries": int(arr.size),
            "negative_ratio": int(ratio),
            "candidate_count_mean": float(np.mean(np.asarray(n_pos) + np.asarray(n_neg))) if n_pos else 0.0,
            "positive_count_mean": float(np.mean(n_pos)) if n_pos else 0.0,
            "Top1_accuracy": float(np.mean(arr <= 1)) if arr.size else None,
            "Top5_accuracy": float(np.mean(arr <= 5)) if arr.size else None,
            "Top10_accuracy": float(np.mean(arr <= 10)) if arr.size else None,
            "MRR": float(np.mean(1.0 / arr)) if arr.size else None,
            "median_rank": float(np.median(arr)) if arr.size else None,
        })
    out: Dict[str, Any] = {"num_repeats": int(repeats), "repeats": reps}
    for key in ["Top1_accuracy", "Top5_accuracy", "Top10_accuracy", "MRR", "median_rank"]:
        vals = np.asarray([x[key] for x in reps if x[key] is not None], dtype=np.float64)
        out[f"{key}_mean"] = float(vals.mean()) if vals.size else None
        out[f"{key}_std"] = float(vals.std(ddof=0)) if vals.size else None
    return out


def sampled_metrics(z_c: np.ndarray, z_p: np.ndarray, c_codes: np.ndarray, p_codes: np.ndarray, ratios: Sequence[int], repeats: int, seed: int) -> Dict[str, Any]:
    out = {"compound_to_profile": {}, "profile_to_compound": {}, "positive_label": "entity_key"}
    for ratio in ratios:
        out["compound_to_profile"][f"1:{ratio}"] = sampled_one_direction(z_c, z_p, c_codes, p_codes, ratio, repeats, seed)
        out["profile_to_compound"][f"1:{ratio}"] = sampled_one_direction(z_p, z_c, p_codes, c_codes, ratio, repeats, seed)
    return out


def bidirectional_top10(metrics: Dict[str, Any], ratio: int = 100) -> Dict[str, Any]:
    key = f"1:{int(ratio)}"
    s = metrics.get("mocop_protocol_sampled", {})
    c2p = s.get("compound_to_profile", {}).get(key, {}).get("Top10_accuracy_mean")
    p2c = s.get("profile_to_compound", {}).get(key, {}).get("Top10_accuracy_mean")
    vals = [float(x) for x in [c2p, p2c] if x is not None]
    return {"ratio": key, "compound_to_profile": c2p, "profile_to_compound": p2c, "bidirectional_mean": float(np.mean(vals)) if vals else None}


def evaluate_rows(
    model: CGPAlignGNNModel,
    graph_store: GraphStore,
    features: np.ndarray,
    entity_rows: np.ndarray,
    entity_to_reps: Sequence[np.ndarray],
    entity_codes: np.ndarray,
    norm: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    repeats: int,
    ratios: Sequence[int],
) -> Dict[str, Any]:
    z_c = encode_compounds(model, graph_store, entity_rows, device, args.eval_batch_size)
    c_codes = entity_codes[entity_rows]
    if args.eval_profile_mode == "entity_mean":
        mean_profiles = compute_entity_mean_profiles(features, entity_to_reps, norm)[entity_rows]
        chunks = []
        with torch.no_grad():
            for start in range(0, len(entity_rows), args.eval_batch_size):
                x = tensor_from_numpy(mean_profiles[start : start + args.eval_batch_size], device)
                chunks.append(model.encode_profile(x, source_ids(x.shape[0], device)).detach().cpu().numpy().astype(np.float32))
        z_p = np.vstack(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)
        p_codes = c_codes
        num_profiles = len(entity_rows)
    else:
        gallery = build_replicate_gallery(entity_rows, entity_to_reps, entity_codes, args.eval_max_replicates_per_entity, args.seed)
        z_p = encode_profiles(model, features, gallery["feature_rows"], norm, device, args.eval_batch_size)
        p_codes = gallery["entity_codes"]
        num_profiles = len(p_codes)
    out: Dict[str, Any] = {
        "profile_eval_mode": str(args.eval_profile_mode),
        "num_entities": int(len(entity_rows)),
        "num_profile_replicates": int(num_profiles),
        "num_entity_keys": int(len(set(c_codes.astype(int).tolist()))),
        "full_gallery": {
            "compound_to_profile": full_gallery_metrics_general(z_c, z_p, c_codes, p_codes),
            "profile_to_compound": full_gallery_metrics_general(z_p, z_c, p_codes, c_codes),
        },
        "mocop_protocol_sampled": sampled_metrics(z_c, z_p, c_codes, p_codes, ratios, repeats, args.seed),
    }
    out["top10_100"] = bidirectional_top10(out, 100)
    return out


def train(args: argparse.Namespace, model: CGPAlignGNNModel, entities: pd.DataFrame, reps: pd.DataFrame, features: np.ndarray, splits: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    entity_to_reps = build_entity_replicate_index(reps, len(entities))
    codes = entity_key_codes(entities)
    train_entities = eligible_entities(entities, splits[args.split_name]["train"], entity_to_reps)
    val_entities = eligible_entities(entities, splits[args.split_name]["val"], entity_to_reps)
    test_entities = eligible_entities(entities, splits[args.split_name]["test"], entity_to_reps)
    full_train_entities = int(len(train_entities))
    train_entities = subsample_train_entities(train_entities, float(args.train_fraction), int(args.seed) + 101)
    if args.smoke_test:
        train_entities = train_entities[: min(2048, len(train_entities))]
        val_entities = val_entities[: min(256, len(val_entities))]
        test_entities = test_entities[: min(256, len(test_entities))]
        args.epochs = min(args.epochs, 2)
    norm = fit_profile_normalizer(features, entity_to_reps, train_entities, args.profile_norm)
    mean_profiles: Optional[np.ndarray] = compute_entity_mean_profiles(features, entity_to_reps, norm) if args.train_profile_mode == "mean_profile" else None
    graph_store = GraphStore(entities["canonical_smiles"].fillna("").astype(str).tolist())

    ckpt_dir = args.checkpoint_dir / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"{args.run_name}_train_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    config = {
        **vars(args),
        "num_train_entities": int(len(train_entities)),
        "num_full_train_entities": int(full_train_entities),
        "train_fraction_requested": float(args.train_fraction),
        "train_fraction_actual": float(len(train_entities) / max(1, full_train_entities)),
        "num_val_entities": int(len(val_entities)),
        "num_test_entities": int(len(test_entities)),
        "profile_dim": int(features.shape[1]),
        "profile_normalizer": {"mode": norm.get("mode"), "fit_split": norm.get("fit_split")},
        "training_profile_sampling": str(args.train_profile_mode),
        "eval_profile_mode": str(args.eval_profile_mode),
        "backbone": "GGNN compound encoder + profile MLP encoder",
        "positive_label": "compound entity_key / InChIKey; all replicate profiles of the same entity are positives",
        "strict_cold_compound": True,
    }
    write_json(ckpt_dir / "config.json", config)
    write_json(ckpt_dir / "split_guard.json", {
        "train_val_overlap": int(len(set(train_entities.tolist()) & set(val_entities.tolist()))),
        "train_test_overlap": int(len(set(train_entities.tolist()) & set(test_entities.tolist()))),
        "val_test_overlap": int(len(set(val_entities.tolist()) & set(test_entities.tolist()))),
        "passed": not (set(train_entities.tolist()) & set(val_entities.tolist()) or set(train_entities.tolist()) & set(test_entities.tolist()) or set(val_entities.tolist()) & set(test_entities.tolist())),
    })

    rng = np.random.default_rng(args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    steps = int(args.steps_per_epoch or max(1, np.ceil(len(train_entities) / max(1, args.batch_size))))
    best = -float("inf")
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: List[float] = []
        for _ in range(steps):
            batch = rng.choice(train_entities, size=min(args.batch_size, len(train_entities)), replace=False)
            code = torch.from_numpy(codes[batch].astype(np.int64)).long().to(device)
            with autocast(enabled=bool(args.amp and device.type == "cuda")):
                z_c = model.encode_compound_graphs(graph_store.get_many(batch.tolist()), device)
                if mean_profiles is None:
                    rep_rows = sample_replicates(batch, entity_to_reps, rng)
                    profile_x = tensor_from_numpy(transform_rows(features, rep_rows, norm), device)
                else:
                    profile_x = tensor_from_numpy(mean_profiles[batch], device)
                z_p = model.encode_profile(profile_x, source_ids(len(batch), device))
                logits = z_c @ z_p.T / float(args.temperature)
                loss = multipositive_loss(logits, code)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu().item()))
        rec: Dict[str, Any] = {"epoch": int(epoch), "train_loss": float(np.mean(losses)) if losses else None, "steps_per_epoch": int(steps)}
        if epoch % int(args.val_every) == 0 or epoch == int(args.epochs):
            val_rows = val_entities
            if args.val_max_queries > 0 and len(val_rows) > args.val_max_queries:
                val_rows = rng.choice(val_rows, size=args.val_max_queries, replace=False)
            val = evaluate_rows(model, graph_store, features, val_rows, entity_to_reps, codes, norm, args, device, args.val_repeats, [args.val_negative_ratio])
            top = val["top10_100"]
            rec.update({
                "val_mocop_alignment_score": top.get("bidirectional_mean"),
                "val_mocop_c2profile_Top10": top.get("compound_to_profile"),
                "val_mocop_profile2c_Top10": top.get("profile_to_compound"),
                "val_num_profile_replicates": val.get("num_profile_replicates"),
            })
            if rec["val_mocop_alignment_score"] is not None and float(rec["val_mocop_alignment_score"]) > best:
                best = float(rec["val_mocop_alignment_score"])
                torch.save({"model_state_dict": model.state_dict(), "epoch": int(epoch), "best_score": best, "config": config, "profile_normalizer": norm}, ckpt_dir / "best_model.pt")
        append_jsonl(log_path, rec)
        print(json.dumps(rec, sort_keys=True), flush=True)
    if not (ckpt_dir / "best_model.pt").exists():
        torch.save({"model_state_dict": model.state_dict(), "epoch": int(args.epochs), "best_score": best, "config": config, "profile_normalizer": norm}, ckpt_dir / "best_model.pt")
    return {"norm": norm, "entity_to_reps": entity_to_reps, "codes": codes, "test_entities": test_entities}


def evaluate(args: argparse.Namespace, model: CGPAlignGNNModel, entities: pd.DataFrame, reps: pd.DataFrame, features: np.ndarray, splits: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    ckpt = args.checkpoint_dir / args.run_name / "best_model.pt"
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    norm = payload.get("profile_normalizer", {"mode": "none"})
    entity_to_reps = build_entity_replicate_index(reps, len(entities))
    codes = entity_key_codes(entities)
    test_entities = eligible_entities(entities, splits[args.split_name]["test"], entity_to_reps)
    if args.smoke_test:
        test_entities = test_entities[: min(256, len(test_entities))]
    graph_store = GraphStore(entities["canonical_smiles"].fillna("").astype(str).tolist())
    metrics = evaluate_rows(model, graph_store, features, test_entities, entity_to_reps, codes, norm, args, device, args.num_repeats, parse_ints(args.negative_ratios))
    metrics.update({
        "run_name": args.run_name,
        "data_dir": str(args.data_dir),
        "split_name": args.split_name,
        "query_split": "test",
        "checkpoint_epoch": payload.get("epoch"),
        "best_score": payload.get("best_score"),
        "profile_normalizer": {"mode": norm.get("mode"), "fit_split": norm.get("fit_split")},
        "profile_eval_mode": str(args.eval_profile_mode),
        "training_profile_sampling": str(args.train_profile_mode),
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / f"{args.run_name}_metrics.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    entities, reps, features, splits = load_data(args.data_dir)
    model = build_model(int(features.shape[1]), args, device)
    if not args.eval_only:
        train(args, model, entities, reps, features, splits, device)
    if not args.train_only:
        if args.eval_only:
            model = build_model(int(features.shape[1]), args, device)
        evaluate(args, model, entities, reps, features, splits, device)


if __name__ == "__main__":
    main()
