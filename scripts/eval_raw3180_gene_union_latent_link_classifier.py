from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import train_cgp_align_replicate as cgp
from eval_cgp_align_crossmodal_bridge import namespace_from_config
from eval_cgp_align_target_enrichment_replicate import parse_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frozen CGP latent target-edge classifier with ORF/CRISPR collapsed by gene symbol.")
    p.add_argument("--rows", required=True)
    p.add_argument("--target_edges", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--source", default="all")
    p.add_argument(
        "--within_sources",
        default="",
        help="Comma-separated source tokens. If set, evaluate random/cold splits inside each source-specific edge set.",
    )
    p.add_argument(
        "--source_filter_before_dedup",
        action="store_true",
        help="Filter source-specific target rows before compound-gene pair de-duplication.",
    )
    p.add_argument("--split_modes", default="random_edge,cold_compound,cold_gene")
    p.add_argument("--holdout_sources", default="")
    p.add_argument("--gene_modalities", default="orf,crispr")
    p.add_argument("--negative_ratio", type=int, default=10)
    p.add_argument("--max_train_pos", type=int, default=30000)
    p.add_argument("--max_eval_pos", type=int, default=10000)
    p.add_argument("--eval_batch_size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--device", default="auto")
    p.add_argument("--only_usable_typed_prior", action="store_true")
    return p.parse_args()


def split_tokens(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {str(x).strip().lower() for x in str(value).split("|") if str(x).strip()}


def source_mask(df: pd.DataFrame, source: str) -> pd.Series:
    source = str(source).strip().lower()
    if source in {"", "all", "all_broad"}:
        return pd.Series(True, index=df.index)
    if source == "typed_usable":
        if "usable_for_typed_prior" not in df.columns:
            raise SystemExit("source=typed_usable requires usable_for_typed_prior")
        return df["usable_for_typed_prior"].astype(bool)
    src = df.get("source_database", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str).str.lower()
    ids = df.get("source_id_types", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str).str.lower()
    return src.str.contains(source) | ids.str.contains(source)


def normalize_targets(path: Path, only_usable_typed_prior: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if only_usable_typed_prior:
        if "usable_for_typed_prior" not in df.columns:
            raise SystemExit("--only_usable_typed_prior requires usable_for_typed_prior")
        df = df[df["usable_for_typed_prior"].astype(bool)].copy()
    compound_col = "compound_id" if "compound_id" in df.columns else "inchikey"
    gene_col = "gene_id" if "gene_id" in df.columns else "gene_symbol"
    out = df.copy()
    out["compound_key"] = out[compound_col].fillna("").astype(str)
    out["gene_symbol_key"] = out[gene_col].fillna("").astype(str).str.upper()
    out = out[(out["compound_key"] != "") & (out["gene_symbol_key"] != "")]
    return out.reset_index(drop=True)


def dedup_pairs(edges: pd.DataFrame) -> pd.DataFrame:
    return edges.drop_duplicates(["compound_key", "gene_symbol_key"]).reset_index(drop=True)


def load_targets(path: Path, only_usable_typed_prior: bool = False) -> pd.DataFrame:
    return dedup_pairs(normalize_targets(path, only_usable_typed_prior))


def all_rows(data: Dict[str, Any], branch: str) -> np.ndarray:
    return np.unique(
        np.concatenate(
            [
                np.asarray(data[branch]["train_entities"], dtype=np.int64),
                np.asarray(data[branch]["val_entities"], dtype=np.int64),
                np.asarray(data[branch]["test_entities"], dtype=np.int64),
            ]
        )
    )


def filter_gene_rows(data: Dict[str, Any], rows: np.ndarray, modalities: List[str]) -> np.ndarray:
    allowed = {str(x).lower() for x in modalities}
    vals = (
        data["gene"]["entities"].iloc[np.asarray(rows, dtype=np.int64)]["perturbation_modality"]
        .fillna("")
        .astype(str)
        .str.lower()
        .to_numpy()
    )
    return np.asarray(rows, dtype=np.int64)[np.asarray([x in allowed for x in vals], dtype=bool)]


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def load_model_gene_union_embeddings(
    checkpoint: Path,
    device: torch.device,
    eval_batch_size: int,
    modalities: List[str],
) -> Tuple[Dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray, List[str], np.ndarray]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = namespace_from_config(payload.get("config", {}))
    args.eval_batch_size = int(eval_batch_size)
    args.smoke_test = False
    data = cgp.build_data(args)
    model, _ = cgp.build_model(args, data, device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    c_rows = all_rows(data, "compound")
    g_rows = filter_gene_rows(data, all_rows(data, "gene"), modalities)
    z_c = cgp.encode_compounds(model, data["compound"]["graph_store"], c_rows, device, eval_batch_size)
    z_g_entity = cgp.encode_gene_entities(
        model,
        data["gene"]["entities"],
        data["gene"]["protein_embeddings"],
        data["gene"]["modality_ids"],
        g_rows,
        device,
        eval_batch_size,
    )
    symbols: Dict[str, List[int]] = {}
    for local_idx, row in enumerate(np.asarray(g_rows, dtype=np.int64).tolist()):
        sym = str(data["gene"]["entities"].iloc[int(row)].get("gene_symbol", "")).upper()
        if sym:
            symbols.setdefault(sym, []).append(int(local_idx))
    gene_symbols = sorted(symbols)
    z_gene = np.vstack([z_g_entity[symbols[sym]].mean(axis=0) for sym in gene_symbols]).astype(np.float32)
    z_gene = l2_normalize(z_gene).astype(np.float32)
    c_entities = data["compound"]["entities"].copy()
    return payload, c_entities, c_rows, z_c.astype(np.float32), gene_symbols, z_gene


def compound_key_to_idx(c_entities: pd.DataFrame, c_rows: np.ndarray) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for local_idx, row in enumerate(np.asarray(c_rows, dtype=np.int64).tolist()):
        ent = c_entities.iloc[int(row)]
        for key in {str(ent.get("compound_id", "")), str(ent.get("inchikey", "")), str(ent.get("entity_id", ""))}:
            if key and key not in out:
                out[key] = int(local_idx)
    return out


def mapped_pairs(edges: pd.DataFrame, c_entities: pd.DataFrame, gene_symbols: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    c_rows = np.arange(len(c_entities), dtype=np.int64)
    c_lookup = compound_key_to_idx(c_entities, c_rows)
    g_lookup = {sym: i for i, sym in enumerate(gene_symbols)}
    pairs: set[Tuple[int, int]] = set()
    mapped_edges = 0
    for rec in edges[["compound_key", "gene_symbol_key"]].drop_duplicates().itertuples(index=False):
        ci = c_lookup.get(str(rec.compound_key))
        gi = g_lookup.get(str(rec.gene_symbol_key).upper())
        if ci is None or gi is None:
            continue
        mapped_edges += 1
        pairs.add((int(ci), int(gi)))
    arr = np.asarray(sorted(pairs), dtype=np.int64)
    return arr, {"mapped_edges": int(mapped_edges), "positive_pairs": int(len(arr))}


def mapped_pairs_with_compound_rows(
    edges: pd.DataFrame,
    c_entities: pd.DataFrame,
    c_rows: np.ndarray,
    gene_symbols: List[str],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    c_lookup = compound_key_to_idx(c_entities, c_rows)
    g_lookup = {sym: i for i, sym in enumerate(gene_symbols)}
    pairs: set[Tuple[int, int]] = set()
    mapped_edges = 0
    for rec in edges[["compound_key", "gene_symbol_key"]].drop_duplicates().itertuples(index=False):
        ci = c_lookup.get(str(rec.compound_key))
        gi = g_lookup.get(str(rec.gene_symbol_key).upper())
        if ci is None or gi is None:
            continue
        mapped_edges += 1
        pairs.add((int(ci), int(gi)))
    arr = np.asarray(sorted(pairs), dtype=np.int64)
    return arr, {"mapped_edges": int(mapped_edges), "positive_pairs": int(len(arr))}


def split_pairs(pos: np.ndarray, mode: str, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(len(pos))
    if n < 10:
        raise ValueError(f"Not enough positive pairs for {mode}: {n}")
    if mode == "random_edge":
        order = rng.permutation(n)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        return pos[order[:train_end]], pos[order[train_end:val_end]], pos[order[val_end:]]
    axis = 0 if mode == "cold_compound" else 1
    keys = np.unique(pos[:, axis])
    keys = keys[rng.permutation(len(keys))]
    train_keys = set(keys[: int(0.7 * len(keys))].tolist())
    val_keys = set(keys[int(0.7 * len(keys)) : int(0.85 * len(keys))].tolist())
    m_train = np.asarray([x in train_keys for x in pos[:, axis]])
    m_val = np.asarray([x in val_keys for x in pos[:, axis]])
    m_test = ~(m_train | m_val)
    return pos[m_train], pos[m_val], pos[m_test]


def source_holdout_pairs(
    edges: pd.DataFrame,
    token: str,
    c_entities: pd.DataFrame,
    c_rows: np.ndarray,
    gene_symbols: List[str],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    token = str(token).lower()
    if "source_database" not in edges.columns:
        raise ValueError("source_holdout requires source_database column")
    has_token = edges["source_database"].map(lambda x: token in split_tokens(x))
    train_edges = edges[~has_token].drop_duplicates(["compound_key", "gene_symbol_key"]).reset_index(drop=True)
    test_edges = edges[has_token].drop_duplicates(["compound_key", "gene_symbol_key"]).reset_index(drop=True)
    train_pos, _ = mapped_pairs_with_compound_rows(train_edges, c_entities, c_rows, gene_symbols)
    test_pos, _ = mapped_pairs_with_compound_rows(test_edges, c_entities, c_rows, gene_symbols)
    if len(train_pos) < 10 or len(test_pos) < 10:
        raise ValueError(f"Not enough source-holdout positives for {token}: train={len(train_pos)} test={len(test_pos)}")
    order = rng.permutation(len(train_pos))
    val_n = max(1, int(round(0.15 * len(train_pos))))
    val_pos = train_pos[order[:val_n]]
    train_pos = train_pos[order[val_n:]]
    return train_pos, val_pos, test_pos


def downsample(arr: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if int(max_n) <= 0 or len(arr) <= int(max_n):
        return arr
    return arr[rng.choice(len(arr), size=int(max_n), replace=False)]


def sample_negatives(
    positives: np.ndarray,
    all_pos: set[Tuple[int, int]],
    n_genes: int,
    ratio: int,
    rng: np.random.Generator,
) -> np.ndarray:
    target_n = int(len(positives) * ratio)
    out: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    attempts = 0
    max_attempts = max(1000, target_n * 200)
    while len(out) < target_n and attempts < max_attempts:
        ci = int(positives[int(rng.integers(0, len(positives))), 0])
        gi = int(rng.integers(0, n_genes))
        pair = (ci, gi)
        if pair not in all_pos and pair not in seen:
            out.append(pair)
            seen.add(pair)
        attempts += 1
    if len(out) < target_n:
        raise RuntimeError(f"Could only sample {len(out)} negatives, requested {target_n}")
    return np.asarray(out, dtype=np.int64)


def pair_features(z_c: np.ndarray, z_g: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    c = z_c[pairs[:, 0]]
    g = z_g[pairs[:, 1]]
    cos = np.sum(c * g, axis=1, keepdims=True)
    return np.concatenate([c, g, c * g, np.abs(c - g), cos], axis=1).astype(np.float32)


def train_eval_one(
    z_c: np.ndarray,
    z_g: np.ndarray,
    train_pos: np.ndarray,
    val_pos: np.ndarray,
    test_pos: np.ndarray,
    all_pos: set[Tuple[int, int]],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    train_pos = downsample(train_pos, int(args.max_train_pos), rng)
    val_pos = downsample(val_pos, int(args.max_eval_pos), rng)
    test_pos = downsample(test_pos, int(args.max_eval_pos), rng)
    train_neg = sample_negatives(train_pos, all_pos, z_g.shape[0], int(args.negative_ratio), rng)
    val_neg = sample_negatives(val_pos, all_pos, z_g.shape[0], int(args.negative_ratio), rng)
    test_neg = sample_negatives(test_pos, all_pos, z_g.shape[0], int(args.negative_ratio), rng)

    x_train = np.vstack([pair_features(z_c, z_g, train_pos), pair_features(z_c, z_g, train_neg)])
    y_train = np.concatenate([np.ones(len(train_pos), dtype=np.int64), np.zeros(len(train_neg), dtype=np.int64)])
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", n_jobs=4)
    clf.fit(x_train, y_train)

    def eval_split(pos: np.ndarray, neg: np.ndarray) -> Dict[str, float]:
        x_raw = np.vstack([pair_features(z_c, z_g, pos), pair_features(z_c, z_g, neg)])
        y = np.concatenate([np.ones(len(pos), dtype=np.int64), np.zeros(len(neg), dtype=np.int64)])
        pred = clf.predict_proba(scaler.transform(x_raw))[:, 1]
        cos = x_raw[:, -1]
        return {
            "auroc": float(roc_auc_score(y, pred)),
            "auprc": float(average_precision_score(y, pred)),
            "cosine_auroc": float(roc_auc_score(y, cos)),
            "cosine_auprc": float(average_precision_score(y, cos)),
            "num_pos": int(len(pos)),
            "num_neg": int(len(neg)),
        }

    return {
        "train_pos": int(len(train_pos)),
        "train_neg": int(len(train_neg)),
        "val": eval_split(val_pos, val_neg),
        "test": eval_split(test_pos, test_neg),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    modalities = [x.strip().lower() for x in str(args.gene_modalities).split(",") if x.strip()]
    raw_edges_full = normalize_targets(args.target_edges, args.only_usable_typed_prior)
    within_sources = [x.strip().lower() for x in str(args.within_sources).split(",") if x.strip()]
    if within_sources:
        source_edge_sets: Dict[str, pd.DataFrame] = {}
        for token in within_sources:
            sub = raw_edges_full[source_mask(raw_edges_full, token)].copy()
            sub = dedup_pairs(sub)
            if sub.empty:
                raise SystemExit(f"No edges after within-source filter: {token}")
            source_edge_sets[token] = sub
        edges = dedup_pairs(raw_edges_full[source_mask(raw_edges_full, args.source)].copy())
    else:
        if bool(args.source_filter_before_dedup):
            edges = dedup_pairs(raw_edges_full[source_mask(raw_edges_full, args.source)].copy())
        else:
            raw_edges = dedup_pairs(raw_edges_full)
            edges = raw_edges[source_mask(raw_edges, args.source)].copy().reset_index(drop=True)
        if edges.empty:
            raise SystemExit(f"No edges after source filter: {args.source}")
    rows = parse_rows(args.rows)
    split_modes = [x.strip() for x in str(args.split_modes).split(",") if x.strip()]
    holdout_sources = [x.strip().lower() for x in str(args.holdout_sources).split(",") if x.strip()]

    flat_rows: List[Dict[str, Any]] = []
    all_metrics: Dict[str, Any] = {}
    for row_idx, (label, checkpoint) in enumerate(rows):
        payload, c_entities, c_rows, z_c, gene_symbols, z_gene = load_model_gene_union_embeddings(
            checkpoint, device, int(args.eval_batch_size), modalities
        )
        pos_pairs, map_meta = mapped_pairs_with_compound_rows(edges, c_entities, c_rows, gene_symbols)
        all_pos = {tuple(map(int, p)) for p in pos_pairs.tolist()}
        metrics: Dict[str, Any] = {
            "metadata": {
                **map_meta,
                "checkpoint": str(checkpoint),
                "epoch": payload.get("epoch"),
                "target_edges": str(args.target_edges),
                "source": str(args.source),
                "within_sources": ",".join(within_sources),
                "source_filter_before_dedup": bool(args.source_filter_before_dedup),
                "only_usable_typed_prior": bool(args.only_usable_typed_prior),
                "num_compounds": int(z_c.shape[0]),
                "num_gene_symbols": int(z_gene.shape[0]),
                "gene_modalities": ",".join(modalities),
            },
            "splits": {},
        }
        split_edge_sets = source_edge_sets if within_sources else {str(args.source): edges}
        for source_label, split_edges in split_edge_sets.items():
            split_pos_pairs, split_map_meta = mapped_pairs_with_compound_rows(split_edges, c_entities, c_rows, gene_symbols)
            split_all_pos = {tuple(map(int, p)) for p in split_pos_pairs.tolist()}
            if len(split_pos_pairs) < 10:
                raise ValueError(f"Not enough positive pairs for source={source_label}: {len(split_pos_pairs)}")
            metrics["splits"].setdefault(f"within_source::{source_label}", {"metadata": split_map_meta, "splits": {}})
            for split_mode in split_modes:
                rng = np.random.default_rng(
                    int(args.seed) + row_idx * 1009 + sum(ord(c) for c in f"{source_label}__{split_mode}")
                )
                train_pos, val_pos, test_pos = split_pairs(split_pos_pairs, split_mode, rng)
                out = train_eval_one(z_c, z_gene, train_pos, val_pos, test_pos, split_all_pos, args, rng)
                if within_sources:
                    metrics["splits"][f"within_source::{source_label}"]["splits"][split_mode] = out
                else:
                    metrics["splits"][split_mode] = out
                for part in ["val", "test"]:
                    rec = {
                        "row": label,
                        "source": source_label,
                        "split_mode": split_mode,
                        "fold": part,
                        "epoch": payload.get("epoch"),
                        "mapped_edges": split_map_meta["mapped_edges"],
                        "positive_pairs": split_map_meta["positive_pairs"],
                        "train_pos": out["train_pos"],
                        **out[part],
                    }
                    flat_rows.append(rec)
                    print(json.dumps({"completed": f"{label}__{source_label}__{split_mode}__{part}", **rec}, sort_keys=True), flush=True)
        for token in holdout_sources:
            rng = np.random.default_rng(int(args.seed) + row_idx * 1009 + sum(ord(c) for c in f"holdout_{token}"))
            train_pos, val_pos, test_pos = source_holdout_pairs(edges, token, c_entities, c_rows, gene_symbols, rng)
            out = train_eval_one(z_c, z_gene, train_pos, val_pos, test_pos, all_pos, args, rng)
            split_name = f"source_holdout_{token}"
            metrics["splits"][split_name] = out
            for part in ["val", "test"]:
                rec = {
                    "row": label,
                    "source": args.source,
                    "split_mode": split_name,
                    "fold": part,
                    "epoch": payload.get("epoch"),
                    "mapped_edges": map_meta["mapped_edges"],
                    "positive_pairs": map_meta["positive_pairs"],
                    "train_pos": out["train_pos"],
                    **out[part],
                }
                flat_rows.append(rec)
                print(json.dumps({"completed": f"{label}__{split_name}__{part}", **rec}, sort_keys=True), flush=True)
        all_metrics[label] = metrics

    metrics_path = args.output_dir / "gene_union_latent_link_classifier_metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output_dir / "gene_union_latent_link_classifier_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps({"metrics": str(metrics_path), "csv": str(csv_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
