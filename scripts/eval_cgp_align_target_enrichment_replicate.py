from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pandas as pd
import torch

import train_cgp_align_replicate as cgp
from eval_cgp_align_crossmodal_bridge import namespace_from_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate known compound-target enrichment in replicate CGP-Align C-G latent space.")
    p.add_argument("--rows", required=True, help="Comma-separated label:checkpoint_path rows.")
    p.add_argument("--target_edges", type=Path, default=Path("data/cgp_cpg_full_strict_target_edges/compound_target_edges.parquet"))
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    p.add_argument("--top_ks", default="10,50,100")
    p.add_argument("--eval_batch_size", type=int, default=1024)
    p.add_argument("--device", default="auto")
    p.add_argument("--only_usable_typed_prior", action="store_true")
    p.add_argument("--respect_typed_profile_modality", action="store_true")
    return p.parse_args()


def parse_rows(text: str) -> List[Tuple[str, Path]]:
    rows: List[Tuple[str, Path]] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        label, path = item.split(":", 1)
        rows.append((label.strip(), Path(path.strip())))
    return rows


def parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def load_model_and_data(checkpoint: Path, device: torch.device, eval_batch_size: int) -> Tuple[cgp.ReplicateCGPAlign, Dict[str, Any], Dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = namespace_from_config(payload.get("config", {}))
    args.eval_batch_size = int(eval_batch_size)
    args.smoke_test = False
    data = cgp.build_data(args)
    model, _ = cgp.build_model(args, data, device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, data, payload


def load_targets(path: Path, only_usable_typed_prior: bool, respect_typed_profile_modality: bool) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if only_usable_typed_prior:
        if "usable_for_typed_prior" not in df.columns:
            raise SystemExit("--only_usable_typed_prior requires a usable_for_typed_prior column")
        df = df[df["usable_for_typed_prior"].astype(bool)].copy()
    compound_col = "compound_id" if "compound_id" in df.columns else "inchikey"
    gene_col = "gene_id" if "gene_id" in df.columns else "gene_symbol"
    out = {
        "compound_id": df[compound_col].astype(str),
        "gene_symbol": df[gene_col].astype(str).str.upper(),
    }
    if respect_typed_profile_modality:
        if "typed_profile_modality" not in df.columns:
            raise SystemExit("--respect_typed_profile_modality requires a typed_profile_modality column")
        out["target_modality"] = df["typed_profile_modality"].fillna("").astype(str).str.lower()
    else:
        out["target_modality"] = ""
    out = pd.DataFrame(out)
    out = out[(out["compound_id"] != "") & (out["gene_symbol"] != "")]
    return out.drop_duplicates().reset_index(drop=True)


def build_positive_maps(
    targets: pd.DataFrame,
    c_entities: pd.DataFrame,
    g_entities: pd.DataFrame,
    c_rows: np.ndarray,
    g_rows: np.ndarray,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[str, Any]]:
    c_local = {int(row): i for i, row in enumerate(np.asarray(c_rows, dtype=np.int64).tolist())}
    g_local = {int(row): i for i, row in enumerate(np.asarray(g_rows, dtype=np.int64).tolist())}
    compound_to_rows: Dict[str, List[int]] = {}
    for row in np.asarray(c_rows, dtype=np.int64).tolist():
        ent = c_entities.iloc[int(row)]
        keys = {str(ent.get("compound_id", "")), str(ent.get("inchikey", "")), str(ent.get("entity_id", ""))}
        for key in keys:
            if key:
                compound_to_rows.setdefault(key, []).append(int(row))
    gene_to_rows: Dict[str, List[int]] = {}
    gene_modality_to_rows: Dict[Tuple[str, str], List[int]] = {}
    for row in np.asarray(g_rows, dtype=np.int64).tolist():
        sym = str(g_entities.iloc[int(row)].get("gene_symbol", "")).upper()
        modality = str(g_entities.iloc[int(row)].get("perturbation_modality", "")).lower()
        if sym:
            gene_to_rows.setdefault(sym, []).append(int(row))
            if modality:
                gene_modality_to_rows.setdefault((sym, modality), []).append(int(row))

    c2g: Dict[int, set[int]] = {}
    g2c: Dict[int, set[int]] = {}
    mapped_edges = 0
    expanded_c2g_edges = 0
    expanded_g2c_edges = 0
    for rec in targets.itertuples(index=False):
        cands_c = compound_to_rows.get(str(rec.compound_id), [])
        target_modality = str(getattr(rec, "target_modality", "") or "").lower()
        if target_modality:
            cands_g = gene_modality_to_rows.get((str(rec.gene_symbol).upper(), target_modality), [])
        else:
            cands_g = gene_to_rows.get(str(rec.gene_symbol).upper(), [])
        if not cands_c or not cands_g:
            continue
        mapped_edges += 1
        expanded_c2g_edges += int(len(cands_c) * len(cands_g))
        expanded_g2c_edges += int(len(cands_g) * len(cands_c))
        for c_row in cands_c:
            ci = c_local.get(int(c_row))
            if ci is None:
                continue
            c2g.setdefault(ci, set()).update(g_local[int(g_row)] for g_row in cands_g if int(g_row) in g_local)
        for g_row in cands_g:
            gi = g_local.get(int(g_row))
            if gi is None:
                continue
            g2c.setdefault(gi, set()).update(c_local[int(c_row)] for c_row in cands_c if int(c_row) in c_local)
    c2g_list = {k: sorted(v) for k, v in c2g.items() if v}
    g2c_list = {k: sorted(v) for k, v in g2c.items() if v}
    meta = {
        "target_edges_total": int(len(targets)),
        "target_edges_mapped_to_split": int(mapped_edges),
        "target_edges_expanded_c2g": int(expanded_c2g_edges),
        "target_edges_expanded_g2c": int(expanded_g2c_edges),
        "c2g_num_queries_with_targets": int(len(c2g_list)),
        "g2c_num_queries_with_targets": int(len(g2c_list)),
        "num_compounds": int(len(c_rows)),
        "num_genes": int(len(g_rows)),
    }
    return c2g_list, g2c_list, meta


def random_recall_without_replacement(gallery_n: int, num_positive: int, k: int) -> float:
    n = int(gallery_n)
    p = min(max(int(num_positive), 0), n)
    kk = min(max(int(k), 0), n)
    if p <= 0 or kk <= 0:
        return 0.0
    if kk > n - p:
        return 1.0
    # P(hit >= 1) = 1 - C(N-P, K) / C(N, K), computed in log-space.
    log_no_hit = (
        math.lgamma(n - p + 1)
        - math.lgamma(n - p - kk + 1)
        - math.lgamma(n + 1)
        + math.lgamma(n - kk + 1)
    )
    return float(1.0 - math.exp(log_no_hit))


def multi_positive_rank_metrics(scores: np.ndarray, positives: Dict[int, List[int]], top_ks: Sequence[int]) -> Dict[str, Any]:
    gallery_n = int(scores.shape[1])
    ranks: List[int] = []
    pos_counts: List[int] = []
    hits = {int(k): [] for k in top_ks}
    random_hits = {int(k): [] for k in top_ks}
    top_pos_fracs = {int(k): [] for k in top_ks}
    for q, pos in positives.items():
        pos_arr = np.asarray(pos, dtype=np.int64)
        if pos_arr.size == 0:
            continue
        row = scores[int(q)]
        best = float(np.max(row[pos_arr]))
        rank = int(1 + np.sum(row > best))
        ranks.append(rank)
        pos_counts.append(int(pos_arr.size))
        for k in top_ks:
            kk = min(int(k), gallery_n)
            idx = np.argpartition(-row, kth=kk - 1)[:kk] if kk < gallery_n else np.arange(gallery_n)
            hit_count = int(np.intersect1d(idx, pos_arr, assume_unique=False).size)
            hits[int(k)].append(1.0 if hit_count > 0 else 0.0)
            top_pos_fracs[int(k)].append(float(hit_count) / float(kk))
            random_hits[int(k)].append(random_recall_without_replacement(gallery_n, int(pos_arr.size), kk))
    ranks_np = np.asarray(ranks, dtype=np.float64)
    out: Dict[str, Any] = {
        "num_queries": int(len(ranks)),
        "gallery_size": gallery_n,
        "positive_count_mean": float(np.mean(pos_counts)) if pos_counts else 0.0,
        "median_rank": float(np.median(ranks_np)) if ranks else None,
        "mean_rank": float(np.mean(ranks_np)) if ranks else None,
        "MRR": float(np.mean(1.0 / ranks_np)) if ranks else None,
    }
    for k in top_ks:
        kk = int(k)
        recall = float(np.mean(hits[kk])) if hits[kk] else 0.0
        random_recall = float(np.mean(random_hits[kk])) if random_hits[kk] else 0.0
        density = float(np.mean(top_pos_fracs[kk])) if top_pos_fracs[kk] else 0.0
        random_density = float(np.mean(pos_counts) / gallery_n) if pos_counts else 0.0
        out[f"Recall@{kk}"] = recall
        out[f"RandomRecall@{kk}"] = random_recall
        out[f"Lift@{kk}"] = float(recall / random_recall) if random_recall > 0 else None
        out[f"TargetDensity@{kk}"] = density
        out[f"TargetDensityLift@{kk}"] = float(density / random_density) if random_density > 0 else None
    return out


def flatten(row: str, run: str, metrics: Dict[str, Any], meta: Dict[str, Any], epoch: Any) -> Dict[str, Any]:
    c = metrics["C2G"]
    g = metrics["G2C"]
    return {
        "row": row,
        "run": run,
        "epoch": epoch,
        "mapped_edges": meta["target_edges_mapped_to_split"],
        "C2G_queries": c["num_queries"],
        "G2C_queries": g["num_queries"],
        "C2G_MRR": c["MRR"],
        "G2C_MRR": g["MRR"],
        "C2G_Recall@50": c["Recall@50"],
        "G2C_Recall@50": g["Recall@50"],
        "C2G_Lift@50": c["Lift@50"],
        "G2C_Lift@50": g["Lift@50"],
        "Mean_Lift@50": float(np.mean([c["Lift@50"], g["Lift@50"]])),
        "C2G_TargetDensityLift@50": c["TargetDensityLift@50"],
        "G2C_TargetDensityLift@50": g["TargetDensityLift@50"],
    }


def main() -> None:
    args = parse_args()
    rows = parse_rows(args.rows)
    top_ks = parse_ints(args.top_ks)
    if 50 not in top_ks:
        top_ks = sorted(set(top_ks + [50]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    targets = load_targets(args.target_edges, args.only_usable_typed_prior, args.respect_typed_profile_modality)
    all_metrics: Dict[str, Any] = {}
    flat_rows: List[Dict[str, Any]] = []
    for label, checkpoint in rows:
        model, data, payload = load_model_and_data(checkpoint, device, args.eval_batch_size)
        if args.split == "all":
            c_rows = np.unique(
                np.concatenate(
                    [
                        np.asarray(data["compound"]["train_entities"], dtype=np.int64),
                        np.asarray(data["compound"]["val_entities"], dtype=np.int64),
                        np.asarray(data["compound"]["test_entities"], dtype=np.int64),
                    ]
                )
            )
            g_rows = np.unique(
                np.concatenate(
                    [
                        np.asarray(data["gene"]["train_entities"], dtype=np.int64),
                        np.asarray(data["gene"]["val_entities"], dtype=np.int64),
                        np.asarray(data["gene"]["test_entities"], dtype=np.int64),
                    ]
                )
            )
        else:
            c_rows = np.asarray(data["compound"][f"{args.split}_entities"], dtype=np.int64)
            g_rows = np.asarray(data["gene"][f"{args.split}_entities"], dtype=np.int64)
        c2g_pos, g2c_pos, meta = build_positive_maps(targets, data["compound"]["entities"], data["gene"]["entities"], c_rows, g_rows)
        z_c = cgp.encode_compounds(model, data["compound"]["graph_store"], c_rows, device, args.eval_batch_size)
        z_g = cgp.encode_gene_entities(
            model,
            data["gene"]["entities"],
            data["gene"]["protein_embeddings"],
            data["gene"]["modality_ids"],
            g_rows,
            device,
            args.eval_batch_size,
        )
        scores = z_c @ z_g.T
        metrics = {
            "metadata": {
                **meta,
                "checkpoint": str(checkpoint),
                "checkpoint_epoch": payload.get("epoch"),
                "best_score": payload.get("best_score"),
                "split": args.split,
                "target_edges": str(args.target_edges),
                "only_usable_typed_prior": bool(args.only_usable_typed_prior),
                "respect_typed_profile_modality": bool(args.respect_typed_profile_modality),
            },
            "C2G": multi_positive_rank_metrics(scores, c2g_pos, top_ks),
            "G2C": multi_positive_rank_metrics(scores.T, g2c_pos, top_ks),
        }
        all_metrics[label] = metrics
        flat_rows.append(flatten(label, checkpoint.parent.name, metrics, meta, payload.get("epoch")))
        print(json.dumps({"completed": label, **flat_rows[-1]}, indent=2))
    metrics_path = args.output_dir / "target_enrichment_metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output_dir / "target_enrichment_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)
    md_path = args.output_dir / "target_enrichment_summary.md"
    lines = [
        "# Known compound-target enrichment",
        "",
        f"Target edges: `{args.target_edges}`",
        f"Split: `{args.split}`",
        f"Only usable typed prior: `{bool(args.only_usable_typed_prior)}`",
        f"Respect typed profile modality: `{bool(args.respect_typed_profile_modality)}`",
        "",
        "| model | epoch | mapped edges | C2G queries | G2C queries | C2G MRR | G2C MRR | C2G Recall@50 | G2C Recall@50 | C2G Lift@50 | G2C Lift@50 | Mean Lift@50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in flat_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["row"]),
                    str(r["epoch"]),
                    str(r["mapped_edges"]),
                    str(r["C2G_queries"]),
                    str(r["G2C_queries"]),
                    f"{r['C2G_MRR']:.4f}",
                    f"{r['G2C_MRR']:.4f}",
                    f"{r['C2G_Recall@50']:.4f}",
                    f"{r['G2C_Recall@50']:.4f}",
                    f"{r['C2G_Lift@50']:.4f}",
                    f"{r['G2C_Lift@50']:.4f}",
                    f"{r['Mean_Lift@50']:.4f}",
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": str(metrics_path), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
