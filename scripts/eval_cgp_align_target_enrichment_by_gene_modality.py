from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import torch

import train_cgp_align_replicate as cgp
from eval_cgp_align_target_enrichment_replicate import (
    build_positive_maps,
    flatten,
    load_model_and_data,
    load_targets,
    multi_positive_rank_metrics,
    parse_ints,
    parse_rows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate C-G enrichment after restricting gene gallery to one perturbation modality.")
    p.add_argument("--rows", required=True)
    p.add_argument("--target_edges", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    p.add_argument("--gene_modalities", default="orf,crispr")
    p.add_argument("--top_ks", default="10,50,100")
    p.add_argument("--eval_batch_size", type=int, default=1024)
    p.add_argument("--device", default="auto")
    p.add_argument("--only_usable_typed_prior", action="store_true")
    return p.parse_args()


def split_rows(data: Dict[str, Any], branch: str, split: str) -> np.ndarray:
    if split == "all":
        return np.unique(
            np.concatenate(
                [
                    np.asarray(data[branch]["train_entities"], dtype=np.int64),
                    np.asarray(data[branch]["val_entities"], dtype=np.int64),
                    np.asarray(data[branch]["test_entities"], dtype=np.int64),
                ]
            )
        )
    return np.asarray(data[branch][f"{split}_entities"], dtype=np.int64)


def filter_gene_rows_by_modality(data: Dict[str, Any], g_rows: np.ndarray, modality: str) -> np.ndarray:
    modality = str(modality).lower()
    vals = data["gene"]["entities"].iloc[np.asarray(g_rows, dtype=np.int64)]["perturbation_modality"].fillna("").astype(str).str.lower().to_numpy()
    return np.asarray(g_rows, dtype=np.int64)[vals == modality]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    top_ks = parse_ints(args.top_ks)
    if 50 not in top_ks:
        top_ks = sorted(set(top_ks + [50]))
    modalities = [x.strip().lower() for x in str(args.gene_modalities).split(",") if x.strip()]
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    targets = load_targets(args.target_edges, args.only_usable_typed_prior, respect_typed_profile_modality=False)
    rows = parse_rows(args.rows)

    all_metrics: Dict[str, Any] = {}
    flat_rows: List[Dict[str, Any]] = []
    for label, checkpoint in rows:
        model, data, payload = load_model_and_data(checkpoint, device, args.eval_batch_size)
        c_rows = split_rows(data, "compound", args.split)
        g_all = split_rows(data, "gene", args.split)
        z_c = cgp.encode_compounds(model, data["compound"]["graph_store"], c_rows, device, args.eval_batch_size)
        for modality in modalities:
            g_rows = filter_gene_rows_by_modality(data, g_all, modality)
            z_g = cgp.encode_gene_entities(
                model,
                data["gene"]["entities"],
                data["gene"]["protein_embeddings"],
                data["gene"]["modality_ids"],
                g_rows,
                device,
                args.eval_batch_size,
            )
            c2g_pos, g2c_pos, meta = build_positive_maps(targets, data["compound"]["entities"], data["gene"]["entities"], c_rows, g_rows)
            scores = z_c @ z_g.T
            metrics = {
                "metadata": {
                    **meta,
                    "checkpoint": str(checkpoint),
                    "checkpoint_epoch": payload.get("epoch"),
                    "best_score": payload.get("best_score"),
                    "split": args.split,
                    "gene_modality_filter": modality,
                    "target_edges": str(args.target_edges),
                    "only_usable_typed_prior": bool(args.only_usable_typed_prior),
                    "respect_typed_profile_modality": False,
                },
                "C2G": multi_positive_rank_metrics(scores, c2g_pos, top_ks),
                "G2C": multi_positive_rank_metrics(scores.T, g2c_pos, top_ks),
            }
            key = f"{label}__{modality}"
            all_metrics[key] = metrics
            rec = flatten(label, modality, metrics, meta, payload.get("epoch"))
            rec["gene_modality_filter"] = modality
            flat_rows.append(rec)
            print(json.dumps({"completed": key, **rec}, sort_keys=True), flush=True)

    metrics_path = args.output_dir / "target_enrichment_by_gene_modality_metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output_dir / "target_enrichment_by_gene_modality_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps({"metrics": str(metrics_path), "csv": str(csv_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
