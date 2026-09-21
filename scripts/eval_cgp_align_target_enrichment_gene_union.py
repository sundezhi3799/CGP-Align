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
import torch

import train_cgp_align_replicate as cgp
from eval_cgp_align_target_enrichment_replicate import (
    flatten,
    load_model_and_data,
    load_targets,
    multi_positive_rank_metrics,
    parse_ints,
    parse_rows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate compound-gene target enrichment with ORF/CRISPR collapsed by gene symbol.")
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


def filter_gene_rows_by_modalities(data: Dict[str, Any], g_rows: np.ndarray, modalities: List[str]) -> np.ndarray:
    allowed = {str(x).lower() for x in modalities}
    vals = (
        data["gene"]["entities"].iloc[np.asarray(g_rows, dtype=np.int64)]["perturbation_modality"]
        .fillna("")
        .astype(str)
        .str.lower()
        .to_numpy()
    )
    return np.asarray(g_rows, dtype=np.int64)[np.asarray([x in allowed for x in vals], dtype=bool)]


def compound_key_to_rows(c_entities, c_rows: np.ndarray) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for local_idx, row in enumerate(np.asarray(c_rows, dtype=np.int64).tolist()):
        ent = c_entities.iloc[int(row)]
        keys = {str(ent.get("compound_id", "")), str(ent.get("inchikey", "")), str(ent.get("entity_id", ""))}
        for key in keys:
            if key:
                out.setdefault(key, []).append(int(local_idx))
    return out


def gene_symbol_groups(g_entities, g_rows: np.ndarray) -> Tuple[List[str], Dict[str, int], List[np.ndarray], Dict[str, List[str]]]:
    symbol_to_entity_cols: Dict[str, List[int]] = {}
    symbol_to_modalities: Dict[str, set[str]] = {}
    for local_idx, row in enumerate(np.asarray(g_rows, dtype=np.int64).tolist()):
        ent = g_entities.iloc[int(row)]
        sym = str(ent.get("gene_symbol", "")).upper()
        if not sym:
            continue
        modality = str(ent.get("perturbation_modality", "")).lower()
        symbol_to_entity_cols.setdefault(sym, []).append(int(local_idx))
        if modality:
            symbol_to_modalities.setdefault(sym, set()).add(modality)
    symbols = sorted(symbol_to_entity_cols)
    symbol_to_idx = {sym: i for i, sym in enumerate(symbols)}
    groups = [np.asarray(symbol_to_entity_cols[sym], dtype=np.int64) for sym in symbols]
    modalities = {sym: sorted(symbol_to_modalities.get(sym, set())) for sym in symbols}
    return symbols, symbol_to_idx, groups, modalities


def collapse_entity_scores_to_symbols(entity_scores: np.ndarray, groups: List[np.ndarray]) -> np.ndarray:
    if not groups:
        return np.zeros((entity_scores.shape[0], 0), dtype=np.float32)
    out = np.empty((entity_scores.shape[0], len(groups)), dtype=np.float32)
    for idx, cols in enumerate(groups):
        if len(cols) == 1:
            out[:, idx] = entity_scores[:, int(cols[0])]
        else:
            out[:, idx] = np.max(entity_scores[:, cols], axis=1)
    return out


def build_symbol_positive_maps(
    targets,
    c_entities,
    c_rows: np.ndarray,
    gene_symbol_to_idx: Dict[str, int],
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[str, Any]]:
    compound_to_local = compound_key_to_rows(c_entities, c_rows)
    c2g: Dict[int, set[int]] = {}
    g2c: Dict[int, set[int]] = {}
    mapped_edges = 0
    expanded_c2g_edges = 0
    expanded_g2c_edges = 0
    for rec in targets.itertuples(index=False):
        cands_c = compound_to_local.get(str(rec.compound_id), [])
        gi = gene_symbol_to_idx.get(str(rec.gene_symbol).upper())
        if not cands_c or gi is None:
            continue
        mapped_edges += 1
        expanded_c2g_edges += int(len(cands_c))
        expanded_g2c_edges += int(len(cands_c))
        for ci in cands_c:
            c2g.setdefault(int(ci), set()).add(int(gi))
            g2c.setdefault(int(gi), set()).add(int(ci))
    return (
        {k: sorted(v) for k, v in c2g.items() if v},
        {k: sorted(v) for k, v in g2c.items() if v},
        {
            "target_edges_total": int(len(targets)),
            "target_edges_mapped_to_split": int(mapped_edges),
            "target_edges_expanded_c2g": int(expanded_c2g_edges),
            "target_edges_expanded_g2c": int(expanded_g2c_edges),
            "c2g_num_queries_with_targets": int(len(c2g)),
            "g2c_num_queries_with_targets": int(len(g2c)),
            "num_compounds": int(len(c_rows)),
            "num_gene_symbols": int(len(gene_symbol_to_idx)),
        },
    )


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
        g_rows = filter_gene_rows_by_modalities(data, split_rows(data, "gene", args.split), modalities)
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
        gene_symbols, gene_symbol_to_idx, gene_groups, gene_symbol_modalities = gene_symbol_groups(data["gene"]["entities"], g_rows)
        c2g_pos, g2c_pos, meta = build_symbol_positive_maps(targets, data["compound"]["entities"], c_rows, gene_symbol_to_idx)
        entity_scores = z_c @ z_g.T
        scores = collapse_entity_scores_to_symbols(entity_scores, gene_groups)
        metrics = {
            "metadata": {
                **meta,
                "checkpoint": str(checkpoint),
                "checkpoint_epoch": payload.get("epoch"),
                "best_score": payload.get("best_score"),
                "split": args.split,
                "gene_modality_filter": ",".join(modalities),
                "gene_gallery_mode": "gene_symbol_union_max_orf_crispr",
                "target_edges": str(args.target_edges),
                "only_usable_typed_prior": bool(args.only_usable_typed_prior),
                "respect_typed_profile_modality": False,
                "num_gene_entities": int(len(g_rows)),
                "num_gene_symbols_with_orf_and_crispr": int(sum(1 for sym in gene_symbols if {"orf", "crispr"}.issubset(set(gene_symbol_modalities.get(sym, []))))),
            },
            "C2G": multi_positive_rank_metrics(scores, c2g_pos, top_ks),
            "G2C": multi_positive_rank_metrics(scores.T, g2c_pos, top_ks),
        }
        key = f"{label}__gene_union"
        all_metrics[key] = metrics
        rec = flatten(label, "gene_union", metrics, meta, payload.get("epoch"))
        rec["gene_gallery_mode"] = "gene_symbol_union_max_orf_crispr"
        rec["gene_modalities"] = ",".join(modalities)
        rec["num_gene_symbols"] = meta["num_gene_symbols"]
        flat_rows.append(rec)
        print(json.dumps({"completed": key, **rec}, sort_keys=True), flush=True)

    metrics_path = args.output_dir / "target_enrichment_gene_union_metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output_dir / "target_enrichment_gene_union_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps({"metrics": str(metrics_path), "csv": str(csv_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
