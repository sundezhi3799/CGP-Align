from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create raw3180 data dirs with seed-specific entity splits.")
    p.add_argument("--compound_src", type=Path, required=True)
    p.add_argument("--gene_src", type=Path, required=True)
    p.add_argument("--out_root", type=Path, required=True)
    p.add_argument("--seeds", default="31,37,41")
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac", type=float, default=0.1)
    return p.parse_args()


def split_indices(indices: np.ndarray, seed: int, train_frac: float, val_frac: float) -> dict[str, list[int]]:
    rng = np.random.default_rng(int(seed))
    values = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(values)
    n = len(values)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_train = min(max(n_train, 0), n)
    n_val = min(max(n_val, 0), n - n_train)
    return {
        "train": values[:n_train].astype(int).tolist(),
        "val": values[n_train : n_train + n_val].astype(int).tolist(),
        "test": values[n_train + n_val :].astype(int).tolist(),
    }


def split_by_group(
    entities: pd.DataFrame,
    group_col: str,
    seed: int,
    train_frac: float,
    val_frac: float,
) -> dict[str, list[int]]:
    groups = entities[group_col].fillna("").astype(str).str.upper().to_numpy()
    valid = groups != ""
    unique_groups = np.unique(groups[valid])
    group_split = split_indices(np.arange(len(unique_groups), dtype=np.int64), seed, train_frac, val_frac)
    group_sets = {fold: set(unique_groups[idxs]) for fold, idxs in group_split.items()}
    out: dict[str, list[int]] = {}
    for fold, group_set in group_sets.items():
        rows = np.where(np.isin(groups, list(group_set)))[0]
        out[fold] = rows.astype(int).tolist()
    return out


def link_payload(src: Path, dst: Path, split_filename: str) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name == split_filename:
            continue
        target = dst / item.name
        if target.exists() or target.is_symlink():
            continue
        os.symlink(item.resolve(), target)


def write_split(
    path: Path,
    split_key: str,
    seed: int,
    n_entities: int,
    train_frac: float,
    val_frac: float,
    split: dict[str, list[int]] | None = None,
) -> None:
    split = split or split_indices(np.arange(n_entities, dtype=np.int64), seed, train_frac, val_frac)
    payload = {
        split_key: split,
        "random_entity": split,
        "seed": int(seed),
        "split_index_type": "entity_index",
        "schema": "raw3180_seeded_entity_split_v1",
        "train_frac": float(train_frac),
        "val_frac": float(val_frac),
        "test_frac": float(1.0 - train_frac - val_frac),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    compound_entities = pd.read_parquet(args.compound_src / "compound_mocop_entities.parquet")
    gene_entities = pd.read_parquet(args.gene_src / "gene_mocop_entities.parquet")
    summary = []
    for seed in seeds:
        c_dst = args.out_root / f"seed{seed}" / "compound_source_zscore_plate_well_center_3180"
        g_dst = args.out_root / f"seed{seed}" / "gene_source_zscore_plate_well_center_3180"
        link_payload(args.compound_src, c_dst, "splits_compound_mocop.json")
        link_payload(args.gene_src, g_dst, "splits_gene_mocop.json")
        gene_split = split_by_group(gene_entities, "gene_symbol", seed, args.train_frac, args.val_frac)
        write_split(c_dst / "splits_compound_mocop.json", "cold_compound", seed, len(compound_entities), args.train_frac, args.val_frac)
        write_split(g_dst / "splits_gene_mocop.json", "cold_gene", seed, len(gene_entities), args.train_frac, args.val_frac, gene_split)
        summary.append(
            {
                "seed": seed,
                "compound_dir": str(c_dst),
                "gene_dir": str(g_dst),
                "compound_entities": int(len(compound_entities)),
                "gene_entities": int(len(gene_entities)),
            }
        )
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "seeded_split_dirs_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
