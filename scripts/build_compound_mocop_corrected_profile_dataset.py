from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

COPY_FILES = ["compound_mocop_entities.parquet", "compound_mocop_replicates.parquet", "splits_compound_mocop.json", "feature_columns.json"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train-only source/plate/well corrected compound replicate profile matrices.")
    parser.add_argument("--input_data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["source_zscore_train", "source_zscore_plate_center", "source_zscore_well_center", "source_zscore_plate_well_center"], required=True)
    parser.add_argument("--split_name", default="cold_compound")
    parser.add_argument("--repeat_passes", type=int, default=2)
    return parser.parse_args()


def copy_metadata(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in COPY_FILES:
        src = input_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
    if (input_dir / "dataset_summary.json").exists():
        shutil.copy2(input_dir / "dataset_summary.json", output_dir / "dataset_summary.base.json")


def source_train_zscore(x: np.ndarray, reps: pd.DataFrame, train_entities: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    train_set = set(np.asarray(train_entities, dtype=np.int64).tolist())
    train_mask = reps["entity_index"].astype(int).isin(train_set).to_numpy()
    if not train_mask.any():
        raise RuntimeError("No train rows for source_train_zscore.")
    mean = np.nanmean(x[train_mask], axis=0).astype(np.float32)
    std = np.nanstd(x[train_mask], axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    out = (x - mean) / std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), {
        "mode": "source_train_zscore",
        "num_fit_rows": int(train_mask.sum()),
        "num_apply_rows": int(len(reps)),
        "mean_abs_mean": float(np.mean(np.abs(mean))),
        "median_std": float(np.median(std)),
    }


def subtract_train_label_means(x: np.ndarray, labels: np.ndarray, train_rows: np.ndarray, label_name: str) -> Tuple[np.ndarray, Dict[str, object]]:
    out = x.copy()
    labels = pd.Series(labels).astype(str).to_numpy()
    train_rows = np.asarray(train_rows, dtype=np.int64)
    label_to_mean: Dict[str, np.ndarray] = {}
    label_to_count: Dict[str, int] = {}
    frame = pd.DataFrame({"row": train_rows, "label": labels[train_rows]})
    for label, block in frame.groupby("label", sort=False):
        rows = block["row"].to_numpy(dtype=np.int64)
        label_to_mean[str(label)] = x[rows].mean(axis=0).astype(np.float32)
        label_to_count[str(label)] = int(rows.size)
    unseen = 0
    apply = pd.DataFrame({"row": np.arange(len(labels), dtype=np.int64), "label": labels})
    for label, block in apply.groupby("label", sort=False):
        rows = block["row"].to_numpy(dtype=np.int64)
        mean = label_to_mean.get(str(label))
        if mean is None:
            unseen += int(rows.size)
            continue
        out[rows] -= mean
    counts = np.asarray(list(label_to_count.values()), dtype=np.float64)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), {
        "label": label_name,
        "num_train_rows": int(train_rows.size),
        "num_train_labels": int(len(label_to_mean)),
        "num_all_labels": int(pd.Series(labels).nunique()),
        "num_unseen_apply_rows": int(unseen),
        "train_label_count_median": float(np.median(counts)) if counts.size else 0.0,
        "train_label_count_min": int(np.min(counts)) if counts.size else 0,
        "train_label_count_max": int(np.max(counts)) if counts.size else 0,
    }


def labels_for_mode(mode: str) -> List[str]:
    if mode == "source_zscore_train":
        return []
    if mode == "source_zscore_plate_center":
        return ["Metadata_Plate"]
    if mode == "source_zscore_well_center":
        return ["Metadata_Well"]
    if mode == "source_zscore_plate_well_center":
        return ["Metadata_Plate", "Metadata_Well"]
    raise ValueError(mode)


def categorical_r2(x: np.ndarray, labels: Sequence[str]) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=object)
    codes, uniques = pd.factorize(labels, sort=True)
    global_mean = x.mean(axis=0)
    total = ((x - global_mean) ** 2).sum(axis=0)
    between = np.zeros(x.shape[1], dtype=np.float64)
    for code in range(len(uniques)):
        idx = np.where(codes == code)[0]
        if idx.size == 0:
            continue
        diff = x[idx].mean(axis=0) - global_mean
        between += idx.size * (diff.astype(np.float64) ** 2)
    r2 = between / np.maximum(total, 1e-12)
    return {"groups": int(len(uniques)), "mean_r2": float(np.mean(r2)), "median_r2": float(np.median(r2)), "p90_r2": float(np.quantile(r2, 0.9)), "max_r2": float(np.max(r2))}


def main() -> None:
    args = parse_args()
    copy_metadata(args.input_data_dir, args.output_dir)
    entities = pd.read_parquet(args.input_data_dir / "compound_mocop_entities.parquet")
    reps = pd.read_parquet(args.input_data_dir / "compound_mocop_replicates.parquet")
    x = np.load(args.input_data_dir / "compound_mocop_replicate_features.npy").astype(np.float32)
    splits = json.loads((args.input_data_dir / "splits_compound_mocop.json").read_text(encoding="utf-8"))
    train_entities = np.asarray(splits[args.split_name]["train"], dtype=np.int64)
    train_set = set(train_entities.tolist())
    train_rows = np.where(reps["entity_index"].astype(int).isin(train_set).to_numpy())[0]
    corrected, source_summary = source_train_zscore(x, reps, train_entities)
    steps: List[Dict[str, object]] = [source_summary]
    labels = labels_for_mode(args.mode)
    passes = max(1, int(args.repeat_passes)) if len(labels) > 1 else 1
    for pass_idx in range(passes):
        for label_col in labels:
            corrected, step = subtract_train_label_means(corrected, reps[label_col].astype(str).to_numpy(), train_rows, f"{label_col}_pass{pass_idx + 1}")
            steps.append(step)
    np.save(args.output_dir / "compound_mocop_replicate_features.npy", corrected.astype(np.float32))
    audit = {
        "input_data_dir": str(args.input_data_dir),
        "output_dir": str(args.output_dir),
        "mode": args.mode,
        "split_name": args.split_name,
        "num_entities": int(len(entities)),
        "num_replicates": int(len(reps)),
        "feature_dim": int(corrected.shape[1]),
        "steps": steps,
        "post_correction_r2": {
            "plate": categorical_r2(corrected, reps["Metadata_Plate"].astype(str).to_numpy()),
            "well": categorical_r2(corrected, reps["Metadata_Well"].astype(str).to_numpy()),
            "entity": categorical_r2(corrected, reps["entity_id"].astype(str).to_numpy()),
        },
    }
    (args.output_dir / "profile_correction_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    summary = {"dataset_name": "compound_mocop_replicate_corrected", "input_data_dir": str(args.input_data_dir), "mode": args.mode, "feature_dim": int(corrected.shape[1]), "num_entities": int(len(entities)), "num_replicates": int(len(reps))}
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
