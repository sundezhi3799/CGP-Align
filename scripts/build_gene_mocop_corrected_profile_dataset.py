from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train-only source/plate/well corrected Gene-MoCoP profile matrices.")
    parser.add_argument("--input_data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=[
            "source_zscore_train",
            "source_zscore_plate_center",
            "source_zscore_well_center",
            "source_zscore_plate_well_center",
        ],
        required=True,
    )
    parser.add_argument("--split_name", default="cold_gene")
    parser.add_argument("--repeat_passes", type=int, default=2)
    return parser.parse_args()


def copy_metadata(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ["gene_mocop_entities.parquet", "gene_mocop_replicates.parquet", "splits_gene_mocop.json"]:
        shutil.copy2(input_dir / name, output_dir / name)


def source_train_zscore(
    x: np.ndarray,
    replicates: pd.DataFrame,
    train_entities: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, object]]:
    train_entity_set = set(np.asarray(train_entities, dtype=np.int64).tolist())
    train_mask = replicates["entity_index"].astype(int).isin(train_entity_set).to_numpy()
    modalities = replicates["perturbation_modality"].astype(str).to_numpy()
    out = np.empty_like(x, dtype=np.float32)
    summary: Dict[str, object] = {"mode": "source_train_zscore", "modalities": {}}
    for modality in sorted(set(modalities.tolist())):
        fit_rows = np.where(train_mask & (modalities == modality))[0]
        apply_rows = np.where(modalities == modality)[0]
        if fit_rows.size == 0:
            raise RuntimeError(f"No train replicate rows for modality={modality}")
        mean = np.nanmean(x[fit_rows], axis=0).astype(np.float32)
        std = np.nanstd(x[fit_rows], axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        out[apply_rows] = (x[apply_rows] - mean) / std
        summary["modalities"][modality] = {
            "num_fit_rows": int(fit_rows.size),
            "num_apply_rows": int(apply_rows.size),
            "mean_abs_mean": float(np.mean(np.abs(mean))),
            "median_std": float(np.median(std)),
        }
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), summary


def subtract_train_label_means(
    x: np.ndarray,
    labels: np.ndarray,
    train_rows: np.ndarray,
    label_name: str,
) -> Tuple[np.ndarray, Dict[str, object]]:
    out = x.copy()
    train_rows = np.asarray(train_rows, dtype=np.int64)
    labels = labels.astype(str)
    train_labels = labels[train_rows]
    summary: Dict[str, object] = {
        "label": label_name,
        "num_train_rows": int(train_rows.size),
        "num_train_labels": int(pd.Series(train_labels).nunique()),
        "num_all_labels": int(pd.Series(labels).nunique()),
        "num_unseen_apply_rows": 0,
    }
    label_to_mean: Dict[str, np.ndarray] = {}
    label_to_count: Dict[str, int] = {}
    train_df = pd.DataFrame({"row": train_rows, "label": train_labels})
    for label, block in train_df.groupby("label", sort=False):
        rows = block["row"].to_numpy(dtype=np.int64)
        label_to_mean[str(label)] = x[rows].mean(axis=0).astype(np.float32)
        label_to_count[str(label)] = int(rows.size)
    unseen = 0
    apply_df = pd.DataFrame({"row": np.arange(len(labels), dtype=np.int64), "label": labels})
    for label, block in apply_df.groupby("label", sort=False):
        rows = block["row"].to_numpy(dtype=np.int64)
        mean = label_to_mean.get(str(label))
        if mean is None:
            unseen += int(rows.size)
            continue
        out[rows] -= mean
    counts = np.asarray(list(label_to_count.values()), dtype=np.float64)
    summary.update(
        {
            "num_unseen_apply_rows": int(unseen),
            "train_label_count_median": float(np.median(counts)) if counts.size else 0.0,
            "train_label_count_min": int(np.min(counts)) if counts.size else 0,
            "train_label_count_max": int(np.max(counts)) if counts.size else 0,
        }
    )
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), summary


def correction_labels(mode: str) -> List[str]:
    if mode == "source_zscore_train":
        return []
    if mode == "source_zscore_plate_center":
        return ["Metadata_Plate"]
    if mode == "source_zscore_well_center":
        return ["Metadata_Well"]
    if mode == "source_zscore_plate_well_center":
        return ["Metadata_Plate", "Metadata_Well"]
    raise ValueError(f"Unsupported mode: {mode}")


def label_r2(x: np.ndarray, labels: Sequence[str]) -> Dict[str, float]:
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
    return {
        "groups": int(len(uniques)),
        "mean_r2": float(np.mean(r2)),
        "median_r2": float(np.median(r2)),
        "p90_r2": float(np.quantile(r2, 0.9)),
        "max_r2": float(np.max(r2)),
    }


def main() -> None:
    args = parse_args()
    copy_metadata(args.input_data_dir, args.output_dir)

    entities = pd.read_parquet(args.input_data_dir / "gene_mocop_entities.parquet")
    replicates = pd.read_parquet(args.input_data_dir / "gene_mocop_replicates.parquet")
    features = np.load(args.input_data_dir / "gene_mocop_replicate_features.npy").astype(np.float32)
    splits = json.loads((args.input_data_dir / "splits_gene_mocop.json").read_text(encoding="utf-8"))
    train_entities = np.asarray(splits[args.split_name]["train"], dtype=np.int64)
    train_entity_set = set(train_entities.tolist())
    train_rows = np.where(replicates["entity_index"].astype(int).isin(train_entity_set).to_numpy())[0]

    corrected, source_summary = source_train_zscore(features, replicates, train_entities)
    steps: List[Dict[str, object]] = [source_summary]
    labels = correction_labels(args.mode)
    passes = max(1, int(args.repeat_passes)) if len(labels) > 1 else 1
    for pass_idx in range(passes):
        for label_col in labels:
            corrected, step_summary = subtract_train_label_means(
                corrected,
                replicates[label_col].astype(str).to_numpy(),
                train_rows,
                f"{label_col}_pass{pass_idx + 1}",
            )
            steps.append(step_summary)

    np.save(args.output_dir / "gene_mocop_replicate_features.npy", corrected.astype(np.float32))
    audit = {
        "input_data_dir": str(args.input_data_dir),
        "output_dir": str(args.output_dir),
        "mode": args.mode,
        "split_name": args.split_name,
        "repeat_passes": int(args.repeat_passes),
        "num_entities": int(len(entities)),
        "num_replicates": int(len(replicates)),
        "feature_dim": int(corrected.shape[1]),
        "steps": steps,
        "post_correction_r2": {
            "plate": label_r2(corrected, replicates["Metadata_Plate"].astype(str).to_numpy()),
            "well": label_r2(corrected, replicates["Metadata_Well"].astype(str).to_numpy()),
            "entity": label_r2(corrected, replicates["entity_id"].astype(str).to_numpy()),
        },
    }
    (args.output_dir / "profile_correction_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
