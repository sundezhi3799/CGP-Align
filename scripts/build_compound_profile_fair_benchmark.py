from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


DEFAULT_MATCHED_SEED_CSV = Path(
    "output/cgp_align/baseline_reproduction_20260630/"
    "raw3180_matched_baselines_3seed/summary/raw3180_matched_encoder_seed_level.csv"
)
DEFAULT_MATCHED_SEED_CSV_FALLBACK = Path(
    "output/cgp_align/baseline_reproduction_20260630/"
    "raw3180_matched_baselines_3seed/summary/raw3180_cellclip_style_seed_level.csv"
)
DEFAULT_CGP_SEED_CSV = Path("output/cgp_align/branch_model_comparison/publication_multiseed_seed_level_metrics.csv")
DEFAULT_OUTPUT_DIR = Path("output/cgp_align/baseline_reproduction_20260630/compound_profile_fair_benchmark")


COMPOUND_BASELINE_METHODS = {
    "GGNN molecular graph": {
        "method_key": "matched_ggnn_graph",
        "method": "Matched GGNN graph",
        "method_family": "graph-profile contrastive encoder",
        "input_representation": "molecular graph",
        "reference_positioning": "MoCoP-style matched encoder baseline",
    },
    "MoLFormer SMILES embedding": {
        "method_key": "matched_molformer_smiles",
        "method": "Matched MoLFormer",
        "method_family": "SMILES language embedding profile encoder",
        "input_representation": "MoLFormer SMILES embedding",
        "reference_positioning": "CellCLIP-style language embedding baseline",
    },
    "ChemBERTa SMILES embedding": {
        "method_key": "matched_chemberta_smiles",
        "method": "Matched ChemBERTa",
        "method_family": "SMILES language embedding profile encoder",
        "input_representation": "ChemBERTa SMILES embedding",
        "reference_positioning": "CellCLIP-style language embedding baseline",
    },
    "RDKit/Morgan fixed molecular features": {
        "method_key": "matched_rdkit_morgan",
        "method": "Matched RDKit/Morgan",
        "method_family": "fingerprint/profile contrastive encoder",
        "input_representation": "RDKit descriptors plus Morgan fingerprint",
        "reference_positioning": "CLOOME-style matched fingerprint baseline",
    },
}


METRIC_BASES = [
    "entity_to_profile_1:100_Top10",
    "entity_to_profile_1:100_MRR",
    "entity_to_profile_1:1000_Top10",
    "entity_to_profile_1:1000_MRR",
    "entity_to_profile_full_R@1",
    "entity_to_profile_full_R@5",
    "entity_to_profile_full_R@10",
    "entity_to_profile_full_MRR",
    "entity_to_profile_full_median_rank",
    "profile_to_entity_1:100_Top10",
    "profile_to_entity_1:100_MRR",
    "profile_to_entity_1:1000_Top10",
    "profile_to_entity_1:1000_MRR",
    "profile_to_entity_full_R@1",
    "profile_to_entity_full_R@5",
    "profile_to_entity_full_R@10",
    "profile_to_entity_full_MRR",
    "profile_to_entity_full_median_rank",
    "mean_1:100_Top10",
    "mean_1:100_MRR",
    "mean_1:1000_Top10",
    "mean_1:1000_MRR",
    "mean_full_R@1",
    "mean_full_R@5",
    "mean_full_R@10",
    "mean_full_MRR",
    "mean_full_median_rank",
]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out):
        return None
    return out


def nested(data: Dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def add_bidirectional_means(row: Dict[str, Any]) -> None:
    for ratio in ["1:100", "1:1000"]:
        for metric in ["Top10", "MRR"]:
            vals = [
                as_float(row.get(f"entity_to_profile_{ratio}_{metric}")),
                as_float(row.get(f"profile_to_entity_{ratio}_{metric}")),
            ]
            vals = [v for v in vals if v is not None]
            if vals:
                row[f"mean_{ratio}_{metric}"] = float(sum(vals) / len(vals))
    for metric in ["R@1", "R@5", "R@10", "MRR", "median_rank"]:
        vals = [
            as_float(row.get(f"entity_to_profile_full_{metric}")),
            as_float(row.get(f"profile_to_entity_full_{metric}")),
        ]
        vals = [v for v in vals if v is not None]
        if vals:
            row[f"mean_full_{metric}"] = float(sum(vals) / len(vals))


def load_matched_compound_rows(seed_csv: Path) -> List[Dict[str, Any]]:
    if not seed_csv.exists() and seed_csv == DEFAULT_MATCHED_SEED_CSV and DEFAULT_MATCHED_SEED_CSV_FALLBACK.exists():
        seed_csv = DEFAULT_MATCHED_SEED_CSV_FALLBACK
    if not seed_csv.exists():
        raise FileNotFoundError(f"Missing matched baseline seed-level CSV: {seed_csv}")
    seed_df = pd.read_csv(seed_csv)
    seed_df = seed_df.loc[seed_df["branch"].astype(str) == "compound"].copy()
    rows: List[Dict[str, Any]] = []
    for record in seed_df.to_dict("records"):
        method_info = COMPOUND_BASELINE_METHODS.get(str(record.get("input")))
        if method_info is None:
            continue
        row: Dict[str, Any] = {
            **method_info,
            "seed": int(record["seed"]),
            "status": str(record.get("status", "")),
            "source": "raw3180_matched_encoder_baseline",
            "metrics_path": str(record.get("metrics_path", "")),
            "heldout_metrics_path": str(record.get("heldout_metrics_path", "")),
            "training_protocol": "single compound encoder aligned to the raw3180 entity-mean CellProfiler profile target",
            "profile_target": "raw3180 entity-mean 3180-dimensional CellProfiler profile",
            "split": "cold compound entity split",
            "branch": "compound",
        }
        for col in METRIC_BASES:
            value = as_float(record.get(col))
            if value is not None:
                row[col] = value
        add_bidirectional_means(row)
        rows.append(row)
    return rows


def cgp_row_from_metrics(seed_record: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    metric_path = Path(str(seed_record["metrics_path"]))
    if not metric_path.is_absolute():
        metric_path = repo_root / metric_path
    if not metric_path.exists():
        raise FileNotFoundError(f"Missing CGP-Align metric JSON: {metric_path}")
    payload = read_json(metric_path)
    row: Dict[str, Any] = {
        "method_key": "cgp_align_main",
        "method": "CGP-Align",
        "method_family": "multimodal phenotype-anchored foundation embedding",
        "input_representation": "learned compound branch with profile-anchored multimodal alignment",
        "reference_positioning": "main model",
        "seed": int(seed_record["seed"]),
        "status": "done",
        "source": "publication_multiseed_main_cgp_align",
        "metrics_path": str(metric_path),
        "training_protocol": "compound, ORF, CRISPR and profile branches aligned through shared phenotype anchors",
        "profile_target": "raw3180 entity-mean 3180-dimensional CellProfiler profile",
        "split": "cold compound entity split",
        "branch": "compound",
        "checkpoint_epoch": as_float(payload.get("checkpoint_epoch")),
        "best_score": as_float(payload.get("best_score")),
    }
    direction_map = {
        "entity_to_profile": "compound_structure_to_compound_profile",
        "profile_to_entity": "compound_profile_to_compound_structure",
    }
    for prefix, key in direction_map.items():
        for ratio in ["1:100", "1:1000"]:
            sampled = nested(payload, f"{key}_sampled", ratio) or {}
            top10 = as_float(sampled.get("Top10_accuracy_mean"))
            mrr = as_float(sampled.get("MRR_mean"))
            if top10 is not None:
                row[f"{prefix}_{ratio}_Top10"] = top10
            if mrr is not None:
                row[f"{prefix}_{ratio}_MRR"] = mrr
            top10_std = as_float(sampled.get("Top10_accuracy_std"))
            mrr_std = as_float(sampled.get("MRR_std"))
            if top10_std is not None:
                row[f"{prefix}_{ratio}_Top10_repeat_sd"] = top10_std
            if mrr_std is not None:
                row[f"{prefix}_{ratio}_MRR_repeat_sd"] = mrr_std
        full = payload.get(key, {}) or {}
        for metric, source_key in [
            ("R@1", "Recall@1"),
            ("R@5", "Recall@5"),
            ("R@10", "Recall@10"),
            ("MRR", "MRR"),
            ("median_rank", "median_rank"),
        ]:
            value = as_float(full.get(source_key))
            if value is not None:
                row[f"{prefix}_full_{metric}"] = value
        query_count = as_float(full.get("num_queries"))
        if query_count is not None:
            row[f"{prefix}_num_queries"] = int(query_count)
    add_bidirectional_means(row)
    return row


def load_cgp_main_rows(seed_csv: Path, repo_root: Path) -> List[Dict[str, Any]]:
    if not seed_csv.exists():
        raise FileNotFoundError(f"Missing CGP-Align seed-level CSV: {seed_csv}")
    seed_df = pd.read_csv(seed_csv)
    seed_df = seed_df.loc[seed_df["Method"].astype(str) == "CGP-Align"].copy()
    rows = [cgp_row_from_metrics(row, repo_root) for row in seed_df.to_dict("records")]
    return rows


def summarise_seed_rows(seed_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "method_key",
        "method",
        "method_family",
        "input_representation",
        "reference_positioning",
        "training_protocol",
        "profile_target",
        "split",
    ]
    metric_cols = [
        c
        for c in seed_df.columns
        if c in METRIC_BASES
        or c.endswith("_repeat_sd")
        or c.endswith("_num_queries")
        or c in {"checkpoint_epoch", "best_score"}
    ]
    rows: List[Dict[str, Any]] = []
    for values, sub in seed_df.groupby(group_cols, dropna=False):
        row: Dict[str, Any] = dict(zip(group_cols, values))
        done = sub.loc[sub["status"].astype(str) == "done"].copy()
        row["n_seed"] = int(len(done))
        row["seeds"] = ",".join(str(int(x)) for x in done["seed"].tolist())
        for col in metric_cols:
            vals = pd.to_numeric(done[col], errors="coerce").dropna() if col in done.columns else pd.Series(dtype=float)
            if len(vals):
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    if "mean_1:100_Top10_mean" in out.columns:
        out = out.sort_values("mean_1:100_Top10_mean", ascending=False, na_position="last")
    return out.reset_index(drop=True)


def long_direction_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    direction_specs = [
        ("compound_to_profile", "entity_to_profile", "compound -> profile"),
        ("profile_to_compound", "profile_to_entity", "profile -> compound"),
        ("bidirectional_mean", "mean", "bidirectional mean"),
    ]
    for record in seed_df.to_dict("records"):
        for direction_key, prefix, direction_label in direction_specs:
            for ratio in ["1:100", "1:1000"]:
                top10 = as_float(record.get(f"{prefix}_{ratio}_Top10"))
                mrr = as_float(record.get(f"{prefix}_{ratio}_MRR"))
                if top10 is None and mrr is None:
                    continue
                rows.append(
                    {
                        "method_key": record["method_key"],
                        "method": record["method"],
                        "seed": int(record["seed"]),
                        "direction_key": direction_key,
                        "direction": direction_label,
                        "gallery_protocol": ratio,
                        "top10": top10,
                        "mrr": mrr,
                    }
                )
            for metric in ["R@1", "R@5", "R@10", "MRR", "median_rank"]:
                value = as_float(record.get(f"{prefix}_full_{metric}"))
                if value is None:
                    continue
                rows.append(
                    {
                        "method_key": record["method_key"],
                        "method": record["method"],
                        "seed": int(record["seed"]),
                        "direction_key": direction_key,
                        "direction": direction_label,
                        "gallery_protocol": "full",
                        "metric": metric,
                        "value": value,
                    }
                )
    long = pd.DataFrame(rows)
    if long.empty:
        return long
    sampled = long.loc[long["gallery_protocol"].astype(str) != "full"].copy()
    if sampled.empty:
        return sampled
    return (
        sampled.groupby(["method_key", "method", "direction_key", "direction", "gallery_protocol"], dropna=False)
        .agg(
            top10_mean=("top10", "mean"),
            top10_sd=("top10", "std"),
            mrr_mean=("mrr", "mean"),
            mrr_sd=("mrr", "std"),
            n_seed=("seed", "nunique"),
        )
        .reset_index()
        .sort_values(["gallery_protocol", "direction_key", "top10_mean"], ascending=[True, True, False])
    )


def method_metadata(seed_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "method_key",
        "method",
        "method_family",
        "input_representation",
        "reference_positioning",
        "training_protocol",
        "profile_target",
        "split",
    ]
    return seed_df[cols].drop_duplicates().sort_values("method_key").reset_index(drop=True)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(df: pd.DataFrame, columns: Iterable[str]) -> str:
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return ""
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def write_readme(out_dir: Path, summary_df: pd.DataFrame, seed_df: pd.DataFrame) -> None:
    view_cols = [
        "method",
        "n_seed",
        "mean_1:100_Top10_mean",
        "mean_1:100_Top10_sd",
        "entity_to_profile_1:100_Top10_mean",
        "profile_to_entity_1:100_Top10_mean",
        "mean_1:1000_Top10_mean",
        "mean_full_R@10_mean",
    ]
    text = f"""# Compound-profile fair retrieval benchmark

This directory implements the common compound-profile alignment benchmark requested for the CGP-Align paper.

## Benchmark definition

- Fixed entities: the same held-out raw3180 compound split.
- Fixed profile target: the same entity-mean 3180-dimensional CellProfiler profile for every method.
- Method variation: only the compound-side representation/encoder changes.
- Retrieval directions: compound -> profile and profile -> compound.
- Main metric: Top-10 accuracy under sampled 1:100 and 1:1000 galleries, reported as seed-level values and 3-seed mean +/- SD.
- Additional metrics: full-gallery Recall@1/5/10, MRR and median rank when available.

The matched encoder baselines are not official end-to-end reproductions of CellCLIP, CLOOME or MoCoP. They are profile-alignment baselines adapted to the same raw3180 protocol.

## Main 1:100 retrieval summary

{markdown_table(summary_df, view_cols)}

## Files

- `compound_profile_fair_benchmark_seed_level.csv`: one row per method and seed.
- `compound_profile_fair_benchmark_summary.csv`: method-level mean +/- SD.
- `compound_profile_fair_benchmark_direction_summary.csv`: direction-level sampled retrieval summary.
- `compound_profile_fair_benchmark_method_metadata.csv`: method positioning and input definitions.
- `compound_profile_fair_benchmark_manifest.json`: paths and benchmark contract.

## Scope

This benchmark is designed for the manuscript claim that CGP-Align provides a phenotype-anchored compound representation that improves profile retrieval under a shared evaluation protocol. It should be described as a matched benchmark, not as an official reproduction of external methods.
"""
    out_dir.joinpath("README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified compound-profile retrieval benchmark summary.")
    parser.add_argument("--matched_seed_csv", type=Path, default=DEFAULT_MATCHED_SEED_CSV)
    parser.add_argument("--cgp_seed_csv", type=Path, default=DEFAULT_CGP_SEED_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo_root", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    rows = load_cgp_main_rows(args.cgp_seed_csv, repo_root)
    rows.extend(load_matched_compound_rows(args.matched_seed_csv))
    seed_df = pd.DataFrame(rows)
    summary_df = summarise_seed_rows(seed_df)
    direction_df = long_direction_summary(seed_df)
    metadata_df = method_metadata(seed_df)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_df.to_csv(out_dir / "compound_profile_fair_benchmark_seed_level.csv", index=False)
    summary_df.to_csv(out_dir / "compound_profile_fair_benchmark_summary.csv", index=False)
    direction_df.to_csv(out_dir / "compound_profile_fair_benchmark_direction_summary.csv", index=False)
    metadata_df.to_csv(out_dir / "compound_profile_fair_benchmark_method_metadata.csv", index=False)
    write_json(out_dir / "compound_profile_fair_benchmark_seed_level.json", seed_df.to_dict("records"))
    write_json(out_dir / "compound_profile_fair_benchmark_summary.json", summary_df.to_dict("records"))
    write_json(out_dir / "compound_profile_fair_benchmark_direction_summary.json", direction_df.to_dict("records"))
    write_json(out_dir / "compound_profile_fair_benchmark_method_metadata.json", metadata_df.to_dict("records"))
    write_json(
        out_dir / "compound_profile_fair_benchmark_manifest.json",
        {
            "benchmark": "compound_profile_fair_retrieval_v1",
            "matched_seed_csv": str(args.matched_seed_csv),
            "cgp_seed_csv": str(args.cgp_seed_csv),
            "output_dir": str(out_dir),
            "n_methods": int(summary_df["method_key"].nunique()),
            "n_seed_rows": int(len(seed_df)),
            "fixed_entities": "raw3180 held-out compound entities",
            "fixed_target": "entity-mean 3180-dimensional CellProfiler profile",
            "directions": ["compound -> profile", "profile -> compound"],
            "main_metric": "sampled Top-10 accuracy at 1:100 and 1:1000 galleries",
            "interpretation": "matched compound encoder benchmark under a shared profile-alignment protocol",
        },
    )
    write_readme(out_dir, summary_df, seed_df)
    print(f"[write] {out_dir}", flush=True)
    print(
        summary_df[
            [
                "method",
                "n_seed",
                "mean_1:100_Top10_mean",
                "mean_1:100_Top10_sd",
                "entity_to_profile_1:100_Top10_mean",
                "profile_to_entity_1:100_Top10_mean",
                "mean_1:1000_Top10_mean",
            ]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
