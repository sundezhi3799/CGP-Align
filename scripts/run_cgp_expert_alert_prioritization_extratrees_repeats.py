from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_cgp_expert_alert_prioritization import (  # noqa: E402
    load_alert_libraries,
    match_alert_matrix,
    plot_alert_heatmap,
    ranking_metrics,
)
from run_cgp_structural_alert_mining_pilot import (  # noqa: E402
    compare_alert_signatures,
    json_default,
    load_cgp_inputs,
    load_tasks,
    mine_bit_enrichment,
    resolve_cgp_paths,
    top_unique_alerts,
    write_json,
)


METHOD_ORDER = [
    "structure_alert_count",
    "structure_alert_extratrees",
    "cgp_latent_extratrees",
    "cgp_plus_alert_extratrees",
]

METHOD_LABELS = {
    "structure_alert_count": "Alert count",
    "structure_alert_extratrees": "Expert alerts",
    "cgp_latent_extratrees": "CGP latent",
    "cgp_plus_alert_extratrees": "CGP + alerts",
}

METHOD_COLORS = {
    "structure_alert_count": "#9AA6B2",
    "structure_alert_extratrees": "#5F6F7C",
    "cgp_latent_extratrees": "#0B3C5D",
    "cgp_plus_alert_extratrees": "#D95F02",
}

COMPARABLE_METHODS = {
    "structure_alert_logistic": ("Expert alerts", "Logistic"),
    "cgp_latent_logistic": ("CGP latent", "Logistic"),
    "cgp_plus_alert_logistic": ("CGP + alerts", "Logistic"),
    "structure_alert_extratrees": ("Expert alerts", "ExtraTrees"),
    "cgp_latent_extratrees": ("CGP latent", "ExtraTrees"),
    "cgp_plus_alert_extratrees": ("CGP + alerts", "ExtraTrees"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Repeat CGP-guided expert structural-alert prioritization with ExtraTrees readouts. "
            "This is a robustness control for the logistic-readout experiment."
        )
    )
    p.add_argument(
        "--toxric_dir",
        type=Path,
        default=Path("output/cgp_align/paper/raw_results/tox_np_mechanism/raw_download_probe/toxric/toxric_30_datasets"),
    )
    p.add_argument(
        "--benchmark_dir",
        type=Path,
        default=Path("output/cgp_align/paper/raw_results/tox_np_mechanism/toxric_benchmark_nyan_reuse_smoke"),
    )
    p.add_argument("--cgp_smiles_path", type=Path)
    p.add_argument("--cgp_embeddings_path", type=Path)
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/cgp_align/paper/raw_results/expert_alert_prioritization_with_server_alerts_extratrees_repeated_seed"),
    )
    p.add_argument(
        "--alert_cache_dir",
        type=Path,
        default=Path("output/cgp_align/paper/raw_results/expert_alert_prioritization_with_server_alerts_repeated_seed"),
        help="Existing directory with expert_alert_metadata.csv and compound_expert_alert_matrix.npz.",
    )
    p.add_argument(
        "--logistic_dir",
        type=Path,
        default=Path("output/cgp_align/paper/raw_results/expert_alert_prioritization_with_server_alerts_repeated_seed"),
    )
    p.add_argument("--tasks", default="phenotype")
    p.add_argument("--catalogs", default="PAINS,BRENK,NIH,ZINC")
    p.add_argument(
        "--external_alerts",
        type=Path,
        nargs="*",
        default=[
            Path("output/cgp_align/paper/raw_results/expert_alert_prioritization/external_alert_libraries/structural_alerts_labeled.csv"),
            Path("output/cgp_align/paper/raw_results/expert_alert_prioritization/external_alert_libraries/structural_alerts_toxtree_extra.csv"),
            Path("output/cgp_align/paper/raw_results/expert_alert_prioritization/external_alert_libraries/chembl_structural_alerts.csv"),
        ],
    )
    p.add_argument("--seeds", default="13,41,97,123,2026")
    p.add_argument("--top_k_alerts", type=int, default=20)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--n_estimators", type=int, default=240)
    p.add_argument("--n_jobs", type=int, default=-1)
    return p.parse_args()


def parse_list(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def alert_examples_from_meta(meta: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for rec in meta.itertuples(index=False):
        out[int(rec.alert_col)] = {
            "alert_id": str(rec.alert_id),
            "alert_library": str(rec.library),
            "alert_name": str(rec.alert_name),
            "fragment_smarts": str(rec.smarts),
            "scope": str(rec.scope),
            "reference": str(rec.reference),
        }
    return out


def load_or_build_alert_matrix(
    smiles: Sequence[str],
    cache_dir: Path,
    catalogs: Sequence[str],
    external_alerts: Sequence[Path],
) -> Tuple[pd.DataFrame, sparse.csr_matrix, str]:
    meta_path = cache_dir / "expert_alert_metadata.csv"
    matrix_path = cache_dir / "compound_expert_alert_matrix.npz"
    counts_path = cache_dir / "compound_expert_alert_counts.csv"
    if meta_path.exists() and matrix_path.exists() and counts_path.exists():
        counts = pd.read_csv(counts_path)
        if "smiles" in counts.columns and list(counts["smiles"].astype(str)) == list(map(str, smiles)):
            meta = pd.read_csv(meta_path)
            alert_matrix = sparse.load_npz(matrix_path).tocsr().astype(np.float32)
            if alert_matrix.shape[0] == len(smiles) and alert_matrix.shape[1] == len(meta):
                return meta, alert_matrix, "cache"
    meta, rdkit_catalogs = load_alert_libraries(catalogs, external_alerts)
    alert_matrix = match_alert_matrix(smiles, meta, rdkit_catalogs)
    return meta, alert_matrix, "rebuilt"


def safe_auroc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2 or np.nanstd(score) == 0:
        return float("nan")
    return float(roc_auc_score(y, score))


def oof_extratrees_scores(
    x: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
    n_estimators: int,
    n_jobs: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y, minlength=2)
    splits = int(min(n_splits, counts.min()))
    if splits < 2:
        raise RuntimeError("Not enough class members for stratified CV.")
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=int(seed))
    oof = np.zeros((len(y),), dtype=np.float32)
    rows: List[Dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(y)), y)):
        clf = ExtraTreesClassifier(
            n_estimators=int(n_estimators),
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=int(seed) + int(fold),
            n_jobs=int(n_jobs),
        )
        clf.fit(x[train_idx], y[train_idx])
        score = clf.predict_proba(x[test_idx])[:, 1].astype(np.float32)
        oof[test_idx] = score
        rows.append(
            {
                "fold": int(fold),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "positive_train": int(y[train_idx].sum()),
                "positive_test": int(y[test_idx].sum()),
                "auroc": safe_auroc(y[test_idx], score),
                "auprc": float(average_precision_score(y[test_idx], score)),
            }
        )
    return oof, {
        "folds": int(len(rows)),
        "auroc_mean": float(np.nanmean([r["auroc"] for r in rows])),
        "auroc_std": float(np.nanstd([r["auroc"] for r in rows], ddof=1)),
        "auprc_mean": float(np.mean([r["auprc"] for r in rows])),
        "auprc_std": float(np.std([r["auprc"] for r in rows], ddof=1)),
        "fold_rows": rows,
    }


def summarize_metric_repeats(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed = (
        metrics.groupby(["seed", "method"], as_index=False)
        .agg(
            tasks=("task", "nunique"),
            mean_auroc=("auroc", "mean"),
            mean_auprc=("auprc", "mean"),
            mean_lift_at_top50=("lift_at_top50", "mean"),
            mean_precision_at_top50=("precision_at_top50", "mean"),
            mean_recall_at_top50=("recall_at_top50", "mean"),
        )
        .sort_values(["method", "seed"])
    )
    rows: List[Dict[str, Any]] = []
    for method, g in per_seed.groupby("method", sort=False):
        rec: Dict[str, Any] = {
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "seeds": int(g["seed"].nunique()),
            "tasks": int(g["tasks"].max()),
        }
        for col in [
            "mean_auroc",
            "mean_auprc",
            "mean_lift_at_top50",
            "mean_precision_at_top50",
            "mean_recall_at_top50",
        ]:
            vals = pd.to_numeric(g[col], errors="coerce")
            rec[col] = float(vals.mean())
            rec[col.replace("mean_", "sd_")] = float(vals.std(ddof=1))
            rec[col.replace("mean_", "sem_")] = float(vals.sem(ddof=1))
        rows.append(rec)
    summary = pd.DataFrame(rows)
    summary["rank_by_auprc"] = summary["mean_auprc"].rank(ascending=False, method="min").astype(int)
    return summary.sort_values(["rank_by_auprc", "method"]).reset_index(drop=True), per_seed


def summarize_consistency_repeats(consistency: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed = (
        consistency.groupby("seed", as_index=False)
        .agg(
            tasks=("task", "nunique"),
            mean_spearman_log2_or=("spearman_log2_or", "mean"),
            mean_topk_label_recall=("topk_label_recall", "mean"),
            mean_label_significant_alerts_q05=("label_significant_alerts_q05", "mean"),
            mean_cgp_significant_alerts_q05=("cgp_significant_alerts_q05", "mean"),
        )
        .sort_values("seed")
    )
    rows: List[Dict[str, Any]] = []
    for col in [
        "mean_spearman_log2_or",
        "mean_topk_label_recall",
        "mean_label_significant_alerts_q05",
        "mean_cgp_significant_alerts_q05",
    ]:
        vals = pd.to_numeric(per_seed[col], errors="coerce")
        rows.append(
            {
                "metric": col,
                "seeds": int(per_seed["seed"].nunique()),
                "tasks": int(per_seed["tasks"].max()) if not per_seed.empty else 0,
                "mean": float(vals.mean()),
                "sd": float(vals.std(ddof=1)),
                "sem": float(vals.sem(ddof=1)),
            }
        )
    return per_seed, pd.DataFrame(rows)


def plot_metric_repeats(per_seed: pd.DataFrame, output_dir: Path) -> None:
    plot_df = per_seed[per_seed["method"].isin(METHOD_ORDER)].copy()
    plot_df["method_label"] = plot_df["method"].map(METHOD_LABELS)
    order_labels = [METHOD_LABELS[x] for x in METHOD_ORDER]
    palette = [METHOD_COLORS[x] for x in METHOD_ORDER]
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6))
    specs = [
        ("mean_auprc", "AUPRC"),
        ("mean_auroc", "AUROC"),
        ("mean_lift_at_top50", "Lift@50"),
    ]
    for ax, (metric, title) in zip(axes, specs):
        sns.barplot(
            data=plot_df,
            x="method_label",
            y=metric,
            order=order_labels,
            errorbar="sd",
            palette=palette,
            edgecolor="white",
            linewidth=0.6,
            capsize=0.18,
            err_kws={"linewidth": 1.2},
            ax=ax,
        )
        sns.stripplot(
            data=plot_df,
            x="method_label",
            y=metric,
            order=order_labels,
            color="#222222",
            size=3.2,
            jitter=0.14,
            alpha=0.68,
            ax=ax,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels(order_labels, rotation=25, ha="right")
    fig.suptitle("ExtraTrees expert-alert prioritization", fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "fig_expert_alert_prioritization_extratrees_repeated_seed.png", dpi=320)
    fig.savefig(output_dir / "fig_expert_alert_prioritization_extratrees_repeated_seed.pdf")
    plt.close(fig)


def plot_consistency_repeats(per_seed: pd.DataFrame, output_dir: Path) -> None:
    if per_seed.empty:
        return
    plot_df = per_seed.melt(
        id_vars=["seed"],
        value_vars=["mean_spearman_log2_or", "mean_topk_label_recall"],
        var_name="metric",
        value_name="value",
    )
    labels = {
        "mean_spearman_log2_or": "Alert rank rho",
        "mean_topk_label_recall": "Top-20 recall",
    }
    plot_df["metric_label"] = plot_df["metric"].map(labels)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    sns.barplot(
        data=plot_df,
        x="metric_label",
        y="value",
        order=[labels["mean_spearman_log2_or"], labels["mean_topk_label_recall"]],
        errorbar="sd",
        palette=["#218C7E", "#D95F02"],
        capsize=0.16,
        edgecolor="white",
        linewidth=0.6,
        err_kws={"linewidth": 1.2},
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="metric_label",
        y="value",
        order=[labels["mean_spearman_log2_or"], labels["mean_topk_label_recall"]],
        color="#222222",
        size=3.2,
        jitter=0.14,
        alpha=0.7,
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("ExtraTrees CGP-guided alert consistency", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_expert_alert_consistency_extratrees_repeated_seed.png", dpi=320)
    fig.savefig(output_dir / "fig_expert_alert_consistency_extratrees_repeated_seed.pdf")
    plt.close(fig)


def write_logistic_vs_extratrees_comparison(
    output_dir: Path,
    extratrees_summary: pd.DataFrame,
    logistic_dir: Path,
) -> pd.DataFrame:
    logistic_path = logistic_dir / "expert_alert_prioritization_method_repeat_summary.csv"
    rows: List[pd.DataFrame] = []
    if logistic_path.exists():
        rows.append(pd.read_csv(logistic_path))
    rows.append(extratrees_summary)
    comp = pd.concat(rows, ignore_index=True)
    comp = comp[comp["method"].isin(COMPARABLE_METHODS)].copy()
    comp["feature_set"] = comp["method"].map(lambda x: COMPARABLE_METHODS[str(x)][0])
    comp["readout"] = comp["method"].map(lambda x: COMPARABLE_METHODS[str(x)][1])
    order_feature = ["Expert alerts", "CGP latent", "CGP + alerts"]
    order_readout = ["Logistic", "ExtraTrees"]
    comp["feature_set"] = pd.Categorical(comp["feature_set"], categories=order_feature, ordered=True)
    comp["readout"] = pd.Categorical(comp["readout"], categories=order_readout, ordered=True)
    comp = comp.sort_values(["feature_set", "readout"]).reset_index(drop=True)
    comp.to_csv(output_dir / "logistic_vs_extratrees_method_repeat_summary.csv", index=False)

    if not comp.empty:
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
        fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))
        metrics = [("mean_auprc", "sd_auprc", "AUPRC"), ("mean_lift_at_top50", "sd_lift_at_top50", "Lift@50")]
        colors = {"Logistic": "#8AA4B8", "ExtraTrees": "#D95F02"}
        width = 0.34
        x = np.arange(len(order_feature))
        for ax, (mean_col, sd_col, title) in zip(axes, metrics):
            for offset, readout in [(-width / 2, "Logistic"), (width / 2, "ExtraTrees")]:
                sub = comp[comp["readout"].astype(str).eq(readout)].set_index("feature_set")
                means = [float(sub.loc[f, mean_col]) if f in sub.index else np.nan for f in order_feature]
                errs = [float(sub.loc[f, sd_col]) if f in sub.index else np.nan for f in order_feature]
                ax.bar(
                    x + offset,
                    means,
                    width=width,
                    yerr=errs,
                    label=readout,
                    color=colors[readout],
                    edgecolor="white",
                    linewidth=0.7,
                    capsize=3,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(order_feature, rotation=18, ha="right")
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("")
        axes[0].legend(frameon=False, loc="best")
        fig.suptitle("Readout robustness for expert-alert prioritization", fontweight="bold", y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(output_dir / "fig_logistic_vs_extratrees_readout_comparison.png", dpi=320)
        fig.savefig(output_dir / "fig_logistic_vs_extratrees_readout_comparison.pdf")
        plt.close(fig)
    return comp


def write_summary(
    output_dir: Path,
    method_summary: pd.DataFrame,
    consistency_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    seeds: Sequence[int],
    num_alerts: int,
    alert_hits: int,
    num_compounds: int,
    tasks: Sequence[str],
    n_estimators: int,
    alert_source: str,
) -> None:
    best = method_summary.sort_values("mean_auprc", ascending=False).iloc[0]
    cgp = method_summary[method_summary["method"].eq("cgp_latent_extratrees")].iloc[0]
    alert = method_summary[method_summary["method"].eq("structure_alert_extratrees")].iloc[0]
    joint = method_summary[method_summary["method"].eq("cgp_plus_alert_extratrees")].iloc[0]
    rho = consistency_summary[consistency_summary["metric"].eq("mean_spearman_log2_or")].iloc[0]
    recall = consistency_summary[consistency_summary["metric"].eq("mean_topk_label_recall")].iloc[0]
    delta_joint_alert = float(joint["mean_auprc"] - alert["mean_auprc"])
    delta_joint_cgp = float(joint["mean_auprc"] - cgp["mean_auprc"])
    text = [
        "# ExtraTrees repeated-seed expert structural-alert prioritization",
        "",
        f"- Seeds: {', '.join(map(str, seeds))}",
        f"- Tasks: {len(tasks)} ToxRIC phenotype endpoints",
        f"- Compounds: {num_compounds:,}",
        f"- Expert alerts: {num_alerts:,}",
        f"- Compound-alert hits: {alert_hits:,}",
        f"- ExtraTrees estimators: {int(n_estimators)}",
        f"- Alert matrix source: {alert_source}",
        "",
        "## Mean over repeated seeds",
        "",
        (
            f"- Best ExtraTrees method: {METHOD_LABELS.get(str(best['method']), best['method'])}; "
            f"AUPRC {best['mean_auprc']:.3f} +/- {best['sd_auprc']:.3f}, "
            f"AUROC {best['mean_auroc']:.3f} +/- {best['sd_auroc']:.3f}, "
            f"Lift@50 {best['mean_lift_at_top50']:.2f} +/- {best['sd_lift_at_top50']:.2f}"
        ),
        (
            f"- CGP + alerts vs expert alerts: delta AUPRC {delta_joint_alert:+.3f}; "
            f"delta Lift@50 {joint['mean_lift_at_top50'] - alert['mean_lift_at_top50']:+.2f}"
        ),
        (
            f"- CGP + alerts vs CGP latent: delta AUPRC {delta_joint_cgp:+.3f}; "
            f"delta Lift@50 {joint['mean_lift_at_top50'] - cgp['mean_lift_at_top50']:+.2f}"
        ),
        "",
        "## ExtraTrees CGP-guided alert consistency",
        "",
        f"- Alert ranking rho: {rho['mean']:.3f} +/- {rho['sd']:.3f}",
        f"- Top-20 alert recall: {recall['mean']:.3f} +/- {recall['sd']:.3f}",
    ]
    if not comparison.empty:
        text.extend(
            [
                "",
                "## Logistic vs ExtraTrees",
                "",
            ]
        )
        for feature in ["Expert alerts", "CGP latent", "CGP + alerts"]:
            sub = comparison[comparison["feature_set"].astype(str).eq(feature)]
            vals = {
                str(r.readout): r
                for r in sub.itertuples(index=False)
                if str(r.readout) in {"Logistic", "ExtraTrees"}
            }
            if "Logistic" in vals and "ExtraTrees" in vals:
                text.append(
                    f"- {feature}: ExtraTrees AUPRC {vals['ExtraTrees'].mean_auprc:.3f} vs "
                    f"Logistic {vals['Logistic'].mean_auprc:.3f}; "
                    f"Lift@50 {vals['ExtraTrees'].mean_lift_at_top50:.2f} vs "
                    f"{vals['Logistic'].mean_lift_at_top50:.2f}"
                )
    text.append("")
    text.append("Values are reported as mean +/- SD over repeated CV seeds after endpoint averaging.")
    (output_dir / "summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in parse_list(args.seeds)]
    catalogs = parse_list(args.catalogs)

    smiles_path, emb_path = resolve_cgp_paths(args)
    smiles, z_all, smiles_to_idx = load_cgp_inputs(smiles_path, emb_path)
    tasks = load_tasks(args.toxric_dir, args.tasks)
    meta, alert_matrix, alert_source = load_or_build_alert_matrix(smiles, args.alert_cache_dir, catalogs, args.external_alerts)
    alert_matrix_dense = alert_matrix.toarray().astype(np.float32)
    alert_count_all = np.asarray(alert_matrix.sum(axis=1)).ravel().astype(np.float32)
    any_alert_all = (alert_count_all > 0).astype(np.float32)
    examples = alert_examples_from_meta(meta)

    task_cache: List[Dict[str, Any]] = []
    for task, df_task in tasks:
        idx_rows: List[int] = []
        labels: List[int] = []
        missing = 0
        for rec in df_task.itertuples(index=False):
            idx = smiles_to_idx.get(str(rec.smiles))
            if idx is None:
                missing += 1
                continue
            idx_rows.append(idx)
            labels.append(int(rec.label))
        row_idx = np.asarray(idx_rows, dtype=np.int64)
        y = np.asarray(labels, dtype=np.int64)
        if len(y) < 50 or len(np.unique(y)) < 2:
            continue
        label_alerts = mine_bit_enrichment(
            alert_matrix_dense,
            y,
            row_idx,
            task=task,
            target_name="label",
            examples=examples,
            min_support=10,
            max_prevalence=0.85,
        )
        task_cache.append(
            {
                "task": task,
                "row_idx": row_idx,
                "y": y,
                "missing": missing,
                "label_alerts": label_alerts,
            }
        )

    metric_rows: List[Dict[str, Any]] = []
    cv_rows: List[Dict[str, Any]] = []
    consistency_rows: List[Dict[str, Any]] = []
    all_alerts: List[pd.DataFrame] = [x["label_alerts"] for x in task_cache]

    for seed in seeds:
        for item in task_cache:
            task = item["task"]
            row_idx = item["row_idx"]
            y = item["y"]
            z = z_all[row_idx].astype(np.float32)
            x_alert = alert_matrix_dense[row_idx].astype(np.float32)
            x_combined = np.hstack([z, x_alert]).astype(np.float32)

            cgp_score, cgp_cv = oof_extratrees_scores(
                z,
                y,
                int(args.n_splits),
                int(seed),
                int(args.n_estimators),
                int(args.n_jobs),
            )
            alert_score, alert_cv = oof_extratrees_scores(
                x_alert,
                y,
                int(args.n_splits),
                int(seed),
                int(args.n_estimators),
                int(args.n_jobs),
            )
            combined_score, combined_cv = oof_extratrees_scores(
                x_combined,
                y,
                int(args.n_splits),
                int(seed),
                int(args.n_estimators),
                int(args.n_jobs),
            )
            alert_count = alert_count_all[row_idx]
            any_alert = any_alert_all[row_idx]
            scores = {
                "structure_any_alert": any_alert,
                "structure_alert_count": alert_count,
                "structure_alert_extratrees": alert_score,
                "cgp_latent_extratrees": cgp_score,
                "cgp_plus_alert_extratrees": combined_score,
            }
            for method, score in scores.items():
                rec = ranking_metrics(y, score, method=method, task=task)
                rec["seed"] = int(seed)
                metric_rows.append(rec)
            cv_rows.append(
                {
                    "seed": int(seed),
                    "task": task,
                    "n": int(len(y)),
                    "positive": int(y.sum()),
                    "negative": int(len(y) - y.sum()),
                    "missing_from_cgp_cache": int(item["missing"]),
                    "alert_coverage": float(any_alert.mean()),
                    "mean_alert_count": float(alert_count.mean()),
                    "cgp_auroc_mean": cgp_cv["auroc_mean"],
                    "cgp_auprc_mean": cgp_cv["auprc_mean"],
                    "alert_extratrees_auroc_mean": alert_cv["auroc_mean"],
                    "alert_extratrees_auprc_mean": alert_cv["auprc_mean"],
                    "combined_auroc_mean": combined_cv["auroc_mean"],
                    "combined_auprc_mean": combined_cv["auprc_mean"],
                }
            )

            highrisk_count = int(y.sum())
            highrisk = np.zeros_like(y)
            highrisk[np.argsort(-cgp_score)[:highrisk_count]] = 1
            cgp_alerts = mine_bit_enrichment(
                alert_matrix_dense,
                highrisk,
                row_idx,
                task=task,
                target_name="cgp_highrisk",
                examples=examples,
                min_support=10,
                max_prevalence=0.85,
            )
            if not cgp_alerts.empty:
                all_alerts.append(cgp_alerts.assign(seed=int(seed)))
            comp = compare_alert_signatures(
                item["label_alerts"],
                cgp_alerts,
                min(int(args.top_k_alerts), max(1, len(item["label_alerts"]))),
            )
            comp.update(
                {
                    "seed": int(seed),
                    "task": task,
                    "n": int(len(y)),
                    "positive": int(y.sum()),
                    "label_significant_alerts_q05": int((item["label_alerts"]["qvalue"] < 0.05).sum())
                    if not item["label_alerts"].empty
                    else 0,
                    "cgp_significant_alerts_q05": int((cgp_alerts["qvalue"] < 0.05).sum()) if not cgp_alerts.empty else 0,
                    "cgp_auprc_mean": cgp_cv["auprc_mean"],
                }
            )
            consistency_rows.append(comp)
        print(f"finished seed {seed}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    cv_summary = pd.DataFrame(cv_rows).sort_values(["seed", "task"])
    consistency = pd.DataFrame(consistency_rows).sort_values(["seed", "task"])
    alerts = pd.concat([x for x in all_alerts if x is not None and not x.empty], ignore_index=True)
    method_summary, per_seed_metrics = summarize_metric_repeats(metrics)
    per_seed_consistency, consistency_summary = summarize_consistency_repeats(consistency)
    comparison = write_logistic_vs_extratrees_comparison(args.output_dir, method_summary, args.logistic_dir)

    meta.to_csv(args.output_dir / "expert_alert_metadata.csv", index=False)
    pd.DataFrame({"smiles": smiles, "any_alert": any_alert_all.astype(int), "alert_count": alert_count_all}).to_csv(
        args.output_dir / "compound_expert_alert_counts.csv",
        index=False,
    )
    sparse.save_npz(args.output_dir / "compound_expert_alert_matrix.npz", alert_matrix)
    metrics.to_csv(args.output_dir / "expert_alert_prioritization_extratrees_metrics_by_seed.csv", index=False)
    per_seed_metrics.to_csv(args.output_dir / "expert_alert_prioritization_extratrees_method_by_seed.csv", index=False)
    method_summary.to_csv(args.output_dir / "expert_alert_prioritization_extratrees_method_repeat_summary.csv", index=False)
    cv_summary.to_csv(args.output_dir / "expert_alert_extratrees_cv_summary_by_seed.csv", index=False)
    consistency.to_csv(args.output_dir / "expert_alert_extratrees_consistency_by_seed.csv", index=False)
    per_seed_consistency.to_csv(args.output_dir / "expert_alert_extratrees_consistency_seed_mean.csv", index=False)
    consistency_summary.to_csv(args.output_dir / "expert_alert_extratrees_consistency_repeat_summary.csv", index=False)
    alerts.to_csv(args.output_dir / "expert_alert_extratrees_enrichment_long_by_seed.csv", index=False)
    top_unique_alerts(alerts, "label", n_per_task=12).to_csv(args.output_dir / "top_label_expert_alerts.csv", index=False)
    top_unique_alerts(alerts, "cgp_highrisk", n_per_task=12).to_csv(
        args.output_dir / "top_cgp_highrisk_expert_alerts.csv",
        index=False,
    )

    plot_metric_repeats(per_seed_metrics, args.output_dir)
    plot_consistency_repeats(per_seed_consistency, args.output_dir)
    plot_alert_heatmap(alerts, args.output_dir)
    write_summary(
        args.output_dir,
        method_summary,
        consistency_summary,
        comparison,
        seeds,
        int(len(meta)),
        int(alert_matrix.nnz),
        int(len(smiles)),
        [x["task"] for x in task_cache],
        int(args.n_estimators),
        alert_source,
    )
    write_json(
        args.output_dir / "manifest.json",
        {
            "toxric_dir": args.toxric_dir,
            "benchmark_dir": args.benchmark_dir,
            "cgp_smiles_path": smiles_path,
            "cgp_embeddings_path": emb_path,
            "catalogs": catalogs,
            "external_alerts": [str(x) for x in args.external_alerts],
            "alert_cache_dir": args.alert_cache_dir,
            "alert_matrix_source": alert_source,
            "seeds": seeds,
            "n_splits": int(args.n_splits),
            "n_estimators": int(args.n_estimators),
            "n_jobs": int(args.n_jobs),
            "num_compounds": int(len(smiles)),
            "num_alerts": int(len(meta)),
            "alert_hits": int(alert_matrix.nnz),
            "tasks": [x["task"] for x in task_cache],
            "readout": "ExtraTreesClassifier(max_features='sqrt', class_weight='balanced_subsample')",
            "errorbar_definition": "SD over repeated CV seeds after averaging each seed over endpoints",
            "outputs": {
                "metrics_by_seed": "expert_alert_prioritization_extratrees_metrics_by_seed.csv",
                "method_by_seed": "expert_alert_prioritization_extratrees_method_by_seed.csv",
                "method_repeat_summary": "expert_alert_prioritization_extratrees_method_repeat_summary.csv",
                "logistic_vs_extratrees": "logistic_vs_extratrees_method_repeat_summary.csv",
                "figures": [
                    "fig_expert_alert_prioritization_extratrees_repeated_seed.png",
                    "fig_logistic_vs_extratrees_readout_comparison.png",
                    "fig_expert_alert_consistency_extratrees_repeated_seed.png",
                    "fig_expert_alert_label_heatmap.png",
                ],
            },
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "seeds": seeds,
                "num_alerts": int(len(meta)),
                "alert_hits": int(alert_matrix.nnz),
                "tasks": len(task_cache),
                "best_methods_by_auprc": method_summary[
                    ["method", "mean_auprc", "sd_auprc", "mean_lift_at_top50", "sd_lift_at_top50"]
                ]
                .head(5)
                .to_dict("records"),
                "consistency": consistency_summary.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=json_default,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
