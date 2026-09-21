from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import FilterCatalog
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_cgp_structural_alert_mining_pilot import (  # noqa: E402
    compare_alert_signatures,
    cgp_oof_predictions,
    json_default,
    load_cgp_inputs,
    load_tasks,
    mine_bit_enrichment,
    resolve_cgp_paths,
    top_unique_alerts,
    write_json,
)


RDLogger.DisableLog("rdApp.*")


CATALOG_ENUMS = FilterCatalog.FilterCatalogParams.FilterCatalogs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate expert structural-alert libraries for toxicity prioritization, "
            "and test whether CGP-Align risk scores improve alert prioritization."
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
        default=Path("output/cgp_align/paper/raw_results/expert_alert_prioritization"),
    )
    p.add_argument("--tasks", default="phenotype", help="'phenotype', 'all', or comma-separated TOXRIC task stems.")
    p.add_argument("--catalogs", default="PAINS,BRENK,NIH,ZINC")
    p.add_argument(
        "--external_alerts",
        type=Path,
        nargs="*",
        default=[],
        help="Optional CSV/TSV files with columns: library,name,smarts[,scope,reference].",
    )
    p.add_argument("--top_k_alerts", type=int, default=20)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=41)
    return p.parse_args()


def parse_list(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:+-]+", "_", str(text).strip()).strip("_")[:180]


def entry_prop(entry: Any, key: str, default: str = "") -> str:
    props = set(str(x) for x in entry.GetPropList())
    if key not in props:
        return default
    try:
        return str(entry.GetProp(key))
    except Exception:
        return default


def entry_key(entry: Any, fallback_catalog: str, fallback_idx: int) -> str:
    filter_set = entry_prop(entry, "FilterSet", fallback_catalog) or fallback_catalog
    desc = entry.GetDescription()
    return f"{safe_id(filter_set)}::{safe_id(desc or fallback_idx)}"


def build_rdkit_catalog(catalog_name: str) -> Tuple[str, Optional[FilterCatalog.FilterCatalog], List[Dict[str, Any]]]:
    name = str(catalog_name).strip().upper()
    if not name:
        return name, None, []
    if not hasattr(CATALOG_ENUMS, name):
        raise KeyError(f"RDKit FilterCatalog does not contain catalog {catalog_name!r}")
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(getattr(CATALOG_ENUMS, name))
    catalog = FilterCatalog.FilterCatalog(params)
    rows: List[Dict[str, Any]] = []
    for i in range(catalog.GetNumEntries()):
        entry = catalog.GetEntryWithIdx(i)
        filter_set = entry_prop(entry, "FilterSet", name) or name
        desc = entry.GetDescription()
        rows.append(
            {
                "source": "rdkit",
                "catalog": name,
                "library": filter_set,
                "alert_name": desc,
                "alert_id": entry_key(entry, name, i),
                "catalog_index": int(i),
                "scope": entry_prop(entry, "Scope", ""),
                "reference": entry_prop(entry, "Reference", ""),
                "smarts": "",
            }
        )
    return name, catalog, rows


def read_external_alert_file(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep)
    lower = {c.lower(): c for c in df.columns}
    required = ["smarts"]
    for col in required:
        if col not in lower:
            raise ValueError(f"External alert file {path} lacks required column {col!r}")
    lib_col = lower.get("library") or lower.get("catalog") or lower.get("package") or lower.get("source")
    name_col = lower.get("name") or lower.get("alert_name") or lower.get("description")
    smarts_col = lower["smarts"]
    scope_col = lower.get("scope") or lower.get("endpoints")
    ref_col = lower.get("reference")
    out = pd.DataFrame(
        {
            "source": "external",
            "catalog": path.stem,
            "library": df[lib_col].astype(str) if lib_col else path.stem,
            "alert_name": df[name_col].astype(str) if name_col else [f"{path.stem}_{i}" for i in range(len(df))],
            "smarts": df[smarts_col].astype(str),
            "scope": df[scope_col].astype(str) if scope_col else "",
            "reference": df[ref_col].astype(str) if ref_col else "",
        }
    )
    out["alert_id"] = [f"{safe_id(a)}::{safe_id(b)}" for a, b in zip(out["library"], out["alert_name"])]
    out["catalog_index"] = np.arange(len(out), dtype=int)
    return out


def load_alert_libraries(catalog_names: Sequence[str], external_paths: Sequence[Path]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    metadata_rows: List[Dict[str, Any]] = []
    rdkit_catalogs: Dict[str, Any] = {}
    for name in catalog_names:
        cname, catalog, rows = build_rdkit_catalog(name)
        if catalog is not None:
            rdkit_catalogs[cname] = catalog
            metadata_rows.extend(rows)
    external_rows: List[pd.DataFrame] = []
    for path in external_paths:
        if not path.exists():
            raise FileNotFoundError(f"External alert file not found: {path}")
        external_rows.append(read_external_alert_file(path))
    if external_rows:
        metadata_rows.extend(pd.concat(external_rows, ignore_index=True).to_dict("records"))
    meta = pd.DataFrame(metadata_rows).drop_duplicates("alert_id").reset_index(drop=True)
    meta["alert_col"] = np.arange(len(meta), dtype=int)
    return meta, rdkit_catalogs


def match_alert_matrix(smiles: Sequence[str], meta: pd.DataFrame, rdkit_catalogs: Dict[str, Any]) -> sparse.csr_matrix:
    id_to_col = dict(zip(meta["alert_id"].astype(str), meta["alert_col"].astype(int)))
    ext_patterns: List[Tuple[str, Chem.Mol]] = []
    for rec in meta[meta["source"].eq("external")].itertuples(index=False):
        mol = Chem.MolFromSmarts(str(rec.smarts))
        if mol is not None:
            ext_patterns.append((str(rec.alert_id), mol))

    rows: List[int] = []
    cols: List[int] = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        seen: set[int] = set()
        for catalog_name, catalog in rdkit_catalogs.items():
            for entry in catalog.GetMatches(mol):
                key = entry_key(entry, catalog_name, -1)
                col = id_to_col.get(key)
                if col is not None and col not in seen:
                    rows.append(i)
                    cols.append(col)
                    seen.add(col)
        for key, patt in ext_patterns:
            col = id_to_col.get(key)
            if col is None or col in seen:
                continue
            if mol.HasSubstructMatch(patt):
                rows.append(i)
                cols.append(col)
                seen.add(col)
    data = np.ones((len(rows),), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(smiles), len(meta)), dtype=np.float32)


def oof_logistic_scores(
    x: np.ndarray | sparse.spmatrix,
    y: np.ndarray,
    n_splits: int,
    seed: int,
    sparse_input: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y, minlength=2)
    splits = int(min(n_splits, counts.min()))
    if splits < 2:
        raise RuntimeError("Not enough class members for stratified CV.")
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=int(seed))
    oof = np.zeros((len(y),), dtype=np.float32)
    rows: List[Dict[str, Any]] = []
    scaler = MaxAbsScaler() if sparse_input else StandardScaler()
    for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(y)), y)):
        clf = make_pipeline(
            scaler,
            LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", random_state=int(seed) + fold),
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


def safe_auroc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2 or np.nanstd(score) == 0:
        return float("nan")
    return float(roc_auc_score(y, score))


def ranking_metrics(y: np.ndarray, score: np.ndarray, method: str, task: str) -> Dict[str, Any]:
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    n = int(len(y))
    positives = int(y.sum())
    prevalence = positives / max(1, n)
    order = np.argsort(-score)
    rec: Dict[str, Any] = {
        "task": task,
        "method": method,
        "n": n,
        "positive": positives,
        "prevalence": float(prevalence),
        "auroc": safe_auroc(y, score),
        "auprc": float(average_precision_score(y, score)) if np.nanstd(score) > 0 else float(prevalence),
    }
    for k in [50, 100, max(1, int(round(0.01 * n))), max(1, int(round(0.05 * n)))]:
        key = "top1pct" if k == max(1, int(round(0.01 * n))) else "top5pct" if k == max(1, int(round(0.05 * n))) else f"top{k}"
        kk = min(k, n)
        hits = int(y[order[:kk]].sum())
        precision = hits / max(1, kk)
        recall = hits / max(1, positives)
        rec[f"precision_at_{key}"] = float(precision)
        rec[f"recall_at_{key}"] = float(recall)
        rec[f"lift_at_{key}"] = float(precision / prevalence) if prevalence > 0 else float("nan")
    return rec


def alert_examples_from_meta(meta: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for rec in meta.itertuples(index=False):
        col = int(rec.alert_col)
        out[col] = {
            "alert_id": str(rec.alert_id),
            "alert_library": str(rec.library),
            "alert_name": str(rec.alert_name),
            "fragment_smarts": str(rec.smarts),
            "scope": str(rec.scope),
            "reference": str(rec.reference),
        }
    return out


def plot_metric_comparison(metrics: pd.DataFrame, output_dir: Path) -> None:
    if metrics.empty:
        return
    show_methods = [
        "structure_alert_count",
        "structure_alert_logistic",
        "cgp_latent_logistic",
        "cgp_alert_weighted",
        "cgp_plus_alert_logistic",
    ]
    df = metrics[metrics["method"].isin(show_methods)].copy()
    order = show_methods
    labels = {
        "structure_alert_count": "Alert count",
        "structure_alert_logistic": "Alert logistic",
        "cgp_latent_logistic": "CGP latent",
        "cgp_alert_weighted": "CGP x alert",
        "cgp_plus_alert_logistic": "CGP + alerts",
    }
    df["method_label"] = df["method"].map(labels)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.8))
    plot_specs = [
        ("auprc", "AUPRC"),
        ("auroc", "AUROC"),
        ("lift_at_top50", "Lift@50"),
    ]
    palette = ["#9AA6B2", "#5F6F7C", "#0B3C5D", "#218C7E", "#D95F02"]
    for ax, (metric, title) in zip(axes, plot_specs):
        sns.barplot(
            data=df,
            x="method",
            y=metric,
            order=order,
            errorbar="sd",
            palette=palette,
            ax=ax,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([labels[x] for x in order], rotation=35, ha="right")
    fig.suptitle("CGP-guided prioritization improves expert-alert ranking", fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "fig_expert_alert_prioritization_metrics.png", dpi=320)
    fig.savefig(output_dir / "fig_expert_alert_prioritization_metrics.pdf")
    plt.close(fig)


def plot_consistency(consistency: pd.DataFrame, cv: pd.DataFrame, output_dir: Path) -> None:
    if consistency.empty:
        return
    plot_df = consistency.copy()
    plot_df["endpoint"] = plot_df["task"].astype(str).str.replace("Endocrine Disruption_", "", regex=False)
    cv = cv.copy()
    cv["endpoint"] = cv["task"].astype(str).str.replace("Endocrine Disruption_", "", regex=False)
    order = cv.sort_values("auprc_mean", ascending=False)["endpoint"].tolist()
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7))
    sns.barplot(data=cv, x="endpoint", y="auprc_mean", order=order, color="#0B3C5D", ax=axes[0])
    axes[0].set_title("CGP latent prediction", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("AUPRC")
    axes[0].tick_params(axis="x", rotation=35)
    sns.barplot(data=plot_df, x="endpoint", y="topk_label_recall", order=order, color="#218C7E", ax=axes[1])
    axes[1].set_title("Recovered expert alerts", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel(f"Top-{int(plot_df['topk'].max())} recall")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=35)
    sns.barplot(data=plot_df, x="endpoint", y="spearman_log2_or", order=order, color="#D95F02", ax=axes[2])
    axes[2].axhline(0, color="#333333", lw=0.8)
    axes[2].set_title("Label vs CGP-risk alert ranking", fontweight="bold")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Spearman rho")
    axes[2].set_ylim(-0.1, 1)
    axes[2].tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_expert_alert_consistency.png", dpi=320)
    fig.savefig(output_dir / "fig_expert_alert_consistency.pdf")
    plt.close(fig)


def plot_alert_heatmap(alerts: pd.DataFrame, output_dir: Path, top_n: int = 35) -> None:
    label = alerts[alerts["target"].eq("label")].copy()
    if label.empty:
        return
    top = (
        label.sort_values(["qvalue", "pvalue", "log2_or"], ascending=[True, True, False])
        .drop_duplicates("bit")
        .head(int(top_n))
    )
    bits = top["bit"].tolist()
    pivot = label[label["bit"].isin(bits)].pivot_table(index="bit", columns="task", values="log2_or", aggfunc="max")
    pivot = pivot.reindex(bits)
    pivot.columns = [str(c).replace("Endocrine Disruption_", "") for c in pivot.columns]
    label_by_bit = (
        label.drop_duplicates("bit")
        .set_index("bit")[["alert_library", "alert_name"]]
        .apply(lambda x: f"{x['alert_library']} | {x['alert_name']}", axis=1)
        .to_dict()
    )
    ylabels = []
    for bit in pivot.index:
        text = str(label_by_bit.get(bit, bit))
        if len(text) > 42:
            text = text[:40] + "..."
        ylabels.append(text)
    sns.set_theme(style="white", context="paper", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(10.4, 9.0))
    sns.heatmap(
        pivot,
        cmap=sns.diverging_palette(240, 20, as_cmap=True),
        center=0,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "label-enrichment log2(OR)"},
        ax=ax,
    )
    ax.set_yticklabels(ylabels, rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("Expert alert")
    ax.set_title("Expert structural alerts enriched in toxicity labels", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_expert_alert_label_heatmap.png", dpi=320)
    fig.savefig(output_dir / "fig_expert_alert_label_heatmap.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    smiles_path, emb_path = resolve_cgp_paths(args)
    smiles, z_all, smiles_to_idx = load_cgp_inputs(smiles_path, emb_path)
    tasks = load_tasks(args.toxric_dir, args.tasks)
    catalogs = parse_list(args.catalogs)

    meta, rdkit_catalogs = load_alert_libraries(catalogs, args.external_alerts)
    alert_matrix = match_alert_matrix(smiles, meta, rdkit_catalogs)
    alert_matrix_dense = alert_matrix.toarray().astype(np.uint8)
    examples = alert_examples_from_meta(meta)
    alert_count_all = np.asarray(alert_matrix.sum(axis=1)).ravel().astype(np.float32)
    any_alert_all = (alert_count_all > 0).astype(np.float32)

    metric_rows: List[Dict[str, Any]] = []
    cv_summary_rows: List[Dict[str, Any]] = []
    consistency_rows: List[Dict[str, Any]] = []
    all_alerts: List[pd.DataFrame] = []

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

        cgp_score, cgp_cv = cgp_oof_predictions(z_all[row_idx], y, int(args.n_splits), int(args.seed))
        x_alert = alert_matrix[row_idx]
        alert_logit_score, alert_cv = oof_logistic_scores(x_alert, y, int(args.n_splits), int(args.seed), sparse_input=True)
        x_combined = sparse.hstack([sparse.csr_matrix(z_all[row_idx]), x_alert], format="csr")
        combined_score, combined_cv = oof_logistic_scores(
            x_combined, y, int(args.n_splits), int(args.seed), sparse_input=True
        )

        alert_count = alert_count_all[row_idx]
        any_alert = any_alert_all[row_idx]
        alert_weight = np.log1p(alert_count)
        scores = {
            "structure_any_alert": any_alert,
            "structure_alert_count": alert_count,
            "structure_alert_logistic": alert_logit_score,
            "cgp_latent_logistic": cgp_score,
            "cgp_alert_gated": cgp_score * any_alert,
            "cgp_alert_weighted": cgp_score * alert_weight,
            "cgp_plus_alert_logistic": combined_score,
        }
        for method, score in scores.items():
            metric_rows.append(ranking_metrics(y, score, method=method, task=task))

        cv_summary_rows.append(
            {
                "task": task,
                "n": int(len(y)),
                "positive": int(y.sum()),
                "negative": int(len(y) - y.sum()),
                "missing_from_cgp_cache": int(missing),
                "alert_coverage": float(any_alert.mean()),
                "mean_alert_count": float(alert_count.mean()),
                "cgp_auroc_mean": cgp_cv["auroc_mean"],
                "cgp_auprc_mean": cgp_cv["auprc_mean"],
                "alert_logistic_auroc_mean": alert_cv["auroc_mean"],
                "alert_logistic_auprc_mean": alert_cv["auprc_mean"],
                "combined_auroc_mean": combined_cv["auroc_mean"],
                "combined_auprc_mean": combined_cv["auprc_mean"],
            }
        )

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
        all_alerts.extend([label_alerts, cgp_alerts])
        comp = compare_alert_signatures(label_alerts, cgp_alerts, min(int(args.top_k_alerts), max(1, len(label_alerts))))
        comp.update(
            {
                "task": task,
                "n": int(len(y)),
                "positive": int(y.sum()),
                "label_significant_alerts_q05": int((label_alerts["qvalue"] < 0.05).sum()) if not label_alerts.empty else 0,
                "cgp_significant_alerts_q05": int((cgp_alerts["qvalue"] < 0.05).sum()) if not cgp_alerts.empty else 0,
                "cgp_auprc_mean": cgp_cv["auprc_mean"],
            }
        )
        consistency_rows.append(comp)

    metrics = pd.DataFrame(metric_rows)
    cv_summary = pd.DataFrame(cv_summary_rows).sort_values("task")
    consistency = pd.DataFrame(consistency_rows).sort_values("task")
    alerts = pd.concat([x for x in all_alerts if x is not None and not x.empty], ignore_index=True)

    meta.to_csv(args.output_dir / "expert_alert_metadata.csv", index=False)
    pd.DataFrame(
        {
            "smiles": smiles,
            "any_alert": any_alert_all.astype(int),
            "alert_count": alert_count_all,
        }
    ).to_csv(args.output_dir / "compound_expert_alert_counts.csv", index=False)
    sparse.save_npz(args.output_dir / "compound_expert_alert_matrix.npz", alert_matrix)
    metrics.to_csv(args.output_dir / "expert_alert_prioritization_metrics.csv", index=False)
    metrics.groupby("method", as_index=False).agg(
        tasks=("task", "nunique"),
        mean_auroc=("auroc", "mean"),
        mean_auprc=("auprc", "mean"),
        mean_lift_at_top50=("lift_at_top50", "mean"),
        mean_precision_at_top50=("precision_at_top50", "mean"),
        mean_recall_at_top50=("recall_at_top50", "mean"),
    ).sort_values("mean_auprc", ascending=False).to_csv(
        args.output_dir / "expert_alert_prioritization_method_summary.csv", index=False
    )
    cv_summary.to_csv(args.output_dir / "expert_alert_cv_summary.csv", index=False)
    alerts.to_csv(args.output_dir / "expert_alert_enrichment_long.csv", index=False)
    consistency.to_csv(args.output_dir / "expert_alert_consistency_summary.csv", index=False)
    top_unique_alerts(alerts, "label", n_per_task=12).to_csv(args.output_dir / "top_label_expert_alerts.csv", index=False)
    top_unique_alerts(alerts, "cgp_highrisk", n_per_task=12).to_csv(
        args.output_dir / "top_cgp_highrisk_expert_alerts.csv", index=False
    )

    plot_metric_comparison(metrics, args.output_dir)
    plot_consistency(consistency, cv_summary.rename(columns={"cgp_auprc_mean": "auprc_mean"}), args.output_dir)
    plot_alert_heatmap(alerts, args.output_dir)

    method_summary = pd.read_csv(args.output_dir / "expert_alert_prioritization_method_summary.csv")
    write_json(
        args.output_dir / "manifest.json",
        {
            "toxric_dir": args.toxric_dir,
            "benchmark_dir": args.benchmark_dir,
            "cgp_smiles_path": smiles_path,
            "cgp_embeddings_path": emb_path,
            "catalogs": catalogs,
            "external_alerts": [str(x) for x in args.external_alerts],
            "num_compounds": int(len(smiles)),
            "num_alerts": int(len(meta)),
            "alert_hits": int(alert_matrix.nnz),
            "tasks": [name for name, _ in tasks],
            "outputs": {
                "metadata": "expert_alert_metadata.csv",
                "compound_alert_counts": "compound_expert_alert_counts.csv",
                "alert_matrix": "compound_expert_alert_matrix.npz",
                "metrics": "expert_alert_prioritization_metrics.csv",
                "method_summary": "expert_alert_prioritization_method_summary.csv",
                "cv_summary": "expert_alert_cv_summary.csv",
                "enrichment": "expert_alert_enrichment_long.csv",
                "consistency": "expert_alert_consistency_summary.csv",
                "figures": [
                    "fig_expert_alert_prioritization_metrics.png",
                    "fig_expert_alert_consistency.png",
                    "fig_expert_alert_label_heatmap.png",
                ],
            },
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "num_alerts": int(len(meta)),
                "alert_hits": int(alert_matrix.nnz),
                "tasks": int(len(tasks)),
                "best_methods_by_auprc": method_summary[["method", "mean_auprc", "mean_lift_at_top50"]]
                .head(5)
                .to_dict("records"),
                "mean_alert_rank_rho": float(consistency["spearman_log2_or"].mean()) if not consistency.empty else None,
                "mean_topk_alert_recall": float(consistency["topk_label_recall"].mean()) if not consistency.empty else None,
            },
            ensure_ascii=False,
            indent=2,
            default=json_default,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
