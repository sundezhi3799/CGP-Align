from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Draw, MACCSkeys


RDLogger.DisableLog("rdApp.*")


PHENOTYPE_TASKS = [
    "Endocrine Disruption_NR-ER",
    "Endocrine Disruption_SR-ARE",
    "Endocrine Disruption_SR-MMP",
    "Endocrine Disruption_SR-HSE",
    "Endocrine Disruption_SR-p53",
    "Endocrine Disruption_SR-ATAD5",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Pilot experiment for CGP-Align-assisted structural alert mining. "
            "It mines Morgan fragment enrichments from toxicity labels and from "
            "CGP-latent high-risk predictions, then compares the alert signatures."
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
        default=Path("output/cgp_align/paper/raw_results/structural_alert_mining_pilot"),
    )
    p.add_argument("--tasks", default="phenotype", help="'phenotype', 'all', or comma-separated TOXRIC task stems.")
    p.add_argument("--n_bits", type=int, default=2048)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--min_support", type=int, default=30)
    p.add_argument("--max_prevalence", type=float, default=0.45)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=41)
    return p.parse_args()


def json_default(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    raise TypeError(f"Unsupported JSON type: {type(x)!r}")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def parse_list(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def canonical_smiles(smi: object) -> str:
    text = str(smi or "").strip()
    if not text:
        return ""
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)


def load_tasks(toxric_dir: Path, task_arg: str) -> List[Tuple[str, pd.DataFrame]]:
    if not toxric_dir.exists():
        raise FileNotFoundError(f"TOXRIC directory not found: {toxric_dir}")
    files = sorted(toxric_dir.glob("*.csv"))
    task_text = str(task_arg).strip()
    if task_text.lower() == "phenotype":
        wanted = set(PHENOTYPE_TASKS)
        files = [p for p in files if p.stem in wanted]
    elif task_text.lower() != "all":
        wanted = set(parse_list(task_text))
        files = [p for p in files if p.name in wanted or p.stem in wanted]

    tasks: List[Tuple[str, pd.DataFrame]] = []
    for path in files:
        df = pd.read_csv(path)
        if "Canonical SMILES" not in df.columns or "Toxicity Value" not in df.columns:
            continue
        cols = ["Canonical SMILES", "Toxicity Value"]
        if "Name" in df.columns:
            cols.append("Name")
        if "InChIKey" in df.columns:
            cols.append("InChIKey")
        df = df[cols].copy()
        df["smiles"] = df["Canonical SMILES"].map(canonical_smiles)
        df["label"] = pd.to_numeric(df["Toxicity Value"], errors="coerce")
        df = df[(df["smiles"] != "") & df["label"].isin([0, 1])].copy()
        df["label"] = df["label"].astype(int)
        df = df.drop_duplicates(["smiles", "label"]).reset_index(drop=True)
        conflicts = df.groupby("smiles")["label"].nunique()
        bad = set(conflicts[conflicts > 1].index)
        if bad:
            df = df[~df["smiles"].isin(bad)].copy().reset_index(drop=True)
        if len(df) >= 50 and df["label"].nunique() == 2:
            tasks.append((path.stem, df))

    if task_text.lower() == "phenotype":
        order = {name: i for i, name in enumerate(PHENOTYPE_TASKS)}
        tasks = sorted(tasks, key=lambda x: order.get(x[0], 10_000))
    if not tasks:
        raise RuntimeError(f"No usable tasks found for --tasks {task_arg!r}")
    return tasks


def resolve_cgp_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    smiles_path = args.cgp_smiles_path or (args.benchmark_dir / "toxric_benchmark_unique_smiles.json")
    emb_path = args.cgp_embeddings_path or (args.benchmark_dir / "feature_cache" / "CGP_sep_k32.npy")
    if not smiles_path.exists():
        raise FileNotFoundError(f"CGP smiles path not found: {smiles_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"CGP embeddings path not found: {emb_path}")
    return smiles_path, emb_path


def load_cgp_inputs(smiles_path: Path, emb_path: Path) -> Tuple[List[str], np.ndarray, Dict[str, int]]:
    smiles_raw = json.loads(smiles_path.read_text(encoding="utf-8"))
    smiles = [canonical_smiles(x) for x in smiles_raw]
    if any(not x for x in smiles):
        bad = sum(1 for x in smiles if not x)
        raise RuntimeError(f"CGP smiles contains {bad} invalid entries.")
    z = np.load(emb_path).astype(np.float32)
    if z.shape[0] != len(smiles):
        raise RuntimeError(f"SMILES / embedding row mismatch: {len(smiles)} vs {z.shape[0]}")
    idx = {s: i for i, s in enumerate(smiles)}
    return smiles, z, idx


def fragment_from_bit(mol: Chem.Mol, atom_id: int, radius: int) -> str:
    if radius <= 0:
        return Chem.MolFragmentToSmiles(mol, atomsToUse=[int(atom_id)], canonical=True)
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, int(radius), int(atom_id))
    if not env:
        return Chem.MolFragmentToSmiles(mol, atomsToUse=[int(atom_id)], canonical=True)
    submol = Chem.PathToSubmol(mol, env)
    return Chem.MolToSmiles(submol, canonical=True)


def build_morgan_matrix(
    smiles: Sequence[str],
    n_bits: int,
    radius: int,
) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]]]:
    x = np.zeros((len(smiles), int(n_bits)), dtype=np.uint8)
    examples: Dict[int, Dict[str, Any]] = {}
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        bit_info: Dict[int, List[Tuple[int, int]]] = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, int(radius), nBits=int(n_bits), bitInfo=bit_info)
        arr = np.zeros((int(n_bits),), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        x[i] = arr.astype(np.uint8)
        for bit, envs in bit_info.items():
            if bit in examples or not envs:
                continue
            atom_id, env_radius = envs[0]
            frag = fragment_from_bit(mol, atom_id, env_radius)
            examples[int(bit)] = {
                "fragment_smiles": frag,
                "example_smiles": smi,
                "example_row": int(i),
                "example_atom": int(atom_id),
                "example_radius": int(env_radius),
            }
    return x, examples


def build_maccs_matrix(smiles: Sequence[str]) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]]]:
    n_bits = 167
    x = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    examples: Dict[int, Dict[str, Any]] = {}
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        x[i] = arr.astype(np.uint8)
        for bit in np.where(arr > 0)[0]:
            key = int(bit)
            if key in examples:
                continue
            smarts = MACCSkeys.smartsPatts.get(key, ("", 0))[0]
            examples[key] = {
                "fragment_smarts": smarts,
                "fragment_smiles": smarts,
                "example_smiles": smi,
                "example_row": int(i),
            }
    return x, examples


def bh_qvalues(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=np.float64)
    q = np.full_like(p, np.nan, dtype=np.float64)
    valid = np.isfinite(p)
    if not valid.any():
        return q
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = float(len(ranked))
    vals = ranked * m / np.arange(1, len(ranked) + 1, dtype=np.float64)
    vals = np.minimum.accumulate(vals[::-1])[::-1]
    vals = np.clip(vals, 0.0, 1.0)
    q_valid = np.empty_like(vals)
    q_valid[order] = vals
    q[valid] = q_valid
    return q


def mine_bit_enrichment(
    bit_matrix: np.ndarray,
    y: np.ndarray,
    row_idx: np.ndarray,
    task: str,
    target_name: str,
    examples: Dict[int, Dict[str, Any]],
    min_support: int,
    max_prevalence: float,
) -> pd.DataFrame:
    y = np.asarray(y, dtype=np.int64)
    rows = np.asarray(row_idx, dtype=np.int64)
    x = bit_matrix[rows].astype(bool)
    n = int(len(y))
    positives = int(y.sum())
    negatives = int(n - positives)
    support = x.sum(axis=0).astype(np.int64)
    max_support = int(math.floor(float(max_prevalence) * n))
    candidate_bits = np.where((support >= int(min_support)) & (support <= max_support))[0]

    out: List[Dict[str, Any]] = []
    for bit in candidate_bits:
        present = x[:, int(bit)]
        a = int(np.logical_and(present, y == 1).sum())
        b = int(np.logical_and(present, y == 0).sum())
        c = int(positives - a)
        d = int(negatives - b)
        if a == 0 and b == 0:
            continue
        odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        pvalue = fisher_exact([[a, b], [c, d]], alternative="greater").pvalue
        rec = {
            "task": task,
            "target": target_name,
            "bit": int(bit),
            "n": n,
            "positive": positives,
            "negative": negatives,
            "support": int(a + b),
            "pos_with_bit": a,
            "neg_with_bit": b,
            "pos_without_bit": c,
            "neg_without_bit": d,
            "pos_rate_with_bit": float(a / max(1, a + b)),
            "pos_rate_without_bit": float(c / max(1, c + d)),
            "odds_ratio": float(odds_ratio),
            "log2_or": float(np.log2(odds_ratio)),
            "pvalue": float(pvalue),
        }
        rec.update(examples.get(int(bit), {}))
        out.append(rec)
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["qvalue"] = bh_qvalues(df["pvalue"].to_numpy())
    return df.sort_values(["qvalue", "pvalue", "log2_or"], ascending=[True, True, False]).reset_index(drop=True)


def cgp_oof_predictions(
    z: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y, minlength=2)
    splits = int(min(n_splits, counts.min()))
    if splits < 2:
        raise RuntimeError("Not enough class members for stratified CV.")
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=int(seed))
    oof = np.zeros((len(y),), dtype=np.float32)
    fold_rows: List[Dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(z, y)):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", random_state=int(seed) + fold),
        )
        clf.fit(z[train_idx], y[train_idx])
        score = clf.predict_proba(z[test_idx])[:, 1].astype(np.float32)
        oof[test_idx] = score
        fold_rows.append(
            {
                "fold": int(fold),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "positive_train": int(y[train_idx].sum()),
                "positive_test": int(y[test_idx].sum()),
                "auroc": float(roc_auc_score(y[test_idx], score)),
                "auprc": float(average_precision_score(y[test_idx], score)),
            }
        )
    summary = {
        "folds": int(len(fold_rows)),
        "auroc_mean": float(np.mean([r["auroc"] for r in fold_rows])),
        "auroc_std": float(np.std([r["auroc"] for r in fold_rows], ddof=1)),
        "auprc_mean": float(np.mean([r["auprc"] for r in fold_rows])),
        "auprc_std": float(np.std([r["auprc"] for r in fold_rows], ddof=1)),
        "fold_rows": fold_rows,
    }
    return oof, summary


def compare_alert_signatures(label_df: pd.DataFrame, cgp_df: pd.DataFrame, top_k: int) -> Dict[str, Any]:
    if label_df.empty or cgp_df.empty:
        return {
            "spearman_log2_or": np.nan,
            "spearman_pvalue": np.nan,
            "topk_overlap": 0,
            "topk_jaccard": np.nan,
            "topk_label_recall": np.nan,
        }
    merged = label_df[["bit", "log2_or", "qvalue"]].rename(
        columns={"log2_or": "label_log2_or", "qvalue": "label_qvalue"}
    ).merge(
        cgp_df[["bit", "log2_or", "qvalue"]].rename(columns={"log2_or": "cgp_log2_or", "qvalue": "cgp_qvalue"}),
        on="bit",
        how="inner",
    )
    if len(merged) >= 3:
        rho, pval = spearmanr(merged["label_log2_or"], merged["cgp_log2_or"])
    else:
        rho, pval = np.nan, np.nan
    label_top = set(label_df.sort_values(["qvalue", "pvalue", "log2_or"], ascending=[True, True, False]).head(top_k)["bit"])
    cgp_top = set(cgp_df.sort_values(["qvalue", "pvalue", "log2_or"], ascending=[True, True, False]).head(top_k)["bit"])
    overlap = len(label_top & cgp_top)
    union = len(label_top | cgp_top)
    return {
        "shared_bits_for_correlation": int(len(merged)),
        "spearman_log2_or": float(rho) if np.isfinite(rho) else np.nan,
        "spearman_pvalue": float(pval) if np.isfinite(pval) else np.nan,
        "topk": int(top_k),
        "topk_overlap": int(overlap),
        "topk_jaccard": float(overlap / union) if union else np.nan,
        "topk_label_recall": float(overlap / max(1, len(label_top))),
    }


def top_unique_alerts(df: pd.DataFrame, target: str, n_per_task: int = 8) -> pd.DataFrame:
    rows = []
    sub = df[df["target"] == target].copy()
    for task, g in sub.groupby("task", sort=False):
        rows.append(g.sort_values(["qvalue", "pvalue", "log2_or"], ascending=[True, True, False]).head(int(n_per_task)))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot_representation_benchmark(benchmark_dir: Path, output_dir: Path) -> None:
    path = benchmark_dir / "toxric_representation_model_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    df["endpoint"] = df["task"].astype(str).str.replace("Endocrine Disruption_", "", regex=False)
    order = (
        df[df["representation"].eq("CGP_sep_k32")]
        .sort_values("auprc_mean", ascending=False)["endpoint"]
        .drop_duplicates()
        .tolist()
    )
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), sharey=True)
    palette = {"CGP_sep_k32": "#0B3C5D", "RDKit2D": "#7A8793", "Morgan512": "#D95F02"}
    for ax, metric, title in zip(axes, ["auroc_mean", "auprc_mean"], ["AUROC", "AUPRC"]):
        sns.barplot(
            data=df,
            y="endpoint",
            x=metric,
            hue="representation",
            order=order,
            palette=palette,
            ax=ax,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xlim(0, 1)
        ax.legend_.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("CGP latent improves phenotype-toxicity prediction", fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(output_dir / "fig_representation_toxicity_benchmark.png", dpi=320)
    fig.savefig(output_dir / "fig_representation_toxicity_benchmark.pdf")
    plt.close(fig)


def plot_alert_consistency(summary: pd.DataFrame, cv_summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    plot_df = summary.copy()
    plot_df["endpoint"] = plot_df["task"].astype(str).str.replace("Endocrine Disruption_", "", regex=False)
    cv = cv_summary.copy()
    cv["endpoint"] = cv["task"].astype(str).str.replace("Endocrine Disruption_", "", regex=False)
    order = cv.sort_values("auprc_mean", ascending=False)["endpoint"].tolist()
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7))
    sns.barplot(data=cv, x="endpoint", y="auprc_mean", order=order, color="#0B3C5D", ax=axes[0])
    axes[0].set_title("CGP-latent toxicity prediction", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("AUPRC")
    axes[0].tick_params(axis="x", rotation=35)

    sns.barplot(data=plot_df, x="endpoint", y="topk_label_recall", order=order, color="#218C7E", ax=axes[1])
    axes[1].set_title("Recovered label-enriched alerts", fontweight="bold")
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
    fig.savefig(output_dir / "fig_cgp_guided_alert_consistency.png", dpi=320)
    fig.savefig(output_dir / "fig_cgp_guided_alert_consistency.pdf")
    plt.close(fig)


def plot_alert_heatmap(alerts: pd.DataFrame, output_dir: Path, top_bits: int = 30) -> None:
    label = alerts[alerts["target"] == "label"].copy()
    if label.empty:
        return
    top = (
        label.sort_values(["qvalue", "pvalue", "log2_or"], ascending=[True, True, False])
        .drop_duplicates("bit")
        .head(int(top_bits))
    )
    bits = top["bit"].tolist()
    pivot = label[label["bit"].isin(bits)].pivot_table(index="bit", columns="task", values="log2_or", aggfunc="max")
    pivot = pivot.reindex(bits)
    pivot.columns = [str(c).replace("Endocrine Disruption_", "") for c in pivot.columns]
    ylabels = []
    frag_by_bit = label.drop_duplicates("bit").set_index("bit").get("fragment_smiles", pd.Series(dtype=str)).to_dict()
    for bit in pivot.index:
        frag = str(frag_by_bit.get(bit, ""))
        if len(frag) > 22:
            frag = frag[:20] + "..."
        ylabels.append(f"{int(bit)} | {frag}")
    sns.set_theme(style="white", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(9.2, 8.8))
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
    ax.set_ylabel("Morgan bit | representative fragment")
    ax.set_title("Toxicity-associated structural alert candidates", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_label_alert_heatmap.png", dpi=320)
    fig.savefig(output_dir / "fig_label_alert_heatmap.pdf")
    plt.close(fig)


def plot_maccs_alert_heatmap(alerts: pd.DataFrame, output_dir: Path, top_bits: int = 28) -> None:
    label = alerts[alerts["target"] == "label"].copy()
    if label.empty:
        return
    top = (
        label.sort_values(["qvalue", "pvalue", "log2_or"], ascending=[True, True, False])
        .drop_duplicates("bit")
        .head(int(top_bits))
    )
    bits = top["bit"].tolist()
    pivot = label[label["bit"].isin(bits)].pivot_table(index="bit", columns="task", values="log2_or", aggfunc="max")
    pivot = pivot.reindex(bits)
    pivot.columns = [str(c).replace("Endocrine Disruption_", "") for c in pivot.columns]
    smarts_by_bit = label.drop_duplicates("bit").set_index("bit").get("fragment_smarts", pd.Series(dtype=str)).to_dict()
    ylabels = []
    for bit in pivot.index:
        smarts = str(smarts_by_bit.get(bit, ""))
        if len(smarts) > 34:
            smarts = smarts[:32] + "..."
        ylabels.append(f"MACCS {int(bit)} | {smarts}")
    sns.set_theme(style="white", context="paper", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(9.6, 8.4))
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
    ax.set_ylabel("Interpretable MACCS SMARTS key")
    ax.set_title("Interpretable structural-alert candidates", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_maccs_label_alert_heatmap.png", dpi=320)
    fig.savefig(output_dir / "fig_maccs_label_alert_heatmap.pdf")
    plt.close(fig)


def plot_top_fragment_grid(alerts: pd.DataFrame, output_dir: Path, n: int = 12) -> None:
    label = (
        alerts[(alerts["target"] == "label") & alerts["fragment_smiles"].notna()]
        .sort_values(["qvalue", "pvalue", "log2_or"], ascending=[True, True, False])
        .drop_duplicates("bit")
        .head(int(n))
        .copy()
    )
    mols = []
    legends = []
    for rec in label.itertuples(index=False):
        mol = Chem.MolFromSmiles(str(rec.fragment_smiles))
        if mol is None:
            continue
        mols.append(mol)
        endpoint = str(rec.task).replace("Endocrine Disruption_", "")
        legends.append(f"{endpoint} | bit {int(rec.bit)}\nlog2OR {float(rec.log2_or):.2f}, q {float(rec.qvalue):.1e}")
    if not mols:
        return
    img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(290, 210), legends=legends, useSVG=False)
    img.save(output_dir / "fig_top_structural_alert_fragments.png")


def plot_label_vs_cgp_scatter(alerts: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for task, g in alerts.groupby("task", sort=False):
        label = g[g["target"] == "label"][["bit", "log2_or", "qvalue"]].rename(
            columns={"log2_or": "label_log2_or", "qvalue": "label_qvalue"}
        )
        cgp = g[g["target"] == "cgp_highrisk"][["bit", "log2_or", "qvalue"]].rename(
            columns={"log2_or": "cgp_log2_or", "qvalue": "cgp_qvalue"}
        )
        merged = label.merge(cgp, on="bit", how="inner")
        merged["task"] = task
        rows.append(merged)
    if not rows:
        return
    df = pd.concat(rows, ignore_index=True)
    if df.empty:
        return
    df["endpoint"] = df["task"].astype(str).str.replace("Endocrine Disruption_", "", regex=False)
    sig = (df["label_qvalue"] < 0.05) | (df["cgp_qvalue"] < 0.05)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    sns.scatterplot(
        data=df,
        x="label_log2_or",
        y="cgp_log2_or",
        hue="endpoint",
        style=sig.map({True: "FDR<0.05", False: "n.s."}),
        s=36,
        alpha=0.72,
        linewidth=0,
        ax=ax,
    )
    low = float(np.nanmin([df["label_log2_or"].min(), df["cgp_log2_or"].min(), -0.5]))
    high = float(np.nanmax([df["label_log2_or"].max(), df["cgp_log2_or"].max(), 0.5]))
    ax.plot([low, high], [low, high], color="#333333", ls="--", lw=1)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Label-enriched fragment log2(OR)")
    ax.set_ylabel("CGP-risk-enriched fragment log2(OR)")
    ax.set_title("CGP high-risk regions preserve structural-alert ranking", fontweight="bold")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_label_vs_cgp_alert_scatter.png", dpi=320)
    fig.savefig(output_dir / "fig_label_vs_cgp_alert_scatter.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    smiles_path, emb_path = resolve_cgp_paths(args)
    smiles, z_all, smiles_to_idx = load_cgp_inputs(smiles_path, emb_path)
    bit_matrix, examples = build_morgan_matrix(smiles, int(args.n_bits), int(args.radius))
    maccs_matrix, maccs_examples = build_maccs_matrix(smiles)
    tasks = load_tasks(args.toxric_dir, args.tasks)

    all_alerts: List[pd.DataFrame] = []
    all_maccs_alerts: List[pd.DataFrame] = []
    cv_summaries: List[Dict[str, Any]] = []
    cv_fold_rows: List[Dict[str, Any]] = []
    consistency_rows: List[Dict[str, Any]] = []
    maccs_consistency_rows: List[Dict[str, Any]] = []
    task_rows: List[Dict[str, Any]] = []

    for task, df_task in tasks:
        idx_rows = []
        labels = []
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

        oof, cv = cgp_oof_predictions(z_all[row_idx], y, int(args.n_splits), int(args.seed))
        cv_summary = {k: v for k, v in cv.items() if k != "fold_rows"}
        cv_summary.update(
            {
                "task": task,
                "n": int(len(y)),
                "positive": int(y.sum()),
                "negative": int(len(y) - y.sum()),
                "missing_from_cgp_cache": int(missing),
            }
        )
        cv_summaries.append(cv_summary)
        for fold_row in cv["fold_rows"]:
            fold_rec = dict(fold_row)
            fold_rec["task"] = task
            cv_fold_rows.append(fold_rec)

        label_alerts = mine_bit_enrichment(
            bit_matrix,
            y,
            row_idx,
            task=task,
            target_name="label",
            examples=examples,
            min_support=int(args.min_support),
            max_prevalence=float(args.max_prevalence),
        )
        highrisk_count = int(y.sum())
        highrisk = np.zeros_like(y)
        top_order = np.argsort(-oof)[:highrisk_count]
        highrisk[top_order] = 1
        cgp_alerts = mine_bit_enrichment(
            bit_matrix,
            highrisk,
            row_idx,
            task=task,
            target_name="cgp_highrisk",
            examples=examples,
            min_support=int(args.min_support),
            max_prevalence=float(args.max_prevalence),
        )
        all_alerts.extend([label_alerts, cgp_alerts])

        maccs_label_alerts = mine_bit_enrichment(
            maccs_matrix,
            y,
            row_idx,
            task=task,
            target_name="label",
            examples=maccs_examples,
            min_support=int(args.min_support),
            max_prevalence=float(args.max_prevalence),
        )
        maccs_cgp_alerts = mine_bit_enrichment(
            maccs_matrix,
            highrisk,
            row_idx,
            task=task,
            target_name="cgp_highrisk",
            examples=maccs_examples,
            min_support=int(args.min_support),
            max_prevalence=float(args.max_prevalence),
        )
        all_maccs_alerts.extend([maccs_label_alerts, maccs_cgp_alerts])

        comp = compare_alert_signatures(label_alerts, cgp_alerts, int(args.top_k))
        comp.update(
            {
                "task": task,
                "n": int(len(y)),
                "positive": int(y.sum()),
                "cgp_auroc_mean": cv_summary["auroc_mean"],
                "cgp_auprc_mean": cv_summary["auprc_mean"],
                "label_significant_alerts_q05": int((label_alerts["qvalue"] < 0.05).sum()) if not label_alerts.empty else 0,
                "cgp_significant_alerts_q05": int((cgp_alerts["qvalue"] < 0.05).sum()) if not cgp_alerts.empty else 0,
            }
        )
        consistency_rows.append(comp)
        maccs_comp = compare_alert_signatures(maccs_label_alerts, maccs_cgp_alerts, min(int(args.top_k), 20))
        maccs_comp.update(
            {
                "task": task,
                "n": int(len(y)),
                "positive": int(y.sum()),
                "cgp_auroc_mean": cv_summary["auroc_mean"],
                "cgp_auprc_mean": cv_summary["auprc_mean"],
                "label_significant_alerts_q05": int((maccs_label_alerts["qvalue"] < 0.05).sum())
                if not maccs_label_alerts.empty
                else 0,
                "cgp_significant_alerts_q05": int((maccs_cgp_alerts["qvalue"] < 0.05).sum())
                if not maccs_cgp_alerts.empty
                else 0,
            }
        )
        maccs_consistency_rows.append(maccs_comp)
        task_rows.append(
            {
                "task": task,
                "raw_rows": int(len(df_task)),
                "used_rows": int(len(y)),
                "positive": int(y.sum()),
                "negative": int(len(y) - y.sum()),
                "missing_from_cgp_cache": int(missing),
                "candidate_bits": int(bit_matrix[row_idx].sum(axis=0).astype(bool).sum()),
            }
        )

    alerts = pd.concat([x for x in all_alerts if x is not None and not x.empty], ignore_index=True)
    maccs_alerts = pd.concat([x for x in all_maccs_alerts if x is not None and not x.empty], ignore_index=True)
    cv_summary_df = pd.DataFrame(cv_summaries).sort_values("task")
    cv_folds_df = pd.DataFrame(cv_fold_rows)
    consistency_df = pd.DataFrame(consistency_rows).sort_values("task")
    maccs_consistency_df = pd.DataFrame(maccs_consistency_rows).sort_values("task")
    task_summary_df = pd.DataFrame(task_rows).sort_values("task")

    alerts.to_csv(args.output_dir / "structural_alert_candidates_long.csv", index=False)
    maccs_alerts.to_csv(args.output_dir / "maccs_alert_candidates_long.csv", index=False)
    top_unique_alerts(alerts, "label", n_per_task=12).to_csv(args.output_dir / "top_label_structural_alert_candidates.csv", index=False)
    top_unique_alerts(alerts, "cgp_highrisk", n_per_task=12).to_csv(
        args.output_dir / "top_cgp_highrisk_structural_alert_candidates.csv", index=False
    )
    top_unique_alerts(maccs_alerts, "label", n_per_task=10).to_csv(
        args.output_dir / "top_label_maccs_alert_candidates.csv", index=False
    )
    top_unique_alerts(maccs_alerts, "cgp_highrisk", n_per_task=10).to_csv(
        args.output_dir / "top_cgp_highrisk_maccs_alert_candidates.csv", index=False
    )
    cv_summary_df.to_csv(args.output_dir / "cgp_latent_toxicity_cv_summary.csv", index=False)
    cv_folds_df.to_csv(args.output_dir / "cgp_latent_toxicity_cv_folds.csv", index=False)
    consistency_df.to_csv(args.output_dir / "cgp_guided_alert_consistency_summary.csv", index=False)
    maccs_consistency_df.to_csv(args.output_dir / "cgp_guided_maccs_alert_consistency_summary.csv", index=False)
    task_summary_df.to_csv(args.output_dir / "task_data_summary.csv", index=False)

    plot_representation_benchmark(args.benchmark_dir, args.output_dir)
    plot_alert_consistency(consistency_df, cv_summary_df, args.output_dir)
    plot_alert_heatmap(alerts, args.output_dir)
    plot_maccs_alert_heatmap(maccs_alerts, args.output_dir)
    plot_label_vs_cgp_scatter(alerts, args.output_dir)
    plot_top_fragment_grid(alerts, args.output_dir)

    write_json(
        args.output_dir / "manifest.json",
        {
            "toxric_dir": args.toxric_dir,
            "benchmark_dir": args.benchmark_dir,
            "cgp_smiles_path": smiles_path,
            "cgp_embeddings_path": emb_path,
            "output_dir": args.output_dir,
            "tasks": [name for name, _ in tasks],
            "num_cgp_smiles": len(smiles),
            "cgp_embedding_shape": list(z_all.shape),
            "morgan_n_bits": int(args.n_bits),
            "morgan_radius": int(args.radius),
            "min_support": int(args.min_support),
            "max_prevalence": float(args.max_prevalence),
            "top_k": int(args.top_k),
            "n_splits": int(args.n_splits),
            "seed": int(args.seed),
            "outputs": {
                "alerts": "structural_alert_candidates_long.csv",
                "top_label_alerts": "top_label_structural_alert_candidates.csv",
                "top_cgp_highrisk_alerts": "top_cgp_highrisk_structural_alert_candidates.csv",
                "maccs_alerts": "maccs_alert_candidates_long.csv",
                "top_label_maccs_alerts": "top_label_maccs_alert_candidates.csv",
                "top_cgp_highrisk_maccs_alerts": "top_cgp_highrisk_maccs_alert_candidates.csv",
                "cgp_cv_summary": "cgp_latent_toxicity_cv_summary.csv",
                "alert_consistency": "cgp_guided_alert_consistency_summary.csv",
                "maccs_alert_consistency": "cgp_guided_maccs_alert_consistency_summary.csv",
                "figures": [
                    "fig_representation_toxicity_benchmark.png",
                    "fig_cgp_guided_alert_consistency.png",
                    "fig_label_alert_heatmap.png",
                    "fig_maccs_label_alert_heatmap.png",
                    "fig_label_vs_cgp_alert_scatter.png",
                    "fig_top_structural_alert_fragments.png",
                ],
            },
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "tasks": len(tasks),
                "alerts": int(len(alerts)),
                "maccs_alerts": int(len(maccs_alerts)),
                "mean_cgp_auprc": float(cv_summary_df["auprc_mean"].mean()) if not cv_summary_df.empty else None,
                "mean_topk_recall": float(consistency_df["topk_label_recall"].mean()) if not consistency_df.empty else None,
                "mean_alert_rank_rho": float(consistency_df["spearman_log2_or"].mean()) if not consistency_df.empty else None,
                "mean_maccs_topk_recall": float(maccs_consistency_df["topk_label_recall"].mean())
                if not maccs_consistency_df.empty
                else None,
                "mean_maccs_alert_rank_rho": float(maccs_consistency_df["spearman_log2_or"].mean())
                if not maccs_consistency_df.empty
                else None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
