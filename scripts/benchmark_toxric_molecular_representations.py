from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import ExtraTreeClassifier

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Avalon import pyAvalonTools
    from rdkit.Chem import Descriptors, MACCSkeys, rdMolDescriptors
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

    RDLogger.DisableLog("rdApp.*")
except Exception as exc:  # pragma: no cover
    Chem = None
    DataStructs = None
    pyAvalonTools = None
    Descriptors = None
    MACCSkeys = None
    rdMolDescriptors = None
    GetMorganGenerator = None
    RDKit_IMPORT_ERROR = exc
else:
    RDKit_IMPORT_ERROR = None

try:
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover
    lgb = None

try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None

try:
    from mordred import Calculator, descriptors  # type: ignore
except Exception:  # pragma: no cover
    Calculator = None
    descriptors = None


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
        description="NYAN-reuse-style molecular representation and surrogate-model benchmark on TOXRIC."
    )
    p.add_argument("--toxric_dir", type=Path, required=True, help="Directory containing TOXRIC 30 CSV files.")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--tasks", default="phenotype", help="'phenotype', 'all', or comma-separated task stems.")
    p.add_argument(
        "--representations",
        default="Morgan512,Morgan1024,Morgan2048,ECFP2_1024,MACCS,Avalon1024,AtomPair2048,TopologicalTorsion2048,RDKit2D,CGP_sep_k32",
    )
    p.add_argument("--models", default="logistic,extratrees")
    p.add_argument("--cgp_smiles_path", type=Path)
    p.add_argument("--cgp_embeddings_path", type=Path)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--n_estimators", type=int, default=240)
    p.add_argument("--max_iter", type=int, default=2000)
    p.add_argument("--force_recompute_features", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
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


def canonical_smiles(smi: str) -> str:
    text = str(smi or "").strip()
    if not text:
        return ""
    if Chem is None:
        return text
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
        df = df[["Canonical SMILES", "Toxicity Value"]].copy()
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


def mols_from_smiles(smiles: Sequence[str]) -> List[Any]:
    if Chem is None:
        raise RuntimeError(f"RDKit is required but failed to import: {RDKit_IMPORT_ERROR}")
    mols = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            mol = Chem.MolFromSmiles("")
        mols.append(mol)
    return mols


def bitvect_to_array(fp: Any, n_bits: int) -> np.ndarray:
    arr = np.zeros((int(n_bits),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


def fp_matrix(mols: Sequence[Any], name: str) -> np.ndarray:
    lname = name.lower()
    if lname.startswith("morgan"):
        n_bits = int(name.replace("Morgan", ""))
        generator = GetMorganGenerator(radius=2, fpSize=n_bits)
        return np.vstack([bitvect_to_array(generator.GetFingerprint(m), n_bits) for m in mols]).astype(np.float32)
    if lname.startswith("ecfp2"):
        n_bits = int(name.split("_")[-1])
        generator = GetMorganGenerator(radius=1, fpSize=n_bits)
        return np.vstack([bitvect_to_array(generator.GetFingerprint(m), n_bits) for m in mols]).astype(np.float32)
    if lname == "maccs":
        n_bits = 167
        return np.vstack([bitvect_to_array(MACCSkeys.GenMACCSKeys(m), n_bits) for m in mols]).astype(np.float32)
    if lname.startswith("avalon"):
        if pyAvalonTools is None:
            raise RuntimeError("RDKit Avalon is unavailable in this environment.")
        n_bits = int(name.replace("Avalon", ""))
        return np.vstack([bitvect_to_array(pyAvalonTools.GetAvalonFP(m, nBits=n_bits), n_bits) for m in mols]).astype(
            np.float32
        )
    if lname.startswith("atompair"):
        n_bits = int(name.replace("AtomPair", ""))
        return np.vstack(
            [bitvect_to_array(rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=n_bits), n_bits) for m in mols]
        ).astype(np.float32)
    if lname.startswith("topologicaltorsion"):
        n_bits = int(name.replace("TopologicalTorsion", ""))
        return np.vstack(
            [
                bitvect_to_array(rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=n_bits), n_bits)
                for m in mols
            ]
        ).astype(np.float32)
    raise KeyError(name)


def rdkit2d_matrix(mols: Sequence[Any]) -> np.ndarray:
    names = [name for name, _ in Descriptors._descList]
    funcs = [func for _, func in Descriptors._descList]
    x = np.zeros((len(mols), len(funcs)), dtype=np.float64)
    for i, mol in enumerate(mols):
        vals = []
        for func in funcs:
            try:
                vals.append(float(func(mol)))
            except Exception:
                vals.append(np.nan)
        x[i] = np.asarray(vals, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, -1.0e12, 1.0e12)
    return x.astype(np.float32)


def mordred_matrix(mols: Sequence[Any]) -> np.ndarray:
    if Calculator is None or descriptors is None:
        raise RuntimeError("mordred is not installed.")
    calc = Calculator(descriptors, ignore_3D=True)
    df = calc.pandas(list(mols), quiet=True)
    x = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    return x


def load_cgp_embeddings(all_smiles: Sequence[str], smiles_path: Path, embeddings_path: Path) -> np.ndarray:
    source_smiles = json.loads(smiles_path.read_text(encoding="utf-8"))
    source_idx = {str(s): i for i, s in enumerate(source_smiles)}
    if Chem is not None:
        for i, smi in enumerate(source_smiles):
            canon = canonical_smiles(str(smi))
            if canon and canon not in source_idx:
                source_idx[canon] = i
    missing = [s for s in all_smiles if s not in source_idx]
    if missing:
        raise RuntimeError(f"CGP embedding cache missing {len(missing)} smiles; example: {missing[:3]}")
    z_all = np.load(embeddings_path)
    rows = np.asarray([source_idx[str(s)] for s in all_smiles], dtype=np.int64)
    return z_all[rows].astype(np.float32)


def feature_cache_path(output_dir: Path, rep: str) -> Path:
    safe = rep.replace(":", "_").replace("/", "_").replace("+", "plus")
    return output_dir / "feature_cache" / f"{safe}.npy"


def build_features(
    reps: Sequence[str],
    all_smiles: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mols: Optional[List[Any]] = None
    features: Dict[str, np.ndarray] = {}
    meta: Dict[str, Any] = {"skipped_representations": {}, "feature_shapes": {}}
    for rep in reps:
        path = feature_cache_path(args.output_dir, rep)
        if path.exists() and not args.force_recompute_features:
            features[rep] = np.load(path)
            meta["feature_shapes"][rep] = list(features[rep].shape)
            continue
        try:
            if rep.startswith("CGP_"):
                if args.cgp_smiles_path is None or args.cgp_embeddings_path is None:
                    raise RuntimeError("CGP representation requires --cgp_smiles_path and --cgp_embeddings_path.")
                x = load_cgp_embeddings(all_smiles, args.cgp_smiles_path, args.cgp_embeddings_path)
            elif rep == "RDKit2D":
                if mols is None:
                    mols = mols_from_smiles(all_smiles)
                x = rdkit2d_matrix(mols)
            elif rep == "Mordred":
                if mols is None:
                    mols = mols_from_smiles(all_smiles)
                x = mordred_matrix(mols)
            else:
                if mols is None:
                    mols = mols_from_smiles(all_smiles)
                x = fp_matrix(mols, rep)
        except Exception as exc:
            meta["skipped_representations"][rep] = f"{type(exc).__name__}: {exc}"
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, x.astype(np.float32))
        features[rep] = x.astype(np.float32)
        meta["feature_shapes"][rep] = list(features[rep].shape)
    return features, meta


def make_model(name: str, seed: int, n_estimators: int, max_iter: int, n_features: int) -> Any:
    lname = name.lower()
    if lname == "logistic":
        solver = "liblinear" if n_features <= 5000 else "saga"
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=max_iter,
                class_weight="balanced",
                solver=solver,
                penalty="l2",
                random_state=seed,
                n_jobs=1 if solver == "liblinear" else -1,
            ),
        )
    if lname == "sgd":
        return make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-4,
                max_iter=max_iter,
                tol=1e-3,
                class_weight="balanced",
                random_state=seed,
            ),
        )
    if lname == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    if lname == "extratree":
        return ExtraTreeClassifier(class_weight="balanced", random_state=seed)
    if lname == "rf":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    if lname == "svm_rbf":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", probability=False, random_state=seed),
        )
    if lname == "gbdt":
        return HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.04,
            l2_regularization=0.01,
            random_state=seed,
        )
    if lname == "gradientboosting":
        return GradientBoostingClassifier(
            n_estimators=max(80, min(n_estimators, 220)),
            learning_rate=0.04,
            max_depth=3,
            subsample=0.9,
            random_state=seed,
        )
    if lname == "adaboost":
        return AdaBoostClassifier(
            n_estimators=max(80, min(n_estimators, 220)),
            learning_rate=0.5,
            random_state=seed,
        )
    if lname == "lightgbm":
        if lgb is None:
            raise RuntimeError("lightgbm is not installed.")
        return lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    if lname == "xgboost":
        if xgb is None:
            raise RuntimeError("xgboost is not installed.")
        return xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=0.04,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
    raise KeyError(name)


def predict_scores(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    pred = model.predict(x)
    return np.asarray(pred, dtype=np.float32)


def cv_splits(x: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=2)
    splits = int(min(n_splits, counts.min()))
    if splits < 2:
        return []
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    return [(tr.astype(np.int64), te.astype(np.int64)) for tr, te in cv.split(x, y)]


def evaluate_one(
    x: np.ndarray,
    y: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    task: str,
    rep: str,
    model_name: str,
    seed: int,
    n_estimators: int,
    max_iter: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        model = make_model(model_name, seed + fold, n_estimators, max_iter, int(x.shape[1]))
        model.fit(x[train_idx], y[train_idx])
        score = predict_scores(model, x[test_idx])
        rows.append(
            {
                "task": task,
                "representation": rep,
                "model": model_name,
                "fold": int(fold),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "positive_train": int(y[train_idx].sum()),
                "positive_test": int(y[test_idx].sum()),
                "auroc": float(roc_auc_score(y[test_idx], score)),
                "auprc": float(average_precision_score(y[test_idx], score)),
            }
        )
    df = pd.DataFrame(rows)
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=2)
    return rows, {
        "task": task,
        "representation": rep,
        "model": model_name,
        "method": f"{rep}+{model_name}",
        "n": int(len(y)),
        "positive": int(counts[1]),
        "negative": int(counts[0]),
        "folds": int(len(rows)),
        "auroc_mean": float(df["auroc"].mean()),
        "auroc_std": float(df["auroc"].std(ddof=1)),
        "auprc_mean": float(df["auprc"].mean()),
        "auprc_std": float(df["auprc"].std(ddof=1)),
    }


def add_comparison_tables(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    summary = summary.copy()
    summary["method"] = summary["representation"].astype(str) + "+" + summary["model"].astype(str)

    rank_rows = []
    for metric in ["auroc_mean", "auprc_mean"]:
        tmp = summary[["task", "method", "representation", "model", metric]].copy()
        tmp["metric"] = metric.replace("_mean", "")
        tmp["rank"] = tmp.groupby("task")[metric].rank(method="average", ascending=False)
        rank_rows.append(tmp)
    ranks = pd.concat(rank_rows, ignore_index=True)
    ranks.to_csv(output_dir / "toxric_representation_model_ranks.csv", index=False)

    method_agg = (
        ranks.groupby(["metric", "method", "representation", "model"], as_index=False)
        .agg(mean_rank=("rank", "mean"), median_rank=("rank", "median"), tasks=("task", "nunique"))
        .sort_values(["metric", "mean_rank", "median_rank"])
    )
    method_agg.to_csv(output_dir / "toxric_method_mean_ranks.csv", index=False)

    metric_agg = (
        summary.groupby(["representation", "model", "method"], as_index=False)
        .agg(
            tasks=("task", "nunique"),
            mean_auroc=("auroc_mean", "mean"),
            mean_auprc=("auprc_mean", "mean"),
            median_auroc=("auroc_mean", "median"),
            median_auprc=("auprc_mean", "median"),
        )
        .sort_values(["mean_auprc", "mean_auroc"], ascending=False)
    )
    metric_agg.to_csv(output_dir / "toxric_method_metric_aggregate.csv", index=False)

    baseline_methods = ["Morgan2048+logistic", "Morgan2048+extratrees"]
    for baseline in baseline_methods:
        base = summary[summary["method"] == baseline][["task", "auroc_mean", "auprc_mean"]].rename(
            columns={"auroc_mean": "baseline_auroc", "auprc_mean": "baseline_auprc"}
        )
        if base.empty:
            continue
        comp = summary.merge(base, on="task", how="inner")
        comp[f"delta_auroc_vs_{baseline}"] = comp["auroc_mean"] - comp["baseline_auroc"]
        comp[f"delta_auprc_vs_{baseline}"] = comp["auprc_mean"] - comp["baseline_auprc"]
        comp.to_csv(output_dir / f"toxric_delta_vs_{baseline.replace('+', '_')}.csv", index=False)

    traditional = summary[~summary["representation"].astype(str).str.startswith("CGP_")].copy()
    if not traditional.empty:
        best_trad = (
            traditional.sort_values(["task", "auprc_mean", "auroc_mean"], ascending=[True, False, False])
            .groupby("task")
            .head(1)[["task", "method", "auroc_mean", "auprc_mean"]]
            .rename(
                columns={
                    "method": "best_traditional_method",
                    "auroc_mean": "best_traditional_auroc",
                    "auprc_mean": "best_traditional_auprc",
                }
            )
        )
        comp = summary.merge(best_trad, on="task", how="left")
        comp["delta_auroc_vs_best_traditional"] = comp["auroc_mean"] - comp["best_traditional_auroc"]
        comp["delta_auprc_vs_best_traditional"] = comp["auprc_mean"] - comp["best_traditional_auprc"]
        comp.to_csv(output_dir / "toxric_delta_vs_best_traditional_by_task.csv", index=False)


def should_run(rep: str, model_name: str) -> bool:
    lname = model_name.lower()
    if lname in {"svm_rbf", "gbdt"}:
        return rep.startswith("CGP_") or rep in {"RDKit2D", "Mordred"}
    return True


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args.toxric_dir, args.tasks)
    all_smiles = sorted({s for _, df in tasks for s in df["smiles"].tolist()})
    smiles_to_idx = {s: i for i, s in enumerate(all_smiles)}
    (args.output_dir / "feature_cache").mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "toxric_benchmark_unique_smiles.json", all_smiles)

    reps = parse_list(args.representations)
    models = parse_list(args.models)
    features, feature_meta = build_features(reps, all_smiles, args)
    if not features:
        raise RuntimeError("No feature representation was available.")

    fold_path = args.output_dir / "toxric_representation_model_folds.csv"
    summary_path = args.output_dir / "toxric_representation_model_summary.csv"
    fold_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    if args.skip_existing:
        if fold_path.exists():
            fold_rows.extend(pd.read_csv(fold_path).to_dict("records"))
        if summary_path.exists():
            summary_rows.extend(pd.read_csv(summary_path).to_dict("records"))
    skipped_models: Dict[str, str] = {}
    print(json.dumps({"tasks": len(tasks), "unique_smiles": len(all_smiles), "features": list(features)}, sort_keys=True), flush=True)

    for task_name, df in tasks:
        idx = np.asarray([smiles_to_idx[s] for s in df["smiles"].tolist()], dtype=np.int64)
        y = df["label"].to_numpy(dtype=np.int64)
        splits = cv_splits(np.zeros((len(y), 1), dtype=np.float32), y, int(args.n_splits), int(args.seed))
        if not splits:
            continue
        for rep, x_all in features.items():
            x = np.asarray(x_all[idx], dtype=np.float32)
            for model_name in models:
                if not should_run(rep, model_name):
                    continue
                key = f"{rep}+{model_name}"
                if args.skip_existing and (args.output_dir / f"done_{task_name}__{key.replace('+', '__')}.txt").exists():
                    continue
                try:
                    rows, summary = evaluate_one(
                        x,
                        y,
                        splits,
                        task_name,
                        rep,
                        model_name,
                        int(args.seed),
                        int(args.n_estimators),
                        int(args.max_iter),
                    )
                except Exception as exc:
                    skipped_models[key] = f"{type(exc).__name__}: {exc}"
                    print(json.dumps({"skip": key, "task": task_name, "reason": skipped_models[key]}, ensure_ascii=False), flush=True)
                    continue
                fold_rows.extend(rows)
                summary_rows.append(summary)
                (args.output_dir / f"done_{task_name}__{key.replace('+', '__')}.txt").write_text("done\n", encoding="utf-8")
                print(json.dumps(summary, sort_keys=True), flush=True)

    fold_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(summary_rows)
    if not fold_df.empty:
        fold_df = fold_df.drop_duplicates(["task", "representation", "model", "fold"], keep="last")
    if not summary_df.empty:
        summary_df = summary_df.drop_duplicates(["task", "representation", "model"], keep="last")
    fold_df.to_csv(fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    add_comparison_tables(summary_df, args.output_dir)
    write_json(
        args.output_dir / "toxric_representation_model_metadata.json",
        {
            "toxric_dir": args.toxric_dir,
            "tasks": [name for name, _ in tasks],
            "unique_smiles": len(all_smiles),
            "representations_requested": reps,
            "representations_available": list(features),
            "models_requested": models,
            "n_splits": int(args.n_splits),
            "seed": int(args.seed),
            "n_estimators": int(args.n_estimators),
            "feature_meta": feature_meta,
            "skipped_models": skipped_models,
            "cgp_smiles_path": args.cgp_smiles_path,
            "cgp_embeddings_path": args.cgp_embeddings_path,
        },
    )
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "aggregate": str(args.output_dir / "toxric_method_metric_aggregate.csv"),
                "ranks": str(args.output_dir / "toxric_method_mean_ranks.csv"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
