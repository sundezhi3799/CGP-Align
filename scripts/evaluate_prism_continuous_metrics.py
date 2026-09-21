from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors

try:
    from rdkit.Avalon import pyAvalonTools
except Exception:  # pragma: no cover
    pyAvalonTools = None

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover
    wilcoxon = None

from cgp_common import CGPBundle
from prism_nyan_utils import load_nyan_similarity


RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Continuous PRISM functional similarity metrics for CGP-Align retrieval.")
    p.add_argument(
        "--input_dir",
        type=Path,
        default=Path("output/cgp_align/paper/manuscript_cgp_align_profile/prism_functional_analogue_v1"),
    )
    p.add_argument("--cgp_data_dir", type=Path, default=Path("data/cgp_cpg_full_motive_edges"))
    p.add_argument("--cgp_protein_embedding_dir", type=Path, default=Path("protein_embeddings/cgp_cpg_full_esm"))
    p.add_argument("--fingerprint_bits", type=int, default=2048)
    p.add_argument("--topks", default="10,50,100")
    p.add_argument("--low_tanimoto_thresholds", default="0.30,0.25,0.20,0.15")
    p.add_argument("--functional_topk", type=int, default=10)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--nyan_cache", type=Path, default=None)
    return p.parse_args()


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(denom, eps)).astype(np.float32)


def cosine_matrix(x: np.ndarray) -> np.ndarray:
    z = l2_normalize(x)
    sim = z @ z.T
    np.fill_diagonal(sim, -np.inf)
    return sim.astype(np.float32)


def fingerprint_list(smiles: Iterable[str], kind: str, n_bits: int) -> List[Any]:
    fps: List[Any] = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            fps.append(None)
            continue
        if kind == "morgan":
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits))
        elif kind == "atompair":
            fps.append(rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits))
        elif kind == "avalon":
            fps.append(None if pyAvalonTools is None else pyAvalonTools.GetAvalonFP(mol, nBits=n_bits))
        else:
            raise ValueError(kind)
    return fps


def tanimoto_matrix(fps: List[Any]) -> np.ndarray:
    n = len(fps)
    valid = np.asarray([fp is not None for fp in fps], dtype=bool)
    sim = np.full((n, n), -np.inf, dtype=np.float32)
    replacement = next((fp for fp in fps if fp is not None), None)
    if replacement is None:
        return sim
    filled = [fp if fp is not None else replacement for fp in fps]
    for i, fp in enumerate(fps):
        if fp is None:
            continue
        row = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, filled), dtype=np.float32)
        row[~valid] = -np.inf
        sim[i] = row
    np.fill_diagonal(sim, -np.inf)
    return sim


def ndcg_at_k(rel_ordered: np.ndarray, ideal_rel: np.ndarray, k: int) -> float:
    k = min(int(k), len(rel_ordered), len(ideal_rel))
    if k <= 0:
        return float("nan")
    discount = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    dcg = float(np.sum(rel_ordered[:k] * discount))
    idcg = float(np.sum(ideal_rel[:k] * discount))
    if idcg <= 1e-12:
        return float("nan")
    return dcg / idcg


def evaluate_score_matrix(
    method: str,
    score: np.ndarray,
    response_sim: np.ndarray,
    candidate_mask: np.ndarray,
    topks: List[int],
    functional_topk: int,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    n = score.shape[0]
    max_k = max(topks)
    for i in range(n):
        cand = np.flatnonzero(candidate_mask[i])
        cand = cand[cand != i]
        if cand.size <= max(2, functional_topk):
            continue
        resp = response_sim[i, cand].astype(np.float64)
        finite_resp = np.isfinite(resp)
        cand = cand[finite_resp]
        resp = resp[finite_resp]
        if cand.size <= max(2, functional_topk):
            continue

        rel = np.maximum(resp, 0.0)
        ideal_order = np.argsort(-rel, kind="mergesort")
        ideal_rel = rel[ideal_order]
        oracle_corr: Dict[int, float] = {}
        for k in topks:
            kk = min(k, len(resp))
            oracle_corr[k] = float(np.mean(resp[np.argsort(-resp, kind="mergesort")[:kk]]))
        random_expected_corr = float(np.mean(resp))

        method_scores = score[i, cand].astype(np.float64)
        finite_score = np.isfinite(method_scores)
        cand2 = cand[finite_score]
        resp2 = resp[finite_score]
        rel2 = rel[finite_score]
        method_scores = method_scores[finite_score]
        if cand2.size <= max(2, functional_topk):
            continue
        if rng is not None and method.lower() == "random":
            method_scores = rng.random(cand2.size)
        order = np.argsort(-method_scores, kind="mergesort")
        ranked = cand2[order]
        resp_ranked = resp2[order]
        rel_ranked = rel2[order]

        functional = cand[np.argsort(-resp, kind="mergesort")[:functional_topk]]
        rank_position = {int(j): r + 1 for r, j in enumerate(ranked)}
        func_ranks = np.asarray([rank_position.get(int(j), np.nan) for j in functional], dtype=np.float64)
        denom = max(1, len(ranked) - 1)
        func_rank_percentile = (func_ranks - 1.0) / denom
        base: Dict[str, Any] = {
            "method": method,
            "query_idx": int(i),
            "num_candidates": int(len(ranked)),
            "random_expected_corr": random_expected_corr,
            "mean_functional_neighbor_rank_percentile": float(np.nanmean(func_rank_percentile)),
            "best_functional_neighbor_rank_percentile": float(np.nanmin(func_rank_percentile)),
        }
        for k in topks:
            kk = min(k, len(ranked))
            top_resp = resp_ranked[:kk]
            top_rel = rel_ranked[:kk]
            base[f"top{k}_corr"] = float(np.mean(top_resp))
            base[f"top{k}_poscorr"] = float(np.mean(top_rel))
            base[f"ndcg{k}"] = ndcg_at_k(rel_ranked, ideal_rel, kk)
            base[f"oracle_top{k}_corr"] = oracle_corr[k]
            denom_ret = oracle_corr[k] - random_expected_corr
            base[f"retention{k}"] = (
                float((np.mean(top_resp) - random_expected_corr) / denom_ret) if abs(denom_ret) > 1e-12 else float("nan")
            )
            base[f"functional_top{functional_topk}_in_top{k}_rate"] = float(np.isin(functional, ranked[:kk]).mean())
        rows.append(base)
    return pd.DataFrame(rows)


def summarize(query_metrics: pd.DataFrame, topks: List[int], functional_topk: int) -> pd.DataFrame:
    metric_cols = ["num_candidates", "random_expected_corr", "mean_functional_neighbor_rank_percentile", "best_functional_neighbor_rank_percentile"]
    for k in topks:
        metric_cols += [
            f"top{k}_corr",
            f"top{k}_poscorr",
            f"ndcg{k}",
            f"retention{k}",
            f"functional_top{functional_topk}_in_top{k}_rate",
            f"oracle_top{k}_corr",
        ]
    rows = []
    for (setting, method), sub in query_metrics.groupby(["setting", "method"], sort=False):
        row: Dict[str, Any] = {"setting": setting, "method": method, "num_queries": int(len(sub))}
        for col in metric_cols:
            row[col] = float(sub[col].mean()) if col in sub else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def paired_stats(query_metrics: pd.DataFrame, key_metrics: List[str], bootstrap: int, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for setting in sorted(query_metrics["setting"].unique()):
        sub = query_metrics.loc[query_metrics["setting"] == setting]
        cgp = sub.loc[sub["method"] == "CGP-Align"].set_index("query_idx")
        for baseline in sorted(set(sub["method"]) - {"CGP-Align", "Random"}):
            base = sub.loc[sub["method"] == baseline].set_index("query_idx")
            common = cgp.index.intersection(base.index)
            if len(common) == 0:
                continue
            for metric in key_metrics:
                c = cgp.loc[common, metric].to_numpy(dtype=float)
                b = base.loc[common, metric].to_numpy(dtype=float)
                finite = np.isfinite(c) & np.isfinite(b)
                c = c[finite]
                b = b[finite]
                if len(c) == 0:
                    continue
                lower_is_better = metric.endswith("rank_percentile")
                delta = (b - c) if lower_is_better else (c - b)
                boot = []
                for _ in range(int(bootstrap)):
                    idx = rng.integers(0, len(delta), len(delta))
                    boot.append(float(np.mean(delta[idx])))
                ci = np.percentile(boot, [2.5, 97.5])
                p = float("nan")
                if wilcoxon is not None and np.any(np.abs(delta) > 1e-12):
                    try:
                        p = float(wilcoxon(delta, alternative="greater").pvalue)
                    except Exception:
                        p = float("nan")
                rows.append(
                    {
                        "setting": setting,
                        "baseline": baseline,
                        "metric": metric,
                        "direction": "baseline_minus_cgp" if lower_is_better else "cgp_minus_baseline",
                        "n_queries": int(len(delta)),
                        "mean_delta": float(np.mean(delta)),
                        "ci95_low": float(ci[0]),
                        "ci95_high": float(ci[1]),
                        "paired_win_rate": float(np.mean(delta > 0)),
                        "wilcoxon_greater_p": p,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    topks = parse_int_list(args.topks)
    thresholds = parse_float_list(args.low_tanimoto_thresholds)
    mat_path = args.input_dir / "prism_functional_retrieval_matrices.npz"
    overlap_path = args.input_dir / "prism_cgp_overlap_compounds.csv"
    if not mat_path.exists() or not overlap_path.exists():
        raise FileNotFoundError("Run scripts/run_prism_functional_analogue_retrieval.py first.")

    mat = np.load(mat_path, allow_pickle=True)
    overlap = pd.read_csv(overlap_path)
    response_sim = mat["response_similarity"].astype(np.float32)
    cgp_sim = mat["cgp_similarity"].astype(np.float32)
    morgan_sim = mat["morgan_tanimoto"].astype(np.float32)
    cgp_rows = mat["cgp_rows"].astype(np.int64)

    cgp_bundle = CGPBundle(args.cgp_data_dir, args.cgp_protein_embedding_dir)
    smiles = overlap["canonical_smiles"].astype(str).tolist()
    scores: Dict[str, np.ndarray] = {
        "CGP-Align": cgp_sim,
        "RDKit2D": cosine_matrix(cgp_bundle.structure_features[cgp_rows]),
        "NYAN": load_nyan_similarity(args.input_dir, smiles, args.nyan_cache),
        "Morgan": morgan_sim,
        "AtomPair": tanimoto_matrix(fingerprint_list(smiles, "atompair", args.fingerprint_bits)),
        "Random": np.zeros_like(cgp_sim, dtype=np.float32),
    }
    avalon = tanimoto_matrix(fingerprint_list(smiles, "avalon", args.fingerprint_bits))
    if np.isfinite(avalon).any():
        scores["Avalon"] = avalon

    n = response_sim.shape[0]
    masks: Dict[str, np.ndarray] = {"all_candidates": np.ones((n, n), dtype=bool)}
    np.fill_diagonal(masks["all_candidates"], False)
    for thr in thresholds:
        mask = (morgan_sim < float(thr)) & np.isfinite(morgan_sim)
        np.fill_diagonal(mask, False)
        masks[f"low_morgan_lt_{thr:.2f}"] = mask

    all_query_rows: List[pd.DataFrame] = []
    for setting, mask in masks.items():
        for method, score in scores.items():
            rng = np.random.default_rng(args.seed + 10007) if method == "Random" else None
            q = evaluate_score_matrix(
                method=method,
                score=score,
                response_sim=response_sim,
                candidate_mask=mask,
                topks=topks,
                functional_topk=args.functional_topk,
                rng=rng,
            )
            q.insert(0, "setting", setting)
            all_query_rows.append(q)
    query_metrics = pd.concat(all_query_rows, ignore_index=True)
    summary = summarize(query_metrics, topks, args.functional_topk)

    method_order = {"CGP-Align": 0, "RDKit2D": 1, "NYAN": 2, "Morgan": 3, "AtomPair": 4, "Avalon": 5, "Random": 9}
    summary["_method_order"] = summary["method"].map(method_order).fillna(8)
    summary = summary.sort_values(["setting", "_method_order"]).drop(columns=["_method_order"]).reset_index(drop=True)

    key_metrics = [
        "top50_corr",
        "ndcg50",
        "retention50",
        f"functional_top{args.functional_topk}_in_top50_rate",
        "mean_functional_neighbor_rank_percentile",
    ]
    stats = paired_stats(query_metrics, key_metrics, args.bootstrap, args.seed)

    query_metrics.to_csv(args.input_dir / "prism_continuous_query_metrics.csv", index=False)
    summary.to_csv(args.input_dir / "prism_continuous_metric_summary.csv", index=False)
    stats.to_csv(args.input_dir / "prism_continuous_paired_stats.csv", index=False)

    report = {
        "input_dir": str(args.input_dir),
        "num_compounds": int(n),
        "topks": topks,
        "functional_topk": int(args.functional_topk),
        "settings": list(masks.keys()),
        "methods": list(scores.keys()),
        "outputs": {
            "summary": str(args.input_dir / "prism_continuous_metric_summary.csv"),
            "query_metrics": str(args.input_dir / "prism_continuous_query_metrics.csv"),
            "paired_stats": str(args.input_dir / "prism_continuous_paired_stats.csv"),
        },
    }
    (args.input_dir / "prism_continuous_metric_run_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    show_cols = [
        "setting",
        "method",
        "num_queries",
        "top50_corr",
        "ndcg50",
        "retention50",
        f"functional_top{args.functional_topk}_in_top50_rate",
        "mean_functional_neighbor_rank_percentile",
    ]
    print(summary[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
