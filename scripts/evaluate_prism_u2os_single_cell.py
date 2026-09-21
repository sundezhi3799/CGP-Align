from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

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
    p = argparse.ArgumentParser(description="U2OS-only PRISM retrieval evaluation for CGP-Align.")
    p.add_argument(
        "--input_dir",
        type=Path,
        default=Path("output/cgp_align/paper/manuscript_cgp_align_profile/prism_functional_analogue_v1"),
    )
    p.add_argument("--cell_line", default="ACH-000364", help="DepMap ID for U2OS.")
    p.add_argument("--cgp_data_dir", type=Path, default=Path("data/cgp_cpg_full_motive_edges"))
    p.add_argument("--cgp_protein_embedding_dir", type=Path, default=Path("protein_embeddings/cgp_cpg_full_esm"))
    p.add_argument("--fingerprint_bits", type=int, default=2048)
    p.add_argument("--low_tanimoto_thresholds", default="0.30,0.25,0.20,0.15")
    p.add_argument("--active_thresholds", default="-1.0,-1.5,-2.0")
    p.add_argument("--topks", default="10,50,100")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--nyan_cache", type=Path, default=None)
    return p.parse_args()


def parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_floats(text: str) -> List[float]:
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


def unordered_pair(i: int, j: int) -> Tuple[int, int]:
    return (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))


def evaluate_method(
    method: str,
    score: np.ndarray,
    u2os_response: np.ndarray,
    active: np.ndarray,
    query_active: np.ndarray,
    candidate_mask: np.ndarray,
    topk: int,
    rng: np.random.Generator | None = None,
) -> Tuple[Dict[str, Any], pd.DataFrame, Set[Tuple[int, int]]]:
    rows: List[Dict[str, Any]] = []
    pairs: Set[Tuple[int, int]] = set()
    n = score.shape[0]
    query_rows: List[Dict[str, Any]] = []
    for i in range(n):
        if not bool(query_active[i]) or not np.isfinite(u2os_response[i]):
            continue
        cand = np.flatnonzero(candidate_mask[i])
        cand = cand[(cand != i) & np.isfinite(u2os_response[cand])]
        if cand.size == 0:
            continue
        s = score[i, cand].copy()
        finite_score = np.isfinite(s)
        cand = cand[finite_score]
        s = s[finite_score]
        if cand.size == 0:
            continue
        if rng is not None and method.lower() == "random":
            s = rng.random(cand.size)
        order = np.argsort(-s, kind="mergesort")
        kk = min(int(topk), len(order))
        ranked = cand[order[:kk]]
        ranked_score = s[order[:kk]]
        yq = float(u2os_response[i])
        y = u2os_response[ranked].astype(float)
        a = active[ranked].astype(bool)
        base_rate = float(active[cand].mean()) if len(cand) else float("nan")
        active_hits = np.flatnonzero(a)
        for loc in active_hits:
            j = int(ranked[int(loc)])
            pairs.add(unordered_pair(i, j))
            rows.append(
                {
                    "method": method,
                    "query_idx": int(i),
                    "retrieved_idx": j,
                    "rank": int(loc + 1),
                    "method_score": float(ranked_score[int(loc)]),
                    "query_u2os_lfc": yq,
                    "retrieved_u2os_lfc": float(u2os_response[j]),
                    "u2os_absdiff": float(abs(yq - float(u2os_response[j]))),
                }
            )
        top_active_frac = float(a.mean()) if kk else float("nan")
        query_rows.append(
            {
                "method": method,
                "query_idx": int(i),
                "num_candidates": int(len(cand)),
                "topk_used": int(kk),
                "candidate_active_rate": base_rate,
                "top_active_fraction": top_active_frac,
                "active_lift": float(top_active_frac / base_rate) if base_rate > 0 else float("nan"),
                "top_active_count": int(a.sum()),
                "has_active_hit": float(a.any()),
                "top_mean_lfc": float(np.mean(y)) if len(y) else float("nan"),
                "top_mean_absdiff": float(np.mean(np.abs(y - yq))) if len(y) else float("nan"),
                "top_median_absdiff": float(np.median(np.abs(y - yq))) if len(y) else float("nan"),
            }
        )
    q = pd.DataFrame(query_rows)
    h = pd.DataFrame(rows)
    if q.empty:
        summary = {
            "method": method,
            "active_queries": 0,
            "topk": int(topk),
            "directed_active_hits": 0,
            "unique_active_pairs": 0,
            "query_hit_rate": float("nan"),
            "top_active_fraction": float("nan"),
            "candidate_active_rate": float("nan"),
            "active_lift": float("nan"),
            "top_mean_absdiff": float("nan"),
            "top_median_absdiff": float("nan"),
            "hit_yield_per_1000_ranked": float("nan"),
        }
    else:
        summary = {
            "method": method,
            "active_queries": int(len(q)),
            "topk": int(topk),
            "directed_active_hits": int(len(h)),
            "unique_active_pairs": int(len(pairs)),
            "query_hit_rate": float(q["has_active_hit"].mean()),
            "top_active_fraction": float(q["top_active_fraction"].mean()),
            "candidate_active_rate": float(q["candidate_active_rate"].mean()),
            "active_lift": float(q["active_lift"].mean()),
            "top_mean_lfc": float(q["top_mean_lfc"].mean()),
            "top_mean_absdiff": float(q["top_mean_absdiff"].mean()),
            "top_median_absdiff": float(q["top_median_absdiff"].mean()),
            "hit_yield_per_1000_ranked": float(len(h) / max(1, int(q["topk_used"].sum())) * 1000.0),
        }
    return summary, q, pairs


def paired_stats(query_metrics: pd.DataFrame, metrics: List[str], bootstrap: int, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for setting in sorted(query_metrics["setting"].unique()):
        for active_thr in sorted(query_metrics["active_threshold"].unique()):
            for topk in sorted(query_metrics["topk"].unique()):
                sub = query_metrics[
                    (query_metrics["setting"] == setting)
                    & (query_metrics["active_threshold"] == active_thr)
                    & (query_metrics["topk"] == topk)
                ]
                cgp = sub[sub["method"] == "CGP-Align"].set_index("query_idx")
                for baseline in sorted(set(sub["method"]) - {"CGP-Align", "Random"}):
                    base = sub[sub["method"] == baseline].set_index("query_idx")
                    common = cgp.index.intersection(base.index)
                    if len(common) == 0:
                        continue
                    for metric in metrics:
                        c = cgp.loc[common, metric].to_numpy(dtype=float)
                        b = base.loc[common, metric].to_numpy(dtype=float)
                        finite = np.isfinite(c) & np.isfinite(b)
                        c = c[finite]
                        b = b[finite]
                        if len(c) == 0:
                            continue
                        lower_is_better = metric.endswith("absdiff")
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
                                "active_threshold": float(active_thr),
                                "topk": int(topk),
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


def make_case_table(hit_df: pd.DataFrame, meta: pd.DataFrame, morgan_sim: np.ndarray, max_rows: int = 40) -> pd.DataFrame:
    if hit_df.empty:
        return pd.DataFrame()
    sub = hit_df.sort_values(["rank", "u2os_absdiff"], ascending=[True, True]).drop_duplicates(["query_idx", "retrieved_idx"])
    rows: List[Dict[str, Any]] = []
    for row in sub.head(max_rows).itertuples(index=False):
        i = int(row.query_idx)
        j = int(row.retrieved_idx)
        rows.append(
            {
                "query_name": meta.loc[i, "prism_name"],
                "query_moa": meta.loc[i, "prism_moa"],
                "retrieved_name": meta.loc[j, "prism_name"],
                "retrieved_moa": meta.loc[j, "prism_moa"],
                "rank": int(row.rank),
                "query_u2os_lfc": float(row.query_u2os_lfc),
                "retrieved_u2os_lfc": float(row.retrieved_u2os_lfc),
                "u2os_absdiff": float(row.u2os_absdiff),
                "morgan_tanimoto": float(morgan_sim[i, j]),
                "method_score": float(row.method_score),
                "query_inchikey": meta.loc[i, "inchikey"],
                "retrieved_inchikey": meta.loc[j, "inchikey"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    mat_path = args.input_dir / "prism_functional_retrieval_matrices.npz"
    overlap_path = args.input_dir / "prism_cgp_overlap_compounds.csv"
    if not mat_path.exists() or not overlap_path.exists():
        raise FileNotFoundError("Missing PRISM matrix/overlap outputs.")
    mat = np.load(mat_path, allow_pickle=True)
    meta = pd.read_csv(overlap_path).reset_index(drop=True)
    cell_lines = [str(x) for x in mat["cell_lines"]]
    if args.cell_line not in cell_lines:
        raise ValueError(f"{args.cell_line} not found in PRISM matrix cell lines.")
    cell_idx = cell_lines.index(args.cell_line)
    y = mat["prism_response"][:, cell_idx].astype(np.float32)
    cgp_sim = mat["cgp_similarity"].astype(np.float32)
    morgan_sim = mat["morgan_tanimoto"].astype(np.float32)
    cgp_rows = mat["cgp_rows"].astype(np.int64)

    cgp_bundle = CGPBundle(args.cgp_data_dir, args.cgp_protein_embedding_dir)
    smiles = meta["canonical_smiles"].astype(str).tolist()
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

    low_thresholds = parse_floats(args.low_tanimoto_thresholds)
    active_thresholds = parse_floats(args.active_thresholds)
    topks = parse_ints(args.topks)

    summary_rows: List[Dict[str, Any]] = []
    query_frames: List[pd.DataFrame] = []
    hit_frames: List[pd.DataFrame] = []
    pair_sets: Dict[Tuple[str, float, float, int], Set[Tuple[int, int]]] = {}
    for low_thr in low_thresholds:
        candidate_mask = (morgan_sim < float(low_thr)) & np.isfinite(morgan_sim)
        np.fill_diagonal(candidate_mask, False)
        setting = f"low_morgan_lt_{low_thr:.2f}"
        for active_thr in active_thresholds:
            active = np.isfinite(y) & (y <= float(active_thr))
            query_active = active.copy()
            for topk in topks:
                for method, score in scores.items():
                    rng = np.random.default_rng(args.seed + 10007) if method == "Random" else None
                    summary, q, pairs = evaluate_method(
                        method=method,
                        score=score,
                        u2os_response=y,
                        active=active,
                        query_active=query_active,
                        candidate_mask=candidate_mask,
                        topk=topk,
                        rng=rng,
                    )
                    summary["setting"] = setting
                    summary["low_tanimoto"] = float(low_thr)
                    summary["active_threshold"] = float(active_thr)
                    summary["num_active_compounds"] = int(active.sum())
                    summary_rows.append(summary)
                    pair_sets[(method, float(low_thr), float(active_thr), int(topk))] = pairs
                    if not q.empty:
                        q.insert(0, "setting", setting)
                        q.insert(1, "low_tanimoto", float(low_thr))
                        q.insert(2, "active_threshold", float(active_thr))
                        q.insert(3, "topk", int(topk))
                        query_frames.append(q)

    summary_df = pd.DataFrame(summary_rows)
    query_df = pd.concat(query_frames, ignore_index=True) if query_frames else pd.DataFrame()
    method_order = {"CGP-Align": 0, "RDKit2D": 1, "NYAN": 2, "Morgan": 3, "AtomPair": 4, "Avalon": 5, "Random": 9}
    summary_df["_method_order"] = summary_df["method"].map(method_order).fillna(8)
    summary_df = summary_df.sort_values(
        ["low_tanimoto", "active_threshold", "topk", "_method_order"]
    ).drop(columns=["_method_order"]).reset_index(drop=True)

    stats = paired_stats(
        query_df,
        metrics=["top_active_fraction", "active_lift", "top_mean_absdiff", "top_median_absdiff"],
        bootstrap=args.bootstrap,
        seed=args.seed,
    )

    delta_rows = []
    for low_thr in low_thresholds:
        for active_thr in active_thresholds:
            for topk in topks:
                cgp = summary_df[
                    (summary_df["method"] == "CGP-Align")
                    & (summary_df["low_tanimoto"] == float(low_thr))
                    & (summary_df["active_threshold"] == float(active_thr))
                    & (summary_df["topk"] == int(topk))
                ].iloc[0]
                cgp_pairs = pair_sets[("CGP-Align", float(low_thr), float(active_thr), int(topk))]
                for baseline in [m for m in scores if m not in {"CGP-Align", "Random"}]:
                    base = summary_df[
                        (summary_df["method"] == baseline)
                        & (summary_df["low_tanimoto"] == float(low_thr))
                        & (summary_df["active_threshold"] == float(active_thr))
                        & (summary_df["topk"] == int(topk))
                    ].iloc[0]
                    base_pairs = pair_sets[(baseline, float(low_thr), float(active_thr), int(topk))]
                    delta_rows.append(
                        {
                            "low_tanimoto": float(low_thr),
                            "active_threshold": float(active_thr),
                            "topk": int(topk),
                            "baseline": baseline,
                            "directed_active_hits_delta": int(cgp["directed_active_hits"] - base["directed_active_hits"]),
                            "directed_active_hits_relative": float(cgp["directed_active_hits"] / max(1, base["directed_active_hits"])),
                            "unique_active_pairs_delta": int(cgp["unique_active_pairs"] - base["unique_active_pairs"]),
                            "unique_active_pairs_relative": float(cgp["unique_active_pairs"] / max(1, base["unique_active_pairs"])),
                            "query_hit_rate_delta": float(cgp["query_hit_rate"] - base["query_hit_rate"]),
                            "yield_delta_per_1000": float(cgp["hit_yield_per_1000_ranked"] - base["hit_yield_per_1000_ranked"]),
                            "cgp_only_unique_pairs": int(len(cgp_pairs - base_pairs)),
                            "baseline_only_unique_pairs": int(len(base_pairs - cgp_pairs)),
                            "overlap_unique_pairs": int(len(cgp_pairs & base_pairs)),
                        }
                    )
    delta_df = pd.DataFrame(delta_rows)

    # Build detailed CGP cases for the main setting.
    main_mask = (morgan_sim < 0.20) & np.isfinite(morgan_sim)
    np.fill_diagonal(main_mask, False)
    main_active = np.isfinite(y) & (y <= -1.0)
    _, _, _ = evaluate_method("CGP-Align", cgp_sim, y, main_active, main_active, main_mask, 50)
    # Recompute detailed hit frame for cases.
    rows: List[Dict[str, Any]] = []
    for i in np.flatnonzero(main_active):
        cand = np.flatnonzero(main_mask[i])
        cand = cand[(cand != i) & np.isfinite(y[cand])]
        s = cgp_sim[i, cand]
        finite = np.isfinite(s)
        cand = cand[finite]
        s = s[finite]
        if len(cand) == 0:
            continue
        ranked = cand[np.argsort(-s, kind="mergesort")[:50]]
        ranked_score = cgp_sim[i, ranked]
        for loc, j in enumerate(ranked):
            if main_active[j]:
                rows.append(
                    {
                        "method": "CGP-Align",
                        "query_idx": int(i),
                        "retrieved_idx": int(j),
                        "rank": int(loc + 1),
                        "method_score": float(ranked_score[loc]),
                        "query_u2os_lfc": float(y[i]),
                        "retrieved_u2os_lfc": float(y[j]),
                        "u2os_absdiff": float(abs(float(y[i]) - float(y[j]))),
                    }
                )
    case_hits = pd.DataFrame(rows)
    case_table = make_case_table(case_hits, meta, morgan_sim, max_rows=50)

    out_summary = args.input_dir / "prism_u2os_single_cell_active_retrieval_summary.csv"
    out_query = args.input_dir / "prism_u2os_single_cell_query_metrics.csv"
    out_stats = args.input_dir / "prism_u2os_single_cell_paired_stats.csv"
    out_delta = args.input_dir / "prism_u2os_single_cell_delta_vs_baselines.csv"
    out_cases = args.input_dir / "prism_u2os_single_cell_cgp_cases_tan020_active_m1_top50.csv"
    summary_df.to_csv(out_summary, index=False)
    query_df.to_csv(out_query, index=False)
    stats.to_csv(out_stats, index=False)
    delta_df.to_csv(out_delta, index=False)
    case_table.to_csv(out_cases, index=False)

    report = {
        "cell_line": args.cell_line,
        "cell_line_index": int(cell_idx),
        "num_compounds": int(len(y)),
        "finite_response": int(np.isfinite(y).sum()),
        "response_distribution": {
            "mean": float(np.nanmean(y)),
            "std": float(np.nanstd(y)),
            "q05": float(np.nanquantile(y, 0.05)),
            "q10": float(np.nanquantile(y, 0.10)),
            "q20": float(np.nanquantile(y, 0.20)),
            "median": float(np.nanmedian(y)),
            "min": float(np.nanmin(y)),
            "max": float(np.nanmax(y)),
        },
        "outputs": {
            "summary": str(out_summary),
            "query_metrics": str(out_query),
            "paired_stats": str(out_stats),
            "delta_vs_baselines": str(out_delta),
            "cases": str(out_cases),
        },
    }
    (args.input_dir / "prism_u2os_single_cell_run_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    main = summary_df[
        (summary_df["low_tanimoto"] == 0.20)
        & (summary_df["active_threshold"] == -1.0)
        & (summary_df["topk"] == 50)
    ][
        [
            "method",
            "num_active_compounds",
            "directed_active_hits",
            "unique_active_pairs",
            "query_hit_rate",
            "top_active_fraction",
            "candidate_active_rate",
            "active_lift",
            "top_mean_absdiff",
            "hit_yield_per_1000_ranked",
        ]
    ]
    print(main.to_string(index=False))
    main_delta = delta_df[
        (delta_df["low_tanimoto"] == 0.20)
        & (delta_df["active_threshold"] == -1.0)
        & (delta_df["topk"] == 50)
    ]
    print(main_delta.to_string(index=False))


if __name__ == "__main__":
    main()
