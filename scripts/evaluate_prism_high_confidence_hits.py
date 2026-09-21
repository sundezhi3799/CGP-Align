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

from cgp_common import CGPBundle
from prism_nyan_utils import load_nyan_similarity


RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Count high-confidence PRISM functional analogue hits: low structural similarity, "
            "high response-profile correlation, and high method rank."
        )
    )
    p.add_argument(
        "--input_dir",
        type=Path,
        default=Path("output/cgp_align/paper/manuscript_cgp_align_profile/prism_functional_analogue_v1"),
    )
    p.add_argument("--cgp_data_dir", type=Path, default=Path("data/cgp_cpg_full_motive_edges"))
    p.add_argument("--cgp_protein_embedding_dir", type=Path, default=Path("protein_embeddings/cgp_cpg_full_esm"))
    p.add_argument("--fingerprint_bits", type=int, default=2048)
    p.add_argument("--low_tanimoto_thresholds", default="0.30,0.25,0.20,0.15")
    p.add_argument("--response_corr_thresholds", default="0.25,0.30,0.40")
    p.add_argument("--topks", default="10,50,100")
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


def collect_hits(
    method: str,
    score: np.ndarray,
    response_sim: np.ndarray,
    candidate_mask: np.ndarray,
    corr_threshold: float,
    topk: int,
    rng: np.random.Generator | None = None,
) -> Tuple[Dict[str, Any], pd.DataFrame, Set[Tuple[int, int]]]:
    rows: List[Dict[str, Any]] = []
    unique_pairs: Set[Tuple[int, int]] = set()
    n = score.shape[0]
    query_with_hit = 0
    query_count = 0
    topk_used_sum = 0
    for i in range(n):
        cand = np.flatnonzero(candidate_mask[i])
        cand = cand[cand != i]
        if cand.size == 0:
            continue
        resp = response_sim[i, cand]
        finite_resp = np.isfinite(resp)
        cand = cand[finite_resp]
        resp = resp[finite_resp]
        if cand.size == 0:
            continue
        s = score[i, cand].copy()
        finite_score = np.isfinite(s)
        cand = cand[finite_score]
        resp = resp[finite_score]
        s = s[finite_score]
        if cand.size == 0:
            continue
        if rng is not None and method.lower() == "random":
            s = rng.random(cand.size)
        order = np.argsort(-s, kind="mergesort")
        kk = min(int(topk), len(order))
        ranked = cand[order[:kk]]
        ranked_resp = resp[order[:kk]]
        ranked_score = s[order[:kk]]
        hit_local = np.flatnonzero(ranked_resp >= float(corr_threshold))
        query_count += 1
        topk_used_sum += kk
        if hit_local.size > 0:
            query_with_hit += 1
        for loc in hit_local:
            j = int(ranked[int(loc)])
            pair = unordered_pair(i, j)
            unique_pairs.add(pair)
            rows.append(
                {
                    "method": method,
                    "query_idx": int(i),
                    "retrieved_idx": j,
                    "rank": int(loc + 1),
                    "method_score": float(ranked_score[int(loc)]),
                    "prism_response_corr": float(ranked_resp[int(loc)]),
                }
            )
    hit_df = pd.DataFrame(rows)
    denom = max(1, topk_used_sum)
    summary = {
        "method": method,
        "num_queries": int(query_count),
        "topk": int(topk),
        "corr_threshold": float(corr_threshold),
        "directed_hits": int(len(hit_df)),
        "unique_pair_hits": int(len(unique_pairs)),
        "queries_with_hit": int(query_with_hit),
        "query_hit_rate": float(query_with_hit / max(1, query_count)),
        "hit_yield_per_1000_ranked": float(len(hit_df) / denom * 1000.0),
        "mean_hit_rank": float(hit_df["rank"].mean()) if not hit_df.empty else float("nan"),
        "median_hit_corr": float(hit_df["prism_response_corr"].median()) if not hit_df.empty else float("nan"),
    }
    return summary, hit_df, unique_pairs


def make_case_table(hit_df: pd.DataFrame, meta: pd.DataFrame, morgan_sim: np.ndarray, method: str, max_rows: int = 40) -> pd.DataFrame:
    if hit_df.empty:
        return pd.DataFrame()
    sub = hit_df.loc[hit_df["method"] == method].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.sort_values(["rank", "prism_response_corr"], ascending=[True, False]).drop_duplicates(
        ["query_idx", "retrieved_idx"]
    )
    rows = []
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
                "prism_response_corr": float(row.prism_response_corr),
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
        raise FileNotFoundError("Missing PRISM matrix/overlap outputs. Run run_prism_functional_analogue_retrieval.py first.")
    mat = np.load(mat_path, allow_pickle=True)
    meta = pd.read_csv(overlap_path).reset_index(drop=True)
    response_sim = mat["response_similarity"].astype(np.float32)
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

    thresholds = parse_floats(args.low_tanimoto_thresholds)
    corr_thresholds = parse_floats(args.response_corr_thresholds)
    topks = parse_ints(args.topks)

    summary_rows: List[Dict[str, Any]] = []
    hit_frames: List[pd.DataFrame] = []
    pair_sets: Dict[Tuple[str, float, float, int], Set[Tuple[int, int]]] = {}
    for tan_thr in thresholds:
        mask = (morgan_sim < float(tan_thr)) & np.isfinite(morgan_sim)
        np.fill_diagonal(mask, False)
        setting = f"low_morgan_lt_{tan_thr:.2f}"
        for corr_thr in corr_thresholds:
            for topk in topks:
                for method, score in scores.items():
                    rng = np.random.default_rng(args.seed + 10007) if method == "Random" else None
                    summary, hit_df, pairs = collect_hits(
                        method=method,
                        score=score,
                        response_sim=response_sim,
                        candidate_mask=mask,
                        corr_threshold=corr_thr,
                        topk=topk,
                        rng=rng,
                    )
                    summary["setting"] = setting
                    summary["low_tanimoto"] = float(tan_thr)
                    summary_rows.append(summary)
                    if not hit_df.empty:
                        hit_df.insert(0, "setting", setting)
                        hit_df.insert(1, "low_tanimoto", float(tan_thr))
                        hit_df.insert(2, "corr_threshold", float(corr_thr))
                        hit_df.insert(3, "topk", int(topk))
                        hit_frames.append(hit_df)
                    pair_sets[(method, float(tan_thr), float(corr_thr), int(topk))] = pairs

    summary_df = pd.DataFrame(summary_rows)
    method_order = {"CGP-Align": 0, "RDKit2D": 1, "NYAN": 2, "Morgan": 3, "AtomPair": 4, "Avalon": 5, "Random": 9}
    summary_df["_method_order"] = summary_df["method"].map(method_order).fillna(8)
    summary_df = summary_df.sort_values(
        ["low_tanimoto", "corr_threshold", "topk", "_method_order"]
    ).drop(columns=["_method_order"]).reset_index(drop=True)
    hit_all = pd.concat(hit_frames, ignore_index=True) if hit_frames else pd.DataFrame()

    delta_rows = []
    for tan_thr in thresholds:
        for corr_thr in corr_thresholds:
            for topk in topks:
                cgp = summary_df[
                    (summary_df["method"] == "CGP-Align")
                    & (summary_df["low_tanimoto"] == float(tan_thr))
                    & (summary_df["corr_threshold"] == float(corr_thr))
                    & (summary_df["topk"] == int(topk))
                ].iloc[0]
                cgp_pairs = pair_sets[("CGP-Align", float(tan_thr), float(corr_thr), int(topk))]
                for baseline in [m for m in scores if m not in {"CGP-Align", "Random"}]:
                    base = summary_df[
                        (summary_df["method"] == baseline)
                        & (summary_df["low_tanimoto"] == float(tan_thr))
                        & (summary_df["corr_threshold"] == float(corr_thr))
                        & (summary_df["topk"] == int(topk))
                    ].iloc[0]
                    base_pairs = pair_sets[(baseline, float(tan_thr), float(corr_thr), int(topk))]
                    delta_rows.append(
                        {
                            "low_tanimoto": float(tan_thr),
                            "corr_threshold": float(corr_thr),
                            "topk": int(topk),
                            "baseline": baseline,
                            "directed_hits_delta": int(cgp["directed_hits"] - base["directed_hits"]),
                            "directed_hits_relative": float(cgp["directed_hits"] / max(1, base["directed_hits"])),
                            "unique_pair_hits_delta": int(cgp["unique_pair_hits"] - base["unique_pair_hits"]),
                            "unique_pair_hits_relative": float(cgp["unique_pair_hits"] / max(1, base["unique_pair_hits"])),
                            "query_hit_rate_delta": float(cgp["query_hit_rate"] - base["query_hit_rate"]),
                            "yield_delta_per_1000": float(cgp["hit_yield_per_1000_ranked"] - base["hit_yield_per_1000_ranked"]),
                            "cgp_only_unique_pairs": int(len(cgp_pairs - base_pairs)),
                            "baseline_only_unique_pairs": int(len(base_pairs - cgp_pairs)),
                            "overlap_unique_pairs": int(len(cgp_pairs & base_pairs)),
                        }
                    )
    delta_df = pd.DataFrame(delta_rows)

    out_summary = args.input_dir / "prism_high_confidence_hit_summary.csv"
    out_delta = args.input_dir / "prism_high_confidence_hit_delta_vs_baselines.csv"
    out_hits = args.input_dir / "prism_high_confidence_hit_pairs.csv"
    summary_df.to_csv(out_summary, index=False)
    delta_df.to_csv(out_delta, index=False)
    hit_all.to_csv(out_hits, index=False)

    main_case = hit_all[
        (hit_all["method"] == "CGP-Align")
        & (hit_all["low_tanimoto"] == 0.20)
        & (hit_all["corr_threshold"] == 0.30)
        & (hit_all["topk"] == 50)
    ].copy()
    case_table = make_case_table(main_case, meta, morgan_sim, "CGP-Align", max_rows=40)
    out_cases = args.input_dir / "prism_high_confidence_cgp_cases_tan020_corr030_top50.csv"
    case_table.to_csv(out_cases, index=False)

    report = {
        "input_dir": str(args.input_dir),
        "num_compounds": int(len(meta)),
        "settings": [f"low_morgan_lt_{x:.2f}" for x in thresholds],
        "response_corr_thresholds": corr_thresholds,
        "topks": topks,
        "outputs": {
            "summary": str(out_summary),
            "delta_vs_baselines": str(out_delta),
            "hit_pairs": str(out_hits),
            "main_cases": str(out_cases),
        },
    }
    (args.input_dir / "prism_high_confidence_hit_run_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    main = summary_df[
        (summary_df["low_tanimoto"] == 0.20)
        & (summary_df["corr_threshold"] == 0.30)
        & (summary_df["topk"] == 50)
    ][
        [
            "method",
            "directed_hits",
            "unique_pair_hits",
            "queries_with_hit",
            "query_hit_rate",
            "hit_yield_per_1000_ranked",
            "mean_hit_rank",
            "median_hit_corr",
        ]
    ]
    print(main.to_string(index=False))
    main_delta = delta_df[
        (delta_df["low_tanimoto"] == 0.20)
        & (delta_df["corr_threshold"] == 0.30)
        & (delta_df["topk"] == 50)
    ]
    print(main_delta.to_string(index=False))


if __name__ == "__main__":
    main()
