from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors

try:
    from rdkit.Avalon import pyAvalonTools
except Exception:  # pragma: no cover - optional RDKit build component
    pyAvalonTools = None

from cgp_common import CGPBundle
from cgp_gnn import GraphStore
from eval_tri_intrinsic_gnn_esm650 import encode_compounds, load_model, smiles_list


RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate whether phenotype-aligned CGP-Align compound embeddings retrieve "
            "PRISM functionally similar compounds beyond close structural analogues."
        )
    )
    p.add_argument("--prism_dir", type=Path, default=Path("data/prism_repurposing_19q4"))
    p.add_argument("--cgp_data_dir", type=Path, default=Path("data/cgp_cpg_full_motive_edges"))
    p.add_argument("--cgp_protein_embedding_dir", type=Path, default=Path("protein_embeddings/cgp_cpg_full_esm"))
    p.add_argument("--gene_mocop_data_dir", type=Path, default=Path("output/cgp_align/gene_mocop_reagent/data"))
    p.add_argument(
        "--gene_protein_embedding_dir",
        type=Path,
        default=Path("protein_embeddings/protein_encoder_ablation/esm650_l33_mean_cls"),
    )
    p.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("output/cgp_align/manuscript_freeze/best_seed17_external_validation_20260528_01/checkpoints"),
    )
    p.add_argument("--run_name", default="tri_cgp_calib_adapter_only_compound_favored_seed17")
    p.add_argument("--checkpoint_name", default="best_model.pt")
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/cgp_align/paper/manuscript_cgp_align_profile/prism_functional_analogue_v1"),
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--compound_batch_size", type=int, default=512)
    p.add_argument("--min_cell_lines", type=int, default=200)
    p.add_argument("--positive_topk", type=int, default=20)
    p.add_argument("--eval_topk", type=int, default=50)
    p.add_argument("--low_tanimoto", type=float, default=0.30)
    p.add_argument("--fingerprint_bits", type=int, default=2048)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--force_recompute_embeddings", action="store_true")
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
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return str(x)


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(denom, eps)).astype(np.float32)


def largest_fragment(mol: Chem.Mol) -> Chem.Mol | None:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return mol
    frags = [f for f in frags if f is not None and f.GetNumHeavyAtoms() > 0]
    if not frags:
        return None
    return max(frags, key=lambda m: (m.GetNumHeavyAtoms(), m.GetNumAtoms()))


def mol_from_prism_smiles(raw: Any) -> Tuple[Chem.Mol | None, str]:
    text = "" if pd.isna(raw) else str(raw).strip()
    if not text or text.upper() in {"NA", "NAN", "NONE", "NULL"}:
        return None, ""
    text = text.strip().strip('"').strip("'")
    text = re.sub(r",+$", "", text).strip()
    candidates = [text]
    if "," in text:
        candidates.extend([part.strip() for part in text.split(",") if part.strip()])
    best: Chem.Mol | None = None
    best_smi = ""
    for cand in candidates:
        mol = Chem.MolFromSmiles(cand)
        if mol is None:
            continue
        frag = largest_fragment(mol)
        if frag is None:
            continue
        smi = Chem.MolToSmiles(frag, canonical=True)
        if best is None or frag.GetNumHeavyAtoms() > best.GetNumHeavyAtoms():
            best = frag
            best_smi = smi
    return best, best_smi


def inchikey_from_mol(mol: Chem.Mol | None) -> str:
    if mol is None:
        return ""
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return ""


def load_prism_tables(prism_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    info_path = prism_dir / "primary-screen-replicate-collapsed-treatment-info.csv"
    response_path = prism_dir / "primary-screen-replicate-collapsed-logfold-change.csv"
    if not info_path.exists() or not response_path.exists():
        raise FileNotFoundError(f"Missing PRISM files under {prism_dir}")
    info = pd.read_csv(info_path, keep_default_na=False)
    response = pd.read_csv(response_path, index_col=0, na_values=["NA", "NaN", "nan", ""], keep_default_na=True)
    response = response.apply(pd.to_numeric, errors="coerce")
    return info, response


def standardize_prism_info(info: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for row in info.itertuples(index=False):
        raw_smi = getattr(row, "smiles")
        mol, canonical = mol_from_prism_smiles(raw_smi)
        rows.append(
            {
                "column_name": str(getattr(row, "column_name")),
                "broad_id": str(getattr(row, "broad_id")),
                "name": str(getattr(row, "name")),
                "dose": getattr(row, "dose"),
                "screen_id": str(getattr(row, "screen_id")),
                "moa": str(getattr(row, "moa")),
                "target": str(getattr(row, "target")),
                "smiles": str(raw_smi),
                "canonical_smiles_clean": canonical,
                "inchikey": inchikey_from_mol(mol),
            }
        )
    out = pd.DataFrame(rows)
    out["has_valid_inchikey"] = out["inchikey"].astype(str).str.len() > 0
    return out


def collapse_prism_response_by_inchikey(
    info: pd.DataFrame,
    response: pd.DataFrame,
    min_cell_lines: int,
) -> Tuple[pd.DataFrame, np.ndarray, List[str], Dict[str, Any]]:
    response_cols = set(response.columns.astype(str))
    matched = info.loc[info["column_name"].astype(str).isin(response_cols) & info["has_valid_inchikey"]].copy()
    if matched.empty:
        raise RuntimeError("No PRISM treatment columns matched the response matrix.")

    treatment_cols = matched["column_name"].astype(str).tolist()
    treatment_matrix = response.loc[:, treatment_cols].T.copy()
    treatment_matrix.insert(0, "inchikey", matched["inchikey"].to_numpy())
    numeric_cols = [c for c in treatment_matrix.columns if c != "inchikey"]
    by_key = treatment_matrix.groupby("inchikey", sort=False)[numeric_cols].median()
    valid_counts = by_key.notna().sum(axis=1)
    by_key = by_key.loc[valid_counts >= int(min_cell_lines)].copy()

    meta = (
        matched.sort_values(["inchikey", "column_name"])
        .groupby("inchikey", sort=False)
        .agg(
            prism_name=("name", lambda x: "; ".join(pd.Series(x).astype(str).drop_duplicates().head(3))),
            broad_id=("broad_id", lambda x: "; ".join(pd.Series(x).astype(str).drop_duplicates().head(3))),
            prism_columns=("column_name", "nunique"),
            prism_screens=("screen_id", lambda x: "; ".join(pd.Series(x).astype(str).drop_duplicates().head(5))),
            prism_targets=("target", lambda x: "; ".join([v for v in pd.Series(x).astype(str).drop_duplicates().head(5) if v and v != "NA"])),
            prism_moa=("moa", lambda x: "; ".join([v for v in pd.Series(x).astype(str).drop_duplicates().head(5) if v and v != "NA"])),
            prism_smiles=("canonical_smiles_clean", "first"),
        )
        .reset_index()
    )
    meta = meta.loc[meta["inchikey"].isin(by_key.index)].reset_index(drop=True)
    by_key = by_key.loc[meta["inchikey"].tolist()]

    summary = {
        "response_matrix_shape": [int(response.shape[0]), int(response.shape[1])],
        "treatment_info_rows": int(len(info)),
        "valid_inchikey_rows": int(info["has_valid_inchikey"].sum()),
        "matched_treatment_rows": int(len(matched)),
        "collapsed_unique_inchikey": int(by_key.shape[0]),
        "min_cell_lines": int(min_cell_lines),
        "median_nonmissing_cell_lines": float(valid_counts.loc[by_key.index].median()) if len(by_key) else 0.0,
    }
    return meta, by_key.to_numpy(dtype=np.float32), list(by_key.columns.astype(str)), summary


def match_prism_to_cgp(prism_meta: pd.DataFrame, cgp_bundle: CGPBundle) -> pd.DataFrame:
    compounds = cgp_bundle.compounds.reset_index(drop=True).copy()
    compounds["_cgp_row"] = np.arange(len(compounds), dtype=np.int64)
    if "inchikey" not in compounds.columns:
        compounds["inchikey"] = compounds["compound_id"].astype(str)
    cgp_cols = ["_cgp_row", "compound_id", "canonical_smiles", "inchikey", "has_structure", "has_compound_profile"]
    cgp_cols = [c for c in cgp_cols if c in compounds.columns]
    cgp_sub = compounds[cgp_cols].drop_duplicates("inchikey")
    merged = prism_meta.merge(cgp_sub, on="inchikey", how="inner")
    merged = merged.loc[merged["canonical_smiles"].astype(str).str.len() > 0].copy()
    merged = merged.sort_values(["inchikey", "_cgp_row"]).drop_duplicates("inchikey").reset_index(drop=True)
    return merged


def load_gene_context(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    # Compound-only inference needs gene-side array dimensions to instantiate the
    # original joint model, but does not need gene metadata or splits.
    gene_features = np.load(args.gene_mocop_data_dir / "gene_mocop_replicate_features.npy").astype(np.float32)
    gene_protein_embeddings = np.load(args.gene_protein_embedding_dir / "protein_sequence_embeddings.npy").astype(np.float32)
    return gene_features, gene_protein_embeddings


def load_or_encode_cgp_embeddings(
    args: argparse.Namespace,
    cgp_bundle: CGPBundle,
    cgp_rows: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    cache = args.output_dir / f"cgp_align_{args.run_name}_prism_overlap_embeddings.npz"
    if cache.exists() and not args.force_recompute_embeddings:
        payload = np.load(cache, allow_pickle=True)
        cached_rows = payload["cgp_rows"].astype(np.int64)
        if np.array_equal(cached_rows, cgp_rows.astype(np.int64)):
            return payload["z_cgp"].astype(np.float32)

    gene_features, gene_protein_embeddings = load_gene_context(args)
    model_args = argparse.Namespace(
        run_name=args.run_name,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
    )
    model, _, _ = load_model(model_args, cgp_bundle, gene_features, gene_protein_embeddings, device)
    graph_store = GraphStore(smiles_list(cgp_bundle))
    z = encode_compounds(model, cgp_bundle, graph_store, cgp_rows.astype(np.int64), device, args.compound_batch_size)
    z = l2_normalize(z)
    np.savez_compressed(cache, cgp_rows=cgp_rows.astype(np.int64), z_cgp=z)
    return z


def response_similarity(response: np.ndarray) -> np.ndarray:
    x = np.asarray(response, dtype=np.float32).copy()
    row_median = np.nanmedian(x, axis=1)
    row_median = np.nan_to_num(row_median, nan=0.0)
    missing = ~np.isfinite(x)
    if missing.any():
        x[missing] = np.take(row_median, np.where(missing)[0])
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    x = x / np.maximum(std, 1e-6)
    sim = (x @ x.T) / max(1, x.shape[1])
    sim = np.clip(sim, -1.0, 1.0).astype(np.float32)
    np.fill_diagonal(sim, -np.inf)
    return sim


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
            if pyAvalonTools is None:
                fps.append(None)
            else:
                fps.append(pyAvalonTools.GetAvalonFP(mol, nBits=n_bits))
        else:
            raise ValueError(f"Unknown fingerprint kind: {kind}")
    return fps


def tanimoto_matrix(fps: List[Any]) -> np.ndarray:
    n = len(fps)
    sim = np.zeros((n, n), dtype=np.float32)
    valid = [fp is not None for fp in fps]
    for i, fp in enumerate(fps):
        if fp is None:
            continue
        vals = DataStructs.BulkTanimotoSimilarity(fp, [f if f is not None else fp for f in fps])
        row = np.asarray(vals, dtype=np.float32)
        row[~np.asarray(valid, dtype=bool)] = -np.inf
        sim[i] = row
    np.fill_diagonal(sim, -np.inf)
    return sim


def cosine_matrix(x: np.ndarray) -> np.ndarray:
    z = l2_normalize(x)
    sim = z @ z.T
    np.fill_diagonal(sim, -np.inf)
    return sim.astype(np.float32)


def average_precision(labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    total = int(labels.sum())
    if total == 0:
        return float("nan")
    hits = np.cumsum(labels)
    ranks = np.arange(1, len(labels) + 1)
    return float((hits[labels] / ranks[labels]).mean())


def evaluate_method(
    method_name: str,
    score: np.ndarray,
    response_sim: np.ndarray,
    candidate_mask: np.ndarray,
    positive_topk: int,
    eval_topk: int,
    rng: np.random.Generator | None = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    n = score.shape[0]
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        cand = np.flatnonzero(candidate_mask[i])
        cand = cand[cand != i]
        if cand.size <= positive_topk:
            continue
        resp_scores = response_sim[i, cand]
        finite_resp = np.isfinite(resp_scores)
        cand = cand[finite_resp]
        resp_scores = resp_scores[finite_resp]
        if cand.size <= positive_topk:
            continue
        pos_local = np.argsort(-resp_scores, kind="mergesort")[:positive_topk]
        positives = set(cand[pos_local].tolist())
        method_scores = score[i, cand].copy()
        finite_method = np.isfinite(method_scores)
        if not finite_method.any():
            continue
        cand = cand[finite_method]
        method_scores = method_scores[finite_method]
        if rng is not None and method_name.lower() == "random":
            method_scores = rng.random(cand.size)
        order = np.argsort(-method_scores, kind="mergesort")
        ranked = cand[order]
        labels = np.asarray([int(j) in positives for j in ranked], dtype=bool)
        if labels.sum() == 0:
            continue
        k10 = min(10, len(ranked))
        k_eval = min(eval_topk, len(ranked))
        first_hits = np.flatnonzero(labels)
        first_rank = int(first_hits[0] + 1) if first_hits.size else math.inf
        top10 = ranked[:k10]
        top_eval = ranked[:k_eval]
        base_rate = float(labels.sum() / len(labels))
        top_eval_precision = float(labels[:k_eval].mean()) if k_eval else float("nan")
        rows.append(
            {
                "query_idx": i,
                "num_candidates": int(len(ranked)),
                "num_positives": int(labels.sum()),
                "hit10": float(labels[:k10].any()),
                "recall50": float(labels[:k_eval].sum() / max(1, labels.sum())),
                "precision50": top_eval_precision,
                "lift50": float(top_eval_precision / base_rate) if base_rate > 0 else float("nan"),
                "mrr": float(1.0 / first_rank) if math.isfinite(first_rank) else 0.0,
                "average_precision": average_precision(labels),
                "mean_response_corr_top10": float(np.nanmean(response_sim[i, top10])) if len(top10) else float("nan"),
                "mean_response_corr_top50": float(np.nanmean(response_sim[i, top_eval])) if len(top_eval) else float("nan"),
                "mean_positive_response_corr": float(np.nanmean(response_sim[i, list(positives)])),
            }
        )
    q = pd.DataFrame(rows)
    if q.empty:
        summary = {
            "method": method_name,
            "num_queries": 0,
            "hit10": float("nan"),
            "recall50": float("nan"),
            "precision50": float("nan"),
            "lift50": float("nan"),
            "mrr": float("nan"),
            "average_precision": float("nan"),
            "mean_response_corr_top10": float("nan"),
            "mean_response_corr_top50": float("nan"),
            "mean_positive_response_corr": float("nan"),
        }
    else:
        summary = {"method": method_name, "num_queries": int(len(q))}
        for col in [
            "num_candidates",
            "num_positives",
            "hit10",
            "recall50",
            "precision50",
            "lift50",
            "mrr",
            "average_precision",
            "mean_response_corr_top10",
            "mean_response_corr_top50",
            "mean_positive_response_corr",
        ]:
            summary[col] = float(q[col].mean())
    q.insert(0, "method", method_name)
    return summary, q


def make_case_table(
    meta: pd.DataFrame,
    method_score: np.ndarray,
    response_sim: np.ndarray,
    morgan_sim: np.ndarray,
    candidate_mask: np.ndarray,
    max_queries: int = 20,
    topn: int = 5,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    n = method_score.shape[0]
    query_order = np.argsort(-np.nanmax(np.where(candidate_mask, response_sim, -np.inf), axis=1))
    for i in query_order[:max_queries]:
        cand = np.flatnonzero(candidate_mask[i])
        cand = cand[cand != i]
        if cand.size == 0:
            continue
        order = cand[np.argsort(-method_score[i, cand], kind="mergesort")[:topn]]
        for rank, j in enumerate(order, start=1):
            rows.append(
                {
                    "query_rank_group": int(len(rows) // topn + 1),
                    "query_name": meta.loc[i, "prism_name"],
                    "query_inchikey": meta.loc[i, "inchikey"],
                    "query_moa": meta.loc[i, "prism_moa"],
                    "retrieved_rank": int(rank),
                    "retrieved_name": meta.loc[j, "prism_name"],
                    "retrieved_inchikey": meta.loc[j, "inchikey"],
                    "retrieved_moa": meta.loc[j, "prism_moa"],
                    "cgp_similarity": float(method_score[i, j]),
                    "prism_response_corr": float(response_sim[i, j]),
                    "morgan_tanimoto": float(morgan_sim[i, j]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    rng = np.random.default_rng(args.seed)

    info, response = load_prism_tables(args.prism_dir)
    prism_info = standardize_prism_info(info)
    prism_info.to_csv(args.output_dir / "prism_treatment_info_cleaned.csv", index=False)
    prism_meta_all, prism_response_all, cell_lines, prism_summary = collapse_prism_response_by_inchikey(
        prism_info, response, args.min_cell_lines
    )

    cgp_bundle = CGPBundle(args.cgp_data_dir, args.cgp_protein_embedding_dir)
    matched = match_prism_to_cgp(prism_meta_all, cgp_bundle)
    if matched.empty:
        raise RuntimeError("No PRISM compounds overlapped CGP compound library after InChIKey matching.")
    prism_index = {k: i for i, k in enumerate(prism_meta_all["inchikey"].astype(str))}
    response_rows = np.asarray([prism_index[k] for k in matched["inchikey"].astype(str)], dtype=np.int64)
    prism_response = prism_response_all[response_rows]

    cgp_rows = matched["_cgp_row"].to_numpy(dtype=np.int64)
    z_cgp = load_or_encode_cgp_embeddings(args, cgp_bundle, cgp_rows, device)
    cgp_sim = cosine_matrix(z_cgp)

    smiles = matched["canonical_smiles"].astype(str).tolist()
    morgan_fps = fingerprint_list(smiles, "morgan", args.fingerprint_bits)
    morgan_sim = tanimoto_matrix(morgan_fps)
    method_scores: Dict[str, np.ndarray] = {
        "CGP-Align": cgp_sim,
        "Morgan": morgan_sim,
        "AtomPair": tanimoto_matrix(fingerprint_list(smiles, "atompair", args.fingerprint_bits)),
        "Random": np.zeros_like(cgp_sim, dtype=np.float32),
    }
    avalon_fps = fingerprint_list(smiles, "avalon", args.fingerprint_bits)
    if any(fp is not None for fp in avalon_fps):
        method_scores["Avalon"] = tanimoto_matrix(avalon_fps)
    structure_features = cgp_bundle.structure_features[cgp_rows]
    method_scores["RDKit2D"] = cosine_matrix(structure_features)

    resp_sim = response_similarity(prism_response)
    n = len(matched)
    all_mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(all_mask, False)
    low_structure_mask = (morgan_sim < float(args.low_tanimoto)) & np.isfinite(morgan_sim)
    np.fill_diagonal(low_structure_mask, False)

    all_summaries: List[Dict[str, Any]] = []
    all_query_rows: List[pd.DataFrame] = []
    for setting, mask in [("all_candidates", all_mask), (f"low_morgan_lt_{args.low_tanimoto:.2f}", low_structure_mask)]:
        for method, score in method_scores.items():
            method_rng = np.random.default_rng(args.seed + 1009) if method == "Random" else None
            summary, query_df = evaluate_method(
                method,
                score,
                resp_sim,
                mask,
                positive_topk=args.positive_topk,
                eval_topk=args.eval_topk,
                rng=method_rng,
            )
            summary["setting"] = setting
            all_summaries.append(summary)
            query_df.insert(0, "setting", setting)
            all_query_rows.append(query_df)

    summary_df = pd.DataFrame(all_summaries)
    summary_df = summary_df.sort_values(["setting", "average_precision"], ascending=[True, False]).reset_index(drop=True)
    query_metrics = pd.concat(all_query_rows, ignore_index=True) if all_query_rows else pd.DataFrame()

    matched_out = matched.copy()
    matched_out["num_nonmissing_prism_cell_lines"] = np.isfinite(prism_response).sum(axis=1)
    matched_out.to_csv(args.output_dir / "prism_cgp_overlap_compounds.csv", index=False)
    summary_df.to_csv(args.output_dir / "prism_functional_retrieval_summary.csv", index=False)
    query_metrics.to_csv(args.output_dir / "prism_functional_retrieval_query_metrics.csv", index=False)
    np.savez_compressed(
        args.output_dir / "prism_functional_retrieval_matrices.npz",
        inchikey=matched["inchikey"].astype(str).to_numpy(),
        response_similarity=resp_sim.astype(np.float32),
        cgp_similarity=cgp_sim.astype(np.float32),
        morgan_tanimoto=morgan_sim.astype(np.float32),
        prism_response=prism_response.astype(np.float32),
        cgp_rows=cgp_rows.astype(np.int64),
        cell_lines=np.asarray(cell_lines, dtype=object),
    )

    case_df = make_case_table(
        matched.reset_index(drop=True),
        cgp_sim,
        resp_sim,
        morgan_sim,
        low_structure_mask,
        max_queries=20,
        topn=5,
    )
    case_df.to_csv(args.output_dir / "prism_cgp_low_structure_retrieval_cases.csv", index=False)

    run_summary = {
        "device": str(device),
        "checkpoint_dir": args.checkpoint_dir,
        "run_name": args.run_name,
        "cgp_data_dir": args.cgp_data_dir,
        "prism": prism_summary,
        "cgp_library_compounds": int(len(cgp_bundle.compounds)),
        "matched_prism_cgp_compounds": int(n),
        "cell_lines_used": int(prism_response.shape[1]),
        "positive_topk": int(args.positive_topk),
        "eval_topk": int(args.eval_topk),
        "low_morgan_tanimoto_threshold": float(args.low_tanimoto),
        "outputs": {
            "overlap": str(args.output_dir / "prism_cgp_overlap_compounds.csv"),
            "summary": str(args.output_dir / "prism_functional_retrieval_summary.csv"),
            "query_metrics": str(args.output_dir / "prism_functional_retrieval_query_metrics.csv"),
            "cases": str(args.output_dir / "prism_cgp_low_structure_retrieval_cases.csv"),
            "matrices": str(args.output_dir / "prism_functional_retrieval_matrices.npz"),
        },
    }
    (args.output_dir / "prism_functional_retrieval_run_summary.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )

    print(json.dumps(run_summary, indent=2, ensure_ascii=False, default=json_default))
    keep_cols = ["setting", "method", "num_queries", "hit10", "recall50", "lift50", "mrr", "average_precision", "mean_response_corr_top10"]
    print(summary_df[keep_cols].to_string(index=False))


if __name__ == "__main__":
    main()
