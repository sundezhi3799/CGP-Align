from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Scope:
    name: str
    query_mask: np.ndarray
    gallery_mask: np.ndarray
    same_pool: bool
    baseline: str | None


@dataclass(frozen=True)
class MethodSpec:
    name: str
    query_array: str
    gallery_array: str


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def iter_blocks(n: int, block_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, n, block_size):
        yield start, min(start + block_size, n)


def topn_by_similarity(
    query: np.ndarray,
    gallery: np.ndarray,
    n_top: int,
    block_size: int,
    exclude_gallery_local: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return top-n gallery indices and similarities for each query row."""
    n_query = query.shape[0]
    n_gallery = gallery.shape[0]
    if n_top >= n_gallery:
        raise ValueError("n_top must be smaller than gallery size")
    if exclude_gallery_local is not None and exclude_gallery_local.shape[0] != n_query:
        raise ValueError("exclude_gallery_local must have one entry per query")

    top_idx = np.empty((n_query, n_top), dtype=np.int32)
    top_sim = np.empty((n_query, n_top), dtype=np.float32)

    for start, end in iter_blocks(n_query, block_size):
        scores = query[start:end] @ gallery.T
        if exclude_gallery_local is not None:
            local_exclude = exclude_gallery_local[start:end]
            valid = local_exclude >= 0
            scores[np.flatnonzero(valid), local_exclude[valid]] = -np.inf
        part = np.argpartition(-scores, kth=n_top - 1, axis=1)[:, :n_top]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        sorted_idx = np.take_along_axis(part, order, axis=1)
        sorted_scores = np.take_along_axis(part_scores, order, axis=1)
        top_idx[start:end] = sorted_idx.astype(np.int32)
        top_sim[start:end] = sorted_scores.astype(np.float32)

    return top_idx, top_sim


def evaluate_hidden_neighbours(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
    positive_idx: np.ndarray,
    eval_ks: list[int],
    block_size: int,
    exclude_gallery_local: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate whether embedding-nearest neighbours recover profile-defined positives."""
    n_query = query_emb.shape[0]
    n_gallery = gallery_emb.shape[0]
    n_pos = positive_idx.shape[1]
    max_k = max(eval_ks)
    if max_k >= n_gallery:
        raise ValueError("max eval K must be smaller than gallery size")
    if exclude_gallery_local is not None and exclude_gallery_local.shape[0] != n_query:
        raise ValueError("exclude_gallery_local must have one entry per query")

    rows: list[dict[str, float]] = []
    query_rows: list[dict[str, float]] = []

    overlap_sum = {k: 0.0 for k in eval_ks}
    hit_sum = {k: 0.0 for k in eval_ks}
    best_rank = np.empty(n_query, dtype=np.int32)

    for start, end in iter_blocks(n_query, block_size):
        scores = query_emb[start:end] @ gallery_emb.T
        if exclude_gallery_local is not None:
            local_exclude = exclude_gallery_local[start:end]
            valid = local_exclude >= 0
            scores[np.flatnonzero(valid), local_exclude[valid]] = -np.inf

        pos = positive_idx[start:end]
        pos_scores = np.take_along_axis(scores, pos, axis=1)
        best_pos_scores = pos_scores.max(axis=1)
        ranks = 1 + (scores > best_pos_scores[:, None]).sum(axis=1)
        best_rank[start:end] = ranks.astype(np.int32)

        top_idx_unsorted = np.argpartition(-scores, kth=max_k - 1, axis=1)[:, :max_k]
        top_scores = np.take_along_axis(scores, top_idx_unsorted, axis=1)
        top_order = np.argsort(-top_scores, axis=1)
        top_idx = np.take_along_axis(top_idx_unsorted, top_order, axis=1)
        pos_sets = [set(map(int, row)) for row in pos]
        for local_i, candidates in enumerate(top_idx):
            qid = start + local_i
            qrow = {"query_index": qid, "best_positive_rank": int(best_rank[qid])}
            for k in eval_ks:
                overlap = len(pos_sets[local_i].intersection(map(int, candidates[:k])))
                overlap_sum[k] += overlap
                hit_sum[k] += float(overlap > 0)
                qrow[f"overlap_at_{k}"] = int(overlap)
            query_rows.append(qrow)

    effective_gallery = n_gallery - 1 if exclude_gallery_local is not None and np.any(exclude_gallery_local >= 0) else n_gallery
    for k in eval_ks:
        precision = overlap_sum[k] / (n_query * k)
        recall = overlap_sum[k] / (n_query * n_pos)
        hit = hit_sum[k] / n_query
        expected_precision = n_pos / effective_gallery
        expected_recall = min(k, effective_gallery) / effective_gallery
        expected_hit = 1.0 - float(np.prod([(effective_gallery - n_pos - i) / (effective_gallery - i) for i in range(min(k, effective_gallery - n_pos))])) if effective_gallery > n_pos else 1.0
        rows.append(
            {
                "k": k,
                "n_query": n_query,
                "n_gallery": effective_gallery,
                "n_hidden_positive_per_query": n_pos,
                "hit_at_k": hit,
                "recall_at_k": recall,
                "precision_at_k": precision,
                "random_precision_at_k": expected_precision,
                "random_recall_at_k": expected_recall,
                "random_hit_at_k": expected_hit,
                "precision_enrichment": precision / expected_precision if expected_precision > 0 else np.nan,
                "recall_enrichment": recall / expected_recall if expected_recall > 0 else np.nan,
                "mean_best_positive_rank": float(best_rank.mean()),
                "median_best_positive_rank": float(np.median(best_rank)),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(query_rows)


def build_feature_tables(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    emb = np.load(args.embedding_npz, allow_pickle=True)
    prof = np.load(args.profile_input_npz, allow_pickle=True)

    compound_ids = emb["compound_ids"].astype(str)
    gene_ids = emb["gene_entity_ids"].astype(str)
    ids = np.concatenate([compound_ids, gene_ids])

    if prof["x"].shape[0] != ids.shape[0]:
        raise ValueError("profile input row count does not match embedding row count")
    if not np.array_equal(prof["ids"].astype(str), ids):
        mismatch = np.where(prof["ids"].astype(str) != ids)[0][:5]
        raise ValueError(f"profile input IDs are not aligned with embeddings; first mismatches: {mismatch}")

    modality = np.concatenate(
        [
            np.repeat("compound", len(compound_ids)),
            emb["gene_modalities"].astype(str),
        ]
    )
    gene_symbol = np.concatenate(
        [
            np.repeat("", len(compound_ids)),
            emb["gene_symbols"].astype(str),
        ]
    )

    meta = pd.DataFrame(
        {
            "row_index": np.arange(ids.shape[0], dtype=np.int32),
            "entity_id": ids,
            "modality": modality,
            "gene_symbol": gene_symbol,
        }
    )

    z_entity = np.vstack([emb["z_compound"], emb["z_protein"]]).astype(np.float32)
    z_profile = np.vstack([emb["z_compound_profile"], emb["z_gene_profile"]]).astype(np.float32)
    profile_input = np.asarray(prof["x"], dtype=np.float32)

    arrays: dict[str, np.ndarray] = {
        "CGP-Align entity": l2_normalize(z_entity),
        "CGP profile encoder": l2_normalize(z_profile),
        "Raw profile": l2_normalize(profile_input),
    }

    if args.structure_features and Path(args.structure_features).exists():
        structure_all = np.load(args.structure_features, mmap_mode="r")
        compound_structure = np.asarray(structure_all[emb["compound_indices"].astype(np.int64)], dtype=np.float32)
        structure_rows = np.zeros((ids.shape[0], compound_structure.shape[1]), dtype=np.float32)
        structure_rows[: len(compound_ids)] = compound_structure
        arrays["Structure feature baseline"] = l2_normalize(structure_rows)

    if args.protein_features and Path(args.protein_features).exists():
        protein_all = np.load(args.protein_features, mmap_mode="r")
        protein_sub = np.asarray(protein_all[emb["gene_indices"].astype(np.int64)], dtype=np.float32)
        protein_rows = np.zeros((ids.shape[0], protein_sub.shape[1]), dtype=np.float32)
        protein_rows[len(compound_ids) :] = protein_sub
        arrays["ESM-2 baseline"] = l2_normalize(protein_rows)

    return meta, arrays


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover profile-defined hidden phenotype neighbours from frozen CGP-Align embeddings."
    )
    parser.add_argument(
        "--embedding-npz",
        type=Path,
        default=Path("output/cgp_align/manuscript_figure_source_data/figure3_latents/v3_test_embeddings.npz"),
    )
    parser.add_argument(
        "--profile-input-npz",
        type=Path,
        default=Path(
            "output/cgp_align/tri_intrinsic/eval_adapter_only_batch/adapter_only_compound_favored_umap/gene20_input_profile_vectors.npz"
        ),
    )
    parser.add_argument(
        "--structure-features",
        type=Path,
        default=Path("data/cgp_cpg_full_motive_edges/compound_structure_features.npy"),
    )
    parser.add_argument(
        "--protein-features",
        type=Path,
        default=Path("protein_embeddings/protein_encoder_ablation/esm650_l33_mean_cls/protein_sequence_embeddings.npy"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/cgp_align/paper/manuscript_cgp_align_profile/hidden_phenotype_neighbour_recovery_20260908"),
    )
    parser.add_argument("--positive-topn", type=int, default=10)
    parser.add_argument("--eval-ks", type=str, default="10,50,100,200")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-query-per-scope", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    eval_ks = parse_ints(args.eval_ks)

    meta, arrays = build_feature_tables(args)
    meta.to_csv(args.output_dir / "entity_metadata.csv", index=False)

    masks = {m: (meta["modality"].to_numpy() == m) for m in ["compound", "orf", "crispr"]}
    scopes = [
        Scope("compound_to_compound", masks["compound"], masks["compound"], True, "Structure feature baseline"),
        Scope("orf_to_orf", masks["orf"], masks["orf"], True, "ESM-2 baseline"),
        Scope("crispr_to_crispr", masks["crispr"], masks["crispr"], True, "ESM-2 baseline"),
        Scope("compound_to_orf", masks["compound"], masks["orf"], False, None),
        Scope("compound_to_crispr", masks["compound"], masks["crispr"], False, None),
        Scope("orf_to_compound", masks["orf"], masks["compound"], False, None),
        Scope("crispr_to_compound", masks["crispr"], masks["compound"], False, None),
        Scope("orf_to_crispr", masks["orf"], masks["crispr"], False, "ESM-2 baseline"),
        Scope("crispr_to_orf", masks["crispr"], masks["orf"], False, "ESM-2 baseline"),
    ]

    summaries: list[pd.DataFrame] = []
    query_tables: list[pd.DataFrame] = []
    hidden_pair_rows: list[pd.DataFrame] = []

    for scope in scopes:
        q_global = np.flatnonzero(scope.query_mask)
        g_global = np.flatnonzero(scope.gallery_mask)
        if args.max_query_per_scope and q_global.shape[0] > args.max_query_per_scope:
            q_global = np.sort(rng.choice(q_global, size=args.max_query_per_scope, replace=False))
        if g_global.shape[0] <= max(args.positive_topn, max(eval_ks)):
            continue

        q_profile = arrays["Raw profile"][q_global]
        g_profile = arrays["Raw profile"][g_global]
        gallery_lookup = {int(row): i for i, row in enumerate(g_global)}
        exclude_gallery_local = np.asarray([gallery_lookup.get(int(row), -1) for row in q_global], dtype=np.int32)
        if not scope.same_pool:
            exclude_gallery_local[:] = -1
        hidden_idx_local, hidden_sim = topn_by_similarity(
            q_profile,
            g_profile,
            args.positive_topn,
            args.block_size,
            exclude_gallery_local=exclude_gallery_local,
        )

        hidden_pairs = []
        for qi, qrow in enumerate(q_global):
            for rank_pos, (local_g, sim) in enumerate(zip(hidden_idx_local[qi], hidden_sim[qi]), start=1):
                grow = g_global[int(local_g)]
                hidden_pairs.append(
                    {
                        "scope": scope.name,
                        "query_row": int(qrow),
                        "query_entity_id": str(meta.loc[qrow, "entity_id"]),
                        "query_modality": str(meta.loc[qrow, "modality"]),
                        "gallery_row": int(grow),
                        "gallery_entity_id": str(meta.loc[grow, "entity_id"]),
                        "gallery_modality": str(meta.loc[grow, "modality"]),
                        "profile_neighbour_rank": rank_pos,
                        "profile_cosine_similarity": float(sim),
                    }
                )
        hidden_pair_rows.append(pd.DataFrame(hidden_pairs))

        methods = [
            MethodSpec("CGP-Align entity", "CGP-Align entity", "CGP-Align entity"),
            MethodSpec("CGP entity -> profile anchor", "CGP-Align entity", "CGP profile encoder"),
            MethodSpec("CGP profile encoder", "CGP profile encoder", "CGP profile encoder"),
        ]
        if scope.baseline and scope.baseline in arrays:
            methods.append(MethodSpec(scope.baseline, scope.baseline, scope.baseline))

        for method in methods:
            if method.name in {"Structure feature baseline", "ESM-2 baseline"} and scope.baseline != method.name:
                continue
            q_emb = arrays[method.query_array][q_global]
            g_emb = arrays[method.gallery_array][g_global]
            summary, qtab = evaluate_hidden_neighbours(
                q_emb,
                g_emb,
                hidden_idx_local,
                eval_ks,
                args.block_size,
                exclude_gallery_local=exclude_gallery_local,
            )
            summary.insert(0, "scope", scope.name)
            summary.insert(1, "method", method.name)
            summary.insert(2, "hidden_defined_by", "Raw profile cosine top-N")
            summary.insert(3, "mean_hidden_profile_similarity", float(hidden_sim.mean()))
            summary.insert(4, "median_hidden_profile_similarity", float(np.median(hidden_sim)))
            summaries.append(summary)
            qtab.insert(0, "scope", scope.name)
            qtab.insert(1, "method", method.name)
            qtab.insert(2, "query_row_global", q_global[qtab["query_index"].to_numpy(dtype=int)])
            query_tables.append(qtab)

    if not summaries:
        raise RuntimeError("No scope was evaluated")

    summary_df = pd.concat(summaries, ignore_index=True)
    query_df = pd.concat(query_tables, ignore_index=True)
    hidden_df = pd.concat(hidden_pair_rows, ignore_index=True)

    summary_df.to_csv(args.output_dir / "hidden_phenotype_neighbour_recovery_summary.csv", index=False)
    query_df.to_csv(args.output_dir / "hidden_phenotype_neighbour_recovery_query_level.csv", index=False)
    hidden_df.to_csv(args.output_dir / "hidden_profile_neighbour_pairs.csv", index=False)

    compact = summary_df[summary_df["k"].isin([50, 100])].copy()
    compact = compact[
        [
            "scope",
            "method",
            "k",
            "n_query",
            "n_gallery",
            "hit_at_k",
            "recall_at_k",
            "precision_enrichment",
            "mean_best_positive_rank",
        ]
    ]
    compact.to_csv(args.output_dir / "hidden_phenotype_neighbour_recovery_compact.csv", index=False)

    meta_json = {
        "embedding_npz": str(args.embedding_npz),
        "profile_input_npz": str(args.profile_input_npz),
        "structure_features": str(args.structure_features),
        "protein_features": str(args.protein_features),
        "positive_topn": args.positive_topn,
        "eval_ks": eval_ks,
        "max_query_per_scope": args.max_query_per_scope,
        "seed": args.seed,
        "interpretation_boundary": (
            "Hidden phenotype neighbours are defined from held-out raw profile cosine similarity. "
            "The analysis evaluates whether frozen perturbation-side embeddings recover these profile-defined "
            "neighbourhoods; it is not an independent biological-label benchmark."
        ),
    }
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps(meta_json, indent=2), encoding="utf-8")

    print(compact.to_string(index=False))
    print(f"\nWrote outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
