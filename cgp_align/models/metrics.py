from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=eps)


def harmonic_mean(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(value) for value in values if value is not None and float(value) > 0]
    if len(vals) != len(values) or not vals:
        return None
    return float(len(vals) / sum(1.0 / value for value in vals))


def random_ranking_hit_expectation(n_positive: float, n_negative: int, k: int) -> float:
    n_total = float(n_positive) + float(n_negative)
    if n_total <= 0:
        return float("nan")
    return float(min(float(k), n_total) * float(n_positive) / n_total)


def sampled_topk_retrieval(
    query: np.ndarray,
    gallery: np.ndarray,
    query_code: np.ndarray,
    gallery_code: np.ndarray,
    n_negatives: int = 100,
    repeats: int = 5,
    topk: int = 10,
    random_state: int = 0,
) -> Dict[str, Any]:
    """Sampled retrieval with all same-code positives and sampled negatives."""
    if len(query) == 0 or len(gallery) == 0:
        return {"n_queries": int(len(query)), "hit_rate": float("nan")}

    rng = np.random.default_rng(int(random_state))
    all_gallery = np.arange(len(gallery), dtype=np.int64)
    query = np.asarray(query, dtype=np.float32)
    gallery = np.asarray(gallery, dtype=np.float32)
    query_code = np.asarray(query_code)
    gallery_code = np.asarray(gallery_code)

    hit_rates: List[float] = []
    positive_counts: List[int] = []
    for repeat_index in range(int(repeats)):
        hits = 0
        evaluated = 0
        for i in range(len(query)):
            positive_pool = all_gallery[gallery_code == query_code[i]]
            negative_pool = all_gallery[gallery_code != query_code[i]]
            if positive_pool.size == 0 or negative_pool.size == 0:
                continue
            n_sampled = min(int(n_negatives), int(negative_pool.size))
            sampled_negatives = rng.choice(negative_pool, size=n_sampled, replace=False)
            candidates = np.concatenate([positive_pool, sampled_negatives])
            scores = np.einsum("ij,j->i", gallery[candidates], query[i], optimize=False)
            ranked = candidates[np.argsort(-scores)]
            positive_set = set(positive_pool.tolist())
            hits += int(any(int(candidate) in positive_set for candidate in ranked[: min(int(topk), len(ranked))]))
            evaluated += 1
            if repeat_index == 0:
                positive_counts.append(int(positive_pool.size))
        if evaluated:
            hit_rates.append(float(hits / evaluated))

    return {
        "n_queries": int(len(query)),
        "n_gallery": int(len(gallery)),
        "n_evaluated": int(len(positive_counts)),
        "topk": int(topk),
        "n_negatives": int(n_negatives),
        "hit_rate_mean": float(np.mean(hit_rates)) if hit_rates else float("nan"),
        "hit_rate_sd": float(np.std(hit_rates)) if hit_rates else float("nan"),
        "positive_count_mean": float(np.mean(positive_counts)) if positive_counts else float("nan"),
        "random_ranking_expectation": random_ranking_hit_expectation(
            float(np.mean(positive_counts)) if positive_counts else 1.0, int(n_negatives), int(topk)
        ),
    }


def full_gallery_retrieval(
    query: np.ndarray,
    gallery: np.ndarray,
    query_code: np.ndarray,
    gallery_code: np.ndarray,
    ks: Sequence[int] = (1, 5, 10),
    chunk_size: int = 512,
) -> Dict[str, Any]:
    query = np.asarray(query, dtype=np.float32)
    gallery = np.asarray(gallery, dtype=np.float32)
    query_code = np.asarray(query_code)
    gallery_code = np.asarray(gallery_code)
    hit_counts = {int(k): 0 for k in ks}
    ranks: List[int] = []

    for start in range(0, len(query), max(1, int(chunk_size))):
        scores = np.einsum("ij,kj->ik", query[start : start + int(chunk_size)], gallery, optimize=False)
        for local_i in range(scores.shape[0]):
            i = start + local_i
            positives = gallery_code == query_code[i]
            if not bool(positives.any()):
                continue
            best_positive_score = float(scores[local_i, positives].max())
            rank = int(1 + np.sum(scores[local_i] > best_positive_score))
            ranks.append(rank)
            for k in ks:
                hit_counts[int(k)] += int(rank <= int(k))

    if not ranks:
        return {"n_queries": int(len(query)), "n_gallery": int(len(gallery)), "n_evaluated": 0}

    return {
        "n_queries": int(len(query)),
        "n_gallery": int(len(gallery)),
        "n_evaluated": int(len(ranks)),
        **{f"Recall@{int(k)}": float(hit_counts[int(k)] / len(ranks)) for k in ks},
        "MRR": float(np.mean(1.0 / np.asarray(ranks, dtype=np.float64))),
        "median_rank": float(np.median(ranks)),
    }


def lift_at_k(labels: np.ndarray, scores: np.ndarray, k: int = 50) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or labels.sum() == 0:
        return float("nan")
    selected = np.argsort(-scores)[: min(int(k), labels.size)]
    observed = float(labels[selected].mean())
    expected = float(labels.mean())
    return observed / expected if expected > 0 else float("nan")
