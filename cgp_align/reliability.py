from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TransformFn = Callable[[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]], np.ndarray]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def _pairwise_cosine_mean(x: np.ndarray) -> Optional[float]:
    if x.shape[0] < 2:
        return None
    z = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    sim = z @ z.T
    mask = ~np.eye(sim.shape[0], dtype=bool)
    if not np.any(mask):
        return None
    return float(np.mean(sim[mask]))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))).astype(np.float32)


def _zscore(x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.float32)
    mean = float(np.mean(x[finite]))
    std = float(np.std(x[finite]))
    if std < 1e-6:
        std = 1.0
    return ((x - mean) / std).astype(np.float32)


@dataclass
class ReliabilityArtifacts:
    entity_weights: np.ndarray
    theta_by_entity_key: np.ndarray
    theta_valid: np.ndarray
    table: pd.DataFrame
    theta_table: pd.DataFrame
    summary: Dict[str, Any]

    @property
    def theta_by_gene(self) -> np.ndarray:
        """Backward-compatible alias; rows are entity_key codes in Phase-1 mixed mode."""
        return self.theta_by_entity_key


DEFAULT_MODALITY_NAMES = {0: "orf", 1: "crispr"}


def gene_id_values(entities: pd.DataFrame) -> np.ndarray:
    column = "gene_id" if "gene_id" in entities.columns else "gene_symbol"
    return entities[column].fillna("").astype(str).to_numpy()


def entity_key_strings_from_entities(entities: pd.DataFrame) -> np.ndarray:
    gene_ids = gene_id_values(entities)
    modalities = entities["perturbation_modality"].astype(str).str.lower().to_numpy()
    return np.asarray([f"{g}::{m}" for g, m in zip(gene_ids.tolist(), modalities.tolist())], dtype=object)


def entity_key_codes_from_entities(entities: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    keys = entity_key_strings_from_entities(entities)
    codes, uniques = pd.factorize(keys, sort=True)
    return codes.astype(np.int64), np.asarray(uniques, dtype=object)


def parse_allowed_modalities(value: Any) -> List[str]:
    raw = str(value or "").strip().lower().replace(" ", "")
    if raw in {"", "all", "mixed", "crispr_orf_mixed", "crispr+orf", "orf+crispr", "crispr,orf", "orf,crispr"}:
        return ["orf", "crispr"]
    parts = [p for p in raw.replace("+", ",").split(",") if p]
    aliases = {"crispr_only": "crispr", "orf_only": "orf"}
    out = [aliases.get(p, p) for p in parts]
    bad = sorted(set(out) - {"orf", "crispr"})
    if bad:
        raise ValueError(f"Unsupported strict Phase-1 modalities: {bad}")
    return sorted(set(out), key=["orf", "crispr"].index)


def estimate_train_only_gene_reliability(
    entities: pd.DataFrame,
    replicates: pd.DataFrame,
    features: np.ndarray,
    entity_to_reps: Sequence[np.ndarray],
    entity_mods: np.ndarray,
    gene_codes_all: np.ndarray,
    train_entities: np.ndarray,
    norm: Dict[str, Any],
    transform_feature_rows: TransformFn,
    allowed_modality_ids: Optional[Sequence[int]] = None,
    modality_id_to_name: Optional[Dict[int, str]] = None,
    default_guide_consistency: float = 0.5,
) -> ReliabilityArtifacts:
    """Estimate train entity-key reliability without touching val/test profiles.

    The function intentionally receives only train entity rows. It does not inspect
    split dictionaries or non-train profiles; callers are expected to run leakage
    assertions around it.
    """

    modality_id_to_name = {**DEFAULT_MODALITY_NAMES, **(modality_id_to_name or {})}
    if allowed_modality_ids is None:
        allowed_modality_ids = sorted(set(int(x) for x in entity_mods.tolist()))
    allowed = {int(x) for x in allowed_modality_ids}
    allowed_names = [modality_id_to_name.get(int(x), str(int(x))) for x in sorted(allowed)]
    train_entities = np.asarray(train_entities, dtype=np.int64)
    train_entities = train_entities[np.asarray([int(entity_mods[i]) in allowed for i in train_entities], dtype=bool)]
    num_entities = len(entities)
    entity_key_codes_all, entity_key_uniques = entity_key_codes_from_entities(entities)
    entity_key_strings = entity_key_strings_from_entities(entities)
    gene_ids = gene_id_values(entities)
    num_entity_keys = int(np.max(entity_key_codes_all)) + 1 if len(entity_key_codes_all) else 0
    entity_weights = np.ones(num_entities, dtype=np.float32)
    theta_by_entity_key = np.zeros((num_entity_keys, int(features.shape[1])), dtype=np.float32)
    theta_valid = np.zeros(num_entity_keys, dtype=bool)
    if train_entities.size == 0:
        empty = pd.DataFrame(
            columns=[
                "gene_code",
                "gene_id",
                "modality",
                "entity_key",
                "entity_key_code",
                "n_replicates",
                "profile_activity",
                "activity_z",
                "replicate_consistency",
                "guide_consistency",
                "reagent_consistency",
                "missing_guide_consistency",
                "missing_reagent_consistency",
                "w_entity",
            ]
        )
        return ReliabilityArtifacts(
            entity_weights,
            theta_by_entity_key,
            theta_valid,
            empty,
            empty.copy(),
            {
                "enabled": True,
                "strict_train_only": True,
                "modalities": allowed_names,
                "num_train_entity_keys": 0,
            },
        )

    entity_profiles: Dict[int, np.ndarray] = {}
    entity_rep_profiles: Dict[int, np.ndarray] = {}
    profiles_by_modality: Dict[int, List[np.ndarray]] = {}
    for entity_idx in train_entities.astype(int).tolist():
        reps = np.asarray(entity_to_reps[entity_idx], dtype=np.int64)
        if reps.size == 0:
            continue
        rep_mod = np.full(reps.size, int(entity_mods[entity_idx]), dtype=np.int64)
        x = transform_feature_rows(features, reps, rep_mod, norm).astype(np.float32)
        entity_rep_profiles[entity_idx] = x
        centroid = x.mean(axis=0).astype(np.float32)
        entity_profiles[entity_idx] = centroid
        profiles_by_modality.setdefault(int(entity_mods[entity_idx]), []).append(centroid)

    if not entity_profiles:
        raise RuntimeError("No train replicate profiles available for selected modality reliability estimation.")

    modality_fallback_centroids: Dict[int, np.ndarray] = {
        int(modality_id): np.vstack(vals).astype(np.float32).mean(axis=0).astype(np.float32)
        for modality_id, vals in profiles_by_modality.items()
        if vals
    }
    global_fallback_centroid = np.vstack(list(entity_profiles.values())).astype(np.float32).mean(axis=0).astype(np.float32)
    train_by_entity_key: Dict[int, List[int]] = {}
    for entity_idx in entity_profiles:
        train_by_entity_key.setdefault(int(entity_key_codes_all[entity_idx]), []).append(int(entity_idx))

    rows: List[Dict[str, Any]] = []
    key_order = sorted(train_by_entity_key)
    for entity_key_code in key_order:
        entity_idxs = train_by_entity_key[entity_key_code]
        reps_all = [entity_rep_profiles[e] for e in entity_idxs if e in entity_rep_profiles]
        if not reps_all:
            continue
        rep_stack = np.vstack(reps_all).astype(np.float32)
        theta = rep_stack.mean(axis=0).astype(np.float32)
        theta_by_entity_key[int(entity_key_code)] = theta
        theta_valid[int(entity_key_code)] = True
        modality_id = int(entity_mods[entity_idxs[0]])
        modality_name = str(modality_id_to_name.get(modality_id, str(modality_id))).lower()
        activity_centroid = modality_fallback_centroids.get(modality_id, global_fallback_centroid)
        activity = 1.0 - _cosine(theta, activity_centroid)
        rep_vals = []
        reagent_centroids = []
        for entity_idx in entity_idxs:
            x = entity_rep_profiles.get(entity_idx)
            if x is None:
                continue
            val = _pairwise_cosine_mean(x)
            if val is not None:
                rep_vals.append(float(val))
            reagent_centroids.append(entity_profiles[entity_idx])
        rep_cons = float(np.mean(rep_vals)) if rep_vals else 0.0
        reagent_stack = np.vstack(reagent_centroids).astype(np.float32) if reagent_centroids else np.zeros((0, features.shape[1]), dtype=np.float32)
        reagent_val = _pairwise_cosine_mean(reagent_stack)
        missing_reagent = reagent_val is None
        consistency_for_weight = float(default_guide_consistency if missing_reagent else reagent_val)
        is_crispr = modality_name == "crispr"
        guide_consistency = float(consistency_for_weight) if is_crispr else np.nan
        reagent_consistency = float(consistency_for_weight)
        missing_guide = bool(is_crispr and missing_reagent)
        reagent_ids = entities.iloc[entity_idxs]["reagent_id"].fillna("").astype(str).tolist() if "reagent_id" in entities.columns else [str(x) for x in entity_idxs]
        gene_code = int(gene_codes_all[entity_idxs[0]])
        entity_key = str(entity_key_strings[entity_idxs[0]])
        rows.append(
            {
                "gene_code": int(gene_code),
                "gene_id": str(gene_ids[entity_idxs[0]]),
                "gene_symbol": str(entities.loc[entity_idxs[0], "gene_symbol"]) if "gene_symbol" in entities.columns else str(gene_ids[entity_idxs[0]]),
                "modality": modality_name,
                "entity_key": entity_key,
                "entity_key_code": int(entity_key_code),
                "entity_indices": "|".join(str(x) for x in entity_idxs),
                "reagent_ids": "|".join(reagent_ids),
                "n_replicates": int(rep_stack.shape[0]),
                "n_guides": int(len(entity_idxs)) if is_crispr else 0,
                "n_reagents": int(len(entity_idxs)),
                "profile_activity": float(activity),
                "replicate_consistency": float(rep_cons),
                "guide_consistency": guide_consistency,
                "reagent_consistency": reagent_consistency,
                "consistency_for_weight": float(consistency_for_weight),
                "missing_guide_consistency": bool(missing_guide),
                "missing_reagent_consistency": bool((not is_crispr) and missing_reagent),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError("Reliability estimation produced no train entity keys.")
    activity_z = _zscore(table["profile_activity"].to_numpy(dtype=np.float32))
    max_n = max(float(table["n_replicates"].max()), 1.0)
    n_term = np.log1p(table["n_replicates"].to_numpy(dtype=np.float32)) / np.log1p(max_n)
    rep = np.clip(table["replicate_consistency"].to_numpy(dtype=np.float32), 0.0, 1.0)
    reagent = np.clip(table["consistency_for_weight"].to_numpy(dtype=np.float32), 0.0, 1.0)
    weights = (_sigmoid(activity_z) * rep * reagent * n_term).astype(np.float32)
    weights = np.clip(weights, 1e-4, 1.0)
    table["activity_z"] = activity_z
    table["w_entity"] = weights
    table["w_gene"] = weights
    key_to_weight = {int(k): float(w) for k, w in zip(table["entity_key_code"], weights)}
    for entity_idx in train_entities.astype(int).tolist():
        entity_weights[entity_idx] = float(key_to_weight.get(int(entity_key_codes_all[entity_idx]), 1.0))
    theta_table = table[
        [
            "gene_code",
            "gene_id",
            "gene_symbol",
            "modality",
            "entity_key",
            "entity_key_code",
            "n_replicates",
            "n_reagents",
        ]
    ].copy()
    theta_table["theta_valid"] = True

    summary: Dict[str, Any] = {
        "enabled": True,
        "strict_train_only": True,
        "modalities": allowed_names,
        "num_train_entities": int(train_entities.size),
        "num_train_genes": int(table["gene_code"].nunique()),
        "num_train_entity_keys": int(table.shape[0]),
        "activity_fallback": "activity uses modality-specific train global centroid fallback, not explicit negative controls",
        "negative_control_source": "modality_specific_train_global_centroid_fallback_not_explicit_negative_controls",
        "default_guide_consistency": float(default_guide_consistency),
        "missing_guide_consistency_fraction": float(table.loc[table["modality"] == "crispr", "missing_guide_consistency"].mean())
        if np.any(table["modality"].to_numpy() == "crispr")
        else 0.0,
        "missing_reagent_consistency_fraction": float(table.loc[table["modality"] != "crispr", "missing_reagent_consistency"].mean())
        if np.any(table["modality"].to_numpy() != "crispr")
        else 0.0,
        "profile_activity_mean": float(table["profile_activity"].mean()),
        "replicate_consistency_mean": float(table["replicate_consistency"].mean()),
        "guide_consistency_mean": float(table["guide_consistency"].dropna().mean()) if table["guide_consistency"].notna().any() else None,
        "reagent_consistency_mean": float(table["reagent_consistency"].mean()),
        "w_entity_mean": float(table["w_entity"].mean()),
        "w_entity_median": float(table["w_entity"].median()),
        "w_entity_min": float(table["w_entity"].min()),
        "w_entity_max": float(table["w_entity"].max()),
        "w_gene_mean": float(table["w_entity"].mean()),
        "train_entity_key_modality_counts": {str(k): int(v) for k, v in table["modality"].value_counts().items()},
    }
    return ReliabilityArtifacts(entity_weights.astype(np.float32), theta_by_entity_key, theta_valid, table, theta_table, summary)


def json_safe_modality_counts(counts: Dict[Any, Any]) -> str:
    return ",".join(f"{str(k)}:{int(v)}" for k, v in sorted(counts.items(), key=lambda x: str(x[0])))


def estimate_train_only_crispr_reliability(
    entities: pd.DataFrame,
    replicates: pd.DataFrame,
    features: np.ndarray,
    entity_to_reps: Sequence[np.ndarray],
    entity_mods: np.ndarray,
    gene_codes_all: np.ndarray,
    train_entities: np.ndarray,
    norm: Dict[str, Any],
    transform_feature_rows: TransformFn,
    crispr_modality_id: int = 1,
    default_guide_consistency: float = 0.5,
) -> ReliabilityArtifacts:
    """Backward-compatible CRISPR-only wrapper."""
    return estimate_train_only_gene_reliability(
        entities=entities,
        replicates=replicates,
        features=features,
        entity_to_reps=entity_to_reps,
        entity_mods=entity_mods,
        gene_codes_all=gene_codes_all,
        train_entities=train_entities,
        norm=norm,
        transform_feature_rows=transform_feature_rows,
        allowed_modality_ids=[int(crispr_modality_id)],
        default_guide_consistency=default_guide_consistency,
    )


def strict_no_leakage_assertions(
    entities: pd.DataFrame,
    splits: Dict[str, Any],
    split_name: str,
    gene_codes_all: np.ndarray,
    reliability_table: Optional[pd.DataFrame],
    theta_valid: Optional[np.ndarray],
    config: Dict[str, Any],
    entity_key_codes_all: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    train_rows = np.asarray(splits[split_name]["train"], dtype=np.int64)
    val_rows = np.asarray(splits[split_name]["val"], dtype=np.int64)
    test_rows = np.asarray(splits[split_name]["test"], dtype=np.int64)
    train_genes = set(gene_codes_all[train_rows].astype(int).tolist())
    val_genes = set(gene_codes_all[val_rows].astype(int).tolist())
    test_genes = set(gene_codes_all[test_rows].astype(int).tolist())
    if train_genes & val_genes:
        raise RuntimeError("strict_no_leakage failed: train/val gene overlap detected.")
    if train_genes & test_genes:
        raise RuntimeError("strict_no_leakage failed: train/test gene overlap detected.")
    if val_genes & test_genes:
        raise RuntimeError("strict_no_leakage failed: val/test gene overlap detected.")

    modes = parse_allowed_modalities(config.get("modalities", "orf,crispr"))
    if not modes:
        raise RuntimeError("strict_no_leakage failed: at least one modality is required.")
    observed_modes = set(entities["perturbation_modality"].astype(str).str.lower().unique().tolist())
    if not observed_modes <= set(modes):
        raise RuntimeError(f"strict_no_leakage failed: entity table contains modalities outside config: {sorted(observed_modes - set(modes))}.")
    if str(config.get("split_name", "")) != "cold_gene":
        raise RuntimeError("strict_no_leakage failed: split_name must be cold_gene.")
    if len(modes) > 1:
        if bool(config.get("disable_modality_context", False)):
            raise RuntimeError("strict_no_leakage failed: mixed CRISPR+ORF mode requires modality context in the gene encoder.")
        if str(config.get("protein_conditioning", "concat")) == "none":
            raise RuntimeError("strict_no_leakage failed: mixed CRISPR+ORF mode requires protein_conditioning != none.")
        if float(config.get("same_gene_crossmod_positive_weight", 0.0) or 0.0) > 0:
            raise RuntimeError("strict_no_leakage failed: same-gene cross-modality pairs cannot be hard positives in mixed entity-key retrieval.")

    entity_feature_mode = str(config.get("entity_feature_mode", "none"))
    if entity_feature_mode in {"metadata_plate", "metadata_well"}:
        raise RuntimeError(f"strict_no_leakage failed: forbidden entity_feature_mode={entity_feature_mode}.")
    if float(config.get("profile_adversary_well_weight", 0.0) or 0.0) > 0:
        raise RuntimeError("strict_no_leakage failed: well metadata cannot enter this strict phase-1 path.")
    if float(config.get("profile_adversary_plate_weight", 0.0) or 0.0) > 0:
        raise RuntimeError("strict_no_leakage failed: plate metadata cannot enter this strict phase-1 path.")
    fit_split = str(config.get("profile_normalizer_fit_split", "train"))
    if fit_split not in {"train", "none"}:
        raise RuntimeError(f"strict_no_leakage failed: profile normalizer fit_split must be train/none, got {fit_split!r}.")

    if reliability_table is not None and len(reliability_table):
        rel_genes = set(reliability_table["gene_code"].astype(int).tolist())
        if not rel_genes <= train_genes:
            raise RuntimeError("strict_no_leakage failed: reliability table contains non-train genes.")
        if entity_key_codes_all is not None and "entity_key_code" in reliability_table.columns:
            train_entity_keys = set(np.asarray(entity_key_codes_all)[train_rows].astype(int).tolist())
            rel_entity_keys = set(reliability_table["entity_key_code"].astype(int).tolist())
            if not rel_entity_keys <= train_entity_keys:
                raise RuntimeError("strict_no_leakage failed: reliability table contains non-train entity_keys.")
    if theta_valid is not None:
        theta_keys = {int(i) for i in np.where(theta_valid)[0].astype(int).tolist()}
        if entity_key_codes_all is not None:
            train_entity_keys = set(np.asarray(entity_key_codes_all)[train_rows].astype(int).tolist())
            if not theta_keys <= train_entity_keys:
                raise RuntimeError("strict_no_leakage failed: theta table contains non-train entity_keys.")
        else:
            if not theta_keys <= train_genes:
                raise RuntimeError("strict_no_leakage failed: theta table contains non-train genes.")

    return {
        "strict_no_leakage": True,
        "passed": True,
        "num_train_genes": int(len(train_genes)),
        "num_val_genes": int(len(val_genes)),
        "num_test_genes": int(len(test_genes)),
        "num_train_entity_keys": int(len(set(np.asarray(entity_key_codes_all)[train_rows].astype(int).tolist()))) if entity_key_codes_all is not None else None,
        "num_val_entity_keys": int(len(set(np.asarray(entity_key_codes_all)[val_rows].astype(int).tolist()))) if entity_key_codes_all is not None else None,
        "num_test_entity_keys": int(len(set(np.asarray(entity_key_codes_all)[test_rows].astype(int).tolist()))) if entity_key_codes_all is not None else None,
        "modalities": modes,
        "modality_context_required": bool(len(modes) > 1),
        "reliability_train_only": reliability_table is not None,
        "theta_train_only": theta_valid is not None,
    }
