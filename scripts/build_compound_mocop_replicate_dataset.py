from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build replicate-level compound profiles from CPG all-profile parquet.")
    parser.add_argument("--profile_parquet", type=Path, default=Path("raw/cpg_profiles/profiles_all_featselect_sphering_harmony.parquet"))
    parser.add_argument("--profile_inventory", type=Path, default=Path("data/cpg_full_profile_inventory.parquet"))
    parser.add_argument("--compound_table", type=Path, default=Path("data/cgp_cpg_full/compounds.parquet"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--include_controls", action="store_true")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def inspect_feature_columns(profile_parquet: Path) -> List[str]:
    cols = pq.read_schema(profile_parquet).names
    x_cols = [c for c in cols if re.fullmatch(r"X_\d+", str(c))]
    if x_cols:
        return sorted(x_cols, key=lambda x: int(str(x).split("_", 1)[1]))
    prefixes = ("Cells_", "Cytoplasm_", "Nuclei_", "Image_")
    cp_cols = [str(c) for c in cols if str(c).startswith(prefixes)]
    if cp_cols:
        return cp_cols
    raise RuntimeError(f"No profile feature columns found in {profile_parquet}")


def joined_unique(values: pd.Series, limit: int = 256) -> str:
    uniq = sorted({clean_text(v) for v in values if clean_text(v)})
    if len(uniq) > limit:
        return "|".join(uniq[:limit]) + f"|...(+{len(uniq) - limit})"
    return "|".join(uniq)


def split_random(n: int, train_frac: float, val_frac: float, seed: int) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.arange(n, dtype=np.int64))
    n_train = min(max(int(round(n * train_frac)), 0), n)
    n_val = min(max(int(round(n * val_frac)), 0), n - n_train)
    return {
        "train": order[:n_train].astype(int).tolist(),
        "val": order[n_train : n_train + n_val].astype(int).tolist(),
        "test": order[n_train + n_val :].astype(int).tolist(),
    }


def summarize_entities(entities: pd.DataFrame, split: Dict[str, List[int]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for fold, idx in split.items():
        df = entities.iloc[idx]
        out[fold] = {
            "num_entities": int(len(df)),
            "replicate_count_sum": int(df["num_replicates"].sum()) if len(df) else 0,
            "replicate_count_mean": float(df["num_replicates"].mean()) if len(df) else 0.0,
        }
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_cols = inspect_feature_columns(args.profile_parquet)

    inv = pd.read_parquet(args.profile_inventory).copy()
    required = {"JCP2022", "standard_key", "plate_type", "pert_type"}
    missing = required - set(inv.columns)
    if missing:
        raise RuntimeError(f"Missing inventory columns: {sorted(missing)}")
    inv["JCP2022"] = inv["JCP2022"].map(clean_text)
    inv["standard_key"] = inv["standard_key"].map(clean_text)
    inv["plate_type"] = inv["plate_type"].astype(str).str.lower().str.strip()
    inv["pert_type"] = inv["pert_type"].astype(str).str.lower().str.strip()
    inv = inv[inv["plate_type"].eq("compound")].copy()
    if not args.include_controls:
        inv = inv[inv["pert_type"].eq("trt")].copy()
    inv = inv[(inv["JCP2022"] != "") & (inv["standard_key"] != "")].copy()
    inv_meta = inv[["JCP2022", "standard_key", "pert_type"]].drop_duplicates()

    use_cols = ["Metadata_Source", "Metadata_Plate", "Metadata_Well", "Metadata_JCP2022"] + feature_cols
    profiles = pd.read_parquet(args.profile_parquet, columns=use_cols)
    profiles = profiles.merge(inv_meta, left_on="Metadata_JCP2022", right_on="JCP2022", how="inner")
    if profiles.empty:
        raise RuntimeError("No compound replicate profiles matched inventory.")
    profiles = profiles.reset_index(drop=True)
    profiles["compound_id"] = profiles["standard_key"].map(clean_text)
    profiles["entity_key"] = profiles["compound_id"]

    features = profiles[feature_cols].to_numpy(dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    np.save(args.output_dir / "compound_mocop_replicate_features.npy", features.astype(np.float32))

    group = profiles.groupby("entity_key", sort=True)
    entities = group.agg(
        compound_id=("compound_id", "first"),
        num_replicates=("Metadata_JCP2022", "size"),
        num_jcp2022_ids=("JCP2022", "nunique"),
        num_plates=("Metadata_Plate", "nunique"),
        num_sources=("Metadata_Source", "nunique"),
        jcp2022_ids=("JCP2022", joined_unique),
        metadata_sources=("Metadata_Source", joined_unique),
        metadata_plates=("Metadata_Plate", joined_unique),
        metadata_wells=("Metadata_Well", joined_unique),
    ).reset_index(drop=True)
    entities = entities.sort_values("compound_id").reset_index(drop=True)
    entities["entity_index"] = np.arange(len(entities), dtype=np.int64)
    entities["entity_id"] = entities["compound_id"].astype(str)
    entities["perturbation_modality"] = "compound"

    if args.compound_table.exists():
        comp = pd.read_parquet(args.compound_table).copy()
        keep = [c for c in ["compound_id", "canonical_smiles", "inchikey", "pubchem_cid", "has_structure", "structure_feature_index"] if c in comp.columns]
        comp = comp[keep].drop_duplicates("compound_id")
        entities = entities.merge(comp, on="compound_id", how="left")
    for col in ["canonical_smiles", "inchikey", "pubchem_cid"]:
        if col not in entities.columns:
            entities[col] = ""
        entities[col] = entities[col].fillna("").astype(str)
    for col in ["has_structure"]:
        if col not in entities.columns:
            entities[col] = False
        entities[col] = entities[col].fillna(False).astype(bool)
    for col in ["structure_feature_index"]:
        if col not in entities.columns:
            entities[col] = -1
        entities[col] = entities[col].fillna(-1).astype(np.int64)

    entity_to_idx = dict(zip(entities["compound_id"].astype(str), entities["entity_index"].astype(int)))
    replicates = pd.DataFrame({
        "replicate_id": [f"CMREP_{i:08d}" for i in range(len(profiles))],
        "entity_key": profiles["entity_key"].astype(str),
        "entity_index": profiles["entity_key"].map(entity_to_idx).astype(np.int64),
        "entity_id": profiles["entity_key"].astype(str),
        "compound_id": profiles["compound_id"].astype(str),
        "jcp2022_id": profiles["JCP2022"].astype(str),
        "perturbation_modality": "compound",
        "Metadata_Source": profiles["Metadata_Source"].astype(str),
        "Metadata_Plate": profiles["Metadata_Plate"].astype(str),
        "Metadata_Well": profiles["Metadata_Well"].astype(str),
        "feature_index": np.arange(len(profiles), dtype=np.int64),
    })

    entity_cols = [
        "entity_index", "entity_id", "compound_id", "canonical_smiles", "inchikey", "pubchem_cid",
        "has_structure", "structure_feature_index", "perturbation_modality", "num_replicates",
        "num_jcp2022_ids", "num_plates", "num_sources", "jcp2022_ids", "metadata_sources",
        "metadata_plates", "metadata_wells",
    ]
    entities = entities[[c for c in entity_cols if c in entities.columns]]
    entities.to_parquet(args.output_dir / "compound_mocop_entities.parquet", index=False)
    replicates.to_parquet(args.output_dir / "compound_mocop_replicates.parquet", index=False)

    split = split_random(len(entities), args.train_frac, args.val_frac, args.seed)
    splits = {"cold_compound": split, "random_entity": split, "split_index_type": "entity_index", "seed": int(args.seed)}
    (args.output_dir / "splits_compound_mocop.json").write_text(json.dumps(splits, indent=2, sort_keys=True), encoding="utf-8")
    (args.output_dir / "feature_columns.json").write_text(json.dumps({"feature_columns": feature_cols}, indent=2), encoding="utf-8")

    summary = {
        "dataset_name": "compound_mocop_replicate",
        "profile_parquet": str(args.profile_parquet),
        "profile_inventory": str(args.profile_inventory),
        "official_processed_profile_used": True,
        "raw_images_downloaded": False,
        "controls_included": bool(args.include_controls),
        "entity_level": "compound standard_key / InChIKey",
        "profile_level": "replicate/well-level processed profile",
        "feature_dim": int(features.shape[1]),
        "num_entities": int(len(entities)),
        "num_replicates": int(len(replicates)),
        "num_unique_jcp2022": int(replicates["jcp2022_id"].nunique()),
        "replicate_count_summary": {str(k): float(v) for k, v in entities["num_replicates"].describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).to_dict().items()},
        "split_sizes": {"cold_compound": summarize_entities(entities, split)},
    }
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
