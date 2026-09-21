from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None

from cgp_common import set_seed, write_json


GENE_MOCOP_ROOT = Path("output/cgp_align/gene_mocop")
GENE_MOCOP_DATA_DIR = GENE_MOCOP_ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build replicate-level ORF/CRISPR gene perturbation profiles for MoCoP-style protein-profile CLIP."
    )
    parser.add_argument("--profile_parquet", type=Path, default=Path("raw/cpg_profiles/profiles_all_featselect_sphering_harmony.parquet"))
    parser.add_argument("--profile_inventory", type=Path, default=Path("data/cpg_full_profile_inventory.parquet"))
    parser.add_argument("--source_gene_table", type=Path, default=Path("data/cgp_cpg_full_motive_edges/genes.parquet"))
    parser.add_argument("--output_dir", type=Path, default=GENE_MOCOP_DATA_DIR)
    parser.add_argument("--modalities", default="orf,crispr")
    parser.add_argument(
        "--entity_level",
        choices=["gene_modality", "reagent"],
        default="gene_modality",
        help=(
            "gene_modality keeps the original gene_symbol + perturbation_modality entity. "
            "reagent uses gene_symbol + perturbation_modality + JCP2022 so ORF clones/constructs are not merged."
        ),
    )
    parser.add_argument("--include_controls", action="store_true")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--max_genes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def slug(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", clean_text(value)).strip("_")


def inspect_feature_columns(profile_parquet: Path) -> List[str]:
    if pq is not None:
        cols = [field.name for field in pq.read_schema(profile_parquet)]
    else:
        cols = list(pd.read_parquet(profile_parquet).columns)
    feature_cols = [c for c in cols if re.fullmatch(r"X_\d+", str(c))]
    feature_cols = sorted(feature_cols, key=lambda x: int(str(x).split("_", 1)[1]))
    if not feature_cols:
        raise RuntimeError(f"No X_* feature columns found in {profile_parquet}")
    return feature_cols


def load_inventory(path: Path, modalities: Sequence[str], include_controls: bool) -> pd.DataFrame:
    inv = pd.read_parquet(path).copy()
    required = {"JCP2022", "standard_key", "plate_type", "pert_type"}
    missing = required - set(inv.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in {path}: {sorted(missing)}")
    inv["JCP2022"] = inv["JCP2022"].map(clean_text)
    inv["standard_key"] = inv["standard_key"].map(clean_text)
    inv["plate_type"] = inv["plate_type"].astype(str).str.lower().str.strip()
    inv["pert_type"] = inv["pert_type"].astype(str).str.lower().str.strip()
    keep = {m.strip().lower() for m in modalities if m.strip()}
    inv = inv[inv["plate_type"].isin(keep)].copy()
    if not include_controls:
        inv = inv[inv["pert_type"].eq("trt")].copy()
    inv = inv[(inv["JCP2022"] != "") & (inv["standard_key"] != "")].copy()
    return inv


def source_gene_lookup(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "gene_symbol",
                "source_gene_index",
                "uniprot_id",
                "protein_sequence",
                "has_protein_sequence",
                "protein_embedding_index",
            ]
        )
    genes = pd.read_parquet(path).reset_index(drop=True).copy()
    genes["source_gene_index"] = np.arange(len(genes), dtype=np.int64)
    pieces = []
    for col in [c for c in ["gene_id", "gene_symbol"] if c in genes.columns]:
        keep_cols = [
            col,
            "source_gene_index",
            "uniprot_id",
            "protein_sequence",
            "has_protein_sequence",
            "protein_embedding_index",
        ]
        piece = genes[keep_cols].rename(columns={col: "gene_symbol"}).copy()
        pieces.append(piece)
    if not pieces:
        return pd.DataFrame(columns=["gene_symbol"])
    lookup = pd.concat(pieces, ignore_index=True)
    lookup["gene_symbol"] = lookup["gene_symbol"].map(clean_text)
    lookup = lookup[lookup["gene_symbol"] != ""].drop_duplicates("gene_symbol")
    return lookup


def joined_unique(values: pd.Series, limit: int = 128) -> str:
    uniq = sorted({clean_text(v) for v in values if clean_text(v)})
    if len(uniq) > limit:
        return "|".join(uniq[:limit]) + f"|...(+{len(uniq) - limit})"
    return "|".join(uniq)


def split_random_entities(n: int, train_frac: float, val_frac: float, seed: int) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.arange(n, dtype=np.int64))
    n_train = min(max(int(round(n * train_frac)), 0), n)
    n_val = min(max(int(round(n * val_frac)), 0), n - n_train)
    return {
        "train": order[:n_train].astype(int).tolist(),
        "val": order[n_train : n_train + n_val].astype(int).tolist(),
        "test": order[n_train + n_val :].astype(int).tolist(),
    }


def split_cold_gene(entities: pd.DataFrame, train_frac: float, val_frac: float, seed: int) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed)
    genes = rng.permutation(entities["gene_symbol"].drop_duplicates().astype(str).to_numpy())
    n = len(genes)
    n_train = min(max(int(round(n * train_frac)), 0), n)
    n_val = min(max(int(round(n * val_frac)), 0), n - n_train)
    gene_sets = {
        "train": set(genes[:n_train]),
        "val": set(genes[n_train : n_train + n_val]),
        "test": set(genes[n_train + n_val :]),
    }
    out = {}
    for fold, gene_set in gene_sets.items():
        out[fold] = entities.index[entities["gene_symbol"].astype(str).isin(gene_set)].astype(int).tolist()
    return out


def summarize_entities(entities: pd.DataFrame, split: Dict[str, List[int]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for fold, idx in split.items():
        df = entities.iloc[idx]
        out[fold] = {
            "num_entities": int(len(df)),
            "num_genes": int(df["gene_symbol"].nunique()),
            "modality_counts": {str(k): int(v) for k, v in df["perturbation_modality"].value_counts().items()},
            "replicate_count_sum": int(df["num_replicates"].sum()) if len(df) else 0,
        }
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    modalities = [m.strip().lower() for m in args.modalities.split(",") if m.strip()]

    feature_cols = inspect_feature_columns(args.profile_parquet)
    inventory = load_inventory(args.profile_inventory, modalities, args.include_controls)
    if args.smoke_test or args.max_genes:
        max_genes = int(args.max_genes or 512)
        keep_genes = sorted(inventory["standard_key"].drop_duplicates().astype(str).tolist())[:max_genes]
        inventory = inventory[inventory["standard_key"].isin(keep_genes)].copy()

    meta_cols = ["JCP2022", "standard_key", "plate_type", "pert_type"]
    for optional in ["NCBI_Gene_ID", "broad_sample"]:
        if optional in inventory.columns:
            meta_cols.append(optional)
    inv_meta = inventory[meta_cols].drop_duplicates()

    use_cols = ["Metadata_JCP2022", "Metadata_Source", "Metadata_Plate", "Metadata_Well"] + feature_cols
    profiles = pd.read_parquet(args.profile_parquet, columns=use_cols)
    profiles = profiles.merge(inv_meta, left_on="Metadata_JCP2022", right_on="JCP2022", how="inner")
    if profiles.empty:
        raise RuntimeError("No replicate profiles matched the selected ORF/CRISPR inventory.")
    profiles = profiles.reset_index(drop=True)
    profiles["gene_symbol"] = profiles["standard_key"].map(clean_text)
    profiles["perturbation_modality"] = profiles["plate_type"].astype(str).str.lower()
    profiles["reagent_id"] = profiles["JCP2022"].map(clean_text)
    if args.entity_level == "reagent":
        profiles["entity_key"] = profiles["gene_symbol"] + "|" + profiles["perturbation_modality"] + "|" + profiles["reagent_id"]
    else:
        profiles["entity_key"] = profiles["gene_symbol"] + "|" + profiles["perturbation_modality"]

    features = profiles[feature_cols].to_numpy(dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    np.save(args.output_dir / "gene_mocop_replicate_features.npy", features.astype(np.float32))

    entity_group = profiles.groupby("entity_key", sort=True)
    entities = entity_group.agg(
        gene_symbol=("gene_symbol", "first"),
        perturbation_modality=("perturbation_modality", "first"),
        num_replicates=("Metadata_JCP2022", "size"),
        num_jcp2022_ids=("JCP2022", "nunique"),
        num_plates=("Metadata_Plate", "nunique"),
        num_sources=("Metadata_Source", "nunique"),
        jcp2022_ids=("JCP2022", joined_unique),
        reagent_id=("reagent_id", "first"),
        metadata_sources=("Metadata_Source", joined_unique),
        metadata_plates=("Metadata_Plate", joined_unique),
    ).reset_index()
    if "NCBI_Gene_ID" in profiles.columns:
        entrez = entity_group["NCBI_Gene_ID"].agg(lambda s: joined_unique(s, limit=8)).reset_index(drop=True)
        entities["entrez_id"] = entrez
    else:
        entities["entrez_id"] = ""
    if "broad_sample" in profiles.columns:
        broad = entity_group["broad_sample"].agg(lambda s: joined_unique(s, limit=16)).reset_index(drop=True)
        entities["broad_sample_ids"] = broad
    else:
        entities["broad_sample_ids"] = ""

    sort_cols = ["gene_symbol", "perturbation_modality"]
    if "reagent_id" in entities.columns:
        sort_cols.append("reagent_id")
    entities = entities.sort_values(sort_cols).reset_index(drop=True)
    entities["entity_index"] = np.arange(len(entities), dtype=np.int64)
    if args.entity_level == "reagent":
        entities["entity_id"] = [
            f"{slug(gene)}|{slug(modality)}|{slug(reagent)}"
            for gene, modality, reagent in zip(
                entities["gene_symbol"].astype(str),
                entities["perturbation_modality"].astype(str),
                entities["reagent_id"].astype(str),
            )
        ]
    else:
        entities["entity_id"] = [
            f"{slug(gene)}|{slug(modality)}"
            for gene, modality in zip(entities["gene_symbol"].astype(str), entities["perturbation_modality"].astype(str))
        ]

    lookup = source_gene_lookup(args.source_gene_table)
    entities = entities.merge(lookup, on="gene_symbol", how="left")
    entities["source_gene_index"] = entities["source_gene_index"].fillna(-1).astype(np.int64)
    entities["protein_embedding_index"] = entities["protein_embedding_index"].fillna(entities["source_gene_index"]).astype(np.int64)
    entities["uniprot_id"] = entities["uniprot_id"].fillna("").astype(str)
    entities["protein_sequence"] = entities["protein_sequence"].fillna("").astype(str)
    entities["has_protein_sequence"] = entities["has_protein_sequence"].fillna(False).astype(bool) & entities["protein_sequence"].str.len().gt(0)

    entity_key_to_idx = dict(zip(entities["entity_key"], entities["entity_index"]))
    replicates = pd.DataFrame(
        {
            "replicate_id": [f"GMREP_{i:08d}" for i in range(len(profiles))],
            "entity_key": profiles["entity_key"].astype(str),
            "entity_index": profiles["entity_key"].map(entity_key_to_idx).astype(np.int64),
            "entity_id": profiles["entity_key"].map(lambda k: entities.loc[entity_key_to_idx[k], "entity_id"]),
            "gene_symbol": profiles["gene_symbol"].astype(str),
            "perturbation_modality": profiles["perturbation_modality"].astype(str),
            "jcp2022_id": profiles["JCP2022"].astype(str),
            "reagent_id": profiles["reagent_id"].astype(str),
            "Metadata_Source": profiles["Metadata_Source"].astype(str),
            "Metadata_Plate": profiles["Metadata_Plate"].astype(str),
            "Metadata_Well": profiles["Metadata_Well"].astype(str),
            "feature_index": np.arange(len(profiles), dtype=np.int64),
        }
    )

    # Make entity_index match the final row order.
    old_to_new = dict(zip(entities["entity_index"].astype(int), np.arange(len(entities), dtype=np.int64)))
    entities["entity_index"] = np.arange(len(entities), dtype=np.int64)
    replicates["entity_index"] = replicates["entity_index"].map(old_to_new).astype(np.int64)
    replicates["entity_id"] = entities.loc[replicates["entity_index"].to_numpy(), "entity_id"].to_numpy()

    entity_cols = [
        "entity_index",
        "entity_id",
        "gene_symbol",
        "perturbation_modality",
        "entrez_id",
        "uniprot_id",
        "source_gene_index",
        "protein_embedding_index",
        "protein_sequence",
        "has_protein_sequence",
        "num_replicates",
        "num_jcp2022_ids",
        "num_plates",
        "num_sources",
        "jcp2022_ids",
        "reagent_id",
        "broad_sample_ids",
        "metadata_sources",
        "metadata_plates",
    ]
    entities = entities[[c for c in entity_cols if c in entities.columns]]
    entities.to_parquet(args.output_dir / "gene_mocop_entities.parquet", index=False)
    replicates.to_parquet(args.output_dir / "gene_mocop_replicates.parquet", index=False)

    random_split = split_random_entities(len(entities), args.train_frac, args.val_frac, args.seed)
    cold_gene_split = split_cold_gene(entities, args.train_frac, args.val_frac, args.seed)
    splits = {
        "random_entity": random_split,
        "cold_gene": cold_gene_split,
        "split_index_type": "entity_index",
        "seed": int(args.seed),
    }
    write_json(args.output_dir / "splits_gene_mocop.json", splits)

    summary = {
        "dataset_name": "gene_mocop_replicate",
        "synthetic_smoke": bool(args.smoke_test),
        "profile_parquet": str(args.profile_parquet),
        "profile_inventory": str(args.profile_inventory),
        "source_gene_table": str(args.source_gene_table),
        "official_processed_profile_used": True,
        "raw_images_downloaded": False,
        "entity_level": "gene_symbol + perturbation_modality + JCP2022" if args.entity_level == "reagent" else "gene_symbol + perturbation_modality",
        "entity_level_mode": str(args.entity_level),
        "profile_level": "replicate/well-level processed profile",
        "modalities": modalities,
        "controls_included": bool(args.include_controls),
        "feature_dim": int(features.shape[1]),
        "num_entities": int(len(entities)),
        "num_replicates": int(len(replicates)),
        "num_unique_genes": int(entities["gene_symbol"].nunique()),
        "num_orf_entities": int((entities["perturbation_modality"] == "orf").sum()),
        "num_crispr_entities": int((entities["perturbation_modality"] == "crispr").sum()),
        "num_entities_with_multiple_jcp2022_ids": int((entities["num_jcp2022_ids"] > 1).sum()),
        "num_orf_entities_with_multiple_jcp2022_ids": int(((entities["perturbation_modality"] == "orf") & (entities["num_jcp2022_ids"] > 1)).sum()),
        "num_unique_reagents": int(replicates["jcp2022_id"].nunique()),
        "num_orf_replicates": int((replicates["perturbation_modality"] == "orf").sum()),
        "num_crispr_replicates": int((replicates["perturbation_modality"] == "crispr").sum()),
        "genes_with_both_orf_and_crispr": int(entities.groupby("gene_symbol")["perturbation_modality"].nunique().ge(2).sum()),
        "entities_with_protein_sequence": int(entities["has_protein_sequence"].astype(bool).sum()),
        "missing_protein_sequence_entities": int((~entities["has_protein_sequence"].astype(bool)).sum()),
        "replicate_count_by_modality": {
            str(k): {str(stat): float(val) for stat, val in desc.items()}
            for k, desc in entities.groupby("perturbation_modality")["num_replicates"].describe().to_dict(orient="index").items()
        },
        "split_sizes": {
            "random_entity": summarize_entities(entities, random_split),
            "cold_gene": summarize_entities(entities, cold_gene_split),
        },
    }
    write_json(args.output_dir / "dataset_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
