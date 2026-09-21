from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export PRISM overlap SMILES for the official NYAN encoder and convert "
            "the official TSV output into a PRISM-order .npz cache."
        )
    )
    p.add_argument(
        "--overlap_csv",
        type=Path,
        default=Path("output/cgp_align/paper/manuscript_cgp_align_profile/prism_functional_analogue_v1/prism_cgp_overlap_compounds.csv"),
    )
    p.add_argument("--smiles_column", default="canonical_smiles")
    p.add_argument("--smi_path", type=Path, required=True)
    p.add_argument("--tsv_path", type=Path, required=True)
    p.add_argument("--output_cache", type=Path, required=True)
    p.add_argument("--export_only", action="store_true")
    p.add_argument("--convert_only", action="store_true")
    return p.parse_args()


def load_smiles(overlap_csv: Path, smiles_column: str) -> np.ndarray:
    df = pd.read_csv(overlap_csv)
    if smiles_column not in df.columns:
        raise KeyError(f"{smiles_column!r} not found in {overlap_csv}")
    smiles = df[smiles_column].astype(str).to_numpy()
    if smiles.ndim != 1 or len(smiles) == 0:
        raise ValueError(f"No SMILES loaded from {overlap_csv}")
    return smiles


def export_smi(smiles: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for i, smi in enumerate(smiles):
            handle.write(f"{smi} mol_{i}\n")


def convert_tsv(smiles: np.ndarray, tsv_path: Path, output_cache: Path) -> None:
    latent_by_smiles: dict[str, np.ndarray] = {}
    with tsv_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 65:
                continue
            latent_by_smiles[parts[0]] = np.asarray(parts[1:65], dtype=np.float32)

    latents = np.zeros((len(smiles), 64), dtype=np.float32)
    valid = np.zeros((len(smiles),), dtype=bool)
    for i, smi in enumerate(smiles):
        latent = latent_by_smiles.get(str(smi))
        if latent is None:
            continue
        latents[i] = latent
        valid[i] = True

    output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_cache, smiles=smiles.astype(str), latents=latents, valid=valid)
    print(
        {
            "reference_smiles": int(len(smiles)),
            "encoded_smiles": int(len(latent_by_smiles)),
            "valid": int(valid.sum()),
            "invalid": int((~valid).sum()),
            "output_cache": str(output_cache),
        }
    )


def main() -> None:
    args = parse_args()
    if args.export_only and args.convert_only:
        raise ValueError("--export_only and --convert_only are mutually exclusive")
    smiles = load_smiles(args.overlap_csv, args.smiles_column)

    if not args.convert_only:
        export_smi(smiles, args.smi_path)
        print({"exported_smiles": int(len(smiles)), "smi_path": str(args.smi_path)})

    if not args.export_only:
        convert_tsv(smiles, args.tsv_path, args.output_cache)


if __name__ == "__main__":
    main()
