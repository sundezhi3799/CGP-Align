from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_NYAN_CACHE_NAME = "prism_official_nyan_latents_cache.npz"


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(denom, eps)).astype(np.float32)


def _align_cache_rows(data: np.lib.npyio.NpzFile, smiles: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    target = [str(x) for x in smiles]
    cached_smiles = data["smiles"].astype(str).tolist()
    latents = data["latents"].astype(np.float32)
    valid = data["valid"].astype(bool)
    if cached_smiles == target:
        return latents, valid

    pos = {s: i for i, s in enumerate(cached_smiles)}
    out_latents = np.zeros((len(target), latents.shape[1]), dtype=np.float32)
    out_valid = np.zeros((len(target),), dtype=bool)
    for i, smi in enumerate(target):
        j = pos.get(smi)
        if j is None:
            continue
        out_latents[i] = latents[j]
        out_valid[i] = bool(valid[j])
    return out_latents, out_valid


def load_nyan_similarity(input_dir: Path, smiles: Sequence[str], cache_path: Path | None = None) -> np.ndarray:
    cache = cache_path or (input_dir / DEFAULT_NYAN_CACHE_NAME)
    if not cache.exists():
        raise FileNotFoundError(
            f"Missing NYAN latent cache: {cache}. Run scripts/build_prism_official_nyan_cache.py "
            "with the official NYAN encoder output first."
        )
    data = np.load(cache, allow_pickle=False)
    latents, valid = _align_cache_rows(data, smiles)
    if latents.shape[0] != len(smiles):
        raise ValueError(f"NYAN cache row mismatch: {latents.shape[0]} vs {len(smiles)}")
    z = l2_normalize(latents)
    sim = z @ z.T
    invalid = ~valid
    if invalid.any():
        sim[invalid, :] = -np.inf
        sim[:, invalid] = -np.inf
    np.fill_diagonal(sim, -np.inf)
    return sim.astype(np.float32)
