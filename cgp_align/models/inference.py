from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch

from .graph import GraphStore
from .model import CGPAlign, MODALITY_TO_ID


def load_model_weights(model: CGPAlign, checkpoint_path: str | Path, map_location: str = "cpu") -> CGPAlign:
    payload = torch.load(str(checkpoint_path), map_location=map_location)
    state_dict: Dict[str, Any] = payload.get("model_state_dict", payload)
    model.load_state_dict(state_dict, strict=True)
    return model


def encode_compounds_from_smiles(
    model: CGPAlign,
    smiles: Sequence[str],
    batch_size: int = 256,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    device = torch.device(device)
    model.to(device).eval()
    store = GraphStore(smiles)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(smiles), max(1, int(batch_size))):
            rows = list(range(start, min(start + int(batch_size), len(smiles))))
            z = model.encode_compound_graphs(store.get_many(rows), device=device)
            outputs.append(z.cpu().numpy().astype(np.float32))
    return np.vstack(outputs) if outputs else np.zeros((0, model.config.embed_dim), dtype=np.float32)


def encode_gene_embeddings(
    model: CGPAlign,
    protein_embeddings: np.ndarray,
    modalities: Sequence[str | int],
    batch_size: int = 256,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    device = torch.device(device)
    model.to(device).eval()
    protein_embeddings = np.asarray(protein_embeddings, dtype=np.float32)
    modality_ids = np.asarray([MODALITY_TO_ID.get(str(m).lower(), int(m)) if not isinstance(m, str) else MODALITY_TO_ID[str(m).lower()] for m in modalities], dtype=np.int64)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(protein_embeddings), max(1, int(batch_size))):
            x = torch.as_tensor(protein_embeddings[start : start + int(batch_size)], dtype=torch.float32, device=device)
            m = torch.as_tensor(modality_ids[start : start + int(batch_size)], dtype=torch.long, device=device)
            z = model.encode_gene(x, m)
            outputs.append(z.cpu().numpy().astype(np.float32))
    return np.vstack(outputs) if outputs else np.zeros((0, model.config.embed_dim), dtype=np.float32)


def encode_profiles(
    model: CGPAlign,
    profiles: np.ndarray,
    source_ids: Sequence[str | int],
    batch_size: int = 256,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    device = torch.device(device)
    model.to(device).eval()
    profiles = np.asarray(profiles, dtype=np.float32)
    ids = np.asarray([MODALITY_TO_ID.get(str(s).lower(), int(s)) if not isinstance(s, str) else MODALITY_TO_ID[str(s).lower()] for s in source_ids], dtype=np.int64)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(profiles), max(1, int(batch_size))):
            x = torch.as_tensor(profiles[start : start + int(batch_size)], dtype=torch.float32, device=device)
            s = torch.as_tensor(ids[start : start + int(batch_size)], dtype=torch.long, device=device)
            z = model.encode_profile(x, s)
            outputs.append(z.cpu().numpy().astype(np.float32))
    return np.vstack(outputs) if outputs else np.zeros((0, model.config.embed_dim), dtype=np.float32)
