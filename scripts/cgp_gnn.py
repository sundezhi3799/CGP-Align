from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cgp_common import ProfileEncoder
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgp_align.models.modules import MLPEncoder

try:
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.warning")
except Exception:  # pragma: no cover
    Chem = None


ATOM_NUMS = [1, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20, 26, 35, 53]
BOND_TYPES = []
if Chem is not None:
    BOND_TYPES = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]


@dataclass
class MolGraph:
    atom_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray


def one_hot_unknown(value: object, choices: Sequence[object]) -> List[float]:
    return [float(value == choice) for choice in choices] + [float(value not in choices)]


def atom_feature_vector(atom: object) -> List[float]:
    hyb_choices = []
    if Chem is not None:
        hyb_choices = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3]
    features: List[float] = []
    features += one_hot_unknown(int(atom.GetAtomicNum()), ATOM_NUMS)
    features += one_hot_unknown(int(atom.GetTotalDegree()), [0, 1, 2, 3, 4, 5])
    features += one_hot_unknown(int(atom.GetFormalCharge()), [-2, -1, 0, 1, 2])
    features += one_hot_unknown(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4])
    features += one_hot_unknown(atom.GetHybridization(), hyb_choices)
    features += [
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(atom.HasProp("_ChiralityPossible")),
    ]
    return features


def bond_feature_vector(bond: object) -> List[float]:
    features: List[float] = []
    features += one_hot_unknown(bond.GetBondType(), BOND_TYPES)
    features += [
        float(bond.GetIsConjugated()),
        float(bond.GetIsAromatic()),
        float(bond.IsInRing()),
    ]
    return features


def atom_feature_dim() -> int:
    if Chem is None:
        return 47
    mol = Chem.MolFromSmiles("CC")
    return len(atom_feature_vector(mol.GetAtomWithIdx(0)))


def bond_feature_dim() -> int:
    if Chem is None:
        return 8
    mol = Chem.MolFromSmiles("CC")
    return len(bond_feature_vector(mol.GetBondWithIdx(0)))


def graph_from_smiles(smiles: str) -> MolGraph:
    if Chem is None:
        raise RuntimeError("RDKit is required for GNN compound encoding.")
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        x = np.zeros((1, atom_feature_dim()), dtype=np.float32)
        e = np.zeros((2, 0), dtype=np.int64)
        ef = np.zeros((0, bond_feature_dim()), dtype=np.float32)
        return MolGraph(x, e, ef)
    atom_x = np.asarray([atom_feature_vector(atom) for atom in mol.GetAtoms()], dtype=np.float32)
    edge_pairs: List[Tuple[int, int]] = []
    edge_attr: List[List[float]] = []
    for bond in mol.GetBonds():
        i = int(bond.GetBeginAtomIdx())
        j = int(bond.GetEndAtomIdx())
        bf = bond_feature_vector(bond)
        edge_pairs.append((i, j))
        edge_attr.append(bf)
        edge_pairs.append((j, i))
        edge_attr.append(bf)
    if edge_pairs:
        edge_index = np.asarray(edge_pairs, dtype=np.int64).T
        edge_features = np.asarray(edge_attr, dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_features = np.zeros((0, bond_feature_dim()), dtype=np.float32)
    return MolGraph(atom_x, edge_index, edge_features)


class GraphStore:
    def __init__(self, smiles: Sequence[str]):
        self.smiles = [str(s) for s in smiles]

    @lru_cache(maxsize=200000)
    def get(self, index: int) -> MolGraph:
        return graph_from_smiles(self.smiles[int(index)])

    def get_many(self, indices: Sequence[int]) -> List[MolGraph]:
        return [self.get(int(i)) for i in indices]


def batch_graphs(graphs: Sequence[MolGraph], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    atom_parts: List[np.ndarray] = []
    edge_parts: List[np.ndarray] = []
    edge_feature_parts: List[np.ndarray] = []
    batch_index_parts: List[np.ndarray] = []
    offset = 0
    for graph_id, graph in enumerate(graphs):
        n = int(graph.atom_features.shape[0])
        atom_parts.append(graph.atom_features)
        batch_index_parts.append(np.full(n, graph_id, dtype=np.int64))
        if graph.edge_index.shape[1] > 0:
            edge_parts.append(graph.edge_index + offset)
            edge_feature_parts.append(graph.edge_features)
        offset += n
    atom_x = torch.as_tensor(np.vstack(atom_parts), dtype=torch.float32, device=device)
    batch_index = torch.as_tensor(np.concatenate(batch_index_parts), dtype=torch.long, device=device)
    if edge_parts:
        edge_index = torch.as_tensor(np.concatenate(edge_parts, axis=1), dtype=torch.long, device=device)
        edge_features = torch.as_tensor(np.vstack(edge_feature_parts), dtype=torch.float32, device=device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_features = torch.zeros((0, bond_feature_dim()), dtype=torch.float32, device=device)
    return atom_x, edge_index, edge_features, batch_index


class GGNNCompoundEncoder(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 6,
        projection_hidden_dims: Sequence[int] = (512,),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.atom_input = nn.Linear(atom_dim, hidden_dim)
        self.bond_input = nn.Linear(bond_dim, hidden_dim)
        self.message_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.update_layers = nn.ModuleList([nn.GRUCell(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.projection = MLPEncoder(hidden_dim, list(projection_hidden_dims), embed_dim, dropout)

    def forward(self, atom_x: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.atom_input(atom_x))
        if edge_features.numel() > 0:
            bond_h = F.gelu(self.bond_input(edge_features))
        else:
            bond_h = edge_features.new_zeros((0, h.shape[-1]))
        for msg_layer, update_layer, norm_layer in zip(self.message_layers, self.update_layers, self.norm_layers):
            agg = torch.zeros_like(h)
            if edge_index.numel() > 0:
                src = edge_index[0]
                dst = edge_index[1]
                msg = msg_layer(torch.cat([h[src], bond_h], dim=-1))
                agg.index_add_(0, dst, msg.to(dtype=agg.dtype))
            h = update_layer(agg, h)
            h = norm_layer(h)
        num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
        pooled = h.new_zeros((num_graphs, h.shape[-1]))
        pooled.index_add_(0, batch_index, h)
        counts = torch.bincount(batch_index, minlength=num_graphs).to(h.dtype).clamp_min(1.0).unsqueeze(-1)
        pooled = pooled / counts
        return self.projection(pooled)


class CGPAlignGNNModel(nn.Module):
    def __init__(
        self,
        protein_dim: int,
        profile_dim: int,
        embed_dim: int = 256,
        gnn_hidden_dim: int = 256,
        gnn_layers: int = 6,
        dropout: float = 0.1,
        feature_dim_by_source: Dict[str, int] | None = None,
        protein_hidden_dims: Sequence[int] = (512,),
        profile_hidden_dim: int = 1024,
    ):
        super().__init__()
        self.compound_encoder = GGNNCompoundEncoder(
            atom_feature_dim(),
            bond_feature_dim(),
            embed_dim=embed_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_layers,
            projection_hidden_dims=(512,),
            dropout=dropout,
        )
        self.protein_encoder = MLPEncoder(protein_dim, list(protein_hidden_dims), embed_dim, dropout)
        self.profile_encoder = ProfileEncoder(
            profile_dim,
            embed_dim,
            hidden_dim=profile_hidden_dim,
            dropout=dropout,
            feature_dim_by_source=feature_dim_by_source,
        )

    def encode_compound_graphs(self, graphs: Sequence[MolGraph], device: torch.device) -> torch.Tensor:
        atom_x, edge_index, edge_features, batch_index = batch_graphs(graphs, device)
        return self.compound_encoder(atom_x, edge_index, edge_features, batch_index)

    def encode_protein(self, x: torch.Tensor) -> torch.Tensor:
        return self.protein_encoder(x)

    def encode_profile(self, x: torch.Tensor, source_ids: torch.Tensor) -> torch.Tensor:
        return self.profile_encoder(x, source_ids)
