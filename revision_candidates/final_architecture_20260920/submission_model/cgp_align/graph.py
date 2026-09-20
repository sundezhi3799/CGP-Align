from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np
import torch

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
    if Chem is None:
        raise RuntimeError("RDKit is required to featurize atoms from SMILES.")
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
    if Chem is None:
        raise RuntimeError("RDKit is required to featurize bonds from SMILES.")
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
        return 44
    mol = Chem.MolFromSmiles("CC")
    return len(atom_feature_vector(mol.GetAtomWithIdx(0)))


def bond_feature_dim() -> int:
    if Chem is None:
        return 8
    mol = Chem.MolFromSmiles("CC")
    return len(bond_feature_vector(mol.GetBondWithIdx(0)))


def graph_from_smiles(smiles: str) -> MolGraph:
    if Chem is None:
        raise RuntimeError("RDKit is required for SMILES-to-graph conversion.")
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        atom_x = np.zeros((1, atom_feature_dim()), dtype=np.float32)
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_features = np.zeros((0, bond_feature_dim()), dtype=np.float32)
        return MolGraph(atom_x, edge_index, edge_features)

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
        self.smiles = [str(value) for value in smiles]

    @lru_cache(maxsize=200000)
    def get(self, index: int) -> MolGraph:
        return graph_from_smiles(self.smiles[int(index)])

    def get_many(self, indices: Sequence[int]) -> List[MolGraph]:
        return [self.get(int(index)) for index in indices]


def batch_graphs(graphs: Sequence[MolGraph], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    atom_parts: List[np.ndarray] = []
    edge_parts: List[np.ndarray] = []
    edge_feature_parts: List[np.ndarray] = []
    batch_index_parts: List[np.ndarray] = []
    offset = 0

    for graph_id, graph in enumerate(graphs):
        n_atoms = int(graph.atom_features.shape[0])
        atom_parts.append(graph.atom_features)
        batch_index_parts.append(np.full(n_atoms, graph_id, dtype=np.int64))
        if graph.edge_index.shape[1] > 0:
            edge_parts.append(graph.edge_index + offset)
            edge_feature_parts.append(graph.edge_features)
        offset += n_atoms

    atom_x = torch.as_tensor(np.vstack(atom_parts), dtype=torch.float32, device=device)
    batch_index = torch.as_tensor(np.concatenate(batch_index_parts), dtype=torch.long, device=device)
    if edge_parts:
        edge_index = torch.as_tensor(np.concatenate(edge_parts, axis=1), dtype=torch.long, device=device)
        edge_features = torch.as_tensor(np.vstack(edge_feature_parts), dtype=torch.float32, device=device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_features = torch.zeros((0, bond_feature_dim()), dtype=torch.float32, device=device)

    return atom_x, edge_index, edge_features, batch_index
