from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn

from .graph import atom_feature_dim, batch_graphs, bond_feature_dim
from .modules import GGNNCompoundEncoder, MLPEncoder, SharedProfileEncoder


MODALITY_TO_ID = {"compound": 0, "orf": 1, "crispr": 2}


@dataclass
class CGPAlignConfig:
    profile_dim: int = 3180
    protein_dim: int = 2560
    embed_dim: int = 256
    gnn_hidden_dim: int = 256
    gnn_layers: int = 6
    compound_projection_hidden_dims: Sequence[int] = field(default_factory=lambda: (512,))
    gene_hidden_dims: Sequence[int] = field(default_factory=lambda: (512,))
    profile_hidden_dims: Sequence[int] = field(default_factory=lambda: (512,))
    dropout: float = 0.2
    split_gene_modality_branches: bool = True
    profile_source_embedding: bool = True
    num_profile_sources: int = 3
    profile_input_layer_norm: bool = False
    profile_feature_dropout: float = 0.0
    profile_mlp_layer_norm: bool = False

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "CGPAlignConfig":
        return cls(**{key: value for key, value in values.items() if key in cls.__dataclass_fields__})


class CGPAlign(nn.Module):
    """Four-branch CGP-Align encoder.

    The model maps compound graphs, ORF protein embeddings, CRISPR protein
    embeddings and Cell Painting profile vectors into one L2-normalized latent
    space. The profile encoder is shared across compound, ORF and CRISPR profile
    inputs and acts as the phenotype anchor.
    """

    def __init__(self, config: CGPAlignConfig | None = None):
        super().__init__()
        self.config = config or CGPAlignConfig()
        cfg = self.config

        self.compound_encoder = GGNNCompoundEncoder(
            atom_dim=atom_feature_dim(),
            bond_dim=bond_feature_dim(),
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.gnn_hidden_dim,
            num_layers=cfg.gnn_layers,
            projection_hidden_dims=cfg.compound_projection_hidden_dims,
            dropout=cfg.dropout,
        )
        self.orf_gene_encoder = MLPEncoder(cfg.protein_dim, cfg.gene_hidden_dims, cfg.embed_dim, cfg.dropout)
        self.crispr_gene_encoder = MLPEncoder(cfg.protein_dim, cfg.gene_hidden_dims, cfg.embed_dim, cfg.dropout)
        self.profile_encoder = SharedProfileEncoder(
            profile_dim=cfg.profile_dim,
            hidden_dims=cfg.profile_hidden_dims,
            embed_dim=cfg.embed_dim,
            dropout=cfg.dropout,
            num_sources=cfg.num_profile_sources,
            use_source_embedding=cfg.profile_source_embedding,
            input_layer_norm=cfg.profile_input_layer_norm,
            feature_dropout=cfg.profile_feature_dropout,
            mlp_layer_norm=cfg.profile_mlp_layer_norm,
        )

    def encode_compound_graphs(self, graphs: Sequence[Any], device: torch.device | None = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        atom_x, edge_index, edge_features, batch_index = batch_graphs(graphs, device)
        return self.compound_encoder(atom_x, edge_index, edge_features, batch_index)

    def encode_gene(self, protein_x: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
        output = protein_x.new_empty((int(protein_x.shape[0]), int(self.config.embed_dim)))
        orf_mask = modality.long().eq(MODALITY_TO_ID["orf"])
        crispr_mask = modality.long().eq(MODALITY_TO_ID["crispr"])
        if bool(orf_mask.any()):
            output[orf_mask] = self.orf_gene_encoder(protein_x[orf_mask])
        if bool(crispr_mask.any()):
            output[crispr_mask] = self.crispr_gene_encoder(protein_x[crispr_mask])
        other_mask = ~(orf_mask | crispr_mask)
        if bool(other_mask.any()):
            output[other_mask] = self.orf_gene_encoder(protein_x[other_mask])
        return output

    def encode_profile(self, profile_x: torch.Tensor, source_id: torch.Tensor) -> torch.Tensor:
        return self.profile_encoder(profile_x, source_id)

    def forward(
        self,
        compound_graphs: Sequence[Any] | None = None,
        gene_protein_x: torch.Tensor | None = None,
        gene_modality: torch.Tensor | None = None,
        profile_x: torch.Tensor | None = None,
        profile_source_id: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        if compound_graphs is not None:
            outputs["compound"] = self.encode_compound_graphs(compound_graphs)
        if gene_protein_x is not None and gene_modality is not None:
            outputs["gene"] = self.encode_gene(gene_protein_x, gene_modality)
        if profile_x is not None and profile_source_id is not None:
            outputs["profile"] = self.encode_profile(profile_x, profile_source_id)
        return outputs
