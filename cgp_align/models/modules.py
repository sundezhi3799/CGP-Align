from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        embed_dim: int,
        dropout: float,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        previous_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim)
            layers.append(nn.Linear(previous_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.extend([nn.GELU(), nn.Dropout(float(dropout))])
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, int(embed_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class SharedProfileEncoder(nn.Module):
    def __init__(
        self,
        profile_dim: int,
        hidden_dims: Sequence[int],
        embed_dim: int,
        dropout: float,
        num_sources: int = 3,
        use_source_embedding: bool = True,
        input_layer_norm: bool = False,
        feature_dropout: float = 0.0,
        mlp_layer_norm: bool = False,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(int(profile_dim)) if input_layer_norm else None
        self.feature_dropout = nn.Dropout(float(feature_dropout)) if float(feature_dropout) > 0 else None
        self.source_embedding = nn.Embedding(int(num_sources), int(profile_dim)) if use_source_embedding else None
        self.encoder = MLPEncoder(int(profile_dim), hidden_dims, int(embed_dim), float(dropout), use_layer_norm=mlp_layer_norm)

    def forward(self, profile_x: torch.Tensor, source_id: torch.Tensor) -> torch.Tensor:
        h = profile_x
        if self.input_norm is not None:
            h = self.input_norm(h)
        if self.feature_dropout is not None:
            h = self.feature_dropout(h)
        if self.source_embedding is not None:
            h = h + self.source_embedding(source_id.long())
        return self.encoder(h)


class GGNNCompoundEncoder(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 6,
        projection_hidden_dims: Sequence[int] = (512,),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.atom_input = nn.Linear(int(atom_dim), int(hidden_dim))
        self.bond_input = nn.Linear(int(bond_dim), int(hidden_dim))
        self.message_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.update_layers = nn.ModuleList([nn.GRUCell(int(hidden_dim), int(hidden_dim)) for _ in range(int(num_layers))])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(int(hidden_dim)) for _ in range(int(num_layers))])
        self.projection = MLPEncoder(int(hidden_dim), list(projection_hidden_dims), int(embed_dim), float(dropout))

    def forward(
        self,
        atom_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        h = F.gelu(self.atom_input(atom_x))
        if edge_features.numel() > 0:
            bond_h = F.gelu(self.bond_input(edge_features))
        else:
            bond_h = edge_features.new_zeros((0, h.shape[-1]))

        for message_layer, update_layer, norm_layer in zip(self.message_layers, self.update_layers, self.norm_layers):
            aggregated = torch.zeros_like(h)
            if edge_index.numel() > 0:
                src = edge_index[0]
                dst = edge_index[1]
                messages = message_layer(torch.cat([h[src], bond_h], dim=-1))
                aggregated.index_add_(0, dst, messages)
            h = update_layer(aggregated, h)
            h = norm_layer(h)

        n_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
        pooled = h.new_zeros((n_graphs, h.shape[-1]))
        pooled.index_add_(0, batch_index, h)
        counts = torch.bincount(batch_index, minlength=n_graphs).to(h.dtype).clamp_min(1.0).unsqueeze(-1)
        pooled = pooled / counts
        return self.projection(pooled)
