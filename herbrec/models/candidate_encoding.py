from __future__ import annotations
import torch
from torch import nn


class SymptomEncoder(nn.Module):
    def __init__(
        self, symptom_dim: int = 390, hidden_dim: int = 128, dropout: float = 0.1
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(symptom_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x.float())


class CandidateTokenEncoder(nn.Module):
    def __init__(
        self,
        n_herbs: int = 811,
        herb_embedding_dim: int = 128,
        property_embedding_dim: int = 64,
        frequency_embedding_dim: int = 16,
        numeric_feature_dim: int = 9,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.herb_embedding = nn.Embedding(n_herbs, herb_embedding_dim)
        self.property_encoder = nn.Sequential(
            nn.Linear(23, property_embedding_dim),
            nn.GELU(),
            nn.LayerNorm(property_embedding_dim),
        )
        self.group_embedding = nn.Embedding(4, frequency_embedding_dim)
        in_dim = (
            herb_embedding_dim
            + property_embedding_dim
            + frequency_embedding_dim
            + numeric_feature_dim
            + hidden_dim
            + property_embedding_dim
        )
        self.target_property_encoder = nn.Sequential(
            nn.Linear(23, property_embedding_dim),
            nn.GELU(),
            nn.LayerNorm(property_embedding_dim),
        )
        self.token = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self, symptom_repr, herb_ids, numeric_features, herb_23d, target_23d, group_ids
    ):
        herb = self.herb_embedding(herb_ids.long())
        prop = self.property_encoder(herb_23d.float())
        prop_available = (
            herb_23d.float().abs().sum(dim=-1, keepdim=True) > 1e-08
        ).float()
        prop = prop * prop_available
        target_available = (
            target_23d.float().abs().sum(dim=-1, keepdim=True) > 1e-08
        ).float()
        target = self.target_property_encoder(target_23d.float()) * target_available
        target = target.unsqueeze(1).expand(-1, herb_ids.shape[1], -1)
        group = self.group_embedding(group_ids.long().clamp(min=0, max=3))
        sym = symptom_repr.unsqueeze(1).expand(-1, herb_ids.shape[1], -1)
        return self.token(
            torch.cat(
                [herb, prop, group, numeric_features.float(), sym, target], dim=-1
            )
        )
