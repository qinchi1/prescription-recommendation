from __future__ import annotations
import torch
from torch import nn
from .candidate_encoding import SymptomEncoder, CandidateTokenEncoder


class CandidateSetReranker(nn.Module):
    def __init__(
        self,
        n_herbs: int = 811,
        symptom_dim: int = 390,
        hidden_dim: int = 128,
        herb_embedding_dim: int = 128,
        property_embedding_dim: int = 64,
        frequency_embedding_dim: int = 16,
        numeric_feature_dim: int = 9,
        num_heads: int = 4,
        num_set_layers: int = 3,
        dropout: float = 0.1,
        max_candidate_logit_residual: float = 0.5,
        enable_tcm_auxiliary_heads: bool = False,
        syndrome_dim: int = 50,
        treatment_dim: int = 61,
        bounded_candidate_residual: bool = True,
    ):
        super().__init__()
        self.symptom_encoder = SymptomEncoder(symptom_dim, hidden_dim, dropout)
        self.token_encoder = CandidateTokenEncoder(
            n_herbs=n_herbs,
            herb_embedding_dim=herb_embedding_dim,
            property_embedding_dim=property_embedding_dim,
            frequency_embedding_dim=frequency_embedding_dim,
            numeric_feature_dim=numeric_feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=num_set_layers)
        self.selection_head = nn.Linear(hidden_dim, 1)
        self.max_candidate_logit_residual = float(max_candidate_logit_residual)
        self.enable_tcm_auxiliary_heads = bool(enable_tcm_auxiliary_heads)
        self.bounded_candidate_residual = bool(bounded_candidate_residual)
        if self.enable_tcm_auxiliary_heads:
            self.syndrome_aux_head = nn.Linear(hidden_dim, int(syndrome_dim))
            self.treatment_aux_head = nn.Linear(hidden_dim, int(treatment_dim))
        nn.init.zeros_(self.selection_head.weight)
        nn.init.zeros_(self.selection_head.bias)

    def forward(self, batch):
        symptom_repr = self.symptom_encoder(batch["symptom"])
        tokens = self.token_encoder(
            symptom_repr,
            batch["candidate_ids"],
            batch["candidate_features"],
            batch["candidate_23d"],
            batch["target_23d"],
            batch["frequency_group_id"],
        )
        mask = ~batch["candidate_mask"].bool()
        encoded = self.set_encoder(tokens, src_key_padding_mask=mask)
        encoded = encoded.masked_fill(mask.unsqueeze(-1), 0.0)
        denom = batch["candidate_mask"].float().sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = encoded.sum(dim=1) / denom
        raw_selection_residual = self.selection_head(encoded).squeeze(-1)
        selection_residual = (
            self.max_candidate_logit_residual * torch.tanh(raw_selection_residual)
            if self.bounded_candidate_residual
            else raw_selection_residual
        )
        base_selection = batch["gat_logits"].float()
        output = {
            "selection_logits": base_selection + selection_residual,
            "selection_residual": selection_residual,
            "pooled_repr": pooled,
        }
        if self.enable_tcm_auxiliary_heads:
            output.update(
                {
                    "syndrome_logits": self.syndrome_aux_head(pooled),
                    "treatment_logits": self.treatment_aux_head(pooled),
                }
            )
        return output
