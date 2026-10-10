from __future__ import annotations
import torch
from torch import nn

N_SYMPTOMS = 390
N_HERBS = 811
PAD_TOKEN_ID = 811
EMPTY_SET_TOKEN_ID = 812


class SymptomEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_SYMPTOMS, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, symptoms: torch.Tensor) -> torch.Tensor:
        return self.net(symptoms.float())


class CandidateScorer(nn.Module):
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        context: torch.Tensor,
        symptom_repr: torch.Tensor,
        candidate_emb: torch.Tensor,
    ) -> torch.Tensor:
        batch = context.shape[0]
        cand = candidate_emb.unsqueeze(0).expand(batch, -1, -1)
        ctx = context.unsqueeze(1).expand(-1, cand.shape[1], -1)
        sym = symptom_repr.unsqueeze(1).expand(-1, cand.shape[1], -1)
        feats = torch.cat([ctx, sym, cand, ctx * cand], dim=-1)
        return self.net(feats).squeeze(-1)


class MaskedHerbTeacher(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_set_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if embedding_dim != hidden_dim:
            raise ValueError(
                "This lightweight implementation expects embedding_dim == hidden_dim."
            )
        self.symptom_encoder = SymptomEncoder(hidden_dim, dropout)
        self.herb_embedding = nn.Embedding(
            N_HERBS + 2, hidden_dim, padding_idx=PAD_TOKEN_ID
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
        self.pool_query = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.scorer = CandidateScorer(hidden_dim, dropout)

    def forward(
        self,
        symptoms: torch.Tensor,
        visible_herbs: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> torch.Tensor:
        symptom_repr = self.symptom_encoder(symptoms)
        herb_tokens = self.herb_embedding(visible_herbs.long())
        symptom_token = symptom_repr.unsqueeze(1)
        tokens = torch.cat([symptom_token, herb_tokens], dim=1)
        symptom_pad = torch.zeros(
            (visible_mask.shape[0], 1), dtype=torch.bool, device=visible_mask.device
        )
        key_padding_mask = torch.cat([symptom_pad, visible_mask.bool()], dim=1)
        encoded = self.set_encoder(tokens, src_key_padding_mask=key_padding_mask)
        attn_logits = torch.matmul(encoded, self.pool_query)
        attn_logits = attn_logits.masked_fill(key_padding_mask, -10000.0)
        attn = torch.softmax(attn_logits, dim=1)
        context = torch.sum(encoded * attn.unsqueeze(-1), dim=1)
        context = self.context_norm(context)
        candidate_emb = self.herb_embedding.weight[:N_HERBS]
        return self.scorer(context, symptom_repr, candidate_emb)


def mask_visible_logits(
    logits: torch.Tensor,
    visible_herbs: torch.Tensor,
    visible_mask: torch.Tensor,
    fill_value: float = -10000.0,
) -> torch.Tensor:
    out = logits.clone()
    for i in range(out.shape[0]):
        valid = visible_herbs[i][~visible_mask[i].bool()]
        valid = valid[(valid >= 0) & (valid < N_HERBS)]
        if valid.numel() > 0:
            out[i, valid.long()] = fill_value
    return out
