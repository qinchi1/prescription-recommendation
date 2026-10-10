from __future__ import annotations
from typing import Any
import torch
from torch import nn


class SymptomEncoder(nn.Module):
    def __init__(
        self, input_dim: int, encoder_dim: int, hidden_dim: int, dropout: float
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, encoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ClinicalTeacherBackbone(nn.Module):
    def __init__(self, input_dim: int, herb_dim: int, config: dict[str, Any]):
        super().__init__()
        model = config["model"]
        hidden = int(model["hidden_dim"])
        intermediate = int(model["intermediate_dim"])
        syndrome_dim = int(model["syndrome_dim"])
        treatment_dim = int(model["treatment_dim"])
        dropout = float(model["dropout"])
        self.encoder = SymptomEncoder(
            input_dim, model["encoder_hidden_dim"], hidden, dropout
        )
        self.syndrome_head = nn.Linear(hidden, syndrome_dim)
        self.syndrome_representation = nn.Sequential(
            nn.Linear(syndrome_dim, intermediate), nn.GELU()
        )
        self.treatment_hidden = nn.Sequential(
            nn.Linear(hidden + intermediate, hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.treatment_head = nn.Linear(hidden, treatment_dim)
        self.treatment_representation = nn.Sequential(
            nn.Linear(treatment_dim, intermediate), nn.GELU()
        )
        self.herb_head = nn.Sequential(
            nn.Linear(hidden + 2 * intermediate, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, herb_dim),
        )

    def _forward_common(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(x)
        syndrome_logits = self.syndrome_head(hidden)
        syndrome_rep = self.syndrome_representation(torch.sigmoid(syndrome_logits))
        treatment_hidden = self.treatment_hidden(
            torch.cat([hidden, syndrome_rep], dim=-1)
        )
        treatment_logits = self.treatment_head(treatment_hidden)
        treatment_rep = self.treatment_representation(torch.sigmoid(treatment_logits))
        progressive_embedding = torch.cat([hidden, syndrome_rep, treatment_rep], dim=-1)
        return {
            "herb_logits": self.herb_head(progressive_embedding),
            "syndrome_logits": syndrome_logits,
            "treatment_logits": treatment_logits,
            "patient_embedding": progressive_embedding,
            "direct_patient_embedding": hidden,
        }


class ClinicalTeacher(ClinicalTeacherBackbone):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._forward_common(x)
