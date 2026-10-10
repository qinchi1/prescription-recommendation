from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F


def masked_soft_bce_per_sample(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    support_mask: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    mask = support_mask.to(student_logits.device, dtype=student_logits.dtype)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0).expand_as(student_logits)
    target = torch.sigmoid(teacher_logits.detach() / float(temperature))
    element = (
        F.binary_cross_entropy_with_logits(
            student_logits / float(temperature), target, reduction="none"
        )
        * float(temperature) ** 2
    )
    return (element * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def candidate_kl_per_sample(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    valid = candidate_mask.bool()
    fill = torch.full_like(student_logits, -10000.0)
    student = torch.where(valid, student_logits / float(temperature), fill)
    teacher = torch.where(valid, teacher_logits.detach() / float(temperature), fill)
    target = torch.softmax(teacher, dim=1)
    element = F.kl_div(torch.log_softmax(student, dim=1), target, reduction="none")
    return element.sum(dim=1) * float(temperature) ** 2


class CoverageAwareReliabilityGate(nn.Module):
    def __init__(self, input_dim: int = 5, hidden_dim: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        self.network[-1].bias.data.copy_(
            torch.tensor([math.log(math.expm1(2.0)), math.log(math.expm1(1.0))])
        )

    def forward(
        self,
        gate_features: torch.Tensor,
        coverage: torch.Tensor,
        has_supported_symptom: torch.Tensor,
    ) -> torch.Tensor:
        raw = F.softplus(self.network(gate_features)) + 1e-08
        tcm_constraint = coverage.clamp(0.0, 1.0) * has_supported_symptom.clamp(
            0.0, 1.0
        )
        raw_mhm = raw[:, 0]
        raw_tcm = raw[:, 1] * tcm_constraint
        denominator = (raw_mhm + raw_tcm).clamp_min(1e-08)
        tcm_weight = torch.where(
            tcm_constraint > 0, raw_tcm / denominator, torch.zeros_like(raw_tcm)
        )
        return torch.stack([1.0 - tcm_weight, tcm_weight], dim=1)


def make_gate_features(
    coverage: torch.Tensor,
    has_supported_symptom: torch.Tensor,
    syndrome_uncertainty: torch.Tensor,
    treatment_uncertainty: torch.Tensor,
    mhm_uncertainty: torch.Tensor,
    mhm_logits: torch.Tensor,
) -> torch.Tensor:
    sorted_logits = torch.sort(mhm_logits.detach(), dim=1, descending=True).values
    margin_confidence = torch.sigmoid(sorted_logits[:, 0] - sorted_logits[:, 1])
    dispersion_confidence = 1.0 / (
        1.0 + mhm_uncertainty.detach().mean(dim=1).clamp_min(0.0)
    )
    mhm_confidence = 0.5 * (margin_confidence + dispersion_confidence)
    return torch.stack(
        [
            coverage.detach().clamp(0.0, 1.0),
            has_supported_symptom.detach().clamp(0.0, 1.0),
            1.0 - syndrome_uncertainty.detach().clamp(0.0, 1.0),
            1.0 - treatment_uncertainty.detach().clamp(0.0, 1.0),
            mhm_confidence,
        ],
        dim=1,
    )
