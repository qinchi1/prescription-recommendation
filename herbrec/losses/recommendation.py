from __future__ import annotations
import torch
import torch.nn.functional as F


def class_weighted_bce(logits, labels, mask=None, pos_weight_cap: float = 8.0):
    if F is None:
        raise RuntimeError("PyTorch is required.")
    labels = labels.float()
    if mask is None:
        mask = torch.ones_like(labels)
    pos = (labels * mask).sum().clamp_min(1.0)
    neg = ((1.0 - labels) * mask).sum().clamp_min(1.0)
    pos_weight = (neg / pos).clamp(max=pos_weight_cap)
    loss = F.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=pos_weight, reduction="none"
    )
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def listwise_positive_negative_loss(logits, labels, mask=None, margin: float = 0.2):
    if F is None:
        raise RuntimeError("PyTorch is required.")
    if mask is None:
        mask = torch.ones_like(labels)
    losses = []
    for row_logits, row_labels, row_mask in zip(logits, labels, mask):
        pos = row_logits[(row_labels > 0) & row_mask.bool()]
        neg = row_logits[(row_labels <= 0) & row_mask.bool()]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        losses.append(F.relu(margin - pos[:, None] + neg[None, :]).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def boundary_ranking_loss(logits, late_true_mask, high_false_mask, margin: float = 0.4):
    if F is None:
        raise RuntimeError("PyTorch is required.")
    losses = []
    for row, late, high_false in zip(
        logits, late_true_mask.bool(), high_false_mask.bool()
    ):
        pos = row[late]
        neg = row[high_false]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        losses.append(F.relu(margin - pos[:, None] + neg[None, :]).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()
