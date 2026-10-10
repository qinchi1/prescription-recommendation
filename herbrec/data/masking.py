from __future__ import annotations
import math
from typing import Any
import numpy as np


def partial_contexts(
    true_herbs: list[int],
    candidates: np.ndarray,
    ranks: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> list[dict[str, Any]]:
    true = sorted(set((int(h) for h in true_herbs)))
    if not true:
        return [{"visible": [], "masked": [], "strategy": "empty"}]
    rng = np.random.default_rng(int(seed))
    rank_map = {
        int(h): int(r) for h, r in zip(candidates.astype(int), ranks.astype(int))
    }
    contexts: list[dict[str, Any]] = []
    random_drop_n = max(1, int(round(len(true) * 0.35)))
    random_drop = set(
        rng.choice(np.asarray(true), size=min(random_drop_n, len(true)), replace=False)
        .astype(int)
        .tolist()
    )
    contexts.append({"strategy": "random_mask", "masked": sorted(random_drop)})
    late = sorted(true, key=lambda h: rank_map.get(h, 10000), reverse=True)
    contexts.append(
        {
            "strategy": "late_rank_mask",
            "masked": late[: max(1, min(len(late), math.ceil(len(true) * 0.25)))],
        }
    )
    tail = [h for h in true if str(groups[h]) in {"Tail", "Rare"}]
    if not tail:
        tail = late
    contexts.append(
        {
            "strategy": "tail_rare_mask",
            "masked": tail[: max(1, min(len(tail), math.ceil(len(true) * 0.25)))],
        }
    )
    multi_n = max(1, int(round(len(true) * float(rng.uniform(0.3, 0.5)))))
    multi = set(
        rng.choice(np.asarray(true), size=min(multi_n, len(true)), replace=False)
        .astype(int)
        .tolist()
    )
    contexts.append({"strategy": "multi_mask", "masked": sorted(multi)})
    covered = set((int(h) for context in contexts for h in context["masked"]))
    for offset, herb in enumerate((h for h in true if h not in covered)):
        contexts[offset % len(contexts)]["masked"].append(int(herb))
    for context in contexts:
        context["masked"] = sorted(set((int(h) for h in context["masked"])))
        masked = set(context["masked"])
        context["visible"] = [h for h in true if h not in masked]
    return contexts


def pad_herb_sets(formulas: list[list[int]], pad_id: int = 811, empty_id: int = 812):
    import torch

    width = max(1, max((len(row) for row in formulas), default=1))
    ids = torch.full((len(formulas), width), int(pad_id), dtype=torch.long)
    mask = torch.ones((len(formulas), width), dtype=torch.bool)
    lengths = torch.zeros(len(formulas), dtype=torch.float32)
    for i, formula in enumerate(formulas):
        if formula:
            ids[i, : len(formula)] = torch.tensor(formula, dtype=torch.long)
            mask[i, : len(formula)] = False
            lengths[i] = float(len(formula))
        else:
            ids[i, 0] = int(empty_id)
            mask[i, 0] = False
    return (ids, mask, lengths)


def collate_mhm(
    rows: list[dict[str, Any]], positive_weights: np.ndarray, cfg: dict[str, Any]
) -> dict[str, Any]:
    import torch

    visible, visible_mask, _ = pad_herb_sets([row["visible"] for row in rows])
    labels = torch.zeros((len(rows), 811), dtype=torch.float32)
    weights = torch.full_like(labels, float(cfg.get("negative_weight", 0.05)))
    positives = []
    negatives = []
    for i, row in enumerate(rows):
        if row["visible"]:
            weights[i, torch.tensor(row["visible"], dtype=torch.long)] = 0.0
        pos = [int(h) for h in row["masked"]]
        labels[i, torch.tensor(pos, dtype=torch.long)] = 1.0
        weights[i, torch.tensor(pos, dtype=torch.long)] = torch.tensor(
            positive_weights[pos], dtype=torch.float32
        )
        positives.append(pos)
        negatives.append([int(h) for h in row["hard_negatives"]])
    return {
        "symptoms": torch.tensor(
            np.vstack([row["symptoms"] for row in rows]), dtype=torch.float32
        ),
        "visible": visible,
        "visible_mask": visible_mask,
        "labels": labels,
        "weights": weights,
        "positives": positives,
        "negatives": negatives,
    }


def pair_ranking_loss(
    logits, positives: list[list[int]], negatives: list[list[int]], margin: float
):
    import torch

    losses = []
    for i, (pos, neg) in enumerate(zip(positives, negatives)):
        if not pos or not neg:
            continue
        p = logits[i, torch.tensor(pos, dtype=torch.long, device=logits.device)].mean()
        n = logits[i, torch.tensor(neg, dtype=torch.long, device=logits.device)]
        losses.append(torch.relu(float(margin) - p + n).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def evaluate_mhm(model, loader, device) -> dict[str, float]:
    import torch

    model.eval()
    recalls = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(
                batch["symptoms"].to(device),
                batch["visible"].to(device),
                batch["visible_mask"].to(device),
            )
            for i, visible in enumerate(batch["visible"]):
                valid = visible[~batch["visible_mask"][i]].long().to(device)
                valid = valid[(valid >= 0) & (valid < logits.shape[1])]
                if valid.numel() > 0:
                    logits[i, valid] = -10000.0
            top10 = torch.topk(logits, 10, dim=1).indices.cpu().numpy()
            for i, positives in enumerate(batch["positives"]):
                recalls.append(
                    len(set(positives) & set(top10[i].tolist()))
                    / max(len(positives), 1)
                )
    return {
        "masked_recall_at_10": float(np.mean(recalls)) if recalls else 0.0,
        "context_n": len(recalls),
    }
