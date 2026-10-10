import numpy as np


def ranking_metrics(scores, candidate_ids, labels):
    if (
        scores.shape != candidate_ids.shape
        or scores.shape[1] != 50
        or labels.shape != (len(scores), 811)
    ):
        raise ValueError(
            "Ranking evaluation requires [N,50] predictions and [N,811] labels"
        )
    if any(len(np.unique(row)) != 50 for row in candidate_ids):
        raise ValueError("Duplicate candidate IDs")
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.take_along_axis(candidate_ids, order, axis=1)
    truth = labels > 0
    result = {}
    for k in (5, 10, 20):
        hit = int(np.take_along_axis(truth, ranks[:, :k], axis=1).sum())
        precision = hit / max(len(scores) * k, 1)
        recall = hit / max(int(truth.sum()), 1)
        result.update(
            {
                f"P@{k}": precision,
                f"R@{k}": recall,
                f"F1@{k}": 2 * precision * recall / max(precision + recall, 1e-12),
            }
        )
    result["CandidateRecall@50"] = float(
        np.take_along_axis(truth, candidate_ids, axis=1).sum()
        / max(int(truth.sum()), 1)
    )
    return result


def coverage_metrics(scores, candidate_ids, labels, coverage):
    groups = {
        "Zero": coverage == 0,
        "Low": (coverage > 0) & (coverage <= 0.25),
        "Medium": (coverage > 0.25) & (coverage <= 0.5),
        "High": coverage > 0.5,
        "All": np.ones(len(coverage), bool),
    }
    return {
        name: {
            "n": int(mask.sum()),
            "fraction": float(mask.mean()),
            "metrics": ranking_metrics(scores[mask], candidate_ids[mask], labels[mask])
            if mask.any()
            else None,
        }
        for name, mask in groups.items()
    }


def mean_case_f1(scores, labels, k=10):
    ids = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    hits = np.take_along_axis(labels > 0, ids, axis=1).sum(1)
    precision = hits / k
    recall = hits / np.maximum((labels > 0).sum(1), 1)
    return float(
        (2 * precision * recall / np.maximum(precision + recall, 1e-12)).mean()
    )
