from __future__ import annotations
from pathlib import Path
from itertools import combinations
import numpy as np


def build_symptom_herb_edges(
    x: np.ndarray, y: np.ndarray, train_idx: np.ndarray
) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for sample_id in train_idx:
        symptoms = np.flatnonzero(x[sample_id] > 0)
        herbs = np.flatnonzero(y[sample_id] > 0)
        for s in symptoms:
            for h in herbs:
                edges.add((int(s), int(h)))
    return (
        np.array(sorted(edges), dtype=np.int64).T
        if edges
        else np.zeros((2, 0), dtype=np.int64)
    )


def build_herb_herb_edges(
    y: np.ndarray, train_idx: np.ndarray, min_count: int = 2
) -> np.ndarray:
    counts: dict[tuple[int, int], int] = {}
    for sample_id in train_idx:
        herbs = sorted(np.flatnonzero(y[sample_id] > 0).astype(int).tolist())
        for a, b in combinations(herbs, 2):
            counts[a, b] = counts.get((a, b), 0) + 1
    edges = [(a, b) for (a, b), c in counts.items() if c >= min_count]
    return (
        np.array(sorted(edges), dtype=np.int64).T
        if edges
        else np.zeros((2, 0), dtype=np.int64)
    )


def build_herb_property_edges(feature_dir: Path, num_herbs: int) -> np.ndarray:
    path = feature_dir / "herb_features_23d.npy"
    if not path.exists():
        return np.zeros((2, 0), dtype=np.int64)
    feat = np.load(path)
    if feat.shape[0] != num_herbs:
        raise ValueError(
            f"{path} first dimension {feat.shape[0]} != num_herbs {num_herbs}"
        )
    rows, cols = np.nonzero(feat > 0)
    return (
        np.vstack([rows.astype(np.int64), cols.astype(np.int64)])
        if len(rows)
        else np.zeros((2, 0), dtype=np.int64)
    )


def to_global_symptom_herb(edge: np.ndarray, num_symptoms: int) -> np.ndarray:
    out = edge.copy()
    out[1] += num_symptoms
    return out


def to_global_herb_herb(edge: np.ndarray, num_symptoms: int) -> np.ndarray:
    out = edge.copy()
    out += num_symptoms
    return out


def to_global_herb_property(
    edge: np.ndarray, num_symptoms: int, num_herbs: int
) -> np.ndarray:
    out = edge.copy()
    out[0] += num_symptoms
    out[1] += num_symptoms + num_herbs
    return out
