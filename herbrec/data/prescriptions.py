from pathlib import Path
import numpy as np
from ..io import read_json


class PrescriptionDataset:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.symptoms = np.load(
            self.directory / "X_symptom_multihot.npy", mmap_mode="r"
        )
        self._labels = None
        self.symptom_names = read_json(self.directory / "symptom_vocab.json")
        self.herb_names = read_json(self.directory / "herb_vocab.json")
        self.splits = {
            name: np.load(self.directory / file).astype(np.int64)
            for name, file in [
                ("train", "train_idx.npy"),
                ("validation", "val_idx.npy"),
                ("test", "test_idx.npy"),
            ]
        }
        if (
            self.symptoms.shape != (len(self.symptoms), 390)
            or len(self.symptom_names) != 390
            or len(self.herb_names) != 811
        ):
            raise ValueError(
                "Prescription dimensions do not match the frozen vocabulary"
            )
        ids = np.concatenate(list(self.splits.values()))
        if (
            len(ids) != len(self.symptoms)
            or len(np.unique(ids)) != len(ids)
            or ids.min() != 0
            or ids.max() != len(ids) - 1
        ):
            raise ValueError(
                "Frozen split indices must be unique, disjoint and complete"
            )

    def labels(self, ids, allow_test=False):
        ids = np.asarray(ids, dtype=np.int64)
        if not allow_test and np.isin(ids, self.splits["test"]).any():
            raise ValueError("Test labels require an explicit evaluation request")
        if self._labels is None:
            self._labels = np.load(
                self.directory / "Y_herb_multihot.npy", mmap_mode="r"
            )
            if self._labels.shape != (len(self.symptoms), 811):
                raise ValueError("Prescription label shape mismatch")
        values = np.asarray(self._labels[ids], dtype=np.float32)
        if not np.isin(values, [0, 1]).all():
            raise ValueError("Prescription labels must be binary")
        return values

    def validate_rows(self, ids):
        values = np.asarray(self.symptoms[ids], dtype=np.float32)
        if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
            raise ValueError("Symptoms must be finite binary values")
        return values
