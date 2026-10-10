import numpy as np
from ..io import load_arrays, read_json, sha256, save_arrays, save_json


INPUT_FIELDS = (
    "sample_ids",
    "symptom",
    "candidate_ids",
    "candidate_mask",
    "candidate_features",
    "candidate_23d",
    "target_23d",
    "frequency_group_id",
    "gat_logits",
)
TRAIN_FIELDS = INPUT_FIELDS + ("hard_labels", "late_true_mask", "high_false_mask")


def validate_candidates(data):
    n = len(data["sample_ids"])
    if len(np.unique(data["sample_ids"])) != n:
        raise ValueError("Duplicate sample IDs")
    expected = {
        "symptom": (n, 390),
        "candidate_ids": (n, 50),
        "candidate_mask": (n, 50),
        "candidate_features": (n, 50, 9),
        "candidate_23d": (n, 50, 23),
        "target_23d": (n, 23),
        "frequency_group_id": (n, 50),
        "gat_logits": (n, 50),
    }
    for key, shape in expected.items():
        value = data[key]
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{key}: invalid shape or nonfinite values")
    ids = data["candidate_ids"]
    if ids.min() < 0 or ids.max() >= 811 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Invalid candidate IDs")
    if any(len(np.unique(row)) != 50 for row in ids):
        raise ValueError("Candidate IDs must be unique per case")
    if not np.all(data["candidate_mask"] == 1):
        raise ValueError("The model requires a complete Top-50 candidate set")


def normalize_logits(logits):
    return (
        (logits - logits.mean(1, keepdims=True))
        / np.maximum(logits.std(1, keepdims=True), 1e-6)
    ).astype(np.float32)


def build_candidate_inputs(sample_ids, symptoms, full_logits, assets):
    ids = np.argsort(-full_logits, axis=1, kind="stable")[:, :50]
    logits = np.take_along_axis(full_logits, ids, axis=1).astype(np.float32)
    props = assets["herb_23d"][ids].astype(np.float32)
    target = np.zeros((len(ids), 23), np.float32)
    cooccurrence = np.zeros((len(ids), 50), np.float32)
    alignment = np.zeros_like(cooccurrence)
    for row in range(len(ids)):
        active = np.flatnonzero(symptoms[row] > 0)
        if len(active):
            values = assets["symptom_23d"][active]
            values = values[np.linalg.norm(values, axis=1) > 1e-8]
            if len(values):
                target[row] = values.mean(0)
            cooccurrence[row] = assets["symptom_herb"][np.ix_(active, ids[row])].mean(0)
        denom = np.linalg.norm(props[row], axis=1) * np.linalg.norm(target[row])
        alignment[row] = (props[row] @ target[row]) / np.maximum(denom, 1e-8)
    available = (
        (np.linalg.norm(props, axis=2) > 1e-8)
        & (np.linalg.norm(target, axis=1, keepdims=True) > 1e-8)
    ).astype(np.float32)
    frequency = assets["herb_freq"][ids].astype(np.float32)
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
    ranks = np.broadcast_to(np.arange(1, 51, dtype=np.float32), ids.shape)
    numeric = np.stack(
        [
            probabilities,
            logits,
            ranks / 50,
            frequency,
            np.log1p(frequency * len(assets["fit_sample_ids"])),
            alignment,
            available,
            cooccurrence,
            logits.max(1, keepdims=True) - logits,
        ],
        axis=2,
    ).astype(np.float32)
    data = {
        "sample_ids": np.asarray(sample_ids, dtype=np.int64),
        "symptom": np.asarray(symptoms, dtype=np.float32),
        "candidate_ids": ids.astype(np.int64),
        "candidate_mask": np.ones(ids.shape, np.float32),
        "candidate_features": numeric,
        "candidate_23d": props,
        "target_23d": target,
        "frequency_group_id": assets["frequency_group_id"][ids].astype(np.int64),
        "gat_logits": normalize_logits(logits),
    }
    validate_candidates(data)
    return data


def load_inference_inputs(paths, split, limit=None):
    preferred = paths.output / "reranking_inputs" / f"{split}.npz"
    original = paths.inference_data / f"{split}_student_inputs.npz"
    data = load_arrays(preferred if preferred.exists() else original, INPUT_FIELDS)
    if limit is not None:
        data = {key: value[:limit] for key, value in data.items()}
    validate_candidates(data)
    return data


def load_training_inputs(paths, dataset, limit=None):
    prepared = paths.output / "reranking_inputs/train.npz"
    data = load_arrays(
        prepared
        if prepared.exists()
        else paths.student_data / "student_training_data.npz",
        TRAIN_FIELDS,
    )
    if not np.array_equal(data["sample_ids"], dataset.splits["train"]):
        raise ValueError("Training sample IDs differ from frozen training split")
    validate_candidates(data)
    if limit is not None:
        data = {key: value[:limit] for key, value in data.items()}
    mhm_path = paths.output / "teacher_signals/masked_herb.npz"
    mhm_manifest = paths.output / "teacher_signals/masked_herb_manifest.json"
    if not mhm_path.exists():
        mhm_path = paths.mhm / "oof_mhm_teacher.npz"
        mhm_manifest = paths.mhm / "oof_manifest.json"
    manifest = read_json(mhm_manifest)
    if (
        not manifest.get("strict_oof")
        or manifest.get("validation_used")
        or manifest.get("test_used")
        or manifest.get("artifact_sha256") != sha256(mhm_path)
    ):
        raise ValueError("MHM OOF provenance verification failed")
    mhm = load_arrays(mhm_path)
    clinical_path = paths.output / "teacher_signals/clinical_train.npz"
    if not clinical_path.exists():
        clinical_path = paths.clinical_features / "ptm_train_tcm_lung_features.npz"
    clinical = load_arrays(
        clinical_path,
        [
            "sample_ids",
            "syndrome_logits",
            "treatment_logits",
            "symptom_support_coverage",
            "has_supported_symptom",
            "syndrome_uncertainty",
            "treatment_uncertainty",
        ],
    )
    count = len(data["sample_ids"])
    if not np.array_equal(
        data["sample_ids"], mhm["sample_ids"][:count]
    ) or not np.array_equal(data["candidate_ids"], mhm["candidate_ids"][:count]):
        raise ValueError("MHM candidate/sample alignment failed")
    if not np.array_equal(data["sample_ids"], clinical["sample_ids"][:count]):
        raise ValueError("Clinical teacher sample alignment failed")
    if not np.all(mhm["available_mask"][:count] == 1):
        raise ValueError("Formal MHM cache must cover all candidate positions")
    coverage = clinical["symptom_support_coverage"][:count]
    if (
        not np.isfinite(coverage).all()
        or ((coverage < 0) | (coverage > 1)).any()
        or not np.array_equal(
            clinical["has_supported_symptom"][:count] > 0, coverage > 0
        )
    ):
        raise ValueError("Inconsistent clinical coverage/support fields")
    data.update(
        {
            key: mhm[key][:count].astype(np.float32)
            for key in ["mhm_logits", "mhm_uncertainty", "available_mask"]
        }
    )
    data.update(
        {
            f"clinical_{key}": value[:count].astype(np.float32)
            for key, value in clinical.items()
            if key != "sample_ids"
        }
    )
    return data


def prepare_student_inputs(paths, dataset, split, full_logits=None):
    from .statistics import load_feature_assets

    assets = load_feature_assets(paths, dataset)
    if not np.isin(assets["fit_sample_ids"], dataset.splits["train"]).all():
        raise ValueError("Static features include non-training samples")
    ids = dataset.splits[split]
    if full_logits is None:
        generated = paths.output / "candidates" / f"{split}.npz"
        if split != "train" and generated.exists():
            predicted = load_arrays(generated, ["sample_ids", "full_logits"])
            if not np.array_equal(ids, predicted["sample_ids"]):
                raise ValueError(
                    "Generated candidate IDs differ from the complete frozen split"
                )
            full_logits = predicted["full_logits"]
        elif split == "train":
            merged = (
                paths.output / "candidates/oof"
                if (paths.output / "candidates/oof/sample_ids.npy").exists()
                else paths.oof / "merged"
            )
            canonical = np.load(merged / "sample_ids.npy")
            if not np.array_equal(ids, canonical):
                raise ValueError("OOF IDs do not match train split")
            full_logits = np.load(merged / "gat_oof_logits.npy")
        elif split == "validation":
            full_logits = np.load(
                paths.candidate_artifacts / "validation/gat_validation_logits.npy"
            )
        else:
            raise ValueError(
                "Test inputs must be generated by frozen candidate-model inference"
            )
    data = build_candidate_inputs(ids, dataset.validate_rows(ids), full_logits, assets)
    if split == "train":
        labels = dataset.labels(ids)
        hard = np.take_along_axis(labels, data["candidate_ids"], axis=1)
        ranks = np.arange(1, 51)[None, :]
        lengths = labels.sum(1, keepdims=True)
        data.update(
            {
                "hard_labels": hard,
                "late_true_mask": ((hard > 0) & (ranks > lengths)).astype(np.int8),
                "high_false_mask": ((hard == 0) & (ranks <= lengths)).astype(np.int8),
            }
        )
    save_arrays(paths.output / "reranking_inputs" / f"{split}.npz", data)
    save_json(
        paths.output / "reranking_inputs" / f"{split}_manifest.json",
        {
            "split": split,
            "sample_n": len(ids),
            "labels_loaded": split == "train",
            "static_fit_ids": assets["fit_sample_ids"].tolist(),
        },
    )
    return data
