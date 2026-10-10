import numpy as np
from ..io import read_json, save_arrays


def frequency_groups(counts):
    counts = np.asarray(counts, dtype=np.int64)
    order = np.argsort(-counts, kind="stable")
    cumulative = 0.0
    total = max(float(counts.sum()), 1.0)
    groups = np.empty(len(counts), dtype="U8")
    for index in order:
        cumulative += float(counts[index])
        fraction = cumulative / total
        groups[index] = (
            "Head"
            if fraction <= 0.70
            else "Middle"
            if fraction <= 0.90
            else "Tail"
            if fraction <= 0.98
            else "Rare"
        )
    return groups


def build_feature_assets(paths, dataset):
    merged = (
        paths.output / "candidates/oof"
        if (paths.output / "candidates/oof/sample_ids.npy").exists()
        else paths.oof / "merged"
    )
    ids = np.load(merged / "sample_ids.npy")
    folds = np.load(merged / "fold_ids.npy")
    if not np.array_equal(ids, dataset.splits["train"]):
        raise ValueError("Feature-statistics fold IDs are not aligned")
    fit = np.sort(ids[folds != 0])
    x = dataset.validate_rows(fit)
    y = dataset.labels(fit)
    features = paths.output / "knowledge_features/herb_features_23d.npy"
    if not features.exists():
        features = paths.features / "herb_features_23d.npy"
    herb_23d = np.load(features)
    mapping = read_json(paths.knowledge / "syndrome_to_23D_property_mapping.json")
    coverage = read_json(paths.knowledge / "syndrome_coverage_report.json")
    symptom_vectors = {}
    for syndrome, symptoms in coverage.get("syndrome_to_symptoms", {}).items():
        item = mapping.get(syndrome)
        if not isinstance(item, dict) or "target_vector" not in item:
            continue
        vector = np.asarray(item["target_vector"], dtype=np.float32)
        if vector.shape != (23,):
            raise ValueError("Invalid syndrome knowledge vector")
        for name in symptoms:
            symptom_vectors.setdefault(str(name).strip(), []).append(vector)
    matrix = np.zeros((390, 23), np.float32)
    links = np.zeros(390, np.int16)
    for index, name in enumerate(dataset.symptom_names):
        values = symptom_vectors.get(str(name).strip(), [])
        if values:
            matrix[index] = np.mean(values, axis=0)
            links[index] = len(values)
    counts = y.sum(0)
    groups = frequency_groups(counts)
    group_ids = np.asarray(
        [{"Head": 0, "Middle": 1, "Tail": 2, "Rare": 3}[name] for name in groups],
        dtype=np.int64,
    )
    frozen_file = paths.student_data / "static_feature_assets.npz"
    if frozen_file.exists():
        with np.load(frozen_file, allow_pickle=False) as frozen:
            if np.array_equal(fit, frozen["fit_sample_ids"]) and np.allclose(
                (counts + 1) / (len(fit) + 2), frozen["herb_freq"]
            ):
                group_ids = frozen["frequency_group_id"].astype(np.int64)
    if group_ids.shape != (811,) or ((group_ids < 0) | (group_ids > 3)).any():
        raise ValueError("Invalid frozen frequency-group IDs")
    assets = {
        "fit_sample_ids": fit,
        "herb_23d": herb_23d.astype(np.float32),
        "symptom_23d": matrix,
        "symptom_23d_link_count": links,
        "herb_freq": ((counts + 1) / (len(fit) + 2)).astype(np.float32),
        "symptom_herb": ((x.T @ y + 1) / (x.sum(0)[:, None] + 2)).astype(np.float32),
        "frequency_group_id": group_ids,
    }
    save_arrays(paths.output / "feature_assets/static_feature_assets.npz", assets)
    return assets


def load_feature_assets(paths, dataset=None):
    from ..io import load_arrays

    file = paths.output / "feature_assets/static_feature_assets.npz"
    if not file.exists():
        file = paths.student_data / "static_feature_assets.npz"
    if file.exists():
        return load_arrays(file)
    if dataset is None:
        raise FileNotFoundError(file)
    return build_feature_assets(paths, dataset)
