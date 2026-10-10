from __future__ import annotations
from pathlib import Path
import re
import json
import numpy as np
import pandas as pd


def norm_name(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub("\\s+", "", str(value).strip())


def split_names(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [norm_name(x) for x in re.split("[|,;，；、/]+", str(value)) if norm_name(x)]


def build_name_to_chp(d6: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for _, row in d6.iterrows():
        chp_id = str(row.get("CHP_ID", "")).strip()
        names = [norm_name(row.get("Chinese_herbal_pieces", ""))]
        names.extend(split_names(row.get("Chinese_synonyms", "")))
        for name in names:
            if name and name not in mapping:
                mapping[name] = chp_id
    return mapping


def build_knowledge_features(knowledge_dir, herb_vocabulary, output_dir):
    knowledge_dir = Path(knowledge_dir)
    names = json.loads(Path(herb_vocabulary).read_text(encoding="utf-8"))
    if len(names) != 811:
        raise ValueError("Expected the frozen 811-herb vocabulary")
    pieces = pd.read_csv(
        knowledge_dir / "D6_Chinese_herbal_pieces.tsv",
        sep="\t",
        dtype=str,
        on_bad_lines="skip",
    )
    encoder = pd.read_csv(knowledge_dir / "CHP_Encoder.tsv", sep="\t")
    relations = pd.read_csv(
        knowledge_dir / "D7_CHP_Medicinal_properties.tsv",
        sep="\t",
        dtype=str,
        on_bad_lines="skip",
    )
    properties = pd.read_csv(knowledge_dir / "property_vocab.csv", sep="\t")
    mapping = build_name_to_chp(pieces)
    columns = [c for c in encoder.columns if c != "CHP_ID"]
    if len(columns) != 90 or len(properties) != 23:
        raise ValueError("Expected 90 knowledge features and 23 properties")
    property_ids = {
        str(row["prop_name_en"]).strip(): int(row["prop_id"])
        for _, row in properties.iterrows()
    }
    vectors = {
        str(row["CHP_ID"]): row[columns].to_numpy(dtype=np.float32)
        for _, row in encoder.iterrows()
    }
    property_sets = {}
    for _, row in relations.iterrows():
        identifier = str(row.get("CHP_ID", "")).strip()
        value = property_ids.get(str(row.get("Medicinal_properties", "")).strip())
        if identifier and value is not None:
            property_sets.setdefault(identifier, set()).add(value)
    matrix = np.zeros((811, 90), np.float32)
    features = np.zeros((811, 23), np.float32)
    for index, name in enumerate(names):
        identifier = mapping.get(norm_name(name))
        if identifier in vectors:
            matrix[index] = vectors[identifier]
        for value in property_sets.get(identifier, []):
            if 0 <= value < 23:
                features[index, value] = 1
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite knowledge features")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "herb_features_90d.npy", matrix)
    np.save(destination / "herb_features_23d.npy", features)
    return matrix, features
