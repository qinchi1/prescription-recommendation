from pathlib import Path
import numpy as np
import torch
from .models.knowledge_graph import KnowledgeGraphRecommender
from .models.reranker import CandidateSetReranker
from .teachers.masked_herb import MaskedHerbTeacher
from .teachers.clinical import ClinicalTeacher
from .io import read_json, sha256, resolve_device


def build_candidate_model(config):
    kwargs = {
        key: config[key]
        for key in [
            "hidden_dim",
            "num_layers",
            "num_heads",
            "dropout",
            "use_jpa",
            "use_23d",
            "use_kg",
        ]
        if key in config
    }
    kwargs["use_bge"] = bool(config.get("use_bge", False))
    kwargs["bge_symptom_dim"] = 768 if kwargs["use_bge"] else None
    kwargs["backend"] = "pyg"
    return KnowledgeGraphRecommender(**kwargs)


def build_reranker(config):
    values = config["student"]
    return CandidateSetReranker(
        hidden_dim=values["hidden_dim"],
        herb_embedding_dim=values["herb_embedding_dim"],
        property_embedding_dim=values["property_embedding_dim"],
        frequency_embedding_dim=values["frequency_embedding_dim"],
        numeric_feature_dim=9,
        num_heads=values["num_heads"],
        num_set_layers=values["num_set_layers"],
        dropout=values["dropout"],
        max_candidate_logit_residual=values["max_candidate_logit_residual"],
        enable_tcm_auxiliary_heads=True,
        syndrome_dim=50,
        treatment_dim=61,
        bounded_candidate_residual=True,
    )


def load_knowledge_tensors(paths, config, device):
    feature_dir = (
        paths.output / "knowledge_features"
        if (paths.output / "knowledge_features/herb_features_90d.npy").exists()
        else paths.features
    )
    jpa = np.load(feature_dir / "herb_features_90d.npy", allow_pickle=False)
    props = np.load(feature_dir / "herb_features_23d.npy", allow_pickle=False)
    if (
        jpa.shape != (811, 90)
        or props.shape != (811, 23)
        or not np.isfinite(jpa).all()
        or not np.isfinite(props).all()
    ):
        raise ValueError("Knowledge feature alignment failed")
    semantic = None
    if config.get("use_bge", False):
        sem_dir = (
            paths.output / "symptom_embeddings"
            if (
                paths.output
                / "symptom_embeddings/symptom_embeddings_bge_base_zh_v1_5.npy"
            ).exists()
            else paths.semantic
        )
        file = sem_dir / "symptom_embeddings_bge_base_zh_v1_5.npy"
        semantic = np.load(file, allow_pickle=False)
        manifest = read_json(sem_dir / "bge_symptom_embedding_manifest.json")
        if semantic.shape != (390, 768) or not np.allclose(
            np.linalg.norm(semantic, axis=1), 1, atol=1e-5
        ):
            raise ValueError("BGE shape or normalization mismatch")
        if manifest["symptom_vocab_sha256"] != sha256(
            paths.dataset / "symptom_vocab.json"
        ):
            raise ValueError("BGE vocabulary order mismatch")
        if manifest.get("outputs", {}).get("npy", {}).get("sha256") != sha256(file):
            raise ValueError("BGE cache checksum mismatch")
        semantic = torch.from_numpy(semantic.copy()).float().to(device)
    return (
        torch.from_numpy(jpa).float().to(device),
        torch.from_numpy(props).float().to(device),
        semantic,
    )


def load_graph(directory, use_properties=True):
    names = [
        "edge_index_symptom_herb_train_global.npy",
        "edge_index_herb_herb_train_global.npy",
    ]
    if use_properties:
        names.append("edge_index_herb_property_global.npy")
    arrays = [
        np.load(directory / name, allow_pickle=False).astype(np.int64) for name in names
    ]
    edge = np.concatenate(arrays, axis=1)
    if edge.shape[0] != 2 or edge.min() < 0 or edge.max() >= 1224:
        raise ValueError("Graph indices are invalid")
    loops = np.vstack([np.arange(int(edge.max()) + 1)] * 2)
    return torch.from_numpy(np.concatenate([edge, edge[[1, 0]], loops], axis=1)).long()


def load_candidate_generator(paths, checkpoint=None, device_name="cpu"):
    device = resolve_device(device_name)
    checkpoint = checkpoint or paths.gat / "best_model.pt"
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = payload.get("config") or read_json(paths.gat / "config.json")
    model = build_candidate_model(config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if "edge_index" in payload:
        edge = torch.as_tensor(payload["edge_index"], device=device, dtype=torch.long)
    else:
        graph_directory = Path(checkpoint).parent / "graph"
        if not graph_directory.is_dir():
            if Path(checkpoint).parent.name.startswith("fold_"):
                raise FileNotFoundError(
                    f"Missing fold-specific graph: {graph_directory}"
                )
            graph_directory = paths.graph
        edge = load_graph(graph_directory, config.get("use_23d", True)).to(device)
    tensors = load_knowledge_tensors(paths, config, device)
    return model.eval(), edge, tensors, device, payload


def load_student(checkpoint, device_name="cpu"):
    device = resolve_device(device_name)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_reranker(payload["config"]).to(device)
    excluded = (
        "length_head.",
        "confidence_head.",
        "formula_quality_head.",
        "patient_embedding_projector.",
    )
    model.load_state_dict(
        {
            key: value
            for key, value in payload["model_state_dict"].items()
            if not key.startswith(excluded)
        },
        strict=True,
    )
    return model.eval(), device, payload


def load_masked_herb_teacher(checkpoint, device_name="cpu"):
    device = resolve_device(device_name)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = payload["config"]
    keys = ["hidden_dim", "embedding_dim", "num_heads", "num_set_layers", "dropout"]
    model = MaskedHerbTeacher(**{key: config[key] for key in keys}).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval(), device, payload


def load_clinical_teacher(
    checkpoint, config, herb_dim, device_name="cpu", mapping_hash=None
):
    device = resolve_device(device_name)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if (
        mapping_hash is not None
        and payload.get("mapping_manifest_sha256") != mapping_hash
    ):
        raise ValueError("Clinical checkpoint mapping provenance mismatch")
    model = ClinicalTeacher(390, herb_dim, config).to(device)
    model.load_state_dict(
        payload.get("state_dict", payload.get("model_state_dict")), strict=True
    )
    return model.eval(), device, payload
