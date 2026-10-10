import numpy as np
from .data.knowledge_features import build_knowledge_features
from .io import save_json, sha256


def prepare_knowledge(paths):
    matrices = build_knowledge_features(
        paths.knowledge,
        paths.dataset / "herb_vocab.json",
        paths.output / "knowledge_features",
    )
    save_json(
        paths.output / "knowledge_features/manifest.json",
        {
            "herb_vocab_sha256": sha256(paths.dataset / "herb_vocab.json"),
            "shapes": [list(v.shape) for v in matrices],
        },
    )
    return matrices


def encode_symptoms(paths, device_name="cpu", batch_size=64):
    from .data.semantic_encoding import read_vocab, encode
    from .io import resolve_device

    vocabulary = paths.dataset / "symptom_vocab.json"
    vectors, metadata = encode(
        paths.bge_model,
        read_vocab(vocabulary),
        resolve_device(device_name),
        batch_size,
        32,
    )
    if vectors.shape != (390, 768):
        raise ValueError("BGE must produce 390x768 vectors")
    output = paths.output / "symptom_embeddings"
    output.mkdir(parents=True, exist_ok=True)
    file = output / "symptom_embeddings_bge_base_zh_v1_5.npy"
    np.save(file, vectors)
    save_json(
        output / "bge_symptom_embedding_manifest.json",
        {
            "declared_model_id": "BAAI/bge-base-zh-v1.5",
            "embedding_shape": list(vectors.shape),
            "symptom_vocab_sha256": sha256(vocabulary),
            "outputs": {"npy": {"sha256": sha256(file)}},
            "encoding": metadata,
        },
    )
    return file
