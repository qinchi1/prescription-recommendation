from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def read_vocab(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        names = [str(value).strip() for value in raw]
    elif isinstance(raw, dict) and "index_to_name" in raw:
        index_to_name = raw["index_to_name"]
        names = [
            str(index_to_name[str(index)]).strip()
            for index in range(len(index_to_name))
        ]
    else:
        raise ValueError(f"Unsupported symptom vocabulary schema: {path}")
    if len(names) != 390:
        raise ValueError(f"Expected 390 symptoms, got {len(names)} from {path}")
    if any((not name for name in names)) or len(set(names)) != len(names):
        raise ValueError("Symptom vocabulary contains empty or duplicate names")
    return names


def encode(
    model_dir: Path,
    texts: list[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    model = AutoModel.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    ).to(device)
    model.eval()
    untruncated = tokenizer(
        texts, add_special_tokens=True, truncation=False, padding=False
    )
    token_lengths = [len(ids) for ids in untruncated["input_ids"]]
    vectors: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            pooled = outputs.last_hidden_state[:, 0]
            vectors.append(F.normalize(pooled, p=2, dim=1).cpu())
    matrix = torch.cat(vectors, dim=0).numpy().astype(np.float32, copy=False)
    metadata = {
        "resolved_device": str(device),
        "pooling": "CLS / last_hidden_state[:, 0]",
        "l2_normalized": True,
        "max_length": int(max_length),
        "maximum_untruncated_token_length": int(max(token_lengths)),
        "truncated_text_n": int(sum((length > max_length for length in token_lengths))),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (matrix, metadata)
