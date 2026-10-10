from pathlib import Path
import hashlib
import json
import random
import numpy as np
import torch


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=lambda v: v.tolist() if isinstance(v, np.ndarray) else str(v),
        )
        + "\n",
        encoding="utf-8",
    )


def load_arrays(path, keys=None):
    with np.load(path, allow_pickle=False) as file:
        requested = file.files if keys is None else keys
        missing = set(requested) - set(file.files)
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        return {key: np.asarray(file[key]) for key in requested}


def save_arrays(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_indices(ids, batch_size, shuffle=False, seed=0):
    values = np.asarray(ids).copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(values)
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.zeros(1, device=device).add_(1)
    return device
