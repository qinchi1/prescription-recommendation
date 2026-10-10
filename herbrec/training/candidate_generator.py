import numpy as np
import torch
import torch.nn.functional as F
from ..io import (
    read_json,
    save_arrays,
    save_json,
    save_checkpoint,
    set_seed,
    batch_indices,
    resolve_device,
)
from ..data.graph import (
    build_symptom_herb_edges,
    build_herb_herb_edges,
    to_global_symptom_herb,
    to_global_herb_herb,
    to_global_herb_property,
)
from ..model_loading import (
    build_candidate_model,
    load_knowledge_tensors,
    load_candidate_generator,
)
from ..inference import predict_candidates
from ..evaluation import mean_case_f1
from ..configuration import CANDIDATE_CONFIGURATION


def build_training_graph(
    dataset, fit_ids, properties, use_properties=True, min_count=2
):
    if not np.isin(fit_ids, dataset.splits["train"]).all():
        raise ValueError("Graph construction requires only frozen training IDs")
    labels = dataset.labels(fit_ids)
    x = dataset.validate_rows(fit_ids)
    local = np.arange(len(fit_ids))
    parts = [
        to_global_symptom_herb(build_symptom_herb_edges(x, labels, local), 390),
        to_global_herb_herb(build_herb_herb_edges(labels, local, min_count), 390),
    ]
    if use_properties:
        rows, cols = np.nonzero(properties > 0)
        parts.append(to_global_herb_property(np.vstack([rows, cols]), 390, 811))
    edge = np.concatenate(parts, axis=1)
    nodes = 1224 if use_properties else 1201
    return torch.from_numpy(
        np.concatenate([edge, edge[[1, 0]], np.vstack([np.arange(nodes)] * 2)], axis=1)
    ).long()


def train_candidate_generator(
    paths,
    dataset,
    device_name="cpu",
    epochs=None,
    limit=None,
    max_batches=None,
    fold=None,
    use_bge=None,
    use_herb_features=True,
):
    config = (
        read_json(paths.gat / "config.json")
        if (paths.gat / "config.json").exists()
        else CANDIDATE_CONFIGURATION.copy()
    )
    if use_bge is not None:
        config["use_bge"] = use_bge
    if not use_herb_features:
        config.update(use_jpa=False, use_23d=False)
    if fold is None:
        fit = dataset.splits["train"]
        calibration = dataset.splits["validation"]
        run = "full"
    else:
        directory = paths.oof / "checkpoints" / f"fold_{fold}"
        fit = np.load(directory / "fit_ids.npy")
        calibration = np.load(directory / "calibration_ids.npy")
        holdout = np.load(directory / "holdout_ids.npy")
        if np.intersect1d(np.concatenate([fit, calibration]), holdout).size:
            raise ValueError("OOF candidate leakage")
        if not np.isin(
            np.concatenate([fit, calibration, holdout]), dataset.splits["train"]
        ).all():
            raise ValueError("OOF candidate fold is outside frozen training split")
        run = f"fold_{fold}"
    full_fit = fit.copy()
    if limit is not None:
        fit = fit[:limit]
        calibration = calibration[:limit]
    device = resolve_device(device_name)
    set_seed(config.get("seed", 42) + (fold or 0))
    model = build_candidate_model(config).to(device)
    tensors = load_knowledge_tensors(paths, config, device)
    edge = build_training_graph(
        dataset, full_fit, tensors[1].cpu().numpy(), config.get("use_23d", True)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    best = -1
    stale = 0
    history = []
    destination = paths.output / "checkpoints/candidate_generator" / run
    for epoch in range(1, (epochs or config["epochs"]) + 1):
        model.train()
        losses = []
        for index, ids in enumerate(
            batch_indices(fit, config["batch_size"], True, config["seed"] + epoch)
        ):
            if max_batches is not None and index >= max_batches:
                break
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                torch.as_tensor(dataset.validate_rows(ids), device=device),
                edge,
                tensors[0],
                tensors[1],
                bge_symptom_features=tensors[2],
            )
            loss = F.binary_cross_entropy_with_logits(
                logits, torch.as_tensor(dataset.labels(ids), device=device)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append(float(loss.detach()))
        logits = predict_candidates(
            model, dataset, calibration, edge, tensors, device, config["batch_size"]
        )
        ids = np.argsort(-logits, axis=1, kind="stable")[:, :50]
        metric = mean_case_f1(logits, dataset.labels(calibration), 10)
        history.append(
            {"epoch": epoch, "loss": float(np.mean(losses)), "mean_case_F1@10": metric}
        )
        if metric > best:
            best = metric
            stale = 0
            save_checkpoint(
                destination / "best_model.pt",
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "edge_index": edge.cpu(),
                    "epoch": epoch,
                    "fit_sample_ids": full_fit,
                    "calibration_sample_ids": calibration,
                    "fold": fold,
                    "limited_run": limit is not None or max_batches is not None,
                    "test_labels_used": False,
                },
            )
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    save_json(destination / "training_history.json", history)
    return destination / "best_model.pt"


def generate_candidates(
    paths, dataset, split="validation", checkpoint=None, device_name="cpu", limit=None
):
    model, edge, tensors, device, payload = load_candidate_generator(
        paths, checkpoint, device_name
    )
    ids = dataset.splits[split]
    if limit is not None:
        ids = ids[:limit]
    logits = predict_candidates(model, dataset, ids, edge, tensors, device)
    candidate_ids = np.argsort(-logits, axis=1, kind="stable")[:, :50]
    arrays = {
        "sample_ids": ids,
        "candidate_ids": candidate_ids,
        "candidate_logits": np.take_along_axis(logits, candidate_ids, axis=1),
        "full_logits": logits,
    }
    save_arrays(paths.output / "candidates" / f"{split}.npz", arrays)
    return arrays


def generate_oof_candidates(paths, dataset, device_name="cpu", train_missing=False):
    merged_ids = dataset.splits["train"]
    positions = {int(s): i for i, s in enumerate(merged_ids)}
    logits = np.zeros((len(merged_ids), 811), np.float32)
    fold_ids = np.full(len(merged_ids), -1, np.int16)
    seen = np.zeros(len(merged_ids), np.int8)
    for fold in range(5):
        directory = paths.oof / "checkpoints" / f"fold_{fold}"
        heldout = np.load(directory / "holdout_ids.npy")
        fit = np.load(directory / "fit_ids.npy")
        cal = np.load(directory / "calibration_ids.npy")
        if np.intersect1d(heldout, np.concatenate([fit, cal])).size:
            raise ValueError("OOF heldout rows overlap training/calibration")
        local = (
            paths.output
            / "checkpoints/candidate_generator"
            / f"fold_{fold}"
            / "best_model.pt"
        )
        checkpoint = local if local.exists() else directory / "best_model.pt"
        if not checkpoint.exists() and train_missing:
            checkpoint = train_candidate_generator(
                paths, dataset, device_name, fold=fold
            )
        model, edge, tensors, device, payload = load_candidate_generator(
            paths, checkpoint, device_name
        )
        if payload.get("limited_run"):
            raise ValueError(
                "Limited checkpoints cannot be used as formal OOF candidate generators"
            )
        values = predict_candidates(model, dataset, heldout, edge, tensors, device)
        for i, sid in enumerate(heldout):
            pos = positions[int(sid)]
            if seen[pos]:
                raise ValueError("Duplicate OOF prediction")
            logits[pos] = values[i]
            fold_ids[pos] = fold
            seen[pos] = 1
    if not np.all(seen == 1):
        raise ValueError("Incomplete OOF candidate coverage")
    out = paths.output / "candidates/oof"
    out.mkdir(parents=True, exist_ok=True)
    for name, value in [
        ("sample_ids", merged_ids),
        ("gat_oof_logits", logits),
        ("fold_ids", fold_ids),
    ]:
        np.save(out / f"{name}.npy", value)
    save_json(
        out / "oof_manifest.json",
        {
            "strict_oof": True,
            "sample_n": len(merged_ids),
            "validation_used": False,
            "test_used": False,
        },
    )
    return out
