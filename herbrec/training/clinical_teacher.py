import numpy as np
import torch
import torch.nn.functional as F
from ..io import (
    load_arrays,
    read_json,
    save_arrays,
    save_json,
    save_checkpoint,
    batch_indices,
    sha256,
    set_seed,
    resolve_device,
)
from ..teachers.clinical import ClinicalTeacher
from ..model_loading import load_clinical_teacher
from ..evaluation import mean_case_f1
from ..configuration import CLINICAL_CONFIGURATION
from copy import deepcopy


def clinical_configuration(paths):
    file = paths.clinical / "config_resolved.json"
    return read_json(file) if file.exists() else deepcopy(CLINICAL_CONFIGURATION)


def clinical_source(paths):
    return load_arrays(paths.clinical / "source_pilot_arrays.npz")


def support_masks(source):
    return {
        "input_symptom_support_mask": np.any(source["x_symptom390"] > 0, axis=0),
        "syndrome_output_support_mask": np.any(source["y_syndrome50"] > 0, axis=0),
        "treatment_output_support_mask": np.any(source["y_treatment61"] > 0, axis=0),
    }


def train_clinical_teacher(
    paths, seed=42, fold=0, device_name="cpu", epochs=None, limit=None, max_batches=None
):
    source = clinical_source(paths)
    config = clinical_configuration(paths)
    folds = load_arrays(
        paths.clinical / "source_oof_predictions.npz", ["seed_ids", "fold_ids_by_seed"]
    )
    position = np.flatnonzero(folds["seed_ids"] == seed)
    if len(position) != 1:
        raise ValueError("No frozen source folds for requested seed")
    ids = folds["fold_ids_by_seed"][position[0]]
    fit = np.flatnonzero(ids != fold)
    validation = np.flatnonzero(ids == fold)
    if not len(validation) or not len(fit):
        raise ValueError("Empty source fold")
    if set(source["source_group_hash"][fit]) & set(
        source["source_group_hash"][validation]
    ):
        raise ValueError("Clinical source-group leakage")
    device = resolve_device(device_name)
    set_seed(seed + fold)
    model = ClinicalTeacher(390, source["y_herb_common"].shape[1], config).to(device)
    settings = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    pos_weights = {}
    for key in ["y_herb_common", "y_syndrome50", "y_treatment61"]:
        positive = source[key][fit].sum(0)
        pos_weights[key] = torch.as_tensor(
            np.clip(
                (len(fit) - positive) / np.maximum(positive, 1),
                settings["pos_weight_min"],
                settings["pos_weight_max"],
            ),
            device=device,
            dtype=torch.float32,
        )
    full_fit = fit.copy()
    if limit is not None:
        fit = fit[:limit]
        validation = validation[:limit]
    destination = (
        paths.output / "checkpoints/clinical_teacher" / f"seed_{seed}" / f"fold_{fold}"
    )
    best = -1
    stale = 0
    history = []
    for epoch in range(1, (epochs or settings["max_epochs"]) + 1):
        model.train()
        losses = []
        for index, positions in enumerate(
            batch_indices(fit, settings["batch_size"], True, seed + epoch)
        ):
            if max_batches is not None and index >= max_batches:
                break
            optimizer.zero_grad(set_to_none=True)
            output = model(
                torch.as_tensor(
                    source["x_symptom390"][positions],
                    device=device,
                    dtype=torch.float32,
                )
            )
            loss = 0
            sample_weights = torch.as_tensor(
                source["sample_weight"][positions], device=device, dtype=torch.float32
            )
            for key, head, weight in [
                ("y_herb_common", "herb_logits", settings["lambda_herb"]),
                ("y_syndrome50", "syndrome_logits", settings["lambda_syndrome"]),
                ("y_treatment61", "treatment_logits", settings["lambda_treatment"]),
            ]:
                elements = F.binary_cross_entropy_with_logits(
                    output[head],
                    torch.as_tensor(
                        source[key][positions], device=device, dtype=torch.float32
                    ),
                    pos_weight=pos_weights[key],
                    reduction="none",
                ).mean(1)
                loss += (
                    weight
                    * (elements * sample_weights).sum()
                    / sample_weights.sum().clamp_min(1e-8)
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings["gradient_clip_norm"]
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        predictions = []
        with torch.inference_mode():
            for positions in batch_indices(validation, settings["batch_size"]):
                predictions.append(
                    model(
                        torch.as_tensor(
                            source["x_symptom390"][positions],
                            device=device,
                            dtype=torch.float32,
                        )
                    )["herb_logits"]
                    .cpu()
                    .numpy()
                )
        scores = np.concatenate(predictions)
        truth = source["y_herb_common"][validation] > 0
        metric = mean_case_f1(scores, truth, 10)
        history.append(
            {"epoch": epoch, "loss": float(np.mean(losses)), "source_F1@10": metric}
        )
        if metric > best:
            best = metric
            stale = 0
            save_checkpoint(
                destination / "best_model.pt",
                {
                    "state_dict": model.state_dict(),
                    "resolved_config": config,
                    "seed": seed,
                    "fold": fold,
                    "fit_positions": full_fit,
                    "validation_positions": validation,
                    "mapping_manifest_sha256": sha256(
                        paths.clinical / "exact_mapping_manifest.json"
                    ),
                    "limited_run": limit is not None or max_batches is not None,
                },
            )
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    save_json(destination / "training_history.json", history)
    return destination / "best_model.pt"


def clinical_uncertainty(logits, mask):
    if not np.any(mask):
        return np.ones(len(logits), np.float32)
    probabilities = np.clip(
        1 / (1 + np.exp(-np.clip(logits[:, mask], -30, 30))), 1e-7, 1 - 1e-7
    )
    return (
        -(
            probabilities * np.log(probabilities)
            + (1 - probabilities) * np.log(1 - probabilities)
        ).mean(1)
        / np.log(2)
    ).astype(np.float32)


def export_clinical_targets(
    paths, dataset, split="train", device_name="cpu", limit=None
):
    source = clinical_source(paths)
    config = clinical_configuration(paths)
    masks = support_masks(source)
    ids = dataset.splits[split]
    if limit is not None:
        ids = ids[:limit]
    symptoms = dataset.validate_rows(ids)
    masked = symptoms * masks["input_symptom_support_mask"][None, :]
    outputs = {
        "syndrome_logits": np.zeros((len(ids), 50), np.float32),
        "treatment_logits": np.zeros((len(ids), 61), np.float32),
    }
    mapping_hash = sha256(paths.clinical / "exact_mapping_manifest.json")
    checkpoints = []
    for seed in config["seed_list"]:
        for fold in range(config["folds"]):
            generated = (
                paths.output
                / "checkpoints/clinical_teacher"
                / f"seed_{seed}"
                / f"fold_{fold}"
                / "best_model.pt"
            )
            original = (
                paths.clinical
                / "checkpoints"
                / f"seed_{seed}"
                / "progressive"
                / f"fold_{fold}.pt"
            )
            checkpoint = generated if generated.exists() else original
            model, device, payload = load_clinical_teacher(
                checkpoint,
                config,
                source["y_herb_common"].shape[1],
                device_name,
                mapping_hash,
            )
            if payload.get("limited_run"):
                raise ValueError(
                    "Limited source checkpoints cannot replace formal clinical teacher features"
                )
            with torch.inference_mode():
                for positions in batch_indices(np.arange(len(ids)), 1024):
                    prediction = model(
                        torch.as_tensor(
                            masked[positions], device=device, dtype=torch.float32
                        )
                    )
                    for key in outputs:
                        outputs[key][positions] += prediction[key].cpu().numpy()
            checkpoints.append({"path": str(checkpoint), "sha256": sha256(checkpoint)})
    for key in outputs:
        outputs[key] /= len(checkpoints)
    counts = np.count_nonzero(symptoms, axis=1)
    supported = np.count_nonzero(masked, axis=1)
    coverage = (supported / np.maximum(counts, 1)).astype(np.float32)
    result = {
        "sample_ids": ids,
        **outputs,
        "symptom_support_coverage": coverage,
        "has_supported_symptom": supported > 0,
        "supported_symptom_count": supported,
        "total_symptom_count": counts,
        "syndrome_uncertainty": clinical_uncertainty(
            outputs["syndrome_logits"], masks["syndrome_output_support_mask"]
        ),
        "treatment_uncertainty": clinical_uncertainty(
            outputs["treatment_logits"], masks["treatment_output_support_mask"]
        ),
    }
    save_arrays(paths.output / "teacher_signals" / f"clinical_{split}.npz", result)
    save_arrays(paths.output / "teacher_signals/support_masks.npz", masks)
    save_json(
        paths.output / "teacher_signals" / f"clinical_{split}_manifest.json",
        {
            "labels_loaded": False,
            "mapping_sha256": mapping_hash,
            "checkpoints": checkpoints,
            "limited": limit is not None,
        },
    )
    return result
