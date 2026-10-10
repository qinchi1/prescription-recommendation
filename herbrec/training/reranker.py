from copy import deepcopy
import numpy as np
import torch
from ..configuration import student_configuration
from ..data.candidates import load_training_inputs, load_inference_inputs
from ..evaluation import ranking_metrics
from ..inference import student_batch, predict_student
from ..io import (
    load_arrays,
    batch_indices,
    save_checkpoint,
    save_arrays,
    save_json,
    set_seed,
    resolve_device,
)
from ..losses.recommendation import (
    class_weighted_bce,
    listwise_positive_negative_loss,
    boundary_ranking_loss,
)
from ..losses.distillation import (
    CoverageAwareReliabilityGate,
    candidate_kl_per_sample,
    masked_soft_bce_per_sample,
    make_gate_features,
)
from ..model_loading import build_reranker


def training_objective(
    model,
    gate,
    data,
    positions,
    support,
    config,
    device,
    mhm=True,
    clinical=True,
    gated=True,
):
    batch = student_batch(data, positions, device)
    output = model(batch)
    labels = torch.as_tensor(data["hard_labels"][positions], device=device)
    weights = config["loss"]
    hard = weights["hard_select"] * class_weighted_bce(
        output["selection_logits"], labels, batch["candidate_mask"]
    )
    hard += weights["listwise"] * listwise_positive_negative_loss(
        output["selection_logits"], labels, batch["candidate_mask"]
    )
    hard += weights["boundary"] * boundary_ranking_loss(
        output["selection_logits"],
        torch.as_tensor(data["late_true_mask"][positions], device=device),
        torch.as_tensor(data["high_false_mask"][positions], device=device),
    )
    zero = output["selection_logits"].sum() * 0
    losses_m = torch.zeros(len(positions), device=device)
    losses_t = torch.zeros_like(losses_m)
    tensor = lambda key: torch.as_tensor(
        data[key][positions], device=device, dtype=torch.float32
    )
    teacher_logits = tensor("mhm_logits")
    coverage = tensor("clinical_symptom_support_coverage")
    available = tensor("clinical_has_supported_symptom")
    if mhm:
        losses_m = candidate_kl_per_sample(
            output["selection_logits"],
            teacher_logits,
            batch["candidate_mask"],
            weights["mhm_temperature"],
        )
    if clinical:
        syndrome = masked_soft_bce_per_sample(
            output["syndrome_logits"],
            tensor("clinical_syndrome_logits"),
            support["syndrome_output_support_mask"],
            weights["tcm_temperature"],
        )
        treatment = masked_soft_bce_per_sample(
            output["treatment_logits"],
            tensor("clinical_treatment_logits"),
            support["treatment_output_support_mask"],
            weights["tcm_temperature"],
        )
        losses_t = (
            weights["syndrome_fraction"] * syndrome
            + weights["treatment_fraction"] * treatment
        )
    gate_weights = None
    if gated and mhm and clinical:
        features = make_gate_features(
            coverage,
            available,
            tensor("clinical_syndrome_uncertainty"),
            tensor("clinical_treatment_uncertainty"),
            tensor("mhm_uncertainty"),
            teacher_logits,
        )
        gate_weights = gate(features, coverage, available)
        kd = (
            weights["gated_teacher_total"]
            * (gate_weights[:, 0] * losses_m + gate_weights[:, 1] * losses_t).mean()
        )
    else:
        kd = zero
        if mhm:
            kd += weights["mhm_kd"] * losses_m.mean()
        if clinical:
            kd += (
                weights["tcm_kd"]
                * (losses_t * available).sum()
                / available.sum().clamp_min(1e-8)
            )
    return (
        hard + kd,
        {
            "hard": hard,
            "distillation": kd,
            "mhm": losses_m.mean(),
            "clinical": losses_t.mean(),
            "weights": gate_weights,
        },
        output,
    )


def train_reranker(
    paths,
    dataset,
    config=None,
    device_name="cpu",
    epochs=None,
    limit=None,
    max_batches=None,
    variant="full",
):
    config = deepcopy(config or student_configuration())
    device = resolve_device(device_name)
    set_seed(config["formal_seed"])
    torch.set_num_threads(config["training"].get("torch_num_threads", 2))
    train = load_training_inputs(paths, dataset, limit)
    validation = load_inference_inputs(paths, "validation", limit)
    labels = dataset.labels(validation["sample_ids"])
    support_path = paths.output / "teacher_signals/support_masks.npz"
    if not support_path.exists():
        support_path = paths.clinical_features / "support_masks.npz"
    support = load_arrays(
        support_path, ["syndrome_output_support_mask", "treatment_output_support_mask"]
    )
    support = {
        key: torch.as_tensor(value, device=device, dtype=torch.float32)
        for key, value in support.items()
    }
    model = build_reranker(config).to(device)
    gate = CoverageAwareReliabilityGate(hidden_dim=config["gate"]["hidden_dim"]).to(
        device
    )
    flags = {
        "full": (True, True, True),
        "without_mhm": (False, True, False),
        "without_clinical": (True, False, False),
        "fixed_distillation": (True, True, False),
        "without_distillation": (False, False, False),
    }
    if variant not in flags:
        raise ValueError(variant)
    mhm, clinical, gated = flags[variant]
    parameters = list(model.parameters()) + (list(gate.parameters()) if gated else [])
    settings = config["training"]
    optimizer = torch.optim.AdamW(
        parameters, lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    scores, _ = predict_student(
        model, validation, device, settings["evaluation_batch_size"]
    )
    best = ranking_metrics(scores, validation["candidate_ids"], labels)["F1@10"]
    best_state = deepcopy(model.state_dict())
    best_gate = deepcopy(gate.state_dict())
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, (epochs or settings["max_epochs"]) + 1):
        model.train()
        gate.train()
        totals = []
        for index, positions in enumerate(
            batch_indices(
                np.arange(len(train["sample_ids"])),
                settings["batch_size"],
                True,
                config["formal_seed"] * 10000 + epoch,
            )
        ):
            if max_batches is not None and index >= max_batches:
                break
            optimizer.zero_grad(set_to_none=True)
            loss, diagnostics, output = training_objective(
                model,
                gate,
                train,
                positions,
                support,
                config,
                device,
                mhm,
                clinical,
                gated,
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite reranker loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_clip_norm"])
            optimizer.step()
            totals.append(
                {
                    key: float(value.detach())
                    for key, value in diagnostics.items()
                    if key != "weights"
                }
            )
        scores, _ = predict_student(
            model, validation, device, settings["evaluation_batch_size"]
        )
        metric = ranking_metrics(scores, validation["candidate_ids"], labels)["F1@10"]
        history.append(
            {
                "epoch": epoch,
                "validation_F1@10": metric,
                "losses": {
                    key: float(np.mean([r[key] for r in totals])) for key in totals[0]
                }
                if totals
                else {},
            }
        )
        if metric > best + settings["minimum_delta"]:
            best = metric
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            best_gate = deepcopy(gate.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= settings["early_stopping_patience"]:
            break
    destination = paths.output / "checkpoints/reranker" / variant
    checkpoint = destination / "best_model.pt"
    save_checkpoint(
        checkpoint,
        {
            "model_state_dict": best_state,
            "gate_state_dict": best_gate,
            "config": config,
            "best_epoch": best_epoch,
            "selected_on": "validation F1@10",
            "test_labels_used": False,
            "limited_run": limit is not None or max_batches is not None,
        },
    )
    model.load_state_dict(best_state)
    scores, residual = predict_student(
        model, validation, device, settings["evaluation_batch_size"]
    )
    save_arrays(
        destination / "validation_predictions.npz",
        {
            "sample_ids": validation["sample_ids"],
            "candidate_ids": validation["candidate_ids"],
            "scores": scores,
            "residuals": residual,
        },
    )
    save_json(destination / "training_history.json", history)
    return checkpoint
