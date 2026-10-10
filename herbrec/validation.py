from pathlib import Path
import numpy as np
import torch
from .data.prescriptions import PrescriptionDataset
from .data.candidates import load_inference_inputs, load_training_inputs
from .data.masking import pad_herb_sets
from .io import load_arrays, read_json, save_json, sha256
from .model_loading import (
    load_candidate_generator,
    load_student,
    load_masked_herb_teacher,
    load_clinical_teacher,
)
from .inference import predict_candidates, predict_student, recommend
from .training.reranker import training_objective, train_reranker
from .training.candidate_generator import train_candidate_generator
from .training.masked_herb import train_masked_herb_teacher
from .training.clinical_teacher import (
    train_clinical_teacher,
    clinical_source,
    support_masks,
)
from .losses.distillation import (
    CoverageAwareReliabilityGate,
    masked_soft_bce_per_sample,
)


def validate_project(paths, device_name="cpu", limit=4, train_steps=False):
    if limit < 1:
        raise ValueError("limit must be positive")
    torch.set_num_threads(2)
    dataset = PrescriptionDataset(paths.dataset)
    ids = dataset.splits["validation"][:limit]
    gat, edge, tensors, device, payload = load_candidate_generator(
        paths, device_name=device_name
    )
    candidate_logits = predict_candidates(gat, dataset, ids, edge, tensors, device)
    expected = np.load(
        paths.candidate_artifacts / "validation/gat_validation_logits.npy",
        mmap_mode="r",
    )[:limit]
    delta = float(np.abs(candidate_logits - expected).max())
    if delta > 2e-4:
        raise ValueError(f"Reorganized GAT differs from frozen prediction: {delta}")
    inputs = load_inference_inputs(paths, "validation", limit)
    student, device, student_payload = load_student(
        paths.student_checkpoint, device_name
    )
    scores, residual = predict_student(student, inputs, device, limit)
    original_predictions = (
        paths.student_checkpoint.parent / "validation_predictions.npz"
    )
    if original_predictions.exists():
        reference = load_arrays(original_predictions, ["selection_scores"])[
            "selection_scores"
        ][:limit]
        student_delta = float(np.abs(scores - reference).max())
        if student_delta > 1e-5:
            raise ValueError("Reorganized Student differs from frozen prediction")
    else:
        student_delta = None
    train = load_training_inputs(paths, dataset, limit)
    mask = load_arrays(
        paths.clinical_features / "support_masks.npz",
        ["syndrome_output_support_mask", "treatment_output_support_mask"],
    )
    support = {
        k: torch.as_tensor(v, device=device, dtype=torch.float32)
        for k, v in mask.items()
    }
    gate = CoverageAwareReliabilityGate(
        hidden_dim=student_payload["config"]["gate"]["hidden_dim"]
    ).to(device)
    if student_payload.get("gate_state_dict"):
        gate.load_state_dict(student_payload["gate_state_dict"])
    student.train()
    gate.train()
    loss, diagnostics, output = training_objective(
        student,
        gate,
        train,
        np.arange(len(train["sample_ids"])),
        support,
        student_payload["config"],
        device,
    )
    loss.backward()
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite real-data distillation loss")
    if not any(
        p.grad is not None and torch.count_nonzero(p.grad) > 0
        for p in student.parameters()
    ):
        raise ValueError("Missing Student gradients")
    gate_weights = gate(
        torch.zeros(2, 5, device=device),
        torch.tensor([0.0, 1.0], device=device),
        torch.ones(2, device=device),
    )
    if gate_weights[0, 1].item() != 0 or not torch.allclose(
        gate_weights.sum(1), torch.ones(2, device=device)
    ):
        raise ValueError("Coverage gate constraint failed")
    logits = output["syndrome_logits"].detach().requires_grad_(True)
    masked_soft_bce_per_sample(
        logits,
        torch.as_tensor(train["clinical_syndrome_logits"], device=device),
        support["syndrome_output_support_mask"],
    ).mean().backward()
    if torch.count_nonzero(
        logits.grad[:, ~support["syndrome_output_support_mask"].bool()]
    ):
        raise ValueError(
            "Unsupported clinical dimensions receive distillation gradients"
        )
    mhm, device, mhm_payload = load_masked_herb_teacher(
        paths.mhm / "oof/checkpoints/fold_0/best_model.pt", device_name
    )
    true_ids = dataset.splits["train"][:limit]
    formulas = [np.flatnonzero(row > 0).tolist() for row in dataset.labels(true_ids)]
    visible, padding, _ = pad_herb_sets([row[1:] for row in formulas])
    with torch.inference_mode():
        mhm_output = mhm(
            torch.as_tensor(dataset.validate_rows(true_ids), device=device),
            visible.to(device),
            padding.to(device),
        )
    source = clinical_source(paths)
    clinical_config = read_json(paths.clinical / "config_resolved.json")
    masks = support_masks(source)
    clinical, device, clinical_payload = load_clinical_teacher(
        paths.clinical / "checkpoints/seed_42/progressive/fold_0.pt",
        clinical_config,
        source["y_herb_common"].shape[1],
        device_name,
        sha256(paths.clinical / "exact_mapping_manifest.json"),
    )
    with torch.inference_mode():
        clinical_output = clinical(
            torch.as_tensor(
                dataset.validate_rows(ids) * masks["input_symptom_support_mask"],
                device=device,
            )
        )
    result = {
        "real_rows": limit,
        "graph_backend": gat.graph_backend,
        "candidate_checkpoint_max_delta": delta,
        "student_checkpoint_max_delta": student_delta,
        "mhm_shape": list(mhm_output.shape),
        "clinical_syndrome_shape": list(clinical_output["syndrome_logits"].shape),
        "clinical_treatment_shape": list(clinical_output["treatment_logits"].shape),
        "real_distillation_loss": float(loss.detach()),
        "zero_coverage_constraint": True,
        "unsupported_gradient_zero": True,
        "test_labels_loaded": False,
        "training_steps": {},
    }
    if train_steps:
        result["training_steps"]["candidate_generator"] = str(
            train_candidate_generator(
                paths,
                dataset,
                device_name,
                epochs=1,
                limit=limit,
                max_batches=1,
                use_bge=True,
            )
        )
        result["training_steps"]["masked_herb_teacher"] = str(
            train_masked_herb_teacher(
                paths, dataset, 0, device_name, epochs=1, limit=limit, max_batches=1
            )
        )
        result["training_steps"]["clinical_teacher"] = str(
            train_clinical_teacher(
                paths, 42, 0, device_name, epochs=1, limit=limit, max_batches=1
            )
        )
        result["training_steps"]["reranker"] = str(
            train_reranker(
                paths,
                dataset,
                device_name=device_name,
                epochs=1,
                limit=limit,
                max_batches=1,
            )
        )
        prediction = recommend(
            Path(result["training_steps"]["reranker"]),
            inputs,
            paths.output / "predictions/validated_predictions.npz",
            device_name,
        )
        if np.max(np.abs(prediction["residuals"])) > 0.500001:
            raise ValueError("Residual exceeded its bound")
    save_json(paths.output / "validation_result.json", result)
    return result
