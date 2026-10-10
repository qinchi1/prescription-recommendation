import numpy as np
import torch
from .data.candidates import INPUT_FIELDS, validate_candidates
from .io import batch_indices, save_arrays
from .model_loading import load_student


def student_batch(data, positions, device):
    keys = INPUT_FIELDS[1:]
    integers = {"candidate_ids", "frequency_group_id"}
    return {
        key: torch.as_tensor(
            data[key][positions],
            device=device,
            dtype=torch.long if key in integers else torch.float32,
        )
        for key in keys
    }


def predict_student(model, inputs, device, batch_size=512):
    validate_candidates(inputs)
    scores = []
    residuals = []
    model.eval()
    with torch.inference_mode():
        for positions in batch_indices(
            np.arange(len(inputs["sample_ids"])), batch_size
        ):
            output = model(student_batch(inputs, positions, device))
            scores.append(output["selection_logits"].cpu().numpy())
            residuals.append(output["selection_residual"].cpu().numpy())
    return np.concatenate(scores), np.concatenate(residuals)


def predict_candidates(
    model, dataset, sample_ids, edge, tensors, device, batch_size=256
):
    values = []
    model.eval()
    with torch.inference_mode():
        for ids in batch_indices(sample_ids, batch_size):
            x = torch.as_tensor(dataset.validate_rows(ids), device=device)
            values.append(
                model(x, edge, tensors[0], tensors[1], bge_symptom_features=tensors[2])
                .cpu()
                .numpy()
            )
    return np.concatenate(values)


def recommend(checkpoint, inputs, output, device_name="cpu"):
    forbidden = [key for key in inputs if key not in INPUT_FIELDS]
    if forbidden:
        raise ValueError(f"Inference accepts only label-free model inputs: {forbidden}")
    model, device, payload = load_student(checkpoint, device_name)
    scores, residuals = predict_student(model, inputs, device)
    ranks = np.argsort(-scores, axis=1, kind="stable")
    result = {
        "sample_ids": inputs["sample_ids"],
        "candidate_ids": inputs["candidate_ids"],
        "scores": scores,
        "residuals": residuals,
        "ranked_ids": np.take_along_axis(inputs["candidate_ids"], ranks, axis=1),
    }
    save_arrays(output, result)
    return result
