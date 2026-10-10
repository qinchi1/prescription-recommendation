from .data.prescriptions import PrescriptionDataset
from .data.candidates import prepare_student_inputs, build_candidate_inputs
from .io import save_arrays, save_json
from .training.candidate_generator import (
    generate_candidates,
    generate_oof_candidates,
    train_candidate_generator,
)
from .training.masked_herb import export_masked_herb_targets, train_masked_herb_teacher
from .training.clinical_teacher import export_clinical_targets, train_clinical_teacher
from .training.reranker import train_reranker
from .inference import recommend
from .evaluation import ranking_metrics
from .data.statistics import load_feature_assets, build_feature_assets


def run_pipeline(
    paths, device_name="cpu", retrain=False, limit=None, epochs=None, max_batches=None
):
    if retrain and (limit is not None or max_batches is not None):
        raise ValueError(
            "Full retraining and OOF generation require all frozen training samples"
        )
    dataset = PrescriptionDataset(paths.dataset)
    checkpoint = None
    if retrain:
        from .preprocessing import prepare_knowledge, encode_symptoms

        prepare_knowledge(paths)
        encode_symptoms(paths, device_name)
        build_feature_assets(paths, dataset)
        checkpoint = train_candidate_generator(
            paths, dataset, device_name, epochs, limit, max_batches, use_bge=True
        )
        for fold in range(5):
            train_candidate_generator(
                paths, dataset, device_name, epochs, fold=fold, use_bge=True
            )
        generate_oof_candidates(paths, dataset, device_name)
        for fold in range(5):
            train_masked_herb_teacher(paths, dataset, fold, device_name, epochs)
        export_masked_herb_targets(paths, dataset, device_name)
        from .training.clinical_teacher import clinical_configuration

        config = clinical_configuration(paths)
        for seed in config["seed_list"]:
            for fold in range(config["folds"]):
                train_clinical_teacher(paths, seed, fold, device_name, epochs)
        export_clinical_targets(paths, dataset, "train", device_name)
    candidate_output = generate_candidates(
        paths, dataset, "validation", checkpoint, device_name, limit
    )
    assets = load_feature_assets(paths, dataset)
    validation = build_candidate_inputs(
        candidate_output["sample_ids"],
        dataset.validate_rows(candidate_output["sample_ids"]),
        candidate_output["full_logits"],
        assets,
    )
    save_arrays(paths.output / "reranking_inputs/validation.npz", validation)
    prepare_student_inputs(paths, dataset, "train")
    checkpoint = train_reranker(
        paths,
        dataset,
        device_name=device_name,
        epochs=epochs,
        limit=limit,
        max_batches=max_batches,
    )
    prediction = recommend(
        checkpoint, validation, paths.output / "predictions/validation.npz", device_name
    )
    result = ranking_metrics(
        prediction["scores"],
        prediction["candidate_ids"],
        dataset.labels(prediction["sample_ids"]),
    )
    save_json(paths.output / "evaluation/validation_metrics.json", result)
    return {
        "checkpoint": str(checkpoint),
        "prediction": str(paths.output / "predictions/validation.npz"),
        "metrics": result,
        "limited_run": limit is not None or max_batches is not None,
    }
