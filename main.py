from pathlib import Path
import argparse
import json
from herbrec.paths import ProjectPaths
from herbrec.data.prescriptions import PrescriptionDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "prepare-knowledge",
            "prepare-assets",
            "encode-symptoms",
            "train-candidates",
            "generate-candidates",
            "generate-oof",
            "train-masked-teacher",
            "export-masked-targets",
            "train-clinical-teacher",
            "export-clinical-targets",
            "prepare-reranking",
            "train-reranker",
            "predict",
            "evaluate",
            "run",
            "validate",
        ],
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--existing-project", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--split", choices=["train", "validation", "test"], default="validation"
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--train-steps", action="store_true")
    parser.add_argument("--without-semantics", action="store_true")
    parser.add_argument("--without-herb-features", action="store_true")
    parser.add_argument(
        "--variant",
        choices=[
            "full",
            "without_mhm",
            "without_clinical",
            "fixed_distillation",
            "without_distillation",
        ],
        default="full",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.epochs is not None and args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.max_batches is not None and args.max_batches <= 0:
        parser.error("--max-batches must be positive")
    paths = ProjectPaths(args.source_root, args.output_root, args.existing_project)
    if args.stage == "prepare-knowledge":
        from herbrec.preprocessing import prepare_knowledge

        result = [list(v.shape) for v in prepare_knowledge(paths)]
    elif args.stage == "prepare-assets":
        from herbrec.data.statistics import build_feature_assets

        result = {
            "fit_samples": len(
                build_feature_assets(paths, PrescriptionDataset(paths.dataset))[
                    "fit_sample_ids"
                ]
            )
        }
    elif args.stage == "encode-symptoms":
        from herbrec.preprocessing import encode_symptoms

        result = str(encode_symptoms(paths, args.device))
    elif args.stage == "validate":
        from herbrec.validation import validate_project

        result = validate_project(paths, args.device, args.limit or 4, args.train_steps)
    elif args.stage == "run":
        from herbrec.pipeline import run_pipeline

        result = run_pipeline(
            paths, args.device, args.retrain, args.limit, args.epochs, args.max_batches
        )
    else:
        dataset = PrescriptionDataset(paths.dataset)
        if args.stage == "train-candidates":
            from herbrec.training.candidate_generator import train_candidate_generator

            result = str(
                train_candidate_generator(
                    paths,
                    dataset,
                    args.device,
                    args.epochs,
                    args.limit,
                    args.max_batches,
                    args.fold,
                    not args.without_semantics,
                    not args.without_herb_features,
                )
            )
        elif args.stage == "generate-candidates":
            from herbrec.training.candidate_generator import generate_candidates

            result = {
                "sample_n": len(
                    generate_candidates(
                        paths,
                        dataset,
                        args.split,
                        args.checkpoint,
                        args.device,
                        args.limit,
                    )["sample_ids"]
                )
            }
        elif args.stage == "generate-oof":
            from herbrec.training.candidate_generator import generate_oof_candidates

            result = str(
                generate_oof_candidates(paths, dataset, args.device, args.retrain)
            )
        elif args.stage == "train-masked-teacher":
            from herbrec.training.masked_herb import train_masked_herb_teacher

            result = str(
                train_masked_herb_teacher(
                    paths,
                    dataset,
                    args.fold or 0,
                    args.device,
                    args.epochs,
                    args.limit,
                    args.max_batches,
                )
            )
        elif args.stage == "export-masked-targets":
            from herbrec.training.masked_herb import export_masked_herb_targets

            result = str(export_masked_herb_targets(paths, dataset, args.device))
        elif args.stage == "train-clinical-teacher":
            from herbrec.training.clinical_teacher import train_clinical_teacher

            result = str(
                train_clinical_teacher(
                    paths,
                    args.seed,
                    args.fold or 0,
                    args.device,
                    args.epochs,
                    args.limit,
                    args.max_batches,
                )
            )
        elif args.stage == "export-clinical-targets":
            from herbrec.training.clinical_teacher import export_clinical_targets

            result = {
                "sample_n": len(
                    export_clinical_targets(
                        paths, dataset, args.split, args.device, args.limit
                    )["sample_ids"]
                )
            }
        elif args.stage == "prepare-reranking":
            from herbrec.data.candidates import prepare_student_inputs

            result = {
                "sample_n": len(
                    prepare_student_inputs(paths, dataset, args.split)["sample_ids"]
                )
            }
        elif args.stage == "train-reranker":
            from herbrec.training.reranker import train_reranker

            result = str(
                train_reranker(
                    paths,
                    dataset,
                    device_name=args.device,
                    epochs=args.epochs,
                    limit=args.limit,
                    max_batches=args.max_batches,
                    variant=args.variant,
                )
            )
        elif args.stage == "predict":
            from herbrec.data.candidates import load_inference_inputs
            from herbrec.inference import recommend

            checkpoint = (
                args.checkpoint
                or paths.output
                / "checkpoints/reranker"
                / args.variant
                / "best_model.pt"
            )
            if not checkpoint.exists():
                checkpoint = paths.student_checkpoint
            inputs = load_inference_inputs(paths, args.split, args.limit)
            result = {
                "sample_n": len(
                    recommend(
                        checkpoint,
                        inputs,
                        paths.output / "predictions" / f"{args.split}.npz",
                        args.device,
                    )["sample_ids"]
                )
            }
        else:
            from herbrec.io import load_arrays, save_json
            from herbrec.evaluation import ranking_metrics

            file = (
                args.predictions or paths.output / "predictions" / f"{args.split}.npz"
            )
            predictions = load_arrays(file, ["sample_ids", "candidate_ids", "scores"])
            result = ranking_metrics(
                predictions["scores"],
                predictions["candidate_ids"],
                dataset.labels(
                    predictions["sample_ids"], allow_test=args.split == "test"
                ),
            )
            save_json(
                paths.output / "evaluation" / f"{args.split}_metrics.json", result
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
