# Herbal Prescription Recommendation

Research implementation for symptom-based herbal set recommendation. The model takes a multi-hot symptom vector from a fixed vocabulary, generates herb candidates, and reranks the candidate set with a student model trained through dual-teacher distillation.

```text
390-dimensional symptom input
              |
     Graph candidate model
              |
    Scores over 811 herbs
              |
       Top-50 candidates
              |
      Set-based reranking
              ^
    Distillation during training
  Masked-herb teacher + Syndrome/treatment-principle teacher
```

The output consists of herb IDs and ranking scores. The model does not generate dosages, and this release does not provide evidence of clinical efficacy or clinical prescription review.

## Implementation

| Component | Location | Function |
|---|---|---|
| Data and features | `herbrec/data/` | Fixed vocabularies and splits, 90-dimensional knowledge features, 23-dimensional attributes, BGE symptom embeddings, and candidate features |
| Candidate model | `herbrec/models/knowledge_graph.py` | GATConv, TransformerConv, and cross-attention for scoring 811 herbs |
| Teachers | `herbrec/teachers/` | Symptom-conditioned masked-herb prediction and a multitask teacher for syndromes, treatment principles, and herbs |
| Reranker | `herbrec/models/reranker.py` | A set Transformer that adds bounded residuals to standardized candidate scores |
| Losses and training | `herbrec/losses/`, `herbrec/training/` | Supervised ranking losses, dual-teacher distillation, coverage-aware gating, and ablation interfaces |
| Inference and evaluation | `herbrec/inference.py`, `herbrec/evaluation.py` | Student inference without labels or teacher targets, ranking outputs, and set recommendation metrics |

Default parameters are defined in `herbrec/configuration.py`. The command-line entry point is `main.py`.

## Environment

The previously checked CPU environment used Python 3.11, NumPy 1.26.4, pandas 2.2.3, PyTorch 2.6.0, torch-geometric 2.6.1, and transformers 4.57.6. These versions are not a complete environment lock, and GPU compatibility has not been comprehensively verified for this release.

```powershell
python install_dependencies.py
python main.py --help
```

The dependency installer uses broad version ranges for some packages. Successful installation or imports do not establish that training on the full dataset has completed.

## Required Assets

This repository does not include patient records, herb knowledge files, a pretrained BGE model, teacher caches, or model weights. It does not automatically download these assets. Public sources, versions, access instructions, and licensing information must be supplied when the corresponding assets are released. The repository alone is therefore insufficient to independently reproduce the paper's results.

`--source-root` specifies the asset root, and `--output-root` specifies a new output directory. The default asset layout is defined in `herbrec/paths.py`.

| Relative asset directory | Required contents |
|---|---|
| `data/prescriptions/` | `X_symptom_multihot.npy` (N × 390), `Y_herb_multihot.npy` (N × 811), symptom/herb vocabulary JSON files, `train_idx.npy`, `val_idx.npy`, and `test_idx.npy` |
| `data/knowledge/`, `data/knowledge_features/` | Herb knowledge tables, syndrome-to-attribute mappings, and 90-/23-dimensional features |
| `bge-base-zh-v1.5/`, `data/symptom_embeddings/` | A local BGE model, symptom embeddings, and vocabulary-order verification metadata |
| `data/candidate_folds/` | Per-fold fit/calibration/holdout manifests, corresponding models and graphs, and merged out-of-fold (OOF) arrays |
| `data/reranking/`, `artifacts/reranking_inputs/` | Static statistics, training tensors, and label-free validation or test inputs |
| `artifacts/masked_herb_teacher/` | Per-fold manifests, weights, frequency groups, and OOF caches for the masked-herb teacher |
| `data/clinical_teacher/`, `artifacts/clinical_teacher/` | Clinical source arrays, fixed source folds, mapping manifests, teacher weights, outputs, and support masks |
| `checkpoints/`, `data/knowledge_graph/` | Candidate/student model weights and training graphs |

Vocabulary order, sample IDs, candidate order, and caches must agree. The code consumes existing standardized arrays and fixed splits; it does not provide a complete workflow for rebuilding all assets from arbitrary raw patient records.

## Running with an Existing Research Workspace

If the required assets are available in a `GraphAI-for-TCM-main` workspace, use `--existing-project` to support its original directory layout. Run the following commands from this repository's root. Set `$source` to the actual asset directory and use a new directory for `$output`.

```powershell
$source = "../GraphAI-for-TCM-main"
$output = "../prescription_recommendation_runs/check"

python main.py validate --source-root $source --output-root $output --existing-project --device cpu --limit 4 --train-steps
```

This command requires existing reference weights and prediction files. It checks model loading, computation on real samples, gradients, and small-batch updates for the trainers.

With candidate OOF arrays and both teacher caches available, the following command checks the connection between candidate prediction, student training, inference, and validation evaluation:

```powershell
$output = "../prescription_recommendation_runs/pipeline_check"
python main.py run --source-root $source --output-root $output --existing-project --device cpu --limit 4 --epochs 1 --max-batches 1
```

This is a limited-sample execution check. Its weights and metrics are not formal experimental results. Removing `--limit`, `--epochs`, and `--max-batches` uses the configured full student training budget, while still requiring the prepared assets.

To run student prediction separately, specify the selected checkpoint explicitly:

```powershell
python main.py predict --source-root $source --output-root $output --existing-project --device cpu --split validation --checkpoint "$output/checkpoints/reranker/full/best_model.pt"
```

`run --retrain` retrains the models using prepared fixed folds and some existing OOF assets. It is not an independent reproduction command starting from an empty directory or arbitrary raw data.

## Outputs and Metrics

Outputs are written under `--output-root`: candidate predictions to `candidates/`, student inputs to `reranking_inputs/`, weights to `checkpoints/`, ranking results to `predictions/`, and evaluation results to `evaluation/`.

Prediction NPZ files contain `sample_ids`, `candidate_ids`, `scores`, `residuals`, and `ranked_ids`. Herb IDs refer to the fixed herb vocabulary in the asset directory. Scores are used for ranking and should not be interpreted as clinical medication probabilities.

The default `ranking_metrics` aggregates hits across samples to compute micro P/R/F1@5/10/20. CandidateRecall@50 also aggregates hits across all ground-truth herbs. Candidate-model and clinical-teacher checkpoint selection uses `mean_case_f1`, which averages F1 over samples. These two F1 definitions cannot be compared as the same metric when reporting improvements.

## Validation Scope and Current Limitations

During the original code-packaging checks, four real samples were used to verify model computation, one batch update per trainer, and the connection from candidate prediction to student training, inference, and validation evaluation. Predictions from the existing student checkpoint matched the reference; the maximum difference for the candidate checkpoint was approximately 3.8 × 10⁻⁶. Full training, multi-seed experiments, and independent test evaluation were not rerun. Small-batch checks do not establish performance improvements or publication readiness.

The current version still has the following limitations:

- OOF export checks fold separation but does not fully verify the checkpoint's actual training/calibration samples or graph provenance. `strict_oof=true` is not a substitute for model-provenance evidence.
- Static herb-frequency and symptom co-occurrence features are fitted after excluding only fold 0. The implementation cannot claim that every training input uses per-sample OOF statistics. This is separate from using training-only statistics for validation features.
- Default checkpoint paths and prediction fields are not fully aligned across training and validation commands. Some entry points may fall back to checkpoints in the source directory.
- Negative sample-ID checks and evaluation-partition checks need strengthening. Sample membership must be verified before formal evaluation.
- The CLI `--seed` does not yet control every trainer. Teacher-free ablations still read teacher assets.

Reproducible result releases require these issues to be addressed, asset access instructions to be supplied, and baselines, ablations, and statistical results to be reported under matched splits and consistent metrics. This README does not claim that these issues have been resolved.

This release does not include an open-source license. Redistribution conditions for third-party data, models, and code must be checked separately.
