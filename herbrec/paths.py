from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectPaths:
    source: Path
    output: Path
    legacy: bool = False

    def __post_init__(self):
        self.source = Path(self.source).resolve()
        self.output = Path(self.output).resolve()
        if (
            self.source == self.output
            or self.source in self.output.parents
            and self.legacy
        ):
            raise ValueError(
                "Runtime output must be outside the original source project"
            )

    @property
    def dataset(self):
        return self.source / (
            "output/herb_recommendation/dataset_split_622_seed42"
            if self.legacy
            else "data/prescriptions"
        )

    @property
    def features(self):
        return self.source / (
            "output/herb_recommendation/split622/features"
            if self.legacy
            else "data/knowledge_features"
        )

    @property
    def knowledge(self):
        return self.source / ("Data" if self.legacy else "data/knowledge")

    @property
    def semantic(self):
        return self.source / (
            "output/herb_recommendation/bge_symptom_features_v1"
            if self.legacy
            else "data/symptom_embeddings"
        )

    @property
    def bge_model(self):
        return self.source / "bge-base-zh-v1.5"

    @property
    def gat(self):
        return self.source / (
            "output/herb_recommendation/split622/v5pro_full/seed_42"
            if self.legacy
            else "checkpoints/candidate_generator"
        )

    @property
    def graph(self):
        return self.source / (
            "output/herb_recommendation/split622/graph"
            if self.legacy
            else "data/knowledge_graph"
        )

    @property
    def oof(self):
        return self.source / (
            "output/formula_evaluation/gat_stage1_artifacts_v1/oof"
            if self.legacy
            else "data/candidate_folds"
        )

    @property
    def candidate_artifacts(self):
        return self.source / (
            "output/formula_evaluation/gat_stage1_artifacts_v1"
            if self.legacy
            else "artifacts/candidates"
        )

    @property
    def student_data(self):
        return self.source / (
            "output/formula_evaluation/multi_teacher_distillation_v1/student_data"
            if self.legacy
            else "data/reranking"
        )

    @property
    def mhm(self):
        return self.source / (
            "output/formula_evaluation/multi_teacher_distillation_v1/teacher_signals/mhm"
            if self.legacy
            else "artifacts/masked_herb_teacher"
        )

    @property
    def clinical(self):
        return self.source / (
            "private_outputs/tcm_lung_teacher_pilot_exact"
            if self.legacy
            else "data/clinical_teacher"
        )

    @property
    def clinical_features(self):
        return self.source / (
            "artifacts/tcm_lung_teacher_features_v1"
            if self.legacy
            else "artifacts/clinical_teacher"
        )

    @property
    def inference_data(self):
        return self.source / (
            "artifacts/route2_heterogeneous_dual_teacher_v1"
            if self.legacy
            else "artifacts/reranking_inputs"
        )

    @property
    def student_checkpoint(self):
        return self.source / (
            "reports/route2_heterogeneous_dual_teacher_v1/checkpoints/single_seed/seed_7319/R5/best_model.pt"
            if self.legacy
            else "checkpoints/reranker/best_model.pt"
        )
