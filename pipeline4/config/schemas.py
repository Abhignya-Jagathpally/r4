"""Pydantic v2 configuration schemas for all pipeline stages."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class BaseConfig(BaseModel):
    """Global pipeline configuration."""

    seed: int = Field(default=42, description="Random seed for reproducibility")
    device: str = Field(default="auto", description="Compute device: auto, cpu, cuda")
    log_level: str = Field(default="INFO", description="Logging level")
    data_dir: str = Field(default="data", description="Root data directory")
    results_dir: str = Field(default="results", description="Results output directory")
    checkpoints_dir: str = Field(default="checkpoints", description="Checkpoint directory")
    pipeline_name: str = Field(default="R4-MM-Clinical")
    pipeline_version: str = Field(default="0.1.0")

    @field_validator("device")
    @classmethod
    def resolve_device(cls, v: str) -> str:
        if v == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return v


class IngestConfig(BaseModel):
    """Data ingestion configuration."""

    geo_accessions: List[str] = Field(default=["GSE24080"])
    expression_source: str = Field(default="r3_pseudobulk")
    expression_path: str = Field(default="data/raw/r3_pseudobulk.h5ad")
    clinical_columns: Dict[str, str] = Field(default_factory=lambda: {
        "time_col": "survival_time",
        "event_col": "event",
        "stage_col": "iss_stage",
        "treatment_col": "treatment",
        "cytogenetics_col": "cytogenetic_risk",
        "age_col": "age",
        "sex_col": "sex",
    })
    min_patients: int = Field(default=20, ge=1)
    demo_mode: bool = Field(default=False, description="Generate synthetic data for demo")
    demo_n_patients: int = Field(default=500, ge=50)


class FeaturesConfig(BaseModel):
    """Feature engineering configuration."""

    variance_threshold: float = Field(default=0.01, ge=0.0)
    n_top_genes: int = Field(default=2000, ge=10)
    pathway_scoring: bool = Field(default=True)
    pathway_db: str = Field(default="hallmark")
    n_pca_components: int = Field(default=50, ge=2)
    encode_clinical: bool = Field(default=True)
    hallmark_pathways: List[str] = Field(default_factory=lambda: [
        "MYC_TARGETS_V1", "MTORC1_SIGNALING", "IL6_JAK_STAT3_SIGNALING",
        "INTERFERON_GAMMA_RESPONSE", "UNFOLDED_PROTEIN_RESPONSE", "HYPOXIA",
        "P53_PATHWAY", "TNFA_SIGNALING_VIA_NFKB",
    ])


class CohortConfig(BaseModel):
    """Cohort construction and splitting configuration."""

    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    val_size: float = Field(default=0.15, gt=0.0, lt=1.0)
    cv_folds: int = Field(default=5, ge=2)
    stratify_by: str = Field(default="event")
    random_state: int = Field(default=42)
    min_events_per_fold: int = Field(default=5, ge=1)


class TrainConfig(BaseModel):
    """Model training configuration."""

    enabled_models: List[str] = Field(
        default=["cox_ph", "deepsurv", "rsf", "response_classifier", "attention_fusion"]
    )
    model_params: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {
        "cox_ph": {"penalizer": 0.1, "l1_ratio": 0.0},
        "deepsurv": {"hidden_dims": [256, 128, 64], "dropout": 0.3, "weight_decay": 1e-4},
        "rsf": {"n_estimators": 100, "min_samples_leaf": 15, "max_features": "sqrt"},
        "response_classifier": {"model_type": "xgboost"},
        "attention_fusion": {"hidden_dim": 128, "n_heads": 4, "dropout": 0.2},
    })
    n_epochs: int = Field(default=100, ge=1)
    early_stopping_patience: int = Field(default=10, ge=1)
    batch_size: int = Field(default=64, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)


class EvaluateConfig(BaseModel):
    """Evaluation configuration."""

    metrics: List[str] = Field(
        default=["c_index", "td_auc", "brier_score", "calibration"]
    )
    bootstrap_n: int = Field(default=1000, ge=10)
    ci_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    fairness_groups: List[str] = Field(default=["iss_stage"])
    time_horizons: List[float] = Field(default=[12.0, 24.0, 36.0, 60.0])
    risk_groups: int = Field(default=3, ge=2)


class InterpretConfig(BaseModel):
    """Interpretation configuration."""

    shap_max_samples: int = Field(default=500, ge=10)
    top_k_genes: int = Field(default=50, ge=1)
    run_pathway_enrichment: bool = Field(default=True)
    attention_extraction: bool = Field(default=True)
    stability_n_bootstrap: int = Field(default=100, ge=10)
    stability_threshold: float = Field(default=0.6, gt=0.0, le=1.0)
    pathway_pvalue_threshold: float = Field(default=0.05, gt=0.0, lt=1.0)


class ReportConfig(BaseModel):
    """Report generation configuration."""

    output_format: str = Field(default="html")
    figure_dpi: int = Field(default=300, ge=72)
    template_name: str = Field(default="default")
    include_sections: List[str] = Field(default_factory=lambda: [
        "kaplan_meier", "model_comparison", "calibration",
        "shap_summary", "biomarker_table", "fairness_dashboard",
    ])


class AutotuneConfig(BaseModel):
    """Hyperparameter autotune configuration."""

    n_trials: int = Field(default=50, ge=1)
    timeout_seconds: int = Field(default=3600, ge=60)
    target_model: str = Field(default="deepsurv")
    target_metric: str = Field(default="c_index")
    search_spaces: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    pruner: str = Field(default="hyperband")
    pruner_params: Dict[str, Any] = Field(default_factory=lambda: {
        "min_resource": 5, "max_resource": 100, "reduction_factor": 3,
    })


class PipelineConfig(BaseModel):
    """Root configuration aggregating all stage configs."""

    base: BaseConfig = Field(default_factory=BaseConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    cohort: CohortConfig = Field(default_factory=CohortConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    evaluate: EvaluateConfig = Field(default_factory=EvaluateConfig)
    interpret: InterpretConfig = Field(default_factory=InterpretConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    autotune: AutotuneConfig = Field(default_factory=AutotuneConfig)

    @classmethod
    def from_dir(cls, config_dir: str) -> "PipelineConfig":
        """Load configuration from a directory of JSON files.

        Each JSON file maps to a section: base.json -> base, train.json -> train, etc.
        """
        config_path = Path(config_dir)
        merged: Dict[str, Any] = {}

        field_names = set(cls.model_fields.keys())

        for json_file in sorted(config_path.glob("*.json")):
            section = json_file.stem
            with open(json_file) as f:
                data = json.load(f)

            if section in field_names:
                merged[section] = data
            else:
                logger.debug(f"Skipping unknown config section: {section}")

        config = cls(**merged)
        logger.info(
            f"Loaded config from {config_dir}: "
            f"{len(list(config_path.glob('*.json')))} JSON files"
        )
        return config

    def save(self, output_dir: str) -> None:
        """Save configuration to JSON files."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for field_name in self.model_fields:
            section = getattr(self, field_name)
            with open(out / f"{field_name}.json", "w") as f:
                json.dump(section.model_dump(), f, indent=2)
