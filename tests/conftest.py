"""Shared pytest fixtures for R4 pipeline tests."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_clinical_data():
    """Generate small synthetic clinical DataFrame."""
    rng = np.random.RandomState(42)
    n = 100
    return pd.DataFrame({
        "patient_id": [f"PT_{i:03d}" for i in range(n)],
        "survival_time": rng.weibull(1.5, n) * 40,
        "event": rng.binomial(1, 0.6, n),
        "iss_stage": rng.choice([1, 2, 3], n),
        "cytogenetic_risk": rng.choice(["standard", "high"], n),
        "treatment": rng.choice(["VRd", "Rd"], n),
        "age": rng.normal(65, 10, n).astype(int),
        "sex": rng.choice(["M", "F"], n),
        "treatment_response": rng.binomial(1, 0.5, n),
    }).set_index("patient_id")


@pytest.fixture
def synthetic_expression_data():
    """Generate small synthetic expression DataFrame."""
    rng = np.random.RandomState(42)
    n, g = 100, 500
    genes = [f"GENE_{i:04d}" for i in range(g)]
    return pd.DataFrame(
        rng.lognormal(2, 1, (n, g)).clip(0),
        index=[f"PT_{i:03d}" for i in range(n)],
        columns=genes,
    )


@pytest.fixture
def small_survival_data():
    """Return (X, T, E) tuple for quick model tests."""
    rng = np.random.RandomState(42)
    n, p = 80, 20
    X = rng.randn(n, p).astype(np.float32)
    T = rng.weibull(1.5, n) * 30
    E = rng.binomial(1, 0.6, n)
    return X, T.astype(np.float64), E.astype(np.float64)


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Create temp directory with valid JSON configs."""
    configs = {
        "base": {"seed": 42, "device": "cpu", "log_level": "WARNING",
                 "data_dir": str(tmp_path / "data"), "results_dir": str(tmp_path / "results"),
                 "checkpoints_dir": str(tmp_path / "ckpt")},
        "ingest": {"demo_mode": True, "demo_n_patients": 50},
        "features": {"n_top_genes": 100, "n_pca_components": 10, "pathway_scoring": False},
        "cohort": {"test_size": 0.2, "val_size": 0.15, "cv_folds": 3},
        "train": {"enabled_models": ["cox_ph"], "n_epochs": 5, "batch_size": 32},
        "evaluate": {"bootstrap_n": 10, "time_horizons": [12, 24]},
        "interpret": {"shap_max_samples": 20, "top_k_genes": 10, "run_pathway_enrichment": False},
        "report": {"output_format": "html"},
        "autotune": {"n_trials": 2, "timeout_seconds": 30},
    }
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    for name, data in configs.items():
        with open(cfg_dir / f"{name}.json", "w") as f:
            json.dump(data, f)
    return str(cfg_dir)
