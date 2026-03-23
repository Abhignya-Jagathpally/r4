"""Test evaluation metrics."""

import numpy as np
import pytest


def test_concordance_index_perfect():
    from pipeline4.evaluation.survival_metrics import concordance_index
    T = np.array([10, 20, 30, 40, 50], dtype=float)
    E = np.array([1, 1, 1, 1, 1], dtype=float)
    risk = np.array([50, 40, 30, 20, 10], dtype=float)  # Higher risk = shorter time
    ci = concordance_index(T, E, risk)
    assert ci > 0.9


def test_concordance_index_random():
    from pipeline4.evaluation.survival_metrics import concordance_index
    rng = np.random.RandomState(42)
    T = rng.exponential(30, 100)
    E = rng.binomial(1, 0.6, 100).astype(float)
    risk = rng.randn(100)
    ci = concordance_index(T, E, risk)
    assert 0.3 < ci < 0.7  # Random predictor near 0.5


def test_bootstrap_ci():
    from pipeline4.evaluation.bootstrap import bootstrap_ci
    from pipeline4.evaluation.survival_metrics import concordance_index
    rng = np.random.RandomState(42)
    T = rng.exponential(30, 50)
    E = rng.binomial(1, 0.6, 50).astype(float)
    risk = -T + rng.randn(50) * 5  # Correlated with time
    result = bootstrap_ci(concordance_index, T, E, risk, n_iterations=50)
    assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]


def test_classification_report():
    from pipeline4.evaluation.classification_metrics import full_classification_report
    y_true = np.array([0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.3, 0.6, 0.8, 0.9])
    report = full_classification_report(y_true, y_prob)
    assert "auroc" in report
    assert "f1" in report
    assert report["auroc"] > 0.5
