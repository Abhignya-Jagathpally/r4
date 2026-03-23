"""Tests for baselines module (M35: fixed variable names)."""
import numpy as np
import pytest


def test_delong():
    from baselines.evaluation import MetricEvaluator
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    p1 = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9, 0.95])
    p2 = np.array([0.1, 0.2, 0.35, 0.55, 0.65, 0.75, 0.85, 0.9])
    ev = MetricEvaluator()
    result = ev.delong_test(y, p1, p2)
    assert "p_value" in result
    assert 0 <= result["p_value"] <= 1


def test_bca_bootstrap():
    from baselines.evaluation import MetricEvaluator
    rng = np.random.RandomState(42)
    data = rng.normal(5, 1, 100)
    ev = MetricEvaluator()
    result = ev.bca_bootstrap_ci(np.mean, data, n_iterations=200)
    # M35: FIXED variable names
    assert result["ci_lower"] <= result["point_estimate"]
    assert result["point_estimate"] <= result["ci_upper"]


def test_nested_cv():
    from baselines.biomarker_ranking import BiomarkerRanker
    rng = np.random.RandomState(42)
    X = rng.randn(100, 20)
    y = rng.binomial(1, 0.5, 100)
    ranker = BiomarkerRanker({"outer_folds": 3, "inner_folds": 2})
    result = ranker.nested_cv_rank(X, y)
    assert len(result) == 20
    assert "combined_score" in result.columns


def test_drug_sensitivity():
    from baselines.drug_sensitivity import DrugSensitivityPredictor
    rng = np.random.RandomState(42)
    X = rng.randn(50, 10)
    y = rng.randn(50)
    pred = DrugSensitivityPredictor({"outer_folds": 2, "inner_folds": 2})
    result = pred.predict(X, y)
    assert "elasticnet" in result
