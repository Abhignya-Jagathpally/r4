"""Tests for orchestration (M92: budget enforcement)."""
import numpy as np
import pytest


def test_budget_validation():
    from orchestration.agentic_tuning import AgenticTuner
    with pytest.raises(ValueError, match="budget"):
        AgenticTuner({})  # M85: both None should fail


def test_budget_experiments_enforced():
    """M92: Assert budget_experiments limit is actually enforced."""
    from orchestration.agentic_tuning import AgenticTuner
    tuner = AgenticTuner({"budget_experiments": 5, "n_restarts": 1})

    def dummy_objective(params):
        return np.random.random()

    result = tuner.tune(dummy_objective, {"x": (0.0, 1.0)})
    assert result["n_experiments"] <= 5  # M92: budget enforced


def test_convergence_detection():
    from orchestration.agentic_tuning import AgenticTuner
    tuner = AgenticTuner({"budget_experiments": 100, "convergence_window": 3,
                          "convergence_threshold": 0.01, "n_restarts": 1})

    def constant_objective(params):
        return 0.5  # Always same -> should converge

    result = tuner.tune(constant_objective, {"x": (0.0, 1.0)})
    assert result["n_experiments"] < 100  # Should stop early due to convergence


def test_reproducibility_weight_comparison():
    from orchestration.reproducibility import ReproducibilityManager
    rm = ReproducibilityManager()
    w1 = {"layer1": np.array([1.0, 2.0]), "layer2": np.array([3.0, 4.0])}
    w2 = {"layer1": np.array([1.0, 2.0]), "layer2": np.array([3.0, 4.0])}
    result = rm.validate_reproducibility(w1, w2)
    assert result["reproducible"] is True


def test_seed_propagation():
    from orchestration.reproducibility import ReproducibilityManager
    import os
    rm = ReproducibilityManager()
    rm.set_seeds(123)
    assert os.environ["PYTHONHASHSEED"] == "123"
    assert os.environ["NUMBA_DISABLE_JIT"] == "1"  # M90
