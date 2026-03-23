"""Tests for DepMap benchmark (M69)."""
import numpy as np
import pandas as pd
import pytest


def test_study_level_splits():
    from depmap_benchmark.benchmark_definition import BenchmarkSuite
    df = pd.DataFrame({"study": ["A"]*20 + ["B"]*20 + ["C"]*20, "y": range(60)})
    suite = BenchmarkSuite()
    splits = suite.study_level_splits(df, "study", n_splits=3)
    assert len(splits) >= 2
    for train_idx, test_idx in splits:
        train_studies = set(df.iloc[train_idx]["study"])
        test_studies = set(df.iloc[test_idx]["study"])
        assert train_studies.isdisjoint(test_studies)


def test_evaluate_real_metrics():
    from depmap_benchmark.benchmark_definition import BenchmarkSuite
    y_true = np.array([1, 2, 3, 4, 5], dtype=float)
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.0], dtype=float)
    suite = BenchmarkSuite()
    metrics = suite.evaluate(y_true, y_pred)
    assert metrics["pearson_r"] > 0.9
    assert metrics["rmse"] < 0.5


def test_mm_filter():
    from depmap_benchmark.depmap_loader import DepMapLoader
    loader = DepMapLoader()
    df = pd.DataFrame(np.random.randn(50, 10),
                       index=["U266", "MM1S", "RPMI8226", "HeLa", "MCF7"] + [f"other_{i}" for i in range(45)])
    filtered = loader.filter_to_mm_lines(df)
    assert len(filtered) >= 3
    assert "HeLa" not in filtered.index


def test_pathway_recovery():
    from depmap_benchmark.mm_pathway_oracle import MMPathwayOracle
    oracle = MMPathwayOracle()
    result = oracle.pathway_recovery_score(["NFKB1", "RELA", "TNF", "FAKE1"], "NFKB_SIGNALING")
    assert result["sensitivity"] > 0
    assert result["n_recovered"] == 3
