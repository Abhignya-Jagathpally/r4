"""Tests for data_pipeline module."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    rng = np.random.RandomState(42)
    return pd.DataFrame(rng.lognormal(20, 2, (50, 100)),
                        index=[f"S{i}" for i in range(50)],
                        columns=[f"P{i}" for i in range(100)])


def test_normalization_fit_transform(sample_data):
    from data_pipeline.normalization import NormalizationPipeline
    norm = NormalizationPipeline()
    norm.fit(sample_data)
    assert norm.is_fitted
    result = norm.transform(sample_data)
    assert result.shape == sample_data.shape


def test_median_centering_postcondition(sample_data):
    from data_pipeline.normalization import NormalizationPipeline
    norm = NormalizationPipeline()
    centered = norm.median_centering(sample_data)
    medians = centered.median(axis=1)
    assert (medians.abs() < 1e-10).all()


def test_missingness_classify(sample_data):
    from data_pipeline.missingness import MissingnessAnalyzer
    df = sample_data.copy()
    df.iloc[:10, :20] = np.nan
    analyzer = MissingnessAnalyzer()
    result = analyzer.classify_mechanism(df)
    assert "mechanism" in result
    assert result["mechanism"] in ("MCAR", "MAR", "MNAR", "complete")


def test_missingness_impute(sample_data):
    from data_pipeline.missingness import MissingnessAnalyzer
    df = sample_data.copy()
    df.iloc[:10, :20] = np.nan
    analyzer = MissingnessAnalyzer()
    imputed = analyzer.impute(df, mechanism="MCAR")
    assert imputed.isna().sum().sum() == 0


def test_pathway_gsva(sample_data):
    from data_pipeline.pathway_aggregation import PathwayScorer
    scorer = PathwayScorer()
    gene_sets = {"test_set": sample_data.columns[:10].tolist()}
    scores = scorer.gsva_score(sample_data, gene_sets)
    assert scores.shape[0] == sample_data.shape[0]
