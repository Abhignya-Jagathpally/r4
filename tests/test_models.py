"""Test model training, prediction, and save/load."""

import numpy as np
import pandas as pd
import pytest


def test_cox_ph(small_survival_data, tmp_path):
    from pipeline4.models.cox_ph import CoxPHModel
    X, T, E = small_survival_data
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

    model = CoxPHModel(penalizer=1.0)
    metrics = model.fit(X_df, T, E)
    assert "c_index" in metrics
    assert 0 < metrics["c_index"] < 1

    risk = model.predict(X_df)
    assert risk.shape == (len(X),)

    model.save(str(tmp_path / "cox.pkl"))
    model2 = CoxPHModel()
    model2.load(str(tmp_path / "cox.pkl"))
    risk2 = model2.predict(X_df)
    np.testing.assert_array_almost_equal(risk, risk2)


def test_deepsurv(small_survival_data, tmp_path):
    from pipeline4.models.deepsurv import DeepSurvModel
    X, T, E = small_survival_data

    model = DeepSurvModel(input_dim=X.shape[1], hidden_dims=[32, 16], lr=1e-3)
    metrics = model.fit(X, T, E, n_epochs=5, batch_size=32, patience=3)
    assert "val_c_index" in metrics or "n_epochs_trained" in metrics

    risk = model.predict(X)
    assert risk.shape == (len(X),)

    model.save(str(tmp_path / "deepsurv.pt"))
    model2 = DeepSurvModel()
    model2.load(str(tmp_path / "deepsurv.pt"))
    risk2 = model2.predict(X)
    assert risk2.shape == risk.shape


def test_rsf(small_survival_data, tmp_path):
    from pipeline4.models.rsf import RSFModel
    X, T, E = small_survival_data

    model = RSFModel(n_estimators=10, min_samples_leaf=5)
    metrics = model.fit(X, T, E)
    assert "c_index" in metrics

    risk = model.predict(X)
    assert risk.shape == (len(X),)

    imp = model.feature_importance()
    assert len(imp) == X.shape[1]


def test_treatment_classifier(small_survival_data, tmp_path):
    from pipeline4.models.treatment_classifier import TreatmentResponseClassifier
    X, T, E = small_survival_data
    y = (T > np.median(T)).astype(int)

    model = TreatmentResponseClassifier(model_type="xgboost")
    metrics = model.fit(X, y)
    assert "train_auroc" in metrics

    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
