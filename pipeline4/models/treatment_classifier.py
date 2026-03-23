"""Treatment response classifier (XGBoost/LightGBM)."""

import logging
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TreatmentResponseClassifier:
    """Binary classifier for treatment response prediction."""

    def __init__(self, model_type: str = "xgboost", params: Optional[Dict] = None, **kwargs):
        self.model_type = model_type
        self.params = params or {}
        self.model = None
        self._fitted = False
        self._init_model()

    def _init_model(self) -> None:
        if self.model_type == "xgboost":
            import xgboost as xgb
            defaults = {
                "n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
                "subsample": 0.8, "colsample_bytree": 0.8, "eval_metric": "logloss",
                "random_state": 42, "use_label_encoder": False,
            }
            defaults.update(self.params)
            self.model = xgb.XGBClassifier(**defaults)
        elif self.model_type == "lightgbm":
            import lightgbm as lgb
            defaults = {
                "n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
                "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42,
                "verbose": -1,
            }
            defaults.update(self.params)
            self.model = lgb.LGBMClassifier(**defaults)
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(random_state=42, **self.params)

    def fit(
        self, X: np.ndarray, y: np.ndarray,
        val_X: Optional[np.ndarray] = None, val_y: Optional[np.ndarray] = None,
    ) -> Dict:
        """Fit classifier with optional early stopping."""
        fit_params = {}
        if val_X is not None and self.model_type in ("xgboost", "lightgbm"):
            fit_params["eval_set"] = [(val_X, val_y)]
            if self.model_type == "xgboost":
                fit_params["verbose"] = False
            elif self.model_type == "lightgbm":
                pass  # LightGBM handles via callbacks

        self.model.fit(X, y, **fit_params)
        self._fitted = True

        from sklearn.metrics import roc_auc_score, average_precision_score
        y_prob = self.model.predict_proba(X)[:, 1]
        metrics = {
            "train_auroc": float(roc_auc_score(y, y_prob)),
            "train_auprc": float(average_precision_score(y, y_prob)),
        }
        logger.info(f"Classifier fit: AUROC={metrics['train_auroc']:.4f}")
        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def feature_importance(self, feature_names: Optional[list] = None) -> pd.Series:
        imp = self.model.feature_importances_
        if feature_names:
            return pd.Series(imp, index=feature_names).sort_values(ascending=False)
        return pd.Series(imp).sort_values(ascending=False)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
        self._fitted = True
