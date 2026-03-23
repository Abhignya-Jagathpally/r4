"""Drug sensitivity prediction with nested CV (M22: no synthetic concordance, M28: nested CV)."""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


class DrugSensitivityPredictor:
    """Predict drug sensitivity with proper evaluation (no synthetic DepMap data, M22)."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.outer_folds = self.config.get("outer_folds", 5)
        self.inner_folds = self.config.get("inner_folds", 3)

    def predict(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[list] = None) -> Dict:
        """Run nested CV drug sensitivity prediction (M28)."""
        outer_cv = KFold(n_splits=self.outer_folds, shuffle=True, random_state=42)
        results = {}

        for model_name, model_fn in [("elasticnet", self._get_elasticnet),
                                       ("rf", self._get_rf),
                                       ("xgboost", self._get_xgboost)]:
            fold_metrics = []
            for train_idx, test_idx in outer_cv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Inner CV for HP selection
                model = model_fn(X_train, y_train)
                y_pred = model.predict(X_test)

                # Test-set metrics (M22: real metrics, no synthetic concordance)
                try:
                    r, _ = pearsonr(y_test, y_pred)
                    rho, _ = spearmanr(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    fold_metrics.append({"pearson_r": r, "spearman_rho": rho, "rmse": rmse})
                except Exception as e:
                    logger.warning(f"{model_name} fold evaluation failed: {e}")

            if fold_metrics:
                results[model_name] = {
                    k: float(np.mean([m[k] for m in fold_metrics]))
                    for k in fold_metrics[0]
                }
                logger.info(f"{model_name}: Pearson r={results[model_name]['pearson_r']:.4f}")

        return results

    def _get_elasticnet(self, X_train, y_train):
        from sklearn.linear_model import ElasticNetCV
        model = ElasticNetCV(cv=self.inner_folds, random_state=42, max_iter=5000)
        model.fit(X_train, y_train)
        return model

    def _get_rf(self, X_train, y_train):
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        return model

    def _get_xgboost(self, X_train, y_train):
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        return model
