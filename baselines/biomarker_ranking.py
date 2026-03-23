"""Biomarker ranking with nested cross-validation.

Fixes: M16 (nested CV), M17 (test-fold AUC), M30 (no bare except), M33 (config values), M34 (SHAP).
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class BiomarkerRanker:
    """Biomarker discovery with proper nested CV evaluation."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.outer_folds = self.config.get("outer_folds", 5)  # M33: from config
        self.inner_folds = self.config.get("inner_folds", 3)
        self.l1_ratios = self.config.get("l1_ratios", [0.1, 0.5, 0.7, 0.9, 0.95])
        self.random_state = self.config.get("random_state", 42)

    def nested_cv_rank(
        self, X: np.ndarray, y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Nested CV: inner for HP tuning, outer for evaluation (M16, M17)."""
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        outer_cv = StratifiedKFold(n_splits=self.outer_folds, shuffle=True, random_state=self.random_state)
        inner_cv = StratifiedKFold(n_splits=self.inner_folds, shuffle=True, random_state=self.random_state + 1)

        fold_aucs = []
        fold_coefs = []
        all_shap_values = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Inner CV for HP tuning (M16)
            try:
                en_cv = ElasticNetCV(
                    l1_ratio=self.l1_ratios,
                    cv=inner_cv.split(X_train, y_train),
                    random_state=self.random_state,
                    max_iter=5000,
                )
                en_cv.fit(X_train, y_train)
                best_l1 = en_cv.l1_ratio_
                best_alpha = en_cv.alpha_
            except ValueError as e:
                logger.warning(f"Inner CV fold {fold_idx} failed: {e}")  # M30: specific exception
                best_l1 = 0.5
                best_alpha = 0.1

            # Refit on full training fold
            model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1, max_iter=5000, random_state=self.random_state)
            model.fit(X_train, y_train)

            # Evaluate on TEST fold (M17: not training AUC)
            y_pred = model.predict(X_test)
            try:
                auc = roc_auc_score(y_test, y_pred)
            except ValueError as e:
                logger.warning(f"AUC computation failed fold {fold_idx}: {e}")  # M30
                auc = 0.5
            fold_aucs.append(auc)
            fold_coefs.append(np.abs(model.coef_))

            # SHAP explanations (M34: actually called)
            try:
                import shap
                explainer = shap.LinearExplainer(model, X_train)
                shap_vals = explainer.shap_values(X_test)
                all_shap_values.append(np.abs(shap_vals).mean(axis=0))
            except ImportError:
                logger.debug("SHAP not available, skipping explanations")
            except Exception as e:
                logger.warning(f"SHAP failed fold {fold_idx}: {e}")  # M30

            logger.debug(f"Outer fold {fold_idx}: test AUC={auc:.4f}, l1={best_l1}, alpha={best_alpha:.4f}")

        # Aggregate
        mean_coefs = np.mean(fold_coefs, axis=0) if fold_coefs else np.zeros(X.shape[1])
        mean_shap = np.mean(all_shap_values, axis=0) if all_shap_values else mean_coefs

        result = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_coef": mean_coefs,
            "mean_abs_shap": mean_shap,
            "combined_score": 0.5 * mean_coefs / (mean_coefs.max() + 1e-10) + 0.5 * mean_shap / (mean_shap.max() + 1e-10),
        }).sort_values("combined_score", ascending=False).set_index("feature")

        logger.info(
            f"Nested CV complete: mean test AUC={np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}, "
            f"top feature={result.index[0]}"
        )
        return result
