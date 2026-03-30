"""Random Survival Forest model wrapper."""

import logging
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RSFModel:
    """Random Survival Forest using scikit-survival."""

    def __init__(
        self, n_estimators: int = 100, max_depth: Optional[int] = None,
        min_samples_leaf: int = 15, max_features: str = "sqrt",
        random_state: int = 42, **kwargs,
    ):
        from sksurv.ensemble import RandomSurvivalForest
        self.model = RandomSurvivalForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, T: np.ndarray, E: np.ndarray) -> Dict:
        """Fit RSF on survival data."""
        # scikit-survival requires structured array
        y = np.array(
            [(bool(e), t) for e, t in zip(E, T)],
            dtype=[("event", bool), ("time", float)],
        )
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.model.fit(X_df, y)
        self._fitted = True

        c_index = self.model.score(X_df, y)
        logger.info(f"RSF fit: C-index={c_index:.4f}")
        return {"c_index": c_index}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return risk scores (higher = higher risk).

        sksurv's RSF.predict() already returns risk scores where higher = higher risk.
        No negation needed — survival_metrics.concordance_index handles the lifelines
        sign convention.
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_df)

    def predict_survival_function(
        self, X: np.ndarray, times: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Return survival function estimates."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        sfs = self.model.predict_survival_function(X_df)
        if times is not None:
            result = np.zeros((len(X_df), len(times)))
            for i, sf in enumerate(sfs):
                result[i] = np.interp(times, sf.x, sf.y)
            return result
        return sfs

    def feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.Series:
        """Return feature importance scores."""
        imp = self.model.feature_importances_
        if feature_names:
            return pd.Series(imp, index=feature_names).sort_values(ascending=False)
        return pd.Series(imp).sort_values(ascending=False)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
        self._fitted = True
