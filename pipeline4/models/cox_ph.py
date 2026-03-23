"""Cox Proportional Hazards model wrapper."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CoxPHModel:
    """Cox PH survival model using lifelines."""

    def __init__(self, penalizer: float = 0.1, l1_ratio: float = 0.0, **kwargs):
        from lifelines import CoxPHFitter
        self.fitter = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self._fitted = False

    def fit(self, X: pd.DataFrame, T: np.ndarray, E: np.ndarray) -> Dict:
        """Fit Cox PH model."""
        df = X.copy()
        df["T"] = T
        df["E"] = E

        self.fitter.fit(df, duration_col="T", event_col="E")
        self._fitted = True

        c_index = self.fitter.concordance_index_
        logger.info(f"CoxPH fit: C-index={c_index:.4f}, penalizer={self.penalizer}")
        return {"c_index": c_index, "log_likelihood": float(self.fitter.log_likelihood_)}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return partial hazard scores (higher = higher risk)."""
        return self.fitter.predict_partial_hazard(X).values.flatten()

    def predict_survival_function(
        self, X: pd.DataFrame, times: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """Return survival probabilities at specified times."""
        sf = self.fitter.predict_survival_function(X)
        if times is not None:
            sf = sf.loc[sf.index.isin(times) | True]  # Interpolate to nearest
        return sf

    def get_coefficients(self) -> pd.DataFrame:
        """Return feature coefficients with statistics."""
        return self.fitter.summary

    def save(self, path: str) -> None:
        joblib.dump({"fitter": self.fitter, "config": {"penalizer": self.penalizer}}, path)

    def load(self, path: str) -> None:
        state = joblib.load(path)
        self.fitter = state["fitter"]
        self._fitted = True
