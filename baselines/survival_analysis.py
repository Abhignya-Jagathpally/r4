"""Survival analysis with Uno's td-AUC, IPCW Brier score, Schoenfeld test.

Fixes: M19 (Uno td-AUC), M20 (IPCW Brier), M21 (test-set C-index), M24 (Schoenfeld).
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

logger = logging.getLogger(__name__)


class SurvivalAnalyzer:
    """Survival analysis with proper test-set evaluation."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    def compute_c_index(
        self, T_test: np.ndarray, E_test: np.ndarray, risk_test: np.ndarray,
    ) -> float:
        """C-index on TEST set (M21: not training set)."""
        ci = concordance_index(T_test, -risk_test, E_test)
        logger.info(f"Test C-index: {ci:.4f}")
        return float(ci)

    def uno_td_auc(
        self, T_train: np.ndarray, E_train: np.ndarray,
        T_test: np.ndarray, E_test: np.ndarray,
        risk_scores: np.ndarray, times: List[float],
    ) -> Dict:
        """Uno's time-dependent AUC with IPCW weights (M19, Uno et al 2011)."""
        from sksurv.metrics import cumulative_dynamic_auc

        y_train = np.array([(bool(e), t) for e, t in zip(E_train, T_train)],
                           dtype=[("event", bool), ("time", float)])
        y_test = np.array([(bool(e), t) for e, t in zip(E_test, T_test)],
                          dtype=[("event", bool), ("time", float)])

        max_time = min(T_train.max(), T_test.max()) * 0.9
        valid_times = sorted([t for t in times if T_test.min() < t < max_time])

        if not valid_times:
            logger.warning("No valid time horizons for Uno td-AUC")
            return {"mean_auc": float("nan")}

        aucs, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, valid_times)
        result = {"mean_auc": float(mean_auc)}
        for t, auc in zip(valid_times, aucs):
            result[f"auc_t{int(t)}"] = float(auc)
        return result

    def ipcw_brier_score(
        self, T_train: np.ndarray, E_train: np.ndarray,
        T_test: np.ndarray, E_test: np.ndarray,
        surv_probs: np.ndarray, times: List[float],
    ) -> Dict:
        """IPCW Brier score (M20, Gerds & Schumacher 2006)."""
        from sksurv.metrics import brier_score as bs_fn

        y_train = np.array([(bool(e), t) for e, t in zip(E_train, T_train)],
                           dtype=[("event", bool), ("time", float)])
        y_test = np.array([(bool(e), t) for e, t in zip(E_test, T_test)],
                          dtype=[("event", bool), ("time", float)])

        valid_times = sorted([t for t in times if T_test.min() < t < T_test.max() * 0.9])
        if not valid_times:
            return {"integrated_brier": float("nan")}

        _, bs = bs_fn(y_train, y_test, surv_probs, np.array(valid_times))
        result = {"integrated_brier": float(np.trapz(bs, valid_times) / (valid_times[-1] - valid_times[0]))}
        for t, s in zip(valid_times, bs):
            result[f"brier_t{int(t)}"] = float(s)
        return result

    def schoenfeld_test(
        self, X: pd.DataFrame, T: np.ndarray, E: np.ndarray,
    ) -> Dict:
        """Schoenfeld residuals test for PH assumption (M24, Grambsch & Therneau 1994)."""
        df = X.copy()
        df["T"] = T
        df["E"] = E

        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(df, duration_col="T", event_col="E")

        results = cph.check_assumptions(df, p_value_threshold=0.05, show_plots=False)

        # Parse results
        violations = {}
        if results is not None:
            for test_name, test_result in results:
                if hasattr(test_result, "iloc"):
                    for _, row in test_result.iterrows():
                        if row.get("p", 1.0) < 0.05:
                            violations[row.get("variable", test_name)] = float(row["p"])

        ph_holds = len(violations) == 0
        logger.info(f"Schoenfeld test: PH {'holds' if ph_holds else 'violated'} ({len(violations)} violations)")
        return {"ph_holds": ph_holds, "violations": violations, "n_violations": len(violations)}

    def fit_cox(self, X_train: pd.DataFrame, T_train: np.ndarray, E_train: np.ndarray) -> CoxPHFitter:
        """Fit Cox PH model on training data."""
        df = X_train.copy()
        df["T"] = T_train
        df["E"] = E_train
        cph = CoxPHFitter(penalizer=self.config.get("penalizer", 0.1))
        cph.fit(df, duration_col="T", event_col="E")
        return cph
