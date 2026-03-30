"""Fairness audit across demographic subgroups."""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FairnessAuditor:
    """Evaluate model fairness across protected attribute subgroups."""

    def __init__(self, protected_attributes: List[str]):
        self.protected_attributes = protected_attributes

    def survival_fairness(
        self, T: np.ndarray, E: np.ndarray,
        risk_scores: np.ndarray, groups: pd.Series,
    ) -> Dict:
        """Compute C-index per subgroup."""
        from lifelines.utils import concordance_index as ci_fn
        result = {}
        for group_val in groups.unique():
            mask = groups.values == group_val
            n_group = int(mask.sum())
            n_events = int(E[mask].sum())
            if n_group < 10:
                continue
            if n_events < 2:
                logger.warning(
                    f"Subgroup '{group_val}': only {n_events} events in {n_group} patients — "
                    f"C-index undefined, skipping"
                )
                result[str(group_val)] = {"c_index": float("nan"), "n": n_group, "n_events": n_events}
                continue
            try:
                ci = ci_fn(T[mask], -risk_scores[mask], E[mask])
                result[str(group_val)] = {"c_index": float(ci), "n": n_group, "n_events": n_events}
            except Exception:
                result[str(group_val)] = {"c_index": float("nan"), "n": n_group, "n_events": n_events}

        # Compute disparity
        cis = [v["c_index"] for v in result.values() if np.isfinite(v["c_index"])]
        if len(cis) >= 2:
            result["disparity"] = float(max(cis) - min(cis))

        return result

    def demographic_parity(
        self, y_pred: np.ndarray, groups: pd.Series,
    ) -> Dict:
        """Positive prediction rate per group."""
        result = {}
        for group_val in groups.unique():
            mask = groups.values == group_val
            result[str(group_val)] = float(y_pred[mask].mean())
        return result

    def equalized_odds(
        self, y_true: np.ndarray, y_pred: np.ndarray, groups: pd.Series,
    ) -> Dict:
        """TPR and FPR per group."""
        result = {}
        for group_val in groups.unique():
            mask = groups.values == group_val
            yt, yp = y_true[mask], y_pred[mask]
            tp = ((yt == 1) & (yp == 1)).sum()
            fn = ((yt == 1) & (yp == 0)).sum()
            fp = ((yt == 0) & (yp == 1)).sum()
            tn = ((yt == 0) & (yp == 0)).sum()
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            result[str(group_val)] = {"tpr": float(tpr), "fpr": float(fpr)}
        return result
