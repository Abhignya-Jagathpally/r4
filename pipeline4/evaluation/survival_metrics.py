"""Survival analysis metrics."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def concordance_index(
    T: np.ndarray, E: np.ndarray, risk_scores: np.ndarray,
) -> float:
    """Compute Harrell's concordance index.

    Sign convention:
        All models return risk_scores where **higher = higher risk**.
        lifelines.concordance_index expects predicted_scores where
        **higher = longer survival (lower risk)**, so we negate once here.
        Models must NOT pre-negate their scores.
    """
    from lifelines.utils import concordance_index as ci_fn
    return ci_fn(T, -risk_scores, E)


def time_dependent_auc(
    T_train: np.ndarray, E_train: np.ndarray,
    T_test: np.ndarray, E_test: np.ndarray,
    risk_scores: np.ndarray, times: List[float],
) -> Dict[str, float]:
    """Compute time-dependent AUC using IPCW estimator."""
    from sksurv.metrics import cumulative_dynamic_auc

    y_train = np.array(
        [(bool(e), t) for e, t in zip(E_train, T_train)],
        dtype=[("event", bool), ("time", float)],
    )
    y_test = np.array(
        [(bool(e), t) for e, t in zip(E_test, T_test)],
        dtype=[("event", bool), ("time", float)],
    )

    # Filter times within range
    max_time = min(T_train.max(), T_test.max()) * 0.9
    valid_times = [t for t in times if t < max_time and t > T_test.min()]

    if not valid_times:
        logger.warning("No valid time horizons for td-AUC")
        return {}

    aucs, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, valid_times)
    result = {"mean_auc": float(mean_auc)}
    for t, auc in zip(valid_times, aucs):
        result[f"auc_t{int(t)}"] = float(auc)
    return result


def brier_score(
    T_train: np.ndarray, E_train: np.ndarray,
    T_test: np.ndarray, E_test: np.ndarray,
    survival_probs: np.ndarray, times: List[float],
) -> Dict[str, float]:
    """Compute integrated Brier score."""
    from sksurv.metrics import brier_score as bs_fn

    y_train = np.array(
        [(bool(e), t) for e, t in zip(E_train, T_train)],
        dtype=[("event", bool), ("time", float)],
    )
    y_test = np.array(
        [(bool(e), t) for e, t in zip(E_test, T_test)],
        dtype=[("event", bool), ("time", float)],
    )

    valid_times = [t for t in times if t < T_test.max() * 0.9]
    if not valid_times:
        return {}

    _, bs = bs_fn(y_train, y_test, survival_probs, np.array(valid_times))
    result = {}
    for t, score in zip(valid_times, bs):
        result[f"brier_t{int(t)}"] = float(score)
    result["integrated_brier"] = float(np.trapz(bs, valid_times) / (valid_times[-1] - valid_times[0]))
    return result
