"""Bootstrap confidence intervals for metrics."""

import logging
from typing import Any, Callable, Dict

import numpy as np

logger = logging.getLogger(__name__)


def bootstrap_ci(
    metric_fn: Callable, *args,
    n_iterations: int = 1000, ci_level: float = 0.95, random_state: int = 42,
    point_estimate: Optional[float] = None,
) -> Dict[str, float]:
    """Compute bootstrap confidence interval for a metric function.

    metric_fn is called as metric_fn(*resampled_args) where each arg
    is resampled with replacement along axis 0.

    If point_estimate is provided, it is used directly instead of
    recomputing metric_fn on the full data.
    """
    rng = np.random.RandomState(random_state)
    n = len(args[0])
    scores = []

    for _ in range(n_iterations):
        idx = rng.choice(n, size=n, replace=True)
        resampled = [a[idx] if isinstance(a, np.ndarray) else a for a in args]
        try:
            score = metric_fn(*resampled)
            if np.isfinite(score):
                scores.append(score)
        except Exception:
            continue

    if not scores:
        return {"point_estimate": float("nan"), "ci_lower": float("nan"),
                "ci_upper": float("nan"), "std": float("nan")}

    if point_estimate is None:
        point_estimate = float(metric_fn(*args))

    scores = np.array(scores)
    alpha = (1 - ci_level) / 2
    return {
        "point_estimate": point_estimate,
        "ci_lower": float(np.percentile(scores, alpha * 100)),
        "ci_upper": float(np.percentile(scores, (1 - alpha) * 100)),
        "std": float(scores.std()),
    }
