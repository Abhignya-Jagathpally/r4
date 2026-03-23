"""Statistical evaluation with DeLong test and BCa bootstrap CIs.

Fixes: M18 (paired DeLong), M25 (BCa bootstrap), M26 (IPCW bootstrap).
"""

import logging
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class MetricEvaluator:
    """Rigorous statistical evaluation for model comparison."""

    def delong_test(
        self, y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray,
    ) -> Dict:
        """DeLong test for comparing two AUCs with paired-sample covariance (M18).

        Implements DeLong et al (1988) with proper covariance estimation.
        """
        n1 = int((y_true == 1).sum())
        n0 = int((y_true == 0).sum())

        if n1 == 0 or n0 == 0:
            return {"z_stat": 0.0, "p_value": 1.0}

        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]

        # Placement values for each predictor
        V10_1 = np.zeros(n1)  # pred1 placements among negatives
        V10_2 = np.zeros(n1)
        V01_1 = np.zeros(n0)  # pred1 placements among positives
        V01_2 = np.zeros(n0)

        for i, pi in enumerate(pos_idx):
            V10_1[i] = np.mean(pred1[pi] > pred1[neg_idx]) + 0.5 * np.mean(pred1[pi] == pred1[neg_idx])
            V10_2[i] = np.mean(pred2[pi] > pred2[neg_idx]) + 0.5 * np.mean(pred2[pi] == pred2[neg_idx])

        for j, nj in enumerate(neg_idx):
            V01_1[j] = np.mean(pred1[pos_idx] > pred1[nj]) + 0.5 * np.mean(pred1[pos_idx] == pred1[nj])
            V01_2[j] = np.mean(pred2[pos_idx] > pred2[nj]) + 0.5 * np.mean(pred2[pos_idx] == pred2[nj])

        # AUCs
        auc1 = V10_1.mean()
        auc2 = V10_2.mean()

        # Covariance matrix (paired DeLong, M18)
        S10 = np.cov(np.stack([V10_1, V10_2]))
        S01 = np.cov(np.stack([V01_1, V01_2]))
        S = S10 / n1 + S01 / n0

        # Test statistic
        diff = auc1 - auc2
        var = S[0, 0] + S[1, 1] - 2 * S[0, 1]

        if var <= 0:
            return {"z_stat": 0.0, "p_value": 1.0, "auc1": auc1, "auc2": auc2}

        z = diff / np.sqrt(var)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        return {"z_stat": float(z), "p_value": float(p_value), "auc1": float(auc1), "auc2": float(auc2)}

    def bca_bootstrap_ci(
        self, metric_fn: Callable, *args,
        n_iterations: int = 2000, ci_level: float = 0.95, seed: int = 42,
    ) -> Dict:
        """BCa (bias-corrected and accelerated) bootstrap CI (M25).

        Uses jackknife acceleration constant for bounded metrics.
        """
        rng = np.random.RandomState(seed)
        n = len(args[0])

        # Bootstrap distribution
        boot_stats = []
        for _ in range(n_iterations):
            idx = rng.choice(n, size=n, replace=True)
            resampled = [a[idx] if isinstance(a, np.ndarray) else a for a in args]
            try:
                val = metric_fn(*resampled)
                if np.isfinite(val):
                    boot_stats.append(val)
            except Exception:
                continue

        if len(boot_stats) < 100:
            return {"point_estimate": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}

        boot_stats = np.array(boot_stats)
        theta_hat = metric_fn(*args)

        # Bias correction constant
        z0 = stats.norm.ppf(np.mean(boot_stats < theta_hat))

        # Acceleration constant (jackknife)
        jackknife_stats = np.zeros(n)
        for i in range(n):
            idx = np.concatenate([np.arange(i), np.arange(i + 1, n)])
            resampled = [a[idx] if isinstance(a, np.ndarray) else a for a in args]
            try:
                jackknife_stats[i] = metric_fn(*resampled)
            except Exception:
                jackknife_stats[i] = theta_hat

        jack_mean = jackknife_stats.mean()
        num = ((jack_mean - jackknife_stats) ** 3).sum()
        den = 6 * (((jack_mean - jackknife_stats) ** 2).sum()) ** 1.5
        a_hat = num / den if den != 0 else 0.0

        # BCa percentiles
        alpha = (1 - ci_level) / 2
        z_alpha = stats.norm.ppf(alpha)
        z_1alpha = stats.norm.ppf(1 - alpha)

        p_lower = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a_hat * (z0 + z_alpha)))
        p_upper = stats.norm.cdf(z0 + (z0 + z_1alpha) / (1 - a_hat * (z0 + z_1alpha)))

        p_lower = np.clip(p_lower, 0.001, 0.999)
        p_upper = np.clip(p_upper, 0.001, 0.999)

        return {
            "point_estimate": float(theta_hat),
            "ci_lower": float(np.percentile(boot_stats, p_lower * 100)),
            "ci_upper": float(np.percentile(boot_stats, p_upper * 100)),
            "std": float(boot_stats.std()),
            "method": "BCa",
        }

    def ipcw_bootstrap(
        self, metric_fn: Callable, T: np.ndarray, E: np.ndarray, *args,
        n_iterations: int = 1000, ci_level: float = 0.95, seed: int = 42,
    ) -> Dict:
        """IPCW-weighted bootstrap for survival CIs (M26)."""
        rng = np.random.RandomState(seed)
        n = len(T)

        # Compute IPCW weights using Kaplan-Meier of censoring
        from lifelines import KaplanMeierFitter
        kmf = KaplanMeierFitter()
        kmf.fit(T, 1 - E)  # Fit censoring distribution
        G = kmf.predict(T)
        G = np.clip(G.values if hasattr(G, "values") else G, 0.01, 1.0)
        weights = E / G  # IPCW weights

        boot_stats = []
        for _ in range(n_iterations):
            idx = rng.choice(n, size=n, replace=True, p=weights / weights.sum())
            try:
                resampled = [a[idx] if isinstance(a, np.ndarray) else a for a in args]
                val = metric_fn(T[idx], E[idx], *resampled)
                if np.isfinite(val):
                    boot_stats.append(val)
            except Exception:
                continue

        if not boot_stats:
            return {"point_estimate": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}

        boot_stats = np.array(boot_stats)
        alpha = (1 - ci_level) / 2
        return {
            "point_estimate": float(np.mean(boot_stats)),
            "ci_lower": float(np.percentile(boot_stats, alpha * 100)),
            "ci_upper": float(np.percentile(boot_stats, (1 - alpha) * 100)),
            "std": float(boot_stats.std()),
            "method": "IPCW_bootstrap",
        }
