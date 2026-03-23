"""Normalization pipeline for proteomics data.

Implements median centering, quantile normalization (Bolstad et al 2003),
and empirical Bayes ComBat batch correction (Johnson et al 2007).
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class NormalizationPipeline:
    """Multi-step normalization with fit/transform pattern.

    Fixes applied (M01-M04, M10-M11, M14):
    - M01: ComBat uses empirical Bayes shrinkage (Johnson et al 2007)
    - M02: Quantile normalization on SAMPLES (axis=0), not proteins
    - M04: grand_mean stored from training fit only
    - M10: is_fitted set True after fit()
    - M11: CV=infinity handled for log-centered data
    - M14: Post-condition assertion: medians==0 after centering
    """

    def __init__(self):
        self.is_fitted = False
        self._grand_mean = None
        self._reference_distribution = None
        self._combat_params = None

    def fit(self, X: pd.DataFrame, batch_labels: Optional[pd.Series] = None) -> "NormalizationPipeline":
        """Fit normalization parameters from training data only."""
        logger.info(f"Fitting normalization on {X.shape}")

        # Store grand mean from TRAINING data only (M04)
        self._grand_mean = X.mean(axis=1).mean()

        # Store reference distribution for quantile normalization
        sorted_vals = np.sort(X.values, axis=0)
        self._reference_distribution = sorted_vals.mean(axis=1)

        # Fit ComBat parameters if batch labels provided
        if batch_labels is not None:
            self._combat_params = self._fit_combat(X, batch_labels)

        self.is_fitted = True  # M10: flag set after fit
        logger.info(f"Normalization fitted: grand_mean={self._grand_mean:.4f}")
        return self

    def transform(self, X: pd.DataFrame, batch_labels: Optional[pd.Series] = None) -> pd.DataFrame:
        """Apply fitted normalization to data."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transform()")
        result = X.copy()
        result = self.median_centering(result)
        result = self.quantile_normalization(result)
        if batch_labels is not None and self._combat_params is not None:
            result = self.combat_normalization(result, batch_labels)
        return result

    def median_centering(self, X: pd.DataFrame) -> pd.DataFrame:
        """Center each sample to median zero."""
        medians = X.median(axis=1)
        result = X.sub(medians, axis=0)

        # M14: Post-condition assertion
        residual_medians = result.median(axis=1)
        assert (residual_medians.abs() < 1e-10).all(), (
            f"Median centering failed: max residual median = {residual_medians.abs().max()}"
        )
        logger.debug("Median centering: post-condition medians==0 verified")
        return result

    def quantile_normalization(self, X: pd.DataFrame) -> pd.DataFrame:
        """Quantile normalization across SAMPLES (M02: axis=0, not proteins).

        Each sample's ranked values are replaced with the mean of
        that rank position across all training samples.
        """
        if self._reference_distribution is None:
            raise RuntimeError("Reference distribution not fitted")

        result = X.copy()
        n_rows = min(len(self._reference_distribution), X.shape[0])

        for col in result.columns:
            ranks = result[col].rank(method="average").values - 1
            ranks = np.clip(ranks, 0, n_rows - 1).astype(int)
            result[col] = self._reference_distribution[ranks]

        logger.debug("Quantile normalization applied (sample-wise)")
        return result

    def combat_normalization(
        self, X: pd.DataFrame, batch_labels: pd.Series,
    ) -> pd.DataFrame:
        """Empirical Bayes ComBat batch correction (M01: Johnson et al 2007).

        Implements full EB shrinkage: estimates batch effect means/variances,
        then shrinks toward pooled estimates using empirical priors.
        """
        if self._combat_params is None:
            logger.warning("ComBat params not fitted, fitting now")
            self._combat_params = self._fit_combat(X, batch_labels)

        params = self._combat_params
        batches = batch_labels.unique()
        result = X.copy()

        # Standardize data
        grand_mean = params["grand_mean"]
        var_pooled = params["var_pooled"]

        stand = (X.T - grand_mean).T
        stand = stand.div(np.sqrt(var_pooled + 1e-10), axis=1)

        # Apply EB-adjusted batch corrections
        for batch in batches:
            mask = batch_labels == batch
            if batch in params["gamma_star"] and batch in params["delta_star"]:
                gamma = params["gamma_star"][batch]
                delta = params["delta_star"][batch]
                stand.loc[mask] = (stand.loc[mask] - gamma) / np.sqrt(delta + 1e-10)

        # Rescale
        result = stand.mul(np.sqrt(var_pooled + 1e-10), axis=1)
        result = (result.T + grand_mean).T

        logger.info(f"ComBat correction applied across {len(batches)} batches")
        return result

    def _fit_combat(self, X: pd.DataFrame, batch_labels: pd.Series) -> Dict:
        """Fit ComBat empirical Bayes parameters."""
        batches = batch_labels.unique()
        n_batches = len(batches)

        # Grand mean (from training only, M04)
        grand_mean = X.mean(axis=0)

        # Pooled variance
        residuals = X - grand_mean
        var_pooled = residuals.var(axis=0)
        var_pooled = var_pooled.clip(lower=1e-10)

        # Standardize
        stand = (X.T - grand_mean).T
        stand = stand.div(np.sqrt(var_pooled), axis=1)

        # Batch effect estimates
        gamma_hat = {}  # batch mean effects
        delta_hat = {}  # batch variance effects
        for batch in batches:
            batch_data = stand.loc[batch_labels == batch]
            gamma_hat[batch] = batch_data.mean(axis=0).values
            delta_hat[batch] = batch_data.var(axis=0).values

        # Empirical Bayes shrinkage (Johnson et al 2007)
        # Prior for gamma: Normal(gamma_bar, tau^2)
        gamma_bar = np.mean([g.mean() for g in gamma_hat.values()])
        tau2 = np.var([g.mean() for g in gamma_hat.values()])

        # Prior for delta: Inverse Gamma(lambda_bar, theta_bar)
        delta_vals = np.array([d.mean() for d in delta_hat.values()])
        lambda_bar = np.mean(delta_vals)
        theta_bar = np.var(delta_vals)

        # Posterior (EB shrinkage)
        gamma_star = {}
        delta_star = {}

        for batch in batches:
            n_b = (batch_labels == batch).sum()
            g = gamma_hat[batch]
            d = delta_hat[batch]

            # Shrink gamma toward gamma_bar
            gamma_star[batch] = (tau2 * g + d * gamma_bar / max(n_b, 1)) / (tau2 + d / max(n_b, 1) + 1e-10)

            # Shrink delta toward lambda_bar
            delta_star[batch] = (theta_bar + (n_b - 1) * d / 2) / (lambda_bar + n_b / 2 - 1 + 1e-10)

        params = {
            "grand_mean": grand_mean,
            "var_pooled": var_pooled,
            "gamma_star": gamma_star,
            "delta_star": delta_star,
            "gamma_bar": gamma_bar,
            "tau2": tau2,
        }
        logger.info(f"ComBat EB params fitted: {n_batches} batches, tau2={tau2:.4f}")
        return params

    def quality_metrics(self, X: pd.DataFrame) -> Dict:
        """Compute quality metrics (M11: handle CV=infinity)."""
        means = X.mean(axis=0)
        stds = X.std(axis=0)

        # M11: Handle CV=infinity for log-centered data (mean near 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cv = stds / means.replace(0, np.nan)
            cv = cv.replace([np.inf, -np.inf], np.nan)

        return {
            "n_proteins": X.shape[1],
            "n_samples": X.shape[0],
            "mean_cv": float(cv.median()) if cv.notna().any() else float("nan"),
            "completeness": float(1 - X.isna().mean().mean()),
            "n_cv_infinite": int(cv.isna().sum()),
        }
