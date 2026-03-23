"""Quality reporting for proteomics data."""

import logging
from typing import Dict
import warnings
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class QualityReporter:
    """Generate quality metrics for proteomics data."""

    def generate_report(self, df: pd.DataFrame) -> Dict:
        """Comprehensive quality report with CV=inf handling (M11)."""
        means = df.mean(axis=0)
        stds = df.std(axis=0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = stds / means.replace(0, np.nan)
            cv = cv.replace([np.inf, -np.inf], np.nan)

        missing_per_protein = df.isna().mean(axis=0)
        missing_per_sample = df.isna().mean(axis=1)

        return {
            "n_samples": df.shape[0],
            "n_proteins": df.shape[1],
            "completeness": float(1 - df.isna().mean().mean()),
            "median_cv": float(cv.median()) if cv.notna().any() else None,
            "n_cv_infinite": int((stds > 0).sum() - cv.notna().sum()),
            "proteins_gt50pct_missing": int((missing_per_protein > 0.5).sum()),
            "samples_gt50pct_missing": int((missing_per_sample > 0.5).sum()),
            "dynamic_range_log2": float(np.log2(df.max().max() / max(df[df > 0].min().min(), 1e-10))),
            "median_intensity": float(df.median().median()) if df.notna().any().any() else None,
        }
