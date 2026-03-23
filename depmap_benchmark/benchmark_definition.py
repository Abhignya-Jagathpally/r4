"""Benchmark suite (M54: study-level splits, M55: real metrics)."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


class BenchmarkSuite:
    """Benchmark definition with study-level splits and real metrics."""

    def study_level_splits(
        self, data: pd.DataFrame, study_col: str = "study", n_splits: int = 5, seed: int = 42,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split by study to prevent information leakage (M54)."""
        if study_col not in data.columns:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            return list(kf.split(data))

        rng = np.random.RandomState(seed)
        studies = data[study_col].unique()
        rng.shuffle(studies)
        splits = []

        fold_size = max(1, len(studies) // n_splits)
        for i in range(n_splits):
            test_studies = set(studies[i * fold_size:(i + 1) * fold_size])
            train_mask = ~data[study_col].isin(test_studies)
            test_mask = data[study_col].isin(test_studies)
            if test_mask.sum() > 0:
                splits.append((np.where(train_mask)[0], np.where(test_mask)[0]))

        logger.info(f"Study-level splits: {len(splits)} folds from {len(studies)} studies")
        return splits

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Real evaluation metrics (M55: not hardcoded 0.75)."""
        results = {}
        try:
            results["pearson_r"] = float(pearsonr(y_true, y_pred)[0])
        except Exception:
            results["pearson_r"] = float("nan")
        try:
            results["spearman_rho"] = float(spearmanr(y_true, y_pred)[0])
        except Exception:
            results["spearman_rho"] = float("nan")
        results["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        results["mse"] = float(mean_squared_error(y_true, y_pred))

        # C-index for drug sensitivity ranking
        from lifelines.utils import concordance_index
        try:
            results["c_index"] = float(concordance_index(y_true, y_pred))
        except Exception:
            results["c_index"] = float("nan")

        logger.info(f"Benchmark: r={results['pearson_r']:.4f}, rho={results['spearman_rho']:.4f}")
        return results
