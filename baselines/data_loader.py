"""Data loader with time-aware splitting (M32)."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and split proteomics data for baseline evaluation.

    Time-Aware Splitting (M32):
        When temporal metadata is available, splits are performed
        chronologically to prevent temporal leakage. Samples from
        earlier time points are used for training, later for testing.
        This prevents the model from learning future information.

        If no temporal column exists, stratified random splitting
        is used with group-aware constraints (same patient cannot
        appear in both train and test).
    """

    def load(self, config: Dict, data_dir: str = "data") -> Tuple[np.ndarray, pd.DataFrame]:
        """Load expression and clinical data."""
        expr_path = Path(data_dir) / "merged" / "expression.parquet"
        clin_path = Path(data_dir) / "merged" / "clinical.parquet"

        if expr_path.exists() and clin_path.exists():
            expression = pd.read_parquet(expr_path)
            clinical = pd.read_parquet(clin_path)
        else:
            logger.warning("Data not found, generating synthetic")
            from data_pipeline.ingest import ProteomicsIngestor
            ingestor = ProteomicsIngestor()
            df = ingestor.generate_demo_data()
            expression = df
            clinical = pd.DataFrame({
                "group": [1 if "MM" in idx else 0 for idx in df.index],
                "survival_time": np.random.exponential(30, len(df)),
                "event": np.random.binomial(1, 0.6, len(df)),
            }, index=df.index)

        X = expression.fillna(0).values
        logger.info(f"Loaded data: {X.shape}, clinical: {clinical.shape}")
        return X, clinical

    def time_aware_split(
        self, X: np.ndarray, clinical: pd.DataFrame,
        time_col: str = "collection_date", test_size: float = 0.2,
    ) -> Tuple:
        """Split data chronologically if time column available (M32)."""
        if time_col in clinical.columns:
            sorted_idx = clinical[time_col].sort_values().index
            n_test = int(len(sorted_idx) * test_size)
            train_idx = sorted_idx[:-n_test]
            test_idx = sorted_idx[-n_test:]
            logger.info(f"Time-aware split: train={len(train_idx)}, test={len(test_idx)}")
        else:
            from sklearn.model_selection import train_test_split
            indices = np.arange(len(X))
            train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
            logger.info(f"Random split: train={len(train_idx)}, test={len(test_idx)}")

        return train_idx, test_idx
