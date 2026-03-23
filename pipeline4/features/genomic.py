"""Genomic feature engineering (mutations, CNV)."""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MM_DRIVER_GENES = [
    "KRAS", "NRAS", "BRAF", "TP53", "FAM46C", "DIS3", "TRAF3",
    "RB1", "CYLD", "MAX", "IRF4", "PRDM1", "SP140", "EGR1",
]


class GenomicFeatures:
    """Build genomic features from mutation and CNV data."""

    def load_mutation_data(self, path: str) -> pd.DataFrame:
        """Load mutation calls if available."""
        p = Path(path)
        if not p.exists():
            logger.warning(f"Mutation data not found at {path}")
            return pd.DataFrame()
        df = pd.read_csv(path, index_col=0) if path.endswith(".csv") else pd.read_parquet(path)
        logger.info(f"Loaded mutation data: {df.shape}")
        return df

    def encode_mutations(
        self, df: pd.DataFrame, genes: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Create binary mutation matrix for key MM genes."""
        if df.empty:
            return df
        genes = genes or MM_DRIVER_GENES
        available = [g for g in genes if g in df.columns]
        if not available:
            logger.warning("No target mutation genes found in data")
            return pd.DataFrame(index=df.index)
        result = (df[available] > 0).astype(float)
        result.columns = [f"mut_{g}" for g in available]
        logger.info(f"Encoded {len(available)} mutation features")
        return result

    def compute_cnv_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute copy number alteration features if available."""
        if df.empty:
            return df
        # Binarize: gain (>2) and loss (<2) for each gene
        gains = (df > 2).astype(float)
        gains.columns = [f"gain_{c}" for c in gains.columns]
        losses = (df < 2).astype(float)
        losses.columns = [f"loss_{c}" for c in losses.columns]
        result = pd.concat([gains, losses], axis=1)
        logger.info(f"Computed {result.shape[1]} CNV features")
        return result

    def generate_synthetic_mutations(
        self, patient_ids: pd.Index, seed: int = 42,
    ) -> pd.DataFrame:
        """Generate synthetic mutation data for demo."""
        rng = np.random.RandomState(seed)
        n = len(patient_ids)
        # MM mutation frequencies from literature
        freqs = {
            "KRAS": 0.23, "NRAS": 0.20, "BRAF": 0.04, "TP53": 0.08,
            "FAM46C": 0.11, "DIS3": 0.11, "TRAF3": 0.05, "RB1": 0.04,
        }
        data = {gene: rng.binomial(1, freq, n) for gene, freq in freqs.items()}
        df = pd.DataFrame(data, index=patient_ids)
        logger.info(f"Generated synthetic mutations: {df.shape}")
        return df
