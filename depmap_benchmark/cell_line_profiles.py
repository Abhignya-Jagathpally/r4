"""Cell line profiling (M66: real features, M67: data-driven thresholds)."""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CellLineProfiler:
    """Build molecular profiles from DepMap data (M66, M67)."""

    def build_profile(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build profile using real molecular features (M66)."""
        profile = pd.DataFrame(index=data.index)

        # Expression statistics
        profile["mean_expression"] = data.mean(axis=1)
        profile["std_expression"] = data.std(axis=1)
        profile["n_expressed"] = (data > 0).sum(axis=1)

        # Gene-specific features if columns match gene names
        dependency_genes = ["BCL2", "MCL1", "PSMB5", "CRBN", "XPO1"]
        for gene in dependency_genes:
            matching = [c for c in data.columns if gene in str(c).upper()]
            if matching:
                profile[f"{gene}_dependency"] = data[matching[0]]

        logger.info(f"Built profiles: {profile.shape}")
        return profile

    def classify_dependency(
        self, data: pd.DataFrame, gene: str, threshold: Optional[float] = None,
    ) -> pd.Series:
        """Data-driven dependency classification (M67: not hardcoded lists)."""
        matching = [c for c in data.columns if gene in str(c).upper()]
        if not matching:
            return pd.Series(False, index=data.index)

        values = data[matching[0]]
        if threshold is None:
            # M67: Data-driven threshold at mean - 1 SD
            threshold = values.mean() - values.std()

        dependent = values < threshold  # More negative = more dependent
        logger.info(f"{gene} dependency: {dependent.sum()}/{len(dependent)} lines (threshold={threshold:.3f})")
        return dependent
