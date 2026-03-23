"""Multi-modal feature combination and normalization."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MultiModalFeatureBuilder:
    """Combine and manage multi-modal feature sets."""

    def combine(
        self,
        expression_features: pd.DataFrame,
        clinical_features: pd.DataFrame,
        genomic_features: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Concatenate feature matrices aligned by patient ID."""
        parts = [expression_features, clinical_features]
        if genomic_features is not None and not genomic_features.empty:
            parts.append(genomic_features)

        # Align on common index
        common_idx = parts[0].index
        for p in parts[1:]:
            common_idx = common_idx.intersection(p.index)

        aligned = [p.loc[common_idx] for p in parts]
        combined = pd.concat(aligned, axis=1)
        combined = combined.fillna(0)

        logger.info(
            f"Combined multi-modal features: {combined.shape} "
            f"({len(common_idx)} patients, {combined.shape[1]} features)"
        )
        return combined

    def get_modality_splits(
        self,
        expression_features: pd.DataFrame,
        clinical_features: pd.DataFrame,
        genomic_features: Optional[pd.DataFrame] = None,
    ) -> Dict[str, List[int]]:
        """Return column index ranges for each modality."""
        splits = {}
        offset = 0

        splits["expression"] = list(range(offset, offset + expression_features.shape[1]))
        offset += expression_features.shape[1]

        splits["clinical"] = list(range(offset, offset + clinical_features.shape[1]))
        offset += clinical_features.shape[1]

        if genomic_features is not None and not genomic_features.empty:
            splits["genomic"] = list(range(offset, offset + genomic_features.shape[1]))

        logger.info(f"Modality splits: { {k: len(v) for k, v in splits.items()} }")
        return splits

    def normalize_modalities(
        self, combined: pd.DataFrame, modality_splits: Dict[str, List[int]],
    ) -> pd.DataFrame:
        """StandardScaler per modality to prevent dominance."""
        result = combined.copy()
        for name, indices in modality_splits.items():
            cols = combined.columns[indices]
            scaler = StandardScaler()
            result[cols] = scaler.fit_transform(result[cols])
        logger.info("Normalized features per modality")
        return result
