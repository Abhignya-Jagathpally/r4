"""Attention weight extraction and visualization."""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AttentionAnalyzer:
    """Extract and analyze cross-attention weights from fusion model."""

    def __init__(self, fusion_model):
        self.model = fusion_model

    def extract_weights(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get cross-attention weight matrices."""
        return self.model.get_attention_weights(X)

    def modality_importance(self, weights: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Aggregate attention to per-modality importance."""
        importance = {}
        for modality, w in weights.items():
            # w shape: (batch, n_heads, query_len, key_len)
            if w.ndim >= 3:
                importance[modality] = float(np.mean(np.abs(w)))
            else:
                importance[modality] = float(np.mean(np.abs(w)))

        total = sum(importance.values()) or 1.0
        normalized = {k: v / total for k, v in importance.items()}

        df = pd.DataFrame([
            {"modality": k, "raw_importance": importance[k], "normalized": normalized[k]}
            for k in importance
        ]).set_index("modality").sort_values("normalized", ascending=False)
        return df

    def plot_modality_importance(
        self, importance: pd.DataFrame, output_path: str,
    ) -> None:
        """Bar chart of modality contributions."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        importance["normalized"].plot.bar(ax=ax, color="steelblue")
        ax.set_ylabel("Normalized Attention Weight")
        ax.set_title("Cross-Modal Attention: Modality Importance")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"Saved modality importance plot to {output_path}")
