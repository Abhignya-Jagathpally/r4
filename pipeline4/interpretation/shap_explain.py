"""SHAP-based model explanation."""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """Compute and visualize SHAP explanations."""

    def __init__(self, model: object, model_type: str = "tree"):
        self.model = model
        self.model_type = model_type
        self._shap_values = None

    def explain(
        self, X: np.ndarray, max_samples: int = 500,
    ) -> np.ndarray:
        """Compute SHAP values."""
        import shap

        X_sample = X[:max_samples]

        if self.model_type == "tree":
            inner = self.model.model if hasattr(self.model, "model") else self.model
            explainer = shap.TreeExplainer(inner)
            self._shap_values = explainer.shap_values(X_sample)
        else:
            # Kernel SHAP fallback
            def predict_fn(x):
                return self.model.predict(x) if hasattr(self.model, "predict") else x
            background = shap.sample(X_sample, min(100, len(X_sample)))
            explainer = shap.KernelExplainer(predict_fn, background)
            self._shap_values = explainer.shap_values(X_sample[:min(100, len(X_sample))])

        if isinstance(self._shap_values, list):
            self._shap_values = self._shap_values[0] if len(self._shap_values) > 0 else self._shap_values

        return self._shap_values

    def top_features(
        self, X: np.ndarray, feature_names: Optional[List[str]] = None, n: int = 50,
    ) -> pd.DataFrame:
        """Return top features by mean absolute SHAP value."""
        if self._shap_values is None:
            self.explain(X)

        mean_abs = np.abs(self._shap_values).mean(axis=0)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(mean_abs))]

        df = pd.DataFrame({
            "feature": feature_names[:len(mean_abs)],
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False)
        return df.head(n).set_index("feature")

    def summary_plot(
        self, X: np.ndarray, feature_names: Optional[List[str]] = None,
        output_path: Optional[str] = None,
    ) -> None:
        """Generate SHAP summary beeswarm plot."""
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if self._shap_values is None:
            self.explain(X)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self._shap_values, X[:len(self._shap_values)],
            feature_names=feature_names, show=False, max_display=20,
        )
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved SHAP summary to {output_path}")
