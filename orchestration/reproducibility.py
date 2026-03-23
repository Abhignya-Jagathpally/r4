"""Reproducibility management (M90: NUMBA_DISABLE_JIT, M91: weight comparison)."""

import hashlib
import logging
import os
import random
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ReproducibilityManager:
    """Ensure and validate reproducibility."""

    def set_seeds(self, seed: int = 42) -> None:
        """Set all seeds including NUMBA_DISABLE_JIT (M90)."""
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["NUMBA_DISABLE_JIT"] = "1"  # M90

        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

        logger.info(f"Seeds set to {seed} (NUMBA_DISABLE_JIT=1)")

    def validate_reproducibility(
        self, model1_weights: Dict[str, np.ndarray],
        model2_weights: Dict[str, np.ndarray],
        rtol: float = 1e-5,
    ) -> Dict:
        """Validate by comparing model WEIGHTS, not just metrics (M91)."""
        if set(model1_weights.keys()) != set(model2_weights.keys()):
            return {"reproducible": False, "reason": "Different weight keys"}

        max_diff = 0.0
        diffs = {}
        for key in model1_weights:
            w1 = np.asarray(model1_weights[key])
            w2 = np.asarray(model2_weights[key])
            if w1.shape != w2.shape:
                return {"reproducible": False, "reason": f"Shape mismatch at {key}"}
            diff = np.max(np.abs(w1 - w2))
            diffs[key] = float(diff)
            max_diff = max(max_diff, diff)

        reproducible = max_diff < rtol
        return {
            "reproducible": reproducible,
            "max_weight_diff": float(max_diff),
            "rtol": rtol,
            "per_layer_diff": diffs,
        }

    def hash_model(self, weights: Dict[str, np.ndarray]) -> str:
        """Hash model weights for quick comparison."""
        hasher = hashlib.sha256()
        for key in sorted(weights.keys()):
            hasher.update(key.encode())
            hasher.update(np.asarray(weights[key]).tobytes())
        return hasher.hexdigest()[:16]
