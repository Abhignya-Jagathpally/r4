"""Reproducibility utilities."""

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int = 42) -> None:
    """Set random seeds for all libraries to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    logger.info(f"Set all random seeds to {seed}")
