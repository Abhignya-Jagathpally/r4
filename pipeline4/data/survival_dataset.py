"""PyTorch Dataset classes for survival and classification tasks."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SurvivalDataset(Dataset):
    """Dataset for (features, time, event) survival tuples."""

    def __init__(self, X: np.ndarray, T: np.ndarray, E: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.T = torch.tensor(T, dtype=torch.float32)
        self.E = torch.tensor(E, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.T[idx], self.E[idx]


class ClassificationDataset(Dataset):
    """Dataset for (features, label) classification tuples."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class MultiModalDataset(Dataset):
    """Dataset for multi-modal fusion: expression + clinical + cell proportions."""

    def __init__(
        self,
        modalities: Dict[str, np.ndarray],
        T: np.ndarray,
        E: np.ndarray,
    ):
        self.modalities = {
            k: torch.tensor(v, dtype=torch.float32) for k, v in modalities.items()
        }
        self.T = torch.tensor(T, dtype=torch.float32)
        self.E = torch.tensor(E, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.T)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        batch = {k: v[idx] for k, v in self.modalities.items()}
        return batch, self.T[idx], self.E[idx]
