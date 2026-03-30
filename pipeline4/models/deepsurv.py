"""DeepSurv: Deep Cox proportional hazards neural network."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class DeepSurvNet(nn.Module):
    """Neural network for Cox partial likelihood optimization."""

    def __init__(
        self, input_dim: int, hidden_dims: List[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128, 64]

        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.BatchNorm1d(hdim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def cox_partial_likelihood_loss(
    risk_pred: torch.Tensor, T: torch.Tensor, E: torch.Tensor,
) -> torch.Tensor:
    """Negative log partial likelihood for Cox model.

    Args:
        risk_pred: Predicted log-risk scores (n_samples,)
        T: Survival times (n_samples,)
        E: Event indicators (n_samples,)
    """
    # Sort by descending survival time
    order = torch.argsort(T, descending=True)
    risk_pred = risk_pred[order]
    E = E[order]

    # Compute log cumulative sum of exp(risk) for risk set
    log_cumsum_exp = torch.logcumsumexp(risk_pred, dim=0)

    # Partial likelihood: sum over events of (risk_i - log_cumsum)
    uncensored_ll = risk_pred - log_cumsum_exp
    loss = -torch.mean(uncensored_ll * E)
    return loss


class DeepSurvModel:
    """DeepSurv survival model with PyTorch training loop."""

    def __init__(
        self, input_dim: int = None, hidden_dims: List[int] = None,
        dropout: float = 0.3, lr: float = 1e-3, weight_decay: float = 1e-4,
        device: str = "cpu", **kwargs,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.model = None
        self._fitted = False

    def fit(
        self, X: np.ndarray, T: np.ndarray, E: np.ndarray,
        val_X: Optional[np.ndarray] = None,
        val_T: Optional[np.ndarray] = None,
        val_E: Optional[np.ndarray] = None,
        n_epochs: int = 100, batch_size: int = 64, patience: int = 10,
    ) -> Dict:
        """Train DeepSurv with early stopping."""
        if self.input_dim is None:
            self.input_dim = X.shape[1]

        self.model = DeepSurvNet(self.input_dim, self.hidden_dims, self.dropout)
        self.model = self.model.to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        # Create data loaders
        train_ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(T, dtype=torch.float32),
            torch.tensor(E, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

        history = {"train_loss": [], "val_c_index": []}
        best_val_ci = 0.0
        best_state = None
        wait = 0

        for epoch in range(n_epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            for X_b, T_b, E_b in train_loader:
                X_b, T_b, E_b = X_b.to(self.device), T_b.to(self.device), E_b.to(self.device)
                optimizer.zero_grad()
                risk = self.model(X_b)
                loss = cox_partial_likelihood_loss(risk, T_b, E_b)
                if torch.isnan(loss):
                    logger.warning(f"NaN loss at epoch {epoch+1}, batch {n_batches+1} — skipping")
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_loss)
            scheduler.step(avg_loss)

            # Validation
            val_ci = 0.5
            if val_X is not None:
                risk_scores = self.predict(val_X)
                from lifelines.utils import concordance_index as ci_fn
                try:
                    val_ci = ci_fn(val_T, -risk_scores, val_E)
                except Exception:
                    val_ci = 0.5
            history["val_c_index"].append(val_ci)

            # Early stopping
            if val_ci > best_val_ci:
                best_val_ci = val_ci
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % max(1, n_epochs // 10) == 0:
                logger.debug(
                    f"DeepSurv epoch {epoch+1}/{n_epochs}: "
                    f"loss={avg_loss:.4f}, val_ci={val_ci:.4f}"
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self._fitted = True

        logger.info(f"DeepSurv trained: best val C-index={best_val_ci:.4f}")
        return {"val_c_index": best_val_ci, "n_epochs_trained": epoch + 1}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return risk scores."""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            risk = self.model(X_t).cpu().numpy()
        return risk

    def save(self, path: str) -> None:
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": {
                "input_dim": self.input_dim, "hidden_dims": self.hidden_dims,
                "dropout": self.dropout, "lr": self.lr,
            },
        }, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        cfg = state["config"]
        self.input_dim = cfg["input_dim"]
        self.hidden_dims = cfg["hidden_dims"]
        self.model = DeepSurvNet(cfg["input_dim"], cfg["hidden_dims"], cfg["dropout"])
        self.model.load_state_dict(state["model_state_dict"])
        self._fitted = True
