"""Multi-modal cross-attention fusion model for survival prediction."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class CrossModalAttentionBlock(nn.Module):
    """Multi-head cross-attention between modalities."""

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self, query: torch.Tensor, key_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # query/key_value: (batch, 1, dim) - single token per modality
        attn_out, attn_weights = self.attention(query, key_value, key_value)
        x = self.norm1(query + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x, attn_weights


class CrossModalFusionNet(nn.Module):
    """Full cross-modal fusion network for survival or classification."""

    def __init__(
        self, modality_dims: Dict[str, int], hidden_dim: int = 128,
        n_heads: int = 4, dropout: float = 0.2, task: str = "survival",
    ):
        super().__init__()
        self.modality_names = sorted(modality_dims.keys())
        self.hidden_dim = hidden_dim

        # Per-modality projection
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            )
            for name, dim in modality_dims.items()
        })

        # Cross-attention blocks
        self.cross_attns = nn.ModuleDict({
            name: CrossModalAttentionBlock(hidden_dim, n_heads, dropout)
            for name in self.modality_names
        })

        # Gated fusion
        n_mod = len(self.modality_names)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * n_mod, n_mod), nn.Softmax(dim=-1),
        )

        # Task head
        if task == "survival":
            self.head = nn.Linear(hidden_dim, 1)
        else:
            self.head = nn.Linear(hidden_dim, 2)
        self.task = task

    def forward(
        self, modalities: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Project each modality
        projected = {}
        for name in self.modality_names:
            x = self.projections[name](modalities[name])  # (batch, hidden)
            projected[name] = x.unsqueeze(1)  # (batch, 1, hidden)

        # Cross-attention: each modality attends to concatenation of others
        attended = {}
        attn_weights = {}
        for name in self.modality_names:
            others = torch.cat(
                [projected[n] for n in self.modality_names if n != name], dim=1,
            )
            out, weights = self.cross_attns[name](projected[name], others)
            attended[name] = out.squeeze(1)  # (batch, hidden)
            attn_weights[name] = weights

        # Gated fusion
        concat = torch.cat([attended[n] for n in self.modality_names], dim=-1)
        gates = self.gate(concat)  # (batch, n_modalities)

        fused = torch.zeros_like(attended[self.modality_names[0]])
        for i, name in enumerate(self.modality_names):
            fused += gates[:, i:i+1] * attended[name]

        # Task head
        output = self.head(fused)
        if self.task == "survival":
            output = output.squeeze(-1)

        return output, attn_weights


class MultiModalAttentionFusion:
    """High-level wrapper for cross-modal attention fusion training."""

    def __init__(
        self, modality_dims: Optional[Dict[str, int]] = None,
        hidden_dim: int = 128, n_heads: int = 4, dropout: float = 0.2,
        task: str = "survival", device: str = "cpu", **kwargs,
    ):
        self.modality_dims = modality_dims
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.task = task
        self.device = device
        self.model = None
        self._fitted = False

    def fit(
        self, X: np.ndarray, T: np.ndarray, E: np.ndarray,
        modality_splits: Dict[str, List[int]],
        val_X: Optional[np.ndarray] = None,
        val_T: Optional[np.ndarray] = None,
        val_E: Optional[np.ndarray] = None,
        n_epochs: int = 100, batch_size: int = 64,
        patience: int = 10, lr: float = 1e-3,
    ) -> Dict:
        """Train cross-modal attention fusion model."""
        # Validate and determine modality dimensions
        if not modality_splits:
            raise ValueError("modality_splits is empty — at least 2 modalities required for cross-attention")
        if len(modality_splits) < 2:
            raise ValueError(f"Cross-attention requires >=2 modalities, got {list(modality_splits.keys())}")
        n_features = X.shape[1]
        for name, indices in modality_splits.items():
            if not indices:
                raise ValueError(f"Modality '{name}' has no feature indices")
            if max(indices) >= n_features:
                raise ValueError(
                    f"Modality '{name}' index {max(indices)} out of bounds for {n_features} features"
                )

        self.modality_dims = {name: len(idx) for name, idx in modality_splits.items()}
        self.modality_splits = modality_splits

        self.model = CrossModalFusionNet(
            self.modality_dims, self.hidden_dim, self.n_heads, self.dropout, self.task,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        history = {"train_loss": [], "val_c_index": []}
        best_val_ci = 0.0
        best_state = None
        wait = 0

        for epoch in range(n_epochs):
            self.model.train()

            # Shuffle
            perm = np.random.permutation(len(X))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(X), batch_size):
                idx = perm[start:start + batch_size]
                if len(idx) < 4:
                    continue

                batch_mods = self._split_modalities(X[idx])
                T_b = torch.tensor(T[idx], dtype=torch.float32).to(self.device)
                E_b = torch.tensor(E[idx], dtype=torch.float32).to(self.device)

                optimizer.zero_grad()
                risk, _ = self.model(batch_mods)

                from pipeline4.models.deepsurv import cox_partial_likelihood_loss
                loss = cox_partial_likelihood_loss(risk, T_b, E_b)
                if torch.isnan(loss):
                    logger.warning(f"NaN loss at epoch {epoch+1} — skipping batch")
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_loss)

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

            if val_ci > best_val_ci:
                best_val_ci = val_ci
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self._fitted = True

        logger.info(f"Attention fusion trained: best val C-index={best_val_ci:.4f}")
        return {"val_c_index": best_val_ci}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return risk scores."""
        self.model.eval()
        mods = self._split_modalities(X)
        with torch.no_grad():
            risk, _ = self.model(mods)
        return risk.cpu().numpy()

    def get_attention_weights(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract cross-attention weight matrices."""
        self.model.eval()
        mods = self._split_modalities(X)
        with torch.no_grad():
            _, attn_weights = self.model(mods)
        return {k: v.cpu().numpy() for k, v in attn_weights.items()}

    def _split_modalities(self, X: np.ndarray) -> Dict[str, torch.Tensor]:
        """Split combined feature matrix into per-modality tensors."""
        result = {}
        for name, indices in self.modality_splits.items():
            result[name] = torch.tensor(X[:, indices], dtype=torch.float32).to(self.device)
        return result

    def save(self, path: str) -> None:
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "modality_dims": self.modality_dims,
            "modality_splits": self.modality_splits,
            "config": {
                "hidden_dim": self.hidden_dim, "n_heads": self.n_heads,
                "dropout": self.dropout, "task": self.task,
            },
        }, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.modality_dims = state["modality_dims"]
        self.modality_splits = state["modality_splits"]
        cfg = state["config"]
        self.model = CrossModalFusionNet(
            self.modality_dims, cfg["hidden_dim"], cfg["n_heads"], cfg["dropout"], cfg["task"],
        )
        self.model.load_state_dict(state["model_state_dict"])
        self._fitted = True
