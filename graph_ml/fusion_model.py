"""Multi-modal fusion models (M38: correct TensorFusion einsum)."""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TensorFusion(nn.Module):
    """Tensor fusion (M38: correct einsum contraction and output shape)."""

    def __init__(self, drug_dim: int, prot_dim: int, hidden_dim: int = 128, out_dim: int = 1):
        super().__init__()
        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        self.prot_proj = nn.Linear(prot_dim, hidden_dim)
        # M38: outer product produces (batch, hidden, hidden), flatten to hidden^2
        self.fused_dim = hidden_dim * hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(self.fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, drug_emb: torch.Tensor, prot_emb: torch.Tensor) -> torch.Tensor:
        d = self.drug_proj(drug_emb)  # (batch, hidden)
        p = self.prot_proj(prot_emb)  # (batch, hidden)
        # M38: correct einsum — outer product per sample, then flatten
        fused = torch.einsum("bi,bj->bij", d, p)  # (batch, hidden, hidden)
        fused = fused.reshape(fused.shape[0], -1)  # (batch, hidden^2)
        return self.fc(fused)


class MultiModalFusion(nn.Module):
    """Combine drug graph + proteomics embeddings for prediction."""

    def __init__(self, drug_dim: int, prot_dim: int, hidden_dim: int = 128,
                 out_dim: int = 1, fusion_type: str = "tensor"):
        super().__init__()
        if fusion_type == "tensor":
            self.fusion = TensorFusion(drug_dim, prot_dim, hidden_dim, out_dim)
        else:
            # Concatenation fusion
            self.fusion = nn.Sequential(
                nn.Linear(drug_dim + prot_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                nn.Linear(hidden_dim // 2, out_dim),
            )
        self.fusion_type = fusion_type

    def forward(self, drug_emb: torch.Tensor, prot_emb: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == "tensor":
            return self.fusion(drug_emb, prot_emb)
        return self.fusion(torch.cat([drug_emb, prot_emb], dim=-1))
