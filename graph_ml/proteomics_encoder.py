"""Proteomics encoder with pathway-aware attention (M39: dim match, M44: padding mask)."""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PathwayAwareEncoder(nn.Module):
    """Encode proteomics data with pathway-aware attention (M39: output dim matches)."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, n_pathways: int = 10,
                 output_dim: int = 128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pathway_embeddings = nn.Parameter(torch.randn(n_pathways, hidden_dim) * 0.01)
        # M39: attention output projected to match expected downstream dim
        self.attn_q = nn.Linear(hidden_dim, hidden_dim)
        self.attn_k = nn.Linear(hidden_dim, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)  # M39: explicit dim match

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Args: x (batch, n_proteins), mask (batch, n_proteins) - True for padding."""
        h = self.input_proj(x)  # (batch, hidden)
        h = h.unsqueeze(1)  # (batch, 1, hidden)

        # Cross-attention with pathway embeddings
        pw = self.pathway_embeddings.unsqueeze(0).expand(x.shape[0], -1, -1)
        q = self.attn_q(h)
        k = self.attn_k(pw)
        v = self.attn_v(pw)

        scores = torch.bmm(q, k.transpose(1, 2)) / (k.shape[-1] ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attended = torch.bmm(attn, v).squeeze(1)  # (batch, hidden)

        # M39: project to output_dim to match downstream expectations
        return self.output_proj(attended)


class PathwayTransformer(nn.Module):
    """Transformer encoder with padding mask for missing proteins (M44)."""

    def __init__(self, n_proteins: int, hidden_dim: int = 128, n_heads: int = 4,
                 n_layers: int = 2, output_dim: int = 128):
        super().__init__()
        self.protein_embedding = nn.Linear(1, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, n_proteins, hidden_dim) * 0.01)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Args: x (batch, n_proteins), padding_mask (batch, n_proteins) True=pad (M44)."""
        h = self.protein_embedding(x.unsqueeze(-1))  # (batch, n_prot, hidden)
        h = h + self.pos_encoding[:, :h.shape[1], :]

        # M44: Apply padding mask for missing protein data
        h = self.transformer(h, src_key_padding_mask=padding_mask)

        # Pool over non-padded positions
        if padding_mask is not None:
            mask_expanded = (~padding_mask).unsqueeze(-1).float()
            h = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)

        return self.output_proj(h)
