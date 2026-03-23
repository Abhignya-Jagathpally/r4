"""GNN models for molecular property prediction.

Fixes: M36 (MPNN dst_feat), M37 (AttentiveFP softmax dim=1), M42 (residual from layer 0),
M43 (edge dim validation).
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from torch_geometric.nn import GCNConv, GATConv, MessagePassing, global_mean_pool
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    logger.warning("torch_geometric not available, GNN models will use fallback")


class GCNModel(nn.Module):
    """Graph Convolutional Network with residual connections from layer 0 (M42)."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 1,
                 n_layers: int = 3, dropout: float = 0.2, edge_dim: Optional[int] = None):
        super().__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.dropout = dropout
        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                                     nn.Linear(hidden_dim // 2, out_dim))

    def forward(self, data: "Data") -> torch.Tensor:
        # M43: Edge dimension validation
        if self.edge_dim is not None and data.edge_attr is not None:
            assert data.edge_attr.shape[1] == self.edge_dim, \
                f"Edge dim mismatch: expected {self.edge_dim}, got {data.edge_attr.shape[1]}"

        x = self.input_proj(data.x)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = x  # M42: residual from layer 0 (not layer 1)
            x = conv(x, data.edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual  # M42: always add residual

        x = global_mean_pool(x, data.batch)
        return self.output(x)


class GATModel(nn.Module):
    """Graph Attention Network."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 1,
                 n_layers: int = 3, n_heads: int = 4, dropout: float = 0.2,
                 edge_dim: Optional[int] = None):
        super().__init__()
        self.edge_dim = edge_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // n_heads, heads=n_heads, dropout=dropout,
                    edge_dim=edge_dim)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.dropout = dropout
        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                                     nn.Linear(hidden_dim // 2, out_dim))

    def forward(self, data: "Data") -> torch.Tensor:
        if self.edge_dim is not None and data.edge_attr is not None:
            assert data.edge_attr.shape[1] == self.edge_dim

        x = self.input_proj(data.x)
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, data.edge_index, edge_attr=data.edge_attr if self.edge_dim else None)
            x = norm(x)
            x = F.relu(x) + residual
        x = global_mean_pool(x, data.batch)
        return self.output(x)


class MPNNModel(nn.Module if HAS_PYG else object):
    """Message Passing Neural Network (M36: message uses BOTH src AND dst features)."""

    def __init__(self, node_dim: int, edge_dim: int = 4, hidden_dim: int = 128,
                 out_dim: int = 1, n_layers: int = 3, dropout: float = 0.2):
        if not HAS_PYG:
            raise ImportError("torch_geometric required for MPNN")
        super().__init__()
        self.edge_dim = edge_dim
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        # M36: message function includes dst_feat via 2*hidden + edge
        self.edge_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden_dim + edge_dim, hidden_dim), nn.ReLU(),
                          nn.Linear(hidden_dim, hidden_dim))
            for _ in range(n_layers)
        ])
        self.node_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU())
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.dropout = dropout
        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                                     nn.Linear(hidden_dim // 2, out_dim))

    def forward(self, data: "Data") -> torch.Tensor:
        assert data.edge_attr is not None, "MPNN requires edge features"
        assert data.edge_attr.shape[1] == self.edge_dim, \
            f"Edge dim mismatch: expected {self.edge_dim}, got {data.edge_attr.shape[1]}"

        x = self.node_proj(data.x)
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        for edge_mlp, node_mlp, norm in zip(self.edge_mlps, self.node_mlps, self.norms):
            residual = x
            src, dst = edge_index
            # M36: message = f(src_feat, dst_feat, edge_feat)
            msg_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)
            messages = edge_mlp(msg_input)

            # Aggregate
            agg = torch.zeros_like(x)
            agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)

            # Update
            x = node_mlp(torch.cat([x, agg], dim=-1))
            x = norm(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual

        x = global_mean_pool(x, data.batch)
        return self.output(x)


class AttentiveFPModel(nn.Module if HAS_PYG else object):
    """AttentiveFP (M37: softmax on dim=1, not dim=0)."""

    def __init__(self, node_dim: int, edge_dim: int = 4, hidden_dim: int = 128,
                 out_dim: int = 1, n_layers: int = 3, dropout: float = 0.2):
        if not HAS_PYG:
            raise ImportError("torch_geometric required")
        super().__init__()
        self.edge_dim = edge_dim
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.attn_layers = nn.ModuleList([
            nn.Linear(3 * hidden_dim, 1) for _ in range(n_layers)
        ])
        self.update_layers = nn.ModuleList([
            nn.GRUCell(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.dropout = dropout
        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                                     nn.Linear(hidden_dim // 2, out_dim))

    def forward(self, data: "Data") -> torch.Tensor:
        assert data.edge_attr is not None and data.edge_attr.shape[1] == self.edge_dim

        x = self.node_proj(data.x)
        edge_feat = self.edge_proj(data.edge_attr)
        edge_index = data.edge_index

        for attn_layer, update in zip(self.attn_layers, self.update_layers):
            residual = x
            src, dst = edge_index

            # Compute attention scores
            attn_input = torch.cat([x[src], x[dst], edge_feat], dim=-1)
            scores = attn_layer(attn_input).squeeze(-1)

            # M37: softmax on dim=1 (across neighbors), NOT dim=0 (across batch)
            # Group by destination node and softmax within each group
            attn_weights = torch.zeros_like(scores)
            for node_idx in dst.unique():
                mask = dst == node_idx
                if mask.any():
                    attn_weights[mask] = F.softmax(scores[mask], dim=0)  # within neighbors of this node

            # Weighted message
            weighted_msg = attn_weights.unsqueeze(-1) * x[src]
            agg = torch.zeros_like(x)
            agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted_msg), weighted_msg)

            x = update(agg, x)
            x = x + residual

        x = global_mean_pool(x, data.batch)
        return self.output(x)
