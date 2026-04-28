"""Scene graph encoder — graph neural network over detected objects.

This module is *optional*; it enriches the visual representation with
explicit object-relation structure when bounding-box detections are available.
Without detections the module is a no-op (returns the unmodified patch tokens).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    """Single graph convolutional layer (mean-aggregation variant)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   [N, in_dim]  node features
            adj: [N, N]       normalised adjacency matrix
        Returns:
            [N, out_dim]
        """
        return self.norm(torch.relu(self.lin(adj @ x)))


class SceneGraphEncoder(nn.Module):
    """Encodes an object-level scene graph into enriched node embeddings.

    Usage:
        If bounding-box region features and an adjacency matrix are available,
        pass them to forward() to get graph-enhanced features.  Otherwise the
        module returns the raw patch_tokens unchanged.
    """

    def __init__(self, node_dim: int = 768, hidden_dim: int = 768, num_layers: int = 2) -> None:
        super().__init__()
        dims = [node_dim] + [hidden_dim] * num_layers
        self.gcn = nn.ModuleList(
            [GCNLayer(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            patch_tokens: [B, N, D]
            adj:          [B, N, N] or None
        Returns:
            enriched tokens: [B, N, D]
        """
        if adj is None:
            return patch_tokens

        B, N, D = patch_tokens.shape
        x = patch_tokens.view(B * N, D)
        adj_flat = adj.view(B * N, N)  # rough batching — replace with scatter for large N

        for layer in self.gcn:
            x = layer(x, adj_flat)

        x = self.out_proj(x).view(B, N, -1)
        return patch_tokens + x  # residual
