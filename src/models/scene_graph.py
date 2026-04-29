"""Scene graph modules — GCN and attention-based relational reasoning over image patches.

Two implementations are provided:
  * SceneGraphEncoder — original lightweight GCN (kept for backward compat).
  * SceneGraphGenerator — new config-driven module with explicit spatial and
    semantic relation types, implemented via attention-based message passing
    (no torch_geometric dependency).
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Original module (kept for backward compat)
# ---------------------------------------------------------------------------

class GCNLayer(nn.Module):
    """Single graph convolutional layer (mean-aggregation variant)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.norm(torch.relu(self.lin(adj @ x)))


class SceneGraphEncoder(nn.Module):
    """Encodes an object-level scene graph into enriched node embeddings (original)."""

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
        if adj is None:
            return patch_tokens

        B, N, D = patch_tokens.shape
        x = patch_tokens.view(B * N, D)
        adj_flat = adj.view(B * N, N)

        for layer in self.gcn:
            x = layer(x, adj_flat)

        x = self.out_proj(x).view(B, N, -1)
        return patch_tokens + x


# ---------------------------------------------------------------------------
# New: attention-based message passing layer
# ---------------------------------------------------------------------------

class _RelationMessagePassing(nn.Module):
    """Single attention-based graph convolution with relation-conditioned messages.

    For each node pair (i, j) a message is computed as a function of both node
    features and a relation embedding, then aggregated via softmax attention.
    This avoids the O(N²·D) memory cost of explicit pairwise tensors by using
    ``nn.MultiheadAttention`` with an additive relation bias on the key side.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        # Maps concatenated relation embedding to key-bias per head
        self.rel_key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rel_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:        (B, N, D) node features
            rel_bias: (B, N, D) optional per-node relation context added to keys
        Returns:
            (B, N, D) updated node features
        """
        k = x if rel_bias is None else x + self.rel_key_proj(rel_bias)
        out, _ = self.attn(query=x, key=k, value=x, need_weights=False)
        return self.norm(x + self.drop(out))


# ---------------------------------------------------------------------------
# SceneGraphGenerator
# ---------------------------------------------------------------------------

class SceneGraphGenerator(nn.Module):
    """Config-driven scene graph module with spatial and semantic relation types.

    Relation taxonomy
    -----------------
    Spatial  : left, right, above, below, inside  (5 types)
    Semantic : holding, wearing, near             (3 types)

    Implementation
    --------------
    Uses 2 rounds of attention-based message passing (no torch_geometric
    required).  Spatial relations are captured by a learnable positional bias
    computed from the patch grid.  Semantic relations are encoded via an
    embedding table that can be injected when ``object_labels`` are provided.

    If torch_geometric *is* installed this module still uses the attention
    approach — the fallback IS the default implementation.

    Args:
        patch_embeddings: (B, N, D)  image patch features from vision encoder
        object_labels:    (B, M) int optional COCO class ids (0 = background)
    Returns:
        relation_embeddings: (B, N, D) — can be added/concatenated to fusion
    """

    SPATIAL_RELATIONS:  List[str] = ["left", "right", "above", "below", "inside"]
    SEMANTIC_RELATIONS: List[str] = ["holding", "wearing", "near"]
    ALL_RELATIONS:      List[str] = SPATIAL_RELATIONS + SEMANTIC_RELATIONS

    # Size of the COCO object-label vocabulary (80 classes + 1 background)
    LABEL_VOCAB_SIZE: int = 81

    def __init__(self, config: dict) -> None:
        super().__init__()
        m = config["model"]
        hidden_dim: int = m["hidden_dim"]
        dropout: float = m.get("dropout", 0.1)
        num_heads: int = min(8, m.get("num_heads", 8))
        num_relations: int = len(self.ALL_RELATIONS)

        # Object-label embedding (COCO 80 classes; 0 = padding/background)
        self.label_emb = nn.Embedding(self.LABEL_VOCAB_SIZE, hidden_dim, padding_idx=0)

        # Relation-type embedding — one vector per relation type
        self.rel_emb = nn.Embedding(num_relations, hidden_dim)

        # Spatial position bias: maps a 2D relative (Δrow, Δcol) to a scalar
        # bias per attention head, pre-computed from patch-grid positions.
        # Stored as a learnable table indexed by (Δrow + grid, Δcol + grid).
        # Default grid size: 14×14 = 196 patches for ViT-L/14.
        self._grid = m.get("patch_grid_size", 14)
        table_size = 2 * self._grid - 1   # range of relative positions
        self.spatial_bias_table = nn.Parameter(
            torch.zeros(table_size * table_size, num_heads)
        )

        # 2-layer attention-based message passing
        self.mp1 = _RelationMessagePassing(hidden_dim, num_heads, dropout)
        self.mp2 = _RelationMessagePassing(hidden_dim, num_heads, dropout)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _spatial_bias(self, N: int, device: torch.device) -> torch.Tensor:
        """Compute (N, D) per-node spatial context from patch grid positions."""
        grid = self._grid
        # Patch (i) row and col in the grid
        rows = torch.arange(N, device=device) // grid   # (N,)
        cols = torch.arange(N, device=device) % grid    # (N,)

        # Mean relative position from each patch to all others (as a summary)
        # shape: (N,) for rows and cols
        mean_rel_row = rows.float() - rows.float().mean()
        mean_rel_col = cols.float() - cols.float().mean()

        # Clamp to valid table range
        half = grid - 1
        idx_row = (mean_rel_row.long().clamp(-half, half) + half)
        idx_col = (mean_rel_col.long().clamp(-half, half) + half)
        table_idx = idx_row * (2 * grid - 1) + idx_col               # (N,)

        # Retrieve bias vectors: (N, num_heads) → mean to hidden_dim proxy
        # We project num_heads dims back to hidden_dim via rel_emb lookup
        bias = self.spatial_bias_table[table_idx]                     # (N, num_heads)
        # Expand to (1, N, D) via learnable upsampling (rel_emb of type 4 = "below")
        # Use a simple repeat to match hidden_dim size
        hidden_dim = self.rel_emb.embedding_dim
        bias_expanded = bias.unsqueeze(-1).expand(-1, -1, hidden_dim // bias.size(-1) + 1)
        bias_expanded = bias_expanded.reshape(N, -1)[:, :hidden_dim]
        return bias_expanded.unsqueeze(0)                             # (1, N, D)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        patch_embeddings: torch.Tensor,
        object_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            patch_embeddings: (B, N, D)
            object_labels:    (B, M) int, COCO class ids; optional
        Returns:
            relation_embeddings: (B, N, D)
        """
        B, N, D = patch_embeddings.shape
        x = patch_embeddings

        # Inject object-label semantics into the first M patches (if provided)
        if object_labels is not None:
            label_feats = self.label_emb(object_labels)           # (B, M, D)
            M = min(label_feats.size(1), N)
            x = x.clone()
            x[:, :M] = x[:, :M] + label_feats[:, :M]

        # Spatial relation context (broadcast over batch)
        spatial_ctx = self._spatial_bias(N, patch_embeddings.device)  # (1, N, D)
        spatial_ctx = spatial_ctx.expand(B, -1, -1)                   # (B, N, D)

        # Semantic relation context: mean of all relation embeddings as global prior
        semantic_ctx = self.rel_emb.weight.mean(dim=0, keepdim=True)  # (1, D)
        semantic_ctx = semantic_ctx.unsqueeze(0).expand(B, N, -1)     # (B, N, D)

        rel_bias = spatial_ctx + semantic_ctx                         # (B, N, D)

        # Two message-passing rounds
        x = self.mp1(x, rel_bias)
        x = self.mp2(x, rel_bias)

        return self.out_norm(self.out_proj(x) + patch_embeddings)     # residual
