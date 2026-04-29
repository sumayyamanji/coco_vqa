"""Multimodal fusion modules — bidirectional cross-modal attention and bilinear pooling."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class CrossModalBlock(nn.Module):
    """Bidirectional cross-modal attention block.

    Runs two cross-attention operations in parallel:
      Q→V  (question tokens attend to image patches, updating text side)
      V→Q  (image patches attend to question tokens, updating vision side)

    Each side also has its own FFN with hidden_dim*4 intermediate size.
    All sub-layers use post-norm (Add & Norm) with residual connections.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        mha_kw = dict(embed_dim=hidden_dim, num_heads=num_heads,
                      dropout=dropout, batch_first=True)

        # Q→V: text queries attend to image patches
        self.q_to_v_attn = nn.MultiheadAttention(**mha_kw)
        self.norm_q_cross = nn.LayerNorm(hidden_dim)
        self.ffn_q = self._ffn(hidden_dim, dropout)
        self.norm_q_ffn = nn.LayerNorm(hidden_dim)

        # V→Q: image patches attend to text tokens
        self.v_to_q_attn = nn.MultiheadAttention(**mha_kw)
        self.norm_v_cross = nn.LayerNorm(hidden_dim)
        self.ffn_v = self._ffn(hidden_dim, dropout)
        self.norm_v_ffn = nn.LayerNorm(hidden_dim)

        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _ffn(dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        text_tokens: torch.Tensor,
        image_patches: torch.Tensor,
        text_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            text_tokens:            (B, T, D)
            image_patches:          (B, N, D)
            text_key_padding_mask:  (B, T) bool — True for pad positions
        Returns:
            text_tokens:     (B, T, D)  updated
            image_patches:   (B, N, D)  updated
            q_attn_weights:  (B, T, N)  cross-attention weights (Q→V side)
        """
        # --- Q→V side ---
        q_cross, q_attn_w = self.q_to_v_attn(
            query=text_tokens, key=image_patches, value=image_patches,
            need_weights=True, average_attn_weights=True,
        )
        text_tokens = self.norm_q_cross(text_tokens + self.drop(q_cross))
        text_tokens = self.norm_q_ffn(text_tokens + self.ffn_q(text_tokens))

        # --- V→Q side ---
        v_cross, _ = self.v_to_q_attn(
            query=image_patches, key=text_tokens, value=text_tokens,
            key_padding_mask=text_key_padding_mask,
            need_weights=False,
        )
        image_patches = self.norm_v_cross(image_patches + self.drop(v_cross))
        image_patches = self.norm_v_ffn(image_patches + self.ffn_v(image_patches))

        return text_tokens, image_patches, q_attn_w


# ---------------------------------------------------------------------------
# Full cross-modal fusion stack
# ---------------------------------------------------------------------------

class CrossModalFusion(nn.Module):
    """Stack of CrossModalBlocks that fuses text and vision representations.

    After the final block both sequences are mean-pooled and concatenated,
    then projected back to hidden_dim to produce a single fused vector.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        m = config["model"]
        hidden_dim: int = m["hidden_dim"]
        num_heads: int = m["num_heads"]
        num_layers: int = m["fusion_layers"]
        dropout: float = m.get("dropout", 0.1)

        self.blocks = nn.ModuleList(
            [CrossModalBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        # Concat pooled text + pooled vision → hidden_dim
        self.pool_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.pool_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        text_tokens: torch.Tensor,
        image_patches: torch.Tensor,
        text_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            text_tokens:           (B, T, D)
            image_patches:         (B, N, D)
            text_key_padding_mask: (B, T) bool — True for pad positions
        Returns:
            fused:              (B, D)   pooled + projected fused vector
            text_tokens:        (B, T, D) final text states (for viz)
            image_patches:      (B, N, D) final patch states (for viz)
            cross_attn_weights: (B, T, N) weights from last block Q→V
        """
        attn_weights: Optional[torch.Tensor] = None
        for block in self.blocks:
            text_tokens, image_patches, attn_weights = block(
                text_tokens, image_patches, text_key_padding_mask
            )

        text_pooled = text_tokens.mean(dim=1)        # (B, D)
        img_pooled = image_patches.mean(dim=1)       # (B, D)
        fused = self.pool_norm(
            self.pool_proj(torch.cat([text_pooled, img_pooled], dim=-1))
        )
        return fused, text_tokens, image_patches, attn_weights


# ---------------------------------------------------------------------------
# Bilinear fusion (Tucker decomposition style)
# ---------------------------------------------------------------------------

class BilinearFusion(nn.Module):
    """Low-rank bilinear pooling via Tucker decomposition.

    Computes: fused = W_out( W1(v1) ⊙ W2(v2) )
    where ⊙ is element-wise product and rank << hidden_dim, giving a compact
    bilinear interaction without the full O(D²) parameter cost.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        m = config["model"]
        hidden_dim: int = m["hidden_dim"]
        rank: int = m.get("bilinear_rank", 512)

        self.W1 = nn.Linear(hidden_dim, rank, bias=False)
        self.W2 = nn.Linear(hidden_dim, rank, bias=False)
        self.out_proj = nn.Linear(rank, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v1: (B, D)  — e.g. vision CLS
            v2: (B, D)  — e.g. text  CLS
        Returns:
            fused: (B, D)
        """
        return self.norm(self.out_proj(self.W1(v1) * self.W2(v2)))


# ---------------------------------------------------------------------------
# Backward-compat alias
# ---------------------------------------------------------------------------

CrossAttentionFusion = CrossModalFusion
