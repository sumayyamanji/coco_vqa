"""Cross-attention fusion — lets question tokens attend to image patches."""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class CrossAttentionBlock(nn.Module):
    """Single cross-attention layer: query from text, key/value from vision."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        text_tokens: torch.Tensor,
        image_patches: torch.Tensor,
        text_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # cross: text queries attend to image keys/values
        x, _ = self.cross_attn(
            query=text_tokens, key=image_patches, value=image_patches
        )
        x = self.norm1(text_tokens + x)
        # self: text tokens attend to each other
        s, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=text_key_padding_mask)
        x = self.norm2(x + s)
        x = self.norm3(x + self.ff(x))
        return x


class CrossAttentionFusion(nn.Module):
    """Stack of CrossAttentionBlocks that fuses visual and language features.

    The CLS representation after the last block is used for answer prediction.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        text_tokens: torch.Tensor,
        image_patches: torch.Tensor,
        text_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            text_tokens:  [B, seq_len, D]
            image_patches: [B, num_patches, D]
            text_key_padding_mask: [B, seq_len] bool mask (True = pad)
        Returns:
            fused_cls: [B, D]  — CLS token after final fusion layer
        """
        x = text_tokens
        for layer in self.layers:
            x = layer(x, image_patches, text_key_padding_mask)
        return x[:, 0]  # CLS token
