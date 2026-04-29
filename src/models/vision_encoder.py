"""CLIP-based vision encoder — extracts patch embeddings and CLS token."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from transformers import CLIPVisionModel


class CLIPVisionEncoder(nn.Module):
    """Wraps ``openai/clip-vit-large-patch14`` and projects to model hidden_dim.

    Returns patch_embeddings (spatial tokens) first and cls_embedding (global
    token) second, so callers can directly unpack for cross-attention fusion.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config["model"]
        train_cfg = config.get("training", {})

        model_name: str = model_cfg.get("vision_backbone", "openai/clip-vit-large-patch14")
        hidden_dim: int = model_cfg["hidden_dim"]
        freeze: bool = model_cfg.get("freeze_vision", False)
        grad_ckpt: bool = train_cfg.get("gradient_checkpointing", False)

        self.backbone = CLIPVisionModel.from_pretrained(model_name)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        if grad_ckpt:
            self.backbone.gradient_checkpointing_enable()

        clip_dim: int = self.backbone.config.hidden_size
        self.proj = (
            nn.Linear(clip_dim, hidden_dim) if clip_dim != hidden_dim else nn.Identity()
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pixel_values: (B, 3, H, W)
        Returns:
            patch_embeddings: (B, num_patches, hidden_dim)
            cls_embedding:    (B, hidden_dim)
        """
        out = self.backbone(pixel_values=pixel_values, output_hidden_states=False)
        hidden = out.last_hidden_state          # (B, 1+num_patches, clip_dim)
        cls_embedding = self.norm(self.proj(hidden[:, 0]))       # (B, D)
        patch_embeddings = self.norm(self.proj(hidden[:, 1:]))   # (B, N, D)
        return patch_embeddings, cls_embedding


# Backward-compat alias used by existing training script
VisionEncoder = CLIPVisionEncoder
