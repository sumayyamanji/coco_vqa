"""CLIP-based vision encoder — extracts patch embeddings and CLS token."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPVisionConfig


class VisionEncoder(nn.Module):
    """Wraps a pretrained CLIP ViT and projects its output to hidden_dim.

    Returns both the CLS (global) token and the grid of patch tokens so
    that downstream fusion layers can attend to spatial image features.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        hidden_dim: int = 768,
        freeze_backbone: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = CLIPVisionModel.from_pretrained(model_name)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        clip_dim = self.backbone.config.hidden_size
        self.proj = nn.Linear(clip_dim, hidden_dim) if clip_dim != hidden_dim else nn.Identity()

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pixel_values: [B, 3, H, W]
        Returns:
            cls_token:    [B, hidden_dim]
            patch_tokens: [B, num_patches, hidden_dim]
        """
        outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=False)
        # last_hidden_state: [B, 1 + num_patches, clip_dim]
        hidden = outputs.last_hidden_state
        cls_token = self.proj(hidden[:, 0])
        patch_tokens = self.proj(hidden[:, 1:])
        return cls_token, patch_tokens
