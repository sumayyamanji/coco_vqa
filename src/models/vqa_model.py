"""VQAModel — assembles all components into a single nn.Module."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .vision_encoder import VisionEncoder
from .text_encoder import TextEncoder
from .fusion import CrossAttentionFusion
from .answer_heads import AnswerClassifier
from .scene_graph import SceneGraphEncoder


class VQAModel(nn.Module):
    """End-to-end Visual Question Answering model.

    Architecture:
        1. VisionEncoder   : CLIP ViT  → patch tokens + CLS
        2. TextEncoder     : BERT      → token embeddings + CLS
        3. SceneGraphEncoder (optional): GCN enrichment of patch tokens
        4. CrossAttentionFusion        : text tokens attend to image patches
        5. AnswerClassifier            : fused CLS → answer logits
    """

    def __init__(
        self,
        vision_backbone: str = "openai/clip-vit-large-patch14",
        text_encoder: str = "bert-base-uncased",
        hidden_dim: int = 768,
        num_heads: int = 8,
        fusion_layers: int = 4,
        num_answer_classes: int = 3129,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
        use_scene_graph: bool = False,
        mode: str = "multimodal",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.vision_encoder = VisionEncoder(
            vision_backbone, hidden_dim, gradient_checkpointing=gradient_checkpointing
        )
        self.text_encoder = TextEncoder(
            text_encoder, hidden_dim, gradient_checkpointing=gradient_checkpointing
        )
        self.scene_graph = SceneGraphEncoder(hidden_dim, hidden_dim) if use_scene_graph else None
        self.fusion = CrossAttentionFusion(hidden_dim, num_heads, fusion_layers, dropout)
        self.classifier = AnswerClassifier(hidden_dim, num_answer_classes, dropout)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values:   [B, 3, H, W]
            input_ids:      [B, seq_len]
            attention_mask: [B, seq_len]
            adj:            [B, N, N] optional scene-graph adjacency
        Returns:
            logits: [B, num_answer_classes]
        """
        _, patch_tokens = self.vision_encoder(pixel_values)
        _, token_embeds = self.text_encoder(input_ids, attention_mask)

        if self.mode == "text_only":
            patch_tokens = torch.zeros_like(patch_tokens)
        elif self.mode == "image_only":
            token_embeds = torch.zeros_like(token_embeds)

        if self.scene_graph is not None:
            patch_tokens = self.scene_graph(patch_tokens, adj)

        # Convert HuggingFace attention mask (1=attend, 0=pad)
        # to PyTorch key_padding_mask (True=ignore)
        key_padding_mask = attention_mask == 0

        fused = self.fusion(token_embeds, patch_tokens, key_padding_mask)
        return self.classifier(fused)

    @torch.no_grad()
    def predict(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        top_k: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience method returning top-k answer indices and probabilities."""
        logits = self.forward(pixel_values, input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        return probs.topk(top_k, dim=-1)
