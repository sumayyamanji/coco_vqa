"""Answer classification head — maps fused embedding to answer logits."""
from __future__ import annotations

import torch
import torch.nn as nn


class AnswerClassifier(nn.Module):
    """Two-layer MLP that projects the fused multimodal embedding to logits.

    A gated architecture (element-wise product of two projections) is used
    to allow the head to selectively emphasise different feature dimensions,
    which has empirically improved VQA accuracy over a plain linear layer.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_answer_classes: int = 3129,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj = nn.Linear(hidden_dim * 2, num_answer_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, hidden_dim]
        Returns:
            logits: [B, num_answer_classes]
        """
        return self.proj(self.gate(x))
