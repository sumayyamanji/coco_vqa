"""Loss functions for VQA training."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQALoss(nn.Module):
    """Soft binary cross-entropy as used in the original VQA v2 paper.

    Each answer class has a soft target in [0, 1] derived from human consensus,
    rather than a hard one-hot label.  This formulation has been shown to
    produce better-calibrated models than standard cross-entropy.
    """

    def __init__(self, label_smoothing: float = 0.1) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, soft_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:      [B, num_classes]
            soft_scores: [B, num_classes]  values in [0, 1]
        Returns:
            scalar loss
        """
        if self.label_smoothing > 0.0:
            num_classes = soft_scores.size(-1)
            soft_scores = (
                soft_scores * (1.0 - self.label_smoothing)
                + self.label_smoothing / num_classes
            )
        log_probs = F.log_softmax(logits, dim=-1)
        return -(soft_scores * log_probs).sum(dim=-1).mean()
