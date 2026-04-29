"""Loss functions for VQA training."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Original loss (kept for backward compat)
# ---------------------------------------------------------------------------

class VQALoss(nn.Module):
    """Soft cross-entropy loss as used in the original VQA v2 paper (legacy)."""

    def __init__(self, label_smoothing: float = 0.1) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, soft_scores: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0.0:
            num_classes = soft_scores.size(-1)
            soft_scores = (
                soft_scores * (1.0 - self.label_smoothing)
                + self.label_smoothing / num_classes
            )
        log_probs = F.log_softmax(logits, dim=-1)
        return -(soft_scores * log_probs).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# New losses
# ---------------------------------------------------------------------------

class VQASoftLoss(nn.Module):
    """Soft binary cross-entropy — the standard VQA v2 training objective.

    Each answer class has a soft target t_c = min(count_c / 3, 1.0) derived
    from annotator consensus.  BCEWithLogitsLoss treats every class as an
    independent binary prediction, which matches the multi-label nature of VQA
    (multiple valid answers per question) better than hard cross-entropy.

    Label smoothing is applied manually before passing targets to BCE:
        t̃_c = t_c * (1 − α) + α / C
    where α = label_smoothing and C = num_answer_classes.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.label_smoothing: float = config["training"]["label_smoothing"]

    def forward(self, logits: torch.Tensor, soft_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:      (B, C) raw model logits
            soft_scores: (B, C) soft targets in [0, 1]  (already in VQA scale)
        Returns:
            scalar BCE loss averaged over all (B, C) elements
        """
        if self.label_smoothing > 0.0:
            num_classes = soft_scores.size(-1)
            soft_scores = (
                soft_scores * (1.0 - self.label_smoothing)
                + self.label_smoothing / num_classes
            )
        return F.binary_cross_entropy_with_logits(logits, soft_scores)


class AnswerTypeLoss(nn.Module):
    """Standard cross-entropy for the 3-way answer-type classifier.

    Class mapping: 0 = yes/no  |  1 = number  |  2 = other
    """

    def __init__(self) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, type_logits: torch.Tensor, type_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            type_logits: (B, 3)  raw logits from AnswerTypeClassifier
            type_labels: (B,)    integer class indices in {0, 1, 2}
        Returns:
            scalar cross-entropy loss
        """
        return self.ce(type_logits, type_labels)


class TotalLoss(nn.Module):
    """Weighted combination of the two training objectives.

    total = 1.0 * vqa_loss + 0.5 * type_loss
    """

    VQA_WEIGHT: float = 1.0
    TYPE_WEIGHT: float = 0.5

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.vqa_loss = VQASoftLoss(config)
        self.type_loss = AnswerTypeLoss()

    def forward(
        self,
        answer_logits: torch.Tensor,
        type_logits: torch.Tensor,
        answer_scores: torch.Tensor,
        type_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            answer_logits:  (B, num_classes)  open-ended head output
            type_logits:    (B, 3)            type classifier output
            answer_scores:  (B, num_classes)  VQA soft targets from dataset
            type_labels:    (B,)              integer type class indices
        Returns:
            (total_loss, {"vqa_loss": float, "type_loss": float})
        """
        vqa = self.vqa_loss(answer_logits, answer_scores)
        typ = self.type_loss(type_logits, type_labels)
        total = self.VQA_WEIGHT * vqa + self.TYPE_WEIGHT * typ
        return total, {"vqa_loss": float(vqa), "type_loss": float(typ)}
