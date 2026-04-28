"""Supplementary accuracy metrics beyond the official VQA score."""
from __future__ import annotations

import torch


def soft_accuracy(logits: torch.Tensor, soft_scores: torch.Tensor) -> float:
    """Average soft score of the top-1 predicted answer."""
    preds = logits.argmax(dim=-1)
    scores = soft_scores[torch.arange(len(preds)), preds]
    return float(scores.mean())


def top_k_accuracy(
    logits: torch.Tensor,
    soft_scores: torch.Tensor,
    k: int = 3,
) -> float:
    """Soft accuracy when any of the top-k predictions matches."""
    topk_indices = logits.topk(k, dim=-1).indices  # [B, k]
    best_scores = soft_scores.gather(1, topk_indices).max(dim=-1).values
    return float(best_scores.mean())


def per_type_accuracy(
    logits: torch.Tensor,
    soft_scores: torch.Tensor,
    question_types: list,
) -> dict:
    """Break down soft accuracy by question type (yes/no, number, other)."""
    from collections import defaultdict

    preds = logits.argmax(dim=-1)
    type_scores: dict = defaultdict(list)
    for i, qtype in enumerate(question_types):
        score = float(soft_scores[i, preds[i]].clamp(max=1.0))
        type_scores[qtype].append(score)
    return {qtype: sum(v) / len(v) for qtype, v in type_scores.items()}
