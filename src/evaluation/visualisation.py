"""Visualisation utilities for inspection and error analysis."""
from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


_IMAGENET_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
_IMAGENET_STD = np.array([0.26862954, 0.26130258, 0.27577711])


def _denorm(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalisation to [0, 1] float array."""
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = img * _IMAGENET_STD + _IMAGENET_MEAN
    return np.clip(img, 0, 1)


def visualise_predictions(
    images: torch.Tensor,
    questions: List[str],
    pred_answers: List[str],
    gt_answers: Optional[List[str]] = None,
    max_samples: int = 5,
    save_path: Optional[str] = None,
) -> None:
    """Plot a grid of images with predicted (and optionally ground-truth) answers."""
    n = min(len(images), max_samples)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(_denorm(images[i]))
        ax.axis("off")
        title = f"Q: {questions[i]}\nPred: {pred_answers[i]}"
        if gt_answers:
            title += f"\nGT: {gt_answers[i]}"
        ax.set_title(title, fontsize=9, wrap=True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_answer_distribution(
    soft_scores: torch.Tensor,
    vocab,
    top_k: int = 10,
    save_path: Optional[str] = None,
) -> None:
    """Bar chart of the top-k answer scores for a single sample."""
    vals, idxs = soft_scores.topk(top_k)
    labels = [vocab.idx_to_answer(i.item()) for i in idxs]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels[::-1], vals.cpu().numpy()[::-1])
    ax.set_xlabel("Soft Score")
    ax.set_title("Top-K Answer Distribution")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
