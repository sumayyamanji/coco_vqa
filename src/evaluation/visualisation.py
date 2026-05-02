"""Visualisation utilities for inspection, attention analysis, and error analysis."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend safe for server / notebook export
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
_CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _denorm(tensor: torch.Tensor) -> np.ndarray:
    """Reverse CLIP/ImageNet normalisation → float32 array in [0, 1]."""
    img = tensor.detach().cpu().float().permute(1, 2, 0).numpy()
    return np.clip(img * _CLIP_STD + _CLIP_MEAN, 0.0, 1.0)


def _pil_to_np(img: Any) -> np.ndarray:
    """Accept PIL Image or (C,H,W) tensor; return H×W×3 float32 array in [0,1]."""
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    if isinstance(img, torch.Tensor):
        return _denorm(img)
    return np.asarray(img).astype(np.float32) / 255.0


def _patches_to_heatmap(
    attn: np.ndarray, img_h: int, img_w: int, grid_size: int = 14
) -> np.ndarray:
    """Reshape flat patch attention (N,) → bicubic-upsampled heatmap (img_h, img_w)."""
    side = int(math.sqrt(len(attn))) if grid_size is None else grid_size
    am = attn[: side * side].reshape(side, side).astype(np.float32)
    am = (am - am.min()) / (am.max() - am.min() + 1e-8)
    # PIL bicubic resize
    am_pil = Image.fromarray((am * 255).astype(np.uint8)).resize(
        (img_w, img_h), Image.BICUBIC
    )
    return np.asarray(am_pil).astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# 1. Attention heatmap overlay
# ---------------------------------------------------------------------------

def plot_attention_heatmap(
    image: Any,
    attention_weights: torch.Tensor,
    question: str,
    answer: str,
    alpha: float = 0.55,
    cmap: str = "plasma",
    grid_size: int = 14,
) -> plt.Figure:
    """Overlay cross-attention weights on the original image.

    Args:
        image:             PIL Image or (C,H,W) normalised tensor
        attention_weights: (num_patches,) or (T, num_patches) — if 2-D,
                           the mean across question tokens is used
        question:          question string (shown in title)
        answer:            predicted answer (shown in title)
        alpha:             heatmap opacity
        cmap:              matplotlib colourmap name
        grid_size:         side length of the patch grid (14 for ViT-L/14)
    Returns:
        matplotlib Figure
    """
    img_np = _pil_to_np(image)
    H, W = img_np.shape[:2]

    attn = attention_weights.detach().cpu().float()
    if attn.dim() == 2:
        attn = attn.mean(dim=0)   # (T, N) → (N,)
    attn_np = attn.numpy()

    heatmap = _patches_to_heatmap(attn_np, H, W, grid_size)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_np)
    axes[0].set_title("Original image", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(img_np)
    axes[1].imshow(heatmap, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    axes[1].set_title(
        f"Q: {question[:60]}\nA: {answer}", fontsize=9, wrap=True
    )
    axes[1].axis("off")

    plt.colorbar(
        plt.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(0, 1),
            cmap=cmap,
        ),
        ax=axes[1], fraction=0.046, pad=0.04, label="attention",
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Grad-CAM heatmap
# ---------------------------------------------------------------------------

def gradcam_heatmap(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    question_ids: torch.Tensor,
    target_class: int,
    attention_mask: Optional[torch.Tensor] = None,
    grid_size: int = 14,
    alpha: float = 0.6,
) -> plt.Figure:
    """Compute Grad-CAM for the answer logit of *target_class* and overlay.

    Hooks into the last CLIP transformer encoder block to capture activations
    and gradients.  Works with the ``CLIPVisionEncoder`` from this project.

    Args:
        model:           VQAModel (in eval mode; gradients enabled)
        image_tensor:    (1, C, H, W) normalised tensor
        question_ids:    (1, seq_len) token ids
        target_class:    answer vocabulary index to explain
        attention_mask:  (1, seq_len); ones if None
        grid_size:       patch grid side length (14 for ViT-L/14)
        alpha:           heatmap overlay opacity
    Returns:
        matplotlib Figure with Grad-CAM overlay
    """
    activations: Dict[str, torch.Tensor] = {}
    gradients: Dict[str, torch.Tensor] = {}

    def _fwd_hook(module, inp, out):
        a = out[0] if isinstance(out, tuple) else out
        activations["last"] = a.detach()

    def _bwd_hook(module, gin, gout):
        g = gout[0] if isinstance(gout, tuple) else gout
        if g is not None:
            gradients["last"] = g.detach()

    # Locate last CLIP encoder layer
    try:
        last_block = (
            model.vision_encoder.backbone.vision_model.encoder.layers[-1]
        )
    except AttributeError:
        raise RuntimeError(
            "Cannot locate last CLIP encoder block. "
            "Expected model.vision_encoder.backbone.vision_model.encoder.layers"
        )

    fwd_h = last_block.register_forward_hook(_fwd_hook)
    bwd_h = last_block.register_full_backward_hook(_bwd_hook)

    was_training = model.training
    model.eval()

    try:
        if attention_mask is None:
            attention_mask = torch.ones(
                1, question_ids.size(1),
                dtype=torch.long, device=question_ids.device,
            )
        batch = {
            "image_tensor": image_tensor,
            "question_ids": question_ids,
            "attention_mask": attention_mask,
        }
        out = model(batch)
        logit = out["answer_logits"][0, target_class]
        model.zero_grad()
        logit.backward()
    finally:
        fwd_h.remove()
        bwd_h.remove()
        if was_training:
            model.train()

    # Compute Grad-CAM from patch tokens (skip CLS at index 0)
    acts = activations.get("last")          # (1, N+1, D)
    grads = gradients.get("last")           # (1, N+1, D)
    if acts is None or grads is None:
        raise RuntimeError("Hooks did not capture activations / gradients.")

    acts = acts[0, 1:]                      # (N, D)
    grads = grads[0, 1:]                    # (N, D)
    weights = grads.mean(dim=-1)            # (N,)
    cam = F.relu((weights.unsqueeze(-1) * acts).sum(dim=-1))  # (N,)

    img_np = _denorm(image_tensor.squeeze(0))
    H, W = img_np.shape[:2]
    heatmap = _patches_to_heatmap(cam.cpu().numpy(), H, W, grid_size)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(img_np)
    ax.imshow(heatmap, cmap="jet", alpha=alpha, vmin=0, vmax=1)
    ax.set_title(f"Grad-CAM — class {target_class}", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Scene-graph visualisation
# ---------------------------------------------------------------------------

def plot_scene_graph(
    scene_graph_output: torch.Tensor,
    object_labels: Optional[Sequence[str]] = None,
    max_nodes: int = 20,
    relation_threshold: float = 0.5,
) -> plt.Figure:
    """Draw a scene graph from the SceneGraphGenerator output.

    Node pairs with high cosine similarity are connected with edges labelled
    with the dominant relation type (approximated from embedding similarity).

    Args:
        scene_graph_output: (N, D) relation embeddings from SceneGraphGenerator
        object_labels:      optional list of N node label strings
        max_nodes:          cap the number of nodes drawn for readability
        relation_threshold: cosine similarity threshold to draw an edge
    Returns:
        matplotlib Figure
    """
    try:
        import networkx as nx
    except ImportError:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "networkx not installed", ha="center", va="center")
        return fig

    emb = scene_graph_output.detach().cpu().float()
    N = min(emb.size(0), max_nodes)
    emb = emb[:N]

    # Node labels
    if object_labels is not None:
        labels = {i: str(object_labels[i]) for i in range(N)}
    else:
        labels = {i: f"patch {i}" for i in range(N)}

    # Build graph: connect nodes with high embedding similarity
    G = nx.Graph()
    G.add_nodes_from(range(N))
    norm = F.normalize(emb, dim=-1)
    sim = (norm @ norm.T).numpy()
    for i in range(N):
        for j in range(i + 1, N):
            if sim[i, j] > relation_threshold:
                G.add_edge(i, j, weight=float(sim[i, j]))

    # Colour nodes by rough category (cycle through palette)
    palette = plt.cm.tab20.colors
    node_colors = [palette[i % len(palette)] for i in range(N)]

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, ax=ax)
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)
    edges = G.edges(data=True)
    if edges:
        weights = [d["weight"] for _, _, d in edges]
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6, ax=ax)
    ax.set_title("Scene Graph (patch co-embedding similarity)", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Visual grounding via peak attention patch
# ---------------------------------------------------------------------------

def plot_visual_grounding(
    image: Any,
    cross_attention_weights: torch.Tensor,
    question: str,
    answer: str,
    grid_size: int = 14,
    patch_expand: int = 2,
) -> plt.Figure:
    """Highlight the most attended image region with a bounding box.

    Finds the peak-attention patch, expands it by ``patch_expand`` patches in
    each direction, and draws a red rectangle around that region.

    Args:
        image:                  PIL Image or (C, H, W) normalised tensor
        cross_attention_weights:(T, N) or (N,) attention weights
        question:               question string
        answer:                 predicted answer string
        grid_size:              patch grid side length
        patch_expand:           how many patches to expand around the peak
    Returns:
        matplotlib Figure
    """
    img_np = _pil_to_np(image)
    H, W = img_np.shape[:2]
    patch_h = H / grid_size
    patch_w = W / grid_size

    attn = cross_attention_weights.detach().cpu().float()
    if attn.dim() == 2:
        attn = attn.mean(dim=0)       # (T, N) → (N,)
    attn_np = attn.numpy()

    peak = int(np.argmax(attn_np))
    peak_row = peak // grid_size
    peak_col = peak % grid_size

    r0 = max(0, peak_row - patch_expand)
    c0 = max(0, peak_col - patch_expand)
    r1 = min(grid_size, peak_row + patch_expand + 1)
    c1 = min(grid_size, peak_col + patch_expand + 1)

    x0, y0 = c0 * patch_w, r0 * patch_h
    bw, bh = (c1 - c0) * patch_w, (r1 - r0) * patch_h

    # Heatmap background
    heatmap = _patches_to_heatmap(attn_np, H, W, grid_size)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].imshow(img_np)
    rect = mpatches.FancyBboxPatch(
        (x0, y0), bw, bh,
        boxstyle="round,pad=2", linewidth=2,
        edgecolor="red", facecolor="none",
    )
    axes[0].add_patch(rect)
    axes[0].set_title(
        f"Q: {question[:55]}\nA: {answer}", fontsize=8, wrap=True
    )
    axes[0].axis("off")

    axes[1].imshow(img_np)
    axes[1].imshow(heatmap, cmap="plasma", alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title("Cross-attention heatmap", fontsize=9)
    axes[1].axis("off")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Comparison table
# ---------------------------------------------------------------------------

def plot_comparison_table(
    text_only_results: Dict[str, float],
    multimodal_results: Dict[str, float],
    answer_types: Sequence[str] = ("yes/no", "number", "other"),
    image_only_results: Optional[Dict[str, float]] = None,
) -> plt.Figure:
    """Matplotlib table comparing per-type accuracy across model variants.

    Args:
        text_only_results:   {"accuracy": float, "per_type": {type: float}}
        multimodal_results:  same structure
        answer_types:        column order for answer-type sub-scores
        image_only_results:  optional; same structure
    Returns:
        matplotlib Figure
    """
    rows = []
    row_labels = []

    def _extract(res: dict) -> List[str]:
        overall = f"{100 * res.get('accuracy', 0):.1f}%"
        pt = res.get("per_type", {})
        return [overall] + [f"{100 * pt.get(t, 0):.1f}%" for t in answer_types]

    if text_only_results:
        rows.append(_extract(text_only_results))
        row_labels.append("Text-Only")
    if image_only_results:
        rows.append(_extract(image_only_results))
        row_labels.append("Image-Only")
    if multimodal_results:
        rows.append(_extract(multimodal_results))
        row_labels.append("Multimodal")

    col_labels = ["Overall"] + list(answer_types)

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(col_labels)), 1 + len(rows)))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.3, 1.8)

    # Colour the "Multimodal" row header
    for j in range(len(col_labels)):
        cell = tbl[(len(rows), j)]
        cell.set_facecolor("#d4edda")

    ax.set_title("Model comparison — VQA accuracy", fontsize=12, pad=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Per-category bar chart
# ---------------------------------------------------------------------------

def plot_per_category_bar(
    category_accuracies: Dict[str, float],
    title: str = "VQA accuracy by COCO supercategory",
) -> plt.Figure:
    """Horizontal bar chart coloured green→red by accuracy level.

    Args:
        category_accuracies: {supercategory: accuracy_float}
        title:               chart title
    Returns:
        matplotlib Figure
    """
    cats = sorted(category_accuracies, key=category_accuracies.get, reverse=True)
    vals = [category_accuracies[c] for c in cats]

    # Colour: green for high accuracy, red for low
    cmap = plt.cm.RdYlGn
    colors = [cmap(v) for v in vals]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(cats))))
    bars = ax.barh(cats, vals, color=colors, edgecolor="white")
    ax.set_xlim(0, 1)
    ax.set_xlabel("VQA accuracy")
    ax.set_title(title, fontsize=11)

    for bar, v in zip(bars, vals):
        ax.text(
            min(v + 0.01, 0.98), bar.get_y() + bar.get_height() / 2,
            f"{100 * v:.1f}%",
            va="center", ha="left", fontsize=8,
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compat)
# ---------------------------------------------------------------------------

def visualise_predictions(
    images: torch.Tensor,
    questions: List[str],
    pred_answers: List[str],
    gt_answers: Optional[List[str]] = None,
    max_samples: int = 5,
    save_path: Optional[str] = None,
) -> None:
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
