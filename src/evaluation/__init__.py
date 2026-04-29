"""Evaluation utilities: official VQA metric, custom metrics, and visualisation."""
from .vqa_eval import VQAEvaluator, preprocess_answer
from .metrics import (
    soft_accuracy,
    top_k_accuracy,
    per_type_accuracy,
    compute_confusion_matrix,
    compute_top3_accuracy,
    per_category_accuracy,
    bias_analysis,
    ANSWER_TYPE_MAP,
    ANSWER_TYPE_NAMES,
)
from .visualisation import (
    visualise_predictions,
    plot_attention_heatmap,
    gradcam_heatmap,
    plot_scene_graph,
    plot_visual_grounding,
    plot_comparison_table,
    plot_per_category_bar,
)

__all__ = [
    # vqa_eval
    "VQAEvaluator",
    "preprocess_answer",
    # metrics
    "soft_accuracy",
    "top_k_accuracy",
    "per_type_accuracy",
    "compute_confusion_matrix",
    "compute_top3_accuracy",
    "per_category_accuracy",
    "bias_analysis",
    "ANSWER_TYPE_MAP",
    "ANSWER_TYPE_NAMES",
    # visualisation
    "visualise_predictions",
    "plot_attention_heatmap",
    "gradcam_heatmap",
    "plot_scene_graph",
    "plot_visual_grounding",
    "plot_comparison_table",
    "plot_per_category_bar",
]
