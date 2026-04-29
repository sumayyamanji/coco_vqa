"""Supplementary accuracy metrics beyond the official VQA score."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Original helpers (kept for backward compat)
# ---------------------------------------------------------------------------

def soft_accuracy(logits: torch.Tensor, soft_scores: torch.Tensor) -> float:
    """Average soft score of the top-1 predicted answer (logit-based)."""
    preds = logits.argmax(dim=-1)
    scores = soft_scores[torch.arange(len(preds)), preds]
    return float(scores.mean())


def top_k_accuracy(
    logits: torch.Tensor,
    soft_scores: torch.Tensor,
    k: int = 3,
) -> float:
    """Soft accuracy when the best of the top-k predictions is used."""
    topk_indices = logits.topk(k, dim=-1).indices
    best_scores = soft_scores.gather(1, topk_indices).max(dim=-1).values
    return float(best_scores.mean())


def per_type_accuracy(
    logits: torch.Tensor,
    soft_scores: torch.Tensor,
    question_types: list,
) -> dict:
    """Break down soft accuracy by question type (yes/no, number, other)."""
    preds = logits.argmax(dim=-1)
    type_scores: dict = defaultdict(list)
    for i, qtype in enumerate(question_types):
        score = float(soft_scores[i, preds[i]].clamp(max=1.0))
        type_scores[qtype].append(score)
    return {qtype: sum(v) / len(v) for qtype, v in type_scores.items()}


# ---------------------------------------------------------------------------
# New metrics
# ---------------------------------------------------------------------------

#: Canonical answer-type label → integer index
ANSWER_TYPE_MAP: Dict[str, int] = {"yes/no": 0, "number": 1, "other": 2}
ANSWER_TYPE_NAMES: List[str] = ["yes/no", "number", "other"]


def compute_confusion_matrix(
    y_true_types: List[str],
    y_pred_types: List[Any],
) -> np.ndarray:
    """3×3 confusion matrix for the answer-type classifier.

    Args:
        y_true_types: ground-truth type strings ("yes/no", "number", "other")
        y_pred_types: predicted type — either strings or integer class indices
                      in {0, 1, 2}
    Returns:
        cm: np.ndarray of shape (3, 3), dtype int.
            Rows = true class, columns = predicted class.
    """
    n = len(ANSWER_TYPE_NAMES)
    cm = np.zeros((n, n), dtype=int)
    for true_str, pred in zip(y_true_types, y_pred_types):
        t = ANSWER_TYPE_MAP.get(str(true_str), 2)
        if isinstance(pred, str):
            p = ANSWER_TYPE_MAP.get(pred, 2)
        else:
            p = int(pred)
        cm[t, p] += 1
    return cm


def compute_top3_accuracy(
    predictions: List[Dict[str, Any]],
    annotations: Any,
) -> float:
    """Accuracy when any of the top-3 predicted answers is correct.

    Args:
        predictions:  list of {"question_id": int,
                               "top3_answers": [str, str, str]}
        annotations:  VQA annotations dict (key "annotations") or list
    Returns:
        float  mean score across all questions
    """
    from .vqa_eval import preprocess_answer

    ann_list: List[dict] = (
        annotations.get("annotations", [])
        if isinstance(annotations, dict)
        else list(annotations)
    )
    ann_lookup = {a["question_id"]: a for a in ann_list}

    total = 0.0
    n = 0
    for pred in predictions:
        qid = int(pred["question_id"])
        top3: List[str] = pred.get("top3_answers", [])
        ann = ann_lookup.get(qid)
        if ann is None or not top3:
            continue

        gt_answers = [preprocess_answer(a["answer"]) for a in ann.get("answers", [])]
        best = 0.0
        for answer in top3:
            count = gt_answers.count(preprocess_answer(answer))
            best = max(best, min(1.0, count / 3.0))

        total += best
        n += 1

    return total / max(n, 1)


def per_category_accuracy(
    predictions: List[Dict[str, Any]],
    vqa_annotations: Any,
    coco_annotations: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Accuracy grouped by COCO object supercategory.

    Maps each question → image, then looks up the COCO supercategories of
    objects in that image.  Questions not matched to any category fall into
    "uncategorised".

    Args:
        predictions:      list of {"question_id": int, "score": float}
        vqa_annotations:  VQA annotations dict/list (to get image_id per qid)
        coco_annotations: COCO *instance* annotations dict
                          (keys: "categories", "annotations").
                          If None, groups only by "uncategorised".
    Returns:
        dict  {supercategory_name: mean_accuracy}
    """
    # Build qid → image_id from VQA annotations
    vqa_list: List[dict] = (
        vqa_annotations.get("annotations", [])
        if isinstance(vqa_annotations, dict)
        else list(vqa_annotations)
    )
    qid_to_imgid: Dict[int, int] = {a["question_id"]: a["image_id"] for a in vqa_list}

    # Build image_id → set of supercategories from COCO instance annotations
    img_to_supers: Dict[int, List[str]] = defaultdict(list)
    if coco_annotations is not None:
        cat_to_super: Dict[int, str] = {
            c["id"]: c.get("supercategory", c.get("name", "other"))
            for c in coco_annotations.get("categories", [])
        }
        for ann in coco_annotations.get("annotations", []):
            iid = ann["image_id"]
            supercat = cat_to_super.get(ann.get("category_id", -1), "other")
            if supercat not in img_to_supers[iid]:
                img_to_supers[iid].append(supercat)

    # Group scores by supercategory
    cat_scores: Dict[str, List[float]] = defaultdict(list)
    for pred in predictions:
        qid = int(pred["question_id"])
        score = float(pred.get("score", 0.0))
        iid = qid_to_imgid.get(qid)
        supercats = img_to_supers.get(iid, ["uncategorised"]) if iid else ["uncategorised"]
        for sc in supercats:
            cat_scores[sc].append(score)

    return {cat: sum(v) / len(v) for cat, v in sorted(cat_scores.items())}


def bias_analysis(
    text_only_preds: Dict[int, Dict[str, Any]],
    multimodal_preds: Dict[int, Dict[str, Any]],
    annotations: Any,
    example_limit: int = 5,
) -> Dict[str, Any]:
    """Analyse language bias by comparing text-only vs multimodal predictions.

    For each question classifies the outcome as one of:
      * **language_bias**  — text-only correct, multimodal wrong
        (model can answer from language statistics alone)
      * **multimodal_gain** — multimodal strictly better than text-only
        (visual grounding added measurable value)
      * **both_correct**   — both models answered correctly
      * **both_fail**      — both models failed

    Args:
        text_only_preds:   {question_id: {"answer": str, "score": float}}
        multimodal_preds:  {question_id: {"answer": str, "score": float}}
        annotations:       VQA annotations dict/list (used for question text)
        example_limit:     how many example question_ids to include per group
    Returns:
        dict with keys:
          language_bias_count, multimodal_gain_count,
          both_correct_count, both_fail_count,
          and *_examples lists (question_ids)
    """
    vqa_list: List[dict] = (
        annotations.get("annotations", [])
        if isinstance(annotations, dict)
        else list(annotations)
    )
    all_qids = {a["question_id"] for a in vqa_list}

    groups: Dict[str, List[int]] = defaultdict(list)

    for qid in all_qids:
        t_score = text_only_preds.get(qid, {}).get("score", 0.0)
        m_score = multimodal_preds.get(qid, {}).get("score", 0.0)

        t_correct = t_score > 0.0
        m_correct = m_score > 0.0

        if t_correct and not m_correct:
            groups["language_bias"].append(qid)
        elif m_score > t_score:
            groups["multimodal_gain"].append(qid)
        elif t_correct and m_correct:
            groups["both_correct"].append(qid)
        else:
            groups["both_fail"].append(qid)

    return {
        "language_bias_count": len(groups["language_bias"]),
        "multimodal_gain_count": len(groups["multimodal_gain"]),
        "both_correct_count": len(groups["both_correct"]),
        "both_fail_count": len(groups["both_fail"]),
        "language_bias_examples": groups["language_bias"][:example_limit],
        "multimodal_gain_examples": groups["multimodal_gain"][:example_limit],
        "both_correct_examples": groups["both_correct"][:example_limit],
        "both_fail_examples": groups["both_fail"][:example_limit],
    }
