"""Official VQA evaluation wrapper (mirrors the VQA v2 eval server logic)."""
from __future__ import annotations

from collections import defaultdict
from typing import List

import torch


class VQAEvaluator:
    """Computes the VQA v2 accuracy metric on a full evaluation split.

    VQA accuracy is soft: a predicted answer is scored against the set of
    10 human answers as  min(count_of_answer_in_annotations / 3, 1.0).
    The final score is the average over all questions.
    """

    def __init__(self, vocab) -> None:
        self.vocab = vocab
        self._results: List[dict] = []

    def reset(self) -> None:
        self._results.clear()

    def process_batch(
        self,
        question_ids: List[int],
        logits: torch.Tensor,
        soft_scores: torch.Tensor,
    ) -> None:
        """Accumulate predictions for a single batch."""
        pred_indices = logits.argmax(dim=-1).cpu().tolist()
        for qid, pred_idx, gt_scores in zip(question_ids, pred_indices, soft_scores):
            score = float(gt_scores[pred_idx].clamp(max=1.0))
            self._results.append({"question_id": qid, "score": score})

    def compute(self) -> float:
        """Return mean VQA accuracy over all accumulated batches."""
        if not self._results:
            return 0.0
        return sum(r["score"] for r in self._results) / len(self._results)

    def summarise(self) -> dict:
        return {
            "vqa_accuracy": self.compute(),
            "num_questions": len(self._results),
        }
