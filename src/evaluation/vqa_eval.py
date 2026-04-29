"""Official VQA v2 evaluation metric with faithful answer pre-processing."""
from __future__ import annotations

import re
import string
from collections import defaultdict
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Pre-processing tables (mirrors the official VQA evaluation code)
# ---------------------------------------------------------------------------

_CONTRACTIONS: Dict[str, str] = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "im": "i'm", "isnt": "isn't", "itd": "it'd", "itll": "it'll",
    "ive": "i've", "mightnt": "mightn't", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "oclock": "o'clock", "oughtnt": "oughtn't", "shant": "shan't",
    "shouldve": "should've", "shouldnt": "shouldn't", "somebodys": "somebody's",
    "someones": "someone's", "thats": "that's", "thered": "there'd",
    "therere": "there're", "theres": "there's", "theyd": "they'd",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've",
    "twas": "'twas", "wasnt": "wasn't", "wed": "we'd", "were": "we're",
    "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've",
    "whens": "when's", "whered": "where'd", "wheres": "where's",
    "whereve": "where've", "whod": "who'd", "wholl": "who'll",
    "whos": "who's", "whove": "who've", "whys": "why's",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "youd": "you'd", "youll": "you'll", "youre": "you're", "youve": "you've",
}

# Word-form numbers → digit strings (official VQA normalisation)
_NUMBER_MAP: Dict[str, str] = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10",
}

_ARTICLES: frozenset = frozenset({"a", "an", "the"})

# Strip lone periods (not between digits)
_PERIOD_STRIP = re.compile(r"(?<!\d)\.(?!\d)")
# Remove commas between digits: "1,000" → "1000"
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
# Punctuation to remove (keep "." for floating-point numbers)
_PUNCT: str = ";/[]\"{}()=+\\-_><@`?!"


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def preprocess_answer(answer: str) -> str:
    """Normalise a raw answer string to match the official VQA scoring convention.

    Steps (in order):
    1. Lowercase
    2. Expand contractions (e.g. "wont" → "won't")
    3. Remove commas embedded in numbers
    4. Strip punctuation (preserving periods adjacent to digits)
    5. Remove lone periods
    6. Remove articles (a, an, the)
    7. Normalise number words to digit form (e.g. "two" → "2")
    """
    answer = answer.lower().strip()

    # 1. Expand contractions
    tokens = answer.split()
    tokens = [_CONTRACTIONS.get(t, t) for t in tokens]
    answer = " ".join(tokens)

    # 2. Comma strip inside numbers
    answer = _COMMA_STRIP.sub(r"\1\3", answer)

    # 3. Punctuation strip
    for ch in _PUNCT:
        answer = answer.replace(ch, " ")

    # 4. Period strip (lone periods)
    answer = _PERIOD_STRIP.sub("", answer)

    # 5. Remove articles and normalise number words
    out_tokens = []
    for w in answer.split():
        if w in _ARTICLES:
            continue
        out_tokens.append(_NUMBER_MAP.get(w, w))

    return " ".join(out_tokens).strip()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class VQAEvaluator:
    """Computes the official VQA v2 accuracy metric.

    Soft scoring: for a predicted answer *a*, the score against a question
    with 10 human annotations is:

        score(a) = min(1, count(a in annotations) / 3)

    Both the predicted answer and each human annotation are preprocessed with
    :func:`preprocess_answer` before comparison so that surface variation
    (punctuation, articles, number words, contractions) does not penalise
    equivalent answers.

    Usage
    -----
    Accumulate batches during inference, then call :meth:`summarise`::

        evaluator = VQAEvaluator()
        for batch in val_loader:
            evaluator.process_batch(question_ids, logits, soft_scores)
        print(evaluator.summarise())

    Or evaluate a list of string predictions directly::

        results = evaluator.evaluate(predictions, annotations_dict)
    """

    def __init__(self) -> None:
        self._results: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self._results.clear()

    # ------------------------------------------------------------------
    # Batch-level accumulation (logit-based, for use during training loop)
    # ------------------------------------------------------------------

    def process_batch(
        self,
        question_ids: List[int],
        logits,   # torch.Tensor (B, C)
        soft_scores,  # torch.Tensor (B, C)
    ) -> None:
        """Accumulate soft-score results from a batch of logit predictions."""
        import torch
        pred_indices = logits.argmax(dim=-1).cpu().tolist()
        for qid, pred_idx, gt in zip(question_ids, pred_indices, soft_scores):
            score = float(gt[pred_idx].clamp(max=1.0))
            self._results.append({"question_id": int(qid), "score": score})

    def compute(self) -> float:
        if not self._results:
            return 0.0
        return sum(r["score"] for r in self._results) / len(self._results)

    def summarise(self) -> Dict[str, Any]:
        return {"vqa_accuracy": self.compute(), "num_questions": len(self._results)}

    # ------------------------------------------------------------------
    # String-prediction evaluation (official metric, end-to-end)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        annotations: Any,
    ) -> Dict[str, Any]:
        """Evaluate a list of string predictions against the VQA annotations.

        Args:
            predictions:  list of {"question_id": int, "answer": str}
            annotations:  either the raw annotations JSON dict
                          (with key "annotations") or a list of annotation
                          dicts each containing "question_id", "answers",
                          and optionally "answer_type" and "image_id".
        Returns:
            {
              "accuracy":       float   overall VQA accuracy,
              "per_type":       dict    accuracy per answer_type,
              "n_questions":    int,
              "per_question":   list    [{"question_id", "score", "answer_type"}]
            }
        """
        ann_list: List[dict] = (
            annotations.get("annotations", [])
            if isinstance(annotations, dict)
            else list(annotations)
        )
        ann_lookup: Dict[int, dict] = {a["question_id"]: a for a in ann_list}

        total = 0.0
        type_scores: Dict[str, List[float]] = defaultdict(list)
        per_question: List[dict] = []

        for pred in predictions:
            qid = int(pred["question_id"])
            pred_answer = preprocess_answer(str(pred.get("answer", "")))

            ann = ann_lookup.get(qid)
            if ann is None:
                continue

            gt_answers = [
                preprocess_answer(a["answer"]) for a in ann.get("answers", [])
            ]
            count = gt_answers.count(pred_answer)
            score = min(1.0, count / 3.0)

            answer_type: str = ann.get("answer_type", "other")
            total += score
            type_scores[answer_type].append(score)
            per_question.append({
                "question_id": qid,
                "score": score,
                "answer_type": answer_type,
                "image_id": ann.get("image_id"),
                "predicted": pred_answer,
            })

        n = len(per_question)
        accuracy = total / max(n, 1)
        per_type = {t: sum(v) / len(v) for t, v in type_scores.items()}

        return {
            "accuracy": accuracy,
            "per_type": per_type,
            "n_questions": n,
            "per_question": per_question,
        }
