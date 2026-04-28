"""AnswerVocab — bidirectional answer ↔ index mapping with soft-score support."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import torch


class AnswerVocab:
    """Manages the closed-set of candidate answers used for classification.

    Soft scores follow the VQA v2 convention:
        score(answer) = min(count / 3, 1.0)
    where count is how many of the 10 human annotators gave that answer.
    """

    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, answers: List[str] | None = None) -> None:
        self._idx2ans: List[str] = [self.PAD, self.UNK]
        self._ans2idx: dict[str, int] = {self.PAD: 0, self.UNK: 1}
        if answers:
            for ans in answers:
                self._add(ans)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _add(self, answer: str) -> int:
        if answer not in self._ans2idx:
            self._ans2idx[answer] = len(self._idx2ans)
            self._idx2ans.append(answer)
        return self._ans2idx[answer]

    @classmethod
    def build_from_annotations(cls, annotations_path: str | Path, min_freq: int = 9) -> "AnswerVocab":
        """Build vocab from a local VQA v2 annotations JSON file."""
        import json
        data = json.loads(Path(annotations_path).read_text())
        counter: Counter = Counter()
        for ann in data["annotations"]:
            for ans in ann.get("answers", []):
                counter[ans["answer"]] += 1
        vocab = cls()
        for answer, freq in counter.most_common():
            if freq < min_freq:
                break
            vocab._add(answer)
        return vocab

    @classmethod
    def build_from_dataset(cls, hf_dataset, min_freq: int = 9) -> "AnswerVocab":
        """Scan an iterable of samples and keep answers that appear >= min_freq times."""
        counter: Counter = Counter()
        for sample in hf_dataset:
            for ans in sample.get("answers", []):
                counter[ans["answer"]] += 1
        vocab = cls()
        for answer, freq in counter.most_common():
            if freq < min_freq:
                break
            vocab._add(answer)
        return vocab

    @classmethod
    def load(cls, path: str | Path) -> "AnswerVocab":
        with open(path, "r") as f:
            data = json.load(f)
        vocab = cls()
        vocab._idx2ans = data["idx2ans"]
        vocab._ans2idx = {a: i for i, a in enumerate(vocab._idx2ans)}
        return vocab

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"idx2ans": self._idx2ans}, f, indent=2)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._idx2ans)

    def answer_to_idx(self, answer: str) -> int:
        return self._ans2idx.get(answer, self._ans2idx[self.UNK])

    def idx_to_answer(self, idx: int) -> str:
        return self._idx2ans[idx]

    # ------------------------------------------------------------------
    # Soft scores
    # ------------------------------------------------------------------

    def build_soft_scores(self, answers: List[dict]) -> torch.Tensor:
        """Convert a list of annotator answer dicts to a soft-score vector."""
        scores = torch.zeros(len(self), dtype=torch.float32)
        counter: Counter = Counter(a["answer"] for a in answers)
        for answer, count in counter.items():
            idx = self.answer_to_idx(answer)
            if idx != self._ans2idx[self.UNK]:
                scores[idx] = min(count / 3.0, 1.0)
        return scores
