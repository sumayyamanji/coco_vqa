"""LocalVQADataset — reads VQA v2 JSON annotations and COCO images from disk."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


class LocalVQADataset(Dataset):
    """PyTorch Dataset over local VQA v2 JSON files + COCO image folders.

    Expected JSON layouts
    ---------------------
    questions_path : {"questions": [{"question_id": int, "image_id": int, "question": str}, ...]}
    annotations_path : {"annotations": [{"question_id": int, "image_id": int,
                                          "answers": [{"answer": str, ...}, ...], ...}]}

    Image filenames follow the official COCO convention:
        COCO_{split}2014_{image_id:012d}.jpg
    where split is "train" or "val".

    Each sample returns
    -------------------
        image          : float32 tensor [C, H, W]
        input_ids      : int64  tensor [seq_len]
        attention_mask : int64  tensor [seq_len]
        label          : int64  scalar — argmax of soft scores (-1 if unknown)
        soft_scores    : float32 tensor [num_answer_classes]
        question_id    : int
    """

    def __init__(
        self,
        annotations_path: str | Path,
        questions_path: str | Path,
        images_dir: str | Path,
        split: str,
        vocab,
        image_transform: Optional[Callable] = None,
        tokenizer=None,
        max_question_length: int = 30,
        debug: bool = False,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.split = split
        self.vocab = vocab
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.max_question_length = max_question_length

        questions_raw = json.loads(Path(questions_path).read_text())["questions"]
        annotations_raw = json.loads(Path(annotations_path).read_text())["annotations"]

        # Build lookup: question_id → answers list
        ann_lookup: dict[int, List[dict]] = {
            a["question_id"]: a["answers"] for a in annotations_raw
        }

        self.samples: List[Tuple[int, int, str, List[dict]]] = []
        for q in questions_raw:
            qid = q["question_id"]
            answers = ann_lookup.get(qid, [])
            self.samples.append((qid, q["image_id"], q["question"], answers))

        if debug:
            self.samples = self.samples[: max(1, len(self.samples) // 100)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        question_id, image_id, question, answers = self.samples[idx]

        # ---- image ----
        img_name = f"COCO_{self.split}2014_{image_id:012d}.jpg"
        img_path = self.images_dir / img_name
        image = Image.open(img_path).convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)

        # ---- question ----
        encoding = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_question_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # ---- soft scores / label ----
        soft_scores = self.vocab.build_soft_scores(answers)
        label = int(soft_scores.argmax()) if soft_scores.sum() > 0 else -1

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long),
            "soft_scores": soft_scores,
            "question_id": question_id,
        }
