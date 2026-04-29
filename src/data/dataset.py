"""LocalVQADataset — reads VQA v2 JSON annotations and COCO images from disk."""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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


# ---------------------------------------------------------------------------
# VQADataset — new config-driven dataset with mode support
# ---------------------------------------------------------------------------

_log = logging.getLogger(__name__)


class VQADataset(Dataset):
    """Config-driven VQA v2 dataset with multimodal / text_only / image_only modes.

    Parameters
    ----------
    split:     "train" or "val"
    config:    Parsed config.yaml dict (uses config["data"] and config["model"])
    vocab:     Either an AnswerVocab instance or an ans2idx dict from load_vocab().
               The answer_scores vector length equals len(vocab).
    transform: torchvision transform applied to the PIL image.
    mode:      "multimodal" | "text_only" | "image_only"
    """

    def __init__(
        self,
        split: str,
        config: Dict[str, Any],
        vocab,
        transform: Optional[Callable] = None,
        mode: str = "multimodal",
    ) -> None:
        from transformers import BertTokenizer

        self.split = split
        self.config = config
        self.vocab = vocab
        self.transform = transform
        self.mode = mode

        data_cfg = config["data"]
        self.image_size: int = int(data_cfg.get("image_size", 224))
        self.max_length: int = int(data_cfg.get("max_question_length", 30))

        if split == "train":
            questions_path = data_cfg["questions_train"]
            annotations_path = data_cfg["annotations_train"]
            self.images_dir = Path(data_cfg["images_train"])
            self._img_prefix = "COCO_train2014"
        else:
            questions_path = data_cfg["questions_val"]
            annotations_path = data_cfg["annotations_val"]
            self.images_dir = Path(data_cfg["images_val"])
            self._img_prefix = "COCO_val2014"

        text_model = config.get("model", {}).get("text_encoder", "bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained(text_model)

        questions_raw: List[dict] = json.loads(
            Path(questions_path).read_text(encoding="utf-8")
        )["questions"]
        annotations_raw: List[dict] = json.loads(
            Path(annotations_path).read_text(encoding="utf-8")
        )["annotations"]

        ann_lookup: Dict[int, dict] = {a["question_id"]: a for a in annotations_raw}

        self.samples: List[dict] = []
        for q in questions_raw:
            qid = q["question_id"]
            ann = ann_lookup.get(qid, {})
            self.samples.append(
                {
                    "question_id": qid,
                    "image_id": q["image_id"],
                    "question": q["question"],
                    "answers": ann.get("answers", []),
                    "answer_type": ann.get("answer_type", "other"),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _build_soft_scores(self, answers: List[dict]) -> torch.Tensor:
        if hasattr(self.vocab, "build_soft_scores"):
            return self.vocab.build_soft_scores(answers)

        # vocab is an ans2idx dict
        num_classes = len(self.vocab)
        scores = torch.zeros(num_classes, dtype=torch.float32)
        counter: Counter = Counter(a["answer"] for a in answers)
        for answer, count in counter.items():
            idx = self.vocab.get(answer, -1)
            if idx >= 0:
                scores[idx] = min(count / 3.0, 1.0)
        return scores

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        question_id: int = sample["question_id"]
        image_id: int = sample["image_id"]
        question: str = sample["question"]
        answers: List[dict] = sample["answers"]
        answer_type: str = sample["answer_type"]

        # ---- image ----
        if self.mode != "text_only":
            img_name = f"{self._img_prefix}_{image_id:012d}.jpg"
            img_path = self.images_dir / img_name
            if img_path.exists():
                pil_img = Image.open(img_path).convert("RGB")
                if self.transform is not None:
                    image_tensor: Optional[torch.Tensor] = self.transform(pil_img)
                else:
                    image_tensor = transforms.ToTensor()(pil_img)
            else:
                _log.warning("Image not found: %s", img_path)
                image_tensor = torch.zeros(3, self.image_size, self.image_size)
        else:
            image_tensor = None

        # ---- question tokens ----
        if self.mode != "image_only":
            enc = self.tokenizer(
                question,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            question_ids: Optional[torch.Tensor] = enc["input_ids"].squeeze(0)
            attention_mask: Optional[torch.Tensor] = enc["attention_mask"].squeeze(0)
        else:
            question_ids = None
            attention_mask = None

        answer_scores = self._build_soft_scores(answers)

        return {
            "image_tensor": image_tensor,
            "question_ids": question_ids,
            "attention_mask": attention_mask,
            "answer_scores": answer_scores,
            "answer_type": answer_type,
            "question_id": question_id,
            "raw_question": question,
            "raw_answers": [a["answer"] for a in answers],
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def _collate_fn(batch: List[dict]) -> dict:
    """Collate that handles None image_tensors (text_only) and None question tokens (image_only)."""
    result: dict = {}

    if batch[0]["image_tensor"] is not None:
        result["image_tensor"] = torch.stack([b["image_tensor"] for b in batch])
    else:
        result["image_tensor"] = None

    if batch[0]["question_ids"] is not None:
        result["question_ids"] = torch.stack([b["question_ids"] for b in batch])
        result["attention_mask"] = torch.stack([b["attention_mask"] for b in batch])
    else:
        result["question_ids"] = None
        result["attention_mask"] = None

    result["answer_scores"] = torch.stack([b["answer_scores"] for b in batch])
    result["answer_type"] = [b["answer_type"] for b in batch]
    result["question_id"] = [b["question_id"] for b in batch]
    result["raw_question"] = [b["raw_question"] for b in batch]
    result["raw_answers"] = [b["raw_answers"] for b in batch]

    return result


def get_dataloader(
    split: str,
    config: Dict[str, Any],
    vocab,
    mode: str = "multimodal",
    num_workers: int = 4,
) -> DataLoader:
    """Build a DataLoader for the given split using config-driven paths and transforms.

    Parameters
    ----------
    split:       "train" or "val"
    config:      Parsed config.yaml dict
    vocab:       AnswerVocab instance or ans2idx dict from load_vocab()
    mode:        "multimodal" | "text_only" | "image_only"
    num_workers: Worker processes for the DataLoader
    """
    from .augmentations import get_train_transforms, get_val_transforms

    data_cfg = config["data"]
    image_size: int = int(data_cfg.get("image_size", 224))
    use_aug: bool = bool(data_cfg.get("augmentation", True))

    if split == "train" and use_aug:
        transform = get_train_transforms(image_size)
    else:
        transform = get_val_transforms(image_size)

    dataset = VQADataset(split, config, vocab, transform, mode)

    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
    )
