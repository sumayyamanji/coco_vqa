"""Entry point for training the VQA model."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from src.data.dataset import LocalVQADataset
from src.data.augmentations import get_train_transforms, get_val_transforms
from src.data.answer_vocab import AnswerVocab
from src.models.vqa_model import VQAModel
from src.training.trainer import Trainer
from src.utils.wandb_logger import WandbLogger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VQA model")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--mode", default="multimodal",
                   choices=["multimodal", "text_only", "image_only"],
                   help="Which modalities the model uses (default: multimodal)")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--debug", action="store_true",
                   help="Run on 1%% of data for a quick smoke-test")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  mode: {args.mode}")

    vocab_path = Path(cfg["paths"]["vocab_path"])
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Vocab not found at {vocab_path}. Run `python scripts/build_vocab.py` first."
        )
    vocab = AnswerVocab.load(vocab_path)
    tokenizer = BertTokenizer.from_pretrained(cfg["model"]["text_encoder"])

    d = cfg["data"]
    train_ds = LocalVQADataset(
        annotations_path=d["annotations_train"],
        questions_path=d["questions_train"],
        images_dir=d["images_train"],
        split="train",
        vocab=vocab,
        image_transform=get_train_transforms(d["image_size"]),
        tokenizer=tokenizer,
        max_question_length=d["max_question_length"],
        debug=args.debug,
    )
    val_ds = LocalVQADataset(
        annotations_path=d["annotations_val"],
        questions_path=d["questions_val"],
        images_dir=d["images_val"],
        split="val",
        vocab=vocab,
        image_transform=get_val_transforms(d["image_size"]),
        tokenizer=tokenizer,
        max_question_length=d["max_question_length"],
        debug=args.debug,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=4, pin_memory=True,
    )

    model = VQAModel(
        vision_backbone=cfg["model"]["vision_backbone"],
        text_encoder=cfg["model"]["text_encoder"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_heads=cfg["model"]["num_heads"],
        fusion_layers=cfg["model"]["fusion_layers"],
        num_answer_classes=cfg["model"]["num_answer_classes"],
        dropout=cfg["model"]["dropout"],
        gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        mode=args.mode,
    )

    logger = WandbLogger(
        project=cfg["logging"]["wandb_project"],
        config={**cfg, "mode": args.mode},
        enabled=not args.no_wandb,
    )
    logger.watch(model)

    trainer = Trainer(model, train_loader, val_loader, cfg, device, logger)
    trainer.fit()
    logger.finish()


if __name__ == "__main__":
    main()
