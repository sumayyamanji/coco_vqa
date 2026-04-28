"""Evaluate a trained VQA checkpoint on the validation split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

from src.data.dataset import LocalVQADataset
from src.data.augmentations import get_val_transforms
from src.data.answer_vocab import AnswerVocab
from src.models.vqa_model import VQAModel
from src.evaluation.vqa_eval import VQAEvaluator
from src.evaluation.metrics import top_k_accuracy
from src.utils.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate VQA model")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--output", default="results.json", help="Where to save results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = AnswerVocab.load(cfg["paths"]["vocab_path"])
    tokenizer = BertTokenizer.from_pretrained(cfg["model"]["text_encoder"])

    d = cfg["data"]
    val_ds = LocalVQADataset(
        annotations_path=d["annotations_val"],
        questions_path=d["questions_val"],
        images_dir=d["images_val"],
        split="val",
        vocab=vocab,
        image_transform=get_val_transforms(d["image_size"]),
        tokenizer=tokenizer,
        max_question_length=d["max_question_length"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=4,
    )

    model = VQAModel(
        vision_backbone=cfg["model"]["vision_backbone"],
        text_encoder=cfg["model"]["text_encoder"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_heads=cfg["model"]["num_heads"],
        fusion_layers=cfg["model"]["fusion_layers"],
        num_answer_classes=cfg["model"]["num_answer_classes"],
    ).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    evaluator = VQAEvaluator(vocab)
    topk_scores = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            logits = model(
                pixel_values=batch["image"].to(device),
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            evaluator.process_batch(batch["question_id"], logits, batch["soft_scores"])
            topk_scores.append(
                top_k_accuracy(logits.cpu(), batch["soft_scores"],
                               k=cfg["evaluation"]["top_k_answers"])
            )

    results = evaluator.summarise()
    results["top_k_accuracy"] = sum(topk_scores) / len(topk_scores)
    print(json.dumps(results, indent=2))
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
