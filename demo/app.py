"""Gradio demo — interactive single-image VQA inference."""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import torch
import yaml
from PIL import Image
from transformers import BertTokenizer

from src.data.augmentations import get_val_transforms
from src.data.answer_vocab import AnswerVocab
from src.models.vqa_model import VQAModel
from src.utils.checkpoint import load_checkpoint

# ---- load config ----
_CFG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"
with open(_CFG_PATH) as f:
    cfg = yaml.safe_load(f)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_TRANSFORM = get_val_transforms(cfg["data"]["image_size"])
_TOKENIZER = BertTokenizer.from_pretrained(cfg["model"]["text_encoder"])

# ---- load vocab ----
vocab_path = Path(cfg["paths"]["vocab_path"])
vocab = AnswerVocab.load(vocab_path) if vocab_path.exists() else None

# ---- load model (optional — show a warning if no checkpoint exists) ----
model: VQAModel | None = None
_ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
_ckpts = sorted(_ckpt_dir.glob("checkpoint_epoch*.pt")) if _ckpt_dir.exists() else []
if vocab and _ckpts:
    model = VQAModel(
        vision_backbone=cfg["model"]["vision_backbone"],
        text_encoder=cfg["model"]["text_encoder"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_heads=cfg["model"]["num_heads"],
        fusion_layers=cfg["model"]["fusion_layers"],
        num_answer_classes=len(vocab),
    ).to(_DEVICE)
    load_checkpoint(_ckpts[-1], model, device=_DEVICE)
    model.eval()


def answer_question(image: Image.Image, question: str) -> str:
    if model is None:
        return "(No trained model found. Train first or place a checkpoint in checkpoints/.)"
    if vocab is None:
        return "(No answer vocab found. Run scripts/build_vocab.py first.)"

    pixel_values = _TRANSFORM(image.convert("RGB")).unsqueeze(0).to(_DEVICE)
    enc = _TOKENIZER(
        question,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=cfg["data"]["max_question_length"],
    )
    input_ids = enc["input_ids"].to(_DEVICE)
    attention_mask = enc["attention_mask"].to(_DEVICE)

    with torch.no_grad():
        top_probs, top_indices = model.predict(
            pixel_values, input_ids, attention_mask, top_k=cfg["evaluation"]["top_k_answers"]
        )

    lines = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        lines.append(f"{vocab.idx_to_answer(idx.item())}  ({prob.item()*100:.1f}%)")
    return "\n".join(lines)


demo = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Image(type="pil", label="Image"),
        gr.Textbox(label="Question", placeholder="What is the color of the car?"),
    ],
    outputs=gr.Textbox(label="Top answers"),
    title="COCO-VQA Demo",
    description="Upload an image and ask a question. Top-3 predicted answers are shown.",
    examples=[
        [str(p), "What is in this image?"]
        for p in sorted(Path("demo/examples").glob("*.jpg"))[: cfg["demo"]["max_examples"]]
    ] or None,
)

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
