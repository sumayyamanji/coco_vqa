"""Full pipeline inference demo — loads a checkpoint and answers a question.

Traces every stage of the pipeline so it can be used as a debugging aid or a
showcase of the end-to-end flow.

Example usage
-------------
Answer a question about a local image:
    python scripts/demo_inference.py \\
        --checkpoint checkpoints/best_model.pt \\
        --image data/raw/images/val2014/COCO_val2014_000000000042.jpg \\
        --question "What is the man holding?"

Generate a GradCAM heatmap and save it:
    python scripts/demo_inference.py \\
        --checkpoint checkpoints/best_model.pt \\
        --image path/to/image.jpg \\
        --question "What colour is the car?" \\
        --gradcam --gradcam-out outputs/heatmap.jpg

Run without a real checkpoint (random weights, sanity-check only):
    python scripts/demo_inference.py --image path/to/image.jpg \\
        --question "Is this a dog?" --no-checkpoint
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

# Make the repo root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ROOT_DIR, load_config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run single-sample inference through the VQA pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/config.yaml",
                   help="Path to config YAML")
    p.add_argument("--checkpoint", default=None, metavar="CKPT",
                   help="Path to a .pt checkpoint; omit with --no-checkpoint for random weights")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="Skip checkpoint loading (random weights — sanity-check only)")
    p.add_argument("--image", required=True, metavar="IMG",
                   help="Path to a local image file (.jpg / .png)")
    p.add_argument("--question", required=True,
                   help="Question string to answer")
    p.add_argument("--mode", default="multimodal",
                   choices=["multimodal", "text_only", "image_only"])
    p.add_argument("--top-k", type=int, default=5, metavar="K",
                   help="Number of top answers to display")
    p.add_argument("--gradcam", action="store_true",
                   help="Compute and display a GradCAM heatmap")
    p.add_argument("--gradcam-out", default=None, metavar="PATH",
                   help="Save the GradCAM overlay to this path instead of displaying it")
    p.add_argument("--gradcam-layer", default=None, metavar="LAYER",
                   help="Named module to hook for GradCAM (auto-detected when omitted)")
    p.add_argument("--device", default=None,
                   help="Torch device string, e.g. 'cuda:0' or 'cpu' (auto-detected)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_device(requested: Optional[str]) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_vocab(config: Dict[str, Any]):
    from src.data.answer_vocab import AnswerVocab
    vocab_path = ROOT_DIR / config["paths"]["vocab_path"]
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Vocab file not found at '{vocab_path}'. "
            "Run 'make vocab' (or python scripts/build_vocab.py) first."
        )
    return AnswerVocab.load(vocab_path)


def _build_model(config: Dict[str, Any], vocab_size: int, mode: str) -> torch.nn.Module:
    from src.models.vqa_model import VQAModel
    return VQAModel(config, vocab_size=vocab_size, mode=mode)


def _preprocess_image(image_path: str, image_size: int) -> torch.Tensor:
    """Load and normalise a PIL image → (1, 3, H, W) float32 tensor."""
    from src.data.augmentations import get_val_transforms
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    transform = get_val_transforms(image_size)
    return transform(img).unsqueeze(0)


def _tokenize_question(
    question: str,
    config: Dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenise a question string → (input_ids, attention_mask) tensors."""
    from transformers import AutoTokenizer
    model_name: str = config["model"]["text_encoder"]
    max_len: int = config["data"].get("max_question_length", 30)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    enc = tokenizer(
        question,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


def _auto_gradcam_layer(model: torch.nn.Module, mode: str) -> str:
    """Return a sensible default layer name for GradCAM depending on mode."""
    named = dict(model.named_modules())
    # Prefer the last transformer layer of the vision encoder
    candidates = [
        "vision_encoder.backbone.vision_model.encoder.layers.23",
        "vision_encoder.backbone.vision_model.encoder.layers.11",
        "vision_encoder",
        "fusion",
    ]
    for candidate in candidates:
        if candidate in named:
            return candidate
    # Fall back to the first available module with parameters
    for name, mod in model.named_modules():
        if name and list(mod.parameters(recurse=False)):
            return name
    raise RuntimeError("Could not auto-detect a GradCAM target layer.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = _pick_device(args.device)
    print(f"\n[demo] Device : {device}")

    # ── 1. Config ──────────────────────────────────────────────────────────
    print(f"[demo] Loading config from '{args.config}' …")
    config = load_config(args.config)
    image_size: int = config["data"].get("image_size", 224)

    # ── 2. Vocabulary ──────────────────────────────────────────────────────
    print("[demo] Loading answer vocabulary …")
    vocab = _load_vocab(config)
    print(f"       Vocab size : {len(vocab):,}")

    # ── 3. Model ───────────────────────────────────────────────────────────
    print(f"[demo] Building VQAModel (mode='{args.mode}') …")
    model = _build_model(config, vocab_size=len(vocab), mode=args.mode)
    model.to(device)

    if not args.no_checkpoint:
        ckpt_path = args.checkpoint
        if ckpt_path is None:
            raise ValueError("Provide --checkpoint <path> or use --no-checkpoint")
        print(f"[demo] Loading checkpoint '{ckpt_path}' …")
        from src.utils.checkpoint import load_checkpoint
        epoch = load_checkpoint(ckpt_path, model, device=device)
        print(f"       Restored epoch {epoch}")
    else:
        print("[demo] Skipping checkpoint — using random weights")

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"       Parameters : {total_params:,}")

    # ── 4. Pre-process inputs ──────────────────────────────────────────────
    print(f"[demo] Pre-processing image '{args.image}' …")
    image_tensor = _preprocess_image(args.image, image_size).to(device)
    print(f"       Tensor shape : {tuple(image_tensor.shape)}")

    print(f"[demo] Tokenising question: \"{args.question}\"")
    input_ids, attention_mask = _tokenize_question(args.question, config)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    print(f"       Token IDs shape : {tuple(input_ids.shape)}")

    # ── 5. Forward pass ────────────────────────────────────────────────────
    batch = {
        "image_tensor": image_tensor,
        "question_ids": input_ids,
        "attention_mask": attention_mask,
    }

    print("[demo] Running forward pass …")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(batch)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    answer_logits = out["answer_logits"]          # (1, vocab_size)
    type_logits   = out["answer_type_logits"]     # (1, 3)

    # ── 6. Decode results ──────────────────────────────────────────────────
    type_names = ["yes/no", "number", "other"]
    predicted_type = type_names[type_logits.argmax(dim=-1).item()]

    probs = F.softmax(answer_logits, dim=-1)[0]   # (vocab_size,)
    topk  = probs.topk(args.top_k)

    print("\n" + "─" * 50)
    print(f"  Question : {args.question}")
    print(f"  Predicted type : {predicted_type}")
    print(f"  Confidence     : {out['confidence'].item():.4f}")
    print(f"  Forward pass   : {elapsed_ms:.1f} ms")
    print(f"\n  Top-{args.top_k} answers:")
    for rank, (idx, prob) in enumerate(
        zip(topk.indices.tolist(), topk.values.tolist()), start=1
    ):
        answer = vocab.idx_to_answer(idx)
        print(f"    {rank}. {answer:<30s} (p={prob:.4f})")
    print("─" * 50 + "\n")

    # ── 7. GradCAM (optional) ─────────────────────────────────────────────
    if args.gradcam:
        print("[demo] Computing GradCAM heatmap …")
        from src.utils.gradcam import GradCAM
        from PIL import Image

        layer_name = args.gradcam_layer or _auto_gradcam_layer(model, args.mode)
        print(f"       Target layer : '{layer_name}'")

        gcam = GradCAM(model, layer_name)
        target_class = int(probs.argmax().item())
        heatmap = gcam.compute(image_tensor, input_ids, attention_mask,
                               target_class=target_class)
        gcam.remove_hooks()

        pil_img = Image.open(args.image).convert("RGB")
        overlay = gcam.overlay(pil_img, heatmap, alpha=0.5)

        if args.gradcam_out:
            out_path = Path(args.gradcam_out)
            if not out_path.is_absolute():
                out_path = ROOT_DIR / out_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            overlay.save(str(out_path))
            print(f"       Heatmap saved to '{out_path}'")
        else:
            overlay.show()
            print("       Heatmap displayed (close the window to exit)")


if __name__ == "__main__":
    main()
