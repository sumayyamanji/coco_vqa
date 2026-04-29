"""Evaluate trained VQA checkpoints and produce a full analysis report.

Loads one or more checkpoints (multimodal is required; text-only and
image-only are optional), runs inference on the validation split, computes
the official VQA v2 accuracy and supplementary metrics, generates
visualisation plots, and writes results to ``outputs/results.json``.

Example usage
-------------
Evaluate multimodal checkpoint only:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt

Evaluate all three modes (supply each checkpoint explicitly):
    python scripts/evaluate.py \\
        --checkpoint          checkpoints/best_model.pt \\
        --checkpoint-text     checkpoints/best_model_text.pt \\
        --checkpoint-image    checkpoints/best_model_image.pt

Limit to N validation samples for a quick sanity check:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt \\
        --max-samples 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.answer_vocab import load_vocab
from src.data.augmentations import get_val_transforms
from src.data.dataset import VQADataset, _collate_fn
from src.evaluation.vqa_eval import VQAEvaluator
from src.evaluation.metrics import (
    compute_confusion_matrix,
    compute_top3_accuracy,
    per_category_accuracy,
    bias_analysis,
)
from src.evaluation.visualisation import (
    plot_attention_heatmap,
    plot_comparison_table,
    plot_per_category_bar,
    plot_visual_grounding,
)
from src.models.vqa_model import VQAModel
from src.utils.checkpoint import load_checkpoint
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained VQA checkpoints")
    p.add_argument("--checkpoint", required=True,
                   help="Path to multimodal (or primary) checkpoint .pt file")
    p.add_argument("--checkpoint-text", default=None, metavar="PATH",
                   help="Optional text-only checkpoint")
    p.add_argument("--checkpoint-image", default=None, metavar="PATH",
                   help="Optional image-only checkpoint")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--output-dir", default="outputs", metavar="DIR",
                   help="Directory for results.json and plot images")
    p.add_argument("--max-samples", type=int, default=None, metavar="N",
                   help="Limit validation to N samples (faster debugging)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override config batch size for evaluation")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-plots", action="store_true",
                   help="Skip visualisation plot generation")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    idx2ans: List[str],
    device: torch.device,
    max_samples: Optional[int] = None,
) -> Tuple[List[Dict], Dict[int, Dict]]:
    """Run the model over *loader* and collect per-question predictions.

    Returns:
        predictions:   list of {"question_id", "answer", "image_id",
                                "top3_answers", "score" (placeholder)}
        preds_by_qid:  same data keyed by question_id for bias_analysis
    """
    model.eval()
    predictions: List[Dict] = []
    seen = 0

    with torch.no_grad():
        for batch in loader:
            if max_samples and seen >= max_samples:
                break
            batch_dev = _to_device(batch, device)
            out = model(batch_dev)

            logits = out["answer_logits"].cpu()          # (B, C)
            top3_idx = logits.topk(3, dim=-1).indices   # (B, 3)

            for i, qid in enumerate(batch["question_id"]):
                top1_ans = idx2ans[top3_idx[i, 0]] if top3_idx[i, 0] < len(idx2ans) else ""
                top3_ans = [
                    idx2ans[j] if j < len(idx2ans) else ""
                    for j in top3_idx[i].tolist()
                ]
                predictions.append({
                    "question_id": int(qid),
                    "answer": top1_ans,
                    "top3_answers": top3_ans,
                    "image_id": None,   # filled in by evaluator via annotations
                })
            seen += logits.size(0)

    preds_by_qid = {p["question_id"]: p for p in predictions}
    return predictions, preds_by_qid


# ---------------------------------------------------------------------------
# Single-mode evaluation
# ---------------------------------------------------------------------------

def evaluate_mode(
    checkpoint: str,
    mode: str,
    cfg: dict,
    ans2idx: dict,
    idx2ans: List[str],
    val_annotations: dict,
    device: torch.device,
    max_samples: Optional[int],
    batch_size: int,
    num_workers: int,
) -> Dict[str, Any]:
    """Load checkpoint, run inference, return evaluation dict."""
    print(f"\n{'=' * 60}")
    print(f"  Evaluating mode: {mode}  |  ckpt: {Path(checkpoint).name}")
    print(f"{'=' * 60}")

    vocab_size = len(idx2ans)
    model = VQAModel(cfg, vocab_size=vocab_size, mode=mode).to(device)
    load_checkpoint(checkpoint, model, device=device)
    model.eval()

    dataset = VQADataset(
        "val", cfg, ans2idx,
        transform=get_val_transforms(cfg["data"]["image_size"]),
        mode=mode,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=_collate_fn,
    )

    predictions, preds_by_qid = run_inference(
        model, loader, idx2ans, device, max_samples
    )

    # Official VQA accuracy (string predictions vs raw annotations)
    evaluator = VQAEvaluator()
    eval_result = evaluator.evaluate(predictions, val_annotations)

    # Score predictions for downstream metrics (bias_analysis, per-category)
    for p in predictions:
        per_q_entry = next(
            (x for x in eval_result["per_question"] if x["question_id"] == p["question_id"]),
            None,
        )
        p["score"] = per_q_entry["score"] if per_q_entry else 0.0
        p["image_id"] = per_q_entry["image_id"] if per_q_entry else None

    preds_by_qid = {p["question_id"]: p for p in predictions}

    # Top-3 accuracy
    top3_acc = compute_top3_accuracy(predictions, val_annotations)

    print(
        f"  Overall VQA acc : {100 * eval_result['accuracy']:.2f}%\n"
        f"  Top-3 acc       : {100 * top3_acc:.2f}%"
    )
    pt = eval_result.get("per_type", {})
    for t, v in pt.items():
        print(f"  {t:<12}: {100 * v:.2f}%")

    return {
        "mode": mode,
        "accuracy": eval_result["accuracy"],
        "per_type": pt,
        "top3_accuracy": top3_acc,
        "n_questions": eval_result["n_questions"],
        "predictions": predictions,
        "preds_by_qid": preds_by_qid,
    }


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _save_fig(fig, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt_close(fig)
    print(f"  Saved: {path}")


def plt_close(fig) -> None:
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass


def generate_visualisations(
    results_by_mode: Dict[str, Dict],
    cfg: dict,
    ans2idx: dict,
    idx2ans: List[str],
    val_annotations: dict,
    plot_dir: Path,
    device: torch.device,
) -> None:
    """Generate and save all visualisation plots."""
    print("\nGenerating visualisation plots …")
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Comparison table (all modes)
    mm = results_by_mode.get("multimodal", {})
    to = results_by_mode.get("text_only", {})
    io = results_by_mode.get("image_only", {})
    fig = plot_comparison_table(
        text_only_results=to if to else None,
        multimodal_results=mm if mm else None,
        image_only_results=io if io else None,
    )
    _save_fig(fig, plot_dir / "comparison_table.png")

    # 2. Per-category accuracy bar (multimodal, if available)
    mm_preds = mm.get("predictions", [])
    if mm_preds:
        cat_acc = per_category_accuracy(mm_preds, val_annotations)
        if cat_acc:
            fig = plot_per_category_bar(cat_acc)
            _save_fig(fig, plot_dir / "per_category_accuracy.png")

    # 3. Attention heatmaps + visual grounding — sample a few images
    _generate_sample_plots(results_by_mode, cfg, ans2idx, idx2ans, plot_dir, device)


def _generate_sample_plots(
    results_by_mode: Dict[str, Dict],
    cfg: dict,
    ans2idx: dict,
    idx2ans: List[str],
    plot_dir: Path,
    device: torch.device,
    n_samples: int = 4,
) -> None:
    """Run inference on a handful of val images and produce attention plots."""
    from src.data.dataset import VQADataset, _collate_fn
    from torch.utils.data import DataLoader

    mm_result = results_by_mode.get("multimodal")
    if not mm_result:
        return

    ckpt_path = results_by_mode["multimodal"].get("_checkpoint")
    if not ckpt_path:
        return

    vocab_size = len(idx2ans)
    model = VQAModel(cfg, vocab_size=vocab_size, mode="multimodal").to(device)
    try:
        load_checkpoint(ckpt_path, model, device=device)
    except Exception as exc:
        print(f"  [Warning] Could not reload model for sample plots: {exc}")
        return
    model.eval()

    dataset = VQADataset(
        "val", cfg, ans2idx,
        transform=get_val_transforms(cfg["data"]["image_size"]),
        mode="multimodal",
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=_collate_fn
    )

    sample_idx = 0
    with torch.no_grad():
        for batch in loader:
            if sample_idx >= n_samples:
                break
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            out = model(batch_dev)
            attn = out.get("cross_attention_weights")
            if attn is None:
                sample_idx += 1
                continue

            question = batch["raw_question"][0]
            pred_idx = out["answer_logits"].argmax(dim=-1).item()
            answer = idx2ans[pred_idx] if pred_idx < len(idx2ans) else "?"

            img_tensor = batch.get("image_tensor")
            if img_tensor is None:
                sample_idx += 1
                continue
            img_t = img_tensor[0]   # (C, H, W)

            # Attention heatmap
            fig = plot_attention_heatmap(img_t, attn[0], question, answer)
            _save_fig(fig, plot_dir / f"attn_sample_{sample_idx:02d}.png")

            # Visual grounding
            fig = plot_visual_grounding(img_t, attn[0], question, answer)
            _save_fig(fig, plot_dir / f"grounding_sample_{sample_idx:02d}.png")

            sample_idx += 1


# ---------------------------------------------------------------------------
# Summary table (stdout)
# ---------------------------------------------------------------------------

def print_summary_table(results_by_mode: Dict[str, Dict]) -> None:
    type_cols = ["yes/no", "number", "other"]
    header = f"{'Model':<16} {'Overall':>9} {'Yes/No':>8} {'Number':>8} {'Other':>8}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    order = ["text_only", "image_only", "multimodal"]
    labels = {"text_only": "Text-Only", "image_only": "Image-Only", "multimodal": "Multimodal"}
    for key in order:
        r = results_by_mode.get(key)
        if r is None:
            continue
        pt = r.get("per_type", {})
        row = (
            f"  {labels[key]:<14}"
            f" {100 * r['accuracy']:>8.1f}%"
            f" {100 * pt.get('yes/no', 0):>7.1f}%"
            f" {100 * pt.get('number', 0):>7.1f}%"
            f" {100 * pt.get('other',  0):>7.1f}%"
        )
        print(row)
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "eval_plots"

    # ---- Vocab ----
    vocab_path = Path(cfg["data"]["vocab_path"])
    ans2idx, idx2ans = load_vocab(vocab_path)
    print(f"Vocab: {len(idx2ans):,} answers")

    # ---- Val annotations ----
    import json as _json
    ann_path = Path(cfg["data"]["annotations_val"])
    print(f"Loading val annotations from {ann_path} …")
    with open(ann_path, encoding="utf-8") as fh:
        val_annotations = _json.load(fh)

    batch_size = args.batch_size or cfg["training"]["batch_size"]

    # ---- Determine which modes to evaluate ----
    mode_checkpoints = {"multimodal": args.checkpoint}
    if args.checkpoint_text:
        mode_checkpoints["text_only"] = args.checkpoint_text
    if args.checkpoint_image:
        mode_checkpoints["image_only"] = args.checkpoint_image

    # Also auto-detect sibling checkpoints from the same directory
    ckpt_dir = Path(args.checkpoint).parent
    for mode, fname in [("text_only", "best_model_text.pt"),
                        ("image_only", "best_model_image.pt")]:
        candidate = ckpt_dir / fname
        if candidate.exists() and mode not in mode_checkpoints:
            mode_checkpoints[mode] = str(candidate)
            print(f"  Auto-detected {mode} checkpoint: {candidate}")

    # ---- Run evaluation for each mode ----
    results_by_mode: Dict[str, Dict] = {}
    for mode, ckpt in mode_checkpoints.items():
        try:
            r = evaluate_mode(
                checkpoint=ckpt,
                mode=mode,
                cfg=cfg,
                ans2idx=ans2idx,
                idx2ans=idx2ans,
                val_annotations=val_annotations,
                device=device,
                max_samples=args.max_samples,
                batch_size=batch_size,
                num_workers=args.num_workers,
            )
            r["_checkpoint"] = ckpt
            results_by_mode[mode] = r
        except Exception as exc:
            print(f"[ERROR] Evaluation failed for mode '{mode}': {exc}")

    if not results_by_mode:
        print("No results — aborting.")
        return

    # ---- Bias analysis (text-only vs multimodal) ----
    if "text_only" in results_by_mode and "multimodal" in results_by_mode:
        bias = bias_analysis(
            text_only_preds=results_by_mode["text_only"]["preds_by_qid"],
            multimodal_preds=results_by_mode["multimodal"]["preds_by_qid"],
            annotations=val_annotations,
        )
        print("\nBias analysis:")
        print(f"  Language-bias (text-only correct, multimodal wrong): "
              f"{bias['language_bias_count']:,}")
        print(f"  Multimodal gain (vision helped)                    : "
              f"{bias['multimodal_gain_count']:,}")
        print(f"  Both correct                                       : "
              f"{bias['both_correct_count']:,}")
        print(f"  Both fail                                          : "
              f"{bias['both_fail_count']:,}")
    else:
        bias = {}

    # ---- Summary table ----
    print_summary_table(results_by_mode)

    # ---- Visualisations ----
    if not args.no_plots:
        try:
            generate_visualisations(
                results_by_mode, cfg, ans2idx, idx2ans,
                val_annotations, plot_dir, device,
            )
        except Exception as exc:
            print(f"[Warning] Plot generation failed: {exc}")

    # ---- Save results JSON ----
    serialisable: Dict[str, Any] = {}
    for mode, r in results_by_mode.items():
        serialisable[mode] = {
            "accuracy": r["accuracy"],
            "per_type": r["per_type"],
            "top3_accuracy": r["top3_accuracy"],
            "n_questions": r["n_questions"],
        }
    if bias:
        serialisable["bias_analysis"] = {
            k: v for k, v in bias.items() if not k.endswith("_examples")
        }

    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2)
    print(f"\nResults saved to {results_path}")
    if not args.no_plots:
        print(f"Plots saved to   {plot_dir}/")


if __name__ == "__main__":
    main()
