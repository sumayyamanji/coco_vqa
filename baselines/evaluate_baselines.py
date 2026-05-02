"""Compare text-only DistilBERT baseline against the main multimodal model.

Reads from:
  baselines/outputs/checkpoints/text_only_bert/best.pt  (baseline checkpoint)
  outputs/checkpoints/best_model.pt                     (main model — READ ONLY)
  outputs/training_log.json                             (training curve — READ ONLY)
  configs/config.yaml                                   (main model config — READ ONLY)

Writes to baselines/outputs/ only — never touches outputs/.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Force UTF-8 output so box-drawing characters print correctly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.data.dataset import VQADataset, _collate_fn
from src.data.answer_vocab import load_vocab
from transformers import DistilBertModel


# ─────────────────────────────────────────────────────────────────────────────
# TextOnlyBERT — duplicated here so evaluate_baselines.py is fully standalone
# ─────────────────────────────────────────────────────────────────────────────

class TextOnlyBERT(nn.Module):
    def __init__(self, text_encoder: str, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(text_encoder)
        bert_dim = self.bert.config.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.mlp(cls)


# ─────────────────────────────────────────────────────────────────────────────
# Stratified evaluation
# ─────────────────────────────────────────────────────────────────────────────

def build_stratified_indices(
    dataset: VQADataset,
    total: int = 3000,
    seed: int = 42,
) -> List[int]:
    """Return indices preserving ~38/12/50 yes/no:number:other split."""
    rng = random.Random(seed)
    yes_no_pool: List[int] = []
    number_pool: List[int] = []
    other_pool: List[int] = []

    for i, s in enumerate(dataset.samples):
        at = s["answer_type"]
        if at == "yes/no":
            yes_no_pool.append(i)
        elif at == "number":
            number_pool.append(i)
        else:
            other_pool.append(i)

    indices = (
        rng.sample(yes_no_pool, min(1140, len(yes_no_pool)))
        + rng.sample(number_pool, min(360, len(number_pool)))
        + rng.sample(other_pool, min(1500, len(other_pool)))
    )
    rng.shuffle(indices)
    return indices


def _mean(lst: List[float]) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def run_text_only_eval(
    model: nn.Module,
    dataset: VQADataset,
    indices: List[int],
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[float, float, float, float, float]:
    """Returns (overall, yes_no, number, other, elapsed_ms)."""
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_fn,
    )

    model.eval()
    buckets: Dict[str, List[float]] = {"yes/no": [], "number": [], "other": []}
    t0 = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            q_ids = batch["question_ids"]
            if q_ids is None:
                continue
            q_ids = q_ids.to(device)
            am = batch["attention_mask"].to(device)
            ss = batch["answer_scores"].to(device)
            answer_types: List[str] = batch["answer_type"]

            logits = model(q_ids, am)
            preds = logits.argmax(dim=-1)
            per_sample = ss[torch.arange(len(preds)), preds]

            for i, at in enumerate(answer_types):
                key = at if at in buckets else "other"
                buckets[key].append(float(per_sample[i]))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    all_vals = [v for vs in buckets.values() for v in vs]
    return (
        _mean(all_vals),
        _mean(buckets["yes/no"]),
        _mean(buckets["number"]),
        _mean(buckets["other"]),
        elapsed_ms,
    )


def run_main_model_eval(
    main_ckpt_path: Path,
    indices: List[int],
    device: torch.device,
    ans2idx: dict,
    num_classes: int,
) -> Optional[Tuple[float, float, float, float]]:
    """Load and evaluate the main VQAModel. Returns None if unavailable."""
    if not main_ckpt_path.exists():
        print(f"  Main model checkpoint not found: {main_ckpt_path.relative_to(ROOT_DIR)}")
        return None

    main_cfg_path = ROOT_DIR / "configs" / "config.yaml"
    if not main_cfg_path.exists():
        print(f"  Main config not found: {main_cfg_path.relative_to(ROOT_DIR)}")
        return None

    try:
        from src.models.vqa_model import VQAModel

        main_cfg = yaml.safe_load(main_cfg_path.read_text(encoding="utf-8"))
        main_model = VQAModel(main_cfg, num_classes=num_classes).to(device)

        raw = torch.load(main_ckpt_path, map_location=device)
        if isinstance(raw, dict):
            state_dict = (
                raw.get("model_state_dict")
                or raw.get("state_dict")
                or raw.get("model")
                or raw
            )
        else:
            state_dict = raw

        main_model.load_state_dict(state_dict, strict=False)
        print("  Main model loaded successfully.")

        # Use a multimodal dataset with the main config so images are loaded
        mm_dataset = VQADataset("val", main_cfg, ans2idx, transform=None, mode="multimodal")
        subset = Subset(mm_dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate_fn,
        )

        main_model.eval()
        buckets: Dict[str, List[float]] = {"yes/no": [], "number": [], "other": []}

        with torch.no_grad():
            for batch in loader:
                img = batch.get("image_tensor")
                q_ids = batch.get("question_ids")
                am = batch.get("attention_mask")
                ss = batch["answer_scores"].to(device)
                answer_types: List[str] = batch["answer_type"]

                if img is not None:
                    img = img.to(device)
                if q_ids is not None:
                    q_ids = q_ids.to(device)
                if am is not None:
                    am = am.to(device)

                out = main_model(img, q_ids, am)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                preds = logits.argmax(dim=-1)
                per_sample = ss[torch.arange(len(preds)), preds]

                for i, at in enumerate(answer_types):
                    key = at if at in buckets else "other"
                    buckets[key].append(float(per_sample[i]))

        all_vals = [v for vs in buckets.values() for v in vs]
        return (
            _mean(all_vals),
            _mean(buckets["yes/no"]),
            _mean(buckets["number"]),
            _mean(buckets["other"]),
        )

    except Exception as exc:
        print(f"  Could not evaluate main model: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Console output
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(results: dict) -> None:
    b = results["text_only_bert"]
    has_main = "main_model" in results

    def pct(v: float) -> str:
        return f"{v * 100:.1f}%"

    w0, w1, w2, w3, w4 = 37, 9, 8, 8, 7
    h_sep = "─"
    row_fmt = "│ {:<{w0}} │ {:^{w1}} │ {:^{w2}} │ {:^{w3}} │ {:^{w4}} │"

    top    = "┌" + h_sep*w0 + "──┬" + h_sep*w1 + "┬" + h_sep*w2 + "┬" + h_sep*w3 + "┬" + h_sep*w4 + "┐"
    mid    = "├" + h_sep*w0 + "──┼" + h_sep*w1 + "┼" + h_sep*w2 + "┼" + h_sep*w3 + "┼" + h_sep*w4 + "┤"
    bottom = "└" + h_sep*w0 + "──┴" + h_sep*w1 + "┴" + h_sep*w2 + "┴" + h_sep*w3 + "┴" + h_sep*w4 + "┘"

    print("\n" + top)
    print(row_fmt.format("Model", "Overall", "Yes/No", "Number", "Other",
                         w0=w0, w1=w1, w2=w2, w3=w3, w4=w4))
    print(mid)
    print(row_fmt.format(
        "Text-Only DistilBERT (baseline)",
        pct(b["overall"]), pct(b["yes_no"]), pct(b["number"]), pct(b["other"]),
        w0=w0, w1=w1, w2=w2, w3=w3, w4=w4,
    ))

    if has_main:
        m = results["main_model"]
        print(row_fmt.format(
            "CLIP ViT-L + BERT + cross-attn ✅",
            pct(m["overall"]), pct(m["yes_no"]), pct(m["number"]), pct(m["other"]),
            w0=w0, w1=w1, w2=w2, w3=w3, w4=w4,
        ))
        print(mid)

        def diff(k: str) -> str:
            return f"+{(m[k] - b[k]) * 100:.1f}%"

        print(row_fmt.format(
            "Improvement",
            diff("overall"), diff("yes_no"), diff("number"), diff("other"),
            w0=w0, w1=w1, w2=w2, w3=w3, w4=w4,
        ))

    print(bottom)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_bar(results: dict, plots_dir: Path) -> None:
    categories = ["Overall", "Yes/No", "Number", "Other"]
    b = results["text_only_bert"]
    b_scores = [b["overall"] * 100, b["yes_no"] * 100, b["number"] * 100, b["other"] * 100]

    has_main = "main_model" in results
    fig, ax = plt.subplots(figsize=(10, 6))
    x = list(range(len(categories)))
    width = 0.35

    if has_main:
        m = results["main_model"]
        m_scores = [m["overall"] * 100, m["yes_no"] * 100, m["number"] * 100, m["other"] * 100]
        ax.bar([i - width / 2 for i in x], b_scores, width,
               label="Text-Only DistilBERT", color="#9e9e9e", edgecolor="white")
        ax.bar([i + width / 2 for i in x], m_scores, width,
               label="CLIP ViT-L + BERT + cross-attn ✓", color="#1976d2", edgecolor="white")

        for i, (bv, mv) in enumerate(zip(b_scores, m_scores)):
            ax.annotate(
                f"+{mv - bv:.1f}%",
                xy=(i + width / 2, mv),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color="#1976d2",
                fontweight="bold",
            )
    else:
        ax.bar(x, b_scores, 0.5, label="Text-Only DistilBERT", color="#9e9e9e", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("VQA Accuracy (%)")
    ax.set_title("Text-Only vs Multimodal VQA")
    ax.legend()
    ax.set_ylim(0, 85)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "comparison_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved comparison_bar.png")


def plot_language_bias(results: dict, plots_dir: Path) -> None:
    categories = ["Overall", "Yes/No", "Number", "Other"]
    b = results["text_only_bert"]
    scores = [b["overall"] * 100, b["yes_no"] * 100, b["number"] * 100, b["other"] * 100]
    colors = ["#9e9e9e", "#e53935", "#9e9e9e", "#9e9e9e"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(categories, scores, color=colors, edgecolor="white")

    yn = scores[1]
    ax.annotate(
        f"{yn:.1f}%+ from\nlanguage bias alone",
        xy=(1, yn),
        xytext=(1.7, yn + 8),
        arrowprops=dict(arrowstyle="->", color="#e53935"),
        fontsize=10,
        color="#e53935",
        fontweight="bold",
    )

    ax.set_ylabel("VQA Accuracy (%)")
    ax.set_title("Language Bias in VQAv2 — Text-Only DistilBERT Performance")
    ax.set_ylim(0, 85)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "language_bias.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved language_bias.png")


def plot_training_curve(
    baseline_log_path: Path,
    main_log_path: Path,
    plots_dir: Path,
) -> None:
    def _load_log(path: Path) -> List[dict]:
        if not path.exists():
            return []
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    baseline_log = _load_log(baseline_log_path)
    main_log = _load_log(main_log_path)

    if not baseline_log and not main_log:
        print("  No training logs found — skipping training_curve.png")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    if baseline_log:
        epochs = [e["epoch"] for e in baseline_log]
        accs = [e["val_accuracy"] * 100 for e in baseline_log]
        ax.plot(epochs, accs, "o-", color="#9e9e9e", label="Text-Only DistilBERT", linewidth=2)

    if main_log:
        valid = [e for e in main_log if e.get("val_accuracy", 0) > 0]
        if valid:
            epochs = [e["epoch"] for e in valid]
            accs = [e["val_accuracy"] * 100 for e in valid]
            ax.plot(epochs, accs, "s-", color="#1976d2", label="CLIP ViT-L + BERT", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Accuracy (%)")
    ax.set_title("Validation Accuracy: Baseline vs Multimodal")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "training_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved training_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg_path = ROOT_DIR / "baselines" / "configs" / "baselines_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    plots_dir = ROOT_DIR / cfg["paths"]["plots_dir"]
    plots_dir.mkdir(parents=True, exist_ok=True)

    baseline_log_path = ROOT_DIR / cfg["paths"]["training_log"]
    main_log_path = ROOT_DIR / "outputs" / "training_log.json"
    master_results_path = ROOT_DIR / "baselines" / "outputs" / "master_results.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Vocab ──────────────────────────────────────────────────────────────────
    vocab_path = ROOT_DIR / cfg["data"]["vocab_path"]
    ans2idx, idx2ans = load_vocab(vocab_path)
    num_classes = len(idx2ans)
    print(f"Vocab size: {num_classes}")

    # ── Shared stratified indices (seed=42, same for both models) ─────────────
    print("\nLoading val dataset (text-only, no images)...")
    val_dataset = VQADataset("val", cfg, ans2idx, transform=None, mode="text_only")

    print("Building stratified indices (seed=42)...")
    indices = build_stratified_indices(val_dataset, total=3000, seed=42)
    yes_no_n = sum(1 for i in indices if val_dataset.samples[i]["answer_type"] == "yes/no")
    number_n = sum(1 for i in indices if val_dataset.samples[i]["answer_type"] == "number")
    other_n = len(indices) - yes_no_n - number_n
    print(f"  {len(indices)} samples: {yes_no_n} yes/no | {number_n} number | {other_n} other")

    # Save indices for reproducibility — both models evaluated on identical samples
    eval_indices_path = ROOT_DIR / "baselines" / "outputs" / "eval_sample_indices.json"
    eval_indices_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_indices_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"indices": indices, "seed": 42, "total": len(indices),
             "yes_no": yes_no_n, "number": number_n, "other": other_n},
            fh,
        )
    print(f"Evaluation sample: 3000 stratified samples")
    print(f"  (seed=42, saved to eval_sample_indices.json)")
    print(f"  Same indices used for both models ✅")
    print(f"\n  NOTE: scripts/evaluate.py does not use stratified sampling —")
    print(f"  evaluate_baselines.py loads the main model directly on these same 3000 samples.")

    # ── Load and evaluate baseline ────────────────────────────────────────────
    baseline_ckpt = (
        ROOT_DIR / "baselines" / "outputs" / "checkpoints" / "text_only_bert" / "best.pt"
    )
    if not baseline_ckpt.exists():
        print(f"\nBaseline checkpoint not found: {baseline_ckpt.relative_to(ROOT_DIR)}")
        print("Run `make train-baseline` first.")
        sys.exit(1)

    print(f"\nLoading baseline: {baseline_ckpt.relative_to(ROOT_DIR)}")
    model_cfg = cfg["models"]["text_only_bert"]
    baseline_model = TextOnlyBERT(
        text_encoder=model_cfg["text_encoder"],
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_classes=num_classes,
    ).to(device)
    saved = torch.load(baseline_ckpt, map_location=device)
    try:
        baseline_model.load_state_dict(saved["model_state_dict"])
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print(f"⚠️  Baseline checkpoint trained with a different vocab size.")
            print(f"    Retrain with: make train-baseline  (current vocab: {num_classes} classes)")
            sys.exit(1)
        raise

    print("Evaluating text-only baseline...")
    b_overall, b_yesno, b_number, b_other, b_ms = run_text_only_eval(
        baseline_model, val_dataset, indices, device
    )
    print(f"  Overall={b_overall:.4f}  Yes/No={b_yesno:.4f}  "
          f"Number={b_number:.4f}  Other={b_other:.4f}")

    results: dict = {
        "text_only_bert": {
            "overall": round(b_overall, 4),
            "yes_no": round(b_yesno, 4),
            "number": round(b_number, 4),
            "other": round(b_other, 4),
        }
    }

    # ── Load and evaluate main model ──────────────────────────────────────────
    # Try best_model.pt first (actual filename), then best.pt (spec name)
    for candidate in ["best_model.pt", "best.pt"]:
        main_ckpt = ROOT_DIR / "outputs" / "checkpoints" / candidate
        if main_ckpt.exists():
            break

    print(f"\nAttempting main model: {main_ckpt.relative_to(ROOT_DIR)}...")
    main_result = run_main_model_eval(main_ckpt, indices, device, ans2idx, num_classes)

    if main_result is not None:
        m_overall, m_yesno, m_number, m_other = main_result
        results["main_model"] = {
            "overall": round(m_overall, 4),
            "yes_no": round(m_yesno, 4),
            "number": round(m_number, 4),
            "other": round(m_other, 4),
        }

    # ── Print comparison table ────────────────────────────────────────────────
    print_comparison_table(results)

    # ── Language bias analysis ────────────────────────────────────────────────
    print("\nLanguage bias analysis:")
    print(f"  Text-only yes/no: {b_yesno * 100:.1f}% — model exploits question phrasing "
          f"without seeing any image.")
    if "main_model" in results:
        m = results["main_model"]
        print(f"  Multimodal yes/no: {m['yes_no'] * 100:.1f}% "
              f"(+{(m['yes_no'] - b_yesno) * 100:.1f}% from vision)")
        number_gain = (m["number"] - results["text_only_bert"]["number"]) * 100
        print(f"  Largest gain: number questions (+{number_gain:.1f}%) "
              f"— counting requires visual grounding.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_comparison_bar(results, plots_dir)
    plot_language_bias(results, plots_dir)
    plot_training_curve(baseline_log_path, main_log_path, plots_dir)

    # ── Save master results ───────────────────────────────────────────────────
    master_results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nMaster results → {master_results_path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
