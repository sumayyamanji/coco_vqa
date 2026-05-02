"""Text-only DistilBERT baseline for VQAv2.

Completely standalone — writes only to baselines/outputs/checkpoints/text_only_bert/
and baselines/outputs/. Never touches outputs/, src/, scripts/, or configs/.

Imports from src/ are read-only (no modifications to those files).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from src.training.losses import VQASoftLoss
from src.training.scheduler import build_scheduler
from transformers import DistilBertModel


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class TextOnlyBERT(nn.Module):
    """DistilBERT CLS token → 2-layer MLP → answer logits.

    DistilBERT has NO token_type_ids — they are never passed to self.bert.
    """

    def __init__(self, text_encoder: str, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(text_encoder)
        bert_dim = self.bert.config.hidden_size  # 768
        self.mlp = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # DistilBERT has no token_type_ids — do NOT pass them
        )
        cls = outputs.last_hidden_state[:, 0, :]
        return self.mlp(cls)


# ─────────────────────────────────────────────────────────────────────────────
# Stratified evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_stratified_indices(
    dataset: VQADataset,
    total: int = 3000,
    seed: int = 42,
) -> List[int]:
    """Return indices preserving ~38/12/50 yes/no:number:other VQAv2 distribution."""
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

    yes_no_samples = rng.sample(yes_no_pool, min(1140, len(yes_no_pool)))
    number_samples = rng.sample(number_pool, min(360, len(number_pool)))
    other_samples = rng.sample(other_pool, min(1500, len(other_pool)))

    indices = yes_no_samples + number_samples + other_samples
    rng.shuffle(indices)
    return indices


def evaluate_stratified(
    model: nn.Module,
    dataset: VQADataset,
    device: torch.device,
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[float, float, float, float, float]:
    """Stratified 3000-sample evaluation.

    Returns (overall, yes_no, number, other, total_inference_ms).
    """
    indices = build_stratified_indices(dataset, total=3000, seed=seed)
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
            input_ids = batch["question_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            soft_scores = batch["answer_scores"].to(device)
            answer_types: List[str] = batch["answer_type"]

            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1)
            per_sample = soft_scores[torch.arange(len(preds)), preds]

            for i, at in enumerate(answer_types):
                key = at if at in buckets else "other"
                buckets[key].append(float(per_sample[i]))

    elapsed_ms = (time.perf_counter() - t0) * 1000

    def _mean(lst: List[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    all_vals = [v for vs in buckets.values() for v in vs]
    return (
        _mean(all_vals),
        _mean(buckets["yes/no"]),
        _mean(buckets["number"]),
        _mean(buckets["other"]),
        elapsed_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight check
# ─────────────────────────────────────────────────────────────────────────────

def _preflight_check(
    device: torch.device,
    num_classes: int,
    train_frac: float,
    n_train: int,
    val_dataset: VQADataset,
    trainable_params: float,
) -> bool:
    """Print pre-flight verification report. Returns True if all checks pass."""
    SEP = "━" * 40
    print(f"\n{SEP}")
    print(" PRE-FLIGHT CHECK")
    all_ok = True

    device_mark = "✅" if device.type == "cuda" else "⚠️  (CPU — training will be slow)"
    print(f" Device:         {device} {device_mark}")

    vocab_ok = num_classes == 3129
    vocab_mark = "✅  (matches standard 3129)" if vocab_ok else f"⚠️  ({num_classes} — ensure baseline matches main model vocab)"
    print(f" Vocab size:     {num_classes} {vocab_mark}")

    train_ok = round(train_frac * 100) == 10
    if not train_ok:
        all_ok = False
    train_mark = "✅ (10% — matches main model)" if train_ok else f"❌  (expected 10%, got {train_frac * 100:.0f}%)"
    print(f" Train samples:  {n_train:,} {train_mark}")

    indices = build_stratified_indices(val_dataset, total=3000, seed=42)
    yes_no_n = sum(1 for i in indices if val_dataset.samples[i]["answer_type"] == "yes/no")
    number_n = sum(1 for i in indices if val_dataset.samples[i]["answer_type"] == "number")
    other_n = len(indices) - yes_no_n - number_n
    val_ok = len(indices) == 3000
    if not val_ok:
        all_ok = False
    val_mark = "✅  (stratified, seed=42)" if val_ok else f"❌  (expected 3000, got {len(indices)})"
    print(f" Val samples:    {len(indices)} {val_mark}")
    print(f"   yes/no:       {yes_no_n}")
    print(f"   number:        {number_n}")
    print(f"   other:        {other_n}")

    print(f" Model:          DistilBERT text-only")
    print(f" Trainable params: {trainable_params:.1f}M")

    print(f" {SEP}")
    if all_ok:
        print(" All checks passed ✅ — safe to run full training")
    else:
        print(" ❌ Pre-flight FAILED — fix the issues above before training")
    print(f"{SEP}\n")
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Training log helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_checkpoints(ckpt_dir: Path) -> List[Path]:
    return sorted(ckpt_dir.glob("checkpoint_epoch*.pt"))


def _append_training_log(log_path: Path, entry: dict) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    # ── Config ────────────────────────────────────────────────────────────────
    config_path = ROOT_DIR / args.config
    cfg: dict = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    train_cfg: dict = cfg["training"]
    model_cfg: dict = cfg["models"]["text_only_bert"]
    paths_cfg: dict = cfg["paths"]

    ckpt_dir = ROOT_DIR / paths_cfg["checkpoint_dir"] / "text_only_bert"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_path = ROOT_DIR / paths_cfg["training_log"]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    results_path = ROOT_DIR / paths_cfg["results_path"]
    results_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Debug overrides ────────────────────────────────────────────────────────
    if args.debug:
        train_cfg = dict(train_cfg)
        train_cfg["epochs"] = 1
        train_cfg["batch_size"] = 16
        n_debug_train = 200
        n_debug_val = 100
        print("DEBUG MODE: 200 train samples | 1 epoch | 100 val samples")
    else:
        n_debug_train = None
        n_debug_val = None

    # ── Vocab ─────────────────────────────────────────────────────────────────
    vocab_path = ROOT_DIR / cfg["data"]["vocab_path"]
    ans2idx, idx2ans = load_vocab(vocab_path)
    num_classes = len(idx2ans)
    print(f"Vocab size: {num_classes}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    # mode="text_only" → image_tensor=None, images never loaded or decoded
    train_dataset = VQADataset("train", cfg, ans2idx, transform=None, mode="text_only")
    val_dataset = VQADataset("val", cfg, ans2idx, transform=None, mode="text_only")

    rng = random.Random(42)

    if args.debug:
        train_indices = list(range(min(n_debug_train, len(train_dataset))))
    else:
        frac = float(train_cfg.get("train_subset", 0.03))
        n = max(1, int(len(train_dataset) * frac))
        train_indices = rng.sample(range(len(train_dataset)), n)

    train_subset = Subset(train_dataset, train_indices)
    train_loader = DataLoader(
        train_subset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_fn,
    )

    print(f"Train samples: {len(train_subset)}")
    print(f"Val dataset:   {len(val_dataset)} (stratified eval uses 3000)")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TextOnlyBERT(
        text_encoder=model_cfg["text_encoder"],
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_classes=num_classes,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ) / 1e6
    print(f"Parameters: {total_params:.2f}M total  |  {trainable_params:.2f}M trainable")

    # ── Pre-flight check ───────────────────────────────────────────────────────
    real_frac = float(train_cfg.get("train_subset", 0.10))
    n_real_train = max(1, int(len(train_dataset) * real_frac))
    checks_ok = _preflight_check(
        device=device,
        num_classes=num_classes,
        train_frac=real_frac,
        n_train=n_real_train,
        val_dataset=val_dataset,
        trainable_params=trainable_params,
    )
    if not checks_ok:
        sys.exit(1)
    if args.debug:
        print("(--debug: pre-flight only — no training started)")
        return

    # ── Resume logic ──────────────────────────────────────────────────────────
    start_epoch = 1
    existing_ckpts = _find_checkpoints(ckpt_dir)

    if existing_ckpts:
        print(f"\nFound {len(existing_ckpts)} existing checkpoint(s):")
        for ck in existing_ckpts:
            marker = " ← latest" if ck == existing_ckpts[-1] else ""
            print(f"  {ck.name}{marker}")

        if args.resume:
            do_resume = True
        else:
            ans = input("Resume from latest? [y/n]: ").strip().lower()
            do_resume = ans == "y"

        if do_resume:
            latest = existing_ckpts[-1]
            saved = torch.load(latest, map_location=device)
            try:
                model.load_state_dict(saved["model_state_dict"])
                start_epoch = saved.get("epoch", 1) + 1
                print(f"Resumed from {latest.name} → starting epoch {start_epoch}")
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print(f"⚠️  Checkpoint vocab mismatch — starting from scratch (current vocab: {num_classes} classes)")
                    start_epoch = 1
                else:
                    raise

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    total_epochs = int(train_cfg["epochs"])
    steps_per_epoch = max(1, len(train_loader))
    remaining_epochs = max(0, total_epochs - start_epoch + 1)
    total_steps = max(1, steps_per_epoch * remaining_epochs)

    scheduler = build_scheduler(
        optimizer,
        warmup_steps=int(train_cfg.get("warmup_steps", 100)),
        total_steps=total_steps,
        schedule="cosine",
    )

    loss_fn = VQASoftLoss(cfg)

    use_fp16 = bool(train_cfg.get("fp16", True)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ── Training loop ──────────────────────────────────────────────────────────
    best_acc = 0.0
    epoch_times: List[float] = []
    last_elapsed_ms = 0.0
    last_val_acc = (0.0, 0.0, 0.0, 0.0)
    SEP = "━" * 38

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        epoch_loss = 0.0
        t_epoch = time.perf_counter()

        for batch in train_loader:
            input_ids = batch["question_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            soft_scores = batch["answer_scores"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_fp16):
                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits, soft_scores)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()

        epoch_duration = time.perf_counter() - t_epoch
        epoch_times.append(epoch_duration)
        avg_loss = epoch_loss / steps_per_epoch

        # ── Epoch evaluation ───────────────────────────────────────────────────
        if args.debug:
            dbg_idx = list(range(min(n_debug_val, len(val_dataset))))
            dbg_loader = DataLoader(
                Subset(val_dataset, dbg_idx),
                batch_size=16,
                shuffle=False,
                num_workers=0,
                collate_fn=_collate_fn,
            )
            model.eval()
            buckets: Dict[str, List[float]] = {"yes/no": [], "number": [], "other": []}
            with torch.no_grad():
                for batch in dbg_loader:
                    ids = batch["question_ids"].to(device)
                    am = batch["attention_mask"].to(device)
                    ss = batch["answer_scores"].to(device)
                    logits = model(ids, am)
                    preds = logits.argmax(dim=-1)
                    per = ss[torch.arange(len(preds)), preds]
                    for i, at in enumerate(batch["answer_type"]):
                        k = at if at in buckets else "other"
                        buckets[k].append(float(per[i]))

            def _m(lst):
                return sum(lst) / len(lst) if lst else 0.0

            all_v = [v for vs in buckets.values() for v in vs]
            overall, yes_no, number, other = _m(all_v), _m(buckets["yes/no"]), _m(buckets["number"]), _m(buckets["other"])
            last_elapsed_ms = 0.0
        else:
            overall, yes_no, number, other, last_elapsed_ms = evaluate_stratified(
                model, val_dataset, device, batch_size=int(train_cfg["batch_size"])
            )

        last_val_acc = (overall, yes_no, number, other)

        print(SEP)
        print(f"Epoch {epoch}/{total_epochs} | Text-Only DistilBERT")
        print(f"Train loss={avg_loss:.4f}")
        print(f"Val   vqa_acc={overall:.4f}  yes/no={yes_no:.4f}")
        print(f"      number={number:.4f}   other={other:.4f}")
        print(SEP)

        # ── Save checkpoints ───────────────────────────────────────────────────
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": overall,
        }
        torch.save(ckpt_data, ckpt_dir / f"checkpoint_epoch{epoch:02d}_acc{overall:.4f}.pt")
        torch.save(ckpt_data, ckpt_dir / "latest.pt")

        if overall > best_acc:
            best_acc = overall
            torch.save(ckpt_data, ckpt_dir / "best.pt")

        # ── Append to training log (never overwrite) ───────────────────────────
        log_entry = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 6),
            "val_accuracy": round(overall, 6),
            "val_acc_yesno": round(yes_no, 6),
            "val_acc_number": round(number, 6),
            "val_acc_other": round(other, 6),
            "lr": float(scheduler.get_last_lr()[0]),
            "epoch_duration_secs": round(epoch_duration, 2),
        }
        _append_training_log(log_path, log_entry)

    # ── Efficiency tracking ────────────────────────────────────────────────────
    avg_epoch_mins = (
        (sum(epoch_times) / len(epoch_times)) / 60 if epoch_times else 0.0
    )
    n_val_samples = n_debug_val if args.debug else 3000
    inference_ms = last_elapsed_ms / max(1, n_val_samples)

    peak_vram_gb: Optional[float] = (
        torch.cuda.max_memory_allocated() / 1e9
        if torch.cuda.is_available()
        else None
    )

    efficiency = {
        "model_name": "text_only_bert",
        "text_encoder": model_cfg["text_encoder"],
        "total_params_millions": round(total_params, 3),
        "trainable_params_millions": round(trainable_params, 3),
        "train_time_per_epoch_mins": round(avg_epoch_mins, 3),
        "inference_ms_per_sample": round(inference_ms, 4),
        "peak_vram_gb": round(peak_vram_gb, 4) if peak_vram_gb is not None else None,
        "train_subset": float(train_cfg.get("train_subset", 0.03)),
        "val_samples": n_val_samples,
        "max_seq_length": int(cfg["data"]["max_question_length"]),
        "batch_size": int(train_cfg["batch_size"]),
    }

    # ── Final results ─────────────────────────────────────────────────────────
    overall, yes_no, number, other = last_val_acc
    results = {
        "text_only_bert": {
            "overall": round(overall, 4),
            "yes_no": round(yes_no, 4),
            "number": round(number, 4),
            "other": round(other, 4),
            "efficiency": efficiency,
        }
    }
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nResults   → {results_path.relative_to(ROOT_DIR)}")
    print(f"Best ckpt → {(ckpt_dir / 'best.pt').relative_to(ROOT_DIR)}")
    print(f"Best val accuracy: {best_acc:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train text-only DistilBERT VQA baseline"
    )
    parser.add_argument(
        "--config",
        default="baselines/configs/baselines_config.yaml",
        help="Path to baselines_config.yaml relative to project root",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Auto-resume from latest checkpoint without prompt",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Smoke test: 200 train samples, 1 epoch, 100 val samples",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
