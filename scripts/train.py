"""Entry point for training the VQA model.

Example usage
-------------
Full multimodal run (auto-resumes if checkpoints exist):
    python scripts/train.py --config configs/config.yaml --mode multimodal

Auto-resume from latest checkpoint without a prompt:
    python scripts/train.py --config configs/config.yaml --resume

Resume from the checkpoint with best validation accuracy:
    python scripts/train.py --config configs/config.yaml --resume-best

Resume from a specific checkpoint:
    python scripts/train.py --config configs/config.yaml --resume outputs/checkpoints/checkpoint_epoch05_acc0.4712.pt

Quick smoke-test on 1 % of data:
    python scripts/train.py --config configs/config.yaml --debug
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.answer_vocab import build_vocab, load_vocab
from src.data.augmentations import get_train_transforms, get_val_transforms
from src.data.dataset import VQADataset, _collate_fn
from src.models.vqa_model import VQAModel
from src.training.trainer import Trainer
from src.utils import ROOT_DIR, setup_output_dirs
from src.utils.checkpoint import (
    CheckpointManager,
    find_latest_checkpoint,
    find_best_checkpoint,
    list_checkpoints,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the VQA model")
    p.add_argument("--config", default="configs/config.yaml",
                   help="Path to config YAML (default: configs/config.yaml)")
    p.add_argument("--mode", default="multimodal",
                   choices=["multimodal", "text_only", "image_only"],
                   help="Which modalities to use (default: multimodal)")
    p.add_argument(
        "--resume", nargs="?", const="latest", default=None, metavar="PATH",
        help=(
            "Resume training. No value → latest checkpoint; "
            "PATH → specific checkpoint file"
        ),
    )
    p.add_argument(
        "--resume-best", action="store_true",
        help="Resume from the checkpoint with the highest validation accuracy",
    )
    p.add_argument("--debug", action="store_true",
                   help="Load 1 %% of each split for a quick smoke-test")
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable Weights & Biases logging")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Resume resolution
# ---------------------------------------------------------------------------

def _resolve_resume(
    args: argparse.Namespace,
    ckpt_dir: Path,
) -> tuple[Path | None, bool]:
    """Return (checkpoint_path_or_None, was_auto_resolved).

    Handles --resume-best, --resume PATH, --resume (latest), and the
    interactive prompt when no flag is given but checkpoints exist.
    """
    # --resume-best
    if args.resume_best:
        path = find_best_checkpoint(ckpt_dir)
        if path:
            return path, True
        print("[Resume] No checkpoints found for --resume-best.")
        return None, False

    # --resume PATH  (explicit file)
    if args.resume and args.resume != "latest":
        return Path(args.resume), True

    # --resume  (no value → const "latest")
    if args.resume == "latest":
        path = find_latest_checkpoint(ckpt_dir)
        if path:
            return path, True
        print("[Resume] No checkpoints found — starting fresh.")
        return None, False

    # No flag — check whether checkpoints exist and prompt interactively
    existing = list_checkpoints(ckpt_dir)
    if not existing:
        return None, False

    print(f"\nFound {len(existing)} existing checkpoint(s):")
    for i, ckpt in enumerate(existing):
        tag = "  ← latest" if i == len(existing) - 1 else ""
        print(
            f"  {ckpt['path'].name}"
            f"  (epoch {ckpt['epoch']}, acc {ckpt['accuracy']:.4f})"
            f"{tag}"
        )

    try:
        ans = input("\nResume from latest? [y/n]: ").strip().lower()
    except (EOFError, OSError):
        ans = "y"

    if ans == "y":
        return existing[-1]["path"], True

    try:
        ans2 = input(
            "Start fresh? This will not delete existing checkpoints. [y/n]: "
        ).strip().lower()
    except (EOFError, OSError):
        ans2 = "y"

    if ans2 == "y":
        return None, False

    print("Exiting.")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_loaders(cfg: dict, ans2idx: dict, mode: str, debug: bool):
    from torch.utils.data import DataLoader, Subset

    data_cfg = cfg["data"]
    img_size: int = data_cfg["image_size"]
    use_aug: bool = data_cfg.get("augmentation", True)
    batch_size: int = cfg["training"]["batch_size"]
    num_workers: int = 0 if debug else 4

    train_ds = VQADataset(
        "train", cfg, ans2idx,
        transform=get_train_transforms(img_size) if use_aug else get_val_transforms(img_size),
        mode=mode,
    )
    val_ds = VQADataset(
        "val", cfg, ans2idx,
        transform=get_val_transforms(img_size),
        mode=mode,
    )

    if debug:
        train_ds = Subset(train_ds, range(max(1, len(train_ds) // 100)))
        val_ds = Subset(val_ds, range(max(1, len(val_ds) // 100)))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(not debug), collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(not debug), collate_fn=_collate_fn,
    )
    return train_loader, val_loader


def _init_wandb(cfg: dict, args: argparse.Namespace):
    if args.no_wandb:
        return None
    try:
        import wandb
        run = wandb.init(
            project=cfg["logging"]["wandb_project"],
            config={**cfg, "mode": args.mode, "debug": args.debug},
            resume="allow" if (args.resume or args.resume_best) else None,
        )
        return run
    except Exception as exc:
        print(f"[W&B] Unavailable ({exc}). Logging disabled.")
        return None


def _upload_to_hf(best_ckpt: Path, cfg: dict, mode: str) -> None:
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return
    if not best_ckpt.exists():
        print("[HF] Best checkpoint not found — skipping upload.")
        return
    try:
        from huggingface_hub import HfApi
        repo_id: str = cfg.get("huggingface", {}).get("repo_id", f"coco-vqa-{mode}")
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        url = api.upload_file(
            path_or_fileobj=str(best_ckpt),
            path_in_repo=best_ckpt.name,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"[HF] Uploaded best checkpoint → {url}")
    except Exception as exc:
        print(f"[HF] Upload failed: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT_DIR / args.config
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    setup_output_dirs(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  mode: {args.mode}  |  debug: {args.debug}")

    # ---- Vocab ----
    vocab_path = ROOT_DIR / cfg["paths"]["vocab_path"]
    if vocab_path.exists():
        ans2idx, idx2ans = load_vocab(vocab_path)
        print(f"Loaded vocab: {len(idx2ans):,} answers  ({vocab_path})")
    else:
        ann_path = ROOT_DIR / cfg["data"]["annotations_train"]
        print(f"Building vocab from {ann_path} …")
        ans2idx, idx2ans = build_vocab(ann_path)
        print(f"Built vocab: {len(idx2ans):,} answers  (saved to {vocab_path})")

    vocab_size: int = len(idx2ans)

    # ---- Data ----
    print("Loading datasets …")
    train_loader, val_loader = _build_loaders(cfg, ans2idx, args.mode, args.debug)
    print(
        f"  train batches: {len(train_loader):,}"
        f"  |  val batches: {len(val_loader):,}"
    )

    # ---- W&B ----
    wandb_run = _init_wandb(cfg, args)

    # ---- Model ----
    model = VQAModel(cfg, vocab_size=vocab_size, mode=args.mode)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ---- Trainer ----
    trainer = Trainer(
        model=model, config=cfg,
        train_loader=train_loader, val_loader=val_loader,
        device=device, wandb_run=wandb_run,
    )

    # ---- Resume ----
    ckpt_dir = ROOT_DIR / cfg["paths"]["checkpoint_dir"]
    resume_path, _ = _resolve_resume(args, ckpt_dir)

    start_epoch = 0
    best_val_accuracy = 0.0
    if resume_path is not None:
        _, _, _, _, start_epoch, ckpt_metrics = CheckpointManager.load(
            resume_path, model, trainer.optimizer, trainer.scheduler,
            trainer.scaler, device,
        )
        trainer._global_step = int(ckpt_metrics.get("global_step", 0))
        best_val_accuracy = float(
            ckpt_metrics.get("val_accuracy", ckpt_metrics.get("vqa_accuracy", 0.0))
        )
        print(
            f"Resuming from epoch {start_epoch}, "
            f"best accuracy so far: {best_val_accuracy * 100:.2f}%"
        )

    # ---- Train ----
    total_epochs: int = cfg["training"]["epochs"]
    remaining: int = total_epochs - start_epoch
    if remaining <= 0:
        print("Nothing to train — checkpoint already at full epoch count.")
    else:
        trainer.train(
            num_epochs=remaining,
            start_epoch=start_epoch,
            best_val_accuracy=best_val_accuracy,
        )

    # ---- HuggingFace upload ----
    if trainer.best_ckpt_path:
        _upload_to_hf(trainer.best_ckpt_path, cfg, args.mode)

    # ---- Finish W&B ----
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
