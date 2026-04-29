"""Entry point for training the VQA model.

Example usage
-------------
Full multimodal run:
    python scripts/train.py --config configs/config.yaml --mode multimodal

Text-only ablation:
    python scripts/train.py --config configs/config.yaml --mode text_only

Image-only ablation:
    python scripts/train.py --config configs/config.yaml --mode image_only

Resume from a checkpoint:
    python scripts/train.py --config configs/config.yaml --mode multimodal \\
        --resume checkpoints/checkpoint_epoch0005.pt

Quick smoke-test on 1 % of data:
    python scripts/train.py --config configs/config.yaml --mode multimodal --debug
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

# Make the repo root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.answer_vocab import build_vocab, load_vocab
from src.data.augmentations import get_train_transforms, get_val_transforms
from src.data.dataset import VQADataset, _collate_fn
from src.models.vqa_model import VQAModel
from src.training.trainer import Trainer
from src.utils.checkpoint import CheckpointManager, find_latest_checkpoint


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
    p.add_argument("--resume", default=None, metavar="CHECKPOINT",
                   help="Path to a .pt checkpoint to resume training from")
    p.add_argument("--debug", action="store_true",
                   help="Load 1 %% of each split for a quick smoke-test")
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable Weights & Biases logging")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_loaders(cfg: dict, ans2idx: dict, mode: str, debug: bool):
    """Create train/val DataLoaders, applying 1 % Subset in debug mode."""
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
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(not debug),
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(not debug),
        collate_fn=_collate_fn,
    )
    return train_loader, val_loader


def _init_wandb(cfg: dict, args: argparse.Namespace):
    """Initialise a W&B run, returning the run object or None on failure."""
    if args.no_wandb:
        return None
    try:
        import wandb
        run = wandb.init(
            project=cfg["logging"]["wandb_project"],
            config={**cfg, "mode": args.mode, "debug": args.debug},
            resume="allow" if args.resume else None,
        )
        return run
    except Exception as exc:
        print(f"[W&B] Unavailable ({exc}). Logging disabled.")
        return None


def _upload_to_hf(best_ckpt: Path, cfg: dict, mode: str) -> None:
    """Upload best checkpoint to HuggingFace Hub if HF_TOKEN is set."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return
    if not best_ckpt.exists():
        print("[HF] Best checkpoint not found — skipping upload.")
        return
    try:
        from huggingface_hub import HfApi
        repo_id: str = cfg.get("huggingface", {}).get(
            "repo_id", f"coco-vqa-{mode}"
        )
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

    # ---- Config ----
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  mode: {args.mode}  |  debug: {args.debug}")

    # ---- Vocab ----
    vocab_path = Path(cfg["data"]["vocab_path"])
    if vocab_path.exists():
        ans2idx, idx2ans = load_vocab(vocab_path)
        print(f"Loaded vocab: {len(idx2ans):,} answers  ({vocab_path})")
    else:
        print(f"Building vocab from {cfg['data']['annotations_train']} …")
        ans2idx, idx2ans = build_vocab(cfg["data"]["annotations_train"])
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
        model=model,
        config=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        wandb_run=wandb_run,
    )

    # ---- Resume ----
    # Auto-detect the latest checkpoint when --resume is not given explicitly
    if args.resume is None:
        auto = find_latest_checkpoint(cfg["paths"]["checkpoint_dir"])
        if auto is not None:
            print(f"[Auto-resume] Found checkpoint: {auto}")
            args.resume = str(auto)

    start_epoch = 0
    if args.resume:
        _, _, _, _, start_epoch, ckpt_metrics = CheckpointManager.load(
            args.resume, model, trainer.optimizer, trainer.scheduler,
            trainer.scaler, device,
        )
        trainer._global_step = int(ckpt_metrics.get("global_step", 0))
        print(
            f"Resumed from '{args.resume}' — "
            f"epoch {start_epoch}, step {trainer._global_step}"
        )

    # ---- Train ----
    total_epochs: int = cfg["training"]["epochs"]
    remaining: int = total_epochs - start_epoch
    if remaining <= 0:
        print("Nothing to train — checkpoint already at full epoch count.")
    else:
        trainer.train(num_epochs=remaining, start_epoch=start_epoch)

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
