"""Checkpoint save/load with keep-last-N rotation."""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    ckpt_dir: Path,
    keep_last_n: int = 3,
    extra: Optional[dict] = None,
) -> Path:
    """Save model + optimizer state and delete old checkpoints beyond keep_last_n."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"checkpoint_epoch{epoch:04d}.pt"
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)

    # rotate old checkpoints
    existing = sorted(glob.glob(str(ckpt_dir / "checkpoint_epoch*.pt")))
    for old in existing[:-keep_last_n]:
        os.remove(old)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> int:
    """Load weights into model (and optionally optimizer). Returns the saved epoch."""
    map_location = device or "cpu"
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload.get("epoch", 0)
