"""Checkpoint save/load with keep-last-N rotation and HuggingFace Hub upload."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


# ─────────────────────────────────────────────────────────────────────────────
# CheckpointManager — full-featured class API
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    """Manages saving, loading, rotation, and Hub upload of checkpoints.

    Filename convention: ``checkpoint_epoch{epoch:02d}_acc{val_acc:.4f}.pt``

    Usage
    -----
    ::
        mgr = CheckpointManager("checkpoints/", max_keep=3)
        path = mgr.save(model, opt, sched, scaler, epoch, metrics, config)
        model, opt, sched, scaler, epoch, metrics = CheckpointManager.load(
            path, model, opt, sched, scaler, device=device
        )
        CheckpointManager.keep_last_n("checkpoints/", n=3)
        CheckpointManager.upload_to_hub(path, "user/repo")
    """

    def __init__(self, checkpoint_dir: str | Path, max_keep: int = 3) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_keep = max_keep

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Any,
        scaler: Any,
        epoch: int,
        metrics: Dict[str, Any],
        config: dict,
        path: Optional[str | Path] = None,
    ) -> Path:
        """Save a full training snapshot.

        Saves: model, optimizer, scheduler, scaler state dicts plus epoch,
        metrics dict, and a snapshot of the config.

        Args:
            model:     nn.Module
            optimizer: torch Optimizer (or None)
            scheduler: LR scheduler with state_dict (or None)
            scaler:    GradScaler for mixed-precision (or None)
            epoch:     current epoch index
            metrics:   dict e.g. {"val_accuracy": 0.72, "val_loss": 0.8}
            config:    parsed config dict (stored as metadata, not used for restore)
            path:      explicit save path; auto-generated under checkpoint_dir
                       when None (format: checkpoint_epoch{N:02d}_acc{acc:.4f}.pt)

        Returns:
            Path to the saved file.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if path is None:
            val_acc = float(
                metrics.get("val_accuracy", metrics.get("vqa_accuracy", 0.0))
            )
            fname = f"checkpoint_epoch{epoch:02d}_acc{val_acc:.4f}.pt"
            path = self.checkpoint_dir / fname

        payload: Dict[str, Any] = {
            "epoch": epoch,
            "metrics": metrics,
            "config_snapshot": config,
            "model_state_dict": model.state_dict(),
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            try:
                payload["scheduler_state_dict"] = scheduler.state_dict()
            except Exception:
                pass
        if scaler is not None:
            try:
                payload["scaler_state_dict"] = scaler.state_dict()
            except Exception:
                pass

        torch.save(payload, path)
        self.keep_last_n(self.checkpoint_dir, self.max_keep)
        return Path(path)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    @staticmethod
    def load(
        path: str | Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Any = None,
        scaler: Any = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.nn.Module, Any, Any, Any, int, Dict[str, Any]]:
        """Load a checkpoint into the provided objects (in-place).

        Weights are first loaded to CPU, then moved to ``device`` (if given)
        to avoid CUDA OOM when loading on a different GPU.

        Returns:
            (model, optimizer, scheduler, scaler, epoch, metrics)
            Objects for which no state was saved are returned as-is.
        """
        payload = torch.load(path, map_location="cpu")

        model.load_state_dict(payload["model_state_dict"])
        if device is not None:
            model.to(device)

        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in payload:
            try:
                scheduler.load_state_dict(payload["scheduler_state_dict"])
            except Exception:
                pass

        if scaler is not None and "scaler_state_dict" in payload:
            try:
                scaler.load_state_dict(payload["scaler_state_dict"])
            except Exception:
                pass

        epoch = int(payload.get("epoch", 0))
        metrics: Dict[str, Any] = payload.get("metrics", {})

        return model, optimizer, scheduler, scaler, epoch, metrics

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    @staticmethod
    def keep_last_n(checkpoint_dir: str | Path, n: int) -> None:
        """Delete old checkpoints, keeping the *n* most recent plus the best.

        Files matching ``checkpoint_epoch*.pt`` are managed.
        Files matching ``best_model*.pt`` are never deleted.

        Args:
            checkpoint_dir: directory to scan
            n:              number of most-recent checkpoints to keep
        """
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return

        ckpts = sorted(
            checkpoint_dir.glob("checkpoint_epoch*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        if len(ckpts) <= n:
            return

        best = _extract_best_checkpoint(ckpts)
        keep: set[Path] = set(ckpts[-n:])
        if best is not None:
            keep.add(best)

        for ckpt in ckpts:
            if ckpt not in keep:
                try:
                    ckpt.unlink()
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # HuggingFace Hub upload
    # ------------------------------------------------------------------

    @staticmethod
    def upload_to_hub(
        checkpoint_path: str | Path,
        repo_id: str,
        token: Optional[str] = None,
    ) -> None:
        """Upload a checkpoint file to a HuggingFace Hub model repository.

        Reads ``HF_TOKEN`` from the environment when ``token`` is not given.
        Does nothing silently if no token is available.

        Args:
            checkpoint_path: local ``.pt`` file to upload
            repo_id:         HuggingFace repo, e.g. ``"username/coco-vqa"``
            token:           HF write token; falls back to ``HF_TOKEN`` env var
        """
        token = token or os.environ.get("HF_TOKEN")
        if not token:
            return

        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for Hub uploads. "
                "Install with: pip install huggingface_hub"
            )

        checkpoint_path = Path(checkpoint_path)
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo=checkpoint_path.name,
            repo_id=repo_id,
            repo_type="model",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

_ACC_RE = re.compile(r"_acc([0-9]+\.[0-9]+)\.pt$")


def _extract_best_checkpoint(ckpts: list[Path]) -> Optional[Path]:
    """Return the checkpoint with the highest accuracy encoded in its filename."""
    best_path: Optional[Path] = None
    best_acc = -1.0
    for p in ckpts:
        m = _ACC_RE.search(p.name)
        if m:
            acc = float(m.group(1))
            if acc > best_acc:
                best_acc = acc
                best_path = p
    return best_path


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible module-level functions
# (used by demo/app.py, scripts/train.py, and trainer.py)
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    ckpt_dir: Path,
    keep_last_n: int = 3,
    extra: Optional[dict] = None,
) -> Path:
    """Legacy save: model + optimizer + epoch, with automatic rotation."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"checkpoint_epoch{epoch:04d}.pt"
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    CheckpointManager.keep_last_n(ckpt_dir, keep_last_n)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> int:
    """Legacy load: weights into model (and optionally optimizer). Returns epoch."""
    map_location = device or "cpu"
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload.get("epoch", 0)


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:
    """Return the most recently modified checkpoint_epoch*.pt file, or None."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    ckpts = sorted(
        checkpoint_dir.glob("checkpoint_epoch*.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    return ckpts[-1] if ckpts else None


def find_best_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:
    """Return the checkpoint with the highest accuracy encoded in its filename."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    ckpts = list(checkpoint_dir.glob("checkpoint_epoch*.pt"))
    return _extract_best_checkpoint(ckpts)


def list_checkpoints(checkpoint_dir: str | Path) -> List[Dict[str, Any]]:
    """Return all checkpoint_epoch*.pt files sorted oldest-first.

    Each entry: {"path": Path, "epoch": int, "accuracy": float}
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []
    _EPOCH_RE = re.compile(r"checkpoint_epoch(\d+)")
    ckpts = sorted(
        checkpoint_dir.glob("checkpoint_epoch*.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    result: List[Dict[str, Any]] = []
    for p in ckpts:
        epoch_m = _EPOCH_RE.search(p.name)
        acc_m = _ACC_RE.search(p.name)
        result.append({
            "path": p,
            "epoch": int(epoch_m.group(1)) if epoch_m else 0,
            "accuracy": float(acc_m.group(1)) if acc_m else 0.0,
        })
    return result
