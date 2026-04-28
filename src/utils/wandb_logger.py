"""Thin W&B wrapper that degrades gracefully when wandb is unavailable."""
from __future__ import annotations

from typing import Any


class WandbLogger:
    """Wraps wandb.log() and wandb.init() with a no-op fallback.

    When wandb is not installed or WANDB_MODE=disabled is set the logger
    silently does nothing, so training code can call logger.log() freely.
    """

    def __init__(self, project: str, config: dict, enabled: bool = True) -> None:
        self._run = None
        if not enabled:
            return
        try:
            import wandb
            self._run = wandb.init(project=project, config=config)
        except Exception as exc:
            print(f"[WandbLogger] W&B unavailable ({exc}). Logging disabled.")

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self._run is None:
            return
        try:
            self._run.log(metrics, step=step)
        except Exception:
            pass

    def watch(self, model, log_freq: int = 100) -> None:
        if self._run is None:
            return
        try:
            import wandb
            wandb.watch(model, log_freq=log_freq)
        except Exception:
            pass

    def finish(self) -> None:
        if self._run is not None:
            try:
                self._run.finish()
            except Exception:
                pass
