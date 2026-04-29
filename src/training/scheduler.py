"""Learning-rate scheduling helpers."""
from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Linear warmup then cosine decay from peak LR to 0.

    Args:
        optimizer:    the optimizer to wrap
        warmup_steps: steps over which LR rises linearly from 0 to base LR
        total_steps:  total training steps (warmup + decay)
    Returns:
        LambdaLR scheduler
    """
    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, _lr_lambda)


# ---------------------------------------------------------------------------
# Backward-compat alias with additional schedule types
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    schedule: str = "cosine",
) -> LambdaLR:
    """Return a LambdaLR with linear warmup followed by the chosen decay.

    Args:
        optimizer:    the optimizer to wrap
        warmup_steps: number of linear warm-up steps
        total_steps:  total training steps (epochs * steps_per_epoch)
        schedule:     "cosine" | "linear" | "constant"
    """
    def _lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        if schedule == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        if schedule == "linear":
            return max(0.0, 1.0 - progress)
        return 1.0  # constant

    return LambdaLR(optimizer, _lr_lambda)
