"""Shared utilities: root path, output setup, checkpointing, logging, attribution maps."""
from pathlib import Path

from .checkpoint import (
    CheckpointManager,
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    find_best_checkpoint,
    list_checkpoints,
)
from .wandb_logger import WandbLogger
from .gradcam import GradCAM
from .config_loader import load_config

# Absolute path to the project root (two levels up from src/utils/)
ROOT_DIR: Path = Path(__file__).resolve().parents[2]


def setup_output_dirs(config: dict) -> None:
    """Create all output directories defined in config['paths'] if they don't exist."""
    paths = config.get("paths", {})
    for key in (
        "checkpoint_dir",
        "eval_plots_dir",
        "eda_plots_dir",
        "heatmaps_dir",
        "scene_graphs_dir",
    ):
        if key in paths:
            (ROOT_DIR / paths[key]).mkdir(parents=True, exist_ok=True)
    for key in ("results_path", "training_log"):
        if key in paths:
            (ROOT_DIR / paths[key]).parent.mkdir(parents=True, exist_ok=True)


__all__ = [
    "ROOT_DIR",
    "setup_output_dirs",
    "CheckpointManager",
    "save_checkpoint",
    "load_checkpoint",
    "find_latest_checkpoint",
    "find_best_checkpoint",
    "list_checkpoints",
    "WandbLogger",
    "GradCAM",
    "load_config",
]
