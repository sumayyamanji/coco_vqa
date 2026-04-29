"""YAML config loading with optional override support."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file and return its contents as a plain dict.

    Args:
        path: path to a ``.yaml`` / ``.yml`` file

    Returns:
        Nested dict mirroring the YAML structure.

    Raises:
        FileNotFoundError: if the file does not exist.
        ImportError:       if PyYAML is not installed.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required: pip install pyyaml") from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
