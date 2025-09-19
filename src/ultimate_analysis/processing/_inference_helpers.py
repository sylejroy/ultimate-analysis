"""Internal helpers for inference module.

Separated to keep `inference.py` lean and under line limits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ..config.settings import get_setting


def get_model_training_params(model_path: str) -> Dict[str, Any]:
    """Extract training parameters from a model's args.yaml.

    Args:
        model_path: Path to the model file (.pt)

    Returns:
        Dictionary of training parameters (may be empty if not found).
    """
    try:
        model_path_p = Path(model_path)
        args_yaml_path = model_path_p.parent / "args.yaml"

        if args_yaml_path.exists():
            with open(args_yaml_path, "r", encoding="utf-8") as f:
                args = yaml.safe_load(f)
                print(f"[INFERENCE] Loaded training parameters from {args_yaml_path}")
                return args or {}
        else:
            print(f"[INFERENCE] No args.yaml found at {args_yaml_path}")
    except Exception as e:
        print(f"[INFERENCE] Error reading model training parameters: {e}")

    return {}


def resolve_model_path(model_path: str) -> Optional[str]:
    """Resolve a model path to an absolute file path.

    Handles absolute/relative paths and bare filenames looked up under
    configured models directories.
    """
    # Absolute or explicit path with separators
    p = Path(model_path)
    if p.is_absolute() or ("/" in model_path or "\\" in model_path):
        if p.exists():
            return str(p)
        print(f"[INFERENCE] Absolute path does not exist: {model_path}")
        return None

    # Bare filename: search in configured models path
    models_path = Path(get_setting("models.base_path", "data/models"))

    # Prefer pretrained
    pretrained_path = models_path / "pretrained" / model_path
    if pretrained_path.exists():
        return str(pretrained_path)

    # Try detection folder
    detection_path = models_path / "detection" / model_path
    if detection_path.exists():
        return str(detection_path)

    print(f"[INFERENCE] Model file not found: {model_path}")
    return None
