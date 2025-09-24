"""Configuration settings management for Ultimate Analysis.

Simplified: loads a single YAML file (`configs/default.yaml`) with no environment
variable overrides (environment-based overrides were intentionally removed to
reduce surprise and keep deployments deterministic). Any runtime changes should
be applied by editing the YAML and calling `reload_config()`.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Global config cache
_config_cache: Optional[Dict] = None


def get_config() -> Dict[str, Any]:
    """Get the complete configuration dictionary.

    Returns:
        Dict containing all configuration settings
    """
    global _config_cache

    if _config_cache is None:
        _config_cache = _load_config()

    return _config_cache


def get_setting(key_path: str, default: Any = None) -> Any:
    """Get a setting value using dot notation.

    Args:
        key_path: Dot-separated path to the setting (e.g., "models.detection.confidence_threshold")
        default: Default value if setting not found

    Returns:
        The setting value or default

    Example:
        confidence = get_setting("models.detection.confidence_threshold")
        video_formats = get_setting("video.supported_formats", [".mp4"])
    """
    config = get_config()

    # Navigate through the nested dict using dot notation
    keys = key_path.split(".")
    current = config

    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def _load_config() -> Dict[str, Any]:
    """Load configuration from YAML file (no env overrides)."""
    # Find the project root (contains configs/)
    current_dir = Path(__file__).parent
    project_root = None

    # Walk up the directory tree to find configs/
    for parent in current_dir.parents:
        if (parent / "configs").exists():
            project_root = parent
            break

    if project_root is None:
        raise FileNotFoundError("Could not find configs/ directory in project structure")

    config_file = project_root / "configs" / "default.yaml"

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    # Load the YAML configuration
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config




def reload_config() -> None:
    """Reload configuration from file.

    This can be useful during development or when configuration changes.
    """
    global _config_cache
    _config_cache = None
    _config_cache = _load_config()
