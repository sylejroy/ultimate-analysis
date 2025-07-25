"""Configuration management for Ultimate Analysis."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


# Global configuration cache
_config_cache: Optional[Dict[str, Any]] = None


def get_config() -> Dict[str, Any]:
    """Get the complete configuration dictionary."""
    global _config_cache
    
    if _config_cache is None:
        _load_config()
    
    return _config_cache or {}


def get_setting(key: str, default: Any = None) -> Any:
    """
    Get a configuration setting using dot notation.
    
    Args:
        key: Setting key in dot notation (e.g., "models.detection.confidence_threshold")
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    config = get_config()
    
    # Navigate through nested dict using dot notation
    value = config
    for part in key.split('.'):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return default
    
    return value


def _load_config() -> None:
    """Load configuration from YAML file."""
    global _config_cache
    
    # Find config file
    config_path = Path(__file__).parent.parent.parent.parent / "configs" / "default.yaml"
    
    if not config_path.exists():
        # Fallback to empty config if file not found
        _config_cache = {}
        return
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            _config_cache = yaml.safe_load(f) or {}
        
        # Apply environment overrides if needed
        _apply_env_overrides()
        
    except Exception as e:
        # If config loading fails, use empty config
        _config_cache = {}


def _apply_env_overrides() -> None:
    """Apply environment variable overrides to configuration."""
    # This could be expanded to support environment-specific overrides
    pass


def reload_config() -> None:
    """Force reload of configuration."""
    global _config_cache
    _config_cache = None


# Export the main functions
__all__ = ['get_config', 'get_setting', 'reload_config']
