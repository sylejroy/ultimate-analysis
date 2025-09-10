"""Configuration settings management for Ultimate Analysis.

This module provides access to YAML-based configuration with environment overrides.
Following the KISS principle - simple configuration access patterns.
"""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path


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
    keys = key_path.split('.')
    current = config
    
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def _load_config() -> Dict[str, Any]:
    """Load configuration from YAML file with environment overrides.
    
    Returns:
        Complete configuration dictionary
    """
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
    
    config_file = project_root / "configs" / "processing.yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    # Load the YAML configuration
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Apply environment overrides if they exist
    config = _apply_environment_overrides(config)
    
    return config


def _apply_environment_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to configuration.
    
    Environment variables should be prefixed with UA_ (Ultimate Analysis)
    and use double underscores for nested keys.
    
    Example: UA_MODELS__DETECTION__CONFIDENCE_THRESHOLD=0.7
    
    Args:
        config: Base configuration dictionary
        
    Returns:
        Configuration with environment overrides applied
    """
    import copy
    config = copy.deepcopy(config)
    
    # Look for environment variables with UA_ prefix
    for env_key, env_value in os.environ.items():
        if not env_key.startswith('UA_'):
            continue
            
        # Remove prefix and convert to dot notation
        key_path = env_key[3:].lower().replace('__', '.')
        
        # Convert string value to appropriate type
        value = _convert_env_value(env_value)
        
        # Set the value in config
        _set_nested_value(config, key_path, value)
    
    return config


def _convert_env_value(value: str) -> Any:
    """Convert environment variable string to appropriate Python type.
    
    Args:
        value: String value from environment variable
        
    Returns:
        Converted value (bool, int, float, or string)
    """
    # Handle boolean values
    if value.lower() in ('true', 'yes', '1', 'on'):
        return True
    elif value.lower() in ('false', 'no', '0', 'off'):
        return False
    
    # Try to convert to number
    try:
        if '.' in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        pass
    
    # Return as string
    return value


def _set_nested_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """Set a value in nested dictionary using dot notation.
    
    Args:
        config: Configuration dictionary to modify
        key_path: Dot-separated path to the setting
        value: Value to set
    """
    keys = key_path.split('.')
    current = config
    
    # Navigate to the parent of the target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the final value
    current[keys[-1]] = value


def reload_config() -> None:
    """Reload configuration from file.
    
    This can be useful during development or when configuration changes.
    """
    global _config_cache
    _config_cache = None
    _config_cache = _load_config()
