"""
Application settings and configuration management.

Handles loading and validation of configuration files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..core.exceptions import ConfigurationError
from .constants import DEFAULT_CONFIG


@dataclass
class PathSettings:
    """Path configuration settings."""
    data_dir: Path
    models_dir: Path
    cache_dir: Path
    logs_dir: Path
    raw_videos: Path
    processed_dev_data: Path
    raw_training_data: Path
    raw_dataset: Path
    processed_cropped_dataset: Path
    processed_augmented_dataset: Path
    models_pretrained: Path
    models_finetune: Path
    models_detection: Path
    models_player_id: Path
    models_segmentation: Path


@dataclass
class ModelSettings:
    """Model configuration settings."""
    detection_model: str
    tracking_model: str
    segmentation_model: str
    player_id_model: str
    confidence_threshold: float
    iou_threshold: float


@dataclass
class ProcessingSettings:
    """Processing configuration settings."""
    max_workers: int
    batch_size: int
    cache_enabled: bool
    gpu_enabled: bool


@dataclass
class Settings:
    """Application settings container."""
    paths: PathSettings
    models: ModelSettings
    processing: ProcessingSettings
    config: Dict[str, Any]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Settings':
        """Create Settings instance from configuration dictionary."""
        # Create path settings
        data_dir = Path(config.get('data_dir', 'data'))
        paths = PathSettings(
            data_dir=data_dir,
            models_dir=data_dir / 'models',
            cache_dir=data_dir / 'cache',
            logs_dir=Path(config.get('logs_dir', 'logs')),
            raw_videos=data_dir / 'raw' / 'videos',
            processed_dev_data=data_dir / 'processed' / 'dev_data',
            raw_training_data=data_dir / 'raw' / 'training_data',
            raw_dataset=data_dir / 'raw' / 'dataset',
            processed_cropped_dataset=data_dir / 'processed' / 'cropped_dataset',
            processed_augmented_dataset=data_dir / 'processed' / 'augmented_dataset',
            models_pretrained=data_dir / 'models' / 'pretrained',
            models_finetune=data_dir / 'models' / 'finetune',
            models_detection=data_dir / 'models' / 'detection',
            models_player_id=data_dir / 'models' / 'player_id',
            models_segmentation=data_dir / 'models' / 'segmentation'
        )
        
        # Create model settings
        model_config = config.get('models', {})
        
        # Resolve model paths to absolute paths relative to project root
        # Project root is one level up from src/ultimate_analysis/config/
        project_root = Path(__file__).parent.parent.parent.parent
        
        detection_model_path = model_config.get('detection_model', 'yolov8n.pt')
        if not Path(detection_model_path).is_absolute():
            detection_model_path = str(project_root / detection_model_path)
        
        field_model_path = model_config.get('field_model', '')
        if field_model_path and not Path(field_model_path).is_absolute():
            field_model_path = str(project_root / field_model_path)
        
        player_id_model_path = model_config.get('player_id_model', 'easyocr')
        if player_id_model_path != 'easyocr' and not Path(player_id_model_path).is_absolute():
            player_id_model_path = str(project_root / player_id_model_path)
        
        models = ModelSettings(
            detection_model=detection_model_path,
            tracking_model=model_config.get('tracking_model', 'bytetrack'),
            segmentation_model=field_model_path,
            player_id_model=player_id_model_path,
            confidence_threshold=model_config.get('confidence_threshold', 0.5),
            iou_threshold=model_config.get('iou_threshold', 0.5)
        )
        
        # Create processing settings
        processing_config = config.get('processing', {})
        processing = ProcessingSettings(
            max_workers=processing_config.get('max_workers', 4),
            batch_size=processing_config.get('batch_size', 16),
            cache_enabled=processing_config.get('cache_enabled', True),
            gpu_enabled=processing_config.get('gpu_enabled', True)
        )
        
        return cls(
            paths=paths,
            models=models,
            processing=processing,
            config=config
        )


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file or use defaults.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    if config_path is None:
        # Use default configuration
        return DEFAULT_CONFIG.copy()
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_file, 'r') as f:
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            else:
                raise ConfigurationError(f"Unsupported configuration format: {config_file.suffix}")
        
        # Merge with defaults
        merged_config = DEFAULT_CONFIG.copy()
        merged_config.update(config)
        
        return merged_config
        
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML configuration: {e}")
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration: {e}")


def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path(os.getenv("ULTIMATE_ANALYSIS_DATA_DIR", "data"))


def get_models_dir() -> Path:
    """Get the models directory path."""
    return get_data_dir() / "models"


def get_cache_dir() -> Path:
    """Get the cache directory path."""
    return get_data_dir() / "cache"


def get_logs_dir() -> Path:
    """Get the logs directory path."""
    return Path(os.getenv("ULTIMATE_ANALYSIS_LOGS_DIR", "logs"))


# Global settings instance
_settings: Optional[Settings] = None


def get_settings(config_path: Optional[str] = None) -> Settings:
    """
    Get application settings instance.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Settings instance
    """
    global _settings
    
    if _settings is None:
        config = load_config(config_path)
        _settings = Settings.from_config(config)
    
    return _settings


def reset_settings():
    """Reset global settings instance (mainly for testing)."""
    global _settings
    _settings = None
