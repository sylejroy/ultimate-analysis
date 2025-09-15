"""
Logging utilities for the Ultimate Analysis application.

This module provides a centralized logging system that respects the application's
debug configuration settings.
"""

import logging
import sys
from typing import Optional

from ultimate_analysis.config.settings import get_setting


class DebugLogger:
    """A logger that respects the app's debug configuration."""
    
    _loggers = {}
    _debug_enabled: Optional[bool] = None
    _log_level: Optional[str] = None
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get or create a logger for the given module name."""
        if name not in cls._loggers:
            cls._loggers[name] = cls._create_logger(name)
        return cls._loggers[name]
    
    @classmethod
    def _create_logger(cls, name: str) -> logging.Logger:
        """Create a new logger with proper configuration."""
        logger = logging.getLogger(name)
        
        # Only set up handler if not already done
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                get_setting("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        # Set level based on config
        log_level = cls._get_log_level()
        logger.setLevel(getattr(logging, log_level))
        
        return logger
    
    @classmethod
    def _get_log_level(cls) -> str:
        """Get the log level from config, with caching."""
        if cls._log_level is None:
            cls._log_level = get_setting("app.log_level", "INFO").upper()
        return cls._log_level
    
    @classmethod
    def is_debug_enabled(cls) -> bool:
        """Check if debug mode is enabled in config."""
        if cls._debug_enabled is None:
            cls._debug_enabled = get_setting("app.debug", False)
        return cls._debug_enabled
    
    @classmethod
    def debug_print(cls, message: str, module_name: str = "DEBUG") -> None:
        """Print a debug message only if debug mode is enabled."""
        if cls.is_debug_enabled():
            logger = cls.get_logger(module_name)
            logger.debug(message)
    
    @classmethod
    def conditional_print(cls, message: str, module_name: str = "INFO", level: str = "INFO") -> None:
        """Print a message at the specified level if enabled."""
        logger = cls.get_logger(module_name)
        log_level = getattr(logging, level.upper())
        logger.log(log_level, message)


def get_logger(module_name: str) -> logging.Logger:
    """Convenience function to get a logger for a module."""
    return DebugLogger.get_logger(module_name)


def debug_print(message: str, module_name: str = "DEBUG") -> None:
    """Convenience function for debug printing."""
    DebugLogger.debug_print(message, module_name)


def is_debug_enabled() -> bool:
    """Convenience function to check if debug is enabled."""
    return DebugLogger.is_debug_enabled()