"""Console logging for the application.

Every module gets its logger with get_logger("TAG"). The level comes from app.log_level
and the line format from logging.format in configs/default.yaml.
"""

import logging
import sys
from typing import Dict

from ..config.settings import get_setting

_loggers: Dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
    """Logger that writes to the console under the given name."""
    if name not in _loggers:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter(
                    get_setting(
                        "logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )
                )
            )
            logger.addHandler(handler)
        level = str(get_setting("app.log_level", "INFO")).upper()
        logger.setLevel(getattr(logging, level, logging.INFO))
        _loggers[name] = logger
    return _loggers[name]
