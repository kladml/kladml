"""
Structured Logging Configuration.

Configures Loguru to handle all application logging, including
intercepting standard library logging messages.
"""

import sys
import logging
from pathlib import Path
from loguru import logger

from kladml.config.settings import settings


class InterceptHandler(logging.Handler):
    """
    Redirects standard logging messages to Loguru.
    """
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging():
    """
    Initialize structured logging.
    
    - Removes default handlers
    - Configures stderr sink (colored)
    - Configures file sink (JSON/Text) if enabled
    - Intercepts standard logging (SQLAlchemy, requests, etc.)
    """
    # Remove default handlers
    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(logging.INFO)

    # Remove all existing loguru handlers and configure fresh
    logger.remove()
    
    # Determine level
    log_level = "DEBUG" if settings.debug else "INFO"
    
    # 1. Console handler (Human readable)
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )
    
    # 2. File handler (Robust)
    # Ensure logs directory exists
    log_dir = Path(settings.artifacts_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_dir / "kladml.log",
        rotation="10 MB",
        retention="1 month",
        level="DEBUG", # Always log debug to file for post-mortem
        compression="zip",
        format="{time} | {level} | {name}:{function}:{line} | {message}",
    )
    
    # Intercept specific libraries if needed (optional fine tuning)
    for name in ["uvicorn", "fastapi", "sqlalchemy"]:
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True

    logger.debug(f"Logging initialized. Level: {log_level}")
