"""Lightweight logger factory for consistent logging."""
import logging
from typing import Optional


def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger.
    Args:
        name: Logger name or None for root-like.
        level: Logging level.
    """
    logger = logging.getLogger(name if name else "openquant")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

