"""
Centralized Logging Configuration
==================================
Call ``setup_logging()`` once at application startup (in ``src/main.py``)
to configure a consistent, structured log format across all modules.
"""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger with a structured format.

    Args:
        level: The minimum log level to capture (default: ``logging.INFO``).

    The format includes a timestamp, the module name (padded for alignment),
    the severity level, and the log message.  All output goes to **stderr**
    so it doesn't interfere with JSON API responses on stdout.
    """
    log_format = (
        "%(asctime)s | %(name)-32s | %(levelname)-7s | %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        stream=sys.stderr,
        force=True,  # override any prior basicConfig calls
    )

    # Silence noisy third-party loggers
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging initialised (level=%s)", logging.getLevelName(level)
    )
