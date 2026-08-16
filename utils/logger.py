"""
MedRax Logger Utility
======================
Provides a structured, configurable logging system used across all MedRax modules.

Features:
  - Console output with color-coded levels (via standard logging)
  - Optional rotating file handler for persistent logs
  - Optional Google Cloud Logging integration (production)
  - Consistent format: [TIMESTAMP] [LEVEL] [MODULE] message

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Model loaded successfully")
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_FORMAT = "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "logs"
_MAX_LOG_BYTES = 5 * 1024 * 1024  # 5 MB per file
_BACKUP_COUNT = 3

# Track whether the root MedRax logger has been configured
_configured = False


def _setup_root_logger(
    level: int = logging.INFO,
    log_to_file: bool = True,
    enable_cloud_logging: bool = False,
) -> None:
    """Configure the root 'medrax' logger once.

    This is called lazily on the first ``get_logger()`` call so that
    ``config.py`` is already imported and ``settings`` is available.

    Args:
        level: Minimum logging level.
        log_to_file: Whether to write logs to a rotating file.
        enable_cloud_logging: Whether to attach a Google Cloud Logging handler.
    """
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger("medrax")
    root.setLevel(level)
    root.propagate = False

    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT)

    # ── Console handler ───────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # ── File handler (optional) ───────────────────────────────────────────
    if log_to_file:
        try:
            _LOG_DIR.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                _LOG_DIR / "medrax.log",
                maxBytes=_MAX_LOG_BYTES,
                backupCount=_BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError as exc:
            root.warning("Could not create log file handler: %s", exc)

    # ── Google Cloud Logging handler (optional, production) ───────────────
    if enable_cloud_logging:
        try:
            import google.cloud.logging as cloud_logging

            client = cloud_logging.Client()
            cloud_handler = client.get_default_handler()
            cloud_handler.setLevel(level)
            root.addHandler(cloud_handler)
            root.info("Google Cloud Logging handler attached.")
        except Exception as exc:
            root.warning(
                "Could not attach Google Cloud Logging handler: %s. "
                "Falling back to local logging only.",
                exc,
            )


def get_logger(
    name: str,
    level: Optional[int] = None,
) -> logging.Logger:
    """Get a named logger under the ``medrax`` namespace.

    On the first call, this also configures the root ``medrax`` logger with
    console and (optionally) file handlers.

    Args:
        name: Module name — typically ``__name__``.  The returned logger will
              be named ``medrax.<name>`` so all MedRax logs share a
              common prefix.
        level: Optional override for this specific logger's level.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    # Lazy configuration — avoids circular import with config.py
    if not _configured:
        try:
            from config import settings

            _setup_root_logger(
                level=logging.DEBUG if settings.device != "cpu" else logging.INFO,
                log_to_file=True,
                enable_cloud_logging=settings.enable_cloud_logging,
            )
        except ImportError:
            # Fallback if config is not yet available
            _setup_root_logger()

    logger_name = f"medrax.{name}" if not name.startswith("medrax") else name
    logger = logging.getLogger(logger_name)

    if level is not None:
        logger.setLevel(level)

    return logger
