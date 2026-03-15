"""
app/core/logger.py
------------------
Single logging configuration for the entire project.
Import `get_logger` in every module:

    from app.core.logger import get_logger
    logger = get_logger(__name__)

Log files:
    logs/app.log      — all INFO+ messages (rotating, 5 MB x 3 files)
    logs/errors.log   — ERROR+ only
"""
from __future__ import annotations
import logging
import logging.handlers
from pathlib import Path

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def _build_handler(path: str, level: int) -> logging.Handler:
    handler = logging.handlers.RotatingFileHandler(
        _LOG_DIR / path,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
    return handler


def _build_console_handler() -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
    return handler


def _configure_root() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.DEBUG)
    root.addHandler(_build_console_handler())
    root.addHandler(_build_handler("app.log",    logging.INFO))
    root.addHandler(_build_handler("errors.log", logging.ERROR))

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_configure_root()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
