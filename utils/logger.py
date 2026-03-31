import logging
import sys
from pathlib import Path

_FMT     = "%(asctime)s  %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logger(log_path: Path) -> logging.Logger:
    """Configure the 'toy-qwen' logger with stdout + file handlers. Call once at startup."""
    logger = logging.getLogger("toy-qwen")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(_FMT, datefmt=_DATEFMT)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def get_logger() -> logging.Logger:
    """Return the shared 'toy-qwen' logger. setup_logger() must be called first."""
    return logging.getLogger("toy-qwen")
