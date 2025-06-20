from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional
from datetime import datetime


def init_logger(
    name: str = "uncertimety",
    log_dir: Optional[Path] = None,
    log_file: str = "app.log",
) -> logging.Logger:
    if log_dir is None:
        log_dir = Path(__file__).resolve().parents[2] / "logs"

    # Make sure the directory exists BEFORE handler creation
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / log_file

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        file_handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=5)
        # file_formatter = logging.Formatter(
        #     "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        # )
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s\nMessage: %(message)s\n"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # Log run separator once per program start
    logger.info(
        "=" * 20 + f" New run at {datetime.now():%Y-%m-%d %H:%M:%S} " + "=" * 20
    )

    return logger
