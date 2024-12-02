# logging_config.py
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
        log_level: int = logging.INFO,
        log_file: Optional[str] = None,
        logger_name: str = __name__
) -> logging.Logger:
    """
    Configure and return a logger instance with both console and optional file handlers.

    Args:
        log_level: The logging level (default: logging.INFO)
        log_file: Optional path to a log file. If None, only console logging is setup
        logger_name: Name of the logger (default: module name)

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if log_file is specified
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Default logger instance
default_logger = setup_logger()