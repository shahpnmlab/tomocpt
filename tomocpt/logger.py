import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
import os


class CustomLogger:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once (Singleton pattern)
        if CustomLogger._initialized:
            return

        CustomLogger._initialized = True

        # Create logs directory if it doesn't exist
        self.logs_dir = Path('logs')
        self.logs_dir.mkdir(exist_ok=True)

        # Configure the logger
        self.logger = logging.getLogger('application_logger')
        self.logger.setLevel(logging.INFO)

        # Prevent adding handlers multiple times
        if not self.logger.handlers:
            self._setup_handlers()

    def _setup_handlers(self):
        # Console Handler (INFO level and above)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File Handler (DEBUG level and above, with rotation)
        file_handler = RotatingFileHandler(
            self.logs_dir / 'application.log',
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def get_logger(self):
        """Get the configured logger instance."""
        return self.logger


# Create a function to get the logger instance
def get_logger():
    """Get the application logger instance."""
    return CustomLogger().get_logger()


# Example usage:
if __name__ == '__main__':
    logger = get_logger()

    # Test different logging levels
    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')