import logging
import os
from logging.handlers import RotatingFileHandler

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "portfolio_analyzer.log")

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance with both console and file handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL.upper())
    
    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        return logger

    # Console Handler
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional: File Handler with rotation
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
