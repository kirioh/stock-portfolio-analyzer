import pandas as pd
import os
from typing import Optional
from .logger import get_logger

logger = get_logger(__name__)

def validate_csv_path(csv_path: str) -> bool:
    """
    Validates that a CSV file exists and is readable.
    """
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return False
        
    if not os.path.isfile(csv_path):
        logger.error(f"Path is not a file: {csv_path}")
        return False
        
    if not os.access(csv_path, os.R_OK):
        logger.error(f"CSV file is not readable: {csv_path}")
        return False
        
    return True

def format_currency(value: float) -> str:
    """
    Formats a numeric value as a currency string with 2 decimal places.
    """
    return f"${value:,.2f}"

def format_percent(value: float) -> str:
    """
    Formats a numeric value as a percentage with 2 decimal places.
    """
    return f"{value:.2f}%"

def get_color_for_value(value: float) -> str:
    """
    Returns a color code for displaying numbers (green for positive, red for negative).
    For console output with ANSI color codes.
    """
    if value > 0:
        return "\033[92m"  # green
    elif value < 0:
        return "\033[91m"  # red
    return "\033[0m"      # default

def reset_color() -> str:
    """
    Resets the console color to default.
    """
    return "\033[0m"
