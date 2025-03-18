from .config import load_config
from .portfolio import analyze_portfolio
from .utils import validate_csv_path
from .logger import get_logger

logger = get_logger(__name__)

def run_analysis(csv_file_path: str, config_file_path: str = None) -> dict:
    """
    Run the portfolio analysis with the given CSV and config files.
    
    This is the main programmatic entry point for using the library
    outside of the CLI interface.
    """
    # Load configuration
    config = load_config(config_file_path)
    
    # Validate CSV file path
    if not validate_csv_path(csv_file_path):
        raise ValueError(f"Cannot access CSV file: {csv_file_path}")
    
    # Perform analysis
    return analyze_portfolio(csv_file_path, config)
