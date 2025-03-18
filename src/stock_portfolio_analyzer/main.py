import os
import pandas as pd
from .config import load_config
from .portfolio import analyze_portfolio
from .transactions import parse_statement_transactions
from .time_analysis import analyze_portfolio_over_time, generate_time_plots
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


def run_time_based_analysis(csv_file_path: str, statement_text: str = None, 
                          statement_file_path: str = None, config_file_path: str = None,
                          output_dir: str = None, show_plots: bool = False) -> dict:
    """
    Run time-based portfolio analysis using transaction statement data.
    
    Args:
        csv_file_path: Path to the portfolio CSV file
        statement_text: Raw transaction statement text (optional)
        statement_file_path: Path to transaction statement file (optional)
        config_file_path: Path to config file (optional)
        output_dir: Directory to save plots (optional)
        show_plots: Whether to display plots interactively
        
    Note: Either statement_text or statement_file_path must be provided.
    
    Returns:
        Dictionary with current and time-based analysis results
    """
    # Load configuration
    config = load_config(config_file_path)
    
    # Validate CSV file path
    if not validate_csv_path(csv_file_path):
        raise ValueError(f"Cannot access CSV file: {csv_file_path}")
    
    # Get statement text
    if not statement_text and not statement_file_path:
        raise ValueError("Either statement_text or statement_file_path must be provided")
    
    if not statement_text and statement_file_path:
        if not os.path.exists(statement_file_path):
            raise ValueError(f"Statement file not found: {statement_file_path}")
        
        with open(statement_file_path, 'r') as f:
            statement_text = f.read()
    
    # Parse transactions from statement
    logger.info("Parsing transactions from statement text")
    transactions_df = parse_statement_transactions(statement_text)
    
    if transactions_df.empty:
        logger.error("Failed to parse transactions from statement")
        raise ValueError("No transactions found in the provided statement")
    
    # Run current portfolio analysis
    current_analysis = analyze_portfolio(csv_file_path, config)
    
    # Run time-based analysis
    time_analysis = analyze_portfolio_over_time(transactions_df, config)
    
    # Generate plots if output directory is specified
    plots = {}
    if 'error' not in time_analysis:
        plots = generate_time_plots(time_analysis, output_dir, show_plots)
    
    results = {
        'current_analysis': current_analysis,
        'time_analysis': time_analysis,
        'plots': plots,
        'transactions': transactions_df
    }
    
    return results
