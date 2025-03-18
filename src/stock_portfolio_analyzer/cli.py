import argparse
import sys
import os
from .config import load_config
from .portfolio import analyze_portfolio
from .utils import validate_csv_path, format_currency, format_percent, get_color_for_value, reset_color
from .logger import get_logger

logger = get_logger(__name__)

def main():
    """
    Main entry point for the CLI application.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Analyze a stock portfolio from a CSV file."
    )
    parser.add_argument(
        "--csv-file",
        help="Path to the CSV file containing the portfolio data.",
        default=None
    )
    parser.add_argument(
        "--config-file",
        help="Path to the configuration file (YAML).",
        default=None
    )
    parser.add_argument(
        "--no-color",
        help="Disable colored output.",
        action="store_true"
    )
    parser.add_argument(
        "--debug",
        help="Enable debug logging.",
        action="store_true"
    )

    args = parser.parse_args()
    
    # Set debug log level if requested
    if args.debug:
        os.environ["LOG_LEVEL"] = "DEBUG"
    
    # Load configuration
    config = load_config(args.config_file)
    
    # Handle CSV file path from args, config, or environment
    csv_file_path = args.csv_file or config.get("csv_file_path")
    if not csv_file_path:
        logger.error("No CSV file specified. Use --csv-file option or set csv_file_path in config.")
        print("Error: No CSV file specified. Use --csv-file option or set csv_file_path in config.")
        sys.exit(1)
    
    # Validate CSV file
    if not validate_csv_path(csv_file_path):
        print(f"Error: Cannot access CSV file: {csv_file_path}")
        sys.exit(1)
    
    # Perform analysis
    try:
        logger.info("Starting portfolio analysis...")
        results = analyze_portfolio(csv_file_path, config)
        print_summary(results, use_color=not args.no_color)
    except Exception as e:
        logger.exception(f"Failed to analyze portfolio: {e}")
        print(f"Error: Failed to analyze portfolio: {e}")
        sys.exit(1)

def print_summary(results: dict, use_color: bool = True):
    """
    Pretty-print the results to the console.
    """
    df = results["dataframe"]
    
    # Sort by position value (descending)
    df_sorted = df.sort_values(by="value", ascending=False)
    
    print("\n=== Stock Portfolio Analysis ===\n")
    
    # Format for display
    col_formats = {
        "symbol": "{:<6}",
        "shares": "{:<10.2f}",
        "average_cost": "{:<12.2f}",
        "current_price": "{:<13.2f}",
        "value": "{:<15,.2f}",
        "gain_loss": "{:<15,.2f}",
        "gain_loss_percent": "{:<8.2f}"
    }
    
    # Print header
    header = "Symbol  Shares      Avg Cost     Current Price  Position Value    Gain/Loss        %"
    print(header)
    print("-" * len(header))
    
    # Print each position
    for _, row in df_sorted.iterrows():
        symbol = col_formats["symbol"].format(row["symbol"])
        shares = col_formats["shares"].format(row["shares"])
        avg_cost = col_formats["average_cost"].format(row["average_cost"])
        current = col_formats["current_price"].format(row["current_price"])
        value = col_formats["value"].format(row["value"])
        
        gain_loss = row["gain_loss"]
        gain_loss_pct = row["gain_loss_percent"]
        
        # Color coding for gain/loss values
        gl_color = get_color_for_value(gain_loss) if use_color else ""
        reset = reset_color() if use_color else ""
        
        gain_loss_str = col_formats["gain_loss"].format(gain_loss)
        gain_loss_pct_str = col_formats["gain_loss_percent"].format(gain_loss_pct)
        
        print(f"{symbol} {shares} ${avg_cost} ${current} ${value}  {gl_color}${gain_loss_str} {gain_loss_pct_str}%{reset}")
    
    print("\n=== Portfolio Totals ===\n")
    
    total_value = results["total_value"]
    total_cost = results["total_cost_basis"]
    total_gain_loss = results["total_gain_loss"]
    total_gain_loss_pct = results["total_gain_loss_percent"]
    
    # Color coding for total gain/loss
    gl_color = get_color_for_value(total_gain_loss) if use_color else ""
    reset = reset_color() if use_color else ""
    
    print(f"Total Portfolio Value: ${total_value:,.2f}")
    print(f"Total Cost Basis:      ${total_cost:,.2f}")
    print(f"Overall Gain/Loss:     {gl_color}${total_gain_loss:,.2f} ({total_gain_loss_pct:.2f}%){reset}")
    print()

if __name__ == "__main__":
    main()
