import argparse
import sys
import os
import tempfile
from .config import load_config
from .portfolio import analyze_portfolio
from .utils import validate_csv_path, format_currency, format_percent, get_color_for_value, reset_color
from .logger import get_logger
from .main import run_analysis, run_time_based_analysis

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
        "--statement-file",
        help="Path to a transaction statement file for time-based analysis.",
        default=None
    )
    parser.add_argument(
        "--statement-text",
        help="Provide transaction statement text directly for time-based analysis.",
        default=None
    )
    parser.add_argument(
        "--save-statement",
        help="Save the provided statement text to a file with this name.",
        default=None
    )
    parser.add_argument(
        "--time-analysis",
        help="Perform time-based analysis of portfolio returns.",
        action="store_true"
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save output files like charts.",
        default=None
    )
    parser.add_argument(
        "--show-plots",
        help="Display analysis plots interactively.",
        action="store_true"
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
    
    # Create output directory if specified
    if args.output_dir and not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            logger.info(f"Created output directory: {args.output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")
            print(f"Error: Failed to create output directory: {e}")
            sys.exit(1)
    
    # Save statement text to file if requested
    statement_file_path = args.statement_file
    if args.statement_text and args.save_statement:
        try:
            with open(args.save_statement, 'w') as f:
                f.write(args.statement_text)
            logger.info(f"Saved statement text to {args.save_statement}")
            statement_file_path = args.save_statement
        except Exception as e:
            logger.error(f"Failed to save statement text: {e}")
            print(f"Warning: Failed to save statement text: {e}")
    
    # Perform analysis
    try:
        if args.time_analysis:
            # Check if we have statement data
            if not args.statement_text and not args.statement_file:
                logger.error("Time-based analysis requires statement data. Use --statement-file or --statement-text.")
                print("Error: Time-based analysis requires transaction statement data. "
                      "Use --statement-file or --statement-text.")
                sys.exit(1)
            
            logger.info("Starting time-based portfolio analysis...")
            results = run_time_based_analysis(
                csv_file_path=csv_file_path,
                statement_text=args.statement_text,
                statement_file_path=args.statement_file,
                config_file_path=args.config_file,
                output_dir=args.output_dir,
                show_plots=args.show_plots
            )
            
            print_time_analysis_summary(results, use_color=not args.no_color)
            
            # List generated plot files
            if args.output_dir and results.get('plots'):
                print(f"\nGenerated plots saved to: {args.output_dir}")
                for plot_name, file_path in results['plots'].items():
                    if isinstance(file_path, str):  # Skip error entries
                        print(f"  - {os.path.basename(file_path)}")
        else:
            # Standard analysis
            logger.info("Starting standard portfolio analysis...")
            results = run_analysis(csv_file_path, args.config_file)
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


def print_time_analysis_summary(results: dict, use_color: bool = True):
    """
    Pretty-print the time-based analysis results to the console.
    """
    # Print current portfolio summary
    print_summary(results["current_analysis"], use_color=use_color)
    
    time_analysis = results["time_analysis"]
    if "error" in time_analysis:
        print(f"\nError in time-based analysis: {time_analysis['error']}")
        return
    
    df = time_analysis["dataframe"]
    
    print("\n=== Portfolio Time-Based Analysis ===\n")
    
    # Print header
    header = "Date        Portfolio Value    Cost Basis       Return ($)      Return (%)"
    print(header)
    print("-" * len(header))
    
    # Print each date's portfolio state
    for _, row in df.iterrows():
        date = row["date_label"]
        value = row["portfolio_value"]
        cost = row["portfolio_cost"]
        ret = row["portfolio_return"]
        ret_pct = row["portfolio_return_pct"]
        
        # Color coding for return values
        ret_color = get_color_for_value(ret) if use_color else ""
        reset = reset_color() if use_color else ""
        
        print(f"{date}  ${value:,.2f}      ${cost:,.2f}      {ret_color}${ret:,.2f}      {ret_pct:.2f}%{reset}")
    
    # Print performance metrics
    if len(df) > 1:
        first_row = df.iloc[0]
        last_row = df.iloc[-1]
        
        first_date = first_row["date_label"]
        last_date = last_row["date_label"]
        
        value_change = last_row["portfolio_value"] - first_row["portfolio_value"]
        value_change_pct = (value_change / first_row["portfolio_value"] * 100) if first_row["portfolio_value"] > 0 else 0
        
        print("\n=== Performance Summary ===\n")
        
        change_color = get_color_for_value(value_change) if use_color else ""
        reset = reset_color() if use_color else ""
        
        print(f"Period: {first_date} to {last_date}")
        print(f"Portfolio Value Change: {change_color}${value_change:,.2f} ({value_change_pct:.2f}%){reset}")
        
        # Provide insights if available
        if value_change > 0:
            print("Your portfolio has grown in value over this period.")
        elif value_change < 0:
            print("Your portfolio has decreased in value over this period.")
        else:
            print("Your portfolio value has remained stable over this period.")
    
    print()


if __name__ == "__main__":
    main()
