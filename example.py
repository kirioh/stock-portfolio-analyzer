#!/usr/bin/env python3
"""
Example script demonstrating how to use the stock_portfolio_analyzer library programmatically
rather than through the CLI.
"""

import os
import pandas as pd
from stock_portfolio_analyzer.main import run_analysis
from stock_portfolio_analyzer.utils import format_currency, format_percent, get_color_for_value, reset_color

def main():
    """Run a portfolio analysis and display custom output."""
    # Set up environment for more detailed logging if desired
    os.environ["LOG_LEVEL"] = "INFO"
    
    # Define the path to your portfolio CSV
    csv_path = "./portfolio.csv"
    
    # Optional: specify a custom config file
    config_path = "./config.yaml"
    
    # Run the analysis
    results = run_analysis(csv_path, config_path)
    
    # Access the dataframe with all results
    df = results["dataframe"]
    
    # Print a custom report
    print("\n🚀 PORTFOLIO PERFORMANCE REPORT 🚀\n")
    
    # Sort by gain/loss percent
    df_sorted = df.sort_values(by="gain_loss_percent", ascending=False)
    
    # Print top performers and worst performers
    print("TOP PERFORMERS:")
    for _, row in df_sorted.head(2).iterrows():
        gain_color = get_color_for_value(row["gain_loss"])
        print(f"  {row['symbol']}: {gain_color}{format_percent(row['gain_loss_percent'])}{reset_color()} " +
              f"({format_currency(row['gain_loss'])})")
    
    print("\nWORST PERFORMERS:")
    for _, row in df_sorted.tail(2).iterrows():
        gain_color = get_color_for_value(row["gain_loss"])
        print(f"  {row['symbol']}: {gain_color}{format_percent(row['gain_loss_percent'])}{reset_color()} " +
              f"({format_currency(row['gain_loss'])})")
    
    # Access portfolio totals
    total_value = results["total_value"]
    total_gain_loss = results["total_gain_loss"]
    total_gain_loss_pct = results["total_gain_loss_percent"]
    
    # Print portfolio summary
    print(f"\nTOTAL PORTFOLIO VALUE: {format_currency(total_value)}")
    
    gain_color = get_color_for_value(total_gain_loss)
    print(f"OVERALL PERFORMANCE: {gain_color}{format_percent(total_gain_loss_pct)}{reset_color()} " +
          f"({format_currency(total_gain_loss)})")

if __name__ == "__main__":
    main()