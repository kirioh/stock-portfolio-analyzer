#!/usr/bin/env python3
"""
Custom Revolut-style portfolio report script for Cieran Kelly's investments.
This script generates a report similar to the Revolut account statement format.
"""

import os
import datetime
from stock_portfolio_analyzer.main import run_analysis
from stock_portfolio_analyzer.utils import format_currency, format_percent, get_color_for_value, reset_color

def main():
    """Run a portfolio analysis and display a Revolut-style report."""
    # Set up environment
    os.environ["LOG_LEVEL"] = "INFO"
    
    # Portfolio CSV path
    csv_path = "./portfolio.csv"
    
    # Run the analysis
    results = run_analysis(csv_path)
    
    # Get the dataframe with stock data
    df = results["dataframe"]
    
    # Calculate portfolio percentages
    total_value = results["total_value"]
    df["percentage"] = (df["value"] / total_value) * 100
    
    # Sort by percentage (descending)
    df_sorted = df.sort_values(by="percentage", ascending=False)
    
    # Print the report header
    today = datetime.datetime.now().strftime("%d %b %Y")
    print("\n" + "=" * 80)
    print(f"                  PORTFOLIO STATEMENT - GENERATED ON {today}")
    print("=" * 80)
    print("\nAccount name:    Cieran Kelly")
    print("Account number:  RAWC000141")
    print("\n" + "-" * 80)
    print("ACCOUNT SUMMARY")
    print("-" * 80)
    print(f"Total Portfolio Value:      {format_currency(total_value)}")
    
    gain_color = get_color_for_value(results["total_gain_loss"])
    print(f"Total Gain/Loss:           {gain_color}{format_currency(results['total_gain_loss'])} "
          f"({format_percent(results['total_gain_loss_percent'])}){reset_color()}")
    print("\n" + "-" * 80)
    print("PORTFOLIO BREAKDOWN")
    print("-" * 80)
    
    # Print column headers for the portfolio breakdown
    print(f"{'Symbol':<6} {'Shares':<12} {'Purchase Price':<15} {'Current Price':<15} "
          f"{'Value':<15} {'%':<8} {'Gain/Loss':<15}")
    print("-" * 80)
    
    # Print each position
    for _, row in df_sorted.iterrows():
        symbol = f"{row['symbol']:<6}"
        shares = f"{row['shares']:<12.8f}"
        avg_cost = f"${row['average_cost']:<13.2f}"
        current = f"${row['current_price']:<13.2f}"
        value = f"${row['value']:<13.2f}"
        percentage = f"{row['percentage']:<6.2f}%"
        
        gain_loss = row["gain_loss"]
        gain_loss_pct = row["gain_loss_percent"]
        
        # Color coding for gain/loss values
        gl_color = get_color_for_value(gain_loss)
        gl_str = f"{gl_color}${gain_loss:<6.2f} ({gain_loss_pct:.2f}%){reset_color()}"
        
        print(f"{symbol} {shares} {avg_cost} {current} {value} {percentage} {gl_str}")
    
    # Print footer
    print("-" * 80)
    print("\nNOTES:")
    print("1. Current prices are fetched in real-time from Yahoo Finance")
    print("2. Portfolio based on Revolut statement from 18 Mar 2025")
    print("3. Green values indicate gains, red values indicate losses")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()