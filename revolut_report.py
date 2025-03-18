#!/usr/bin/env python3
"""
Custom Revolut-style portfolio report script for Cieran Kelly's investments.
This script generates a report similar to the Revolut account statement format
and includes time-based analysis of portfolio performance.

Usage:
    python revolut_report.py
    python revolut_report.py --statement-file path/to/statement.txt --time-analysis
"""

import os
import sys
import argparse
import datetime
from src.stock_portfolio_analyzer.main import run_analysis, run_time_based_analysis
from src.stock_portfolio_analyzer.utils import format_currency, format_percent, get_color_for_value, reset_color

# Example statement text for demonstration
EXAMPLE_STATEMENT = """Account Statement
Generated on the 18 Mar 2025
Cieran Kelly
16 Kings Rd
Unit 3
Taringa
4068
AU
Period 09 Mar 2025 - 18 Mar 2025
Account name Cieran Kelly
Account number RAWC000141
Account summary
Starting Ending
Stocks value US$0 US$250.84
Cash value* US$0 US$0
Total US$0 US$250.84
*Cash value is the amount of cash in your stock trading account that has not been invested
Portfolio breakdown
Symbol Company ISIN Quantity Price Value % of Portfolio
PLTR Palantir US69608A1088 0.60975609 US$87.36 US$53.27 21.24%
META Meta Platforms US30303M1027 0.08230452 US$607.26 US$49.98 19.93%
NVDA Nvidia US67066G1040 0.42738695 US$119.49 US$51.07 20.36%
TSLA Tesla US88160R1014 0.19821605 US$238.02 US$47.18 18.81%
AMZN Amazon US0231351067 0.25205424 US$195.75 US$49.34 19.67%
Stocks value US$250.84 100%
Cash value US$0 0%
Total US$250.84
Transactions
Date Symbol Type Quantity Price Side Value Fees Commission
09 Mar 2025 11:28:39 AEDT Cash top-up US$50 US$0 US$0
09 Mar 2025 12:19:19 AEDT Cash top-up US$50 US$0 US$0
11 Mar 2025 00:30:01 AEDT TSLA Trade - Market 0.19821605 US$252.25 Buy US$50 US$0 US$0
11 Mar 2025 00:30:06 AEDT PLTR Trade - Market 0.60975609 US$82 Buy US$50 US$0 US$0
13 Mar 2025 16:56:53 AEDT Cash top-up US$50 US$0 US$0
13 Mar 2025 16:58:54 AEDT Cash top-up US$50 US$0 US$0
14 Mar 2025 00:30:00 AEDT NVDA Trade - Market 0.42738695 US$116.99 Buy US$50 US$0 US$0
14 Mar 2025 00:30:02 AEDT AMZN Trade - Market 0.25205424 US$198.37 Buy US$50 US$0 US$0
15 Mar 2025 09:49:43 AEDT Cash top-up US$50 US$0 US$0
18 Mar 2025 00:30:01 AEDT META Trade - Market 0.08230452 US$607.50 Buy US$50 US$0 US$0
Report lost or stolen card
+61 1300 281 208
Get help directly In app
Scan the QR code
This statement is provided by Revolut Payments Australia Pty Ltd. (Revolut Australia) in respect of your orders which Revolut has transmitted to its third party broker,
DriveWealth LLC (DriveWealth) for execution or onward transmission for execution by one of DriveWealth executing brokers.
Revolut Australia does not provide tax advice. You have the sole responsibility of determining the relevant tax impact to your trading and you should consult an
appropriate professional advisor if you have any questions or doubts in this regard.
Revolut Australia (ABN 21 634 823 180) is a company incorporated in Australia and licensed by the Australian Securities and Investments Commission (AFSL number
517589). Registered address: Level 8, 222 Exhibition Street, Melbourne VIC 3000 Australia.
© 2025 Revolut Payments Australia Pty Ltd"""

def print_standard_report(results):
    """Print a standard Revolut-style portfolio report."""
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


def print_time_based_report(results):
    """Print a time-based portfolio analysis report."""
    # Print standard report first
    print_standard_report(results['current_analysis'])
    
    # Now add time-based analysis
    time_analysis = results['time_analysis']
    if 'error' in time_analysis:
        print(f"\nError in time-based analysis: {time_analysis['error']}")
        return
    
    df = time_analysis['dataframe']
    
    print("\n" + "=" * 80)
    print("                  PORTFOLIO PERFORMANCE OVER TIME")
    print("=" * 80)
    
    # Print header
    print(f"\n{'Date':<12} {'Portfolio Value':<18} {'Cost Basis':<15} {'Return ($)':<15} {'Return (%)':<10}")
    print("-" * 80)
    
    # Print each date's portfolio state
    for _, row in df.iterrows():
        date = row["date_label"]
        value = row["portfolio_value"]
        cost = row["portfolio_cost"]
        ret = row["portfolio_return"]
        ret_pct = row["portfolio_return_pct"]
        
        # Color coding for return values
        ret_color = get_color_for_value(ret)
        
        print(f"{date:<12} ${value:<16.2f} ${cost:<13.2f} {ret_color}${ret:<13.2f} {ret_pct:<8.2f}%{reset_color()}")
    
    # Print performance metrics
    if len(df) > 1:
        first_row = df.iloc[0]
        last_row = df.iloc[-1]
        
        first_date = first_row["date_label"]
        last_date = last_row["date_label"]
        
        value_change = last_row["portfolio_value"] - first_row["portfolio_value"]
        value_change_pct = (value_change / first_row["portfolio_value"] * 100) if first_row["portfolio_value"] > 0 else 0
        
        print("\n" + "-" * 80)
        print("PERFORMANCE SUMMARY")
        print("-" * 80)
        
        change_color = get_color_for_value(value_change)
        
        print(f"Period:                  {first_date} to {last_date}")
        print(f"Portfolio Value Change:  {change_color}${value_change:.2f} ({value_change_pct:.2f}%){reset_color()}")
        
    # List generated plots
    if 'plots' in results and results['plots']:
        print("\n" + "-" * 80)
        print("GENERATED PLOTS")
        print("-" * 80)
        for plot_name, file_path in results['plots'].items():
            if isinstance(file_path, str):
                print(f"- {os.path.basename(file_path)}")
    
    print("\n" + "=" * 80)


def main():
    """Run portfolio analysis with optional time-based analysis."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate a Revolut-style portfolio report with optional time-based analysis."
    )
    parser.add_argument(
        "--csv-file",
        help="Path to the portfolio CSV file",
        default="./portfolio.csv"
    )
    parser.add_argument(
        "--statement-file",
        help="Path to a transaction statement file for time-based analysis",
        default=None
    )
    parser.add_argument(
        "--time-analysis",
        help="Perform time-based analysis of portfolio returns",
        action="store_true"
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save output files like charts",
        default="./output"
    )
    parser.add_argument(
        "--show-plots",
        help="Display analysis plots interactively",
        action="store_true"
    )
    parser.add_argument(
        "--debug",
        help="Enable debug logging",
        action="store_true"
    )
    
    args = parser.parse_args()
    
    # Set up environment
    os.environ["LOG_LEVEL"] = "DEBUG" if args.debug else "INFO"
    
    # Create output directory if it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the appropriate analysis
    try:
        if args.time_analysis:
            # Get statement text
            statement_text = None
            if args.statement_file:
                if os.path.exists(args.statement_file):
                    with open(args.statement_file, 'r') as f:
                        statement_text = f.read()
                else:
                    print(f"Error: Statement file not found: {args.statement_file}")
                    return 1
            else:
                # Use example statement for demonstration
                statement_text = EXAMPLE_STATEMENT
                print("Using example statement data (no statement file provided)")
            
            # Run time-based analysis
            results = run_time_based_analysis(
                csv_file_path=args.csv_file,
                statement_text=statement_text,
                output_dir=args.output_dir,
                show_plots=args.show_plots
            )
            
            # Print the time-based report
            print_time_based_report(results)
            
        else:
            # Run standard analysis
            results = run_analysis(args.csv_file)
            
            # Print the standard report
            print_standard_report(results)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())