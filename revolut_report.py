#!/usr/bin/env python3
"""
Custom Revolut-style portfolio report script for Cieran Kelly's investments.
This script generates a report similar to the Revolut account statement format
and includes time-based analysis of portfolio performance with visualizations.

Usage:
    python revolut_report.py
    python revolut_report.py --statement-file path/to/statement.txt --time-analysis
    python revolut_report.py --time-analysis --show-plots
"""

import os
import sys
import argparse
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from src.stock_portfolio_analyzer.main import run_analysis, run_time_based_analysis
from src.stock_portfolio_analyzer.utils import format_currency, format_percent, get_color_for_value, reset_color
from src.stock_portfolio_analyzer.logger import get_logger

# Add terminal chart capabilities
import shutil

logger = get_logger(__name__)

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


def print_time_based_report(results, terminal_charts=False):
    """
    Print a time-based portfolio analysis report with visualization details.
    
    Args:
        results: Analysis results dictionary
        terminal_charts: Whether to display ASCII charts in the terminal
    """
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
        # Calculate time-based metrics
        time_metrics = calculate_time_metrics(df)
        
        first_row = df.iloc[0]
        last_row = df.iloc[-1]
        
        first_date = first_row["date_label"]
        last_date = last_row["date_label"]
        period_days = time_metrics.get("period_days", 0)
        
        value_change = time_metrics.get("value_change", 0)
        value_change_pct = time_metrics.get("value_change_pct", 0)
        
        print("\n" + "-" * 80)
        print("PORTFOLIO PERFORMANCE SUMMARY")
        print("-" * 80)
        
        change_color = get_color_for_value(value_change)
        
        print(f"Period:                  {first_date} to {last_date} ({period_days} days)")
        print(f"Portfolio Value Change:  {change_color}${value_change:.2f} ({value_change_pct:.2f}%){reset_color()}")
        
        # Win Rate (if available)
        if "win_rate" in time_metrics:
            win_rate = time_metrics["win_rate"]
            win_rate_color = get_color_for_value(win_rate - 50)  # Color based on better than 50%
            print(f"Win Rate:                {win_rate_color}{format_percent(win_rate)}{reset_color()}")
        
        # Add annualized return
        if "annualized_return" in time_metrics:
            ann_return = time_metrics["annualized_return"]
            ann_ret_color = get_color_for_value(ann_return)
            print(f"Annualized Return:       {ann_ret_color}{format_percent(ann_return)}{reset_color()}")
        
        # Add CAGR if period is long enough
        if "cagr" in time_metrics and period_days >= 30:
            cagr = time_metrics["cagr"]
            cagr_color = get_color_for_value(cagr)
            print(f"CAGR:                    {cagr_color}{format_percent(cagr)}{reset_color()}")
        
        # Add risk section if we have risk metrics
        if any(metric in time_metrics for metric in ["volatility", "max_drawdown", "value_at_risk_95", 
                                                    "drawdown_data", "downside_volatility"]):
            print("\n" + "-" * 80)
            print("RISK METRICS")
            print("-" * 80)
            
            # Add volatility metrics
            if "volatility" in time_metrics:
                volatility = time_metrics["volatility"]
                print(f"Volatility:              {format_percent(volatility)}")
                
                if "annualized_volatility" in time_metrics:
                    ann_volatility = time_metrics["annualized_volatility"]
                    print(f"Annualized Volatility:   {format_percent(ann_volatility)}")
            
            # Downside volatility
            if "downside_volatility" in time_metrics:
                downside_vol = time_metrics["downside_volatility"]
                print(f"Downside Volatility:     {format_percent(downside_vol)}")
                
                if "annualized_downside_volatility" in time_metrics:
                    ann_downside_vol = time_metrics["annualized_downside_volatility"]
                    print(f"Ann. Downside Vol.:      {format_percent(ann_downside_vol)}")
            
            # Add drawdown information
            if "max_drawdown" in time_metrics:
                drawdown = time_metrics["max_drawdown"]
                drawdown_color = get_color_for_value(-drawdown)  # Negative color for drawdown
                print(f"Maximum Drawdown:        {drawdown_color}{format_percent(drawdown)}{reset_color()}")
                
                # Show detailed drawdown information if available
                if "drawdown_data" in time_metrics:
                    drawdown_data = time_metrics["drawdown_data"]
                    if drawdown_data["peak_date"] and drawdown_data["trough_date"]:
                        print(f"Drawdown Period:         {drawdown_data['peak_date']} to {drawdown_data['trough_date']}")
                    
                    if drawdown_data["recovery_time"] > 0:
                        print(f"Recovery Time:           {drawdown_data['recovery_time']} days")
            
            # Add Value at Risk metrics
            if "value_at_risk_95" in time_metrics:
                var_95 = time_metrics["value_at_risk_95"]
                var_color = get_color_for_value(-var_95)
                print(f"Value at Risk (95%):     {var_color}{format_percent(var_95)}{reset_color()}")
                
                if "conditional_var_95" in time_metrics:
                    cvar_95 = time_metrics["conditional_var_95"]
                    cvar_color = get_color_for_value(-cvar_95)
                    print(f"Expected Shortfall (95%): {cvar_color}{format_percent(cvar_95)}{reset_color()}")
        
        # Add risk-adjusted metrics section
        if any(metric in time_metrics for metric in ["sharpe_ratio", "sortino_ratio", "calmar_ratio"]):
            print("\n" + "-" * 80)
            print("RISK-ADJUSTED PERFORMANCE METRICS")
            print("-" * 80)
            
            # Sharpe ratio
            if "sharpe_ratio" in time_metrics:
                sharpe = time_metrics["sharpe_ratio"]
                sharpe_color = get_color_for_value(sharpe)
                print(f"Sharpe Ratio (2% RF):    {sharpe_color}{sharpe:.2f}{reset_color()}")
            
            # Sortino ratio
            if "sortino_ratio" in time_metrics:
                sortino = time_metrics["sortino_ratio"]
                sortino_color = get_color_for_value(sortino)
                print(f"Sortino Ratio (2% RF):   {sortino_color}{sortino:.2f}{reset_color()}")
            
            # Calmar ratio
            if "calmar_ratio" in time_metrics:
                calmar = time_metrics["calmar_ratio"]
                calmar_color = get_color_for_value(calmar)
                print(f"Calmar Ratio:            {calmar_color}{calmar:.2f}{reset_color()}")
    
    # Display stock contribution to returns
    print("\n" + "-" * 80)
    print("STOCK CONTRIBUTION TO RETURNS")
    print("-" * 80)
    
    # Extract current portfolio with gain/loss per symbol
    current_df = results['current_analysis']['dataframe']
    if not current_df.empty:
        current_df['contribution'] = current_df['gain_loss'] / current_df['gain_loss'].sum() * 100 if current_df['gain_loss'].sum() != 0 else 0
        
        # Sort by contribution (absolute value)
        sorted_df = current_df.sort_values(by='contribution', key=abs, ascending=False)
        
        for _, row in sorted_df.iterrows():
            symbol = row['symbol']
            contribution = row['contribution']
            gain_loss = row['gain_loss']
            
            contrib_color = get_color_for_value(contribution)
            print(f"{symbol:<6}: {contrib_color}{format_percent(contribution)} contribution ({format_currency(gain_loss)}){reset_color()}")
    
    # Display terminal charts if requested
    if terminal_charts:
        # Generate and print terminal charts
        term_charts = generate_terminal_charts(results)
        print(term_charts)
    
    # List generated plots
    if 'plots' in results and results['plots']:
        print("\n" + "-" * 80)
        print("GENERATED VISUALIZATIONS")
        print("-" * 80)
        for plot_name, file_path in results['plots'].items():
            if isinstance(file_path, str) and 'error' not in plot_name:
                plot_title = {
                    'value_time': 'Portfolio Value vs. Cost Basis Over Time',
                    'returns_time': 'Portfolio Absolute Returns Over Time',
                    'returns_pct_time': 'Portfolio Percentage Returns Over Time'
                }.get(plot_name, os.path.basename(file_path))
                
                print(f"- {plot_title}: {os.path.basename(file_path)}")
        
        # Add guidance for how to view the plots
        if any(isinstance(file_path, str) for file_path in results['plots'].values()):
            output_dir = os.path.dirname(next(file_path for file_path in results['plots'].values() if isinstance(file_path, str)))
            print(f"\nAll visualizations saved to: {output_dir}")
            print("Use --show-plots flag to display them interactively")
    
    print("\n" + "=" * 80)


def main():
    """Run portfolio analysis with optional time-based analysis and visualizations."""
    # Suppress all stdout from yfinance
    import sys
    from io import StringIO
    
    class StdoutRedirector:
        def __init__(self):
            self.old_stdout = sys.stdout
            self.buffer = StringIO()
        
        def __enter__(self):
            sys.stdout = self.buffer
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout = self.old_stdout
            return False
            
    stdout_redirector = StdoutRedirector()
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate a Revolut-style portfolio report with time-based analysis and visualizations."
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
        "--save-plots",
        help="Save plots to the output directory",
        action="store_true",
        default=True
    )
    parser.add_argument(
        "--plot-format",
        help="Format for saved plot files (png, pdf, svg)",
        choices=["png", "pdf", "svg"],
        default="png"
    )
    parser.add_argument(
        "--debug",
        help="Enable debug logging",
        action="store_true"
    )
    parser.add_argument(
        "--additional-plots", "--additional-plot",
        help="Generate additional analysis plots",
        action="store_true"
    )
    parser.add_argument(
        "--terminal-charts",
        help="Display ASCII charts in the terminal",
        action="store_true"
    )
    parser.add_argument(
        "--quiet",
        help="Suppress log messages",
        action="store_true",
        default=True
    )
    
    args = parser.parse_args()
    
    # Set up environment
    import logging
    # Configure root logger
    if args.quiet and not args.debug:
        logging.getLogger().setLevel(logging.ERROR)
        os.environ["LOG_LEVEL"] = "ERROR"
        
        # Disable specific loggers
        for logger_name in ["src.stock_portfolio_analyzer.config", "src.stock_portfolio_analyzer.main", 
                           "src.stock_portfolio_analyzer.transactions", "src.stock_portfolio_analyzer.portfolio",
                           "src.stock_portfolio_analyzer.time_analysis", "src.stock_portfolio_analyzer.fetcher"]:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
    else:
        os.environ["LOG_LEVEL"] = "DEBUG" if args.debug else "INFO"
    
    # Create output directory if it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the appropriate analysis
    try:
        if args.time_analysis or args.show_plots:
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
            
            # Run time-based analysis with stdout redirection to suppress yfinance messages
            with stdout_redirector:
                results = run_time_based_analysis(
                    csv_file_path=args.csv_file,
                    statement_text=statement_text,
                    output_dir=args.output_dir,
                    show_plots=args.show_plots
                )
            
            # Generate additional plots if requested
            if args.additional_plots and 'time_analysis' in results and 'error' not in results['time_analysis']:
                # Create more advanced visualizations
                with stdout_redirector:
                    generate_additional_visualizations(
                        results=results,
                        output_dir=args.output_dir,
                        show_plots=args.show_plots,
                        file_format=args.plot_format
                    )
            
            # Print the time-based report with terminal charts if requested
            print_time_based_report(results, terminal_charts=args.terminal_charts)
            
        else:
            # Run standard analysis
            with stdout_redirector:
                results = run_analysis(args.csv_file)
            
            # Print the standard report
            print_standard_report(results)
            
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

def generate_additional_visualizations(results, output_dir, show_plots=False, file_format="png"):
    """
    Generate additional visualizations beyond the standard time plots.
    
    Args:
        results: Dictionary with analysis results
        output_dir: Directory to save plots
        show_plots: Whether to display plots interactively
        file_format: File format for saved plots (png, pdf, svg)
    
    Returns:
        Dictionary with paths to saved plots
    """
    logger.info("Generating additional visualizations")
    
    if 'time_analysis' not in results or 'error' in results['time_analysis']:
        logger.error("Cannot generate additional visualizations: time analysis data unavailable")
        return {}
    
    saved_plots = {}
    
    # Set aesthetic parameters
    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 12})
    
    # Get data
    time_df = results['time_analysis']['dataframe']
    current_df = results['current_analysis']['dataframe']
    transactions_df = results.get('transactions', pd.DataFrame())
    
    # 1. Portfolio Composition Pie Chart
    try:
        if not current_df.empty:
            plt.figure(figsize=(10, 8))
            
            # Create a pie chart of current portfolio composition
            labels = current_df['symbol']
            sizes = current_df['value']
            colors = sns.color_palette('viridis', len(current_df))
            
            # Create explode effect for the largest holding
            explode = [0.1 if i == sizes.idxmax() else 0 for i in range(len(sizes))]
            
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, 
                   shadow=True, explode=explode, colors=colors)
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.title('Current Portfolio Composition', fontsize=16)
            
            filename = os.path.join(output_dir, f'portfolio_composition.{file_format}')
            plt.savefig(filename)
            saved_plots['composition'] = filename
            
            if show_plots:
                plt.show()
            else:
                plt.close()
    
    except Exception as e:
        logger.error(f"Error generating portfolio composition chart: {e}")
    
    # 2. Performance Comparison with Market Benchmarks (simulated)
    try:
        if len(time_df) > 1:
            plt.figure(figsize=(12, 6))
            
            # Get portfolio returns
            portfolio_returns = time_df['portfolio_return_pct']
            dates = time_df['date']
            
            # Create simulated benchmark data (this would normally come from an API)
            # This is just for demonstration - in a real implementation, fetch actual market data
            np.random.seed(42)  # For reproducibility
            sp500_returns = np.cumsum(np.random.normal(0.0005, 0.005, len(dates)))
            tech_index_returns = np.cumsum(np.random.normal(0.001, 0.008, len(dates)))
            
            # Plot comparison
            plt.plot(dates, portfolio_returns, 'b-', linewidth=2, label='Your Portfolio')
            plt.plot(dates, sp500_returns, 'g--', linewidth=2, label='S&P 500 (Simulated)')
            plt.plot(dates, tech_index_returns, 'r-.', linewidth=2, label='Tech Index (Simulated)')
            
            plt.title('Portfolio Performance vs Market Benchmarks', fontsize=16)
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Return (%)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Format x-axis date labels
            plt.xticks(dates, time_df['date_label'], rotation=45)
            plt.tight_layout()
            
            filename = os.path.join(output_dir, f'benchmark_comparison.{file_format}')
            plt.savefig(filename)
            saved_plots['benchmark'] = filename
            
            if show_plots:
                plt.show()
            else:
                plt.close()
    
    except Exception as e:
        logger.error(f"Error generating benchmark comparison chart: {e}")
    
    # 3. Contribution to Returns - Waterfall Chart
    try:
        if not current_df.empty:
            # Sort by contribution
            current_df['contribution_value'] = current_df['gain_loss']
            sorted_df = current_df.sort_values('contribution_value', ascending=True)
            
            # Prepare data for waterfall chart
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Define colors based on positive/negative values
            colors = ['green' if x >= 0 else 'red' for x in sorted_df['contribution_value']]
            
            # Create the waterfall chart
            ax.bar(sorted_df['symbol'], sorted_df['contribution_value'], color=colors)
            
            # Add a horizontal line at y=0
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add total gain/loss
            total_gain_loss = results['current_analysis']['total_gain_loss']
            ax.text(len(sorted_df['symbol']) - 0.5, total_gain_loss, 
                   f'Total: {format_currency(total_gain_loss)}',
                   fontweight='bold', ha='center', va='bottom')
            
            # Format
            ax.set_title('Contribution to Overall Returns by Symbol', fontsize=16)
            ax.set_xlabel('Symbol', fontsize=14)
            ax.set_ylabel('Contribution ($)', fontsize=14)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            filename = os.path.join(output_dir, f'contribution_waterfall.{file_format}')
            plt.savefig(filename)
            saved_plots['waterfall'] = filename
            
            if show_plots:
                plt.show()
            else:
                plt.close()
    
    except Exception as e:
        logger.error(f"Error generating contribution waterfall chart: {e}")
    
    # 4. Transaction Timeline
    try:
        if not transactions_df.empty:
            # Filter for actual trades
            trade_df = transactions_df[transactions_df['type'] == 'Trade']
            
            if not trade_df.empty:
                plt.figure(figsize=(12, 6))
                
                # Group by date and side
                buy_df = trade_df[trade_df['side'] == 'Buy']
                sell_df = trade_df[trade_df['side'] == 'Sell']
                
                # Plot buy transactions
                for _, row in buy_df.iterrows():
                    plt.scatter(row['date'], row['value'], color='green', 
                              marker='^', s=100, label='Buy' if 'Buy' not in plt.gca().get_legend_handles_labels()[1] else "")
                    plt.text(row['date'], row['value']*1.05, f"{row['symbol']} ${row['value']:.2f}", 
                           ha='center', va='bottom', fontsize=9)
                
                # Plot sell transactions
                for _, row in sell_df.iterrows():
                    plt.scatter(row['date'], -row['value'], color='red', 
                              marker='v', s=100, label='Sell' if 'Sell' not in plt.gca().get_legend_handles_labels()[1] else "")
                    plt.text(row['date'], -row['value']*1.05, f"{row['symbol']} ${row['value']:.2f}", 
                           ha='center', va='top', fontsize=9)
                
                # Format
                plt.title('Transaction Timeline', fontsize=16)
                plt.xlabel('Date', fontsize=14)
                plt.ylabel('Transaction Value ($)', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Add zero line
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                plt.tight_layout()
                
                filename = os.path.join(output_dir, f'transaction_timeline.{file_format}')
                plt.savefig(filename)
                saved_plots['timeline'] = filename
                
                if show_plots:
                    plt.show()
                else:
                    plt.close()
    
    except Exception as e:
        logger.error(f"Error generating transaction timeline chart: {e}")
    
    # 5. Portfolio Value Heatmap
    try:
        if not current_df.empty and len(time_df) > 1:
            # Create a heatmap showing daily portfolio value changes
            # For demonstration, we'll create a simulated daily value chart
            
            # Get the first and last date
            first_date = time_df.iloc[0]['date']
            last_date = time_df.iloc[-1]['date']
            
            # Generate a range of dates
            date_range = pd.date_range(start=first_date, end=last_date, freq='D')
            
            # Create simulated daily values (in real implementation, fetch actual daily data)
            np.random.seed(42)  # For reproducibility
            daily_changes = np.random.normal(0.001, 0.01, len(date_range))
            daily_values = time_df.iloc[0]['portfolio_value'] * np.cumprod(1 + daily_changes)
            
            # Create a DataFrame
            daily_df = pd.DataFrame({
                'date': date_range,
                'value': daily_values
            })
            
            # Extract month and day
            daily_df['month'] = daily_df['date'].dt.month
            daily_df['day'] = daily_df['date'].dt.day
            
            # Create a pivot table for the heatmap
            heatmap_data = pd.pivot_table(
                daily_df, 
                values='value', 
                index='day',
                columns='month'
            )
            
            # Create the heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(heatmap_data, cmap='viridis', annot=False, fmt=".1f", cbar_kws={'label': 'Portfolio Value ($)'})
            
            plt.title('Portfolio Value Heatmap (Day by Month)', fontsize=16)
            plt.xlabel('Month', fontsize=14)
            plt.ylabel('Day', fontsize=14)
            
            plt.tight_layout()
            
            filename = os.path.join(output_dir, f'portfolio_heatmap.{file_format}')
            plt.savefig(filename)
            saved_plots['heatmap'] = filename
            
            if show_plots:
                plt.show()
            else:
                plt.close()
    
    except Exception as e:
        logger.error(f"Error generating portfolio heatmap: {e}")
    
    logger.info(f"Generated {len(saved_plots)} additional visualizations")
    
    return saved_plots


def generate_terminal_bar_chart(labels, values, title, width=None, height=10, colors=None, max_bar_width=50):
    """
    Generate an ASCII bar chart for terminal display.
    
    Args:
        labels: List of labels for each bar
        values: List of values for each bar
        title: Chart title
        width: Terminal width (auto-detected if None)
        height: Chart height in lines
        colors: List of color functions for each bar
        max_bar_width: Maximum width for bars to prevent excessive width
    
    Returns:
        String containing ASCII chart
    """
    if width is None:
        # Auto-detect terminal width, but use a reasonable default
        term_width, _ = shutil.get_terminal_size((80, 20))
        width = min(term_width, 80)  # Limit width to something reasonable
    
    # Reserve space for labels and values
    max_label_len = max(len(str(label)) for label in labels) + 1
    max_value_len = max(len(f"{value:.2f}") for value in values) + 1
    available_width = min(width - max_label_len - max_value_len - 5, max_bar_width)
    
    # Normalize values to fit height
    max_val = max(abs(v) for v in values) if values else 1
    normalized = [value / max_val * (height - 1) for value in values]
    
    # Generate the chart
    chart = []
    
    # Add title
    if title:
        chart.append(f"{title}".center(width))
        chart.append("-" * width)
    
    # Generate bars
    for i, (label, value, norm) in enumerate(zip(labels, values, normalized)):
        bar_width = int(abs(norm) * available_width / height)
        
        # Determine bar character based on value
        if value >= 0:
            bar = "█" * bar_width
        else:
            bar = "▒" * bar_width
        
        # Apply color if provided
        if colors and i < len(colors):
            color_func = colors[i] if callable(colors[i]) else (lambda x: x)
            bar = color_func(bar)
        
        # Format output line with label and value
        line = f"{str(label):<{max_label_len}} │ {bar} {value:.2f}"
        chart.append(line)
    
    return "\n".join(chart)


def generate_terminal_sparkline(values, min_value=None, max_value=None, width=60, label_format="Value"):
    """
    Generate a sparkline for values to display in terminal.
    
    Args:
        values: List of numeric values
        min_value: Override minimum value for scaling
        max_value: Override maximum value for scaling
        width: Width of sparkline
        label_format: Format string for value labels
    
    Returns:
        String containing sparkline with labels
    """
    if not values:
        return ""
    
    # Sample values to fit width
    if len(values) > width:
        # Take samples at regular intervals
        indices = [int(i * (len(values) - 1) / (width - 1)) for i in range(width)]
        samples = [values[i] for i in indices]
    else:
        # Pad with repeated values if too few
        samples = values + [values[-1]] * (width - len(values))
        samples = samples[:width]
    
    # Determine scale
    min_val = min_value if min_value is not None else min(values)
    max_val = max_value if max_value is not None else max(values)
    value_range = max_val - min_val
    
    # Sparkline characters from low to high
    sparkchars = "▁▂▃▄▅▆▇█"
    
    if value_range == 0:
        # All values are the same
        line = sparkchars[3] * width
    else:
        # Map values to sparkline characters
        line = ""
        for value in samples:
            if value >= max_val:
                line += sparkchars[-1]
            elif value <= min_val:
                line += sparkchars[0]
            else:
                # Normalize value to character range
                normalized = (value - min_val) / value_range
                index = min(int(normalized * (len(sparkchars) - 1)), len(sparkchars) - 1)
                line += sparkchars[index]
    
    # Format the output with clear labels
    if label_format == "Currency":
        result = [
            line,
            f"Min: ${min_val:.2f}   Current: ${values[-1]:.2f}   Max: ${max_val:.2f}"
        ]
    elif label_format == "Percent":
        result = [
            line,
            f"Min: {min_val:.2f}%   Current: {values[-1]:.2f}%   Max: {max_val:.2f}%"
        ]
    else:
        result = [
            line,
            f"Min: {min_val:.2f}   Current: {values[-1]:.2f}   Max: {max_val:.2f}"
        ]
    
    return "\n".join(result)


def calculate_time_metrics(time_df):
    """
    Calculate various time-based metrics for portfolio analysis.
    
    Args:
        time_df: DataFrame containing time-based analysis data
    
    Returns:
        Dictionary of time-based metrics
    """
    if time_df.empty or len(time_df) < 2:
        return {}
    
    # Get first and last rows
    first_row = time_df.iloc[0]
    last_row = time_df.iloc[-1]
    
    # Basic metrics
    days_diff = (last_row["date"] - first_row["date"]).days
    value_change = last_row["portfolio_value"] - first_row["portfolio_value"]
    value_change_pct = (value_change / first_row["portfolio_value"] * 100) if first_row["portfolio_value"] > 0 else 0
    
    metrics = {
        "period_days": days_diff,
        "value_change": value_change,
        "value_change_pct": value_change_pct,
    }
    
    # Win Rate - percentage of periods with positive returns
    if len(time_df) >= 3:
        period_returns = []
        positive_periods = 0
        
        for i in range(1, len(time_df)):
            prev_value = time_df.iloc[i-1]["portfolio_value"]
            current_value = time_df.iloc[i]["portfolio_value"]
            if prev_value > 0:
                period_return = ((current_value / prev_value) - 1) * 100
                period_returns.append(period_return)
                if period_return > 0:
                    positive_periods += 1
        
        if period_returns:
            win_rate = (positive_periods / len(period_returns)) * 100
            metrics["win_rate"] = win_rate
    
    # Calculate annualized metrics if period is significant
    if days_diff >= 1:
        # Annualized return
        annualized_return = (((1 + value_change_pct/100) ** (365/days_diff)) - 1) * 100 if days_diff > 0 else 0
        metrics["annualized_return"] = annualized_return
        
        # CAGR (Compound Annual Growth Rate)
        cagr = (((last_row["portfolio_value"] / first_row["portfolio_value"]) ** (365/days_diff)) - 1) * 100 if days_diff > 0 and first_row["portfolio_value"] > 0 else 0
        metrics["cagr"] = cagr
        
        # Calculate volatility and other risk metrics if we have enough data points
        if len(time_df) >= 3:
            returns = []
            negative_returns = []
            
            for i in range(1, len(time_df)):
                prev_value = time_df.iloc[i-1]["portfolio_value"]
                current_value = time_df.iloc[i]["portfolio_value"]
                if prev_value > 0:
                    daily_return = (current_value / prev_value) - 1
                    returns.append(daily_return)
                    
                    # Collect negative returns for downside risk calculations
                    if daily_return < 0:
                        negative_returns.append(daily_return)
            
            if returns:
                # Daily volatility
                volatility = np.std(returns) * 100  # Convert to percentage
                metrics["volatility"] = volatility
                
                # Annualized volatility
                annualized_volatility = volatility * np.sqrt(365)
                metrics["annualized_volatility"] = annualized_volatility
                
                # Sharpe ratio (assuming risk-free rate of 2%)
                risk_free_rate = 2.0  # 2% annual risk-free rate
                daily_risk_free = risk_free_rate / 365
                if volatility > 0:
                    # Sharpe ratio = (Expected Return - Risk Free Rate) / Standard Deviation
                    sharpe = ((annualized_return - risk_free_rate) / annualized_volatility) if annualized_volatility > 0 else 0
                    metrics["sharpe_ratio"] = sharpe
                
                # Sortino ratio (focuses on downside risk)
                if negative_returns:
                    downside_volatility = np.std(negative_returns) * 100
                    metrics["downside_volatility"] = downside_volatility
                    
                    # Annualized downside volatility
                    annualized_downside_vol = downside_volatility * np.sqrt(365)
                    metrics["annualized_downside_volatility"] = annualized_downside_vol
                    
                    # Sortino ratio
                    if annualized_downside_vol > 0:
                        sortino = ((annualized_return - risk_free_rate) / annualized_downside_vol)
                        metrics["sortino_ratio"] = sortino
                
                # Calmar ratio (return / max drawdown)
                if "max_drawdown" in metrics and metrics["max_drawdown"] > 0:
                    calmar = annualized_return / metrics["max_drawdown"]
                    metrics["calmar_ratio"] = calmar
    
    # Calculate max drawdown and related metrics
    if len(time_df) >= 2:
        portfolio_values = time_df["portfolio_value"].tolist()
        max_drawdown = 0
        peak = portfolio_values[0]
        peak_idx = 0
        trough = peak
        trough_idx = 0
        current_drawdown_start = 0
        
        # Data for drawdown analysis
        drawdown_data = {
            "max_drawdown": 0,
            "max_drawdown_pct": 0,
            "recovery_time": 0,
            "peak_date": None,
            "trough_date": None
        }
        
        for i, value in enumerate(portfolio_values):
            if value > peak:
                peak = value
                peak_idx = i
                current_drawdown_start = i
            
            # Calculate current drawdown
            current_drawdown = (peak - value) / peak * 100 if peak > 0 else 0
            
            # If this is a new max drawdown, update metrics
            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown
                trough = value
                trough_idx = i
                
                # Update drawdown data
                drawdown_data["max_drawdown"] = peak - trough
                drawdown_data["max_drawdown_pct"] = max_drawdown
                drawdown_data["peak_date"] = time_df.iloc[peak_idx]["date_label"] if "date_label" in time_df.columns else None
                drawdown_data["trough_date"] = time_df.iloc[trough_idx]["date_label"] if "date_label" in time_df.columns else None
        
        # Calculate recovery time if applicable
        if trough_idx < len(portfolio_values) - 1:
            recovered = False
            for i in range(trough_idx + 1, len(portfolio_values)):
                if portfolio_values[i] >= peak:
                    recovered = True
                    recovery_days = (time_df.iloc[i]["date"] - time_df.iloc[trough_idx]["date"]).days
                    drawdown_data["recovery_time"] = recovery_days
                    break
        
        metrics["max_drawdown"] = max_drawdown
        metrics["drawdown_data"] = drawdown_data
        
        # Value at Risk (VaR) - simplified calculation
        if len(returns) >= 10:  # Need reasonable sample size
            # Convert returns to numpy array for percentile calculation
            returns_array = np.array(returns) * 100  # Convert to percentage
            
            # Calculate Value at Risk (95% confidence)
            var_95 = np.percentile(returns_array, 5)  # 5th percentile for 95% confidence
            metrics["value_at_risk_95"] = -var_95  # Convert to positive number representing loss
            
            # Calculate Conditional VaR (Expected Shortfall)
            cvar_95 = np.mean(returns_array[returns_array <= var_95])
            metrics["conditional_var_95"] = -cvar_95  # Convert to positive number
    
    return metrics


def generate_terminal_charts(results):
    """
    Generate a set of terminal-based charts for portfolio analysis.
    
    Args:
        results: Analysis results dictionary
    
    Returns:
        String containing terminal charts
    """
    output = []
    
    # Setup
    chart_width = 80  # Fixed reasonable width
    
    # Get data
    current_df = results['current_analysis']['dataframe']
    
    # Header for all charts
    output.append("\n" + "╔" + "═" * (chart_width-2) + "╗")
    output.append("║" + "PORTFOLIO VISUALIZATION".center(chart_width-2) + "║")
    output.append("╠" + "═" * (chart_width-2) + "╣")
    
    if 'time_analysis' in results and 'error' not in results['time_analysis']:
        time_df = results['time_analysis']['dataframe']
        
        # Calculate time-based metrics
        time_metrics = calculate_time_metrics(time_df)
        
        # 1. Portfolio Value Over Time Sparkline
        output.append("║" + "PORTFOLIO VALUE TREND".center(chart_width-2) + "║")
        output.append("║" + " " * (chart_width-2) + "║")
        
        # Generate sparkline of portfolio values
        values = time_df['portfolio_value'].tolist()
        sparkline = generate_terminal_sparkline(values, width=chart_width-8, label_format="Currency")
        
        # Format the sparkline to fit in the box
        for line in sparkline.split('\n'):
            output.append("║  " + line.ljust(chart_width-4) + "║")
        
        output.append("║" + " " * (chart_width-2) + "║")
        
        # Add time-based performance metrics if available
        if time_metrics:
            output.append("║" + "PERFORMANCE METRICS".center(chart_width-2) + "║")
            output.append("║" + " " * (chart_width-2) + "║")
            
            # Format period information
            first_date = time_df.iloc[0]["date_label"]
            last_date = time_df.iloc[-1]["date_label"]
            period_days = time_metrics.get("period_days", 0)
            
            period_info = f"Period: {first_date} to {last_date} ({period_days} days)"
            output.append("║  " + period_info.ljust(chart_width-4) + "║")
            
            # Format return metrics
            value_change = time_metrics.get("value_change", 0)
            value_change_pct = time_metrics.get("value_change_pct", 0)
            
            change_color = get_color_for_value(value_change)
            change_str = f"Total Return: {change_color}{format_currency(value_change)} ({format_percent(value_change_pct)}){reset_color()}"
            output.append("║  " + change_str.ljust(chart_width-4) + "║")
            
            # Win Rate
            if "win_rate" in time_metrics:
                win_rate = time_metrics["win_rate"]
                win_rate_color = get_color_for_value(win_rate - 50)  # Color based on better than 50%
                win_rate_str = f"Win Rate: {win_rate_color}{format_percent(win_rate)}{reset_color()}"
                output.append("║  " + win_rate_str.ljust(chart_width-4) + "║")
            
            # Display annualized metrics if available
            if "annualized_return" in time_metrics and period_days >= 7:
                ann_return = time_metrics["annualized_return"]
                ann_return_color = get_color_for_value(ann_return)
                
                ann_return_str = f"Annualized Return: {ann_return_color}{format_percent(ann_return)}{reset_color()}"
                output.append("║  " + ann_return_str.ljust(chart_width-4) + "║")
                
            # Display CAGR if available for longer periods
            if "cagr" in time_metrics and period_days >= 30:
                cagr = time_metrics["cagr"]
                cagr_color = get_color_for_value(cagr)
                cagr_str = f"CAGR: {cagr_color}{format_percent(cagr)}{reset_color()}"
                output.append("║  " + cagr_str.ljust(chart_width-4) + "║")
                
            # Risk metrics header - display if we have any risk metrics
            has_risk_metrics = any(metric in time_metrics for metric in 
                                ["volatility", "max_drawdown", "value_at_risk_95"])
            if has_risk_metrics:
                output.append("║" + " " * (chart_width-2) + "║")
                output.append("║" + "RISK METRICS".center(chart_width-2) + "║")
                
                # Display volatility metrics if available
                if "volatility" in time_metrics:
                    volatility = time_metrics["volatility"]
                    vol_str = f"Volatility: {format_percent(volatility)}"
                    output.append("║  " + vol_str.ljust(chart_width-4) + "║")
                
                # Display max drawdown if available
                if "max_drawdown" in time_metrics:
                    drawdown = time_metrics["max_drawdown"]
                    drawdown_color = get_color_for_value(-drawdown)  # Negative color for drawdown
                    drawdown_str = f"Max Drawdown: {drawdown_color}{format_percent(drawdown)}{reset_color()}"
                    output.append("║  " + drawdown_str.ljust(chart_width-4) + "║")
                
                # Value at Risk metrics
                if "value_at_risk_95" in time_metrics:
                    var_95 = time_metrics["value_at_risk_95"]
                    var_color = get_color_for_value(-var_95)
                    var_str = f"Value at Risk (95%): {var_color}{format_percent(var_95)}{reset_color()}"
                    output.append("║  " + var_str.ljust(chart_width-4) + "║")
            
            # Risk-adjusted metrics header - display if we have any risk-adjusted metrics
            has_risk_adj_metrics = any(metric in time_metrics for metric in 
                                    ["sharpe_ratio", "sortino_ratio", "calmar_ratio"])
            if has_risk_adj_metrics:
                output.append("║" + " " * (chart_width-2) + "║")
                output.append("║" + "RISK-ADJUSTED METRICS".center(chart_width-2) + "║")
                
                # Sharpe ratio
                if "sharpe_ratio" in time_metrics:
                    sharpe = time_metrics["sharpe_ratio"]
                    sharpe_color = get_color_for_value(sharpe)
                    sharpe_str = f"Sharpe Ratio: {sharpe_color}{sharpe:.2f}{reset_color()}"
                    output.append("║  " + sharpe_str.ljust(chart_width-4) + "║")
                
                # Sortino ratio
                if "sortino_ratio" in time_metrics:
                    sortino = time_metrics["sortino_ratio"]
                    sortino_color = get_color_for_value(sortino)
                    sortino_str = f"Sortino Ratio: {sortino_color}{sortino:.2f}{reset_color()}"
                    output.append("║  " + sortino_str.ljust(chart_width-4) + "║")
                
                # Calmar ratio
                if "calmar_ratio" in time_metrics:
                    calmar = time_metrics["calmar_ratio"]
                    calmar_color = get_color_for_value(calmar)
                    calmar_str = f"Calmar Ratio: {calmar_color}{calmar:.2f}{reset_color()}"
                    output.append("║  " + calmar_str.ljust(chart_width-4) + "║")
        
        output.append("║" + " " * (chart_width-2) + "║")
        output.append("╠" + "─" * (chart_width-2) + "╣")
        
        # 2. Returns Bar Chart
        output.append("║" + "PORTFOLIO RETURNS BY DATE".center(chart_width-2) + "║")
        output.append("║" + " " * (chart_width-2) + "║")
        
        returns = time_df['portfolio_return'].tolist()
        dates = time_df['date_label'].tolist()
        colors = [get_color_for_value(val) for val in returns]
        
        bar_chart = generate_terminal_bar_chart(
            dates, returns, "", 
            width=chart_width-8, height=6, 
            colors=colors,
            max_bar_width=40
        )
        
        # Format the bar chart to fit in the box
        for line in bar_chart.split('\n'):
            output.append("║  " + line.ljust(chart_width-4) + "║")
        
        output.append("║" + " " * (chart_width-2) + "║")
        output.append("╠" + "─" * (chart_width-2) + "╣")
    
    # 3. Stock Allocation Chart
    if not current_df.empty:
        output.append("║" + "PORTFOLIO ALLOCATION".center(chart_width-2) + "║")
        output.append("║" + " " * (chart_width-2) + "║")
        
        symbols = current_df['symbol'].tolist()
        allocations = current_df['percentage'].tolist()
        
        # Generate a proportional chart
        allocation_chart = generate_terminal_bar_chart(
            symbols, allocations, "",
            width=chart_width-8, height=5,
            max_bar_width=40
        )
        
        # Format the chart
        for line in allocation_chart.split('\n'):
            output.append("║  " + line.ljust(chart_width-4) + "║")
        
        output.append("║" + " " * (chart_width-2) + "║")
        output.append("╠" + "─" * (chart_width-2) + "╣")
    
    # 4. Stock Performance Comparison
    if not current_df.empty:
        output.append("║" + "STOCK PERFORMANCE COMPARISON".center(chart_width-2) + "║")
        output.append("║" + " " * (chart_width-2) + "║")
        
        symbols = current_df['symbol'].tolist()
        returns_pct = current_df['gain_loss_percent'].tolist()
        colors = [get_color_for_value(val) for val in returns_pct]
        
        perf_chart = generate_terminal_bar_chart(
            symbols, returns_pct, "",
            width=chart_width-8, height=6,
            colors=colors,
            max_bar_width=40
        )
        
        # Format the chart
        for line in perf_chart.split('\n'):
            output.append("║  " + line.ljust(chart_width-4) + "║")
        
        output.append("║" + " " * (chart_width-2) + "║")
        
        # Return summary 
        if 'total_gain_loss' in results['current_analysis']:
            gain_loss = results['current_analysis']['total_gain_loss']
            gain_loss_pct = results['current_analysis']['total_gain_loss_percent']
            color = get_color_for_value(gain_loss)
            
            summary = f"Total Return: {color}{format_currency(gain_loss)} ({format_percent(gain_loss_pct)}){reset_color()}"
            output.append("║  " + summary.ljust(chart_width-4) + "║")
            output.append("║" + " " * (chart_width-2) + "║")
    
    # Footer
    output.append("╚" + "═" * (chart_width-2) + "╝")
    
    return "\n".join(output)


if __name__ == "__main__":
    sys.exit(main())