import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from typing import Dict, List, Optional, Tuple, Union
from .logger import get_logger
from .fetcher import fetch_prices_for_dates
from .transactions import get_transaction_dates, calculate_portfolio_state_at_date
from .utils import format_currency, format_percent

logger = get_logger(__name__)

def analyze_portfolio_over_time(transactions_df: pd.DataFrame, config: dict) -> dict:
    """
    Analyze portfolio performance over time based on transaction history.
    
    Args:
        transactions_df: DataFrame containing transaction data
        config: Configuration dictionary
    
    Returns:
        Dictionary with time-based analysis results
    """
    logger.info("Performing time-based portfolio analysis")
    
    # Get unique transaction dates (when trades occurred)
    transaction_dates = get_transaction_dates(transactions_df)
    if not transaction_dates:
        logger.warning("No transaction dates found for time-based analysis")
        return {"error": "No transaction dates found"}
    
    # Add today's date to the analysis points
    analysis_dates = transaction_dates + [datetime.datetime.now()]
    
    # Get symbols from transaction data
    symbols = transactions_df[transactions_df['type'] == 'Trade']['symbol'].unique().tolist()
    if not symbols:
        logger.warning("No symbols found in transaction data")
        return {"error": "No symbols found in transaction data"}
    
    logger.info(f"Analyzing portfolio of {len(symbols)} symbols across {len(analysis_dates)} dates")
    
    # Fetch historical prices for all required dates
    retries = config.get("retry_attempts", 3)
    historical_prices = fetch_prices_for_dates(symbols, analysis_dates, retries)
    
    # Initialize result data structures
    portfolio_values = []
    portfolio_costs = []
    portfolio_returns = []
    portfolio_returns_pct = []
    date_labels = []
    
    # Calculate portfolio state at each analysis date
    for date in analysis_dates:
        portfolio_state = calculate_portfolio_state_at_date(transactions_df, date)
        
        # Calculate total portfolio value and cost at this date
        total_value = 0.0
        total_cost = 0.0
        
        for symbol, state in portfolio_state.items():
            shares = state['shares']
            cost_basis = state['cost_basis']
            
            # Get price for this symbol at this date
            if symbol in historical_prices and date in historical_prices[symbol]:
                price = historical_prices[symbol][date]
                if price is not None:
                    position_value = price * shares
                    total_value += position_value
            
            total_cost += cost_basis
        
        # Calculate returns
        gain_loss = total_value - total_cost
        gain_loss_pct = (gain_loss / total_cost * 100) if total_cost > 0 else 0.0
        
        # Store results
        portfolio_values.append(total_value)
        portfolio_costs.append(total_cost)
        portfolio_returns.append(gain_loss)
        portfolio_returns_pct.append(gain_loss_pct)
        date_labels.append(date.strftime('%Y-%m-%d'))
        
        logger.debug(f"Date: {date.date()}, Value: ${total_value:.2f}, "
                   f"Cost: ${total_cost:.2f}, Return: ${gain_loss:.2f} ({gain_loss_pct:.2f}%)")
    
    # Create a DataFrame for easier analysis
    results_df = pd.DataFrame({
        'date': analysis_dates,
        'date_label': date_labels,
        'portfolio_value': portfolio_values,
        'portfolio_cost': portfolio_costs,
        'portfolio_return': portfolio_returns,
        'portfolio_return_pct': portfolio_returns_pct
    })
    
    logger.info("Time-based portfolio analysis complete")
    
    return {
        'dataframe': results_df,
        'dates': analysis_dates,
        'values': portfolio_values,
        'costs': portfolio_costs,
        'returns': portfolio_returns,
        'returns_pct': portfolio_returns_pct,
        'date_labels': date_labels
    }


def generate_time_plots(time_analysis_results: dict, output_dir: Optional[str] = None, 
                       show_plots: bool = False) -> Dict[str, str]:
    """
    Generate time-based analysis plots.
    
    Args:
        time_analysis_results: Results from analyze_portfolio_over_time
        output_dir: Directory to save plot files (if None, won't save files)
        show_plots: Whether to display plots interactively
        
    Returns:
        Dictionary with file paths to saved plots
    """
    if 'error' in time_analysis_results:
        logger.error(f"Cannot generate plots: {time_analysis_results['error']}")
        return {'error': time_analysis_results['error']}
    
    if 'dataframe' not in time_analysis_results:
        logger.error("Invalid time analysis results - missing dataframe")
        return {'error': "Invalid time analysis results"}
    
    df = time_analysis_results['dataframe']
    
    # Set aesthetic parameters
    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 12})
    
    saved_plots = {}
    
    # 1. Portfolio Value vs. Cost Over Time
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['date'], df['portfolio_value'], marker='o', linewidth=2, label='Portfolio Value')
        ax.plot(df['date'], df['portfolio_cost'], marker='s', linewidth=2, label='Cost Basis')
        
        # Format
        ax.set_title('Portfolio Value vs. Cost Basis Over Time', fontsize=16)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('USD ($)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis date labels
        plt.xticks(df['date'], df['date_label'], rotation=45)
        plt.tight_layout()
        
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, 'portfolio_value_time.png')
            plt.savefig(filename)
            saved_plots['value_time'] = filename
        
        if show_plots:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        logger.error(f"Error generating portfolio value plot: {e}")
    
    # 2. Portfolio Returns (Absolute) Over Time
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Bar colors based on positive/negative
        colors = ['green' if x >= 0 else 'red' for x in df['portfolio_return']]
        ax.bar(df['date_label'], df['portfolio_return'], color=colors)
        
        # Format
        ax.set_title('Portfolio Returns Over Time', fontsize=16)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Return (USD $)', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Format x-axis date labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if output_dir:
            import os
            filename = os.path.join(output_dir, 'portfolio_returns_time.png')
            plt.savefig(filename)
            saved_plots['returns_time'] = filename
        
        if show_plots:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        logger.error(f"Error generating portfolio returns plot: {e}")
    
    # 3. Portfolio Returns (Percentage) Over Time
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Bar colors based on positive/negative
        colors = ['green' if x >= 0 else 'red' for x in df['portfolio_return_pct']]
        ax.bar(df['date_label'], df['portfolio_return_pct'], color=colors)
        
        # Format
        ax.set_title('Portfolio Percentage Returns Over Time', fontsize=16)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Return (%)', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Format x-axis date labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if output_dir:
            import os
            filename = os.path.join(output_dir, 'portfolio_returns_pct_time.png')
            plt.savefig(filename)
            saved_plots['returns_pct_time'] = filename
        
        if show_plots:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        logger.error(f"Error generating portfolio percentage returns plot: {e}")
    
    logger.info(f"Generated {len(saved_plots)} time-based analysis plots")
    return saved_plots