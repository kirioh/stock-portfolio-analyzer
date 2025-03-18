import pandas as pd
import datetime
from typing import List, Dict, Optional, Tuple
from .logger import get_logger

logger = get_logger(__name__)

def parse_statement_transactions(statement_text: str) -> pd.DataFrame:
    """
    Parse transaction data from a Revolut statement text.
    
    Returns a DataFrame with columns:
    - date: datetime
    - symbol: str
    - type: str (Trade or Cash top-up)
    - quantity: float
    - price: float
    - side: str (Buy or Sell)
    - value: float
    """
    logger.info("Parsing transaction data from statement")
    
    lines = statement_text.split('\n')
    
    # Find the transactions section
    transaction_start = -1
    for i, line in enumerate(lines):
        if "Transactions" in line and i < len(lines) - 1 and "Date" in lines[i+1]:
            transaction_start = i + 2  # Skip the header line
            break
    
    if transaction_start == -1:
        logger.error("Could not find transactions section in statement")
        raise ValueError("Could not find transactions section in statement")
    
    # Parse transactions
    transactions = []
    for i in range(transaction_start, len(lines)):
        line = lines[i].strip()
        if not line or "Report lost or stolen card" in line:
            break
        
        # Parse the transaction line
        parts = line.split()
        if len(parts) < 7:
            continue
        
        try:
            # Extract date (format: DD MMM YYYY HH:MM:SS AEDT)
            date_str = " ".join(parts[:5])
            # Convert to datetime - handle timezone manually since strptime can't reliably parse timezone abbreviations
            try:
                # Try parsing with timezone first
                date = datetime.datetime.strptime(date_str, "%d %b %Y %H:%M:%S %Z")
            except ValueError:
                # If that fails, parse without timezone and handle it separately
                base_date_str = " ".join(parts[:4])
                date = datetime.datetime.strptime(base_date_str, "%d %b %Y %H:%M:%S")
            
            # Extract other fields
            symbol = parts[5] if parts[5] != "Cash" else ""
            txn_type = "Trade" if "Trade" in " ".join(parts[6:9]) else "Cash top-up"
            
            # Default values
            quantity = 0.0
            price = 0.0
            side = ""
            value = 0.0
            
            if txn_type == "Cash top-up":
                value_index = parts.index("US$") if "US$" in parts else -1
                if value_index >= 0 and value_index + 1 < len(parts):
                    value = float(parts[value_index + 1])
            else:  # Trade
                # Find quantity, price, side and value
                for j, part in enumerate(parts):
                    if part.replace(".", "").isdigit() and j + 1 < len(parts) and j > 6:
                        quantity = float(part)
                        price_idx = j + 1
                        if price_idx < len(parts) and parts[price_idx].startswith("US$"):
                            price = float(parts[price_idx].replace("US$", ""))
                        break
                
                side_index = -1
                for j, part in enumerate(parts):
                    if part in ["Buy", "Sell"]:
                        side = part
                        side_index = j
                        break
                
                if side_index >= 0 and side_index + 1 < len(parts) and parts[side_index + 1].startswith("US$"):
                    value = float(parts[side_index + 1].replace("US$", ""))
            
            transactions.append({
                "date": date,
                "symbol": symbol,
                "type": txn_type,
                "quantity": quantity,
                "price": price,
                "side": side,
                "value": value
            })
            
        except Exception as e:
            logger.warning(f"Error parsing transaction line: {line}. Error: {e}")
            continue
    
    df = pd.DataFrame(transactions)
    if not df.empty:
        logger.info(f"Successfully parsed {len(df)} transactions")
    else:
        logger.warning("No transactions found in the statement")
    
    return df


def get_transaction_dates(transactions_df: pd.DataFrame) -> List[datetime.datetime]:
    """
    Extract unique transaction dates from the transactions DataFrame.
    Only returns dates where actual trades occurred (not cash top-ups).
    """
    if transactions_df.empty:
        return []
    
    # Filter for trade transactions only
    trade_df = transactions_df[transactions_df['type'] == 'Trade']
    if trade_df.empty:
        return []
    
    # Get unique dates and sort
    unique_dates = pd.to_datetime(trade_df['date']).dt.date.unique()
    sorted_dates = sorted(unique_dates)
    
    # Convert to datetime objects at start of day
    return [datetime.datetime.combine(date, datetime.time.min) for date in sorted_dates]


def calculate_portfolio_state_at_date(transactions_df: pd.DataFrame, 
                                     target_date: datetime.datetime) -> Dict[str, Dict]:
    """
    Calculate the portfolio state at a specific date based on transaction history.
    
    Returns a dictionary with stock symbols as keys and dictionaries with:
    - shares: number of shares owned
    - cost_basis: total cost basis
    - avg_price: average cost per share
    """
    # Filter transactions up to the target date
    filtered_df = transactions_df[transactions_df['date'] <= target_date]
    
    # Only include actual trades
    trades_df = filtered_df[filtered_df['type'] == 'Trade']
    
    portfolio = {}
    
    for _, trade in trades_df.iterrows():
        symbol = trade['symbol']
        quantity = trade['quantity']
        price = trade['price']
        side = trade['side']
        value = trade['value']
        
        if symbol not in portfolio:
            portfolio[symbol] = {
                'shares': 0.0,
                'cost_basis': 0.0,
                'avg_price': 0.0
            }
        
        if side == 'Buy':
            # Update shares and cost basis
            current_shares = portfolio[symbol]['shares']
            current_cost = portfolio[symbol]['cost_basis']
            
            portfolio[symbol]['shares'] += quantity
            portfolio[symbol]['cost_basis'] += value
            
            # Recalculate average price
            if portfolio[symbol]['shares'] > 0:
                portfolio[symbol]['avg_price'] = portfolio[symbol]['cost_basis'] / portfolio[symbol]['shares']
        elif side == 'Sell':
            # Handle sells similarly but subtract
            portfolio[symbol]['shares'] -= quantity
            
            # For simplicity, we're using FIFO to calculate new cost basis
            # A more sophisticated approach would be to use specific tax lot accounting
            if portfolio[symbol]['shares'] > 0:
                sold_ratio = quantity / (portfolio[symbol]['shares'] + quantity)
                sold_cost = portfolio[symbol]['cost_basis'] * sold_ratio
                portfolio[symbol]['cost_basis'] -= sold_cost
                portfolio[symbol]['avg_price'] = portfolio[symbol]['cost_basis'] / portfolio[symbol]['shares']
            elif portfolio[symbol]['shares'] == 0:
                portfolio[symbol]['cost_basis'] = 0
                portfolio[symbol]['avg_price'] = 0
    
    return portfolio
