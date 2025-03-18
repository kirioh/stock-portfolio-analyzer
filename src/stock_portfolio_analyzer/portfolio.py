import pandas as pd
from .fetcher import fetch_current_price
from .logger import get_logger

logger = get_logger(__name__)

def analyze_portfolio(csv_file: str, config: dict) -> dict:
    """
    Reads a CSV portfolio file, fetches current prices, and returns
    a dictionary of analysis results.
    """
    logger.info(f"Analyzing portfolio from {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        logger.error(f"Failed to read CSV file {csv_file}: {e}")
        raise ValueError(f"Could not read CSV file: {e}")

    # Validate CSV structure
    required_cols = {"symbol", "shares", "average_cost"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        logger.error(f"CSV file is missing required columns: {', '.join(missing)}")
        raise ValueError(f"CSV file must contain columns: {', '.join(required_cols)}")

    # Initialize new columns
    df["current_price"] = 0.0
    df["value"] = 0.0
    df["gain_loss"] = 0.0
    df["gain_loss_percent"] = 0.0

    retries = config.get("retry_attempts", 1)
    logger.info(f"Using {retries} retry attempts for API calls")

    # Process each stock
    for i, row in df.iterrows():
        symbol = row["symbol"]
        shares = float(row["shares"])
        avg_cost = float(row["average_cost"])

        try:
            # Fetch current price
            current_price = fetch_current_price(symbol, retries)
            df.at[i, "current_price"] = current_price
            
            # Calculate position metrics
            position_value = current_price * shares
            df.at[i, "value"] = position_value
            
            cost_basis = avg_cost * shares
            gain_loss_value = position_value - cost_basis
            df.at[i, "gain_loss"] = gain_loss_value
            
            gain_loss_percent = (gain_loss_value / cost_basis) * 100 if cost_basis != 0 else 0
            df.at[i, "gain_loss_percent"] = gain_loss_percent
            
            logger.info(f"Processed {symbol}: ${current_price:.2f}, value: ${position_value:.2f}, "
                      f"gain/loss: ${gain_loss_value:.2f} ({gain_loss_percent:.2f}%)")
            
        except Exception as e:
            logger.error(f"Failed to update row {i} for symbol {symbol}: {e}")
            # Keep zeros in the row, but note the error

    # Calculate portfolio totals
    total_value = df["value"].sum()
    total_cost_basis = (df["shares"] * df["average_cost"]).sum()
    total_gain_loss = total_value - total_cost_basis
    total_gain_loss_percent = (total_gain_loss / total_cost_basis * 100) if total_cost_basis != 0 else 0

    logger.info(f"Portfolio analysis complete. Total value: ${total_value:.2f}, "
               f"Total gain/loss: ${total_gain_loss:.2f} ({total_gain_loss_percent:.2f}%)")

    return {
        "dataframe": df,
        "total_value": total_value,
        "total_cost_basis": total_cost_basis,
        "total_gain_loss": total_gain_loss,
        "total_gain_loss_percent": total_gain_loss_percent,
    }
