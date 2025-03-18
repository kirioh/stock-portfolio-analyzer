import yfinance as yf
import pandas as pd
import time
import datetime
from typing import Dict, List, Optional, Tuple, Union
from .logger import get_logger

logger = get_logger(__name__)

def fetch_current_price(symbol: str, retries: int = 1) -> float:
    """
    Fetch the latest price for a given stock symbol using yfinance,
    retrying up to 'retries' times if necessary.
    """
    attempt = 0
    while attempt < retries:
        try:
            logger.debug(f"Fetching price for {symbol}, attempt {attempt+1}/{retries}")
            ticker_data = yf.Ticker(symbol)
            hist = ticker_data.history(period="1d")
            if hist.empty:
                raise ValueError(f"No data returned for symbol {symbol}")
            price = hist["Close"].iloc[-1]
            logger.debug(f"Successfully fetched price for {symbol}: ${price:.2f}")
            return price
        except Exception as e:
            attempt += 1
            logger.warning(f"Attempt {attempt} - Error fetching data for {symbol}: {e}")
            if attempt < retries:
                time.sleep(1)  # basic backoff
    
    # If we get here, all retries failed
    logger.error(f"Failed to fetch price for {symbol} after {retries} retries.")
    raise ConnectionError(f"Failed to fetch price for {symbol} after {retries} retries.")


def fetch_historical_prices(symbols: List[str], start_date: datetime.datetime, 
                          end_date: Optional[datetime.datetime] = None, 
                          retries: int = 1) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical price data for multiple symbols over a date range.
    
    Args:
        symbols: List of stock symbols to fetch
        start_date: Start date for historical data
        end_date: End date (defaults to today if None)
        retries: Number of retry attempts
        
    Returns:
        Dictionary mapping symbols to DataFrames with historical price data
    """
    if not symbols:
        logger.warning("No symbols provided to fetch_historical_prices")
        return {}
    
    # Default end date to today if not specified
    if end_date is None:
        end_date = datetime.datetime.now()
    
    logger.info(f"Fetching historical prices for {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")
    
    results = {}
    for symbol in symbols:
        attempt = 0
        while attempt < retries:
            try:
                logger.debug(f"Fetching historical data for {symbol}, attempt {attempt+1}/{retries}")
                ticker_data = yf.Ticker(symbol)
                hist = ticker_data.history(start=start_date, end=end_date)
                
                if hist.empty:
                    logger.warning(f"No historical data returned for {symbol}")
                    break
                
                # Keep only essential price columns
                if not all(col in hist.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                    logger.warning(f"Missing expected columns in data for {symbol}")
                    break
                
                results[symbol] = hist
                logger.debug(f"Successfully fetched {len(hist)} days of price data for {symbol}")
                break
                
            except Exception as e:
                attempt += 1
                logger.warning(f"Attempt {attempt} - Error fetching historical data for {symbol}: {e}")
                if attempt < retries:
                    time.sleep(1)  # basic backoff
        
        if symbol not in results:
            logger.error(f"Failed to fetch historical data for {symbol} after {retries} retries")
    
    return results


def fetch_prices_for_dates(symbols: List[str], dates: List[datetime.datetime], 
                           retries: int = 1) -> Dict[str, Dict[datetime.datetime, float]]:
    """
    Fetch closing prices for specific dates for multiple symbols.
    
    Args:
        symbols: List of stock symbols
        dates: List of datetime objects representing the dates to fetch prices for
        retries: Number of retry attempts
        
    Returns:
        Nested dictionary {symbol: {date: price}}
    """
    if not symbols or not dates:
        logger.warning("No symbols or dates provided to fetch_prices_for_dates")
        return {}
    
    # Extend date range by 5 days to handle market closures
    min_date = min(dates) - datetime.timedelta(days=5)
    max_date = max(dates) + datetime.timedelta(days=5)
    
    logger.info(f"Fetching prices for {len(symbols)} symbols on {len(dates)} specific dates")
    
    # Get continuous price history for the entire range
    historical_data = fetch_historical_prices(symbols, min_date, max_date, retries)
    
    # Extract prices for target dates
    results = {}
    for symbol, hist_df in historical_data.items():
        results[symbol] = {}
        
        # For each target date, find the closest available price date
        for target_date in dates:
            # Convert target date to datetime64 for comparison with DataFrame index
            target_datetime64 = pd.Timestamp(target_date)
            
            # Find the closest date in the historical data (on or before the target date)
            # This handles weekends and holidays when markets are closed
            available_dates = hist_df.index[hist_df.index <= target_datetime64]
            
            if len(available_dates) > 0:
                closest_date = available_dates[-1]  # Most recent date on or before target
                price = hist_df.loc[closest_date, 'Close']
                
                # Store the result using the original target_date as the key
                results[symbol][target_date] = price
                logger.debug(f"Found price for {symbol} on {target_date.date()}: ${price:.2f} "
                           f"(using market data from {closest_date.date()})")
            else:
                logger.warning(f"No data available for {symbol} on or before {target_date.date()}")
                # Set a None value to indicate missing data
                results[symbol][target_date] = None
    
    return results
