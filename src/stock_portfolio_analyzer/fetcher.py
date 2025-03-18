import yfinance as yf
import time
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
