import pytest
import pandas as pd
import os
import sys
from unittest.mock import patch, MagicMock

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Create sample portfolio data for testing
@pytest.fixture
def sample_portfolio_csv(tmp_path):
    """Create a temporary CSV file with sample portfolio data for testing."""
    csv_path = tmp_path / "test_portfolio.csv"
    csv_content = "symbol,shares,average_cost\nAAPL,10,120.5\nMSFT,5,220.75\nGOOG,2,1500.25"
    csv_path.write_text(csv_content)
    return str(csv_path)

# Mock config for testing
@pytest.fixture
def mock_config():
    return {
        "data_source": "yfinance",
        "retry_attempts": 1
    }

def test_analyze_portfolio_basic(sample_portfolio_csv, mock_config):
    """Test the basic functionality of analyze_portfolio with mocked price data."""
    # We need to patch directly in the portfolio module to intercept the calls
    with patch('stock_portfolio_analyzer.portfolio.fetch_current_price') as mock_fetch:
        # Configure the mock to return different prices for different symbols
        def side_effect(symbol, retries):
            prices = {
                'AAPL': 150.0,   # Up from 120.5
                'MSFT': 200.0,   # Down from 220.75
                'GOOG': 2000.0   # Up from 1500.25
            }
            return prices.get(symbol, 0.0)
        
        mock_fetch.side_effect = side_effect
        
        # Now import and run the analysis
        from stock_portfolio_analyzer.portfolio import analyze_portfolio
        results = analyze_portfolio(sample_portfolio_csv, mock_config)
        
        # Verify the mock was called for each symbol
        assert mock_fetch.call_count == 3
        
        # Check dataframe exists in results
        assert 'dataframe' in results
        df = results['dataframe']
        
        # Check total calculations
        expected_value = 10 * 150.0 + 5 * 200.0 + 2 * 2000.0
        expected_cost = 10 * 120.5 + 5 * 220.75 + 2 * 1500.25
        assert results['total_value'] == pytest.approx(expected_value)
        assert results['total_cost_basis'] == pytest.approx(expected_cost)
        
        # Check individual rows
        apple_row = df[df['symbol'] == 'AAPL'].iloc[0]
        assert apple_row['current_price'] == 150.0
        assert apple_row['value'] == pytest.approx(10 * 150.0)
        assert apple_row['gain_loss'] == pytest.approx(10 * (150.0 - 120.5))
        
        # Check gain/loss calculations
        msft_row = df[df['symbol'] == 'MSFT'].iloc[0]
        assert msft_row['current_price'] == 200.0
        # MSFT price went down from 220.75 to 200.0, so gain should be negative
        assert msft_row['gain_loss'] < 0
        assert msft_row['gain_loss'] == pytest.approx(5 * (200.0 - 220.75))

def test_analyze_portfolio_missing_columns(tmp_path, mock_config):
    """Test that analysis fails properly with missing columns."""
    # Create a CSV with missing required columns
    invalid_csv = tmp_path / "invalid.csv"
    invalid_csv.write_text("symbol,quantity\nAAPL,10\nMSFT,5")
    
    # Import the function
    from stock_portfolio_analyzer.portfolio import analyze_portfolio
    
    # Check that it raises ValueError
    with pytest.raises(ValueError, match="CSV file must contain columns"):
        analyze_portfolio(str(invalid_csv), mock_config)

def test_analyze_portfolio_api_error(sample_portfolio_csv, mock_config):
    """Test handling of API errors during price fetching."""
    # Patch the function in the portfolio module
    with patch('stock_portfolio_analyzer.portfolio.fetch_current_price') as mock_fetch:
        # Make the first call work but the second throw an exception
        def side_effect(symbol, retries):
            if symbol == 'AAPL':
                return 150.0
            elif symbol == 'MSFT':
                raise ConnectionError("API connection failed")
            else:
                return 2000.0
        
        mock_fetch.side_effect = side_effect
        
        # Import and run the analysis
        from stock_portfolio_analyzer.portfolio import analyze_portfolio
        results = analyze_portfolio(sample_portfolio_csv, mock_config)
        df = results['dataframe']
        
        # Check that AAPL was processed correctly
        apple_row = df[df['symbol'] == 'AAPL'].iloc[0]
        assert apple_row['current_price'] == 150.0
        
        # Check that MSFT has zero values due to the error
        msft_row = df[df['symbol'] == 'MSFT'].iloc[0]
        assert msft_row['current_price'] == 0.0
        
        # GOOG should be processed after the error
        goog_row = df[df['symbol'] == 'GOOG'].iloc[0]
        assert goog_row['current_price'] == 2000.0
