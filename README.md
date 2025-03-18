# Stock Portfolio Analyzer

A modular, production-ready Python tool for analyzing a stock portfolio from a CSV file.

## Features

- Fetch current stock prices using the yfinance API
- Calculate portfolio value, gain/loss, and performance metrics
- Configurable via YAML file or environment variables
- Comprehensive logging with configurable levels
- Colorized console output
- Robust error handling
- Comprehensive test suite
- Both CLI and programmatic usage

## Installation

```bash
# Install in development mode
pip install -e .

# Or install directly from the directory
pip install .
```

## CLI Usage

```bash
# Basic usage with default config
portfolio-analyzer --csv-file /path/to/portfolio.csv

# With a custom config file
portfolio-analyzer --csv-file /path/to/portfolio.csv --config-file /path/to/config.yaml

# Enable debug logging
portfolio-analyzer --csv-file /path/to/portfolio.csv --debug

# Disable colored output
portfolio-analyzer --csv-file /path/to/portfolio.csv --no-color
```

## Programmatic Usage

You can also use the library programmatically in your own Python scripts:

```python
from stock_portfolio_analyzer.main import run_analysis

# Run analysis with default config
results = run_analysis('/path/to/portfolio.csv')

# Access the results
df = results['dataframe']
total_value = results['total_value']
total_gain_loss = results['total_gain_loss']
total_gain_loss_pct = results['total_gain_loss_percent']

# Process the results as needed
print(f"Portfolio value: ${total_value:,.2f}")
print(f"Portfolio gain/loss: ${total_gain_loss:,.2f} ({total_gain_loss_pct:.2f}%)")

# You can also analyze the individual positions
for _, row in df.iterrows():
    print(f"{row['symbol']}: ${row['value']:,.2f}, {row['gain_loss_percent']:.2f}%")
```

See the `example.py` file for a more comprehensive example of programmatic usage.

## CSV Format

The CSV file should have the following columns:
- **symbol**: The stock ticker symbol
- **shares**: Number of shares held
- **average_cost**: Average cost per share

Example:
```
symbol,shares,average_cost
AAPL,10,120.5
MSFT,5,220.75
GOOG,2,1500.25
AMZN,3,140.80
```

## Configuration

Configuration can be provided via a YAML file or environment variables:

```yaml
# config.yaml
data_source: "yfinance"    # Data source for stock prices
csv_file_path: "./portfolio.csv"  # Default CSV file path
retry_attempts: 3  # Number of retries for API calls
```

Environment variables (take precedence over config file):
- `DATA_SOURCE`: Data source for fetching stock prices
- `CSV_FILE_PATH`: Default path to the portfolio CSV file
- `RETRY_ATTEMPTS`: Number of retry attempts for API calls
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `LOG_FILE`: Path to the log file

## Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage report
python -m pytest --cov=src/stock_portfolio_analyzer

# Run specific test
python -m pytest tests/test_portfolio.py::test_analyze_portfolio_basic
```

## Project Structure

```
stock_portfolio_analyzer/
├── pyproject.toml          # Package configuration
├── requirements.txt        # Dependencies
├── README.md               # This file
├── config.yaml             # Default configuration
├── portfolio.csv           # Example portfolio data
├── example.py              # Example script for programmatic usage
├── src/
│   └── stock_portfolio_analyzer/
│       ├── __init__.py     # Package initialization
│       ├── cli.py          # Command-line interface
│       ├── main.py         # Main programmatic entry point
│       ├── config.py       # Configuration management
│       ├── logger.py       # Logging configuration
│       ├── fetcher.py      # Stock price fetching logic
│       ├── portfolio.py    # Portfolio analysis logic
│       └── utils.py        # Helper utilities
└── tests/
    ├── __init__.py
    └── test_portfolio.py   # Tests for portfolio module
```
