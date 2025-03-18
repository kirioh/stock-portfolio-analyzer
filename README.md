# Stock Portfolio Analyzer

A modular, production-ready Python tool for analyzing a stock portfolio from a CSV file, with support for time-based performance analysis.

## Features

- Fetch current stock prices using the yfinance API
- Calculate portfolio value, gain/loss, and performance metrics
- Track portfolio performance over time using transaction history
- Visualize returns as a function of time with interactive plots
- Parse transaction statements (Revolut format supported)
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

### Basic Portfolio Analysis

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

### Time-Based Portfolio Analysis

```bash
# Analyze portfolio performance over time using a transaction statement
portfolio-analyzer --csv-file /path/to/portfolio.csv --statement-file /path/to/statement.txt --time-analysis

# Generate and display plots
portfolio-analyzer --csv-file /path/to/portfolio.csv --statement-file /path/to/statement.txt --time-analysis --show-plots 

# Save plots to a custom directory
portfolio-analyzer --csv-file /path/to/portfolio.csv --statement-file /path/to/statement.txt --time-analysis --output-dir ./charts

# Using the Revolut statement parser
python revolut_report.py --statement-file /path/to/statement.txt --output-dir ./charts --time-analysis
```

## Programmatic Usage

### Basic Portfolio Analysis

You can use the library programmatically in your own Python scripts:

```python
from src.stock_portfolio_analyzer.main import run_analysis

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

### Time-Based Portfolio Analysis

For time-based analysis using transaction history:

```python
from src.stock_portfolio_analyzer.main import run_time_based_analysis

# Run time-based analysis with a transaction statement
results = run_time_based_analysis(
    csv_file_path='/path/to/portfolio.csv',
    statement_file_path='/path/to/statement.txt',
    output_dir='./charts',
    show_plots=True  # Set to True to display interactive plots
)

# Access time-based results
time_df = results['time_analysis']['dataframe']
dates = results['time_analysis']['dates']
values = results['time_analysis']['values']
returns = results['time_analysis']['returns']

# Access plot file paths
plot_files = results['plots']

# Process time-based results
for _, row in time_df.iterrows():
    date = row['date_label']
    value = row['portfolio_value']
    ret_pct = row['portfolio_return_pct']
    print(f"{date}: ${value:.2f}, {ret_pct:.2f}%")
```

See the `example.py` and `example_time_analysis.py` files for more comprehensive examples of programmatic usage.

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

## Transaction Statement Format

The tool supports parsing and analyzing transaction statements in Revolut format. The statement should contain:

- Transaction history with dates, symbols, quantities, prices, and transaction types
- Portfolio information including symbols, quantities, and values

An example of the supported statement format is embedded in the `revolut_report.py` file.

## Project Structure

```
stock_portfolio_analyzer/
├── pyproject.toml             # Package configuration
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── config.yaml                # Default configuration
├── portfolio.csv              # Example portfolio data
├── example.py                 # Example script for programmatic usage
├── example_time_analysis.py   # Example script for time-based analysis
├── revolut_report.py          # Revolut statement parser and reporter
├── src/
│   └── stock_portfolio_analyzer/
│       ├── __init__.py        # Package initialization
│       ├── cli.py             # Command-line interface
│       ├── main.py            # Main programmatic entry point
│       ├── config.py          # Configuration management
│       ├── logger.py          # Logging configuration
│       ├── fetcher.py         # Stock price fetching logic
│       ├── portfolio.py       # Portfolio analysis logic
│       ├── transactions.py    # Transaction parsing and processing
│       ├── time_analysis.py   # Time-based analysis and visualization
│       └── utils.py           # Helper utilities
└── tests/
    ├── __init__.py
    └── test_portfolio.py      # Tests for portfolio module
```
