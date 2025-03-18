"""
Example script showing how to use the time-based analysis features 
of the stock portfolio analyzer.

This script demonstrates how to parse transaction data directly from a Revolut statement,
run time-based analysis, and generate visualizations.
"""

import os
import sys
from src.stock_portfolio_analyzer.main import run_time_based_analysis

# Example transaction statement
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

def main():
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save statement to a file (optional)
    statement_file = os.path.join(output_dir, "revolut_statement.txt")
    with open(statement_file, 'w') as f:
        f.write(EXAMPLE_STATEMENT)
    print(f"Saved transaction statement to {statement_file}")
    
    # CSV portfolio file path
    csv_file = "./portfolio.csv"
    
    # Run time-based analysis
    print("Running time-based portfolio analysis...")
    try:
        results = run_time_based_analysis(
            csv_file_path=csv_file,
            statement_text=EXAMPLE_STATEMENT,
            output_dir=output_dir,
            show_plots=True  # Set to True to display plots interactively
        )
        
        # Print results summary
        print("\nTime-based analysis completed successfully!")
        print(f"Analyzed {len(results['time_analysis']['dates'])} dates")
        
        # List generated plots
        if results.get('plots'):
            print(f"\nGenerated plots saved to: {output_dir}")
            for plot_name, file_path in results['plots'].items():
                if isinstance(file_path, str):
                    print(f"  - {os.path.basename(file_path)}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())