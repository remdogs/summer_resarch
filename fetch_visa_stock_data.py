import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_visa_stock_data():
    # Download historical data for Visa Inc. (ticker: V)
    ticker = 'V'
    visa = yf.Ticker(ticker)
    # You can adjust the period or interval as needed
    df = visa.history(period="max")
    df.reset_index(inplace=True)
    # Save to CSV
    filename = "Visa_stock_up_to_date.csv"
    df.to_csv(filename, index=False)
    print(f"Visa stock data saved to {filename}")

if __name__ == "__main__":
    fetch_visa_stock_data()