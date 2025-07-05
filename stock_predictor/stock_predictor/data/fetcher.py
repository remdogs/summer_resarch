"""Stock data fetching module."""

import yfinance as yf
import pandas as pd
import csv
from datetime import datetime
from typing import Optional, Union
import warnings
from pathlib import Path


class StockDataFetcher:
    """Handles fetching stock data from various sources."""
    
    def __init__(self):
        """Initialize the stock data fetcher."""
        pass
    
    def fetch_jd_sports_data(self, 
                            end_date: str = "2024-12-31",
                            save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch JD Sports stock data from Yahoo Finance.
        
        Args:
            end_date: End date for data fetch in YYYY-MM-DD format
            save_path: Optional path to save the data
            
        Returns:
            DataFrame with stock data
            
        Raises:
            Exception: If data fetching fails
        """
        try:
            # Define the stock ticker for JD Sports (London Stock Exchange: JD.L)
            ticker_symbol = "JD.L"
            
            # Download historical daily data from the earliest available date up to the end date
            jd_sports_data = yf.download(ticker_symbol, end=end_date, progress=True)
            
            if jd_sports_data.empty:
                raise ValueError(f"No data found for ticker {ticker_symbol}")
            
            # Reset index to get Date as a column
            jd_sports_data.reset_index(inplace=True)
            
            # Save if path provided
            if save_path:
                self._save_and_clean_data(jd_sports_data, save_path)
            
            print(f"Downloaded JD Sports share price data up to {end_date}")
            print(f"Data shape: {jd_sports_data.shape}")
            
            return jd_sports_data
            
        except Exception as e:
            raise Exception(f"Failed to fetch JD Sports data: {str(e)}")
    
    def fetch_visa_stock_data(self, 
                             period: str = "max",
                             save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch Visa stock data from Yahoo Finance.
        
        Args:
            period: Period for data fetch (max, 1y, 2y, etc.)
            save_path: Optional path to save the data
            
        Returns:
            DataFrame with stock data
            
        Raises:
            Exception: If data fetching fails
        """
        try:
            # Download historical data for Visa Inc. (ticker: V)
            ticker = 'V'
            visa = yf.Ticker(ticker)
            df = visa.history(period=period)
            
            if df.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            df.reset_index(inplace=True)
            
            # Save if path provided
            if save_path:
                df.to_csv(save_path, index=False)
                print(f"Visa stock data saved to {save_path}")
            
            print(f"Downloaded Visa stock data for period: {period}")
            print(f"Data shape: {df.shape}")
            
            return df
            
        except Exception as e:
            raise Exception(f"Failed to fetch Visa data: {str(e)}")
    
    def fetch_stock_data(self, 
                        ticker: str,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        period: str = "max",
                        save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generic method to fetch stock data for any ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format  
            period: Period if not using start/end dates
            save_path: Optional path to save the data
            
        Returns:
            DataFrame with stock data
            
        Raises:
            Exception: If data fetching fails
        """
        try:
            stock = yf.Ticker(ticker)
            
            if start_date and end_date:
                df = stock.history(start=start_date, end=end_date)
            else:
                df = stock.history(period=period)
            
            if df.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            df.reset_index(inplace=True)
            
            # Save if path provided
            if save_path:
                df.to_csv(save_path, index=False)
                print(f"Stock data for {ticker} saved to {save_path}")
            
            print(f"Downloaded stock data for {ticker}")
            print(f"Data shape: {df.shape}")
            
            return df
            
        except Exception as e:
            raise Exception(f"Failed to fetch data for {ticker}: {str(e)}")
    
    def _save_and_clean_data(self, data: pd.DataFrame, file_path: str) -> None:
        """
        Save data to CSV and perform basic cleaning.
        
        Args:
            data: DataFrame to save
            file_path: Path to save the file
        """
        try:
            # Save the initial data
            data.to_csv(file_path, index=False)
            
            # Read back and clean
            df = pd.read_csv(file_path)
            
            # Ensure Date column exists
            if "Date" not in df.columns and len(df.columns) > 0:
                df.rename(columns={df.columns[0]: "Date"}, inplace=True)
            
            # Remove any rows with ticker or date headers that might have been added
            if 'Date' in df.columns:
                df = df[~df["Date"].astype(str).str.contains("Ticker|Date", na=False)]
            
            # Save cleaned data
            df.to_csv(file_path, index=False)
            
            # Create a cleaned version without unnecessary rows
            cleaned_file = file_path.replace('.csv', '_cleaned.csv')
            self._remove_header_rows(file_path, cleaned_file)
            
        except Exception as e:
            warnings.warn(f"Warning: Could not clean data file {file_path}: {str(e)}")
    
    def _remove_header_rows(self, input_file: str, output_file: str) -> None:
        """
        Remove potential header rows (lines 2 and 3) from CSV file.
        
        Args:
            input_file: Input CSV file path
            output_file: Output CSV file path
        """
        try:
            with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
                reader = csv.reader(infile)
                writer = csv.writer(outfile)
                
                # Iterate through the rows and skip lines 2 and 3 (index 1 and 2)
                for i, row in enumerate(reader):
                    if i not in (1, 2):  # Skip lines 2 and 3
                        writer.writerow(row)
            
            print(f"Cleaned file saved as {output_file}")
            
        except Exception as e:
            warnings.warn(f"Warning: Could not create cleaned file: {str(e)}")