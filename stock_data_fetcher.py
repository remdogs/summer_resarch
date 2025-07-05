"""
Generic stock data fetcher module that supports any ticker symbol and configurable date ranges.
"""
import yfinance as yf
import pandas as pd
from typing import Optional, Union
from datetime import datetime


class StockDataFetcher:
    """
    A generic stock data fetcher that can download historical data for any stock
    from various sources (starting with yfinance).
    """
    
    def __init__(self, data_source: str = "yfinance"):
        """
        Initialize the stock data fetcher.
        
        Args:
            data_source: The data source to use (currently only 'yfinance' supported)
        """
        self.data_source = data_source
        if data_source != "yfinance":
            raise ValueError("Currently only 'yfinance' data source is supported")
    
    def fetch_stock_data(
        self,
        ticker: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical stock data for the specified ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'JD.L', 'MSFT')
            start_date: Start date for data fetch (YYYY-MM-DD format or datetime)
            end_date: End date for data fetch (YYYY-MM-DD format or datetime)
            period: Period to download (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            DataFrame with stock data including Date index and OHLCV columns
        """
        try:
            # Create yfinance ticker object
            stock = yf.Ticker(ticker)
            
            # Download data based on parameters
            if period:
                data = stock.history(period=period, interval=interval)
            else:
                data = stock.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for ticker '{ticker}' with the specified parameters")
            
            # Reset index to make Date a column, then set it back as index with proper name
            data = data.reset_index()
            if 'Date' not in data.columns:
                # Handle different column names that might come from yfinance
                date_col = data.columns[0]  # First column is usually the date
                data.rename(columns={date_col: 'Date'}, inplace=True)
            
            # Ensure Date is datetime
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            
            # Standardize column names
            column_mapping = {
                'Open': 'Open',
                'High': 'High', 
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume',
                'Adj Close': 'Adj Close'
            }
            
            # Rename columns to ensure consistency
            for old_name, new_name in column_mapping.items():
                if old_name in data.columns:
                    data.rename(columns={old_name: new_name}, inplace=True)
            
            return data
            
        except Exception as e:
            raise Exception(f"Error fetching data for {ticker}: {str(e)}")
    
    def save_to_csv(
        self,
        data: pd.DataFrame,
        filename: str,
        clean_data: bool = True
    ) -> None:
        """
        Save stock data to CSV file.
        
        Args:
            data: DataFrame with stock data
            filename: Output filename
            clean_data: Whether to clean the data before saving
        """
        if clean_data:
            # Remove any rows with invalid data
            data_clean = data.dropna()
            # Remove any duplicate dates
            data_clean = data_clean[~data_clean.index.duplicated(keep='first')]
        else:
            data_clean = data
        
        data_clean.to_csv(filename)
        print(f"Stock data saved to {filename}")
    
    def get_stock_info(self, ticker: str) -> dict:
        """
        Get basic information about a stock.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with stock information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'symbol': ticker,
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'currency': info.get('currency', 'N/A'),
                'exchange': info.get('exchange', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A')
            }
        except Exception as e:
            return {'symbol': ticker, 'error': str(e)}


def fetch_multiple_stocks(
    tickers: list,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = None
) -> dict:
    """
    Convenience function to fetch data for multiple stocks.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        period: Period to download
        
    Returns:
        Dictionary with ticker as key and DataFrame as value
    """
    fetcher = StockDataFetcher()
    results = {}
    
    for ticker in tickers:
        try:
            data = fetcher.fetch_stock_data(ticker, start_date, end_date, period)
            results[ticker] = data
            print(f"Successfully fetched data for {ticker}")
        except Exception as e:
            print(f"Failed to fetch data for {ticker}: {e}")
            results[ticker] = None
    
    return results