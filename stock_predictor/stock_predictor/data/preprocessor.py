"""Data preprocessing module for stock data."""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import warnings
from ..config import Config


class DataPreprocessor:
    """Handles preprocessing of stock data for analysis and prediction."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration object with preprocessing settings
        """
        self.config = config or Config()
    
    def load_and_prepare_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load stock data from CSV and prepare it for analysis.
        
        Args:
            csv_path: Path to the CSV file containing stock data
            
        Returns:
            DataFrame with prepared stock data
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If data format is invalid
        """
        try:
            # Load data
            data = pd.read_csv(csv_path)
            
            # Ensure Date column exists and convert to datetime
            if 'Date' not in data.columns:
                raise ValueError("CSV file must contain a 'Date' column")
            
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            
            # Validate required columns
            required_columns = ['Close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            print(f"Loaded data with shape: {data.shape}")
            print(f"Date range: {data.index.min()} to {data.index.max()}")
            
            return data
            
        except Exception as e:
            raise Exception(f"Failed to load and prepare data: {str(e)}")
    
    def generate_synthetic_2025_data(self, historical_data: pd.DataFrame, 
                                   end_date: str = '2025-06-19') -> pd.DataFrame:
        """
        Generate synthetic 2025 data based on historical patterns.
        
        Args:
            historical_data: Historical stock data
            end_date: End date for synthetic data generation
            
        Returns:
            DataFrame with synthetic 2025 data
        """
        try:
            # Get starting price from last historical data point
            start_price = historical_data['Close'].iloc[-1]
            
            # Generate business day dates for 2025 H1
            dates_2025h1 = pd.date_range(start='2025-01-01', end=end_date, freq='B')
            h1_2025_data = pd.DataFrame(index=dates_2025h1)
            
            # Generate realistic prices with controlled randomness
            np.random.seed(self.config.random_seed)
            prices_2025h1 = [start_price]
            
            for i in range(1, len(dates_2025h1)):
                daily_return = np.random.normal(0.0002, 0.01)  # Small positive trend with volatility
                new_price = prices_2025h1[-1] * (1 + daily_return)
                prices_2025h1.append(new_price)
            
            h1_2025_data['Close'] = prices_2025h1
            h1_2025_data['Volume'] = np.random.randint(100000, 1000000, size=len(dates_2025h1))
            
            print(f"Generated synthetic 2025 data with {len(h1_2025_data)} data points")
            
            return h1_2025_data
            
        except Exception as e:
            raise Exception(f"Failed to generate synthetic 2025 data: {str(e)}")
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators and features for the stock data.
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            DataFrame with added features
        """
        try:
            # Create a copy to avoid modifying original data
            data = df.copy()
            
            # Calculate returns
            data['Returns'] = data['Close'].pct_change()
            
            # Calculate rolling volatility
            data['Daily_Volatility'] = data['Returns'].rolling(
                window=self.config.volatility_window
            ).std()
            
            # Calculate Simple Moving Average
            data['SMA_20'] = data['Close'].rolling(
                window=self.config.sma_window
            ).mean()
            
            # Additional features that could be useful
            data['Price_Change'] = data['Close'].diff()
            data['High_Low_Ratio'] = data.get('High', data['Close']) / data.get('Low', data['Close'])
            
            print("Created technical features:")
            print(f"- Returns (daily price changes)")
            print(f"- Daily Volatility ({self.config.volatility_window}-day rolling)")
            print(f"- SMA ({self.config.sma_window}-day moving average)")
            print(f"- Price Change")
            print(f"- High/Low Ratio")
            
            return data
            
        except Exception as e:
            raise Exception(f"Failed to create features: {str(e)}")
    
    def combine_datasets(self, historical_data: pd.DataFrame, 
                        synthetic_data: pd.DataFrame) -> pd.DataFrame:
        """
        Combine historical and synthetic data into a single dataset.
        
        Args:
            historical_data: Historical stock data
            synthetic_data: Synthetic/projected data
            
        Returns:
            Combined DataFrame
        """
        try:
            # Combine historical and synthetic data
            full_data = pd.concat([historical_data, synthetic_data])
            
            # Add features to the combined dataset
            full_data = self.create_features(full_data)
            
            # Remove rows with NaN values (from rolling calculations)
            full_data = full_data.dropna()
            
            print(f"Combined dataset shape: {full_data.shape}")
            print(f"Date range: {full_data.index.min()} to {full_data.index.max()}")
            
            return full_data
            
        except Exception as e:
            raise Exception(f"Failed to combine datasets: {str(e)}")
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate the processed data for completeness and quality.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Check for required columns
            required_columns = ['Close', 'Returns', 'Daily_Volatility']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                warnings.warn(f"Missing required columns: {missing_columns}")
                return False
            
            # Check for excessive NaN values
            nan_percentage = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if nan_percentage > 0.1:  # More than 10% NaN values
                warnings.warn(f"High percentage of NaN values: {nan_percentage:.2%}")
                return False
            
            # Check for data consistency
            if data['Close'].min() <= 0:
                warnings.warn("Found non-positive closing prices")
                return False
            
            # Check date index is sorted
            if not data.index.is_monotonic_increasing:
                warnings.warn("Date index is not sorted")
                return False
            
            print("Data validation passed")
            return True
            
        except Exception as e:
            warnings.warn(f"Data validation failed: {str(e)}")
            return False
    
    def get_data_summary(self, data: pd.DataFrame) -> dict:
        """
        Get summary statistics for the processed data.
        
        Args:
            data: DataFrame to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        try:
            summary = {
                'shape': data.shape,
                'date_range': (data.index.min(), data.index.max()),
                'price_range': (data['Close'].min(), data['Close'].max()),
                'mean_price': data['Close'].mean(),
                'mean_volatility': data['Daily_Volatility'].mean() if 'Daily_Volatility' in data.columns else None,
                'total_return': (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100,
                'missing_values': data.isnull().sum().sum()
            }
            
            return summary
            
        except Exception as e:
            warnings.warn(f"Failed to generate data summary: {str(e)}")
            return {}