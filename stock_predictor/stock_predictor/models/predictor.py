"""Stock prediction model module."""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import warnings
from ..config import Config


class StockPredictor:
    """Handles stock price prediction using statistical modeling."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the stock predictor.
        
        Args:
            config: Configuration object with prediction settings
        """
        self.config = config or Config()
        
    def generate_predictions(self, 
                           last_price: float, 
                           days: int, 
                           historical_data: pd.DataFrame) -> List[float]:
        """
        Generate stock price predictions for specified number of days.
        
        Args:
            last_price: Starting price for predictions
            days: Number of days to predict
            historical_data: Historical data for parameter estimation
            
        Returns:
            List of predicted prices
        """
        try:
            predictions = [last_price]
            
            # Parameters from recent data
            recent_volatility = self._calculate_recent_volatility(historical_data)
            recent_trend = self._calculate_recent_trend(historical_data)
            
            # Conservative bounds based on configuration
            max_price = last_price * (1 + self.config.max_price_change)
            min_price = last_price * (1 - self.config.max_price_change)
            
            # Generate predictions
            for day in range(days):
                prev_price = predictions[-1]
                
                # Smooth transition factor
                transition_factor = min(1.0, day / self.config.transition_days)
                current_trend = recent_trend * (1 - transition_factor)
                
                # Mild seasonal component
                seasonal = 0.001 * np.sin(2 * np.pi * day / 252)
                
                # Controlled volatility
                daily_vol = min(recent_volatility * 0.5, 0.008) * (1 + transition_factor)
                random_component = np.random.normal(0, daily_vol)
                
                # Combined return
                daily_return = current_trend + seasonal + random_component
                
                # Apply limits
                daily_return = np.clip(daily_return, 
                                     -self.config.max_daily_return, 
                                     self.config.max_daily_return)
                new_price = prev_price * (1 + daily_return)
                new_price = np.clip(new_price, min_price, max_price)
                
                predictions.append(new_price)
            
            return predictions[1:]  # Remove the initial price
            
        except Exception as e:
            raise Exception(f"Failed to generate predictions: {str(e)}")
    
    def generate_scenario_predictions(self,
                                    last_price: float,
                                    days: int,
                                    historical_data: pd.DataFrame,
                                    n_scenarios: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate multiple prediction scenarios and calculate statistics.
        
        Args:
            last_price: Starting price for predictions
            days: Number of days to predict
            historical_data: Historical data for parameter estimation
            n_scenarios: Number of scenarios to generate (uses config default if None)
            
        Returns:
            Tuple of (mean_predictions, confidence_lower, confidence_upper)
        """
        try:
            n_scenarios = n_scenarios or self.config.prediction_scenarios
            all_scenarios = []
            
            print(f"Generating {n_scenarios} prediction scenarios...")
            
            for i in range(n_scenarios):
                if i % 10 == 0:
                    print(f"Processing scenario {i+1}/{n_scenarios}")
                
                scenario = self.generate_predictions(last_price, days, historical_data)
                all_scenarios.append(scenario)
            
            # Calculate statistics
            all_scenarios = np.array(all_scenarios)
            mean_predictions = np.mean(all_scenarios, axis=0)
            
            # Calculate confidence intervals
            confidence_level = self.config.confidence_interval
            lower_percentile = (100 - confidence_level) / 2
            upper_percentile = 100 - lower_percentile
            
            confidence_lower = np.percentile(all_scenarios, lower_percentile, axis=0)
            confidence_upper = np.percentile(all_scenarios, upper_percentile, axis=0)
            
            print(f"Generated predictions with {confidence_level}% confidence intervals")
            
            return mean_predictions, confidence_lower, confidence_upper
            
        except Exception as e:
            raise Exception(f"Failed to generate scenario predictions: {str(e)}")
    
    def predict_future_prices(self,
                            historical_data: pd.DataFrame,
                            start_date: str = '2025-06-19',
                            end_date: str = '2025-12-31') -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict future stock prices for a specified date range.
        
        Args:
            historical_data: Historical stock data
            start_date: Start date for predictions
            end_date: End date for predictions
            
        Returns:
            Tuple of (future_dates, predictions, confidence_lower, confidence_upper)
        """
        try:
            # Setup prediction parameters
            current_date = pd.Timestamp(start_date)
            end_date_ts = pd.Timestamp(end_date)
            future_dates = pd.date_range(start=current_date, end=end_date_ts, freq='B')
            n_days = len(future_dates)
            last_known_price = historical_data['Close'].iloc[-1]
            
            # Generate scenario predictions
            predictions, confidence_lower, confidence_upper = self.generate_scenario_predictions(
                last_known_price, n_days, historical_data
            )
            
            print(f"Predicted prices for {n_days} business days")
            print(f"Price range: £{predictions.min():.2f} - £{predictions.max():.2f}")
            
            return future_dates, predictions, confidence_lower, confidence_upper
            
        except Exception as e:
            raise Exception(f"Failed to predict future prices: {str(e)}")
    
    def _calculate_recent_volatility(self, data: pd.DataFrame, window: int = 20) -> float:
        """
        Calculate recent volatility from historical data.
        
        Args:
            data: Historical data with Daily_Volatility column
            window: Window size for calculation
            
        Returns:
            Recent volatility value
        """
        try:
            if 'Daily_Volatility' in data.columns:
                return data['Daily_Volatility'].iloc[-window:].mean()
            else:
                # Fallback: calculate from returns
                returns = data['Close'].pct_change()
                return returns.iloc[-window:].std()
                
        except Exception as e:
            warnings.warn(f"Could not calculate recent volatility: {str(e)}")
            return 0.02  # Default volatility
    
    def _calculate_recent_trend(self, data: pd.DataFrame, window: int = 20) -> float:
        """
        Calculate recent price trend from historical data.
        
        Args:
            data: Historical data with Close prices
            window: Window size for calculation
            
        Returns:
            Recent trend value (daily return)
        """
        try:
            recent_start_price = data['Close'].iloc[-window]
            recent_end_price = data['Close'].iloc[-1]
            return (recent_end_price / recent_start_price - 1) / window
            
        except Exception as e:
            warnings.warn(f"Could not calculate recent trend: {str(e)}")
            return 0.0  # Default neutral trend
    
    def evaluate_prediction_quality(self,
                                  predictions: np.ndarray,
                                  actual_prices: Optional[np.ndarray] = None) -> dict:
        """
        Evaluate the quality of predictions (if actual data is available).
        
        Args:
            predictions: Predicted prices
            actual_prices: Actual prices (if available for validation)
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            metrics = {
                'prediction_count': len(predictions),
                'price_range': (predictions.min(), predictions.max()),
                'mean_price': predictions.mean(),
                'std_price': predictions.std(),
                'coefficient_of_variation': predictions.std() / predictions.mean()
            }
            
            if actual_prices is not None and len(actual_prices) == len(predictions):
                # Calculate error metrics
                errors = predictions - actual_prices
                metrics.update({
                    'mae': np.mean(np.abs(errors)),  # Mean Absolute Error
                    'rmse': np.sqrt(np.mean(errors**2)),  # Root Mean Square Error
                    'mape': np.mean(np.abs(errors / actual_prices)) * 100,  # Mean Absolute Percentage Error
                    'accuracy': 1 - np.mean(np.abs(errors / actual_prices))  # Accuracy (1 - MAPE/100)
                })
            
            return metrics
            
        except Exception as e:
            warnings.warn(f"Could not evaluate prediction quality: {str(e)}")
            return {}
    
    def get_prediction_summary(self,
                             predictions: np.ndarray,
                             confidence_lower: np.ndarray,
                             confidence_upper: np.ndarray,
                             start_price: float) -> dict:
        """
        Get summary statistics for predictions.
        
        Args:
            predictions: Mean predictions
            confidence_lower: Lower confidence bound
            confidence_upper: Upper confidence bound
            start_price: Starting price
            
        Returns:
            Dictionary with prediction summary
        """
        try:
            total_return = (predictions[-1] / start_price - 1) * 100
            volatility = np.std(predictions) / np.mean(predictions) * 100
            
            summary = {
                'start_price': start_price,
                'end_price': predictions[-1],
                'total_return_pct': total_return,
                'volatility_pct': volatility,
                'max_price': predictions.max(),
                'min_price': predictions.min(),
                'confidence_range': (confidence_lower[-1], confidence_upper[-1]),
                'prediction_days': len(predictions)
            }
            
            return summary
            
        except Exception as e:
            warnings.warn(f"Could not generate prediction summary: {str(e)}")
            return {}