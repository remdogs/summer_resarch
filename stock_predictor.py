"""
Stock Predictor class that provides configurable stock price prediction functionality.
"""
from typing import Optional, Dict, List, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from stock_data_fetcher import StockDataFetcher


class StockPredictor:
    """
    A configurable stock price predictor that can work with any stock and 
    supports different prediction strategies and customizable parameters.
    """
    
    def __init__(
        self,
        ticker: str,
        confidence_interval: float = 0.70,
        max_price_change: float = 0.15,
        volatility_window: int = 20,
        n_scenarios: int = 50
    ):
        """
        Initialize the StockPredictor.
        
        Args:
            ticker: Stock ticker symbol
            confidence_interval: Confidence interval for predictions (0.0-1.0)
            max_price_change: Maximum allowed price change as a fraction (e.g., 0.15 = 15%)
            volatility_window: Window size for calculating volatility
            n_scenarios: Number of scenarios to generate for Monte Carlo simulation
        """
        self.ticker = ticker
        self.confidence_interval = confidence_interval
        self.max_price_change = max_price_change
        self.volatility_window = volatility_window
        self.n_scenarios = n_scenarios
        self.data = None
        self.stock_info = None
        self.data_fetcher = StockDataFetcher()
        
    def load_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = "max"
    ) -> None:
        """
        Load historical data for the stock.
        
        Args:
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            period: Period to load if start/end dates not specified
        """
        try:
            self.data = self.data_fetcher.fetch_stock_data(
                self.ticker, start_date, end_date, period
            )
            self.stock_info = self.data_fetcher.get_stock_info(self.ticker)
            print(f"Loaded data for {self.ticker} ({self.stock_info.get('name', 'N/A')})")
            print(f"Data range: {self.data.index[0]} to {self.data.index[-1]}")
            print(f"Total records: {len(self.data)}")
        except Exception as e:
            raise Exception(f"Failed to load data for {self.ticker}: {e}")
    
    def preprocess_data(self) -> None:
        """
        Prepare data for prediction by adding technical indicators and features.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Calculate returns and volatility
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Daily_Volatility'] = self.data['Returns'].rolling(
            window=self.volatility_window
        ).std()
        
        # Simple moving averages
        self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        
        # Additional technical indicators
        self.data['Price_Change'] = self.data['Close'].diff()
        self.data['High_Low_Pct'] = (self.data['High'] - self.data['Low']) / self.data['Close']
        
        # Remove NaN values
        self.data = self.data.dropna()
        
        print(f"Data preprocessed. Available features: {list(self.data.columns)}")
    
    def _generate_single_prediction(
        self,
        last_price: float,
        days: int,
        historical_data: pd.DataFrame
    ) -> List[float]:
        """
        Generate a single prediction scenario.
        
        Args:
            last_price: Starting price for prediction
            days: Number of days to predict
            historical_data: Historical data to base predictions on
            
        Returns:
            List of predicted prices
        """
        predictions = [last_price]
        
        # Parameters from recent data
        recent_volatility = historical_data['Daily_Volatility'].iloc[-self.volatility_window:].mean()
        recent_trend = (
            historical_data['Close'].iloc[-1] / 
            historical_data['Close'].iloc[-self.volatility_window] - 1
        ) / self.volatility_window
        
        # Conservative bounds
        max_price = last_price * (1 + self.max_price_change)
        min_price = last_price * (1 - self.max_price_change)
        
        # Smooth transition parameters
        transition_days = 10
        
        for day in range(days):
            prev_price = predictions[-1]
            
            # Smooth transition
            transition_factor = min(1.0, day / transition_days)
            current_trend = recent_trend * (1 - transition_factor)
            
            # Mild seasonal component
            seasonal = 0.001 * np.sin(2 * np.pi * day / 252)
            
            # Controlled volatility
            daily_vol = min(recent_volatility * 0.5, 0.008) * (1 + transition_factor)
            random_component = np.random.normal(0, daily_vol)
            
            # Combined return
            daily_return = current_trend + seasonal + random_component
            
            # Apply limits
            max_daily_change = 0.015  # Max 1.5% daily move
            daily_return = np.clip(daily_return, -max_daily_change, max_daily_change)
            new_price = prev_price * (1 + daily_return)
            new_price = np.clip(new_price, min_price, max_price)
            
            predictions.append(new_price)
        
        return predictions[1:]
    
    def generate_predictions(
        self,
        start_date: str,
        end_date: str,
        prediction_method: str = "monte_carlo"
    ) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """
        Generate predictions for the specified period.
        
        Args:
            start_date: Start date for predictions (YYYY-MM-DD)
            end_date: End date for predictions (YYYY-MM-DD)
            prediction_method: Method to use for predictions
            
        Returns:
            Dictionary containing predictions and confidence intervals
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if len(self.data) == 0:
            raise ValueError("No preprocessed data available. Call preprocess_data() first.")
        
        # Generate date range for predictions
        future_dates = pd.date_range(start=start_date, end=end_date, freq='B')
        n_days = len(future_dates)
        last_known_price = self.data['Close'].iloc[-1]
        
        print(f"Generating {n_days} days of predictions from {start_date} to {end_date}")
        print(f"Starting price: {last_known_price:.2f}")
        
        if prediction_method == "monte_carlo":
            # Generate multiple scenarios
            all_scenarios = []
            
            for i in range(self.n_scenarios):
                if i % 10 == 0:
                    print(f"Processing scenario {i+1}/{self.n_scenarios}")
                scenario = self._generate_single_prediction(last_known_price, n_days, self.data)
                all_scenarios.append(scenario)
            
            # Calculate statistics
            predictions = np.mean(all_scenarios, axis=0)
            
            # Calculate confidence intervals
            lower_percentile = (1 - self.confidence_interval) / 2 * 100
            upper_percentile = (1 + self.confidence_interval) / 2 * 100
            
            confidence_lower = np.percentile(all_scenarios, lower_percentile, axis=0)
            confidence_upper = np.percentile(all_scenarios, upper_percentile, axis=0)
            
            return {
                'dates': future_dates,
                'predictions': pd.Series(predictions, index=future_dates),
                'confidence_lower': pd.Series(confidence_lower, index=future_dates),
                'confidence_upper': pd.Series(confidence_upper, index=future_dates),
                'all_scenarios': np.array(all_scenarios)
            }
        else:
            raise ValueError(f"Prediction method '{prediction_method}' not supported")
    
    def plot_predictions(
        self,
        predictions: Dict[str, Union[pd.Series, pd.DataFrame]],
        show_confidence: bool = True,
        show_historical: bool = True,
        historical_days: int = 90,
        figsize: tuple = (15, 8)
    ) -> None:
        """
        Plot the predictions with optional confidence intervals and historical data.
        
        Args:
            predictions: Dictionary containing prediction results
            show_confidence: Whether to show confidence intervals
            show_historical: Whether to show recent historical data
            historical_days: Number of historical days to show
            figsize: Figure size tuple
        """
        plt.figure(figsize=figsize)
        
        # Show recent historical data
        if show_historical and self.data is not None:
            historical_end = self.data.index[-1]
            historical_start = historical_end - timedelta(days=historical_days)
            historical_data = self.data[self.data.index >= historical_start]
            
            plt.plot(
                historical_data.index, 
                historical_data['Close'], 
                label='Historical Data', 
                color='blue',
                linewidth=2
            )
        
        # Plot predictions
        plt.plot(
            predictions['dates'], 
            predictions['predictions'], 
            label='Predictions', 
            color='red', 
            linestyle='--',
            linewidth=2
        )
        
        # Plot confidence intervals
        if show_confidence:
            plt.fill_between(
                predictions['dates'], 
                predictions['confidence_lower'], 
                predictions['confidence_upper'],
                color='red', 
                alpha=0.2, 
                label=f'{self.confidence_interval*100:.0f}% Confidence Interval'
            )
        
        # Formatting
        currency = self.stock_info.get('currency', '') if self.stock_info else ''
        currency_symbol = '$' if currency == 'USD' else '£' if currency == 'GBP' else ''
        
        plt.title(f'{self.ticker} Stock Price Predictions\n{self.stock_info.get("name", "") if self.stock_info else ""}')
        plt.xlabel('Date')
        plt.ylabel(f'Price ({currency_symbol})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Format y-axis
        if currency_symbol:
            plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter(f'{currency_symbol}%.2f'))
        
        plt.tight_layout()
        plt.show()
    
    def generate_analysis_report(
        self,
        predictions: Dict[str, Union[pd.Series, pd.DataFrame]]
    ) -> str:
        """
        Generate a detailed analysis report.
        
        Args:
            predictions: Dictionary containing prediction results
            
        Returns:
            Formatted analysis report as string
        """
        if self.data is None:
            return "No historical data available for analysis."
        
        # Current stats
        current_price = self.data['Close'].iloc[-1]
        start_prediction = predictions['predictions'].iloc[0]
        end_prediction = predictions['predictions'].iloc[-1]
        
        # Calculate returns
        predicted_return = (end_prediction / start_prediction - 1) * 100
        
        # Calculate volatility
        prediction_volatility = predictions['predictions'].std() / predictions['predictions'].mean() * 100
        
        # Recent historical performance
        recent_data = self.data.tail(30)  # Last 30 days
        recent_return = (recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[0] - 1) * 100
        historical_volatility = recent_data['Daily_Volatility'].mean() * 100
        
        # Generate report
        currency = self.stock_info.get('currency', '') if self.stock_info else ''
        currency_symbol = '$' if currency == 'USD' else '£' if currency == 'GBP' else ''
        
        report = f"""
{self.ticker} Stock Analysis Report
{'=' * 50}
Company: {self.stock_info.get('name', 'N/A') if self.stock_info else 'N/A'}
Sector: {self.stock_info.get('sector', 'N/A') if self.stock_info else 'N/A'}
Exchange: {self.stock_info.get('exchange', 'N/A') if self.stock_info else 'N/A'}

Current Market Data:
- Current Price: {currency_symbol}{current_price:.2f}
- Data Range: {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}
- Total Trading Days: {len(self.data)}

Recent Performance (Last 30 Days):
- Return: {recent_return:.2f}%
- Average Daily Volatility: {historical_volatility:.2f}%

Prediction Summary:
- Prediction Period: {predictions['dates'][0].strftime('%Y-%m-%d')} to {predictions['dates'][-1].strftime('%Y-%m-%d')}
- Starting Price: {currency_symbol}{start_prediction:.2f}
- Predicted End Price: {currency_symbol}{end_prediction:.2f}
- Predicted Return: {predicted_return:.2f}%
- Prediction Volatility: {prediction_volatility:.2f}%
- Confidence Interval: {self.confidence_interval*100:.0f}%

Price Range Forecast:
- Minimum Expected: {currency_symbol}{predictions['confidence_lower'].min():.2f}
- Maximum Expected: {currency_symbol}{predictions['confidence_upper'].max():.2f}
- Most Likely: {currency_symbol}{predictions['predictions'].mean():.2f}

Model Parameters:
- Number of Scenarios: {self.n_scenarios}
- Volatility Window: {self.volatility_window} days
- Max Price Change Limit: ±{self.max_price_change*100:.1f}%

Risk Assessment:
- Downside Risk: {((predictions['confidence_lower'].min() / current_price - 1) * 100):.2f}%
- Upside Potential: {((predictions['confidence_upper'].max() / current_price - 1) * 100):.2f}%
"""
        
        return report
    
    def compare_with_other_stocks(
        self,
        other_tickers: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, Dict]:
        """
        Compare predictions with other stocks.
        
        Args:
            other_tickers: List of other ticker symbols to compare
            start_date: Start date for comparison
            end_date: End date for comparison
            
        Returns:
            Dictionary containing comparison results
        """
        comparison_results = {}
        
        # Get predictions for current stock
        current_predictions = self.generate_predictions(start_date, end_date)
        current_return = (
            current_predictions['predictions'].iloc[-1] / 
            current_predictions['predictions'].iloc[0] - 1
        ) * 100
        
        comparison_results[self.ticker] = {
            'predicted_return': current_return,
            'volatility': current_predictions['predictions'].std() / current_predictions['predictions'].mean() * 100,
            'predictions': current_predictions
        }
        
        # Compare with other stocks
        for ticker in other_tickers:
            try:
                other_predictor = StockPredictor(ticker)
                other_predictor.load_data()
                other_predictor.preprocess_data()
                other_predictions = other_predictor.generate_predictions(start_date, end_date)
                
                other_return = (
                    other_predictions['predictions'].iloc[-1] / 
                    other_predictions['predictions'].iloc[0] - 1
                ) * 100
                
                comparison_results[ticker] = {
                    'predicted_return': other_return,
                    'volatility': other_predictions['predictions'].std() / other_predictions['predictions'].mean() * 100,
                    'predictions': other_predictions
                }
                
            except Exception as e:
                print(f"Failed to analyze {ticker}: {e}")
                comparison_results[ticker] = {'error': str(e)}
        
        return comparison_results