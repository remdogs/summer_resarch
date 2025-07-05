"""Analysis utilities for stock data and predictions."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
from ..config import Config


class StockAnalyzer:
    """Handles analysis and reporting of stock data and predictions."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the stock analyzer.
        
        Args:
            config: Configuration object with analysis settings
        """
        self.config = config or Config()
    
    def generate_performance_report(self,
                                  historical_data: pd.DataFrame,
                                  predictions: np.ndarray,
                                  future_dates: pd.DatetimeIndex,
                                  current_date: Optional[str] = None) -> Dict:
        """
        Generate a comprehensive performance analysis report.
        
        Args:
            historical_data: Historical stock data
            predictions: Predicted future prices
            future_dates: Dates corresponding to predictions
            current_date: Current date for analysis context
            
        Returns:
            Dictionary containing performance metrics and analysis
        """
        try:
            current_date = current_date or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Get 2025 actual data if available
            data_2025 = historical_data[historical_data.index >= '2025-01-01']
            
            report = {
                'metadata': {
                    'generated_by': 'remdogs',
                    'generated_at': current_date,
                    'analysis_period': f"{historical_data.index.min()} to {historical_data.index.max()}",
                    'prediction_period': f"{future_dates[0]} to {future_dates[-1]}"
                }
            }
            
            # 2025 Performance Analysis
            if not data_2025.empty:
                year_start_price = data_2025['Close'].iloc[0]
                current_price = data_2025['Close'].iloc[-1]
                ytd_return = ((current_price / year_start_price) - 1) * 100
                
                report['2025_performance'] = {
                    'year_start_price': year_start_price,
                    'current_price': current_price,
                    'ytd_return_pct': ytd_return,
                    'ytd_return_status': self._classify_return(ytd_return)
                }
            
            # Predicted Performance
            if len(predictions) > 0:
                start_price = predictions[0] if len(predictions) > 0 else historical_data['Close'].iloc[-1]
                end_price = predictions[-1]
                predicted_return = ((end_price / start_price) - 1) * 100
                
                report['predicted_performance'] = {
                    'start_price': start_price,
                    'predicted_end_price': end_price,
                    'predicted_return_pct': predicted_return,
                    'predicted_return_status': self._classify_return(predicted_return),
                    'prediction_days': len(predictions)
                }
            
            return report
            
        except Exception as e:
            raise Exception(f"Failed to generate performance report: {str(e)}")
    
    def analyze_monthly_performance(self,
                                  historical_data: pd.DataFrame,
                                  predictions: np.ndarray,
                                  future_dates: pd.DatetimeIndex) -> Dict:
        """
        Analyze monthly performance for both actual and predicted data.
        
        Args:
            historical_data: Historical stock data
            predictions: Predicted future prices
            future_dates: Dates corresponding to predictions
            
        Returns:
            Dictionary with monthly analysis
        """
        try:
            monthly_analysis = {
                'actual_months': {},
                'predicted_months': {}
            }
            
            # Analyze actual 2025 data (first half)
            data_2025 = historical_data[historical_data.index >= '2025-01-01']
            
            for month in range(1, 7):  # January to June
                month_data = data_2025[data_2025.index.month == month]
                if len(month_data) > 0:
                    month_name = pd.Timestamp(f'2025-{month:02d}-01').strftime('%B')
                    monthly_analysis['actual_months'][month_name] = {
                        'average_price': month_data['Close'].mean(),
                        'price_range': (month_data['Close'].min(), month_data['Close'].max()),
                        'volatility_pct': month_data['Daily_Volatility'].mean() * 100 if 'Daily_Volatility' in month_data.columns else None,
                        'trading_days': len(month_data)
                    }
            
            # Analyze predicted data (second half)
            prediction_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Price': predictions
            })
            prediction_df.set_index('Date', inplace=True)
            
            for month in range(7, 13):  # July to December
                month_start = pd.Timestamp(f"2025-{month:02d}-01")
                month_end = pd.Timestamp(f"2025-{month:02d}-01") + pd.offsets.MonthEnd(1)
                
                month_predictions = prediction_df[
                    (prediction_df.index >= month_start) & 
                    (prediction_df.index <= month_end)
                ]
                
                if len(month_predictions) > 0:
                    month_name = month_start.strftime('%B')
                    prices = month_predictions['Predicted_Price'].values
                    monthly_analysis['predicted_months'][month_name] = {
                        'average_price': np.mean(prices),
                        'price_range': (min(prices), max(prices)),
                        'volatility_pct': (np.std(prices) / np.mean(prices)) * 100,
                        'trading_days': len(prices)
                    }
            
            return monthly_analysis
            
        except Exception as e:
            raise Exception(f"Failed to analyze monthly performance: {str(e)}")
    
    def calculate_risk_metrics(self, data: pd.DataFrame) -> Dict:
        """
        Calculate various risk metrics for the stock.
        
        Args:
            data: Stock data with returns
            
        Returns:
            Dictionary with risk metrics
        """
        try:
            if 'Returns' not in data.columns:
                data = data.copy()
                data['Returns'] = data['Close'].pct_change()
            
            returns = data['Returns'].dropna()
            
            risk_metrics = {
                'volatility_annual': returns.std() * np.sqrt(252),  # Annualized volatility
                'volatility_daily': returns.std(),
                'var_95': np.percentile(returns, 5),  # Value at Risk (95%)
                'var_99': np.percentile(returns, 1),  # Value at Risk (99%)
                'max_drawdown': self._calculate_max_drawdown(data['Close']),
                'sharpe_ratio': self._calculate_sharpe_ratio(returns),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis()
            }
            
            return risk_metrics
            
        except Exception as e:
            warnings.warn(f"Could not calculate risk metrics: {str(e)}")
            return {}
    
    def identify_trend_patterns(self, data: pd.DataFrame, window: int = 20) -> Dict:
        """
        Identify trend patterns in the stock data.
        
        Args:
            data: Stock data
            window: Window for trend analysis
            
        Returns:
            Dictionary with trend analysis
        """
        try:
            prices = data['Close']
            
            # Calculate moving averages
            short_ma = prices.rolling(window=window//2).mean()
            long_ma = prices.rolling(window=window).mean()
            
            # Current trend
            current_trend = "neutral"
            if short_ma.iloc[-1] > long_ma.iloc[-1]:
                current_trend = "bullish"
            elif short_ma.iloc[-1] < long_ma.iloc[-1]:
                current_trend = "bearish"
            
            # Support and resistance levels
            recent_prices = prices.iloc[-window:]
            support_level = recent_prices.min()
            resistance_level = recent_prices.max()
            
            trend_analysis = {
                'current_trend': current_trend,
                'short_ma': short_ma.iloc[-1],
                'long_ma': long_ma.iloc[-1],
                'support_level': support_level,
                'resistance_level': resistance_level,
                'trend_strength': abs(short_ma.iloc[-1] - long_ma.iloc[-1]) / long_ma.iloc[-1]
            }
            
            return trend_analysis
            
        except Exception as e:
            warnings.warn(f"Could not identify trend patterns: {str(e)}")
            return {}
    
    def generate_text_report(self,
                           performance_report: Dict,
                           monthly_analysis: Dict,
                           risk_metrics: Optional[Dict] = None) -> str:
        """
        Generate a formatted text report.
        
        Args:
            performance_report: Performance analysis results
            monthly_analysis: Monthly analysis results
            risk_metrics: Risk metrics (optional)
            
        Returns:
            Formatted text report
        """
        try:
            report_lines = []
            
            # Header
            report_lines.append("JD Sports Stock Analysis Report")
            report_lines.append("=" * 35)
            report_lines.append(f"Generated by: {performance_report['metadata']['generated_by']}")
            report_lines.append(f"Date and Time: {performance_report['metadata']['generated_at']} UTC")
            report_lines.append("")
            
            # 2025 Performance
            if '2025_performance' in performance_report:
                perf = performance_report['2025_performance']
                report_lines.append("2025 Performance:")
                report_lines.append(f"Year Start: £{perf['year_start_price']:.2f}")
                report_lines.append(f"Current Price: £{perf['current_price']:.2f}")
                report_lines.append(f"YTD Return: {perf['ytd_return_pct']:.1f}% ({perf['ytd_return_status']})")
                report_lines.append("")
            
            # Predicted Performance
            if 'predicted_performance' in performance_report:
                pred = performance_report['predicted_performance']
                report_lines.append("Predicted Performance (H2 2025):")
                report_lines.append(f"Predicted EOY Price: £{pred['predicted_end_price']:.2f}")
                report_lines.append(f"Predicted H2 Return: {pred['predicted_return_pct']:.1f}% ({pred['predicted_return_status']})")
                report_lines.append("")
            
            # Monthly Analysis
            report_lines.append("Monthly Analysis 2025:")
            
            # Actual months
            for month_name, data in monthly_analysis['actual_months'].items():
                report_lines.append(f"\n{month_name} (Actual):")
                report_lines.append(f"Average: £{data['average_price']:.2f}")
                report_lines.append(f"Range: £{data['price_range'][0]:.2f} - £{data['price_range'][1]:.2f}")
                if data['volatility_pct'] is not None:
                    report_lines.append(f"Volatility: {data['volatility_pct']:.1f}%")
            
            # Predicted months
            for month_name, data in monthly_analysis['predicted_months'].items():
                report_lines.append(f"\n{month_name} (Predicted):")
                report_lines.append(f"Average: £{data['average_price']:.2f}")
                report_lines.append(f"Range: £{data['price_range'][0]:.2f} - £{data['price_range'][1]:.2f}")
                report_lines.append(f"Volatility: {data['volatility_pct']:.1f}%")
            
            # Risk Metrics (if provided)
            if risk_metrics:
                report_lines.append("\n\nRisk Analysis:")
                report_lines.append(f"Annual Volatility: {risk_metrics.get('volatility_annual', 0):.1%}")
                report_lines.append(f"Max Drawdown: {risk_metrics.get('max_drawdown', 0):.1%}")
                report_lines.append(f"Value at Risk (95%): {risk_metrics.get('var_95', 0):.1%}")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            raise Exception(f"Failed to generate text report: {str(e)}")
    
    def _classify_return(self, return_pct: float) -> str:
        """Classify return percentage as positive, negative, or neutral."""
        if return_pct > 5:
            return "strong positive"
        elif return_pct > 0:
            return "positive"
        elif return_pct < -5:
            return "strong negative"
        elif return_pct < 0:
            return "negative"
        else:
            return "neutral"
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown from price series."""
        try:
            peak = prices.expanding().max()
            drawdown = (prices - peak) / peak
            return drawdown.min()
        except Exception:
            return 0.0
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio for returns."""
        try:
            excess_returns = returns.mean() * 252 - risk_free_rate  # Annualized
            volatility = returns.std() * np.sqrt(252)  # Annualized
            return excess_returns / volatility if volatility > 0 else 0.0
        except Exception:
            return 0.0
    
    def export_analysis_to_csv(self, 
                              monthly_analysis: Dict,
                              file_path: str = "monthly_analysis.csv") -> None:
        """
        Export monthly analysis to CSV file.
        
        Args:
            monthly_analysis: Monthly analysis results
            file_path: Path to save CSV file
        """
        try:
            data_for_csv = []
            
            # Process actual months
            for month_name, data in monthly_analysis['actual_months'].items():
                data_for_csv.append({
                    'Month': month_name,
                    'Type': 'Actual',
                    'Average_Price': data['average_price'],
                    'Min_Price': data['price_range'][0],
                    'Max_Price': data['price_range'][1],
                    'Volatility_Pct': data['volatility_pct'],
                    'Trading_Days': data['trading_days']
                })
            
            # Process predicted months
            for month_name, data in monthly_analysis['predicted_months'].items():
                data_for_csv.append({
                    'Month': month_name,
                    'Type': 'Predicted',
                    'Average_Price': data['average_price'],
                    'Min_Price': data['price_range'][0],
                    'Max_Price': data['price_range'][1],
                    'Volatility_Pct': data['volatility_pct'],
                    'Trading_Days': data['trading_days']
                })
            
            df = pd.DataFrame(data_for_csv)
            df.to_csv(file_path, index=False)
            print(f"Monthly analysis exported to {file_path}")
            
        except Exception as e:
            warnings.warn(f"Could not export analysis to CSV: {str(e)}")