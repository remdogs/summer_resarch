"""Visualization utilities for stock data and predictions."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import warnings
from ..config import Config


class StockVisualizer:
    """Handles visualization of stock data and predictions."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the stock visualizer.
        
        Args:
            config: Configuration object with visualization settings
        """
        self.config = config or Config()
        
    def plot_stock_prediction(self,
                            historical_data: pd.DataFrame,
                            future_dates: pd.DatetimeIndex,
                            predictions: np.ndarray,
                            confidence_lower: np.ndarray,
                            confidence_upper: np.ndarray,
                            current_date: Optional[str] = None,
                            title: str = "Stock Price Prediction",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive stock prediction plot.
        
        Args:
            historical_data: Historical stock data
            future_dates: Dates for predictions
            predictions: Predicted prices
            confidence_lower: Lower confidence bound
            confidence_upper: Upper confidence bound
            current_date: Current date marker
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        try:
            # Create figure
            fig = plt.figure(figsize=self.config.figure_size)
            
            # Plot 2025 data (if available)
            data_2025 = historical_data[historical_data.index >= '2025-01-01']
            if not data_2025.empty:
                plt.plot(data_2025.index, data_2025['Close'], 
                        label='2025 Actual Data', color='blue', linewidth=2)
            
            # Plot predictions
            plt.plot(future_dates, predictions, 
                    label='Predictions', color='red', linestyle='--', linewidth=2)
            
            # Plot confidence intervals
            plt.fill_between(future_dates, confidence_lower, confidence_upper,
                           color='red', alpha=0.2, 
                           label=f'{self.config.confidence_interval}% Confidence Interval')
            
            # Set y-axis limits for better visualization
            all_values = np.concatenate([
                data_2025['Close'].values if not data_2025.empty else [],
                predictions, confidence_upper, confidence_lower
            ])
            
            if len(all_values) > 0:
                y_min = max(min(all_values) * 0.98, 0)
                y_max = max(all_values) * 1.02
                plt.ylim(y_min, y_max)
            
            # Add current date marker if provided
            if current_date:
                current_date_ts = pd.Timestamp(current_date)
                plt.axvline(x=current_date_ts, color='green', linestyle=':', 
                          linewidth=2, label='Current Time')
            
            # Formatting
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Price (£)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            # Format y-axis as currency
            plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('£%.2f'))
            
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            raise Exception(f"Failed to create stock prediction plot: {str(e)}")
    
    def plot_historical_analysis(self,
                               data: pd.DataFrame,
                               title: str = "Historical Stock Analysis",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create plots for historical stock analysis.
        
        Args:
            data: Historical stock data
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Price plot
            axes[0, 0].plot(data.index, data['Close'], color='blue', linewidth=1)
            axes[0, 0].set_title('Stock Price Over Time')
            axes[0, 0].set_ylabel('Price (£)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Volume plot (if available)
            if 'Volume' in data.columns:
                axes[0, 1].plot(data.index, data['Volume'], color='orange', linewidth=1)
                axes[0, 1].set_title('Trading Volume')
                axes[0, 1].set_ylabel('Volume')
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(0.5, 0.5, 'Volume data\nnot available', 
                              ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Trading Volume')
            
            # Returns plot
            if 'Returns' in data.columns:
                axes[1, 0].plot(data.index, data['Returns'], color='green', linewidth=1)
                axes[1, 0].set_title('Daily Returns')
                axes[1, 0].set_ylabel('Returns')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Volatility plot
            if 'Daily_Volatility' in data.columns:
                axes[1, 1].plot(data.index, data['Daily_Volatility'], color='red', linewidth=1)
                axes[1, 1].set_title('Rolling Volatility')
                axes[1, 1].set_ylabel('Volatility')
                axes[1, 1].grid(True, alpha=0.3)
            
            # Format x-axes
            for ax in axes.flat:
                ax.tick_params(axis='x', rotation=45)
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Historical analysis plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            raise Exception(f"Failed to create historical analysis plot: {str(e)}")
    
    def plot_prediction_scenarios(self,
                                future_dates: pd.DatetimeIndex,
                                all_scenarios: np.ndarray,
                                mean_prediction: np.ndarray,
                                title: str = "Prediction Scenarios",
                                max_scenarios_display: int = 20,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot multiple prediction scenarios to show uncertainty.
        
        Args:
            future_dates: Dates for predictions
            all_scenarios: Array of all prediction scenarios
            mean_prediction: Mean prediction across scenarios
            title: Plot title
            max_scenarios_display: Maximum number of scenarios to display
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        try:
            fig = plt.figure(figsize=self.config.figure_size)
            
            # Plot subset of scenarios
            n_scenarios_display = min(max_scenarios_display, len(all_scenarios))
            indices = np.linspace(0, len(all_scenarios)-1, n_scenarios_display, dtype=int)
            
            for i in indices:
                plt.plot(future_dates, all_scenarios[i], 
                        color='lightblue', alpha=0.3, linewidth=0.5)
            
            # Plot mean prediction
            plt.plot(future_dates, mean_prediction, 
                    color='red', linewidth=2, label='Mean Prediction')
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Price (£)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            # Format y-axis as currency
            plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('£%.2f'))
            
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Prediction scenarios plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            raise Exception(f"Failed to create prediction scenarios plot: {str(e)}")
    
    def plot_comparison(self,
                       data1: pd.DataFrame,
                       data2: pd.DataFrame,
                       labels: Tuple[str, str] = ("Dataset 1", "Dataset 2"),
                       title: str = "Stock Price Comparison",
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare two stock datasets side by side.
        
        Args:
            data1: First dataset
            data2: Second dataset
            labels: Labels for the datasets
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        try:
            fig = plt.figure(figsize=self.config.figure_size)
            
            # Plot both datasets
            plt.plot(data1.index, data1['Close'], 
                    label=labels[0], linewidth=2)
            plt.plot(data2.index, data2['Close'], 
                    label=labels[1], linewidth=2)
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Price (£)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            # Format y-axis as currency
            plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('£%.2f'))
            
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Comparison plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            raise Exception(f"Failed to create comparison plot: {str(e)}")
    
    def show_plot(self) -> None:
        """Display all created plots."""
        try:
            plt.show()
        except Exception as e:
            warnings.warn(f"Could not display plots: {str(e)}")
    
    def close_all_plots(self) -> None:
        """Close all matplotlib figures to free memory."""
        plt.close('all')
        
    def set_style(self, style: str = 'default') -> None:
        """
        Set matplotlib style for plots.
        
        Args:
            style: Matplotlib style name
        """
        try:
            plt.style.use(style)
            print(f"Set plot style to: {style}")
        except Exception as e:
            warnings.warn(f"Could not set style '{style}': {str(e)}")
            
    def get_available_styles(self) -> list:
        """
        Get list of available matplotlib styles.
        
        Returns:
            List of available style names
        """
        return plt.style.available