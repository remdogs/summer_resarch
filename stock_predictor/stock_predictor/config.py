"""Configuration management for the stock predictor package."""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration class for stock predictor settings."""
    
    # Data settings
    data_path: str = "data/"
    default_csv_path: str = "jd_sports_stock_until_2024.csv"
    
    # Prediction settings
    prediction_scenarios: int = 50
    max_daily_return: float = 0.015  # 1.5% max daily move
    max_price_change: float = 0.15   # 15% max price change from current
    transition_days: int = 10
    
    # Feature engineering settings
    volatility_window: int = 20
    sma_window: int = 20
    
    # Visualization settings
    figure_size: tuple = (15, 8)
    confidence_interval: int = 70  # Percentile for confidence bands
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config instance from dictionary."""
        return cls(**config_dict)
    
    @classmethod 
    def from_env(cls) -> 'Config':
        """Create Config instance from environment variables."""
        return cls(
            data_path=os.getenv('STOCK_DATA_PATH', cls.data_path),
            prediction_scenarios=int(os.getenv('PREDICTION_SCENARIOS', cls.prediction_scenarios)),
            max_daily_return=float(os.getenv('MAX_DAILY_RETURN', cls.max_daily_return)),
            max_price_change=float(os.getenv('MAX_PRICE_CHANGE', cls.max_price_change)),
            random_seed=int(os.getenv('RANDOM_SEED', cls.random_seed))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config instance to dictionary."""
        return {
            'data_path': self.data_path,
            'default_csv_path': self.default_csv_path,
            'prediction_scenarios': self.prediction_scenarios,
            'max_daily_return': self.max_daily_return,
            'max_price_change': self.max_price_change,
            'transition_days': self.transition_days,
            'volatility_window': self.volatility_window,
            'sma_window': self.sma_window,
            'figure_size': self.figure_size,
            'confidence_interval': self.confidence_interval,
            'random_seed': self.random_seed
        }