"""
Example usage of the flexible StockPredictor class with different stocks.
This demonstrates how the refactored code can work with any stock dataset.
"""
from stock_predictor import StockPredictor
from stock_data_fetcher import StockDataFetcher
import pandas as pd


def example_basic_usage():
    """Basic example showing how to use StockPredictor with any stock."""
    print("Example 1: Basic Usage with Any Stock")
    print("=" * 50)
    
    # Example 1: Apple Inc. (AAPL)
    print("\nAnalyzing Apple (AAPL)...")
    try:
        predictor = StockPredictor(
            ticker="AAPL",
            confidence_interval=0.80,  # 80% confidence interval
            max_price_change=0.20,     # Allow up to 20% price change
            n_scenarios=30             # Use 30 scenarios for faster execution
        )
        
        # Load last 2 years of data
        predictor.load_data(period="2y")
        predictor.preprocess_data()
        
        # Generate 3-month predictions
        predictions = predictor.generate_predictions("2025-01-01", "2025-03-31")
        
        # Generate report
        report = predictor.generate_analysis_report(predictions)
        print(report[:500] + "...")  # Show first 500 characters
        
    except Exception as e:
        print(f"Network error (expected in this environment): {e}")


def example_custom_parameters():
    """Example showing different parameter configurations."""
    print("\n\nExample 2: Custom Parameter Configurations")
    print("=" * 50)
    
    configurations = [
        {
            "name": "Conservative",
            "confidence_interval": 0.90,
            "max_price_change": 0.10,
            "volatility_window": 30,
            "n_scenarios": 20
        },
        {
            "name": "Aggressive", 
            "confidence_interval": 0.60,
            "max_price_change": 0.25,
            "volatility_window": 10,
            "n_scenarios": 100
        }
    ]
    
    for config in configurations:
        print(f"\n{config['name']} Configuration:")
        print(f"- Confidence Interval: {config['confidence_interval']*100}%")
        print(f"- Max Price Change: ±{config['max_price_change']*100}%")
        print(f"- Volatility Window: {config['volatility_window']} days")
        print(f"- Number of Scenarios: {config['n_scenarios']}")


def example_with_local_data():
    """Example using local CSV data (works without internet)."""
    print("\n\nExample 3: Using Local Data (JD Sports)")
    print("=" * 50)
    
    try:
        # Load existing JD Sports data
        data_path = "/home/runner/work/summer_resarch/summer_resarch/jd_sports_stock_until_2024.csv"
        data = pd.read_csv(data_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        
        # Create predictor and set data manually
        predictor = StockPredictor("JD.L", confidence_interval=0.75, n_scenarios=25)
        predictor.data = data
        predictor.stock_info = {'name': 'JD Sports Fashion Plc', 'currency': 'GBP'}
        
        # Preprocess and predict
        predictor.preprocess_data()
        predictions = predictor.generate_predictions("2025-01-01", "2025-06-30")
        
        print(f"✓ Successfully analyzed JD Sports with local data")
        print(f"  Data range: {data.index[0]} to {data.index[-1]}")
        print(f"  Predicted 6-month return: {((predictions['predictions'].iloc[-1] / predictions['predictions'].iloc[0] - 1) * 100):.2f}%")
        print(f"  Average predicted price: £{predictions['predictions'].mean():.2f}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_comparison_framework():
    """Example showing how to set up stock comparisons."""
    print("\n\nExample 4: Stock Comparison Framework")
    print("=" * 50)
    
    print("To compare multiple stocks, you would:")
    print("1. Create StockPredictor instances for each stock")
    print("2. Generate predictions for the same time period")
    print("3. Compare metrics like predicted returns and volatility")
    
    # Mock comparison results (since we can't fetch real data)
    mock_results = {
        "AAPL": {"return": 12.5, "volatility": 18.2},
        "MSFT": {"return": 8.3, "volatility": 15.7},
        "TSLA": {"return": 25.4, "volatility": 35.8},
        "JD.L": {"return": 3.5, "volatility": 16.9}
    }
    
    print("\nExample comparison results:")
    print("Stock | Return | Volatility")
    print("-" * 30)
    for stock, metrics in mock_results.items():
        print(f"{stock:5} | {metrics['return']:6.1f}% | {metrics['volatility']:9.1f}%")


def example_different_time_periods():
    """Example showing different prediction time periods."""
    print("\n\nExample 5: Different Time Periods")
    print("=" * 50)
    
    time_periods = [
        ("Short-term", "2025-01-01", "2025-02-28"),   # 2 months
        ("Medium-term", "2025-01-01", "2025-06-30"),  # 6 months
        ("Long-term", "2025-01-01", "2025-12-31")     # 1 year
    ]
    
    print("The StockPredictor supports flexible time periods:")
    for name, start, end in time_periods:
        days = pd.date_range(start, end, freq='B')
        print(f"- {name}: {start} to {end} ({len(days)} business days)")


def main():
    """Run all examples."""
    print("StockPredictor Usage Examples")
    print("=" * 60)
    print("This demonstrates the flexibility of the refactored stock prediction system.")
    
    example_basic_usage()
    example_custom_parameters()
    example_with_local_data()
    example_comparison_framework()
    example_different_time_periods()
    
    print("\n\nKey Benefits of the Refactored System:")
    print("=" * 50)
    print("✓ Works with any stock ticker symbol")
    print("✓ Supports different data sources (yfinance)")
    print("✓ Configurable prediction parameters")
    print("✓ Flexible time periods")
    print("✓ Multiple stocks comparison capability")
    print("✓ Fallback to local data when network unavailable")
    print("✓ Reusable, modular design")
    print("✓ Maintains backward compatibility")


if __name__ == "__main__":
    main()