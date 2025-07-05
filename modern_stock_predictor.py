"""
Modernized stock prediction script using the new StockPredictor class.
This script demonstrates how to use the refactored code with any stock.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from stock_data_fetcher import StockDataFetcher
from stock_predictor import StockPredictor


def predict_stock_with_fallback(ticker, data_path=None, **kwargs):
    """
    Predict stock prices with fallback to local data if network fails.
    
    Args:
        ticker: Stock ticker symbol
        data_path: Path to local CSV data (fallback)
        **kwargs: Additional arguments for StockPredictor
    """
    predictor = StockPredictor(ticker, **kwargs)
    
    try:
        # Try to load data from yfinance
        predictor.load_data(period="max")
        print(f"✓ Loaded data from yfinance for {ticker}")
    except Exception as e:
        print(f"Network fetch failed: {e}")
        
        if data_path:
            print(f"Falling back to local data: {data_path}")
            try:
                # Load local CSV data
                data = pd.read_csv(data_path)
                data['Date'] = pd.to_datetime(data['Date'])
                data.set_index('Date', inplace=True)
                
                predictor.data = data
                predictor.stock_info = {
                    'name': 'JD Sports Fashion Plc' if 'JD' in ticker else 'Unknown Company',
                    'currency': 'GBP' if '.L' in ticker else 'USD'
                }
                print(f"✓ Loaded local data for {ticker}")
            except Exception as local_e:
                raise Exception(f"Failed to load both network and local data: {local_e}")
        else:
            raise Exception("No fallback data available")
    
    return predictor


def main():
    """Main function demonstrating the flexible stock prediction system."""
    print("Stock Prediction System - Modernized Version")
    print("=" * 50)
    
    # Configuration
    ticker = "JD.L"  # JD Sports (can be changed to any ticker)
    fallback_data_path = "/home/runner/work/summer_resarch/summer_resarch/jd_sports_stock_until_2024.csv"
    
    # Prediction parameters
    prediction_config = {
        'confidence_interval': 0.70,
        'max_price_change': 0.15,
        'volatility_window': 20,
        'n_scenarios': 50
    }
    
    try:
        # Initialize predictor with configuration
        print(f"\nInitializing predictor for {ticker}...")
        predictor = predict_stock_with_fallback(
            ticker, 
            fallback_data_path, 
            **prediction_config
        )
        
        # Preprocess data
        print("\nPreprocessing data...")
        predictor.preprocess_data()
        
        # Generate predictions
        start_date = "2025-06-19"  # Current date from original script
        end_date = "2025-12-31"
        
        print(f"\nGenerating predictions from {start_date} to {end_date}...")
        predictions = predictor.generate_predictions(start_date, end_date)
        
        # Generate and display analysis report
        print("\nGenerating analysis report...")
        report = predictor.generate_analysis_report(predictions)
        print(report)
        
        # Create visualization
        print("\nGenerating visualization...")
        predictor.plot_predictions(
            predictions,
            show_confidence=True,
            show_historical=True,
            historical_days=180
        )
        
        # Monthly analysis (similar to original script)
        print("\nMonthly Analysis 2025:")
        print("-" * 30)
        
        # For predictions (second half of 2025)
        for month in range(7, 13):
            month_start = pd.Timestamp(f"2025-{month:02d}-01")
            month_end = pd.Timestamp(f"2025-{month:02d}-01") + pd.offsets.MonthEnd(1)
            
            # Filter predictions for this month
            month_mask = (predictions['dates'] >= month_start) & (predictions['dates'] <= month_end)
            month_predictions = predictions['predictions'][month_mask]
            
            if len(month_predictions) > 0:
                month_name = pd.Timestamp(f"2025-{month:02d}-01").strftime('%B')
                print(f"\n{month_name} (Predicted):")
                print(f"Average: £{month_predictions.mean():.2f}")
                print(f"Range: £{month_predictions.min():.2f} - £{month_predictions.max():.2f}")
                print(f"Volatility: {month_predictions.std()/month_predictions.mean()*100:.1f}%")
        
        # Demonstrate flexibility - show how to use with different stocks
        print(f"\n\nDemonstrating Flexibility:")
        print("-" * 40)
        print("The same code can be used with any stock ticker!")
        print("Examples:")
        print("- AAPL (Apple)")
        print("- MSFT (Microsoft)")
        print("- TSLA (Tesla)")
        print("- GOOGL (Google)")
        print("- JD.L (JD Sports - UK)")
        print("- BP.L (BP - UK)")
        
        print(f"\nTo use with a different stock, simply change:")
        print(f'ticker = "AAPL"  # Instead of "JD.L"')
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_multiple_stocks():
    """
    Demonstration function showing how to compare multiple stocks.
    Note: This will only work if network connectivity is available.
    """
    print("\n" + "=" * 60)
    print("MULTI-STOCK COMPARISON DEMO")
    print("=" * 60)
    print("(This demo requires internet connectivity)")
    
    tickers = ["AAPL", "MSFT", "TSLA"]
    results = {}
    
    for ticker in tickers:
        try:
            print(f"\nAnalyzing {ticker}...")
            predictor = StockPredictor(ticker, n_scenarios=10)  # Fewer scenarios for demo
            predictor.load_data(period="1y")
            predictor.preprocess_data()
            
            predictions = predictor.generate_predictions("2025-01-01", "2025-06-30")
            
            # Calculate summary stats
            predicted_return = (
                predictions['predictions'].iloc[-1] / 
                predictions['predictions'].iloc[0] - 1
            ) * 100
            
            results[ticker] = {
                'return': predicted_return,
                'volatility': predictions['predictions'].std() / predictions['predictions'].mean() * 100
            }
            
            print(f"✓ {ticker}: {predicted_return:.1f}% return, {results[ticker]['volatility']:.1f}% volatility")
            
        except Exception as e:
            print(f"✗ Failed to analyze {ticker}: {e}")
    
    if results:
        print(f"\nComparison Summary:")
        for ticker, stats in results.items():
            print(f"{ticker}: {stats['return']:.1f}% return, {stats['volatility']:.1f}% volatility")


if __name__ == "__main__":
    # Run main prediction
    main()
    
    # Optionally run multi-stock demo (commented out due to network issues in environment)
    # demonstrate_multiple_stocks()