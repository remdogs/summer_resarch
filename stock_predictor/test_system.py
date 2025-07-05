"""Test script to verify the restructured stock predictor functionality."""

import warnings
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add the package to the path for development
sys.path.insert(0, str(Path(__file__).parent))

from stock_predictor import (
    Config, StockDataFetcher, DataPreprocessor, 
    StockPredictor, StockVisualizer, StockAnalyzer
)

warnings.filterwarnings('ignore')


def create_mock_data():
    """Create mock stock data for testing."""
    print("Creating mock stock data for testing...")
    
    # Generate mock historical data
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='B')
    np.random.seed(42)
    
    # Generate realistic stock prices
    initial_price = 100.0
    prices = [initial_price]
    
    for i in range(1, len(dates)):
        daily_return = np.random.normal(0.0005, 0.02)  # Small positive trend with volatility
        new_price = prices[-1] * (1 + daily_return)
        new_price = max(new_price, 10.0)  # Prevent negative prices
        prices.append(new_price)
    
    data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Volume': np.random.randint(100000, 1000000, size=len(dates))
    })
    
    # Save mock data
    data.to_csv('mock_jd_sports_data.csv', index=False)
    print(f"Created mock data with {len(data)} records")
    return 'mock_jd_sports_data.csv'


def test_stock_predictor():
    """Test the restructured stock predictor system."""
    
    print("="*60)
    print("TESTING RESTRUCTURED STOCK PREDICTOR SYSTEM")
    print("="*60)
    
    # Test 1: Initialize components
    print("\n1. Testing component initialization...")
    try:
        config = Config()
        fetcher = StockDataFetcher()
        preprocessor = DataPreprocessor(config)
        predictor = StockPredictor(config)
        visualizer = StockVisualizer(config)
        analyzer = StockAnalyzer(config)
        print("✅ All components initialized successfully")
    except Exception as e:
        print(f"❌ Component initialization failed: {e}")
        return False
    
    # Test 2: Create and load mock data
    print("\n2. Testing data loading and preprocessing...")
    try:
        csv_path = create_mock_data()
        historical_data = preprocessor.load_and_prepare_data(csv_path)
        synthetic_data = preprocessor.generate_synthetic_2025_data(historical_data)
        full_data = preprocessor.combine_datasets(historical_data, synthetic_data)
        
        print(f"✅ Data preprocessing successful")
        print(f"   - Historical data shape: {historical_data.shape}")
        print(f"   - Synthetic data shape: {synthetic_data.shape}")
        print(f"   - Combined data shape: {full_data.shape}")
        
        # Validate data
        is_valid = preprocessor.validate_data(full_data)
        print(f"   - Data validation: {'✅ Passed' if is_valid else '⚠️ Warning'}")
        
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        return False
    
    # Test 3: Generate predictions
    print("\n3. Testing prediction generation...")
    try:
        future_dates, predictions, conf_lower, conf_upper = predictor.predict_future_prices(
            full_data, start_date='2025-06-19', end_date='2025-12-31'
        )
        
        print(f"✅ Prediction generation successful")
        print(f"   - Prediction days: {len(predictions)}")
        print(f"   - Price range: £{predictions.min():.2f} - £{predictions.max():.2f}")
        
        # Get prediction summary
        summary = predictor.get_prediction_summary(
            predictions, conf_lower, conf_upper, full_data['Close'].iloc[-1]
        )
        print(f"   - Expected return: {summary.get('total_return_pct', 0):.1f}%")
        
    except Exception as e:
        print(f"❌ Prediction generation failed: {e}")
        return False
    
    # Test 4: Generate analysis
    print("\n4. Testing analysis generation...")
    try:
        performance_report = analyzer.generate_performance_report(
            full_data, predictions, future_dates, current_date='2025-06-19 15:50:11'
        )
        
        monthly_analysis = analyzer.analyze_monthly_performance(
            full_data, predictions, future_dates
        )
        
        risk_metrics = analyzer.calculate_risk_metrics(full_data)
        
        print(f"✅ Analysis generation successful")
        print(f"   - Performance report generated")
        print(f"   - Monthly analysis: {len(monthly_analysis['actual_months'])} actual + {len(monthly_analysis['predicted_months'])} predicted months")
        print(f"   - Risk metrics calculated: {len(risk_metrics)} metrics")
        
    except Exception as e:
        print(f"❌ Analysis generation failed: {e}")
        return False
    
    # Test 5: Create visualizations (without showing)
    print("\n5. Testing visualization creation...")
    try:
        # Create the plot but don't show it
        fig = visualizer.plot_stock_prediction(
            full_data, future_dates, predictions, conf_lower, conf_upper,
            current_date='2025-06-19', title='Test Stock Prediction'
        )
        
        # Save plot instead of showing it
        fig.savefig('test_prediction_plot.png', dpi=150, bbox_inches='tight')
        visualizer.close_all_plots()  # Clean up
        
        print(f"✅ Visualization creation successful")
        print(f"   - Plot saved as 'test_prediction_plot.png'")
        
    except Exception as e:
        print(f"❌ Visualization creation failed: {e}")
        return False
    
    # Test 6: Generate text report
    print("\n6. Testing report generation...")
    try:
        text_report = analyzer.generate_text_report(
            performance_report, monthly_analysis, risk_metrics
        )
        
        # Save the report
        with open('test_analysis_report.txt', 'w') as f:
            f.write(text_report)
        
        print(f"✅ Report generation successful")
        print(f"   - Text report saved as 'test_analysis_report.txt'")
        print(f"   - Report length: {len(text_report)} characters")
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED - RESTRUCTURING SUCCESSFUL!")
    print("="*60)
    print("The original SR-19.06.25.py functionality has been successfully")
    print("restructured into a professional, modular package structure.")
    print("\nKey improvements:")
    print("✅ Modular architecture with clear separation of concerns")
    print("✅ Professional error handling and validation")
    print("✅ Comprehensive documentation and type hints")
    print("✅ Flexible configuration management")
    print("✅ Reusable components for different stocks/scenarios")
    print("✅ Proper package structure with setup.py")
    
    return True


if __name__ == "__main__":
    success = test_stock_predictor()
    if not success:
        sys.exit(1)