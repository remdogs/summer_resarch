"""Main demonstration script for the stock predictor package.

This script replicates the functionality of the original SR-19.06.25.py
using the new modular structure.
"""

import warnings
import os
import sys
from pathlib import Path

# Add the package to the path for development
sys.path.insert(0, str(Path(__file__).parent))

from stock_predictor import (
    Config, StockDataFetcher, DataPreprocessor, 
    StockPredictor, StockVisualizer, StockAnalyzer
)

warnings.filterwarnings('ignore')


def main():
    """Main function demonstrating the stock predictor system."""
    
    print("Stage 1: Import libraries - DONE")
    
    # Initialize configuration
    config = Config()
    
    # Initialize components
    fetcher = StockDataFetcher()
    preprocessor = DataPreprocessor(config)
    predictor = StockPredictor(config)
    visualizer = StockVisualizer(config)
    analyzer = StockAnalyzer(config)
    
    print("Stage 2: Initialize components - DONE")
    
    # Define data path (adjust to actual path)
    csv_path = "jd_sports_stock_until_2024.csv"
    
    # Check if data file exists, if not try to fetch it
    if not os.path.exists(csv_path):
        print("Data file not found. Attempting to fetch JD Sports data...")
        try:
            data = fetcher.fetch_jd_sports_data(save_path=csv_path)
            print("Successfully fetched JD Sports data.")
        except Exception as e:
            print(f"Failed to fetch data: {e}")
            print("Please ensure you have an internet connection or provide the data file.")
            return
    
    # Load and prepare data
    print("Stage 3: Loading and preparing data...")
    try:
        historical_data = preprocessor.load_and_prepare_data(csv_path)
        
        # Generate synthetic 2025 data
        synthetic_2025_data = preprocessor.generate_synthetic_2025_data(
            historical_data, end_date='2025-06-19'
        )
        
        # Combine datasets
        full_data = preprocessor.combine_datasets(historical_data, synthetic_2025_data)
        
        # Validate data
        if not preprocessor.validate_data(full_data):
            print("Warning: Data validation failed. Proceeding with caution.")
        
        print("Stage 3: Data preparation - DONE")
        
    except Exception as e:
        print(f"Error in data preparation: {e}")
        return
    
    # Generate predictions
    print("Stage 4: Generating predictions...")
    try:
        current_date = '2025-06-19 15:50:11'
        end_date = '2025-12-31'
        
        future_dates, predictions, confidence_lower, confidence_upper = predictor.predict_future_prices(
            full_data, start_date=current_date, end_date=end_date
        )
        
        print("Stage 4: Prediction generation - DONE")
        
    except Exception as e:
        print(f"Error in prediction generation: {e}")
        return
    
    # Create visualizations
    print("Stage 5: Creating visualizations...")
    try:
        # Main prediction plot
        fig = visualizer.plot_stock_prediction(
            full_data, future_dates, predictions, confidence_lower, confidence_upper,
            current_date=current_date, title='JD Sports Stock Price - 2025 Full Year'
        )
        
        print("Stage 5: Visualization creation - DONE")
        
    except Exception as e:
        print(f"Error in visualization creation: {e}")
        return
    
    # Generate analysis reports
    print("Stage 6: Generating analysis reports...")
    try:
        # Performance report
        performance_report = analyzer.generate_performance_report(
            full_data, predictions, future_dates, current_date=current_date
        )
        
        # Monthly analysis
        monthly_analysis = analyzer.analyze_monthly_performance(
            full_data, predictions, future_dates
        )
        
        # Risk metrics
        risk_metrics = analyzer.calculate_risk_metrics(full_data)
        
        # Generate text report
        text_report = analyzer.generate_text_report(
            performance_report, monthly_analysis, risk_metrics
        )
        
        # Print the report
        print("\n" + text_report)
        
        # Export monthly analysis to CSV
        analyzer.export_analysis_to_csv(monthly_analysis, "monthly_analysis_report.csv")
        
        print("\nStage 6: Analysis report generation - DONE")
        
    except Exception as e:
        print(f"Error in analysis generation: {e}")
        return
    
    # Show plots
    print("\nStage 7: Displaying visualizations...")
    try:
        visualizer.show_plot()
        print("Stage 7: Visualization display - DONE")
        
    except Exception as e:
        print(f"Error in displaying visualizations: {e}")
    
    # Print summary
    print("\n" + "="*50)
    print("STOCK PREDICTOR ANALYSIS COMPLETE")
    print("="*50)
    print("✅ Data fetched and preprocessed")
    print("✅ Predictions generated with confidence intervals")
    print("✅ Visualizations created")
    print("✅ Analysis reports generated")
    print("✅ Monthly analysis exported to CSV")
    
    # Print next steps suggestion (from original comment)
    print("\n" + "="*50)
    print("FUTURE ENHANCEMENTS")
    print("="*50)
    print("""
The next enhancement would be to integrate real-time news analysis:
1. Pull relevant news for each day/week/month
2. Use GPT to analyze news sentiment and impact
3. Determine how news will influence stock price
4. Incorporate news analysis into prediction model
5. This would make predictions influenced by both technical and fundamental factors
    """)


if __name__ == "__main__":
    main()