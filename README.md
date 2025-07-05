# Stock Prediction System - Modernized

This repository contains a flexible, modular stock prediction system that can work with any stock dataset, not just JD Sports. The system has been refactored to provide configurable parameters, multiple data sources, and reusable components.

## Key Features

✅ **Generic Stock Support**: Works with any stock ticker symbol (AAPL, MSFT, TSLA, JD.L, etc.)  
✅ **Configurable Parameters**: Customize confidence intervals, volatility windows, scenarios, etc.  
✅ **Multiple Data Sources**: Currently supports yfinance, with architecture for additional sources  
✅ **Flexible Time Periods**: Predict for any date range (days, months, years)  
✅ **Fallback Support**: Automatically falls back to local CSV data when network unavailable  
✅ **Modular Design**: Reusable components for data fetching, prediction, and visualization  
✅ **Backward Compatibility**: Original JD Sports functionality preserved  

## Project Structure

```
├── stock_data_fetcher.py      # Generic data fetching module
├── stock_predictor.py         # Main StockPredictor class
├── modern_stock_predictor.py  # Modernized main script
├── usage_examples.py          # Usage examples for different scenarios
├── SR-19.06.25.py            # Original JD Sports script (preserved)
├── fetch_jd_sports_data.py   # Original data fetcher (preserved)
└── jd_sports_stock_*.csv     # Sample data files
```

## Quick Start

### Basic Usage

```python
from stock_predictor import StockPredictor

# Create predictor for any stock
predictor = StockPredictor("AAPL")  # Apple
# predictor = StockPredictor("MSFT")  # Microsoft  
# predictor = StockPredictor("JD.L")  # JD Sports (UK)

# Load data and generate predictions
predictor.load_data(period="2y")
predictor.preprocess_data()
predictions = predictor.generate_predictions("2025-01-01", "2025-12-31")

# Generate analysis report
report = predictor.generate_analysis_report(predictions)
print(report)

# Create visualization
predictor.plot_predictions(predictions)
```

### Custom Configuration

```python
# Conservative configuration
predictor = StockPredictor(
    ticker="AAPL",
    confidence_interval=0.90,  # 90% confidence
    max_price_change=0.10,     # ±10% max change
    volatility_window=30,      # 30-day volatility window
    n_scenarios=50            # 50 Monte Carlo scenarios
)

# Aggressive configuration
predictor = StockPredictor(
    ticker="TSLA", 
    confidence_interval=0.60,  # 60% confidence
    max_price_change=0.25,     # ±25% max change
    volatility_window=10,      # 10-day volatility window
    n_scenarios=100           # 100 scenarios
)
```

### Multiple Stock Comparison

```python
tickers = ["AAPL", "MSFT", "TSLA", "JD.L"]
results = {}

for ticker in tickers:
    predictor = StockPredictor(ticker)
    predictor.load_data(period="1y")
    predictor.preprocess_data()
    predictions = predictor.generate_predictions("2025-01-01", "2025-06-30")
    
    # Calculate metrics
    predicted_return = (predictions['predictions'].iloc[-1] / 
                       predictions['predictions'].iloc[0] - 1) * 100
    results[ticker] = predicted_return

print("Predicted 6-month returns:")
for ticker, return_pct in results.items():
    print(f"{ticker}: {return_pct:.1f}%")
```

## API Reference

### StockPredictor Class

```python
class StockPredictor:
    def __init__(
        self,
        ticker: str,
        confidence_interval: float = 0.70,
        max_price_change: float = 0.15,
        volatility_window: int = 20,
        n_scenarios: int = 50
    )
```

**Parameters:**
- `ticker`: Stock ticker symbol (e.g., 'AAPL', 'JD.L')
- `confidence_interval`: Confidence level for predictions (0.0-1.0)
- `max_price_change`: Maximum allowed price change as fraction (e.g., 0.15 = 15%)
- `volatility_window`: Window size for volatility calculations
- `n_scenarios`: Number of Monte Carlo scenarios

### Key Methods

- `load_data(start_date, end_date, period)`: Load historical data
- `preprocess_data()`: Prepare data with technical indicators
- `generate_predictions(start_date, end_date)`: Generate price predictions
- `plot_predictions(predictions)`: Create visualization
- `generate_analysis_report(predictions)`: Generate detailed report

### StockDataFetcher Class

```python
fetcher = StockDataFetcher()
data = fetcher.fetch_stock_data("AAPL", period="1y")
```

## Running the Examples

1. **Main modernized script:**
   ```bash
   python modern_stock_predictor.py
   ```

2. **Usage examples:**
   ```bash
   python usage_examples.py
   ```

3. **Original JD Sports script (preserved):**
   ```bash
   python SR-19.06.25.py
   ```

## Dependencies

```bash
pip install pandas numpy matplotlib yfinance ta
```

## Changes from Original

### Original System (JD Sports only)
- Hardcoded JD Sports ticker ("JD.L")
- Fixed CSV file path
- Embedded prediction logic in main script
- Single stock analysis only

### Modernized System (Any Stock)
- ✅ Configurable ticker symbol
- ✅ Generic data fetching from multiple sources
- ✅ Modular, reusable StockPredictor class
- ✅ Flexible configuration parameters
- ✅ Multi-stock comparison capability
- ✅ Fallback to local data when needed
- ✅ Maintained backward compatibility

## Example Outputs

The system generates:
- **Predictions**: Daily price forecasts with confidence intervals
- **Visualizations**: Interactive plots with historical data and predictions
- **Analysis Reports**: Detailed statistics, risk assessment, and forecasts
- **Monthly Breakdown**: Month-by-month analysis of predicted performance

## Network Considerations

The system is designed to work both online and offline:
- **Online**: Fetches real-time data from yfinance for any stock
- **Offline**: Falls back to local CSV data when network unavailable
- **Hybrid**: Can mix live data with local historical data

## Future Enhancements

Potential improvements mentioned in the original code:
- Integration with news sentiment analysis
- Additional data sources (Alpha Vantage, Quandl, etc.)
- Machine learning prediction models
- Real-time data streaming
- Web interface for easier interaction