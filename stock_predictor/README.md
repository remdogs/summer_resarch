# Stock Predictor

A professional stock price prediction system with data fetching, preprocessing, modeling, and visualization capabilities.

## Features

- **Data Fetching**: Retrieve stock data from Yahoo Finance for any ticker symbol
- **Data Preprocessing**: Clean and prepare stock data with technical indicators
- **Prediction Models**: Generate statistical predictions with confidence intervals
- **Visualization**: Create comprehensive charts and plots for analysis
- **Analysis Tools**: Generate detailed performance reports and risk metrics
- **Configuration Management**: Flexible configuration system for customization

## Installation

### From Source

1. Clone the repository:
```bash
git clone https://github.com/remdogs/summer_resarch.git
cd summer_resarch/stock_predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
from stock_predictor import (
    Config, StockDataFetcher, DataPreprocessor, 
    StockPredictor, StockVisualizer, StockAnalyzer
)

# Initialize components
config = Config()
fetcher = StockDataFetcher()
preprocessor = DataPreprocessor(config)
predictor = StockPredictor(config)
visualizer = StockVisualizer(config)
analyzer = StockAnalyzer(config)

# Fetch stock data
data = fetcher.fetch_stock_data("JD.L", save_path="jd_sports_data.csv")

# Preprocess data
processed_data = preprocessor.load_and_prepare_data("jd_sports_data.csv")
synthetic_data = preprocessor.generate_synthetic_2025_data(processed_data)
full_data = preprocessor.combine_datasets(processed_data, synthetic_data)

# Generate predictions
future_dates, predictions, conf_lower, conf_upper = predictor.predict_future_prices(
    full_data, start_date='2025-06-19', end_date='2025-12-31'
)

# Create visualizations
fig = visualizer.plot_stock_prediction(
    full_data, future_dates, predictions, conf_lower, conf_upper,
    current_date='2025-06-19', title='JD Sports Stock Prediction'
)

# Generate analysis report
performance_report = analyzer.generate_performance_report(
    full_data, predictions, future_dates, current_date='2025-06-19 15:50:11'
)
monthly_analysis = analyzer.analyze_monthly_performance(
    full_data, predictions, future_dates
)

# Print text report
text_report = analyzer.generate_text_report(performance_report, monthly_analysis)
print(text_report)

# Show plots
visualizer.show_plot()
```

### Supported Stock Tickers

The system supports any stock ticker available on Yahoo Finance:
- **JD.L**: JD Sports (London Stock Exchange)
- **V**: Visa Inc.
- **AAPL**: Apple Inc.
- **MSFT**: Microsoft Corporation
- And many more...

## Configuration

The system uses a flexible configuration system. You can customize settings by:

### Using the Config class:
```python
from stock_predictor import Config

config = Config(
    prediction_scenarios=100,  # Number of prediction scenarios
    max_daily_return=0.02,     # Maximum daily return (2%)
    confidence_interval=80,    # Confidence interval for predictions
    random_seed=42            # Random seed for reproducibility
)
```

### Using environment variables:
```bash
export PREDICTION_SCENARIOS=100
export MAX_DAILY_RETURN=0.02
export CONFIDENCE_INTERVAL=80
```

## Package Structure

```
stock_predictor/
├── README.md
├── requirements.txt
├── setup.py
├── stock_predictor/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetcher.py         # Data fetching from Yahoo Finance
│   │   └── preprocessor.py    # Data preprocessing and feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   └── predictor.py       # Stock prediction algorithms
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── analysis.py        # Analysis and reporting tools
│   │   └── visualization.py   # Plotting and visualization
└── tests/
    └── __init__.py
```

## Modules

### Data Module (`stock_predictor.data`)

- **StockDataFetcher**: Fetch stock data from Yahoo Finance
- **DataPreprocessor**: Clean and prepare data with technical indicators

### Models Module (`stock_predictor.models`)

- **StockPredictor**: Generate price predictions with statistical modeling

### Utils Module (`stock_predictor.utils`)

- **StockVisualizer**: Create charts and visualizations
- **StockAnalyzer**: Generate reports and perform analysis

## Dependencies

- pandas >= 2.0.0
- numpy >= 1.20.0
- matplotlib >= 3.5.0
- yfinance >= 0.2.0
- ta >= 0.10.0

## Contributing

This is a research project. Feel free to submit issues and enhancement requests.

## License

This project is licensed under the MIT License.

## Disclaimer

This software is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Stock predictions are inherently uncertain and past performance does not guarantee future results.

## Author

- **remdogs** - Initial work and research

## Acknowledgments

- Yahoo Finance for providing stock data through the yfinance library
- The open-source Python community for the excellent libraries used in this project