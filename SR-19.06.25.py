<<<<<<< HEAD
import sys
import argparse
import subprocess
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import ta

def get_user_inputs():
    parser = argparse.ArgumentParser(description="Stock Price Predictor (with scenario display and honest uncertainty)")
    parser.add_argument('--csv', type=str, help='Path to the CSV file (from yfinance)', required=True)
    parser.add_argument('--years', type=float, help='Years to predict into the future (e.g., 1, 2.5)', required=True)
    parser.add_argument('--scenarios', type=int, default=20, help='Number of scenario paths to display')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (for reproducibility)')
    args = parser.parse_args()
    return args.csv, args.years, args.scenarios, args.seed

def run_fetcher_if_needed(csv_file):
    if "JD" in csv_file.upper():
        fetcher_script = "fetch_jd_sports_data.py"
        if os.path.exists(fetcher_script):
            print(f"Updating JD stock data using {fetcher_script}...")
            subprocess.run([sys.executable, fetcher_script], check=True)
        else:
            print(f"Fetcher script {fetcher_script} not found. Proceeding with existing CSV.")

def create_features(df):
    df['Returns'] = df['Close'].pct_change()
    df['Daily_Volatility'] = df['Returns'].rolling(window=20).std()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    return df

def generate_predictions(last_price, n_days, historical_data, random_state):
    predictions = [last_price]
    # Use global mean/volatility for all data
    drift = historical_data['Returns'].mean()
    volatility = historical_data['Returns'].std()
    for day in range(n_days):
        prev_price = predictions[-1]
        # Only essential stochasticity for uncertainty
        daily_return = drift + random_state.normal(0, volatility)
        new_price = prev_price * (1 + daily_return)
        predictions.append(new_price)
    return predictions[1:]

def backtest(historical_data, n_days, num_scenarios, seed):
    # Go back through history and test prediction accuracy for each horizon
    min_backtest_year = 2010
    results = []
    for i, current_date in enumerate(historical_data.index[:-n_days]):
        if historical_data.index[i].year < min_backtest_year:
            continue
        last_price = historical_data['Close'].iloc[i]
        true_series = historical_data['Close'].iloc[i+1:i+1+n_days].values
        scenario_errors = []
        for s in range(num_scenarios):
            scenario = generate_predictions(last_price, n_days, historical_data.iloc[:i+1], np.random.RandomState(seed+s))
            scenario = np.array(scenario)
            # Calculate relative error at each horizon
            if len(true_series) < len(scenario):  # Out of data (end of file)
                continue
            error = np.abs(scenario - true_series) / (true_series + 1e-9)
            # dont really know what this line above but without the error parameters code does not work
            scenario_errors.append(error)
        if scenario_errors:
            scenario_errors = np.stack(scenario_errors)
            mean_error = scenario_errors.mean(axis=0)
            results.append(mean_error)
    # Average over all backtests
    if results:
        results = np.stack(results)
        horizon_mse = results.mean(axis=0)
        return horizon_mse
    else:
        return None

def get_confidence_horizon(horizon_mse, future_dates, threshold=0.30):
    """Return the first date where mean relative error exceeds threshold (30% by default)."""
    if horizon_mse is None:
        return None
    for i, mse in enumerate(horizon_mse):
        if mse > threshold:
            return future_dates[i]
    return None

if __name__ == "__main__":
    csv_file, years_to_predict, n_scenarios_to_show, seed = get_user_inputs()
    run_fetcher_if_needed(csv_file)
    df = pd.read_csv(csv_file)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    else:
        print("Error: CSV must have a 'Date' column.")
        sys.exit(1)
    df = pd.read_csv(csv_file)

    # Clean the DataFrame
    df = df[pd.to_numeric(df['Close'], errors='coerce').notnull()]
    for col in ['Close', 'High', 'Low', 'Open']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    df = create_features(df)
    df = df.dropna()

    last_date = df.index[-1]
    n_days = int(252 * years_to_predict)
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=n_days, freq='B')
    last_known_price = df['Close'].iloc[-1]

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Generate and show scenario paths
    all_scenarios = []
    print(f"Generating {n_scenarios_to_show} scenario paths...")
    for i in range(n_scenarios_to_show):
        scenario = generate_predictions(last_known_price, n_days, df, np.random.RandomState(seed+i))
        all_scenarios.append(scenario)
    all_scenarios = np.array(all_scenarios)

    # Calculate mean and confidence intervals for background
    n_background_scenarios = 100
    all_for_stats = []
    for i in range(n_background_scenarios):
        scenario = generate_predictions(last_known_price, n_days, df, np.random.RandomState(seed+1000+i))
        all_for_stats.append(scenario)
    all_for_stats = np.array(all_for_stats)
    mean_pred = np.mean(all_for_stats, axis=0)
    conf_int_95 = np.percentile(all_for_stats, [2.5, 97.5], axis=0)

    # Backtest to get honest uncertainty at each time horizon
    print("Running backtest for honest uncertainty...")
    horizon_mse = backtest(df, n_days, 10, seed)
    confidence_horizon = get_confidence_horizon(horizon_mse, future_dates, threshold=0.30)

    # Plot
    plt.figure(figsize=(15, 8))
    plt.plot(df.index, df['Close'], label='Historical', color='blue')
    # Plot individual scenario paths
    for i, scenario in enumerate(all_scenarios):
        plt.plot(future_dates, scenario, color='orange', alpha=0.5 if n_scenarios_to_show > 3 else 0.8, linewidth=1, label='Scenario' if i == 0 else None)
    # Plot mean and confidence interval as background
    plt.plot(future_dates, mean_pred, color='red', linestyle='--', label='Mean Prediction')
    plt.fill_between(future_dates, conf_int_95[0], conf_int_95[1], color='red', alpha=0.15, label='95% Confidence Interval')

    # Show honest uncertainty window
    if confidence_horizon is not None:
        plt.text(
            0.99, 0.02,
            f"Note: Based on backtesting, predictions after {confidence_horizon.year} are highly uncertain.",
            transform=plt.gca().transAxes,
            ha='right', va='bottom',
            color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='red')
        )
        plt.axvline(confidence_horizon, color='red', linestyle=':', alpha=0.6)
    else:
        plt.text(
            0.99, 0.02,
            f"Backtesting could not determine a confident forecast horizon.",
            transform=plt.gca().transAxes,
            ha='right', va='bottom',
            color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='red')
        )

    plt.title(f'Stock Price Predictions ({last_date.year + 1}-{last_date.year + int(years_to_predict)})')
    plt.xlabel('Date')
    plt.ylabel('Price (£)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print("\nPrediction Analysis:")
    print(f"Starting price: £{mean_pred[0]:.2f}")
    print(f"Ending price (mean): £{mean_pred[-1]:.2f}")
    if confidence_horizon is not None:
        print(f"Model is historically confident up to: {confidence_horizon.strftime('%Y-%m-%d')}")
        print(f"(Predictions after this are likely to be less reliable: >30% mean relative error in backtests)")
    else:
        print("Confidence horizon could not be determined (insufficient backtest data).")
=======
>>>>>>> a9fc37d1ddf5e0259193470d4155f197493927c1
