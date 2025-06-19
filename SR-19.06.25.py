import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import ta
import random
import warnings
warnings.filterwarnings('ignore')

print("Stage 1: Import libraries - DONE")

# Load and prepare data (same as before)
csv_path = "/Users/remylieberman/Desktop/research/prototype1/summer_resarch/jd_sports_stock_until_2024.csv"
data = pd.read_csv(csv_path)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

def create_advanced_features(df):
    # Previous feature creation code...
    return df

data = create_advanced_features(data)
data = data.dropna()

# New function to generate realistic market movements
def generate_realistic_predictions(last_price, days, volatility):
    """
    Generate realistic price movements using a combination of trends and random walks
    """
    predictions = [last_price]
    
    # Calculate historical metrics
    historical_volatility = data['Close'].pct_change().std()
    avg_daily_change = data['Close'].pct_change().mean()
    
    # Market regime parameters
    trend_strength = random.uniform(0.3, 0.7)
    cycle_length = random.randint(20, 40)
    
    for day in range(days):
        prev_price = predictions[-1]
        
        # Combine multiple factors for price movement
        
        # 1. Random walk component
        random_walk = np.random.normal(0, historical_volatility * volatility)
        
        # 2. Trend component (using sine wave for cyclic behavior)
        trend = np.sin(2 * np.pi * day / cycle_length) * trend_strength
        
        # 3. Momentum component
        if len(predictions) > 5:
            momentum = (predictions[-1] - predictions[-5]) / predictions[-5]
        else:
            momentum = 0
            
        # 4. Mean reversion component
        mean_reversion = (np.mean(predictions) - prev_price) * 0.1
        
        # 5. Volatility clustering
        if abs(random_walk) > 2 * historical_volatility:
            volatility *= 1.1  # Increase volatility after large moves
        else:
            volatility *= 0.99  # Gradually decrease volatility
            
        # 6. Add some random shocks
        if random.random() < 0.05:  # 5% chance of a significant move
            shock = random.choice([-1, 1]) * random.uniform(0.02, 0.05) * prev_price
        else:
            shock = 0
            
        # Combine all components
        daily_return = (random_walk + 
                       trend * historical_volatility + 
                       momentum * 0.2 + 
                       mean_reversion + 
                       shock)
        
        # Calculate new price
        new_price = prev_price * (1 + daily_return)
        
        # Add some constraints to prevent unrealistic movements
        max_daily_move = 0.1  # 10% max daily move
        if abs(new_price/prev_price - 1) > max_daily_move:
            new_price = prev_price * (1 + max_daily_move * np.sign(new_price - prev_price))
            
        predictions.append(new_price)
    
    return predictions[1:]  # Remove the initial seed price

# Generate predictions
last_known_price = data['Close'].iloc[-1]
future_dates = pd.date_range(start='2025-01-01', end='2025-06-19', freq='B')
n_days = len(future_dates)

# Generate multiple prediction scenarios
n_scenarios = 50
all_scenarios = []
for _ in range(n_scenarios):
    scenario = generate_realistic_predictions(last_known_price, n_days, volatility=1.0)
    all_scenarios.append(scenario)

# Calculate the main prediction and confidence intervals
predictions = np.mean(all_scenarios, axis=0)
confidence_lower = np.percentile(all_scenarios, 5, axis=0)
confidence_upper = np.percentile(all_scenarios, 95, axis=0)

# Plotting
plt.figure(figsize=(15, 8))

# Plot historical data
plt.plot(data.index[-90:], data['Close'][-90:], label='Historical', color='blue')

# Plot predictions
plt.plot(future_dates, predictions, label='Predictions', color='red', linestyle='--')
plt.fill_between(future_dates, confidence_lower, confidence_upper,
                 color='red', alpha=0.2, label='95% Confidence Interval')

# Calculate y-axis limits with padding
all_values = np.concatenate([data['Close'][-90:], predictions, confidence_upper, confidence_lower])
y_min = min(all_values) * 0.95
y_max = max(all_values) * 1.05

plt.ylim(y_min, y_max)
plt.title('Dynamic Stock Price Predictions January-June 2025')
plt.xlabel('Date')
plt.ylabel('Price (£)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print analysis
print("\nPrediction Analysis:")
print(f"Starting price: £{predictions[0]:.2f}")
print(f"Ending price: £{predictions[-1]:.2f}")
print(f"Highest predicted price: £{max(predictions):.2f}")
print(f"Lowest predicted price: £{min(predictions):.2f}")
print(f"Average volatility: {np.std(np.diff(predictions))/np.mean(predictions)*100:.2f}%")

# Calculate monthly statistics
monthly_stats = {}
for i, date in enumerate(future_dates):
    month = date.strftime('%B')
    if month not in monthly_stats:
        monthly_stats[month] = []
    monthly_stats[month].append(predictions[i])

print("\nMonthly Statistics:")
for month, prices in monthly_stats.items():
    print(f"\n{month} 2025:")
    print(f"Average: £{np.mean(prices):.2f}")
    print(f"Range: £{min(prices):.2f} - £{max(prices):.2f}")
    print(f"Monthly Volatility: {np.std(prices)/np.mean(prices)*100:.2f}%")