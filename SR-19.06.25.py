import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression

print("Stage 1: Import libraries - DONE")

# Step 1: Load the CSV data
csv_path = "/Users/remylieberman/Desktop/research/prototype1/jd_sports_stock_until_2024.csv"
data = pd.read_csv(csv_path)

# Step 2: Preprocess the data
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data.asfreq('B')  # Set frequency to business days
close_prices = data['Close'].fillna(method='ffill')  # Fill missing values

# Step 3: Train a time series model
model = ARIMA(close_prices, order=(5, 1, 0))  # ARIMA(p=5, d=1, q=0)
model_fit = model.fit()

# Step 4: Predict January 2025
start_date = '2025-01-01'
end_date = '2025-01-31'
predictions = model_fit.predict(start=start_date, end=end_date)

# Step 5: Plot the predictions
plt.figure(figsize=(10, 6))
plt.plot(predictions.index, predictions, label="Predicted Prices", color="blue")
plt.title("Stock Price Predictions for January 2025")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()

# Approximate prices from the graph (manually estimated for 3 days)
prices = [
    76.1, 75.6, 75.0, 74.8, 75.2, 75.0, 75.5, 75.3, 75.9, 75.6,
    76.0, 75.7, 76.8, 76.1, 76.4, 75.9, 75.3, 76.0, 75.2, 74.8,
    74.9, 75.1, 74.7, 74.8
]
print("Stage 2: Prices loaded - DONE")

# Create time steps for each price
X = np.arange(len(prices)).reshape(-1, 1)
y = np.array(prices)
print("Stage 3: Time steps created - DONE")

# Use last 6 points to fit a linear regression model
X_recent = X[-6:]
y_recent = y[-6:]
model = LinearRegression()
model.fit(X_recent, y_recent)
print("Stage 4: Linear regression model fitted - DONE")

# Predict the next day (let's say 8 more time steps)
future_steps = 8
X_future = np.arange(len(prices), len(prices) + future_steps).reshape(-1, 1)
y_future = model.predict(X_future)
print("Stage 5: Future predictions made - DONE")

# Combine original and predicted data for plotting
all_prices = np.concatenate((prices, y_future))
print("Stage 6: Data combined for plotting - DONE")

# Plotting
print("Stage 7: Plotting - Starting...")
plt.figure(figsize=(10, 4))
plt.plot(range(len(prices)), prices, label="Actual", color='red')
plt.plot(range(len(prices), len(prices) + future_steps), y_future, label="Predicted", linestyle='--', color='blue')
plt.axvline(x=len(prices)-1, color='gray', linestyle=':', label="Prediction Start")
plt.title("Stock Price: 3 Days + Predicted Next Day")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.ylim(74, 77.5)
plt.show()
print("Stage 8: Plot shown - DONE")