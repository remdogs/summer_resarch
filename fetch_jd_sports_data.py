import yfinance as yf
import pandas as pd
from datetime import datetime

ticker = "JD.L"
today = datetime.now().strftime("%Y-%m-%d")
df = yf.download(ticker, end=today)
df.reset_index(inplace=True)  # Moves Date to a column

# Only keep necessary columns
df = df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]

# Save cleanly to CSV
df.to_csv("jd_sports_stock_up_to_date.csv", index=False)

df = pd.read_csv("jd_sports_stock_up_to_date.csv", skiprows=1)
df.to_csv("jd_sports_stock_up_to_date_cleaned.csv", index=False)

print("Saved JD Sports data to jd_sports_stock_up_to_date.csv")