import pandas as pd

# Load with date as index and make sure it's datetime
df = pd.read_csv("jd_sports_stock_until_2024.csv", index_col=0, parse_dates=True)
dates = df.index

# Show first 5 dates
print("First 5 dates:", dates[:5])

# Calculate increments, now this will work
date_diffs = dates.to_series().diff().dropna()
print("Frequency of increments:\n", date_diffs.value_counts())

if not date_diffs.empty:
    most_common_increment = date_diffs.mode()[0]
    print("\nMost common increment:", most_common_increment)
else:
    print("Not enough data to determine increments.")