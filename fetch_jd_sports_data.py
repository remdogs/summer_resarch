import yfinance as yf
import pandas as pd
import csv

# Define the stock ticker for JD Sports (London Stock Exchange: JD.L)
ticker_symbol = "JD.L"

# Set the end date to the last day of 2024
end_date = "2024-12-31"

# Download historical daily data from the earliest available date up to the end of 2024
jd_sports_data = yf.download(ticker_symbol, end=end_date, progress=True)

# Save the data to a CSV file for further analysis or use
jd_sports_data.to_csv("jd_sports_stock_until_2024.csv")

# Dynamically handle column renaming and ensure CSV structure is consistent
df = pd.read_csv("jd_sports_stock_until_2024.csv")
if "Date" not in df.columns:
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)  # Rename the first column to 'Date' if needed

# Remove unnecessary rows (lines 2 and 3)
df = df[~df["Date"].str.contains("Ticker|Date", na=False)]

df.to_csv("jd_sports_stock_until_2024.csv", index=False)  # Save the cleaned data

print("Downloaded and cleaned JD Sports share price data up to", end_date)
print(df.head())

input_file = 'jd_sports_stock_until_2024.csv'
output_file = 'jd_sports_stock_cleaned.csv'

# Open the input file and create a new output file
with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # Iterate through the rows and skip lines 2 and 3 (index 1 and 2)
    for i, row in enumerate(reader):
        if i not in (1, 2):  # Skip lines 2 and 3
            writer.writerow(row)

print(f"Lines 2 and 3 removed. Cleaned file saved as {output_file}.")