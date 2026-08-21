# Import FRED data reader
from pandas_datareader import data as web

# Import PCA
from sklearn.decomposition import PCA

# Import pandas
import pandas as pd

# Import date tools
import datetime


# Set the start date
start = datetime.datetime(2010, 1, 1)

# Set the end date to today
end = datetime.datetime.today()


# Define the FRED Treasury yield series
series = [
    "DGS3MO",
    "DGS6MO",
    "DGS1",
    "DGS2",
    "DGS3",
    "DGS5",
    "DGS7",
    "DGS10",
    "DGS20",
    "DGS30"
]


# Download the yield data from FRED
yields = web.DataReader(series, "fred", start, end)


# Rename the columns by maturity
yields.columns = [
    "3M",
    "6M",
    "1Y",
    "2Y",
    "3Y",
    "5Y",
    "7Y",
    "10Y",
    "20Y",
    "30Y"
]


# Calculate common curve spreads
yields["2s10s"] = yields["10Y"] - yields["2Y"]
yields["5s30s"] = yields["30Y"] - yields["5Y"]
yields["2s30s"] = yields["30Y"] - yields["2Y"]


# Remove rows containing missing values
yields = yields.dropna()


# Define the maturities used in PCA
maturities = [
    "3M", "6M", "1Y", "2Y", "3Y",
    "5Y", "7Y", "10Y", "20Y", "30Y"
]


# Calculate daily yield changes in basis points
changes = yields[maturities].diff().dropna() * 100


# Use approximately two years of data for each PCA estimation
window = 504


# Create an empty DataFrame to store PCA residuals
residuals = pd.DataFrame(
    index=changes.index,
    columns=maturities,
    dtype=float
)


# Run rolling PCA
for i in range(window, len(changes)):

    # Select the previous 504 trading days
    train = changes.iloc[i-window:i]

    # Select the current day's yield changes
    today = changes.iloc[[i]]

    # Create a PCA model with three factors
    # Which combinations of maturities explain most of the variation in this dataset?
    pca = PCA(n_components=3)

    # Estimate the PCA factors
    pca.fit(train)

    # Reconstruct today's yield changes using the three factors
    expected = pca.inverse_transform(
        pca.transform(today)
    )

    # Calculate actual minus PCA-reconstructed changes
    residuals.iloc[i] = (
        today.values[0] - expected[0]
    )


# Remove rows where PCA residuals were unavailable
residuals = residuals.dropna()


# Use the previous 252 trading days to define the normal residual range
z_window = 252


# Calculate rolling mean using only previous observations
mean = residuals.rolling(z_window).mean().shift(1)

# Calculate rolling standard deviation using only previous observations
std = residuals.rolling(z_window).std().shift(1)

# Calculate rolling z-scores
z_scores = (residuals - mean) / std


# Set the trading threshold
threshold = 1.5


# Create signals for every maturity
signals = pd.DataFrame(
    0,
    index=z_scores.index,
    columns=maturities
)


# Positive residual: expect reversal downward
signals[z_scores > threshold] = -1 # Short position

# Negative residual: expect reversal upward
signals[z_scores < -threshold] = 1 # Long position


# Move next day's residuals onto today's row
next_residuals = residuals.shift(-1)


# Calculate signal performance for every maturity
strategy = signals * next_residuals


# Keep only observations where a trade occurred
trade_results = strategy.where(signals != 0).stack()


# Display current z-scores
print("\nCurrent z-scores:")
print(z_scores.iloc[-1])


# Display current signals
print("\nCurrent signals:")
print(signals.iloc[-1])


# Display aggregate backtest results
print("\nBacktest results:")
print("Number of trades:", len(trade_results))
print("Average result:", trade_results.mean(), "bp")
print("Total result:", trade_results.sum(), "bp")
print("Win rate:", (trade_results > 0).mean())


# Display results by maturity
print("\nResults by maturity:")

for maturity in maturities:

    maturity_results = strategy[maturity][
        signals[maturity] != 0
    ].dropna()

    print(
        maturity,
        "| Trades:", len(maturity_results),
        "| Average:", round(maturity_results.mean(), 3), "bp",
        "| Total:", round(maturity_results.sum(), 3), "bp",
        "| Win rate:", round((maturity_results > 0).mean(), 3)
    )