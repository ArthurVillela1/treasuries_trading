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


# Calculate the 2Y-10Y curve spread
yields["2s10s"] = yields["10Y"] - yields["2Y"]

# Calculate the 5Y-30Y curve spread
yields["5s30s"] = yields["30Y"] - yields["5Y"]

# Calculate the 2Y-30Y curve spread
yields["2s30s"] = yields["30Y"] - yields["2Y"]


# Remove rows containing missing values
yields = yields.dropna()

# Display the first five observations
print(yields.head())

# Display the latest five observations
print(yields.tail())


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


# Repeat the PCA estimation for every day after the initial window
for i in range(window, len(changes)):

    # Select the previous 504 trading days
    train = changes.iloc[i-window:i]

    # Select the current day's yield changes
    today = changes.iloc[[i]]

    # Create a PCA model with three factors
    pca = PCA(n_components=3)

    # Estimate the PCA factors using the previous 504 days
    pca.fit(train)

    # Reconstruct today's yield changes using the three PCA factors
    expected = pca.inverse_transform(pca.transform(today))

    # Calculate actual minus PCA-reconstructed yield changes
    residuals.iloc[i] = today.values[0] - expected[0]


# Remove the initial rows where residuals could not be calculated
residuals = residuals.dropna()


# Display the latest PCA residuals
print("\nLatest PCA residuals:")
print(residuals.tail())

# Calculate the historical mean residual for each maturity
mean = residuals.mean()

# Calculate the historical standard deviation for each maturity
std = residuals.std()

# Calculate the z-score of the latest residual for every maturity
z_scores = (residuals.iloc[-1] - mean) / std

# Display the z-scores
print(z_scores)
