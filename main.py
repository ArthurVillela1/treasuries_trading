from pandas_datareader import data as web
import datetime

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime.today()

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

yields = web.DataReader(series, "fred", start, end)

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

yields["2s10s"] = yields["10Y"] - yields["2Y"]
yields["5s30s"] = yields["30Y"] - yields["5Y"]
yields["2s30s"] = yields["30Y"] - yields["2Y"]

yields = yields.dropna()
print(yields.head())
print(yields.tail())

from sklearn.decomposition import PCA
import pandas as pd

maturities = [
    "3M", "6M", "1Y", "2Y", "3Y",
    "5Y", "7Y", "10Y", "20Y", "30Y"
]

changes = yields[maturities].diff().dropna() * 100

window = 504

residuals = pd.DataFrame(
    index=changes.index,
    columns=maturities
)

for i in range(window, len(changes)):

    train = changes.iloc[i-window:i]
    today = changes.iloc[[i]]

    pca = PCA(n_components=3)
    pca.fit(train)

    expected = pca.inverse_transform(
        pca.transform(today)
    )

    residuals.iloc[i] = today.values[0] - expected[0]

residuals = residuals.dropna()

print(residuals.tail())