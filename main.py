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