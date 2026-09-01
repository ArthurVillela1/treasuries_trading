import numpy as np
import pandas as pd
from pandas_datareader import data as web
from scipy.optimize import minimize_scalar

# --------------------------------------------------
# 1. LOAD TREASURY YIELDS
# --------------------------------------------------

series = {
    "3M": "DGS3MO",
    "6M": "DGS6MO",
    "1Y": "DGS1",
    "2Y": "DGS2",
    "3Y": "DGS3",
    "5Y": "DGS5",
    "7Y": "DGS7",
    "10Y": "DGS10",
    "20Y": "DGS20",
    "30Y": "DGS30"
}

Y = web.DataReader(
    list(series.values()),
    "fred",
    "2005-01-01"
)

Y.columns = series.keys()
Y = Y.dropna()

tau = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])


# --------------------------------------------------
# 2. NELSON-SIEGEL CURVE
# --------------------------------------------------

def loadings(lam):
    slope = (1 - np.exp(-lam * tau)) / (lam * tau)
    curvature = slope - np.exp(-lam * tau)

    return np.column_stack([
        np.ones(len(tau)),
        slope,
        curvature
    ])


def fit_curve(y, lam):
    L = loadings(lam)

    beta = np.linalg.lstsq(
        L,
        y,
        rcond=None
    )[0]

    fitted = L @ beta

    return beta, fitted


# Choose lambda from the first half of the sample

train = Y.iloc[:len(Y)//2]

def objective(lam):
    error = 0

    for y in train.values:
        _, fitted = fit_curve(y, lam)
        error += np.sum((y - fitted) ** 2)

    return error


lam = minimize_scalar(
    objective,
    bounds=(0.01, 3),
    method="bounded"
).x


# --------------------------------------------------
# 3. ESTIMATE DAILY CURVE FACTORS
# --------------------------------------------------

L = loadings(lam)

factors = []

for y in Y.values:
    beta, _ = fit_curve(y, lam)
    factors.append(beta)

F = pd.DataFrame(
    factors,
    index=Y.index,
    columns=["Level", "Slope", "Curvature"]
)


# --------------------------------------------------
# 4. MODEL FACTOR DYNAMICS
# --------------------------------------------------

# F_t = c + A F_(t-1) + error

X = np.column_stack([
    np.ones(len(F) - 1),
    F.shift(1).dropna().values
])

target = F.iloc[1:].values

coef = np.linalg.lstsq(
    X,
    target,
    rcond=None
)[0]

c = coef[0]
A = coef[1:].T


# --------------------------------------------------
# 5. ONE-STEP-AHEAD FITTED CURVE
# --------------------------------------------------

predicted_factors = (
    c
    + F.shift(1).values @ A.T
)

predicted_factors = pd.DataFrame(
    predicted_factors,
    index=F.index,
    columns=F.columns
)

fitted_yields = pd.DataFrame(
    predicted_factors.values @ L.T,
    index=Y.index,
    columns=Y.columns
)


# --------------------------------------------------
# 6. RESIDUALS
# --------------------------------------------------

residuals = Y - fitted_yields


# --------------------------------------------------
# 7. Z-SCORES
# --------------------------------------------------

mean = residuals.rolling(252).mean().shift(1)
std = residuals.rolling(252).std().shift(1)

z = (residuals - mean) / std


# --------------------------------------------------
# 8. SIGNALS
# --------------------------------------------------

# Positive residual:
# yield too high -> bond relatively cheap -> long

# Negative residual:
# yield too low -> bond relatively rich -> short

signals = pd.DataFrame(
    0,
    index=Y.index,
    columns=Y.columns
)

signals[z > 1.5] = 1
signals[z < -1.5] = -1


# --------------------------------------------------
# 9. SIMPLE BACKTEST
# --------------------------------------------------

yield_change_bp = Y.diff() * 100

# Long bond benefits when yield falls,
# hence the minus sign.

returns = (
    -signals.shift(1)
    * yield_change_bp
)

print("Cumulative yield-space return by maturity:")
print(returns.sum().round(2))