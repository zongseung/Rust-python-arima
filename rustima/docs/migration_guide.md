# Migration Guide: statsmodels to sarimax_rs

## Quick Start

### Before (statsmodels)

```python
import statsmodels.api as sm

model = sm.tsa.SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 0, 0, 12), trend="n")
result = model.fit(disp=False)

print(f"AIC: {result.aic}")
print(f"Params: {result.params}")

forecast = result.get_forecast(steps=10)
print(f"Forecast: {forecast.predicted_mean}")
print(f"CI: {forecast.conf_int()}")

residuals = result.resid
```

### After (sarimax_rs — High-level API)

```python
from sarimax_py import SARIMAXModel

model = SARIMAXModel(y, order=(1, 1, 1), seasonal_order=(1, 0, 0, 12))
result = model.fit()

print(f"AIC: {result.aic}")
print(f"Params: {result.params}")

forecast = result.forecast(steps=10)
print(f"Forecast: {forecast.predicted_mean}")
print(f"CI: {forecast.conf_int()}")

residuals = result.resid
```

### After (sarimax_rs — Low-level API)

```python
import numpy as np
import sarimax_rs

# Fit
result = sarimax_rs.sarimax_fit(y, (1, 1, 1), (1, 0, 0, 12))
params = np.array(result["params"])

# Forecast
fc = sarimax_rs.sarimax_forecast(y, (1, 1, 1), (1, 0, 0, 12), params, steps=10)

# Residuals
resid = sarimax_rs.sarimax_residuals(y, (1, 1, 1), (1, 0, 0, 12), params)
```

## API Mapping

| statsmodels | sarimax_rs (high-level) | sarimax_rs (low-level) |
|------------|------------------------|----------------------|
| `SARIMAX(y, ...)` | `SARIMAXModel(y, ...)` | — |
| `model.fit()` | `model.fit()` | `sarimax_fit(y, ...)` |
| `result.llf` | `result.llf` | `result["loglike"]` |
| `result.params` | `result.params` | `result["params"]` |
| `result.aic` | `result.aic` | `result["aic"]` |
| `result.bic` | `result.bic` | `result["bic"]` |
| `result.scale` | `result.scale` | `result["scale"]` |
| `result.nobs` | `result.nobs` | `result["n_obs"]` |
| `result.get_forecast(n)` | `result.forecast(n)` | `sarimax_forecast(...)` |
| `result.resid` | `result.resid` | `sarimax_residuals(...)` |
| `model.loglike(params)` | — | `sarimax_loglike(y, ..., params)` |
| `print(result.summary())` | `print(result.summary())` | — |

## Key Differences

### 1. No `trend` parameter

sarimax_rs always uses `trend="n"` (no trend). To add a constant/linear trend, include it as an exogenous variable:

```python
# statsmodels
model = sm.tsa.SARIMAX(y, order=(1,0,0), trend="c")

# sarimax_rs equivalent
exog = np.ones((len(y), 1))  # constant column
result = sarimax_rs.sarimax_fit(y, (1,0,0), (0,0,0,0), exog=exog)
```

### 2. `concentrate_scale=True` is default

Both libraries default to concentrated scale, but the parameter is explicit in sarimax_rs.

### 3. Batch operations

sarimax_rs provides parallel batch operations not available in statsmodels:

```python
# Fit 1000 series in parallel
series_list = [y_1, y_2, ..., y_1000]
results = sarimax_rs.sarimax_batch_fit(series_list, (1,0,0), (0,0,0,0))
```

### 4. Parameter enforcement

When using `enforce_stationarity=True` (default):
- sarimax_rs `fit()` returns **constrained** parameters
- statsmodels `fit()` returns **unconstrained** parameters
- For direct comparison, use `enforce_stationarity=False` on both

### 5. Exogenous variables

```python
# statsmodels
model = sm.tsa.SARIMAX(y, exog=X, order=(1,0,0))
result = model.fit()
forecast = result.get_forecast(10, exog=X_future)

# sarimax_rs
result = sarimax_rs.sarimax_fit(y, (1,0,0), (0,0,0,0), exog=X)
params = np.array(result["params"])
fc = sarimax_rs.sarimax_forecast(
    y, (1,0,0), (0,0,0,0), params,
    steps=10, exog=X, future_exog=X_future,
)
```

## Performance Tips

1. **Use batch operations** for fitting multiple series — 10-120x faster than sequential
2. **Keep `enforce_stationarity=True`** — helps optimizer converge faster
3. **Use the low-level API** for maximum performance in tight loops
4. **Avoid repeated `result.resid` calls** — the high-level API caches, but the low-level API recomputes each time
