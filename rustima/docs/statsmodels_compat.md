# statsmodels Compatibility Notes

## Overview

`sarimax_rs` is designed to produce results compatible with `statsmodels.tsa.SARIMAX`. This document describes the expected differences and tolerance levels.

## Supported Features

| Feature | sarimax_rs | statsmodels |
|---------|-----------|-------------|
| ARIMA(p,d,q) | Yes | Yes |
| Seasonal ARIMA(P,D,Q,s) | Yes | Yes |
| Exogenous variables (exog) | Yes | Yes |
| Concentrated scale (MLE) | Yes (default) | Yes |
| Enforce stationarity | Yes | Yes |
| Enforce invertibility | Yes | Yes |
| L-BFGS optimizer | Yes | Yes |
| Nelder-Mead optimizer | Yes | Yes |
| Batch parallel fitting | Yes (Rayon) | No |
| Confidence intervals | Yes (normal) | Yes (normal) |
| AIC / BIC | Yes | Yes |
| Standardized residuals | Yes | Yes |

## Not Supported (Out of Scope)

| Feature | Status |
|---------|--------|
| Trend (constant/linear) | Not implemented |
| Time-varying regression | Not implemented |
| Hamilton filter | Not implemented |
| HQIC | Not implemented |
| Exact diffuse initialization | Approximate only |
| In-sample prediction | Not implemented |
| `simulate()` | Not implemented |

## Numerical Differences

### Log-Likelihood

When evaluating the log-likelihood at **the same parameters**, the expected absolute difference is:

| Model type | Typical |Δloglike| |
|-----------|----------------------|
| Non-seasonal ARIMA | < 0.01 |
| Seasonal (s=4) | < 0.05 |
| Seasonal (s=12) | < 0.1 |

These differences arise from:
1. **Initialization**: sarimax_rs uses approximate diffuse initialization which may differ slightly from statsmodels' exact diffuse
2. **Numerical precision**: Different linear algebra backends (nalgebra vs numpy/LAPACK)

### Fitted Parameters

When fitting independently (different optimizers, starting params), parameter differences can be larger:

- Parameters are typically within 10-20% of statsmodels
- Log-likelihood at Rust's optimum is typically **equal or better** than statsmodels
- Different local optima are possible for complex models

### AIC / BIC

Since AIC = -2 * loglike + 2k and BIC = -2 * loglike + k * ln(n):
- AIC/BIC differences are 2x the loglike difference
- Compare trends across models rather than absolute values

### Forecasts

- Forecast means at the same params: typically < 1e-6 absolute difference
- Forecast variance: may differ due to scale estimation differences
- Confidence intervals: proportional to variance differences

## Parameter Layout

Both sarimax_rs and statsmodels use concentrated scale by default, meaning sigma2 is **not** included in the parameter vector.

**Parameter order (sarimax_rs):**
```
[exog_0, ..., exog_k, ar_1, ..., ar_p, ma_1, ..., ma_q, sar_1, ..., sar_P, sma_1, ..., sma_Q]
```

**Parameter order (statsmodels):**
```
[ar.L1, ..., ar.Lp, ma.L1, ..., ma.Lq, ar.S.L{s}, ..., ar.S.L{Ps}, ma.S.L{s}, ..., ma.S.L{Qs}]
```

When `enforce_stationarity=False` and `enforce_invertibility=False`, the parameters from both libraries are directly comparable.

When enforcement is enabled:
- `sarimax_fit` returns **constrained** (direct) parameters
- statsmodels' `result.params` are **unconstrained** (transformed) parameters
- To compare, use `enforce_stationarity=False, enforce_invertibility=False` on both sides

## Migration Quick Reference

| statsmodels | sarimax_rs |
|------------|------------|
| `model = SARIMAX(y, order=(1,1,1))` | `model = SARIMAXModel(y, order=(1,1,1))` |
| `res = model.fit(disp=False)` | `res = model.fit()` |
| `res.llf` | `res.llf` |
| `res.params` | `res.params` |
| `res.aic` | `res.aic` |
| `res.bic` | `res.bic` |
| `res.get_forecast(10)` | `res.forecast(10)` or `res.get_forecast(10)` |
| `res.resid` | `res.resid` |
| `print(res.summary())` | `print(res.summary())` |
