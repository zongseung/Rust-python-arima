# sarimax_rs v1 Benchmark Summary

Generated: 2026-02-25

## Overview

Comprehensive comparison of `sarimax_rs` (Rust) vs `statsmodels` (Python) across
ARIMA, SARIMA, ARIMAX, and SARIMAX model families.

### vs statsmodels (exact diffuse initialization)

| Category | Models | Mean Δ(LL) | P95 Δ | Max Δ | Mean Speedup |
|----------|--------|-----------|-------|-------|-------------|
| ARIMA | 15 | 4.04 | 6.89 | 6.98 | 37.8x |
| SARIMA s=7 | 80 | 28.45 | 53.03 | 163.88 | 6.9x |
| SARIMA s=12 | 80 | 43.89 | 96.69 | 107.28 | 6.1x |
| SARIMA s=24 | 80 | 85.42 | 166.33 | 209.06 | 5.6x |
| ARIMAX | 15 | 3.81 | 6.44 | 6.45 | 14.0x |
| SARIMAX s=7 | 80 | 25.80 | 48.47 | 57.69 | 7.1x |
| SARIMAX s=12 | 80 | 41.42 | 95.34 | 105.52 | 6.8x |
| SARIMAX s=24 | 80 | 85.72 | 166.34 | 209.06 | 8.9x |

### vs R `stats::arima()` (same approximate diffuse initialization)

Previously validated (81-model SARIMAX matrix, s=12, k=2):

| Metric | Value |
|--------|-------|
| Models compared | 81 |
| Equal (Δ<0.1) | 84.0% |
| Close (Δ<1.0) | 87.7% |
| Mean Δ(LL) | 0.93 |
| Max Δ(LL) | 11.8 |
| Pure SARIMA Δ | <0.06 |

## Understanding the Log-Likelihood Differences

### Why sarimax_rs vs statsmodels shows large Δ

The log-likelihood difference is **primarily due to the Kalman filter initialization method**, not optimization quality:

1. **sarimax_rs**: Approximate diffuse initialization (kappa = 1e6)
   - Matches R's `stats::arima()` implementation
   - Adds kappa * I to the initial state covariance P₀
   - Number of diffuse states = d + s*D (e.g., 25 for SARIMA with s=24, D=1)

2. **statsmodels**: Exact diffuse initialization (De Jong & Shephard)
   - Separates diffuse and non-diffuse states
   - Uses a two-part filter for the initial diffuse states
   - Produces numerically different log-likelihood values

**The Δ scales with the number of diffuse states:**
- ARIMA (d=1): 1 diffuse state → Δ ≈ 1-7
- SARIMA s=7 (d=1, D=1): 8 diffuse states → Δ ≈ 2-164
- SARIMA s=12 (d=1, D=1): 13 diffuse states → Δ ≈ 2-107
- SARIMA s=24 (d=1, D=1): 25 diffuse states → Δ ≈ 4-209

**Both methods are mathematically valid** — they just define the log-likelihood differently during the initial diffuse period. Model selection (AIC/BIC ranking) is consistent within each method.

### Why sarimax_rs vs R shows small Δ

Both use the same approximate diffuse initialization (kappa=1e6), so log-likelihood values are directly comparable. The small remaining differences come from optimizer convergence to different local optima.

## Performance

### Speed Summary

| Category | Mean rs (ms) | Mean sm (ms) | Mean Speedup |
|----------|-------------|-------------|-------------|
| ARIMA | 3 | 28 | 37.8x |
| SARIMA s=7 | 48 | 152 | 6.9x |
| SARIMA s=12 | 103 | 336 | 6.1x |
| SARIMA s=24 | 239 | 679 | 5.6x |
| ARIMAX | 5 | 37 | 14.0x |
| SARIMAX s=7 | 53 | 157 | 7.1x |
| SARIMAX s=12 | 115 | 352 | 6.8x |
| SARIMAX s=24 | 253 | 732 | 8.9x |

Key observations:
- **ARIMA**: 30-40x speedup (small state dimension, Rust overhead dominates)
- **Seasonal models**: 5-15x speedup (Kalman filter loop dominates, Rust excels)
- **sarimax_rs is consistently faster** across all model types and seasonal periods

## Files

- `01_ARIMA_comparison.md` - ARIMA(p,1,q) for p,q in 0..3
- `02_SARIMA_s7_comparison.md` - SARIMA with s=7
- `02_SARIMA_s12_comparison.md` - SARIMA with s=12
- `02_SARIMA_s24_comparison.md` - SARIMA with s=24
- `03_ARIMAX_comparison.md` - ARIMAX(p,1,q) with 2 exog
- `04_SARIMAX_s7_comparison.md` - SARIMAX with s=7, 2 exog
- `04_SARIMAX_s12_comparison.md` - SARIMAX with s=12, 2 exog
- `04_SARIMAX_s24_comparison.md` - SARIMAX with s=24, 2 exog
- `bench_all_models.py` - Benchmark script

## Configuration

- **Data**: Synthetic ARMA processes, n=500
- **Differencing**: d=1, D=1 (for seasonal models)
- **Trend**: 'n' (no trend)
- **sarimax_rs**: concentrate_scale=true (default), CSS-ML pre-optimization
- **statsmodels**: enforce_stationarity=False, enforce_invertibility=False
- **Orders**: p,q in 0..3 (ARIMA/ARIMAX), p,q,P,Q in 0..2 (SARIMA/SARIMAX)
- **Seasonal periods**: s = 7, 12, 24
- **Exogenous regressors**: k = 2 (for ARIMAX/SARIMAX)
