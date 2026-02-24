# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**sarimax-rs** is a high-performance SARIMAX (Seasonal ARIMA) engine written in Rust with Python bindings via PyO3. It replaces Python's `statsmodels.tsa.SARIMAX` with native Rust for Kalman filter loops, MLE optimization, and Rayon-based parallel batch processing. All code lives under the `sarimax_rs/` subdirectory.

## Build & Development Commands

All commands run from `sarimax_rs/`.

```bash
# Rust unit tests (154 tests across all modules)
cargo test --all-targets

# Build Python wheel (development)
maturin develop --release

# Build Python wheel (production, using uv)
CARGO_TARGET_DIR=target_wheel uv run maturin build --out /tmp/wheels
uv pip install --force-reinstall /tmp/wheels/sarimax_rs-*.whl

# Python integration tests (requires wheel installed first)
python -m pytest python_tests/ -v

# Run a single Python test
python -m pytest python_tests/test_fit.py::test_arima_111_fit -v

# Run a single Rust test
cargo test test_name

# Regenerate statsmodels reference fixtures
python python_tests/generate_fixtures.py

# Benchmarks (criterion)
cargo bench
```

## Architecture

### Processing Pipeline

```
Python (PyO3) → params.rs (unpack/transform) → polynomial.rs (lag expansion)
  → state_space.rs (Harvey matrices T,Z,R,Q) → initialization.rs (diffuse P₀)
  → kalman.rs (filter/loglike) → optimizer.rs (L-BFGS/Nelder-Mead MLE)
  → forecast.rs (h-step prediction + CI)
```

### Key Modules (sarimax_rs/src/)

- **lib.rs** — PyO3 entry point; exports 10 Python functions (`sarimax_fit`, `sarimax_forecast`, `sarimax_loglike`, `sarimax_residuals`, `sarimax_batch_fit`, `sarimax_batch_forecast`, `sarimax_batch_loglike`, `sarimax_inference`, `sarimax_diagnostics`, `version`). All functions accept `trend=` and `simple_differencing=` parameters.
- **types.rs** — `SarimaxOrder(p,d,q,P,D,Q,s)`, `SarimaxConfig`, `Trend`, `FitResult`
- **params.rs** — `SarimaxParams` struct, flat vector ↔ struct conversion, Monahan/Jones constrained transform for stationarity/invertibility
- **state_space.rs** — Builds Harvey representation matrices. Normal: `k_states = k_states_diff(d+s*D) + k_order`. With `simple_differencing=true`: `k_states = k_order` only (diff states removed)
- **pipeline.rs** — Shared helpers: `kalman_eval`, `kalman_filter_full` with automatic pre-differencing when `simple_differencing=true`
- **kalman.rs** — `kalman_loglike()` (concentrated/full) and `kalman_filter()` (full state history). Joseph form covariance update
- **optimizer.rs** — MLE via L-BFGS-B (primary) with Nelder-Mead fallback. CSS pre-optimization, DARE initialization, multi-start with near-cancellation filtering (P6), finite-difference gradients (center diff, eps=1e-7)
- **forecast.rs** — `forecast_pipeline()` and `residuals_pipeline()`. Trend state intercept propagated in h-step forecast loop. Z-score via Abramowitz & Stegun approximation
- **batch.rs** — `batch_fit()` / `batch_forecast()` using `rayon::par_iter()`. Error isolation per series
- **polynomial.rs** — `reduced_ar` / `reduced_ma` via polynomial multiplication for seasonal lag expansion
- **start_params.rs** — CSS-based initial parameter estimation (Yule-Walker AR, OLS MA)

### Python Layer

- **python/sarimax_py/model.py** — `SARIMAXModel`, `SARIMAXResult`, `ForecastResult`, `PredictionResult` classes (statsmodels-compatible API). Supports trend='n'/'c'/'t'/'ct', simple_differencing=True/False. Polars DataFrames via `to_dataframe()`, `params_table()`. HQIC, `get_prediction()`, statsmodels-style `summary()`.
- **python/sarimax_py/auto.py** — `auto_arima()` with stepwise (Hyndman-Khandakar) and grid search modes. `AutoARIMAResult` with Polars `history_dataframe()`.
- **python_tests/** — 351 pytest integration tests validated against statsmodels reference data (includes test_simple_diff.py for Phase 5-C)

### Test Fixtures

JSON files in `tests/fixtures/` contain statsmodels reference outputs for ground-truth validation. Cross-implementation tolerances: params ≤ 1e-2, loglike ≤ 3.0, AIC/BIC ≤ 6.0.

## Constraints & Limitations

- Seasonal differencing D must be 0 or 1
- Exogenous variables (exog) supported: `[exog_coeffs, ar(p), ma(q), sar(P), sma(Q)]` param layout with exog
- Hessian/OPG-based standard errors available via `inference="hessian"` or `inference="opg"`
- Forecast steps capped at 10,000
- Trend support: 'n' (none), 'c' (constant), 't' (linear), 'ct' (both)
- Flat param vector layout: `[trend(kt) | exog(k) | ar(p) | ma(q) | sar(P) | sma(Q)]`
- `concentrate_scale=true` by default (sigma² concentrated out of likelihood)
- `simple_differencing=false` by default. When `true`: pre-differencing reduces `k_states` by `d + s*D`, n_obs = n - d - s*D (R-style AIC/BIC). DARE init used (vs Lyapunov for normal path), so loglike may differ slightly from statsmodels SD.

## Key Dependencies

- **nalgebra** — Dynamic matrices (DMatrix, DVector)
- **argmin** — L-BFGS + Nelder-Mead optimization
- **rayon** — Parallel batch processing
- **pyo3 + numpy** — Python C-API bindings with zero-copy array access
- **polars** — Columnar DataFrame for Python result tables (replaces pandas)
