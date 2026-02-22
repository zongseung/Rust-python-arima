# API Reference

## Low-Level Functions (sarimax_rs)

These are the direct Rust-Python bindings exposed via PyO3.

---

### `sarimax_rs.version()`

Returns the package version string.

```python
>>> sarimax_rs.version()
'0.1.0'
```

---

### `sarimax_rs.sarimax_loglike(y, order, seasonal, params, exog=None, concentrate_scale=True)`

Compute the SARIMAX concentrated (or full) log-likelihood.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `y` | `np.ndarray` (1D) | Endogenous (observed) time series |
| `order` | `tuple(int, int, int)` | `(p, d, q)` ARIMA order |
| `seasonal` | `tuple(int, int, int, int)` | `(P, D, Q, s)` seasonal ARIMA order |
| `params` | `np.ndarray` (1D) | Flat parameter vector `[exog..., ar..., ma..., sar..., sma...]` |
| `exog` | `np.ndarray` (2D) or `None` | Exogenous variables, shape `(n_obs, n_exog)` |
| `concentrate_scale` | `bool` | If `True`, concentrate sigma2 out of likelihood |

**Returns:** `float` — log-likelihood value

**Example:**
```python
import numpy as np
import sarimax_rs

y = np.random.randn(200)
params = np.array([0.5])  # AR(1) coefficient
ll = sarimax_rs.sarimax_loglike(y, (1, 0, 0), (0, 0, 0, 0), params)
```

---

### `sarimax_rs.sarimax_fit(y, order, seasonal, start_params=None, exog=None, concentrate_scale=True, enforce_stationarity=True, enforce_invertibility=True, method=None, maxiter=None)`

Fit a SARIMAX model via Maximum Likelihood Estimation.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `y` | `np.ndarray` (1D) | — | Endogenous time series |
| `order` | `tuple(int, int, int)` | — | `(p, d, q)` |
| `seasonal` | `tuple(int, int, int, int)` | — | `(P, D, Q, s)` |
| `start_params` | `np.ndarray` or `None` | `None` | Initial parameter guess |
| `exog` | `np.ndarray` (2D) or `None` | `None` | Exogenous variables |
| `concentrate_scale` | `bool` | `True` | Concentrate sigma2 |
| `enforce_stationarity` | `bool` | `True` | Enforce AR stationarity |
| `enforce_invertibility` | `bool` | `True` | Enforce MA invertibility |
| `method` | `str` or `None` | `None` | Optimizer: `"lbfgsb"` (default), `"lbfgsb-multi"`, `"lbfgsb-strict"`, `"lbfgs"`, or `"nelder-mead"` |
| `maxiter` | `int` or `None` | `None` | Maximum optimizer work units (default 500). When `0`, returns start params with `converged=False` immediately. |

**Returns:** `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `params` | `list[float]` | Estimated parameters |
| `loglike` | `float` | Log-likelihood at optimum |
| `aic` | `float` | Akaike Information Criterion |
| `bic` | `float` | Bayesian Information Criterion |
| `scale` | `float` | Estimated variance (sigma2) |
| `converged` | `bool` | Whether optimizer truly converged (`TerminationReason::SolverConverged` or `TargetCostReached`). `False` when `maxiter` was exhausted or `maxiter=0`. |
| `method` | `str` | Optimization method used (e.g. `"lbfgsb"`, `"lbfgs+nm"`, `"nelder-mead (fallback)"`) |
| `n_obs` | `int` | Number of observations |
| `n_iter` | `int` | Total optimizer work units: solver iterations for L-BFGS/NM, function evaluations for L-BFGS-B. Multi-start methods report cumulative sum across all sub-runs, capped by `maxiter`. |
| `n_params` | `int` | Number of estimated parameters |

**Example:**
```python
result = sarimax_rs.sarimax_fit(y, (1, 1, 1), (1, 0, 0, 12))
print(f"Converged: {result['converged']}, AIC: {result['aic']:.2f}")
```

---

### `sarimax_rs.sarimax_forecast(y, order, seasonal, params, steps=10, alpha=0.05, exog=None, future_exog=None, concentrate_scale=True)`

Generate h-step ahead forecasts with confidence intervals.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `y` | `np.ndarray` (1D) | — | Endogenous time series |
| `order` | `tuple` | — | `(p, d, q)` |
| `seasonal` | `tuple` | — | `(P, D, Q, s)` |
| `params` | `np.ndarray` (1D) | — | Fitted parameters |
| `steps` | `int` | `10` | Forecast horizon |
| `alpha` | `float` | `0.05` | CI significance level |
| `exog` | `np.ndarray` (2D) or `None` | `None` | Historical exog |
| `future_exog` | `np.ndarray` (2D) or `None` | `None` | Future exog, shape `(steps, n_exog)` |
| `concentrate_scale` | `bool` | `True` | Concentrate sigma2 |

**Returns:** `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `mean` | `list[float]` | Point forecasts (length = steps) |
| `variance` | `list[float]` | Forecast variance |
| `ci_lower` | `list[float]` | Lower CI bound |
| `ci_upper` | `list[float]` | Upper CI bound |

---

### `sarimax_rs.sarimax_residuals(y, order, seasonal, params, exog=None, concentrate_scale=True)`

Compute one-step-ahead residuals and standardized residuals.

**Returns:** `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `residuals` | `list[float]` | Raw residuals (length = n_obs) |
| `standardized_residuals` | `list[float]` | Standardized residuals |

---

### `sarimax_rs.sarimax_batch_loglike(series_list, order, seasonal, params, exog=None, concentrate_scale=True)`

Parallel log-likelihood for multiple time series (Rayon-based).

**Parameters:**
- `series_list`: `list[np.ndarray]` — N time series
- Other params same as `sarimax_loglike`

**Returns:** `list[dict]` — each with `{"loglike": float}`

---

### `sarimax_rs.sarimax_batch_fit(series_list, order, seasonal, exog=None, enforce_stationarity=True, enforce_invertibility=True, concentrate_scale=True, method=None, maxiter=None)`

Parallel fitting for multiple time series (Rayon-based).

**Parameters:**
- `series_list`: `list[np.ndarray]` — N time series
- Other params same as `sarimax_fit`

**Returns:** `list[dict]` — each dict has same keys as `sarimax_fit` return value

---

### `sarimax_rs.sarimax_batch_forecast(series_list, order, seasonal, params_list, steps=10, alpha=0.05, exog=None, future_exog=None, concentrate_scale=True)`

Parallel forecasting for multiple time series (Rayon-based).

**Parameters:**
- `series_list`: `list[np.ndarray]` — N time series
- `params_list`: `list[np.ndarray]` — N parameter vectors (one per series)
- Other params same as `sarimax_forecast`

**Returns:** `list[dict]` — each dict has same keys as `sarimax_forecast` return value

---

## High-Level API (sarimax_py)

### `SARIMAXModel`

statsmodels-compatible SARIMAX model backed by the Rust engine.

```python
from sarimax_py import SARIMAXModel

model = SARIMAXModel(y, order=(1, 1, 1), seasonal_order=(1, 0, 0, 12))
result = model.fit()
```

**Constructor Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `endog` | array_like | — | Observed time series |
| `order` | tuple | `(1, 0, 0)` | `(p, d, q)` |
| `seasonal_order` | tuple | `(0, 0, 0, 0)` | `(P, D, Q, s)` |
| `enforce_stationarity` | bool | `True` | AR constraint |
| `enforce_invertibility` | bool | `True` | MA constraint |

**Methods:**
- `fit(method=None, maxiter=None, start_params=None)` → `SARIMAXResult`

---

### `SARIMAXResult`

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `params` | `np.ndarray` | Estimated parameters |
| `param_names` | `list[str]` | Parameter names matching params vector (e.g. `["ar.L1", "ma.L1"]`) |
| `llf` | `float` | Log-likelihood |
| `aic` | `float` | AIC |
| `bic` | `float` | BIC |
| `scale` | `float` | Estimated sigma2 |
| `nobs` | `int` | Number of observations |
| `converged` | `bool` | Convergence status |
| `method` | `str` | Optimizer method |
| `resid` | `np.ndarray` | Standardized residuals (lazy-loaded) |

**Methods:**

#### `parameter_summary(alpha=0.05, inference=None, include_inference=None)`

Return parameter summary as a machine-readable dict.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | `float` | `0.05` | Significance level for CI, must be in `(0, 1)` |
| `inference` | `str` or `None` | `None` | Inference mode (see below) |
| `include_inference` | `bool` or `None` | `None` | **Deprecated.** Use `inference` instead |

**Inference modes:**

| Mode | Description | Cost |
|------|-------------|------|
| `"none"` | Coefficients only (default) | Fastest |
| `"hessian"` | Numerical Hessian-based std err / z / CI | O(k²) loglike evals |
| `"statsmodels"` | Fit statsmodels internally, borrow its inference | Requires statsmodels |
| `"both"` | Hessian + statsmodels with delta comparison columns | Both costs combined |

**Return dict keys by mode:**

- All modes: `name`, `coef`, `inference_status`, `inference_message`
- `"none"`: `std_err`, `z`, `p_value`, `ci_lower`, `ci_upper` (all `NaN`)
- `"hessian"` / `"statsmodels"`: `std_err`, `z`, `p_value`, `ci_lower`, `ci_upper`
- `"both"`: all of the above plus `hessian_std_err`, `hessian_z`, `hessian_p_value`, `hessian_ci_lower`, `hessian_ci_upper`, `sm_std_err`, `sm_z`, `sm_p_value`, `sm_ci_lower`, `sm_ci_upper`, `delta_std_err`, `delta_ci_lower`, `delta_ci_upper`, `inference_status_hessian`, `inference_status_sm`. The `summary()` output shows dual p-value columns (`hess_p`, `sm_p`) so the source of each p-value is unambiguous.

**Backward compatibility:** `include_inference=True` maps to `inference="hessian"`, `include_inference=False` maps to `inference="none"`. Using `include_inference` emits a `DeprecationWarning`. When both `inference` and `include_inference` are specified, `inference` takes precedence (with a `DeprecationWarning`); invalid `inference` values always raise `ValueError`.

**Cache policy:** Inference results are cached per `(mode, alpha, params_fingerprint)`. The cache auto-invalidates when `result.params` changes, ensuring recomputation with updated parameter values.

```python
result = model.fit()

# Quick: no inference
ps = result.parameter_summary(inference="none")

# Hessian-based inference
ps = result.parameter_summary(inference="hessian")

# statsmodels-based inference
ps = result.parameter_summary(inference="statsmodels")

# Compare both sources
ps = result.parameter_summary(inference="both")
print(ps["delta_std_err"])  # hessian - statsmodels difference
```

#### `summary(alpha=0.05, inference=None, include_inference=None)`

Return a human-readable summary string. Same parameter semantics as `parameter_summary()`.

```python
# Default: coefficients only
print(result.summary())

# With inference statistics
print(result.summary(inference="hessian"))

# Dual-column comparison
print(result.summary(inference="both"))
```

#### `forecast(steps=1, alpha=0.05, exog=None)` → `ForecastResult`

#### `get_forecast(steps=1, alpha=0.05, exog=None)` → `ForecastResult` (alias)

---

### Parameter Naming Convention

Parameter names follow statsmodels convention:

| Component | Pattern | Example |
|-----------|---------|---------|
| Exogenous | `x1`, `x2`, ... | `x1` |
| AR | `ar.L1`, `ar.L2`, ... | `ar.L1` |
| MA | `ma.L1`, `ma.L2`, ... | `ma.L1` |
| Seasonal AR | `ar.S.L{s}`, `ar.S.L{2s}`, ... | `ar.S.L12` |
| Seasonal MA | `ma.S.L{s}`, `ma.S.L{2s}`, ... | `ma.S.L12` |
| Variance | `sigma2` (only when `concentrate_scale=False`) | `sigma2` |

---

### `ForecastResult`

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `predicted_mean` | `np.ndarray` | Point forecasts |
| `variance` | `np.ndarray` | Forecast variance |
| `ci_lower` | `np.ndarray` | Lower CI bound |
| `ci_upper` | `np.ndarray` | Upper CI bound |

**Methods:**
- `conf_int(alpha=None)` → `np.ndarray` shape `(steps, 2)`. If `alpha` differs from the original, recomputes CI from stored variance. Raises `ValueError` if `alpha` is not in `(0, 1)`.
