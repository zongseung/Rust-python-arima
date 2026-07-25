# Changelog

All notable changes to rustima are documented here.
Versions follow `0.MINOR.PATCH`, where breaking changes bump MINOR.

## 0.3.0 — 2026-07-25

A diagnosis sweep of the engine (~1,800 adversarial inputs, 9 numerical
regimes against statsmodels, and a structural audit) found and fixed four
defects that returned **wrong numbers without raising**. Full report:
`planner/ver9/DIAGNOSIS_V9.md`.

> **Results from 0.2.x are not comparable to 0.3.0.** The log-likelihood,
> AIC and BIC of every model **without differencing** (`d=0, D=0`) were
> biased, and `inference="hessian"` standard errors were inflated. If you
> have published or stored numbers from 0.2.x, re-run them.

### Breaking — corrected numerical results

- **Initial state covariance for non-differenced models.** `P₀` for the
  stationary ARMA block used the DARE fixed point (the steady-state
  *filtering* covariance) instead of the unconditional *Lyapunov*
  covariance. For AR(1) this used `σ²` where the correct prior is
  `σ²/(1-φ²)`. The error grows like `1/(1-φ²)` and does not vanish with
  sample size: `Δloglike` ≈ −0.1 at φ=0.5, −62 at φ=0.995, and `ΔAIC`
  +156 for a seasonal SAR=0.99 model. Fixed-parameter log-likelihoods now
  match statsmodels to 1e-10. Models with `d≥1` were always correct.
- **Standard errors from `inference="hessian"`.** The Hessian was mapped
  from unconstrained to constrained space with the Jacobian instead of its
  inverse, applying the transform twice and inflating AR/MA standard
  errors by up to 6.6×. Now matches statsmodels `cov_params_approx` to
  five decimals. `inference="opg"` was unaffected.
- **False convergence with trend terms.** A deterministic trend made the
  CSS/Burg start-parameter estimator see near-integration, saturating the
  AR start on the constraint boundary where L-BFGS-B stalls and reports
  `converged=True` with a gradient of order 100 — up to 192 nats below the
  optimum. Start parameters are now detrended, and out-of-box estimates
  fall back to zeros instead of clamping.
- **Near-cancellation detection.** AR and MA roots were compared against
  the sign-flipped MA polynomial, so genuine cancellation (`θ = −φ`) was
  never detected — the warning and the multi-start restart filter both ran
  on the wrong condition.
- **Post-estimation ran a differently-initialized filter.** `forecast`,
  `rolling_forecast`, `residuals`, `batch_forecast` and `diagnostics`
  ignored the model's enforcement flags and silently used an approximate
  diffuse initialization, so `res.llf` and `res.resid` / `res.diagnostics()`
  described different models — Ljung-Box tested residuals the likelihood
  never saw.

### Breaking — stricter inputs and API

- Masked arrays are rejected by `SARIMAXModel` and `auto_arima` (previously
  the mask was silently dropped and masked entries used as data).
- `sarimax_inference` validates the parameter-vector length instead of
  returning empty arrays.
- `SARIMAXResult.param_names` raises if the Rust layout and the Python
  naming disagree, instead of padding with `param_N` and printing values
  under the wrong names.
- `sarimax_rolling_forecast` refuses configurations whose origin snapshots
  would exceed 2 GiB (previously unbounded — 4.4 GB measured at n=1000,
  s=365, and an allocation failure aborts the process).
- `sarimax_residuals` returns an additional `prediction_variances` key.
  Code asserting an exact key set must be updated.
- Rust: the never-supported `SarimaxConfig.measurement_error` field is
  removed.

### Added

- `PredictionResult.conf_int(alpha=...)` and `.se_mean` —
  `get_prediction()` now produces intervals across the whole range, not
  only the out-of-sample tail.
- `sarimax_residuals` returns `prediction_variances` (one-step-ahead
  prediction variance).
- `sarimax_forecast`, `sarimax_rolling_forecast`, `sarimax_residuals`,
  `sarimax_batch_forecast` and `sarimax_diagnostics` accept
  `enforce_stationarity` / `enforce_invertibility` (appended to the
  signature; positional calls are unaffected).

### Fixed

- **Deadlock in forked processes.** Rayon's global pool is not fork-safe:
  any batch or grid call in a `fork()`ed child hung forever with no error
  (reachable through `multiprocessing`'s fork context after a single
  earlier batch call). Parallel entry points now fail fast with a clear
  message, and the optimizer's internal parallel sections fall back to
  sequential execution.
- Batch workers isolate panics per series, so the documented per-series
  error isolation holds even if a worker panics.
- The analytical gradient no longer disappears near unit roots: the score
  pass now shares the likelihood pass's `F_t ≤ 0` fallback and
  steady-state adoption guard, keeping the gradient finite wherever the
  log-likelihood is.
- The analytical score accounts for `∂P₀/∂θ` when the initialization
  depends on the parameters.

### Internal

- Single-sourced the trend basis, the flat-parameter count, and
  single-series input validation (the last was duplicated into
  `sarimax_grid_search`, the `auto_arima` hot path). Model flags moved from
  adjacent positional booleans into a named struct.
- Removed dead code: `auto_arima`'s unused evaluation path, the
  `measurement_error` plumbing, and the redundant Python numerical-Hessian
  implementation.
- Python test suite 407 → 429, including regression tests for every defect
  above and eight previously uncovered option-matrix combinations. CI is
  green for the first time since 2026-05-23.

## 0.2.0

Initial tagged release with the SARIMAX engine, `auto_arima`, batch
processing and the statsmodels-compatible Python API.
