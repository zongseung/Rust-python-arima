pub mod batch;
pub mod css;
pub mod error;
pub mod forecast;
pub mod inference;
pub mod initialization;
pub mod kalman;
pub(crate) mod lbfgsb_ffi;
pub(crate) mod lbfgsb_wrapper;
pub mod optimizer;
pub mod params;
pub mod pipeline;
pub mod polynomial;
pub mod score;
pub mod start_params;
pub mod state_space;
#[cfg(test)]
pub(crate) mod test_helpers;
pub mod types;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::params::SarimaxParams;
use crate::types::{SarimaxConfig, SarimaxOrder, Trend};

const MAX_K_STATES: usize = 1024;

// Hard upper bounds on individual model order parameters to prevent
// OOM / compute-DoS from adversarial or erroneous inputs.
const MAX_P: usize = 20;
const MAX_Q: usize = 20;
const MAX_D: usize = 3;
const MAX_PP: usize = 4;
const MAX_QQ: usize = 4;
const MAX_DD: usize = 2;
const MAX_S: usize = 365;
const MAX_N_EXOG: usize = 100;

// ---------------------------------------------------------------------------
// Shared helpers (eliminate boilerplate across PyO3 functions)
// ---------------------------------------------------------------------------

/// Convert a 2D numpy array (n_obs × n_exog, row-major) to column-major Vec<Vec<f64>>.
fn numpy2d_to_cols(arr: &PyReadonlyArray2<f64>) -> Vec<Vec<f64>> {
    let shape = arr.shape();
    let (n_obs, n_exog) = (shape[0], shape[1]);
    let data = arr.as_array();
    (0..n_exog)
        .map(|j| (0..n_obs).map(|t| data[[t, j]]).collect())
        .collect()
}

/// Parse optional exog numpy array → (column-major vecs, n_exog, n_rows).
///
/// Returns `n_rows` from the original numpy shape so that row-count validation
/// works even for 0-column exog arrays (shape `(m, 0)`).
fn parse_exog(exog: Option<&PyReadonlyArray2<f64>>) -> (Option<Vec<Vec<f64>>>, usize, usize) {
    match exog {
        Some(e) => {
            let n_rows = e.shape()[0];
            let cols = numpy2d_to_cols(e);
            let n = cols.len();
            (Some(cols), n, n_rows)
        }
        None => (None, 0, 0),
    }
}

fn validate_finite_cols(cols: &[Vec<f64>], input_name: &str) -> PyResult<()> {
    for (j, col) in cols.iter().enumerate() {
        for (t, v) in col.iter().enumerate() {
            if !v.is_finite() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "{} contains NaN or Inf at column {}, row {}",
                    input_name, j, t
                )));
            }
        }
    }
    Ok(())
}

/// Validate that a 1D endog slice contains no NaN/Inf values.
fn validate_endog_finite(endog: &[f64]) -> PyResult<()> {
    if endog.iter().any(|v| !v.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "input time series contains NaN or Inf values",
        ));
    }
    Ok(())
}

/// Validate that each series in a batch contains no NaN/Inf values.
fn validate_batch_finite(series_list: &[PyReadonlyArray1<'_, f64>]) -> PyResult<Vec<Vec<f64>>> {
    let mut result = Vec::with_capacity(series_list.len());
    for (i, s) in series_list.iter().enumerate() {
        let sl = s.as_slice()?;
        if sl.iter().any(|v| !v.is_finite()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "input time series at index {} contains NaN or Inf values",
                i
            )));
        }
        result.push(sl.to_vec());
    }
    Ok(result)
}

/// Parse and validate a batch exog list: matching lengths, finite values, consistent column counts,
/// and per-series row count matching endog length.
///
/// Converts numpy arrays to column-major Vecs and validates:
/// - `exog_list` length matches `series_len`
/// - All values are finite
/// - All series have the same number of exog columns
/// - Each exog's row count matches the corresponding series length
fn parse_and_validate_batch_exog(
    exog_list: &Option<Vec<PyReadonlyArray2<'_, f64>>>,
    series_len: usize,
    series_lengths: &[usize],
) -> PyResult<(Option<Vec<Vec<Vec<f64>>>>, usize)> {
    if let Some(ref el) = exog_list {
        if el.len() != series_len {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "exog_list length ({}) must match series_list length ({})",
                el.len(),
                series_len
            )));
        }
        let exog_vecs: Vec<Vec<Vec<f64>>> = el.iter().map(|e| numpy2d_to_cols(e)).collect();
        let n_exog = exog_vecs.first().map_or(0, |c| c.len());
        for (i, cols) in exog_vecs.iter().enumerate() {
            validate_finite_cols(cols, &format!("exog_list[{}]", i))?;
            if cols.len() != n_exog {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "exog at index {} has {} columns but expected {} (from first series)",
                    i,
                    cols.len(),
                    n_exog
                )));
            }
            // Per-series row count validation (shape-based via numpy shape[0])
            let exog_nrows = el[i].shape()[0];
            if exog_nrows != series_lengths[i] {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "exog_list[{}] has {} rows but series[{}] has {} observations; they must match",
                    i, exog_nrows, i, series_lengths[i]
                )));
            }
        }
        Ok((Some(exog_vecs), n_exog))
    } else {
        Ok((None, 0))
    }
}

/// Parse and validate a batch forecast exog list (future_exog for batch forecasting).
///
/// Validates:
/// - Length matches `series_len`
/// - All values are finite
/// - Column counts match `n_exog`
/// - Each column has at least `steps` rows
fn parse_and_validate_batch_forecast_exog(
    exog_forecast_list: &Option<Vec<PyReadonlyArray2<'_, f64>>>,
    series_len: usize,
    n_exog: usize,
    steps: usize,
) -> PyResult<Option<Vec<Vec<Vec<f64>>>>> {
    match exog_forecast_list {
        Some(ref el) => {
            if el.len() != series_len {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "exog_forecast_list length ({}) must match series_list length ({})",
                    el.len(),
                    series_len
                )));
            }
            let vecs: Vec<Vec<Vec<f64>>> = el.iter().map(|e| numpy2d_to_cols(e)).collect();
            for (i, cols) in vecs.iter().enumerate() {
                validate_finite_cols(cols, &format!("exog_forecast_list[{}]", i))?;
                if cols.len() != n_exog {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "exog_forecast at index {} has {} columns but expected {}",
                        i,
                        cols.len(),
                        n_exog
                    )));
                }
                for (j, col) in cols.iter().enumerate() {
                    if col.len() < steps {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "exog_forecast_list[{}] column {} has {} rows but {} forecast steps requested",
                            i, j, col.len(), steps
                        )));
                    }
                }
            }
            Ok(Some(vecs))
        }
        None => Ok(None),
    }
}

/// Validate model order bounds and check for arithmetic overflow.
///
/// Consolidates all bounds checking (p, q, d, P, D, Q, s, n_exog) and
/// overflow checking (checked_add/checked_mul chains) into one place.
fn validate_order(
    p: usize,
    d: usize,
    q: usize,
    pp: usize,
    dd: usize,
    qq: usize,
    s: usize,
    n_exog: usize,
) -> PyResult<()> {
    // Individual order bounds (DoS prevention)
    if p > MAX_P {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "AR order p={} exceeds maximum {}",
            p, MAX_P
        )));
    }
    if q > MAX_Q {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "MA order q={} exceeds maximum {}",
            q, MAX_Q
        )));
    }
    if d > MAX_D {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "differencing order d={} exceeds maximum {}",
            d, MAX_D
        )));
    }
    if pp > MAX_PP {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "seasonal AR order P={} exceeds maximum {}",
            pp, MAX_PP
        )));
    }
    if qq > MAX_QQ {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "seasonal MA order Q={} exceeds maximum {}",
            qq, MAX_QQ
        )));
    }
    if dd > MAX_DD {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "seasonal differencing D={} exceeds maximum {}",
            dd, MAX_DD
        )));
    }
    if s > MAX_S {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "seasonal period s={} exceeds maximum {}",
            s, MAX_S
        )));
    }
    if n_exog > MAX_N_EXOG {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "n_exog={} exceeds maximum {}",
            n_exog, MAX_N_EXOG
        )));
    }
    if (pp > 0 || dd > 0 || qq > 0) && s < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "seasonal order (P,D,Q) > 0 requires seasonal period s >= 2",
        ));
    }

    // Overflow checking for state-space dimensions
    let k_ar = p
        .checked_add(s.checked_mul(pp).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("order overflow while computing k_ar")
        })?)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("order overflow while computing k_ar")
        })?;
    let k_ma = q
        .checked_add(s.checked_mul(qq).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("order overflow while computing k_ma")
        })?)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("order overflow while computing k_ma")
        })?;
    let k_order = std::cmp::max(
        k_ar,
        k_ma.checked_add(1).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("order overflow while computing k_order")
        })?,
    );
    let k_states_diff = d
        .checked_add(s.checked_mul(dd).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("order overflow while computing k_states_diff")
        })?)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("order overflow while computing k_states_diff")
        })?;
    let k_states = k_order.checked_add(k_states_diff).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("order overflow while computing k_states")
    })?;
    if k_states > MAX_K_STATES {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "model too large: k_states={} exceeds limit {}",
            k_states, MAX_K_STATES
        )));
    }

    Ok(())
}

/// Validate single-series exog columns: finite values and consistent length.
fn validate_exog(exog_cols: &Option<Vec<Vec<f64>>>) -> PyResult<()> {
    if let Some(ref cols) = exog_cols {
        validate_finite_cols(cols, "exog")?;
    }
    Ok(())
}

/// Validate batch exog list: matching lengths, finite values, consistent column counts,
/// and per-series row count matching endog length.
fn validate_batch_exog(
    exog_list: &Option<Vec<PyReadonlyArray2<'_, f64>>>,
    series_len: usize,
    series_lengths: &[usize],
) -> PyResult<(Option<Vec<Vec<Vec<f64>>>>, usize)> {
    parse_and_validate_batch_exog(exog_list, series_len, series_lengths)
}

/// Build SarimaxConfig from Python-facing tuples and flags.
///
/// Assumes `validate_order()` has already been called to verify bounds
/// and overflow safety.
fn build_config(
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    n_exog: usize,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    concentrate_scale: bool,
    trend: Trend,
    simple_differencing: bool,
) -> SarimaxConfig {
    let (p, d, q) = order;
    let (pp, dd, qq, s) = seasonal;

    SarimaxConfig {
        order: SarimaxOrder::new(p, d, q, pp, dd, qq, s),
        n_exog,
        trend,
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
        simple_differencing,
        measurement_error: false,
    }
}

/// Parse optional trend string from Python into Trend enum.
fn parse_trend(trend: Option<&str>) -> PyResult<Trend> {
    Trend::from_str(trend.unwrap_or("n"))
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Common validation and config-building for single-series PyO3 functions.
///
/// Validates endog (finite), parses and validates exog, checks order bounds,
/// and builds `SarimaxConfig`. Returns `(config, exog_cols)`.
fn prepare_single_request(
    endog: &[f64],
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    exog: Option<&PyReadonlyArray2<f64>>,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    concentrate_scale: bool,
    trend: Option<&str>,
    simple_differencing: bool,
) -> PyResult<(SarimaxConfig, Option<Vec<Vec<f64>>>)> {
    validate_endog_finite(endog)?;
    let (exog_cols, n_exog, exog_nrows) = parse_exog(exog);
    validate_exog(&exog_cols)?;
    // Early exog row-count validation at PyO3 boundary (shape-based, works for 0-col exog too)
    if exog_cols.is_some() && exog_nrows != endog.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "exog has {} rows but endog has {} observations; they must match",
            exog_nrows,
            endog.len()
        )));
    }
    let (p, d, q) = order;
    let (pp, dd, qq, s) = seasonal;
    validate_order(p, d, q, pp, dd, qq, s, n_exog)?;
    let config = build_config(
        order, seasonal, n_exog,
        enforce_stationarity, enforce_invertibility,
        concentrate_scale, parse_trend(trend)?, simple_differencing,
    );
    Ok((config, exog_cols))
}

/// Convert a FitResult to a PyDict with standard keys.
///
/// Warnings from the Rust fit path are emitted via Python's `warnings.warn()`
/// and also stored in the `"warnings"` key (V8.5 P2-1).
fn fit_result_to_pydict<'py>(py: Python<'py>, result: &crate::types::FitResult) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("params", &result.params)?;
    dict.set_item("loglike", result.loglike)?;
    dict.set_item("scale", result.scale)?;
    dict.set_item("aic", result.aic)?;
    dict.set_item("bic", result.bic)?;
    dict.set_item("n_obs", result.n_obs)?;
    dict.set_item("n_params", result.n_params)?;
    dict.set_item("n_iter", result.n_iter)?;
    dict.set_item("converged", result.converged)?;
    dict.set_item("method", &result.method)?;
    dict.set_item("warnings", &result.warnings)?;

    // Emit warnings via Python warnings module
    for msg in &result.warnings {
        let warn_mod = py.import("warnings")?;
        warn_mod.call_method1("warn", (msg.as_str(),))?;
    }

    Ok(dict)
}

/// Convert a ForecastResult to a PyDict.
fn forecast_to_pydict<'py>(py: Python<'py>, result: &crate::forecast::ForecastResult) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("mean", &result.mean)?;
    dict.set_item("variance", &result.variance)?;
    dict.set_item("ci_lower", &result.ci_lower)?;
    dict.set_item("ci_upper", &result.ci_upper)?;
    Ok(dict)
}

/// Convert a ResidualOutput to a PyDict.
fn residuals_to_pydict<'py>(py: Python<'py>, result: &crate::forecast::ResidualOutput) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("residuals", &result.residuals)?;
    dict.set_item("standardized_residuals", &result.standardized_residuals)?;
    Ok(dict)
}

/// Convert an InferenceResult to a PyDict.
fn inference_to_pydict<'py>(py: Python<'py>, result: &crate::inference::InferenceResult) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("method", &result.method)?;
    dict.set_item("std_err", &result.std_err)?;
    dict.set_item("z_stat", &result.z_stat)?;
    dict.set_item("p_value", &result.p_value)?;
    dict.set_item("ci_lower", &result.ci_lower)?;
    dict.set_item("ci_upper", &result.ci_upper)?;
    dict.set_item("cov_params", &result.cov_params)?;
    dict.set_item("n_params", result.n_params)?;
    dict.set_item("status", &result.status)?;
    dict.set_item("message", &result.message)?;
    Ok(dict)
}

/// Convert a DiagnosticsResult to a PyDict.
fn diagnostics_to_pydict<'py>(py: Python<'py>, result: &crate::inference::DiagnosticsResult) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("ljung_box_stat", result.ljung_box_stat)?;
    dict.set_item("ljung_box_pvalue", result.ljung_box_pvalue)?;
    dict.set_item("ljung_box_df", result.ljung_box_df)?;
    dict.set_item("jarque_bera_stat", result.jarque_bera_stat)?;
    dict.set_item("jarque_bera_pvalue", result.jarque_bera_pvalue)?;
    dict.set_item("het_stat", result.het_stat)?;
    dict.set_item("het_pvalue", result.het_pvalue)?;
    Ok(dict)
}

/// Validate that alpha is in the open interval (0, 1).
fn validate_alpha(alpha: f64) -> PyResult<()> {
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "alpha must be in (0, 1), got {}",
            alpha
        )));
    }
    Ok(())
}

/// Validate that forecast steps do not exceed the maximum.
fn validate_steps(steps: usize) -> PyResult<()> {
    if steps > 10_000 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "steps must be <= 10000, got {}",
            steps
        )));
    }
    Ok(())
}

/// Map internal SarimaxError to appropriate Python exception type.
fn to_pyerr(e: crate::error::SarimaxError) -> PyErr {
    use crate::error::SarimaxError;
    match &e {
        SarimaxError::OptimizationFailed(_) | SarimaxError::CholeskyFailed => {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        }
        _ => pyo3::exceptions::PyValueError::new_err(e.to_string()),
    }
}

// ---------------------------------------------------------------------------
// PyO3 functions
// ---------------------------------------------------------------------------

/// Smoke-test function: returns the version string.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Compute the SARIMAX concentrated (or full) log-likelihood.
#[pyfunction]
#[pyo3(signature = (y, order, seasonal, params, exog=None, concentrate_scale=true,
                    enforce_stationarity=true, enforce_invertibility=true,
                    trend=None, simple_differencing=false))]
fn sarimax_loglike<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    params: PyReadonlyArray1<'py, f64>,
    exog: Option<PyReadonlyArray2<'py, f64>>,
    concentrate_scale: bool,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    trend: Option<&str>,
    simple_differencing: bool,
) -> PyResult<f64> {
    let endog = y.as_slice()?;
    let params_flat = params.as_slice()?;
    let (config, exog_cols) = prepare_single_request(
        endog, order, seasonal, exog.as_ref(),
        enforce_stationarity, enforce_invertibility,
        concentrate_scale, trend, simple_differencing,
    )?;

    // Own all data before releasing GIL
    let endog = endog.to_vec();
    let params_flat = params_flat.to_vec();

    py.detach(move || {
        let sarimax_params = SarimaxParams::from_flat(&params_flat, &config)?;
        pipeline::kalman_eval(&endog, &sarimax_params, &config, exog_cols.as_deref())
            .map(|o| o.loglike)
    })
    .map_err(to_pyerr)
}

/// Fit a SARIMAX model via MLE (L-BFGS-B default, with Nelder-Mead fallback).
///
/// Returns a dict with: params, loglike, scale, aic, bic, n_obs, n_iter, converged, method.
#[pyfunction]
#[pyo3(signature = (y, order, seasonal, start_params=None, exog=None,
                    concentrate_scale=true, enforce_stationarity=true,
                    enforce_invertibility=true, method=None, maxiter=None,
                    trend=None, simple_differencing=false))]
fn sarimax_fit<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    start_params: Option<PyReadonlyArray1<'py, f64>>,
    exog: Option<PyReadonlyArray2<'py, f64>>,
    concentrate_scale: bool,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    method: Option<&str>,
    maxiter: Option<u64>,
    trend: Option<&str>,
    simple_differencing: bool,
) -> PyResult<Py<PyDict>> {
    let endog = y.as_slice()?;
    let (config, exog_cols) = prepare_single_request(
        endog, order, seasonal, exog.as_ref(),
        enforce_stationarity, enforce_invertibility,
        concentrate_scale, trend, simple_differencing,
    )?;

    let sp_owned: Option<Vec<f64>> = start_params
        .as_ref()
        .map(|a| a.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    let method_owned = method.map(|s| s.to_string());

    // Own all data before releasing GIL
    let endog = endog.to_vec();

    let result = py
        .detach(move || {
            optimizer::fit(
                &endog,
                &config,
                sp_owned.as_deref(),
                method_owned.as_deref(),
                maxiter,
                exog_cols.as_deref(),
            )
        })
        .map_err(to_pyerr)?;

    let dict = fit_result_to_pydict(py, &result)?;

    Ok(dict.into())
}

/// Compute h-step ahead forecast for a SARIMAX model.
///
/// Returns a dict with: mean, variance, ci_lower, ci_upper.
#[pyfunction]
#[pyo3(signature = (y, order, seasonal, params, steps=10, alpha=0.05,
                    exog=None, future_exog=None, concentrate_scale=true,
                    trend=None, simple_differencing=false))]
fn sarimax_forecast<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    params: PyReadonlyArray1<'py, f64>,
    steps: usize,
    alpha: f64,
    exog: Option<PyReadonlyArray2<'py, f64>>,
    future_exog: Option<PyReadonlyArray2<'py, f64>>,
    concentrate_scale: bool,
    trend: Option<&str>,
    simple_differencing: bool,
) -> PyResult<Py<PyDict>> {
    validate_alpha(alpha)?;
    validate_steps(steps)?;

    let endog = y.as_slice()?;
    let params_flat = params.as_slice()?;
    let (config, exog_cols) = prepare_single_request(
        endog, order, seasonal, exog.as_ref(),
        false, false, concentrate_scale, trend, simple_differencing,
    )?;

    let future_exog_cols = future_exog.as_ref().map(|e| numpy2d_to_cols(e));
    if let Some(ref cols) = future_exog_cols {
        validate_finite_cols(cols, "future_exog")?;
    }

    let n_exog = config.n_exog;
    if let Some(ref fec) = future_exog_cols {
        if fec.len() != n_exog {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "future_exog column count ({}) must match exog column count ({})",
                fec.len(),
                n_exog
            )));
        }
    }

    // Validate future_exog has enough rows for forecast steps
    if let Some(ref fec) = future_exog_cols {
        for (j, col) in fec.iter().enumerate() {
            if col.len() < steps {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "future_exog column {} has {} rows but {} forecast steps requested",
                    j,
                    col.len(),
                    steps
                )));
            }
        }
    }

    // Require future exog for exog models
    if n_exog > 0 && steps > 0 && future_exog.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "model has exogenous variables: future_exog is required for forecasting",
        ));
    }

    // Own all data before releasing GIL
    let endog = endog.to_vec();
    let params_flat = params_flat.to_vec();

    let result = py
        .detach(move || {
            let sarimax_params = SarimaxParams::from_flat(&params_flat, &config)?;
            forecast::forecast_pipeline(
                &endog,
                &config,
                &sarimax_params,
                steps,
                alpha,
                exog_cols.as_deref(),
                future_exog_cols.as_deref(),
            )
        })
        .map_err(to_pyerr)?;

    Ok(forecast_to_pydict(py, &result)?.into())
}

/// Compute residuals and standardized residuals for a SARIMAX model.
///
/// Returns a dict with: residuals, standardized_residuals.
#[pyfunction]
#[pyo3(signature = (y, order, seasonal, params, exog=None, concentrate_scale=true, trend=None, simple_differencing=false))]
fn sarimax_residuals<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    params: PyReadonlyArray1<'py, f64>,
    exog: Option<PyReadonlyArray2<'py, f64>>,
    concentrate_scale: bool,
    trend: Option<&str>,
    simple_differencing: bool,
) -> PyResult<Py<PyDict>> {
    let endog = y.as_slice()?;
    let params_flat = params.as_slice()?;
    let (config, exog_cols) = prepare_single_request(
        endog, order, seasonal, exog.as_ref(),
        false, false, concentrate_scale, trend, simple_differencing,
    )?;

    // Own all data before releasing GIL
    let endog = endog.to_vec();
    let params_flat = params_flat.to_vec();

    let result = py
        .detach(move || {
            let sarimax_params = SarimaxParams::from_flat(&params_flat, &config)?;
            forecast::residuals_pipeline(&endog, &config, &sarimax_params, exog_cols.as_deref())
        })
        .map_err(to_pyerr)?;

    Ok(residuals_to_pydict(py, &result)?.into())
}

/// Compute log-likelihood for multiple time series in parallel (Rayon).
///
/// All series share the same model config and parameters.
/// Returns a list of dicts, each with "loglike" key or "error" key.
#[pyfunction]
#[pyo3(signature = (series_list, order, seasonal, params,
                    exog_list=None, concentrate_scale=true,
                    enforce_stationarity=false, enforce_invertibility=false,
                    trend=None, simple_differencing=false))]
fn sarimax_batch_loglike<'py>(
    py: Python<'py>,
    series_list: Vec<PyReadonlyArray1<'py, f64>>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    params: PyReadonlyArray1<'py, f64>,
    exog_list: Option<Vec<PyReadonlyArray2<'py, f64>>>,
    concentrate_scale: bool,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    trend: Option<&str>,
    simple_differencing: bool,
) -> PyResult<Py<PyList>> {
    let series = validate_batch_finite(&series_list)?;
    let series_lengths: Vec<usize> = series.iter().map(|s| s.len()).collect();
    let (exog_vecs, n_exog) = validate_batch_exog(&exog_list, series_list.len(), &series_lengths)?;

    let (p, d, q) = order;
    let (pp, dd, qq, s) = seasonal;
    validate_order(p, d, q, pp, dd, qq, s, n_exog)?;
    let config = build_config(
        order,
        seasonal,
        n_exog,
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
        parse_trend(trend)?,
        simple_differencing,
    );

    let params_flat = params.as_slice()?.to_vec();
    let sarimax_params = SarimaxParams::from_flat(&params_flat, &config).map_err(to_pyerr)?;

    // Release GIL for Rayon parallel computation
    let results =
        py.detach(|| batch::batch_loglike(&series, &config, &sarimax_params, exog_vecs.as_deref()));

    let mut py_results: Vec<Py<PyDict>> = Vec::with_capacity(results.len());
    for r in results {
        let dict = PyDict::new(py);
        match r {
            Ok(loglike) => {
                dict.set_item("loglike", loglike)?;
            }
            Err(e) => {
                dict.set_item("error", e.to_string())?;
            }
        }
        py_results.push(dict.into());
    }

    let list = PyList::new(py, &py_results)?;
    Ok(list.into())
}

/// Fit SARIMAX models to multiple time series in parallel (Rayon).
///
/// Returns a list of dicts (one per series), each with the same keys as sarimax_fit.
/// Failed series return a dict with "error" key instead.
#[pyfunction]
#[pyo3(signature = (series_list, order, seasonal,
                    enforce_stationarity=true, enforce_invertibility=true,
                    concentrate_scale=true, method=None, maxiter=None,
                    exog_list=None, trend=None, simple_differencing=false))]
fn sarimax_batch_fit<'py>(
    py: Python<'py>,
    series_list: Vec<PyReadonlyArray1<'py, f64>>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    concentrate_scale: bool,
    method: Option<&str>,
    maxiter: Option<u64>,
    exog_list: Option<Vec<PyReadonlyArray2<'py, f64>>>,
    trend: Option<&str>,
    simple_differencing: bool,
) -> PyResult<Py<PyList>> {
    let series = validate_batch_finite(&series_list)?;
    let series_lengths: Vec<usize> = series.iter().map(|s| s.len()).collect();
    let (exog_vecs, n_exog) = validate_batch_exog(&exog_list, series_list.len(), &series_lengths)?;

    let (p, d, q) = order;
    let (pp, dd, qq, s) = seasonal;
    validate_order(p, d, q, pp, dd, qq, s, n_exog)?;
    let config = build_config(
        order,
        seasonal,
        n_exog,
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
        parse_trend(trend)?,
        simple_differencing,
    );

    // Release GIL for Rayon parallel computation
    let method_owned = method.map(|s| s.to_string());
    let results = py.detach(|| {
        batch::batch_fit(
            &series,
            &config,
            method_owned.as_deref(),
            maxiter,
            exog_vecs.as_deref(),
        )
    });

    let mut py_results: Vec<Py<PyDict>> = Vec::with_capacity(results.len());
    for r in results {
        match r {
            Ok(result) => {
                let dict = fit_result_to_pydict(py, &result)?;
                py_results.push(dict.into());
            }
            Err(e) => {
                let dict = PyDict::new(py);
                dict.set_item("error", e.to_string())?;
                dict.set_item("converged", false)?;
                py_results.push(dict.into());
            }
        }
    }

    let list = PyList::new(py, &py_results)?;
    Ok(list.into())
}

/// Fit a single time series with multiple (order, seasonal_order) combinations
/// in parallel using Rayon.
///
/// Returns a list of dicts (one per combination), same structure as sarimax_fit.
/// Failed combinations have an "error" key instead of fit results.
#[pyfunction]
#[pyo3(signature = (y, order_list, seasonal_list,
                    enforce_stationarity=true, enforce_invertibility=true,
                    concentrate_scale=true, method=None, maxiter=None,
                    exog=None, trend=None, simple_differencing=false))]
fn sarimax_grid_search<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    order_list: Vec<(usize, usize, usize)>,
    seasonal_list: Vec<(usize, usize, usize, usize)>,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    concentrate_scale: bool,
    method: Option<&str>,
    maxiter: Option<u64>,
    exog: Option<PyReadonlyArray2<'py, f64>>,
    trend: Option<&str>,
    simple_differencing: bool,
) -> PyResult<Py<PyList>> {
    if order_list.len() != seasonal_list.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "order_list length ({}) must match seasonal_list length ({})",
            order_list.len(),
            seasonal_list.len()
        )));
    }

    let endog = y.as_slice()?;
    validate_endog_finite(endog)?;
    let (exog_cols, n_exog, exog_nrows) = parse_exog(exog.as_ref());
    validate_exog(&exog_cols)?;
    if exog_cols.is_some() && exog_nrows != endog.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "exog has {} rows but endog has {} observations; they must match",
            exog_nrows,
            endog.len()
        )));
    }

    let trend_val = parse_trend(trend)?;

    // Validate each combination and build configs
    let mut configs: Vec<SarimaxConfig> = Vec::with_capacity(order_list.len());
    for (order, seasonal) in order_list.iter().zip(seasonal_list.iter()) {
        let (p, d, q) = *order;
        let (pp, dd, qq, s) = *seasonal;
        validate_order(p, d, q, pp, dd, qq, s, n_exog)?;
        configs.push(build_config(
            *order,
            *seasonal,
            n_exog,
            enforce_stationarity,
            enforce_invertibility,
            concentrate_scale,
            trend_val,
            simple_differencing,
        ));
    }

    // Release GIL for Rayon parallel computation
    let endog_owned = endog.to_vec();
    let method_owned = method.map(|s| s.to_string());
    let results = py.detach(|| {
        batch::grid_search_fit(
            &endog_owned,
            &configs,
            method_owned.as_deref(),
            maxiter,
            exog_cols.as_deref(),
        )
    });

    let mut py_results: Vec<Py<PyDict>> = Vec::with_capacity(results.len());
    for (idx, r) in results.into_iter().enumerate() {
        match r {
            Ok(result) => {
                let dict = fit_result_to_pydict(py, &result)?;
                dict.set_item("order", order_list[idx])?;
                dict.set_item("seasonal_order", seasonal_list[idx])?;
                py_results.push(dict.into());
            }
            Err(e) => {
                let dict = PyDict::new(py);
                dict.set_item("order", order_list[idx])?;
                dict.set_item("seasonal_order", seasonal_list[idx])?;
                dict.set_item("error", e.to_string())?;
                dict.set_item("converged", false)?;
                py_results.push(dict.into());
            }
        }
    }

    let list = PyList::new(py, &py_results)?;
    Ok(list.into())
}

/// Forecast multiple time series in parallel (Rayon).
///
/// Each series uses its own parameter vector from params_list.
/// Returns a list of dicts (one per series).
#[pyfunction]
#[pyo3(signature = (series_list, order, seasonal, params_list,
                    steps=10, alpha=0.05, concentrate_scale=true,
                    exog_list=None, exog_forecast_list=None, trend=None,
                    simple_differencing=false))]
fn sarimax_batch_forecast<'py>(
    py: Python<'py>,
    series_list: Vec<PyReadonlyArray1<'py, f64>>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    params_list: Vec<PyReadonlyArray1<'py, f64>>,
    steps: usize,
    alpha: f64,
    concentrate_scale: bool,
    exog_list: Option<Vec<PyReadonlyArray2<'py, f64>>>,
    exog_forecast_list: Option<Vec<PyReadonlyArray2<'py, f64>>>,
    trend: Option<&str>,
    simple_differencing: bool,
) -> PyResult<Py<PyList>> {
    if series_list.len() != params_list.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "series_list and params_list must have same length: {} vs {}",
            series_list.len(),
            params_list.len()
        )));
    }
    validate_steps(steps)?;
    validate_alpha(alpha)?;

    let series = validate_batch_finite(&series_list)?;
    let series_lengths: Vec<usize> = series.iter().map(|s| s.len()).collect();
    let (exog_vecs, n_exog) = validate_batch_exog(&exog_list, series_list.len(), &series_lengths)?;
    let future_exog_vecs = parse_and_validate_batch_forecast_exog(
        &exog_forecast_list,
        series_list.len(),
        n_exog,
        steps,
    )?;

    let (p, d, q) = order;
    let (pp, dd, qq, s) = seasonal;
    validate_order(p, d, q, pp, dd, qq, s, n_exog)?;
    let config = build_config(order, seasonal, n_exog, false, false, concentrate_scale, parse_trend(trend)?, simple_differencing);

    let params_vecs: Vec<Vec<f64>> = params_list
        .iter()
        .map(|a| a.as_slice().map(|s| s.to_vec()))
        .collect::<std::result::Result<Vec<_>, _>>()?;

    // Release GIL for Rayon parallel computation
    let results = py.detach(|| {
        batch::batch_forecast(
            &series,
            &config,
            &params_vecs,
            steps,
            alpha,
            exog_vecs.as_deref(),
            future_exog_vecs.as_deref(),
        )
    });

    let mut py_results: Vec<Py<PyDict>> = Vec::with_capacity(results.len());
    for r in results {
        match r {
            Ok(result) => {
                py_results.push(forecast_to_pydict(py, &result)?.into());
            }
            Err(e) => {
                let dict = PyDict::new(py);
                dict.set_item("error", e.to_string())?;
                py_results.push(dict.into());
            }
        }
    }

    let list = PyList::new(py, &py_results)?;
    Ok(list.into())
}

/// Compute inference statistics (standard errors, z, p-values, CI) for fitted parameters.
///
/// Returns a dict with: method, std_err, z_stat, p_value, ci_lower, ci_upper,
/// cov_params, n_params, status, message.
#[pyfunction]
#[pyo3(signature = (y, order, seasonal, params, method="hessian", alpha=0.05,
                    exog=None, concentrate_scale=true,
                    enforce_stationarity=true, enforce_invertibility=true,
                    trend=None, simple_differencing=false))]
fn sarimax_inference<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    params: PyReadonlyArray1<'py, f64>,
    method: &str,
    alpha: f64,
    exog: Option<PyReadonlyArray2<'py, f64>>,
    concentrate_scale: bool,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    trend: Option<&str>,
    simple_differencing: bool,
) -> PyResult<Py<PyDict>> {
    validate_alpha(alpha)?;

    let endog = y.as_slice()?;
    let params_flat = params.as_slice()?;
    let (config, exog_cols) = prepare_single_request(
        endog, order, seasonal, exog.as_ref(),
        enforce_stationarity, enforce_invertibility,
        concentrate_scale, trend, simple_differencing,
    )?;

    // Own all data before releasing GIL
    let endog = endog.to_vec();
    let params_flat = params_flat.to_vec();
    let method_owned = method.to_string();

    let result = py
        .detach(move || {
            inference::compute_inference(
                &endog,
                &config,
                &params_flat,
                &method_owned,
                alpha,
                exog_cols.as_deref(),
            )
        })
        .map_err(to_pyerr)?;

    Ok(inference_to_pydict(py, &result)?.into())
}

/// Compute residual diagnostic tests (Ljung-Box, Jarque-Bera, heteroskedasticity).
///
/// Returns a dict with: ljung_box_stat, ljung_box_pvalue, ljung_box_df,
/// jarque_bera_stat, jarque_bera_pvalue, het_stat, het_pvalue.
#[pyfunction]
#[pyo3(signature = (y, order, seasonal, params, exog=None, concentrate_scale=true, trend=None, simple_differencing=false))]
fn sarimax_diagnostics<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    params: PyReadonlyArray1<'py, f64>,
    exog: Option<PyReadonlyArray2<'py, f64>>,
    concentrate_scale: bool,
    trend: Option<&str>,
    simple_differencing: bool,
) -> PyResult<Py<PyDict>> {
    let endog = y.as_slice()?;
    let params_flat = params.as_slice()?;
    let (config, exog_cols) = prepare_single_request(
        endog, order, seasonal, exog.as_ref(),
        false, false, concentrate_scale, trend, simple_differencing,
    )?;

    // Own all data before releasing GIL
    let endog = endog.to_vec();
    let params_flat = params_flat.to_vec();

    let result = py
        .detach(move || -> std::result::Result<_, crate::error::SarimaxError> {
            let sarimax_params = SarimaxParams::from_flat(&params_flat, &config)?;
            let resid_result = forecast::residuals_pipeline(
                &endog, &config, &sarimax_params, exog_cols.as_deref(),
            )?;
            let n_params = params_flat.len();
            Ok(inference::compute_diagnostics(
                &resid_result.standardized_residuals,
                n_params,
            ))
        })
        .map_err(to_pyerr)?;

    Ok(diagnostics_to_pydict(py, &result)?.into())
}

/// Python module definition.
#[pymodule]
fn rustima(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(sarimax_loglike, m)?)?;
    m.add_function(wrap_pyfunction!(sarimax_fit, m)?)?;
    m.add_function(wrap_pyfunction!(sarimax_forecast, m)?)?;
    m.add_function(wrap_pyfunction!(sarimax_residuals, m)?)?;
    m.add_function(wrap_pyfunction!(sarimax_batch_loglike, m)?)?;
    m.add_function(wrap_pyfunction!(sarimax_batch_fit, m)?)?;
    m.add_function(wrap_pyfunction!(sarimax_grid_search, m)?)?;
    m.add_function(wrap_pyfunction!(sarimax_batch_forecast, m)?)?;
    m.add_function(wrap_pyfunction!(sarimax_inference, m)?)?;
    m.add_function(wrap_pyfunction!(sarimax_diagnostics, m)?)?;
    Ok(())
}
