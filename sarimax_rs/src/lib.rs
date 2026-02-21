pub mod batch;
pub mod error;
pub mod forecast;
pub mod initialization;
pub mod kalman;
pub mod optimizer;
pub mod params;
pub mod polynomial;
pub mod score;
pub mod start_params;
pub mod state_space;
pub mod types;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::initialization::KalmanInit;
use crate::params::SarimaxParams;
use crate::state_space::StateSpace;
use crate::types::{SarimaxConfig, SarimaxOrder, Trend};

const MAX_K_STATES: usize = 1024;

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

/// Parse optional exog numpy array → (column-major vecs, n_exog).
fn parse_exog(exog: Option<&PyReadonlyArray2<f64>>) -> (Option<Vec<Vec<f64>>>, usize) {
    match exog {
        Some(e) => {
            let cols = numpy2d_to_cols(e);
            let n = cols.len();
            (Some(cols), n)
        }
        None => (None, 0),
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

/// Build SarimaxConfig from Python-facing tuples and flags.
fn build_config(
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    n_exog: usize,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    concentrate_scale: bool,
) -> PyResult<SarimaxConfig> {
    let (p, d, q) = order;
    let (pp, dd, qq, s) = seasonal;
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

    Ok(SarimaxConfig {
        order: SarimaxOrder::new(p, d, q, pp, dd, qq, s),
        n_exog,
        trend: Trend::None,
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
        simple_differencing: false,
        measurement_error: false,
    })
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
                    enforce_stationarity=true, enforce_invertibility=true))]
fn sarimax_loglike<'py>(
    _py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    params: PyReadonlyArray1<'py, f64>,
    exog: Option<PyReadonlyArray2<'py, f64>>,
    concentrate_scale: bool,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
) -> PyResult<f64> {
    let endog = y.as_slice()?;
    if endog.iter().any(|v| !v.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "input time series contains NaN or Inf values",
        ));
    }
    let params_flat = params.as_slice()?;
    let (exog_cols, n_exog) = parse_exog(exog.as_ref());
    if let Some(ref cols) = exog_cols {
        validate_finite_cols(cols, "exog")?;
    }
    let config = build_config(
        order,
        seasonal,
        n_exog,
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
    )?;

    let sarimax_params = SarimaxParams::from_flat(params_flat, &config).map_err(to_pyerr)?;

    let ss =
        StateSpace::new(&config, &sarimax_params, endog, exog_cols.as_deref()).map_err(to_pyerr)?;
    let init = KalmanInit::from_config(&ss, &config, KalmanInit::default_kappa());
    let output = kalman::kalman_loglike(endog, &ss, &init, concentrate_scale).map_err(to_pyerr)?;

    Ok(output.loglike)
}

/// Fit a SARIMAX model via MLE (L-BFGS-B default, with Nelder-Mead fallback).
///
/// Returns a dict with: params, loglike, scale, aic, bic, n_obs, n_iter, converged, method.
#[pyfunction]
#[pyo3(signature = (y, order, seasonal, start_params=None, exog=None,
                    concentrate_scale=true, enforce_stationarity=true,
                    enforce_invertibility=true, method=None, maxiter=None))]
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
) -> PyResult<Py<PyDict>> {
    let endog = y.as_slice()?;
    if endog.iter().any(|v| !v.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "input time series contains NaN or Inf values",
        ));
    }
    let (exog_cols, n_exog) = parse_exog(exog.as_ref());
    if let Some(ref cols) = exog_cols {
        validate_finite_cols(cols, "exog")?;
    }
    let config = build_config(
        order,
        seasonal,
        n_exog,
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
    )?;

    let sp = start_params.as_ref().map(|a| a.as_slice()).transpose()?;

    let result = optimizer::fit(endog, &config, sp, method, maxiter, exog_cols.as_deref())
        .map_err(to_pyerr)?;

    let dict = PyDict::new(py);
    dict.set_item("params", result.params)?;
    dict.set_item("loglike", result.loglike)?;
    dict.set_item("scale", result.scale)?;
    dict.set_item("aic", result.aic)?;
    dict.set_item("bic", result.bic)?;
    dict.set_item("n_obs", result.n_obs)?;
    dict.set_item("n_params", result.n_params)?;
    dict.set_item("n_iter", result.n_iter)?;
    dict.set_item("converged", result.converged)?;
    dict.set_item("method", result.method)?;

    Ok(dict.into())
}

/// Compute h-step ahead forecast for a SARIMAX model.
///
/// Returns a dict with: mean, variance, ci_lower, ci_upper.
#[pyfunction]
#[pyo3(signature = (y, order, seasonal, params, steps=10, alpha=0.05,
                    exog=None, exog_forecast=None, concentrate_scale=true))]
fn sarimax_forecast<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    params: PyReadonlyArray1<'py, f64>,
    steps: usize,
    alpha: f64,
    exog: Option<PyReadonlyArray2<'py, f64>>,
    exog_forecast: Option<PyReadonlyArray2<'py, f64>>,
    concentrate_scale: bool,
) -> PyResult<Py<PyDict>> {
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "alpha must be in (0, 1), got {}",
            alpha
        )));
    }
    if steps > 10_000 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "steps must be <= 10000, got {}",
            steps
        )));
    }

    let endog = y.as_slice()?;
    if endog.iter().any(|v| !v.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "input time series contains NaN or Inf values",
        ));
    }
    let params_flat = params.as_slice()?;
    let (exog_cols, n_exog) = parse_exog(exog.as_ref());
    let future_exog_cols = exog_forecast.as_ref().map(|e| numpy2d_to_cols(e));
    if let Some(ref cols) = exog_cols {
        validate_finite_cols(cols, "exog")?;
    }
    if let Some(ref cols) = future_exog_cols {
        validate_finite_cols(cols, "exog_forecast")?;
    }

    if let Some(ref fec) = future_exog_cols {
        if fec.len() != n_exog {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "exog_forecast column count ({}) must match exog column count ({})",
                fec.len(),
                n_exog
            )));
        }
    }

    // V-6: Validate exog_forecast has enough rows for forecast steps
    if let Some(ref fec) = future_exog_cols {
        for (j, col) in fec.iter().enumerate() {
            if col.len() < steps {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "exog_forecast column {} has {} rows but {} forecast steps requested",
                    j,
                    col.len(),
                    steps
                )));
            }
        }
    }

    // Require future exog for exog models
    if n_exog > 0 && steps > 0 && exog_forecast.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "model has exogenous variables: exog_forecast is required for forecasting",
        ));
    }

    let config = build_config(order, seasonal, n_exog, false, false, concentrate_scale)?;
    let sarimax_params = SarimaxParams::from_flat(params_flat, &config).map_err(to_pyerr)?;

    let result = forecast::forecast_pipeline(
        endog,
        &config,
        &sarimax_params,
        steps,
        alpha,
        exog_cols.as_deref(),
        future_exog_cols.as_deref(),
    )
    .map_err(to_pyerr)?;

    let dict = PyDict::new(py);
    dict.set_item("mean", result.mean)?;
    dict.set_item("variance", result.variance)?;
    dict.set_item("ci_lower", result.ci_lower)?;
    dict.set_item("ci_upper", result.ci_upper)?;

    Ok(dict.into())
}

/// Compute residuals and standardized residuals for a SARIMAX model.
///
/// Returns a dict with: residuals, standardized_residuals.
#[pyfunction]
#[pyo3(signature = (y, order, seasonal, params, exog=None, concentrate_scale=true))]
fn sarimax_residuals<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    params: PyReadonlyArray1<'py, f64>,
    exog: Option<PyReadonlyArray2<'py, f64>>,
    concentrate_scale: bool,
) -> PyResult<Py<PyDict>> {
    let endog = y.as_slice()?;
    if endog.iter().any(|v| !v.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "input time series contains NaN or Inf values",
        ));
    }
    let params_flat = params.as_slice()?;
    let (exog_cols, n_exog) = parse_exog(exog.as_ref());
    if let Some(ref cols) = exog_cols {
        validate_finite_cols(cols, "exog")?;
    }
    let config = build_config(order, seasonal, n_exog, false, false, concentrate_scale)?;
    let sarimax_params = SarimaxParams::from_flat(params_flat, &config).map_err(to_pyerr)?;

    let result =
        forecast::residuals_pipeline(endog, &config, &sarimax_params, exog_cols.as_deref())
            .map_err(to_pyerr)?;

    let dict = PyDict::new(py);
    dict.set_item("residuals", result.residuals)?;
    dict.set_item("standardized_residuals", result.standardized_residuals)?;

    Ok(dict.into())
}

/// Fit SARIMAX models to multiple time series in parallel (Rayon).
///
/// Returns a list of dicts (one per series), each with the same keys as sarimax_fit.
/// Failed series return a dict with "error" key instead.
#[pyfunction]
#[pyo3(signature = (series_list, order, seasonal,
                    enforce_stationarity=true, enforce_invertibility=true,
                    concentrate_scale=true, method=None, maxiter=None,
                    exog_list=None))]
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
) -> PyResult<Py<PyList>> {
    // V-1: NaN/Inf check for each series
    for (i, s) in series_list.iter().enumerate() {
        let sl = s.as_slice()?;
        if sl.iter().any(|v| !v.is_finite()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "input time series at index {} contains NaN or Inf values",
                i
            )));
        }
    }

    // V-2: exog_list length must match series_list length
    if let Some(ref el) = exog_list {
        if el.len() != series_list.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "exog_list length ({}) must match series_list length ({})",
                el.len(),
                series_list.len()
            )));
        }
    }

    let exog_vecs: Option<Vec<Vec<Vec<f64>>>> = exog_list
        .as_ref()
        .map(|el| el.iter().map(|e| numpy2d_to_cols(e)).collect());
    if let Some(ref ev) = exog_vecs {
        for (i, cols) in ev.iter().enumerate() {
            validate_finite_cols(cols, &format!("exog_list[{}]", i))?;
        }
    }
    let n_exog = exog_vecs
        .as_ref()
        .and_then(|v| v.first())
        .map_or(0, |c| c.len());

    // V-5: n_exog consistency across all batch series
    if let Some(ref ev) = exog_vecs {
        for (i, cols) in ev.iter().enumerate() {
            if cols.len() != n_exog {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "exog at index {} has {} columns but expected {} (from first series)",
                    i,
                    cols.len(),
                    n_exog
                )));
            }
        }
    }

    let config = build_config(
        order,
        seasonal,
        n_exog,
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
    )?;

    let series: Vec<Vec<f64>> = series_list
        .iter()
        .map(|a| a.as_slice().map(|s| s.to_vec()))
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let results = batch::batch_fit(&series, &config, method, maxiter, exog_vecs.as_deref());

    let mut py_results: Vec<Py<PyDict>> = Vec::with_capacity(results.len());
    for r in results {
        let dict = PyDict::new(py);
        match r {
            Ok(result) => {
                dict.set_item("params", result.params)?;
                dict.set_item("loglike", result.loglike)?;
                dict.set_item("scale", result.scale)?;
                dict.set_item("aic", result.aic)?;
                dict.set_item("bic", result.bic)?;
                dict.set_item("n_obs", result.n_obs)?;
                dict.set_item("n_params", result.n_params)?;
                dict.set_item("n_iter", result.n_iter)?;
                dict.set_item("converged", result.converged)?;
                dict.set_item("method", result.method)?;
            }
            Err(e) => {
                dict.set_item("error", e.to_string())?;
                dict.set_item("converged", false)?;
            }
        }
        py_results.push(dict.into());
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
                    exog_list=None, exog_forecast_list=None))]
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
) -> PyResult<Py<PyList>> {
    if series_list.len() != params_list.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "series_list and params_list must have same length: {} vs {}",
            series_list.len(),
            params_list.len()
        )));
    }
    if steps > 10_000 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "steps must be <= 10000, got {}",
            steps
        )));
    }

    // V-4: alpha validation (same as sarimax_forecast)
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "alpha must be between 0 and 1, got {}",
            alpha
        )));
    }

    // V-1: NaN/Inf check for each series
    for (i, s) in series_list.iter().enumerate() {
        let sl = s.as_slice()?;
        if sl.iter().any(|v| !v.is_finite()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "input time series at index {} contains NaN or Inf values",
                i
            )));
        }
    }

    // V-2: exog_list length must match series_list length
    if let Some(ref el) = exog_list {
        if el.len() != series_list.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "exog_list length ({}) must match series_list length ({})",
                el.len(),
                series_list.len()
            )));
        }
    }

    // V-2: exog_forecast_list length must match series_list length
    if let Some(ref el) = exog_forecast_list {
        if el.len() != series_list.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "exog_forecast_list length ({}) must match series_list length ({})",
                el.len(),
                series_list.len()
            )));
        }
    }

    let exog_vecs: Option<Vec<Vec<Vec<f64>>>> = exog_list
        .as_ref()
        .map(|el| el.iter().map(|e| numpy2d_to_cols(e)).collect());
    let future_exog_vecs: Option<Vec<Vec<Vec<f64>>>> = exog_forecast_list
        .as_ref()
        .map(|el| el.iter().map(|e| numpy2d_to_cols(e)).collect());
    if let Some(ref ev) = exog_vecs {
        for (i, cols) in ev.iter().enumerate() {
            validate_finite_cols(cols, &format!("exog_list[{}]", i))?;
        }
    }
    if let Some(ref ev) = future_exog_vecs {
        for (i, cols) in ev.iter().enumerate() {
            validate_finite_cols(cols, &format!("exog_forecast_list[{}]", i))?;
        }
    }
    let n_exog = exog_vecs
        .as_ref()
        .and_then(|v| v.first())
        .map_or(0, |c| c.len());

    // V-5: n_exog consistency across all batch series
    if let Some(ref ev) = exog_vecs {
        for (i, cols) in ev.iter().enumerate() {
            if cols.len() != n_exog {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "exog at index {} has {} columns but expected {} (from first series)",
                    i,
                    cols.len(),
                    n_exog
                )));
            }
        }
    }
    if let Some(ref ev) = future_exog_vecs {
        for (i, cols) in ev.iter().enumerate() {
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
    }

    let config = build_config(order, seasonal, n_exog, false, false, concentrate_scale)?;

    let series: Vec<Vec<f64>> = series_list
        .iter()
        .map(|a| a.as_slice().map(|s| s.to_vec()))
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let params_vecs: Vec<Vec<f64>> = params_list
        .iter()
        .map(|a| a.as_slice().map(|s| s.to_vec()))
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let results = batch::batch_forecast(
        &series,
        &config,
        &params_vecs,
        steps,
        alpha,
        exog_vecs.as_deref(),
        future_exog_vecs.as_deref(),
    );

    let mut py_results: Vec<Py<PyDict>> = Vec::with_capacity(results.len());
    for r in results {
        let dict = PyDict::new(py);
        match r {
            Ok(result) => {
                dict.set_item("mean", result.mean)?;
                dict.set_item("variance", result.variance)?;
                dict.set_item("ci_lower", result.ci_lower)?;
                dict.set_item("ci_upper", result.ci_upper)?;
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

/// Python module definition.
#[pymodule]
fn sarimax_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(sarimax_loglike, m)?)?;
    m.add_function(wrap_pyfunction!(sarimax_fit, m)?)?;
    m.add_function(wrap_pyfunction!(sarimax_forecast, m)?)?;
    m.add_function(wrap_pyfunction!(sarimax_residuals, m)?)?;
    m.add_function(wrap_pyfunction!(sarimax_batch_fit, m)?)?;
    m.add_function(wrap_pyfunction!(sarimax_batch_forecast, m)?)?;
    Ok(())
}
