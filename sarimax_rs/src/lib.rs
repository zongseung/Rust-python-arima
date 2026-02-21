pub mod error;
pub mod types;
pub mod params;
pub mod polynomial;
pub mod state_space;
pub mod initialization;
pub mod kalman;
pub mod start_params;
pub mod optimizer;
pub mod forecast;
pub mod batch;
// Phase 5+
// pub mod information;
// pub mod selection;
// pub mod diagnostics;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::initialization::KalmanInit;
use crate::params::SarimaxParams;
use crate::state_space::StateSpace;
use crate::types::{SarimaxConfig, SarimaxOrder, Trend};

/// Smoke-test function: returns the version string.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Compute the SARIMAX concentrated (or full) log-likelihood.
///
/// # Arguments
/// * `y` - Endogenous (observed) time series
/// * `order` - (p, d, q) ARIMA order
/// * `seasonal` - (P, D, Q, s) seasonal order
/// * `params` - Flat parameter vector [ar..., ma...]
/// * `exog` - Optional exogenous variables (not yet implemented)
/// * `concentrate_scale` - If true, concentrate sigma2 out of likelihood
#[pyfunction]
#[pyo3(signature = (y, order, seasonal, params, exog=None, concentrate_scale=true,
                    enforce_stationarity=false, enforce_invertibility=false))]
fn sarimax_loglike<'py>(
    _py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    params: PyReadonlyArray1<'py, f64>,
    exog: Option<PyReadonlyArray1<'py, f64>>,
    concentrate_scale: bool,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
) -> PyResult<f64> {
    if exog.is_some() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "exog (exogenous variables) is not yet supported"
        ));
    }

    let endog = y.as_slice()?;
    let params_flat = params.as_slice()?;

    let (p, d, q) = order;
    let (pp, dd, qq, s) = seasonal;

    let sarimax_order = SarimaxOrder::new(p, d, q, pp, dd, qq, s);

    let config = SarimaxConfig {
        order: sarimax_order,
        n_exog: 0,
        trend: Trend::None,
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
        simple_differencing: false,
        measurement_error: false,
    };

    // Validate params length before slicing
    let expected_len = p + q + pp + qq;
    if params_flat.len() != expected_len {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "params length mismatch: expected {} (p={} + q={} + P={} + Q={}), got {}",
            expected_len, p, q, pp, qq, params_flat.len()
        )));
    }

    // Split params: [ar(p), ma(q), sar(P), sma(Q)]
    let mut i = 0;
    let ar_coeffs = &params_flat[i..i + p]; i += p;
    let ma_coeffs = &params_flat[i..i + q]; i += q;
    let sar_coeffs = &params_flat[i..i + pp]; i += pp;
    let sma_coeffs = &params_flat[i..i + qq];

    let sarimax_params = SarimaxParams {
        trend_coeffs: vec![],
        exog_coeffs: vec![],
        ar_coeffs: ar_coeffs.to_vec(),
        ma_coeffs: ma_coeffs.to_vec(),
        sar_coeffs: sar_coeffs.to_vec(),
        sma_coeffs: sma_coeffs.to_vec(),
        sigma2: None,
    };

    let ss = StateSpace::new(&config, &sarimax_params, endog, None)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let init = KalmanInit::from_config(&ss, &config, KalmanInit::default_kappa());

    let output = kalman::kalman_loglike(endog, &ss, &init, concentrate_scale)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

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
    exog: Option<PyReadonlyArray1<'py, f64>>,
    concentrate_scale: bool,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    method: Option<&str>,
    maxiter: Option<u64>,
) -> PyResult<Py<PyDict>> {
    if exog.is_some() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "exog (exogenous variables) is not yet supported"
        ));
    }

    let endog = y.as_slice()?;

    let (p, d, q) = order;
    let (pp, dd, qq, s) = seasonal;

    let sarimax_order = SarimaxOrder::new(p, d, q, pp, dd, qq, s);

    let config = SarimaxConfig {
        order: sarimax_order,
        n_exog: 0,
        trend: Trend::None,
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
        simple_differencing: false,
        measurement_error: false,
    };

    let sp = start_params
        .as_ref()
        .map(|a| a.as_slice())
        .transpose()?;

    let result = optimizer::fit(endog, &config, sp, method, maxiter)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    let params_list: Vec<f64> = result.params;
    dict.set_item("params", params_list)?;
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
                    exog=None, concentrate_scale=true))]
fn sarimax_forecast<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    params: PyReadonlyArray1<'py, f64>,
    steps: usize,
    alpha: f64,
    exog: Option<PyReadonlyArray1<'py, f64>>,
    concentrate_scale: bool,
) -> PyResult<Py<PyDict>> {
    if exog.is_some() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "exog (exogenous variables) is not yet supported"
        ));
    }

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
    let params_flat = params.as_slice()?;

    let (p, d, q) = order;
    let (pp, dd, qq, s) = seasonal;

    let expected_len = p + q + pp + qq;
    if params_flat.len() != expected_len {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "params length mismatch: expected {} (p={} + q={} + P={} + Q={}), got {}",
            expected_len, p, q, pp, qq, params_flat.len()
        )));
    }

    let sarimax_order = SarimaxOrder::new(p, d, q, pp, dd, qq, s);

    let config = SarimaxConfig {
        order: sarimax_order,
        n_exog: 0,
        trend: Trend::None,
        enforce_stationarity: false,
        enforce_invertibility: false,
        concentrate_scale,
        simple_differencing: false,
        measurement_error: false,
    };

    let mut i = 0;
    let ar_coeffs = &params_flat[i..i + p]; i += p;
    let ma_coeffs = &params_flat[i..i + q]; i += q;
    let sar_coeffs = &params_flat[i..i + pp]; i += pp;
    let sma_coeffs = &params_flat[i..i + qq];

    let sarimax_params = SarimaxParams {
        trend_coeffs: vec![],
        exog_coeffs: vec![],
        ar_coeffs: ar_coeffs.to_vec(),
        ma_coeffs: ma_coeffs.to_vec(),
        sar_coeffs: sar_coeffs.to_vec(),
        sma_coeffs: sma_coeffs.to_vec(),
        sigma2: None,
    };

    let result = forecast::forecast_pipeline(endog, &config, &sarimax_params, steps, alpha)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

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
    exog: Option<PyReadonlyArray1<'py, f64>>,
    concentrate_scale: bool,
) -> PyResult<Py<PyDict>> {
    if exog.is_some() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "exog (exogenous variables) is not yet supported"
        ));
    }

    let endog = y.as_slice()?;
    let params_flat = params.as_slice()?;

    let (p, d, q) = order;
    let (pp, dd, qq, s) = seasonal;

    let expected_len = p + q + pp + qq;
    if params_flat.len() != expected_len {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "params length mismatch: expected {} (p={} + q={} + P={} + Q={}), got {}",
            expected_len, p, q, pp, qq, params_flat.len()
        )));
    }

    let sarimax_order = SarimaxOrder::new(p, d, q, pp, dd, qq, s);

    let config = SarimaxConfig {
        order: sarimax_order,
        n_exog: 0,
        trend: Trend::None,
        enforce_stationarity: false,
        enforce_invertibility: false,
        concentrate_scale,
        simple_differencing: false,
        measurement_error: false,
    };

    let mut i = 0;
    let ar_coeffs = &params_flat[i..i + p]; i += p;
    let ma_coeffs = &params_flat[i..i + q]; i += q;
    let sar_coeffs = &params_flat[i..i + pp]; i += pp;
    let sma_coeffs = &params_flat[i..i + qq];

    let sarimax_params = SarimaxParams {
        trend_coeffs: vec![],
        exog_coeffs: vec![],
        ar_coeffs: ar_coeffs.to_vec(),
        ma_coeffs: ma_coeffs.to_vec(),
        sar_coeffs: sar_coeffs.to_vec(),
        sma_coeffs: sma_coeffs.to_vec(),
        sigma2: None,
    };

    let result = forecast::residuals_pipeline(endog, &config, &sarimax_params)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

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
                    concentrate_scale=true, method=None, maxiter=None))]
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
) -> PyResult<Py<PyList>> {
    let (p, d, q) = order;
    let (pp, dd, qq, s) = seasonal;

    let config = SarimaxConfig {
        order: SarimaxOrder::new(p, d, q, pp, dd, qq, s),
        n_exog: 0,
        trend: Trend::None,
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
        simple_differencing: false,
        measurement_error: false,
    };

    // Convert Python arrays to Rust Vecs (while holding GIL)
    let series: Vec<Vec<f64>> = series_list
        .iter()
        .map(|a| a.as_slice().map(|s| s.to_vec()))
        .collect::<std::result::Result<Vec<_>, _>>()?;

    // Run parallel computation (Rayon thread pool runs independently of GIL)
    let results = batch::batch_fit(&series, &config, method, maxiter);

    // Convert results to Python dicts
    let py_results: Vec<Py<PyDict>> = results
        .into_iter()
        .map(|r: crate::error::Result<crate::types::FitResult>| {
            let dict = PyDict::new(py);
            match r {
                Ok(result) => {
                    dict.set_item("params", result.params).unwrap();
                    dict.set_item("loglike", result.loglike).unwrap();
                    dict.set_item("scale", result.scale).unwrap();
                    dict.set_item("aic", result.aic).unwrap();
                    dict.set_item("bic", result.bic).unwrap();
                    dict.set_item("n_obs", result.n_obs).unwrap();
                    dict.set_item("n_params", result.n_params).unwrap();
                    dict.set_item("n_iter", result.n_iter).unwrap();
                    dict.set_item("converged", result.converged).unwrap();
                    dict.set_item("method", result.method).unwrap();
                }
                Err(e) => {
                    dict.set_item("error", e.to_string()).unwrap();
                    dict.set_item("converged", false).unwrap();
                }
            }
            dict.into()
        })
        .collect();

    let list = PyList::new(py, &py_results)?;
    Ok(list.into())
}

/// Forecast multiple time series in parallel (Rayon).
///
/// Each series uses its own parameter vector from params_list.
/// Returns a list of dicts (one per series).
#[pyfunction]
#[pyo3(signature = (series_list, order, seasonal, params_list,
                    steps=10, alpha=0.05, concentrate_scale=true))]
fn sarimax_batch_forecast<'py>(
    py: Python<'py>,
    series_list: Vec<PyReadonlyArray1<'py, f64>>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    params_list: Vec<PyReadonlyArray1<'py, f64>>,
    steps: usize,
    alpha: f64,
    concentrate_scale: bool,
) -> PyResult<Py<PyList>> {
    if series_list.len() != params_list.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "series_list and params_list must have same length: {} vs {}",
            series_list.len(),
            params_list.len()
        )));
    }

    let (p, d, q) = order;
    let (pp, dd, qq, s) = seasonal;

    let config = SarimaxConfig {
        order: SarimaxOrder::new(p, d, q, pp, dd, qq, s),
        n_exog: 0,
        trend: Trend::None,
        enforce_stationarity: false,
        enforce_invertibility: false,
        concentrate_scale,
        simple_differencing: false,
        measurement_error: false,
    };

    let series: Vec<Vec<f64>> = series_list
        .iter()
        .map(|a| a.as_slice().map(|s| s.to_vec()))
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let params_vecs: Vec<Vec<f64>> = params_list
        .iter()
        .map(|a| a.as_slice().map(|s| s.to_vec()))
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let results = batch::batch_forecast(&series, &config, &params_vecs, steps, alpha);

    let py_results: Vec<Py<PyDict>> = results
        .into_iter()
        .map(|r: crate::error::Result<crate::forecast::ForecastResult>| {
            let dict = PyDict::new(py);
            match r {
                Ok(result) => {
                    dict.set_item("mean", result.mean).unwrap();
                    dict.set_item("variance", result.variance).unwrap();
                    dict.set_item("ci_lower", result.ci_lower).unwrap();
                    dict.set_item("ci_upper", result.ci_upper).unwrap();
                }
                Err(e) => {
                    dict.set_item("error", e.to_string()).unwrap();
                }
            }
            dict.into()
        })
        .collect();

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
