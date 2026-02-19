pub mod error;
pub mod types;
pub mod params;
pub mod polynomial;
pub mod state_space;
pub mod initialization;
pub mod kalman;
// Phase 3+
// pub mod start_params;
// pub mod optimizer;
// pub mod information;
// pub mod forecast;
// pub mod selection;
// pub mod diagnostics;
// pub mod batch;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

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
#[pyo3(signature = (y, order, seasonal, params, exog=None, concentrate_scale=true))]
fn sarimax_loglike<'py>(
    _py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    params: PyReadonlyArray1<'py, f64>,
    exog: Option<PyReadonlyArray1<'py, f64>>,
    concentrate_scale: bool,
) -> PyResult<f64> {
    let _ = exog; // TODO: Phase 2

    let endog = y.as_slice()?;
    let params_flat = params.as_slice()?;

    let (p, d, q) = order;
    let (pp, dd, qq, s) = seasonal;

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

    let init = KalmanInit::approximate_diffuse(
        ss.k_states,
        KalmanInit::default_kappa(),
    );

    let output = kalman::kalman_loglike(endog, &ss, &init, concentrate_scale)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(output.loglike)
}

/// Python module definition.
#[pymodule]
fn sarimax_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(sarimax_loglike, m)?)?;
    Ok(())
}
