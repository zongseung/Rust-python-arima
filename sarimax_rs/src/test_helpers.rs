//! Shared test utilities for all modules.
//!
//! Centralises `make_config`, `make_params`, `load_fixtures` helpers that were
//! previously duplicated across 10+ test modules.

use crate::params::SarimaxParams;
use crate::types::{SarimaxConfig, SarimaxOrder};

// ---------------------------------------------------------------------------
// Config helpers
// ---------------------------------------------------------------------------

/// Non-seasonal ARIMA config with enforcement disabled.
pub fn make_config(p: usize, d: usize, q: usize) -> SarimaxConfig {
    SarimaxConfig {
        order: SarimaxOrder::new(p, d, q, 0, 0, 0, 0),
        ..SarimaxConfig::default()
    }
}

/// Non-seasonal ARIMA config with stationarity/invertibility enforcement.
pub fn make_config_enforced(p: usize, d: usize, q: usize) -> SarimaxConfig {
    SarimaxConfig {
        order: SarimaxOrder::new(p, d, q, 0, 0, 0, 0),
        enforce_stationarity: true,
        enforce_invertibility: true,
        ..SarimaxConfig::default()
    }
}

/// Non-seasonal ARIMA config with parameterised enforcement.
pub fn make_config_with_enforcement(
    p: usize,
    d: usize,
    q: usize,
    enforce_stat: bool,
    enforce_inv: bool,
) -> SarimaxConfig {
    SarimaxConfig {
        order: SarimaxOrder::new(p, d, q, 0, 0, 0, 0),
        enforce_stationarity: enforce_stat,
        enforce_invertibility: enforce_inv,
        ..SarimaxConfig::default()
    }
}

/// Seasonal SARIMA config with enforcement disabled.
pub fn make_seasonal_config(
    p: usize,
    d: usize,
    q: usize,
    pp: usize,
    dd: usize,
    qq: usize,
    s: usize,
) -> SarimaxConfig {
    SarimaxConfig {
        order: SarimaxOrder::new(p, d, q, pp, dd, qq, s),
        ..SarimaxConfig::default()
    }
}

// ---------------------------------------------------------------------------
// Params helpers
// ---------------------------------------------------------------------------

/// Non-seasonal params with AR and MA coefficients.
pub fn make_params(ar: &[f64], ma: &[f64]) -> SarimaxParams {
    SarimaxParams {
        trend_coeffs: vec![],
        exog_coeffs: vec![],
        ar_coeffs: ar.to_vec(),
        ma_coeffs: ma.to_vec(),
        sar_coeffs: vec![],
        sma_coeffs: vec![],
        sigma2: None,
    }
}

/// Seasonal params with AR, MA, SAR and SMA coefficients.
pub fn make_seasonal_params(
    ar: &[f64],
    ma: &[f64],
    sar: &[f64],
    sma: &[f64],
) -> SarimaxParams {
    SarimaxParams {
        trend_coeffs: vec![],
        exog_coeffs: vec![],
        ar_coeffs: ar.to_vec(),
        ma_coeffs: ma.to_vec(),
        sar_coeffs: sar.to_vec(),
        sma_coeffs: sma.to_vec(),
        sigma2: None,
    }
}

// ---------------------------------------------------------------------------
// Fixture loading
// ---------------------------------------------------------------------------

/// Load the statsmodels reference fixture JSON.
pub fn load_fixtures() -> serde_json::Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/statsmodels_reference.json"
    );
    let data = std::fs::read_to_string(path).expect("fixtures file not found");
    serde_json::from_str(&data).expect("invalid JSON")
}
