//! Rayon-based parallel batch processing for multiple time series.
//!
//! Provides batch versions of loglike, fit, and forecast that process
//! N time series in parallel using Rayon's work-stealing thread pool.

use rayon::prelude::*;

use crate::error::Result;
use crate::forecast::{forecast_pipeline, ForecastResult};
use crate::initialization::KalmanInit;
use crate::kalman::kalman_loglike;
use crate::optimizer;
use crate::params::SarimaxParams;
use crate::state_space::StateSpace;
use crate::types::{FitResult, SarimaxConfig};

/// Compute log-likelihood for multiple time series in parallel.
///
/// All series share the same model config and parameters.
/// If `exog_list` is provided, `exog_list[i]` is the exog for `series[i]`.
pub fn batch_loglike(
    series: &[Vec<f64>],
    config: &SarimaxConfig,
    params: &SarimaxParams,
    exog_list: Option<&[Vec<Vec<f64>>]>,
) -> Vec<Result<f64>> {
    series
        .par_iter()
        .enumerate()
        .map(|(i, endog)| {
            let exog = exog_list.map(|el| &el[i][..]);
            let exog_ref: Option<&[Vec<f64>]> = exog;
            let ss = StateSpace::new(config, params, endog, exog_ref)?;
            let init = KalmanInit::from_config(&ss, config, KalmanInit::default_kappa());
            let output = kalman_loglike(endog, &ss, &init, config.concentrate_scale)?;
            Ok(output.loglike)
        })
        .collect()
}

/// Fit SARIMAX models to multiple time series in parallel.
///
/// All series share the same model config. Each series is fit independently.
/// If `exog_list` is provided, `exog_list[i]` is the exog for `series[i]`.
pub fn batch_fit(
    series: &[Vec<f64>],
    config: &SarimaxConfig,
    method: Option<&str>,
    maxiter: Option<u64>,
    exog_list: Option<&[Vec<Vec<f64>>]>,
) -> Vec<Result<FitResult>> {
    let method_str = method.unwrap_or("lbfgs");
    let maxiter_val = maxiter.unwrap_or(500);

    series
        .par_iter()
        .enumerate()
        .map(|(i, endog)| {
            let exog = exog_list.map(|el| &el[i][..]);
            let exog_ref: Option<&[Vec<f64>]> = exog;
            optimizer::fit(endog, config, None, Some(method_str), Some(maxiter_val), exog_ref)
        })
        .collect()
}

/// Forecast multiple time series in parallel.
///
/// Each series uses its own parameter vector from `params_list`.
/// `params_list[i]` is the flat parameter vector for `series[i]`.
/// `exog_list[i]` and `future_exog_list[i]` are the past/future exog for series i.
pub fn batch_forecast(
    series: &[Vec<f64>],
    config: &SarimaxConfig,
    params_list: &[Vec<f64>],
    steps: usize,
    alpha: f64,
    exog_list: Option<&[Vec<Vec<f64>>]>,
    future_exog_list: Option<&[Vec<Vec<f64>>]>,
) -> Vec<Result<ForecastResult>> {
    series
        .par_iter()
        .zip(params_list.par_iter())
        .enumerate()
        .map(|(i, (endog, flat_params))| {
            let sparams = SarimaxParams::from_flat(flat_params, config)?;
            let exog = exog_list.map(|el| &el[i][..]);
            let future_exog = future_exog_list.map(|el| &el[i][..]);
            forecast_pipeline(endog, config, &sparams, steps, alpha,
                exog, future_exog)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SarimaxConfig, SarimaxOrder, Trend};

    fn load_fixtures() -> serde_json::Value {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/statsmodels_reference.json"
        );
        let data = std::fs::read_to_string(path).expect("fixtures file not found");
        serde_json::from_str(&data).expect("invalid JSON")
    }

    fn make_config(
        p: usize, d: usize, q: usize,
        enforce_stat: bool, enforce_inv: bool,
    ) -> SarimaxConfig {
        SarimaxConfig {
            order: SarimaxOrder::new(p, d, q, 0, 0, 0, 0),
            n_exog: 0,
            trend: Trend::None,
            enforce_stationarity: enforce_stat,
            enforce_invertibility: enforce_inv,
            concentrate_scale: true,
            simple_differencing: false,
            measurement_error: false,
        }
    }

    fn get_ar1_data() -> Vec<f64> {
        let fixtures = load_fixtures();
        fixtures["ar1"]["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect()
    }

    fn get_ar1_params() -> Vec<f64> {
        let fixtures = load_fixtures();
        fixtures["ar1"]["params"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect()
    }

    #[test]
    fn test_batch_fit_single() {
        let data = get_ar1_data();
        let config = make_config(1, 0, 0, true, true);

        // Single series via batch
        let batch_results = batch_fit(&[data.clone()], &config, Some("lbfgs"), Some(500), None);
        assert_eq!(batch_results.len(), 1);
        let batch_result = batch_results[0].as_ref().unwrap();

        // Single series via direct fit
        let direct_result =
            optimizer::fit(&data, &config, None, Some("lbfgs"), Some(500), None).unwrap();

        // Results should be identical
        assert!(
            (batch_result.loglike - direct_result.loglike).abs() < 1e-10,
            "loglike mismatch: batch={}, direct={}",
            batch_result.loglike,
            direct_result.loglike
        );
        for (a, b) in batch_result.params.iter().zip(direct_result.params.iter()) {
            assert!((a - b).abs() < 1e-10, "param mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_batch_fit_multiple() {
        let data = get_ar1_data();
        let config = make_config(1, 0, 0, true, true);

        // Create 10 copies of the same series
        let series: Vec<Vec<f64>> = (0..10).map(|_| data.clone()).collect();
        let results = batch_fit(&series, &config, Some("lbfgs"), Some(500), None);

        assert_eq!(results.len(), 10);
        for (i, r) in results.iter().enumerate() {
            let res = r.as_ref().unwrap();
            assert!(res.converged, "series {} should converge", i);
            assert!(res.loglike.is_finite(), "series {} loglike not finite", i);
        }
    }

    #[test]
    fn test_batch_loglike_matches_single() {
        let data = get_ar1_data();
        let params_vec = get_ar1_params();
        let config = make_config(1, 0, 0, false, false);
        let params = SarimaxParams {
            trend_coeffs: vec![],
            exog_coeffs: vec![],
            ar_coeffs: params_vec[..1].to_vec(),
            ma_coeffs: vec![],
            sar_coeffs: vec![],
            sma_coeffs: vec![],
            sigma2: None,
        };

        // Direct single loglike
        let ss = StateSpace::new(&config, &params, &data, None).unwrap();
        let init = KalmanInit::approximate_diffuse(ss.k_states, KalmanInit::default_kappa());
        let direct_ll = kalman_loglike(&data, &ss, &init, true).unwrap().loglike;

        // Batch loglike
        let series = vec![data.clone(), data.clone()];
        let batch_ll = batch_loglike(&series, &config, &params, None);

        assert_eq!(batch_ll.len(), 2);
        for (i, r) in batch_ll.iter().enumerate() {
            let ll = r.as_ref().unwrap();
            assert!(
                (*ll - direct_ll).abs() < 1e-10,
                "series {} loglike mismatch: batch={}, direct={}",
                i, ll, direct_ll
            );
        }
    }

    #[test]
    fn test_batch_forecast_matches_single() {
        let data = get_ar1_data();
        let params_vec = get_ar1_params();
        let config = make_config(1, 0, 0, false, false);
        let params = SarimaxParams {
            trend_coeffs: vec![],
            exog_coeffs: vec![],
            ar_coeffs: params_vec[..1].to_vec(),
            ma_coeffs: vec![],
            sar_coeffs: vec![],
            sma_coeffs: vec![],
            sigma2: None,
        };

        // Direct single forecast
        let direct = forecast_pipeline(&data, &config, &params, 5, 0.05, None, None).unwrap();

        // Batch forecast
        let series = vec![data.clone(), data.clone()];
        let flat_params = params_vec.clone();
        let params_list = vec![flat_params.clone(), flat_params];
        let batch = batch_forecast(&series, &config, &params_list, 5, 0.05, None, None);

        assert_eq!(batch.len(), 2);
        for (i, r) in batch.iter().enumerate() {
            let fcast = r.as_ref().unwrap();
            for (j, (a, b)) in fcast.mean.iter().zip(direct.mean.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-10,
                    "series {} forecast[{}] mismatch: batch={}, direct={}",
                    i, j, a, b
                );
            }
        }
    }

    #[test]
    fn test_batch_empty() {
        let config = make_config(1, 0, 0, true, true);
        let empty: Vec<Vec<f64>> = vec![];
        let results = batch_fit(&empty, &config, None, None, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_error_handling() {
        let config = make_config(1, 0, 0, true, true);
        let good_data = get_ar1_data();
        let bad_data = vec![]; // Empty series always fails

        let series = vec![good_data, bad_data];
        let results = batch_fit(&series, &config, Some("lbfgs"), Some(500), None);

        assert_eq!(results.len(), 2);
        assert!(results[0].is_ok(), "good series should succeed");
        assert!(results[1].is_err(), "bad series should fail");
    }
}
