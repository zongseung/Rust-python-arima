use crate::error::Result;
use crate::initialization::KalmanInit;
use crate::kalman::{kalman_filter, KalmanFilterOutput};
use crate::params::SarimaxParams;
use crate::state_space::StateSpace;
use crate::types::SarimaxConfig;

/// H-step ahead forecast result.
#[derive(Debug, Clone)]
pub struct ForecastResult {
    /// Forecast means E[y_{n+h}] for h = 1..steps.
    pub mean: Vec<f64>,
    /// Forecast variances Var[y_{n+h}].
    pub variance: Vec<f64>,
    /// Lower confidence interval bounds.
    pub ci_lower: Vec<f64>,
    /// Upper confidence interval bounds.
    pub ci_upper: Vec<f64>,
}

/// Residual diagnostics output.
#[derive(Debug, Clone)]
pub struct ResidualOutput {
    /// Raw innovations v_t.
    pub residuals: Vec<f64>,
    /// Standardized residuals v_t / sqrt(F_t * scale).
    pub standardized_residuals: Vec<f64>,
}

/// Compute h-step ahead forecast from the final Kalman filter state.
///
/// Uses state-space forward propagation:
///   y_hat_h = Z' * a_h
///   F_h     = Z' * P_h * Z * scale
///   a_{h+1} = T * a_h
///   P_{h+1} = T * P_h * T' + R * Q * R'
pub fn forecast(
    ss: &StateSpace,
    filter_output: &KalmanFilterOutput,
    steps: usize,
    alpha: f64,
    future_exog: Option<&[Vec<f64>]>,
    exog_coeffs: &[f64],
) -> Result<ForecastResult> {
    if steps == 0 {
        return Ok(ForecastResult {
            mean: vec![],
            variance: vec![],
            ci_lower: vec![],
            ci_upper: vec![],
        });
    }

    // z-score for confidence interval
    let z_alpha = z_score(1.0 - alpha / 2.0);

    let z = &ss.design;
    let t_mat = &ss.transition;
    let r_mat = &ss.selection;
    let q_mat = &ss.state_cov;

    let rqr = r_mat * q_mat * r_mat.transpose();
    let scale = filter_output.scale;

    // Start from predicted state a_{n+1|n}, P_{n+1|n}
    let mut a = filter_output.predicted_state.clone();
    let mut p = filter_output.predicted_cov.clone();

    let mut mean = Vec::with_capacity(steps);
    let mut variance = Vec::with_capacity(steps);
    let mut ci_lower = Vec::with_capacity(steps);
    let mut ci_upper = Vec::with_capacity(steps);

    for h in 0..steps {
        // Forecast mean: y_hat = Z' * a
        let y_hat = z.dot(&a);

        // Add exogenous contribution: d_h = Σ(x_j[h] * β_j)
        let d_h = match future_exog {
            Some(cols) => cols.iter()
                .zip(exog_coeffs.iter())
                .map(|(col, &b)| if h < col.len() { col[h] * b } else { 0.0 })
                .sum::<f64>(),
            None => 0.0,
        };

        // Forecast variance: F = Z' * P * Z * scale
        let p_z = &p * z;
        let f_h = z.dot(&p_z) * scale;
        let f_safe = f_h.max(0.0);

        let se = f_safe.sqrt();
        mean.push(y_hat + d_h);
        variance.push(f_safe);
        ci_lower.push(y_hat + d_h - z_alpha * se);
        ci_upper.push(y_hat + d_h + z_alpha * se);

        // Propagate state: a_{h+1} = T * a_h
        a = t_mat * &a;
        // Propagate covariance: P_{h+1} = T * P_h * T' + R * Q * R'
        p = t_mat * &p * t_mat.transpose() + &rqr;
    }

    Ok(ForecastResult {
        mean,
        variance,
        ci_lower,
        ci_upper,
    })
}

/// Compute residuals and standardized residuals from Kalman filter output.
pub fn compute_residuals(
    filter_output: &KalmanFilterOutput,
) -> ResidualOutput {
    let scale = filter_output.scale;
    let n = filter_output.innovations.len();

    let mut standardized = Vec::with_capacity(n);
    for i in 0..n {
        let v = filter_output.innovations[i];
        let f = filter_output.innovation_vars[i];
        if f * scale > 0.0 {
            standardized.push(v / (f * scale).sqrt());
        } else {
            standardized.push(0.0);
        }
    }

    ResidualOutput {
        residuals: filter_output.innovations.clone(),
        standardized_residuals: standardized,
    }
}

/// Run forecast pipeline: build state space → filter → forecast.
pub fn forecast_pipeline(
    endog: &[f64],
    config: &SarimaxConfig,
    params: &SarimaxParams,
    steps: usize,
    alpha: f64,
    exog: Option<&[Vec<f64>]>,
    future_exog: Option<&[Vec<f64>]>,
) -> Result<ForecastResult> {
    let ss = StateSpace::new(config, params, endog, exog)?;
    let init = KalmanInit::from_config(&ss, config, KalmanInit::default_kappa());
    let fo = kalman_filter(endog, &ss, &init, config.concentrate_scale)?;
    forecast(&ss, &fo, steps, alpha, future_exog, &params.exog_coeffs)
}

/// Run residuals pipeline: build state space → filter → residuals.
pub fn residuals_pipeline(
    endog: &[f64],
    config: &SarimaxConfig,
    params: &SarimaxParams,
    exog: Option<&[Vec<f64>]>,
) -> Result<ResidualOutput> {
    let ss = StateSpace::new(config, params, endog, exog)?;
    let init = KalmanInit::from_config(&ss, config, KalmanInit::default_kappa());
    let fo = kalman_filter(endog, &ss, &init, config.concentrate_scale)?;
    Ok(compute_residuals(&fo))
}

/// Approximate inverse normal CDF using rational approximation (Abramowitz & Stegun 26.2.23).
fn z_score(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    let sign;
    let q;
    if p < 0.5 {
        sign = -1.0;
        q = p;
    } else {
        sign = 1.0;
        q = 1.0 - p;
    };

    let t = (-2.0 * q.ln()).sqrt();

    // Rational approximation coefficients
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let x = t - (c0 + c1 * t + c2 * t * t)
        / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    sign * x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::SarimaxParams;
    use crate::types::{SarimaxConfig, SarimaxOrder, Trend};

    fn load_fixtures() -> serde_json::Value {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/statsmodels_reference.json"
        );
        let data = std::fs::read_to_string(path).expect("fixtures file not found");
        serde_json::from_str(&data).expect("invalid JSON")
    }

    fn make_config(p: usize, d: usize, q: usize) -> SarimaxConfig {
        SarimaxConfig {
            order: SarimaxOrder::new(p, d, q, 0, 0, 0, 0),
            n_exog: 0,
            trend: Trend::None,
            enforce_stationarity: false,
            enforce_invertibility: false,
            concentrate_scale: true,
            simple_differencing: false,
            measurement_error: false,
        }
    }

    fn make_params(ar: &[f64], ma: &[f64]) -> SarimaxParams {
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

    #[test]
    fn test_z_score_standard() {
        assert!((z_score(0.975) - 1.96).abs() < 0.01);
        assert!((z_score(0.5)).abs() < 1e-10);
        assert!((z_score(0.025) + 1.96).abs() < 0.01);
    }

    #[test]
    fn test_forecast_ar1_mean() {
        // AR(1) with phi=0.65: forecast(h) = phi^h * y_filtered
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();

        let phi = 0.6527425084139002;
        let config = make_config(1, 0, 0);
        let params = make_params(&[phi], &[]);

        let result = forecast_pipeline(&data, &config, &params, 5, 0.05, None, None).unwrap();
        assert_eq!(result.mean.len(), 5);

        // Forecast variance should be increasing
        for i in 1..result.variance.len() {
            assert!(result.variance[i] >= result.variance[i - 1],
                "Variance should be non-decreasing: v[{}]={} < v[{}]={}",
                i, result.variance[i], i - 1, result.variance[i - 1]);
        }
    }

    #[test]
    fn test_forecast_ci_symmetric() {
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();

        let config = make_config(1, 0, 0);
        let params = make_params(&[0.6527425084139002], &[]);

        let result = forecast_pipeline(&data, &config, &params, 5, 0.05, None, None).unwrap();
        for i in 0..5 {
            let lower_dist = (result.mean[i] - result.ci_lower[i]).abs();
            let upper_dist = (result.ci_upper[i] - result.mean[i]).abs();
            assert!((lower_dist - upper_dist).abs() < 1e-10,
                "CI not symmetric at step {}: lower_dist={}, upper_dist={}",
                i, lower_dist, upper_dist);
        }
    }

    #[test]
    fn test_forecast_zero_steps() {
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();

        let config = make_config(1, 0, 0);
        let params = make_params(&[0.6527425084139002], &[]);

        let result = forecast_pipeline(&data, &config, &params, 0, 0.05, None, None).unwrap();
        assert!(result.mean.is_empty());
    }

    #[test]
    fn test_residuals_length() {
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();

        let config = make_config(1, 0, 0);
        let params = make_params(&[0.6527425084139002], &[]);

        let result = residuals_pipeline(&data, &config, &params, None).unwrap();
        assert_eq!(result.residuals.len(), data.len());
        assert_eq!(result.standardized_residuals.len(), data.len());
    }

    #[test]
    fn test_standardized_residuals_scale() {
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let params_vec: Vec<f64> = case["params"]
            .as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();

        let config = make_config(1, 0, 0);
        let params = make_params(&params_vec[..1], &[]);

        let result = residuals_pipeline(&data, &config, &params, None).unwrap();

        // After burn-in, standardized residuals should have variance ~ 1
        let burn = config.order.k_states();
        let std_res: Vec<f64> = result.standardized_residuals[burn..].to_vec();
        let n = std_res.len() as f64;
        let mean = std_res.iter().sum::<f64>() / n;
        let var = std_res.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
        // Variance should be roughly 1 (allow wide margin for finite sample)
        assert!(var > 0.5 && var < 2.0,
            "Standardized residual variance should be ~1, got {}", var);
    }
}
