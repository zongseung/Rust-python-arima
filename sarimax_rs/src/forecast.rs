use crate::css::apply_differencing;
use crate::error::{Result, SarimaxError};
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
///   a_{h+1} = T * a_h + c_{n+h}
///   P_{h+1} = T * P_h * T' + R * Q * R'
pub fn forecast(
    ss: &StateSpace,
    filter_output: &KalmanFilterOutput,
    steps: usize,
    alpha: f64,
    future_exog: Option<&[Vec<f64>]>,
    exog_coeffs: &[f64],
    config: &SarimaxConfig,
    params: &SarimaxParams,
    n_obs: usize,
) -> Result<ForecastResult> {
    // Validate: exog model requires future_exog for forecasting
    if !exog_coeffs.is_empty() && future_exog.is_none() && steps > 0 {
        return Err(SarimaxError::InvalidInput(
            "model has exogenous variables: future_exog is required for forecasting".into(),
        ));
    }
    if let Some(cols) = future_exog {
        if cols.len() != exog_coeffs.len() {
            return Err(SarimaxError::InvalidInput(format!(
                "future_exog column count mismatch: expected {}, got {}",
                exog_coeffs.len(),
                cols.len()
            )));
        }
        for (j, col) in cols.iter().enumerate() {
            if col.len() < steps {
                return Err(SarimaxError::InvalidInput(format!(
                    "future_exog column {} has {} rows but {} forecast steps requested",
                    j,
                    col.len(),
                    steps
                )));
            }
            if col.iter().any(|v| !v.is_finite()) {
                return Err(SarimaxError::InvalidInput(format!(
                    "future_exog column {} contains NaN or Inf values",
                    j
                )));
            }
        }
    }

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
            Some(cols) => cols
                .iter()
                .zip(exog_coeffs.iter())
                .map(|(col, &b)| col[h] * b)
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

        // Propagate state: a_{h+1} = T * a_h + c_{n+h}
        a = t_mat * &a;
        // Add trend state intercept for this forecast step
        if config.trend != crate::types::Trend::None && !params.trend_coeffs.is_empty() {
            let t_abs = n_obs + h; // absolute time index
            let val = match config.trend {
                crate::types::Trend::Constant => params.trend_coeffs[0],
                crate::types::Trend::Linear => params.trend_coeffs[0] * (t_abs as f64),
                crate::types::Trend::Both => {
                    params.trend_coeffs[0] + params.trend_coeffs[1] * (t_abs as f64)
                }
                crate::types::Trend::None => 0.0,
            };
            a[ss.k_states_diff] += val;
        }
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
pub fn compute_residuals(filter_output: &KalmanFilterOutput) -> ResidualOutput {
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
///
/// When `config.simple_differencing=true`:
///   1. Pre-difference the endog (and trim exog accordingly)
///   2. Run Kalman filter on the differenced series (ARMA-only state space)
///   3. Forecast in differenced space
///   4. Undifference the forecast to reconstruct original-scale means and variances
pub fn forecast_pipeline(
    endog: &[f64],
    config: &SarimaxConfig,
    params: &SarimaxParams,
    steps: usize,
    alpha: f64,
    exog: Option<&[Vec<f64>]>,
    future_exog: Option<&[Vec<f64>]>,
) -> Result<ForecastResult> {
    if config.simple_differencing {
        // Pre-difference the data
        let eff_endog = apply_differencing(endog, config);
        let n_drop = endog.len() - eff_endog.len();
        let eff_exog_owned: Option<Vec<Vec<f64>>> = match exog {
            Some(cols) => {
                let mut trimmed = Vec::with_capacity(cols.len());
                for (j, col) in cols.iter().enumerate() {
                    if col.len() < n_drop {
                        return Err(crate::error::SarimaxError::InvalidInput(format!(
                            "exog column {} has {} rows, but simple_differencing requires dropping {} rows",
                            j, col.len(), n_drop
                        )));
                    }
                    trimmed.push(col[n_drop..].to_vec());
                }
                Some(trimmed)
            }
            None => None,
        };
        let eff_exog = eff_exog_owned.as_deref();

        let ss = StateSpace::new(config, params, &eff_endog, eff_exog)?;
        let init = KalmanInit::from_config_default(&ss, config);
        let fo = kalman_filter(&eff_endog, &ss, &init, config.concentrate_scale)?;

        // Forecast in differenced space (future_exog is for future steps, no trimming)
        let diff_fc = forecast(
            &ss,
            &fo,
            steps,
            alpha,
            future_exog,
            &params.exog_coeffs,
            config,
            params,
            eff_endog.len(),
        )?;

        // Undifference back to original scale
        Ok(undifference_forecast(&diff_fc.mean, &diff_fc.variance, endog, config, alpha))
    } else {
        let ss = StateSpace::new(config, params, endog, exog)?;
        let init = KalmanInit::from_config_default(&ss, config);
        let fo = kalman_filter(endog, &ss, &init, config.concentrate_scale)?;
        forecast(&ss, &fo, steps, alpha, future_exog, &params.exog_coeffs, config, params, endog.len())
    }
}

/// Reconstruct original-scale forecast from a differenced-space forecast.
///
/// Reverses the differencing applied by `apply_differencing`:
///   - `apply_differencing` order: seasonal (D times, period s) → non-seasonal (d times)
///   - Undo order: non-seasonal first (d times), then seasonal (D times)
///
/// Mean undifferencing is exact. Variance is approximated as a cumulative
/// sum of the differenced-space variances (ignores cross-step covariances).
///
/// Supports: d = 0..2, D = 0..1 (D > 1 is not supported by the model).
fn undifference_forecast(
    diff_mean: &[f64],
    diff_var: &[f64],
    endog: &[f64],
    config: &SarimaxConfig,
    alpha: f64,
) -> ForecastResult {
    let d = config.order.d;
    let dd = config.order.dd;
    let s = config.order.s;
    let steps = diff_mean.len();
    let n = endog.len();
    let z_alpha = z_score(1.0 - alpha / 2.0);

    // --- Step 1: undo non-seasonal differencing (d times) ---
    // For undo iteration i, the initial "prev" value is ∇^{d-1-i}(∇_s^D y)_n.
    let mut x_mean = diff_mean.to_vec();
    let mut x_var = diff_var.to_vec();

    if d > 0 {
        let undo_inits = compute_undo_d_initials(endog, d, dd, s);
        for &init in &undo_inits {
            let mut prev = init;
            for h in 0..steps {
                x_mean[h] += prev;
                prev = x_mean[h];
            }
            // Variance: cumulative sum (each step adds uncertainty from prior step)
            let mut cum = 0.0;
            for h in 0..steps {
                cum += x_var[h];
                x_var[h] = cum;
            }
        }
    }

    // --- Step 2: undo seasonal differencing (D=1, period s) ---
    let mut y_mean = x_mean;
    let mut y_var = x_var;

    if dd > 0 && s >= 1 {
        // Last s values of original y (used for h < s reconstruction)
        let y_last_s: &[f64] = &endog[n.saturating_sub(s)..n];
        let buf_len = y_last_s.len();

        let mut undiff_mean = vec![0.0_f64; steps];
        let mut undiff_var = vec![0.0_f64; steps];
        for h in 0..steps {
            let (y_prev, v_prev) = if h < buf_len {
                (y_last_s[h], 0.0) // known y → no added variance
            } else {
                (undiff_mean[h - s], undiff_var[h - s])
            };
            undiff_mean[h] = y_mean[h] + y_prev;
            undiff_var[h] = y_var[h] + v_prev;
        }
        y_mean = undiff_mean;
        y_var = undiff_var;
    }

    // Build CI
    let ci_lower = y_mean
        .iter()
        .zip(y_var.iter())
        .map(|(&m, &v)| m - z_alpha * v.max(0.0).sqrt())
        .collect();
    let ci_upper = y_mean
        .iter()
        .zip(y_var.iter())
        .map(|(&m, &v)| m + z_alpha * v.max(0.0).sqrt())
        .collect();

    ForecastResult {
        mean: y_mean,
        variance: y_var,
        ci_lower,
        ci_upper,
    }
}

/// Precompute the initial "prev" values needed for undoing d levels of non-seasonal
/// differencing, given that D levels of seasonal differencing were already applied.
///
/// Returns d values where `result[i]` = last value of ∇^{d-1-i}(∇_s^D y).
///   i=0   → ∇^{d-1}(∇_s^D y)_n  (used to undo the innermost ∇)
///   i=d-1 → ∇^0  (∇_s^D y)_n    (used to undo the outermost ∇)
fn compute_undo_d_initials(endog: &[f64], d: usize, dd: usize, s: usize) -> Vec<f64> {
    if d == 0 {
        return vec![];
    }

    // Apply seasonal differencing first (to get the intermediate series ∇_s^D y)
    let mut series = endog.to_vec();
    for _ in 0..dd {
        if s >= 1 && series.len() > s {
            let prev = series.clone();
            series = (s..prev.len()).map(|t| prev[t] - prev[t - s]).collect();
        }
    }

    // Collect last values of ∇^0 s, ∇^1 s, ..., ∇^{d-1} s
    // derivs[k] = last value of ∇^k(series)
    let mut derivs: Vec<f64> = vec![*series.last().unwrap_or(&0.0)];
    let mut current = series;
    for _ in 0..(d - 1) {
        if current.len() > 1 {
            let next: Vec<f64> =
                (1..current.len()).map(|t| current[t] - current[t - 1]).collect();
            derivs.push(*next.last().unwrap_or(&0.0));
            current = next;
        } else {
            derivs.push(0.0);
        }
    }

    // For undo iteration i: init = derivs[d-1-i]
    (0..d).map(|i| derivs[d - 1 - i]).collect()
}

/// Run residuals pipeline: build state space → filter → residuals.
pub fn residuals_pipeline(
    endog: &[f64],
    config: &SarimaxConfig,
    params: &SarimaxParams,
    exog: Option<&[Vec<f64>]>,
) -> Result<ResidualOutput> {
    let fo = crate::pipeline::kalman_filter_full(endog, params, config, exog)?;
    Ok(compute_residuals(&fo))
}

/// Inverse normal CDF using the Beasley-Springer-Moro algorithm.
/// Error < 1e-9 across the full range, much better than the Abramowitz & Stegun
/// approximation (26.2.23) which has max error ~4.5e-4.
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

    // Rational approximation coefficients (Moro / Beasley-Springer-Moro)
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        // Rational approximation for lower region
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        // Rational approximation for central region
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        // Rational approximation for upper region
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{load_fixtures, make_config, make_params};

    #[test]
    fn test_z_score_standard() {
        // With Beasley-Springer-Moro, error should be < 1e-9
        // z(0.975) = 1.959963984540054...
        assert!(
            (z_score(0.975) - 1.959963984540054).abs() < 1e-8,
            "z(0.975) = {}, expected ~1.959964",
            z_score(0.975)
        );
        assert!((z_score(0.5)).abs() < 1e-10);
        assert!(
            (z_score(0.025) + 1.959963984540054).abs() < 1e-8,
            "z(0.025) = {}, expected ~-1.959964",
            z_score(0.025)
        );
    }

    #[test]
    fn test_forecast_ar1_mean() {
        // AR(1) with phi=0.65: forecast(h) = phi^h * y_filtered
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        let phi = 0.6527425084139002;
        let config = make_config(1, 0, 0);
        let params = make_params(&[phi], &[]);

        let result = forecast_pipeline(&data, &config, &params, 5, 0.05, None, None).unwrap();
        assert_eq!(result.mean.len(), 5);

        // Forecast variance should be increasing
        for i in 1..result.variance.len() {
            assert!(
                result.variance[i] >= result.variance[i - 1],
                "Variance should be non-decreasing: v[{}]={} < v[{}]={}",
                i,
                result.variance[i],
                i - 1,
                result.variance[i - 1]
            );
        }
    }

    #[test]
    fn test_forecast_ci_symmetric() {
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        let config = make_config(1, 0, 0);
        let params = make_params(&[0.6527425084139002], &[]);

        let result = forecast_pipeline(&data, &config, &params, 5, 0.05, None, None).unwrap();
        for i in 0..5 {
            let lower_dist = (result.mean[i] - result.ci_lower[i]).abs();
            let upper_dist = (result.ci_upper[i] - result.mean[i]).abs();
            assert!(
                (lower_dist - upper_dist).abs() < 1e-10,
                "CI not symmetric at step {}: lower_dist={}, upper_dist={}",
                i,
                lower_dist,
                upper_dist
            );
        }
    }

    #[test]
    fn test_forecast_zero_steps() {
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

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
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

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
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let params_vec: Vec<f64> = case["params"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

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
        assert!(
            var > 0.5 && var < 2.0,
            "Standardized residual variance should be ~1, got {}",
            var
        );
    }
}
