use crate::css::apply_differencing;
use crate::error::{Result, SarimaxError};
use crate::initialization::KalmanInit;
use crate::kalman::{kalman_filter, kalman_filter_with_snapshots, KalmanFilterOutput};
use crate::params::SarimaxParams;
use crate::state_space::StateSpace;
use crate::types::SarimaxConfig;
use nalgebra::{DMatrix, DVector};

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
    /// Standardized residuals: v_t / sqrt(F_t) in non-concentrated mode
    /// (F_t already includes sigma2), v_t / sqrt(F_t * scale) in
    /// concentrated mode.
    pub standardized_residuals: Vec<f64>,
}

/// Compute h-step ahead forecast from the final Kalman filter state.
///
/// Uses state-space forward propagation:
///   y_hat_h = Z' * a_h
///   F_h     = Z' * P_h * Z          (non-concentrated; Q=[[sigma2]] is in P)
///           = Z' * P_h * Z * scale  (concentrated; Q=[[1]], restore sigma2)
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
    forecast_from_state(
        ss,
        &filter_output.predicted_state,
        &filter_output.predicted_cov,
        filter_output.scale,
        steps,
        alpha,
        future_exog,
        exog_coeffs,
        config,
        params,
        n_obs,
    )
}

/// Compute h-step ahead forecast from an arbitrary predicted state.
///
/// Same propagation as [`forecast`], but takes (a, P, scale) directly so a
/// mid-filter [`crate::kalman::StateSnapshot`] can serve as the forecast
/// origin (single-pass rolling forecasts). `n_obs` is the number of
/// observations consumed at the origin (used for the absolute-time trend
/// intercept).
#[allow(clippy::too_many_arguments)]
pub fn forecast_from_state(
    ss: &StateSpace,
    a0: &DVector<f64>,
    p0: &DMatrix<f64>,
    scale: f64,
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

    // Start from the supplied predicted state (a_{t|t-1}, P_{t|t-1})
    let mut a = a0.clone();
    let mut p = p0.clone();

    // Effective output scale for variance restoration.
    //
    // Concentrated mode: Q = [[1]] so the filter covariance P is normalized;
    // the reported variance must be restored as Z'PZ * sigma2_hat.
    // Non-concentrated mode (default): Q = [[sigma2]] already carries the
    // innovation variance inside P, so Z'PZ IS the forecast variance —
    // multiplying by scale again would double-count sigma2.
    let eff_scale = if config.concentrate_scale { scale } else { 1.0 };

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

        // Forecast variance: F = Z' * P * Z  (* sigma2_hat in concentrated mode)
        let p_z = &p * z;
        let f_h = z.dot(&p_z) * eff_scale;
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
            let idx = ss.k_states_diff;
            if idx < a.len() {
                a[idx] += val;
            } else {
                return Err(SarimaxError::StateSpaceError(format!(
                    "trend state index {} out of bounds (state dim={})",
                    idx,
                    a.len()
                )));
            }
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
///
/// Concentrated mode: F_t excludes sigma2, so standardize by sqrt(F_t * scale).
/// Non-concentrated mode (default): F_t already includes sigma2 (Q=[[sigma2]]),
/// so standardize by sqrt(F_t) — multiplying by scale again would shrink the
/// residuals by sqrt(sigma2).
pub fn compute_residuals(
    filter_output: &KalmanFilterOutput,
    concentrate_scale: bool,
) -> ResidualOutput {
    let eff_scale = if concentrate_scale {
        filter_output.scale
    } else {
        1.0
    };
    let n = filter_output.innovations.len();

    let mut standardized = Vec::with_capacity(n);
    for i in 0..n {
        let v = filter_output.innovations[i];
        let f = filter_output.innovation_vars[i];
        if f * eff_scale > 0.0 {
            standardized.push(v / (f * eff_scale).sqrt());
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

/// Rolling-origin forecast output: one row per origin.
#[derive(Debug, Clone)]
pub struct RollingForecastOutput {
    /// Forecast origins (observations consumed before each forecast).
    pub origins: Vec<usize>,
    /// Point forecasts, shape `[n_origins][horizon]`.
    pub mean: Vec<Vec<f64>>,
    /// Forecast variances.
    pub variance: Vec<Vec<f64>>,
    /// CI lower bounds (at the requested alpha).
    pub ci_lower: Vec<Vec<f64>>,
    /// CI upper bounds.
    pub ci_upper: Vec<Vec<f64>>,
}

/// Single-pass rolling-origin h-step forecasts with fixed parameters.
///
/// ONE Kalman filter pass over the full series captures the predicted state
/// at every origin `start, start+step, ...`; each origin's h-step forecast
/// is then propagated from its snapshot. Total cost O(T + N·h) — versus
/// O(N·T) for refiltering per origin (extend-chain) — while producing
/// numerically identical forecasts, since the filter's prefix processing is
/// exactly the prefix filter (Markov property).
///
/// Origins run while `origin <= n-1`; models with exogenous regressors are
/// additionally capped at `n - horizon` so the in-sample exog rows
/// `[origin, origin+horizon)` cover each forecast window.
///
/// `simple_differencing=true` is not supported yet (per-origin
/// undifferencing of the raw tail is future work).
#[allow(clippy::too_many_arguments)]
pub fn rolling_forecast_pipeline(
    endog: &[f64],
    config: &SarimaxConfig,
    params: &SarimaxParams,
    start: usize,
    step: usize,
    horizon: usize,
    alpha: f64,
    exog: Option<&[Vec<f64>]>,
) -> Result<RollingForecastOutput> {
    if config.simple_differencing {
        return Err(SarimaxError::InvalidInput(
            "rolling_forecast does not support simple_differencing yet".into(),
        ));
    }
    let n = endog.len();
    if n == 0 {
        return Err(SarimaxError::InvalidInput("endog is empty".into()));
    }
    if start == 0 || start > n - 1 {
        return Err(SarimaxError::InvalidInput(format!(
            "start must be in [1, n-1] = [1, {}], got {}",
            n - 1,
            start
        )));
    }
    if step == 0 {
        return Err(SarimaxError::InvalidInput("step must be >= 1".into()));
    }
    if horizon == 0 {
        return Err(SarimaxError::InvalidInput("horizon must be >= 1".into()));
    }

    let has_exog = exog.is_some_and(|c| !c.is_empty());
    let max_origin = if has_exog {
        // in-sample exog must cover [origin, origin+horizon)
        n.checked_sub(horizon).ok_or_else(|| {
            SarimaxError::InvalidInput(format!(
                "horizon {} exceeds series length {} for exog model",
                horizon, n
            ))
        })?
    } else {
        n - 1
    };
    if start > max_origin {
        return Err(SarimaxError::InvalidInput(format!(
            "start {} exceeds max origin {} (exog models cap origins at n - horizon)",
            start, max_origin
        )));
    }

    let mut origins = Vec::new();
    let mut o = start;
    while o <= max_origin {
        origins.push(o);
        o += step;
    }

    // Every origin snapshot retains a k_states x k_states predicted
    // covariance for the whole call, so peak memory is
    // 8 * n_origins * k_states^2 bytes with NO individual knob capping the
    // product (measured 4.4 GB at n=1000, s=365; DIAGNOSIS_V9 C3). A failed
    // allocation aborts the process rather than raising, so refuse early.
    const MAX_SNAPSHOT_BYTES: usize = 2 << 30; // 2 GiB
    let k = config.order.k_states();
    let snapshot_bytes = origins
        .len()
        .saturating_mul(k)
        .saturating_mul(k)
        .saturating_mul(8);
    if snapshot_bytes > MAX_SNAPSHOT_BYTES {
        return Err(SarimaxError::InvalidInput(format!(
            "rolling forecast would retain {} origin snapshots of {}x{} state \
             covariances (~{:.1} GiB > 2 GiB limit); increase `step`, raise \
             `start`, or split the series into chunks",
            origins.len(),
            k,
            k,
            snapshot_bytes as f64 / (1u64 << 30) as f64,
        )));
    }

    // Single filter pass with snapshots at every origin
    let ss = StateSpace::new(config, params, endog, exog)?;
    let init = KalmanInit::from_config_default(&ss, config);
    let (fo, snaps) =
        kalman_filter_with_snapshots(endog, &ss, &init, config.concentrate_scale, &origins)?;

    let n_o = origins.len();
    let mut out = RollingForecastOutput {
        origins,
        mean: Vec::with_capacity(n_o),
        variance: Vec::with_capacity(n_o),
        ci_lower: Vec::with_capacity(n_o),
        ci_upper: Vec::with_capacity(n_o),
    };

    for snap in &snaps {
        let t0 = snap.origin;
        // Per-origin "future" exog = the actual in-sample rows [t0, t0+horizon)
        let fex_owned: Option<Vec<Vec<f64>>> = exog.map(|cols| {
            cols.iter().map(|c| c[t0..t0 + horizon].to_vec()).collect()
        });
        let fc = forecast_from_state(
            &ss,
            &snap.predicted_state,
            &snap.predicted_cov,
            fo.scale,
            horizon,
            alpha,
            fex_owned.as_deref(),
            &params.exog_coeffs,
            config,
            params,
            t0,
        )?;
        out.mean.push(fc.mean);
        out.variance.push(fc.variance);
        out.ci_lower.push(fc.ci_lower);
        out.ci_upper.push(fc.ci_upper);
    }

    Ok(out)
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
            for xm in x_mean.iter_mut() {
                *xm += prev;
                prev = *xm;
            }
            // Variance: cumulative sum (each step adds uncertainty from prior step)
            let mut cum = 0.0;
            for xv in x_var.iter_mut() {
                cum += *xv;
                *xv = cum;
            }
        }
    }

    // --- Step 2: undo seasonal differencing (D=1, period s) ---
    let mut y_mean = x_mean;
    let mut y_var = x_var;

    // Seasonal undifferencing is only valid when seasonal differencing was
    // actually applied in `apply_differencing` (requires n > s).
    if dd > 0 && s >= 1 && n > s {
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
    Ok(compute_residuals(&fo, config.concentrate_scale))
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
    #[allow(clippy::excessive_precision)]
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
    use crate::test_helpers::{
        load_fixtures, make_config, make_params, make_seasonal_config, make_seasonal_params,
    };

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

    #[test]
    fn test_forecast_simple_diff_short_series_no_panic() {
        // Regression: n < s with simple_differencing + seasonal differencing
        // previously panicked due to h-s underflow in undifference_forecast.
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0]; // n=5
        let mut config = make_seasonal_config(0, 0, 0, 0, 1, 0, 12); // D=1, s=12
        config.simple_differencing = true;
        let params = make_seasonal_params(&[], &[], &[], &[]);

        let result = forecast_pipeline(&data, &config, &params, 10, 0.05, None, None).unwrap();
        assert_eq!(result.mean.len(), 10);
        assert_eq!(result.variance.len(), 10);
    }
}
