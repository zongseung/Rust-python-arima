//! Statistical inference for SARIMAX models.
//!
//! Provides two methods to compute parameter covariance, standard errors,
//! z-statistics, p-values, and confidence intervals:
//!
//! 1. **Numerical Hessian** (`method="hessian"`) — Central-difference second
//!    derivatives of log-likelihood in unconstrained parameter space.
//!    Cost: O(k²) Kalman filter evaluations.
//!
//! 2. **OPG** (`method="opg"`) — Outer Product of per-observation Gradients.
//!    Cost: 1 tangent-linear KF pass + O(n·k²) outer-product summation.
//!
//! Also provides residual diagnostic tests: Ljung-Box, Jarque-Bera,
//! heteroskedasticity (breakvar).

use nalgebra::DMatrix;
use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor, Normal};

use crate::error::{Result, SarimaxError};
use crate::initialization::KalmanInit;
use crate::optimizer::untransform_params;
use crate::params::SarimaxParams;
use crate::score;
use crate::state_space::StateSpace;
use crate::types::SarimaxConfig;

// ---------------------------------------------------------------------------
// Result structs
// ---------------------------------------------------------------------------

/// Result of inference computation.
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// Method used: "hessian" or "opg".
    pub method: String,
    /// k×k covariance matrix (row-major flattened).
    pub cov_params: Vec<f64>,
    /// Standard errors: sqrt(diag(cov)).
    pub std_err: Vec<f64>,
    /// Z-statistics: theta / std_err.
    pub z_stat: Vec<f64>,
    /// P-values: 2·(1 - Phi(|z|)).
    pub p_value: Vec<f64>,
    /// Lower CI bounds: theta - z_crit·SE.
    pub ci_lower: Vec<f64>,
    /// Upper CI bounds: theta + z_crit·SE.
    pub ci_upper: Vec<f64>,
    /// Number of parameters.
    pub n_params: usize,
    /// Status: "ok", "partial" (some SE non-finite), or "failed".
    pub status: String,
    /// Optional message describing issues.
    pub message: Option<String>,
}

/// Result of diagnostic tests on standardized residuals.
#[derive(Debug, Clone)]
pub struct DiagnosticsResult {
    pub ljung_box_stat: f64,
    pub ljung_box_pvalue: f64,
    pub ljung_box_df: usize,
    pub jarque_bera_stat: f64,
    pub jarque_bera_pvalue: f64,
    pub het_stat: f64,
    pub het_pvalue: f64,
}

// ---------------------------------------------------------------------------
// Top-level entry point
// ---------------------------------------------------------------------------

/// Compute inference using the specified method ("hessian" or "opg").
pub fn compute_inference(
    endog: &[f64],
    config: &SarimaxConfig,
    constrained_params: &[f64],
    method: &str,
    alpha: f64,
    exog: Option<&[Vec<f64>]>,
) -> Result<InferenceResult> {
    let k = constrained_params.len();
    if k == 0 {
        return Ok(InferenceResult {
            method: method.to_string(),
            cov_params: vec![],
            std_err: vec![],
            z_stat: vec![],
            p_value: vec![],
            ci_lower: vec![],
            ci_upper: vec![],
            n_params: 0,
            status: "ok".to_string(),
            message: None,
        });
    }

    let info = match method {
        "hessian" => {
            let hess = numerical_hessian(endog, config, constrained_params, exog)?;
            // Information matrix = -Hessian (in constrained space after chain rule)
            -hess
        }
        "opg" => {
            let opg = opg_information_matrix(endog, config, constrained_params, exog)?;
            // Scale by n_eff (matching statsmodels convention)
            let n_eff = compute_n_eff(endog, config)?;
            opg * n_eff as f64
        }
        _ => {
            return Err(SarimaxError::InvalidInput(format!(
                "unknown inference method '{}', expected 'hessian' or 'opg'",
                method
            )));
        }
    };

    Ok(inference_from_information(
        &info,
        constrained_params,
        alpha,
        method,
    ))
}

// ---------------------------------------------------------------------------
// Method 1: Numerical Hessian
// ---------------------------------------------------------------------------

/// Compute numerical Hessian of log-likelihood in constrained parameter space.
///
/// Steps:
/// 1. Untransform to unconstrained space
/// 2. Central-difference Hessian in unconstrained space
/// 3. Chain rule: H_constrained = J^T · H_unconstrained · J
pub fn numerical_hessian(
    endog: &[f64],
    config: &SarimaxConfig,
    constrained_params: &[f64],
    exog: Option<&[Vec<f64>]>,
) -> Result<DMatrix<f64>> {
    let k = constrained_params.len();
    let unconstrained = untransform_params(constrained_params, config)?;

    // Base loglike
    let f0 = eval_loglike_at_unconstrained(&unconstrained, config, endog, exog)?;

    // Step sizes
    let h: Vec<f64> = unconstrained
        .iter()
        .map(|&x| f64::max(1e-5, 1e-4 * f64::max(1.0, x.abs())))
        .collect();

    // Pre-compute f(x ± h_i) for diagonal and mixed terms
    let mut f_plus = vec![0.0_f64; k];
    let mut f_minus = vec![0.0_f64; k];

    for i in 0..k {
        let mut x_p = unconstrained.clone();
        let mut x_m = unconstrained.clone();
        x_p[i] += h[i];
        x_m[i] -= h[i];
        f_plus[i] = eval_loglike_at_unconstrained(&x_p, config, endog, exog)
            .unwrap_or(f64::NAN);
        f_minus[i] = eval_loglike_at_unconstrained(&x_m, config, endog, exog)
            .unwrap_or(f64::NAN);
    }

    let mut hess_unc = DMatrix::zeros(k, k);

    // Diagonal: H_ii = (f(x+h) - 2f(x) + f(x-h)) / h²
    for i in 0..k {
        hess_unc[(i, i)] = (f_plus[i] - 2.0 * f0 + f_minus[i]) / (h[i] * h[i]);
    }

    // Off-diagonal: H_ij = (f++ - f+- - f-+ + f--) / (4·hi·hj)
    for i in 0..k {
        for j in (i + 1)..k {
            let mut x_pp = unconstrained.clone();
            let mut x_pm = unconstrained.clone();
            let mut x_mp = unconstrained.clone();
            let mut x_mm = unconstrained.clone();

            x_pp[i] += h[i];
            x_pp[j] += h[j];
            x_pm[i] += h[i];
            x_pm[j] -= h[j];
            x_mp[i] -= h[i];
            x_mp[j] += h[j];
            x_mm[i] -= h[i];
            x_mm[j] -= h[j];

            let f_pp =
                eval_loglike_at_unconstrained(&x_pp, config, endog, exog).unwrap_or(f64::NAN);
            let f_pm =
                eval_loglike_at_unconstrained(&x_pm, config, endog, exog).unwrap_or(f64::NAN);
            let f_mp =
                eval_loglike_at_unconstrained(&x_mp, config, endog, exog).unwrap_or(f64::NAN);
            let f_mm =
                eval_loglike_at_unconstrained(&x_mm, config, endog, exog).unwrap_or(f64::NAN);

            let val = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h[i] * h[j]);
            hess_unc[(i, j)] = val;
            hess_unc[(j, i)] = val;
        }
    }

    // Chain rule: H_constrained = J' · H_unconstrained · J
    let jac = compute_transform_jacobian(&unconstrained, config)?;
    let hess_con = jac.transpose() * &hess_unc * &jac;

    Ok(hess_con)
}

// ---------------------------------------------------------------------------
// Method 2: OPG (Outer Product of Gradients)
// ---------------------------------------------------------------------------

/// Compute OPG information matrix from per-observation scores.
///
/// Returns: I_OPG = (1/n_eff) · Σ s_t · s_t^T
pub fn opg_information_matrix(
    endog: &[f64],
    config: &SarimaxConfig,
    constrained_params: &[f64],
    exog: Option<&[Vec<f64>]>,
) -> Result<DMatrix<f64>> {
    let sparams = SarimaxParams::from_flat(constrained_params, config)?;
    let ss = StateSpace::new(config, &sparams, endog, exog)?;
    let init = KalmanInit::from_config_default(&ss, config);

    let scores = score::score_obs(
        endog,
        &ss,
        &init,
        config,
        &sparams,
        config.concentrate_scale,
        exog,
    )?;

    let k = constrained_params.len();
    let n_eff = scores.len();
    if n_eff == 0 {
        return Ok(DMatrix::zeros(k, k));
    }

    let mut info = DMatrix::zeros(k, k);
    for s_t in &scores {
        for i in 0..k {
            for j in i..k {
                let val = s_t[i] * s_t[j];
                info[(i, j)] += val;
                if i != j {
                    info[(j, i)] += val;
                }
            }
        }
    }
    info /= n_eff as f64;

    Ok(info)
}

// ---------------------------------------------------------------------------
// Shared inference from information matrix
// ---------------------------------------------------------------------------

/// Compute inference statistics from an information matrix.
fn inference_from_information(
    info: &DMatrix<f64>,
    constrained_params: &[f64],
    alpha: f64,
    method: &str,
) -> InferenceResult {
    let k = constrained_params.len();
    let (cov, singular) = safe_inverse(info);

    let std_err: Vec<f64> = (0..k)
        .map(|i| {
            let v = cov[(i, i)];
            if v > 0.0 {
                v.sqrt()
            } else {
                f64::NAN
            }
        })
        .collect();

    let z_stat: Vec<f64> = (0..k)
        .map(|i| {
            if std_err[i].is_finite() && std_err[i] > 0.0 {
                constrained_params[i] / std_err[i]
            } else {
                f64::NAN
            }
        })
        .collect();

    let norm = match Normal::new(0.0, 1.0) {
        Ok(n) => n,
        Err(_) => {
            // Unreachable with constant args, but keep panic-free for policy
            return InferenceResult {
                method: method.to_string(),
                cov_params: vec![f64::NAN; k * k],
                std_err: vec![f64::NAN; k],
                z_stat: vec![f64::NAN; k],
                p_value: vec![f64::NAN; k],
                ci_lower: vec![f64::NAN; k],
                ci_upper: vec![f64::NAN; k],
                n_params: k,
                status: "failed".to_string(),
                message: Some("internal error: could not create Normal distribution".into()),
            };
        }
    };
    let p_value: Vec<f64> = z_stat
        .iter()
        .map(|&z| {
            if z.is_finite() {
                2.0 * (1.0 - norm.cdf(z.abs()))
            } else {
                f64::NAN
            }
        })
        .collect();

    // z-critical for CI
    let z_crit = norm.inverse_cdf(1.0 - alpha / 2.0);

    let ci_lower: Vec<f64> = (0..k)
        .map(|i| constrained_params[i] - z_crit * std_err[i])
        .collect();
    let ci_upper: Vec<f64> = (0..k)
        .map(|i| constrained_params[i] + z_crit * std_err[i])
        .collect();

    // Flatten cov to row-major Vec
    let mut cov_flat = Vec::with_capacity(k * k);
    for i in 0..k {
        for j in 0..k {
            cov_flat.push(cov[(i, j)]);
        }
    }

    // Determine status
    let n_bad = std_err.iter().filter(|v| !v.is_finite()).count();
    let (status, message) = if n_bad == k {
        (
            "failed".to_string(),
            Some("all standard errors are non-finite; information matrix may be singular".into()),
        )
    } else if n_bad > 0 || singular {
        let detail = if singular && n_bad == 0 {
            "information matrix is near-singular; standard errors may be unreliable. \
             Consider inference='opg'.".to_string()
        } else {
            format!(
                "{} of {} parameters have non-finite SE (singular information matrix)",
                n_bad, k
            )
        };
        (
            "partial".to_string(),
            Some(detail),
        )
    } else {
        ("ok".to_string(), None)
    };

    InferenceResult {
        method: method.to_string(),
        cov_params: cov_flat,
        std_err,
        z_stat,
        p_value,
        ci_lower,
        ci_upper,
        n_params: k,
        status,
        message,
    }
}

// ---------------------------------------------------------------------------
// Diagnostic tests
// ---------------------------------------------------------------------------

/// Compute diagnostic tests on standardized residuals.
pub fn compute_diagnostics(resid: &[f64], n_params: usize) -> DiagnosticsResult {
    // Precompute shared moments to avoid redundant O(n) passes
    let n = resid.len();
    let mean = if n > 0 { resid.iter().sum::<f64>() / n as f64 } else { 0.0 };
    let var_sum: f64 = resid.iter().map(|&x| (x - mean).powi(2)).sum();

    // Ljung-Box (uses mean + var_sum)
    let (lb_stat, lb_pval, lb_df) = ljung_box(resid, n_params, mean, var_sum);

    // Jarque-Bera (uses mean + var_sum/n = m2)
    let (jb_stat, jb_pval) = jarque_bera(resid, mean, var_sum);

    // Heteroskedasticity (uses subset means — independent)
    let (het_stat, het_pval) = heteroskedasticity_breakvar(resid);

    DiagnosticsResult {
        ljung_box_stat: lb_stat,
        ljung_box_pvalue: lb_pval,
        ljung_box_df: lb_df,
        jarque_bera_stat: jb_stat,
        jarque_bera_pvalue: jb_pval,
        het_stat,
        het_pvalue: het_pval,
    }
}

/// Ljung-Box Q test for serial correlation.
///
/// Q = n(n+2) · Σ_{k=1}^{lags} r_k² / (n-k)
/// P-value from chi-squared(lags - n_params) if df > 0.
fn ljung_box(resid: &[f64], n_params: usize, mean: f64, var_sum: f64) -> (f64, f64, usize) {
    let n = resid.len();
    let lags = std::cmp::min(10, n / 5).max(1);

    if var_sum <= 0.0 {
        return (0.0, 1.0, lags);
    }

    // Autocorrelations
    let mut q = 0.0;
    for k in 1..=lags {
        let mut acov = 0.0;
        for t in k..n {
            acov += (resid[t] - mean) * (resid[t - k] - mean);
        }
        let rk = acov / var_sum;
        q += rk * rk / (n - k) as f64;
    }
    q *= n as f64 * (n + 2) as f64;

    let df = if lags > n_params {
        lags - n_params
    } else {
        lags
    };

    let pval = if let Ok(chi2) = ChiSquared::new(df as f64) {
        1.0 - chi2.cdf(q)
    } else {
        f64::NAN
    };

    (q, pval, df)
}

/// Jarque-Bera normality test.
///
/// JB = (n/6) · (S² + K²/4)  where S=skewness, K=excess kurtosis
/// P-value from chi-squared(2).
fn jarque_bera(resid: &[f64], mean: f64, var_sum: f64) -> (f64, f64) {
    let n = resid.len() as f64;
    if n < 3.0 {
        return (0.0, 1.0);
    }

    let m2 = var_sum / n;
    let m3: f64 = resid.iter().map(|&x| (x - mean).powi(3)).sum::<f64>() / n;
    let m4: f64 = resid.iter().map(|&x| (x - mean).powi(4)).sum::<f64>() / n;

    if m2 <= 0.0 {
        return (0.0, 1.0);
    }

    let skew = m3 / m2.powf(1.5);
    let kurt = m4 / (m2 * m2) - 3.0; // excess kurtosis

    let jb = (n / 6.0) * (skew * skew + kurt * kurt / 4.0);

    let pval = if let Ok(chi2) = ChiSquared::new(2.0) {
        1.0 - chi2.cdf(jb)
    } else {
        f64::NAN
    };

    (jb, pval)
}

/// Heteroskedasticity test (breakvar): first third vs last third variance ratio.
///
/// H(h) = var(last h) / var(first h),  h = round(n/3)
/// P-value from F(h, h) distribution.
fn heteroskedasticity_breakvar(resid: &[f64]) -> (f64, f64) {
    let n = resid.len();
    let h = ((n as f64) / 3.0).round() as usize;
    if h < 2 || 2 * h > n {
        return (1.0, 1.0);
    }

    let first = &resid[..h];
    let last = &resid[n - h..];

    let var_first: f64 = {
        let m = first.iter().sum::<f64>() / h as f64;
        first.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (h - 1) as f64
    };
    let var_last: f64 = {
        let m = last.iter().sum::<f64>() / h as f64;
        last.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (h - 1) as f64
    };

    if var_first <= 0.0 {
        return (f64::INFINITY, 0.0);
    }

    let f_stat = var_last / var_first;
    let df = (h - 1) as f64;

    let pval = if let Ok(f_dist) = FisherSnedecor::new(df, df) {
        // Two-sided: 2 · min(cdf, 1-cdf)
        let p = f_dist.cdf(f_stat);
        2.0 * p.min(1.0 - p)
    } else {
        f64::NAN
    };

    (f_stat, pval)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Evaluate loglike at given unconstrained parameters.
fn eval_loglike_at_unconstrained(
    unconstrained: &[f64],
    config: &SarimaxConfig,
    endog: &[f64],
    exog: Option<&[Vec<f64>]>,
) -> Result<f64> {
    let output =
        crate::pipeline::kalman_eval_unconstrained(endog, unconstrained, config, exog)?;
    if output.loglike.is_finite() {
        Ok(output.loglike)
    } else {
        Err(SarimaxError::DataError("non-finite loglike".into()))
    }
}

/// Compute numerical Jacobian of transform_params.
///
/// Delegates to [`crate::params::transform_jacobian`].
fn compute_transform_jacobian(
    unconstrained: &[f64],
    config: &SarimaxConfig,
) -> Result<DMatrix<f64>> {
    crate::params::transform_jacobian(unconstrained, config)
}

/// Safe matrix inverse with pseudo-inverse fallback.
///
/// Returns (inverse_matrix, was_singular).
fn safe_inverse(mat: &DMatrix<f64>) -> (DMatrix<f64>, bool) {
    let k = mat.nrows();
    // Try LU inverse
    if let Some(inv) = mat.clone().try_inverse() {
        return (inv, false);
    }
    // SVD pseudo-inverse fallback
    let threshold = 1e-10 * k as f64;
    match nalgebra::SVD::new(mat.clone(), true, true).pseudo_inverse(threshold) {
        Ok(pinv) => (pinv, true),
        Err(_) => (DMatrix::zeros(k, k), true),
    }
}

/// Compute n_eff = n - burn for the model.
fn compute_n_eff(endog: &[f64], config: &SarimaxConfig) -> Result<usize> {
    let n = endog.len();
    let sd = config.order.k_states_diff();
    let burn = if config.simple_differencing { 0 } else { sd };
    if n <= burn {
        return Err(SarimaxError::DataError(format!(
            "n={} <= burn={}",
            n, burn
        )));
    }
    Ok(n - burn)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::make_config_enforced;

    fn generate_ar1_data(n: usize, phi: f64, seed: u64) -> Vec<f64> {
        let mut y = vec![0.0; n];
        let mut state: u64 = seed;
        for t in 1..n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((state >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0;
            y[t] = phi * y[t - 1] + u;
        }
        // cumsum for integrated
        for t in 1..n {
            y[t] += y[t - 1];
        }
        y
    }

    #[test]
    fn test_numerical_hessian_ar1() {
        let data = generate_ar1_data(500, 0.5, 42);
        let config = make_config_enforced(1, 0, 0);

        // Fit to get params
        let fit = crate::optimizer::fit(&data, &config, None, Some("lbfgsb"), Some(200), None)
            .expect("fit should succeed");

        let hess = numerical_hessian(&data, &config, &fit.params, None)
            .expect("hessian should succeed");

        let k = fit.params.len();
        assert_eq!(hess.nrows(), k);
        assert_eq!(hess.ncols(), k);

        // Hessian should be negative definite at MLE → info matrix positive definite
        // So diagonal of -H should be positive
        for i in 0..k {
            assert!(
                -hess[(i, i)] > 0.0,
                "info matrix diagonal [{},{}] = {} should be positive",
                i,
                i,
                -hess[(i, i)]
            );
        }
    }

    #[test]
    fn test_hessian_symmetry() {
        let data = generate_ar1_data(300, 0.3, 123);
        let config = make_config_enforced(1, 1, 1); // ARIMA(1,1,1), k=2

        let fit = crate::optimizer::fit(&data, &config, None, Some("lbfgsb"), Some(200), None)
            .expect("fit should succeed");

        let hess = numerical_hessian(&data, &config, &fit.params, None)
            .expect("hessian should succeed");

        let k = fit.params.len();
        for i in 0..k {
            for j in (i + 1)..k {
                let diff = (hess[(i, j)] - hess[(j, i)]).abs();
                assert!(
                    diff < 1e-6,
                    "Hessian not symmetric: H[{},{}]={}, H[{},{}]={}, diff={}",
                    i,
                    j,
                    hess[(i, j)],
                    j,
                    i,
                    hess[(j, i)],
                    diff
                );
            }
        }
    }

    #[test]
    fn test_inference_std_err_positive() {
        let data = generate_ar1_data(500, 0.5, 42);
        let config = make_config_enforced(1, 1, 1);

        let fit = crate::optimizer::fit(&data, &config, None, Some("lbfgsb"), Some(200), None)
            .expect("fit should succeed");

        let result =
            compute_inference(&data, &config, &fit.params, "hessian", 0.05, None)
                .expect("inference should succeed");

        assert_eq!(result.n_params, fit.params.len());
        for (i, &se) in result.std_err.iter().enumerate() {
            assert!(
                se.is_finite() && se > 0.0,
                "SE[{}] = {} should be finite and positive",
                i,
                se
            );
        }
    }

    #[test]
    fn test_pvalue_range() {
        let data = generate_ar1_data(500, 0.5, 42);
        let config = make_config_enforced(1, 1, 1);

        let fit = crate::optimizer::fit(&data, &config, None, Some("lbfgsb"), Some(200), None)
            .expect("fit should succeed");

        let result =
            compute_inference(&data, &config, &fit.params, "hessian", 0.05, None)
                .expect("inference should succeed");

        for (i, &p) in result.p_value.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&p),
                "p-value[{}] = {} should be in [0, 1]",
                i,
                p
            );
        }
    }

    #[test]
    fn test_ci_covers_param() {
        let data = generate_ar1_data(500, 0.5, 42);
        let config = make_config_enforced(1, 1, 1);

        let fit = crate::optimizer::fit(&data, &config, None, Some("lbfgsb"), Some(200), None)
            .expect("fit should succeed");

        let result =
            compute_inference(&data, &config, &fit.params, "hessian", 0.05, None)
                .expect("inference should succeed");

        for i in 0..fit.params.len() {
            assert!(
                result.ci_lower[i] <= fit.params[i] && fit.params[i] <= result.ci_upper[i],
                "param[{}]={} not in CI [{}, {}]",
                i,
                fit.params[i],
                result.ci_lower[i],
                result.ci_upper[i]
            );
        }
    }

    #[test]
    fn test_ljung_box_white_noise() {
        // White noise should not be rejected
        let mut rng_state: u64 = 42;
        let mut resid = vec![0.0; 500];
        for r in resid.iter_mut() {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *r = ((rng_state >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0;
        }

        let diag = compute_diagnostics(&resid, 0);
        assert!(
            diag.ljung_box_pvalue > 0.01,
            "Ljung-Box p={} should not reject white noise",
            diag.ljung_box_pvalue
        );
    }

    #[test]
    fn test_jarque_bera_approx_normal() {
        // Approximately normal data should not be strongly rejected
        let mut rng_state: u64 = 123;
        let mut resid = vec![0.0; 1000];
        for r in resid.iter_mut() {
            // Sum of 12 uniform → approx normal (CLT)
            let mut s = 0.0;
            for _ in 0..12 {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                s += (rng_state >> 33) as f64 / (1u64 << 31) as f64;
            }
            *r = s - 6.0; // center at 0
        }

        let diag = compute_diagnostics(&resid, 0);
        assert!(
            diag.jarque_bera_pvalue > 0.01,
            "JB p={} should not strongly reject near-normal data",
            diag.jarque_bera_pvalue
        );
    }
}
