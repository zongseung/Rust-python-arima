//! SARIMAX parameter optimization via L-BFGS with Nelder-Mead fallback.
//!
//! This module provides:
//! - Parameter space transformations (constrained ↔ unconstrained)
//! - Negative log-likelihood objective function for argmin
//! - `fit()` function: the main entry point for model fitting

use argmin::core::{CostFunction, Executor, Gradient, State};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use argmin::solver::neldermead::NelderMead;

use crate::error::{Result, SarimaxError};
use crate::initialization::KalmanInit;
use crate::kalman::kalman_loglike;
use crate::params::{
    self, SarimaxParams,
};
use crate::start_params::compute_start_params;
use crate::state_space::StateSpace;
use crate::types::{FitResult, SarimaxConfig};

// ---------------------------------------------------------------------------
// Parameter transformations (constrained ↔ unconstrained)
// ---------------------------------------------------------------------------

/// Transform constrained parameters to unconstrained space for optimization.
///
/// Layout: `[trend | exog | ar(p) | ma(q) | sar(P) | sma(Q) | sigma2?]`
pub fn untransform_params(constrained: &[f64], config: &SarimaxConfig) -> Result<Vec<f64>> {
    let kt = config.trend.k_trend();
    let n_exog = config.n_exog;
    let p = config.order.p;
    let q = config.order.q;
    let pp = config.order.pp;
    let qq = config.order.qq;

    let mut out = Vec::with_capacity(constrained.len());
    let mut i = 0;

    // Trend + exog: pass through
    out.extend_from_slice(&constrained[i..i + kt + n_exog]);
    i += kt + n_exog;

    // AR coefficients
    if config.enforce_stationarity && p > 0 {
        out.extend(params::unconstrain_stationary(&constrained[i..i + p]));
    } else {
        out.extend_from_slice(&constrained[i..i + p]);
    }
    i += p;

    // MA coefficients
    if config.enforce_invertibility && q > 0 {
        out.extend(params::unconstrain_invertible(&constrained[i..i + q]));
    } else {
        out.extend_from_slice(&constrained[i..i + q]);
    }
    i += q;

    // Seasonal AR
    if config.enforce_stationarity && pp > 0 {
        out.extend(params::unconstrain_stationary(&constrained[i..i + pp]));
    } else {
        out.extend_from_slice(&constrained[i..i + pp]);
    }
    i += pp;

    // Seasonal MA
    if config.enforce_invertibility && qq > 0 {
        out.extend(params::unconstrain_invertible(&constrained[i..i + qq]));
    } else {
        out.extend_from_slice(&constrained[i..i + qq]);
    }
    i += qq;

    // sigma2
    if !config.concentrate_scale && i < constrained.len() {
        out.push(params::unconstrain_variance(constrained[i])?);
    }

    Ok(out)
}

/// Transform unconstrained parameters back to constrained space.
pub fn transform_params(unconstrained: &[f64], config: &SarimaxConfig) -> Vec<f64> {
    let kt = config.trend.k_trend();
    let n_exog = config.n_exog;
    let p = config.order.p;
    let q = config.order.q;
    let pp = config.order.pp;
    let qq = config.order.qq;

    let mut out = Vec::with_capacity(unconstrained.len());
    let mut i = 0;

    // Trend + exog: pass through
    out.extend_from_slice(&unconstrained[i..i + kt + n_exog]);
    i += kt + n_exog;

    // AR
    if config.enforce_stationarity && p > 0 {
        out.extend(params::constrain_stationary(&unconstrained[i..i + p]));
    } else {
        out.extend_from_slice(&unconstrained[i..i + p]);
    }
    i += p;

    // MA
    if config.enforce_invertibility && q > 0 {
        out.extend(params::constrain_invertible(&unconstrained[i..i + q]));
    } else {
        out.extend_from_slice(&unconstrained[i..i + q]);
    }
    i += q;

    // Seasonal AR
    if config.enforce_stationarity && pp > 0 {
        out.extend(params::constrain_stationary(&unconstrained[i..i + pp]));
    } else {
        out.extend_from_slice(&unconstrained[i..i + pp]);
    }
    i += pp;

    // Seasonal MA
    if config.enforce_invertibility && qq > 0 {
        out.extend(params::constrain_invertible(&unconstrained[i..i + qq]));
    } else {
        out.extend_from_slice(&unconstrained[i..i + qq]);
    }
    i += qq;

    // sigma2
    if !config.concentrate_scale && i < unconstrained.len() {
        out.push(params::constrain_variance(unconstrained[i]));
    }

    out
}

// ---------------------------------------------------------------------------
// Objective function for argmin
// ---------------------------------------------------------------------------

/// Negative log-likelihood objective for optimizer.
#[derive(Clone)]
struct SarimaxObjective {
    endog: Vec<f64>,
    config: SarimaxConfig,
}

impl SarimaxObjective {
    /// Evaluate log-likelihood for given unconstrained parameters.
    fn eval_loglike(&self, unconstrained: &[f64]) -> std::result::Result<f64, String> {
        let constrained = transform_params(unconstrained, &self.config);

        let sparams = SarimaxParams::from_flat(&constrained, &self.config)
            .map_err(|e| e.to_string())?;

        let ss = StateSpace::new(&self.config, &sparams, &self.endog, None)
            .map_err(|e| e.to_string())?;

        let init = KalmanInit::from_config(&ss, &self.config, KalmanInit::default_kappa());

        let output = kalman_loglike(&self.endog, &ss, &init, self.config.concentrate_scale)
            .map_err(|e| e.to_string())?;

        if output.loglike.is_finite() {
            Ok(output.loglike)
        } else {
            Err("non-finite log-likelihood".to_string())
        }
    }
}

impl CostFunction for SarimaxObjective {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Vec<f64>) -> std::result::Result<f64, argmin::core::Error> {
        match self.eval_loglike(param) {
            Ok(ll) => Ok(-ll), // minimize negative log-likelihood
            Err(_) => Ok(f64::MAX / 2.0), // penalty for invalid parameters
        }
    }
}

impl Gradient for SarimaxObjective {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, param: &Vec<f64>) -> std::result::Result<Vec<f64>, argmin::core::Error> {
        let n = param.len();
        let mut grad = vec![0.0; n];
        let eps = f64::EPSILON.sqrt(); // ~1.49e-8, optimal for forward-diff

        // Forward-diff: n+1 evaluations (vs center-diff 2n+1)
        let f0 = self.cost(param)?;
        let mut p_work = param.clone(); // single work buffer

        for i in 0..n {
            let orig = p_work[i];
            p_work[i] = orig + eps;
            let f_plus = self.cost(&p_work)?;
            p_work[i] = orig; // restore

            grad[i] = (f_plus - f0) / eps;

            // Fallback to center-diff if forward-diff yields NaN/Inf
            if !grad[i].is_finite() {
                p_work[i] = orig + eps;
                let fp = self.cost(&p_work)?;
                p_work[i] = orig - eps;
                let fm = self.cost(&p_work)?;
                p_work[i] = orig;
                grad[i] = (fp - fm) / (2.0 * eps);
                if !grad[i].is_finite() {
                    grad[i] = 0.0;
                }
            }
        }

        Ok(grad)
    }
}

// ---------------------------------------------------------------------------
// L-BFGS optimization
// ---------------------------------------------------------------------------

fn run_lbfgs(
    objective: SarimaxObjective,
    init_params: Vec<f64>,
    maxiter: u64,
) -> std::result::Result<(Vec<f64>, f64, u64, bool), String> {
    let linesearch = MoreThuenteLineSearch::new();
    let solver = LBFGS::new(linesearch, 7)
        .with_tolerance_grad(1e-7)
        .map_err(|e| e.to_string())?
        .with_tolerance_cost(1e-9)
        .map_err(|e| e.to_string())?;

    let result = Executor::new(objective, solver)
        .configure(|state: argmin::core::IterState<Vec<f64>, Vec<f64>, (), (), (), f64>| {
            state.param(init_params).max_iters(maxiter)
        })
        .run()
        .map_err(|e| format!("L-BFGS failed: {}", e))?;

    let state = result.state();
    let best_param = state.get_best_param()
        .ok_or("L-BFGS: no best parameter found")?
        .clone();
    let best_cost = state.get_best_cost();
    let n_iter = state.get_iter();
    let converged = state.get_termination_status().terminated();

    Ok((best_param, best_cost, n_iter, converged))
}

// ---------------------------------------------------------------------------
// Nelder-Mead fallback
// ---------------------------------------------------------------------------

fn run_nelder_mead(
    objective: SarimaxObjective,
    init_params: Vec<f64>,
    maxiter: u64,
) -> std::result::Result<(Vec<f64>, f64, u64, bool), String> {
    let n = init_params.len();

    // Build simplex: n+1 vertices
    let mut simplex = vec![init_params.clone()];
    for i in 0..n {
        let mut vertex = init_params.clone();
        let delta = if vertex[i].abs() > 1e-8 {
            vertex[i] * 0.05
        } else {
            0.00025
        };
        vertex[i] += delta;
        simplex.push(vertex);
    }

    let solver = NelderMead::new(simplex)
        .with_sd_tolerance(1e-6)
        .map_err(|e| e.to_string())?;

    let result = Executor::new(objective, solver)
        .configure(|state: argmin::core::IterState<Vec<f64>, (), (), (), (), f64>| {
            state.max_iters(maxiter)
        })
        .run()
        .map_err(|e| format!("Nelder-Mead failed: {}", e))?;

    let state = result.state();
    let best_param = state.get_best_param()
        .ok_or("Nelder-Mead: no best parameter found")?
        .clone();
    let best_cost = state.get_best_cost();
    let n_iter = state.get_iter();
    let converged = state.get_termination_status().terminated();

    Ok((best_param, best_cost, n_iter, converged))
}

// ---------------------------------------------------------------------------
// Public fit() entry point
// ---------------------------------------------------------------------------

/// Fit a SARIMAX model using maximum likelihood estimation.
///
/// # Arguments
/// * `endog` — Observed time series
/// * `config` — Model configuration (order, stationarity enforcement, etc.)
/// * `start_params` — Optional initial parameter values (constrained space)
/// * `method` — "lbfgs" (default), "nelder-mead", or "lbfgs+nm" (fallback)
/// * `maxiter` — Maximum iterations (default: 500)
pub fn fit(
    endog: &[f64],
    config: &SarimaxConfig,
    start_params: Option<&[f64]>,
    method: Option<&str>,
    maxiter: Option<u64>,
) -> Result<FitResult> {
    let maxiter = maxiter.unwrap_or(500);
    let method = method.unwrap_or("lbfgs");

    // 1. Get starting parameters
    let constrained_start = match start_params {
        Some(sp) => {
            // Validate start_params length before use
            let expected_len = config.trend.k_trend()
                + config.n_exog
                + config.order.p
                + config.order.q
                + config.order.pp
                + config.order.qq
                + if config.concentrate_scale { 0 } else { 1 };
            if sp.len() != expected_len {
                return Err(SarimaxError::ParamLengthMismatch {
                    expected: expected_len,
                    got: sp.len(),
                });
            }
            sp.to_vec()
        }
        None => compute_start_params(endog, config)?,
    };

    // 2. Transform to unconstrained space
    let unconstrained_start = untransform_params(&constrained_start, config)?;

    let objective = SarimaxObjective {
        endog: endog.to_vec(),
        config: config.clone(),
    };

    // Determine number of restarts based on model complexity
    let n_params_total = unconstrained_start.len();
    let has_seasonal = config.order.pp > 0 || config.order.qq > 0;
    let n_restarts = if n_params_total >= 4 { 3 }
        else if n_params_total >= 3 || has_seasonal { 2 }
        else if n_params_total >= 2 { 1 }
        else { 0 };

    // 3. Run optimization
    let (best_unconstrained, _best_cost, n_iter, converged, used_method) = match method {
        "nelder-mead" | "nm" => {
            let (p, c, n, conv) = run_nelder_mead(objective.clone(), unconstrained_start, maxiter)
                .map_err(|e| SarimaxError::OptimizationFailed(e))?;
            (p, c, n, conv, "nelder-mead".to_string())
        }
        "lbfgs" => {
            // Initial L-BFGS run
            let initial_result = match run_lbfgs(objective.clone(), unconstrained_start.clone(), maxiter) {
                Ok((p, c, n, conv)) => Some((p, c, n, conv, "lbfgs".to_string())),
                Err(_) => None,
            };

            // Multi-start: try perturbed starting points for complex models
            let mut best = initial_result;

            // Helper to update best with a new result
            let mut try_update = |p: Vec<f64>, c: f64, n: u64, conv: bool| {
                match &best {
                    Some((_, best_cost, _, _, _)) if c < *best_cost => {
                        best = Some((p, c, n, conv, "lbfgs".to_string()));
                    }
                    None => {
                        best = Some((p, c, n, conv, "lbfgs".to_string()));
                    }
                    _ => {}
                }
            };

            if n_restarts > 0 {
                // 1. Try starting from zeros in unconstrained space
                let zeros = vec![0.0; n_params_total];
                if let Ok((p, c, n, conv)) = run_lbfgs(objective.clone(), zeros, maxiter) {
                    try_update(p, c, n, conv);
                }

                // 2. For seasonal MA models with enforced invertibility, try Nelder-Mead from grid starts
                if config.enforce_invertibility && config.order.qq > 0 {
                    let kt = config.trend.k_trend();
                    let n_exog = config.n_exog;
                    let ma_start = kt + n_exog + config.order.p;
                    let sma_start = ma_start + config.order.q + config.order.pp;

                    // NM from grid of constrained MA/SMA starts (gradient-free avoids boundary traps)
                    let grid_vals = [-0.3, -0.6, -0.9];
                    for &ma_val in &grid_vals {
                        let mut grid_constrained = vec![0.0; n_params_total];
                        for i in 0..config.order.q {
                            grid_constrained[ma_start + i] = ma_val;
                        }
                        for i in 0..config.order.qq {
                            grid_constrained[sma_start + i] = ma_val;
                        }
                        if let Ok(grid_uncons) = untransform_params(&grid_constrained, config) {
                            if let Ok((p, c, n, conv)) = run_nelder_mead(objective.clone(), grid_uncons, maxiter) {
                                try_update(p, c, n, conv);
                            }
                        }
                    }
                }

                // 3. Deterministic LCG perturbations
                let mut rng_state: u64 = 12345;
                for _ in 0..n_restarts {
                    let mut perturbed = unconstrained_start.clone();
                    for v in perturbed.iter_mut() {
                        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        let u = ((rng_state >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
                        let scale = if v.abs() > 0.1 { v.abs() * 0.5 } else { 0.1 };
                        *v += u * scale;
                    }
                    if let Ok((p, c, n, conv)) = run_lbfgs(objective.clone(), perturbed, maxiter) {
                        try_update(p, c, n, conv);
                    }
                }
            }

            match best {
                Some((best_p, best_c, best_n, _best_conv, _)) => {
                    // Refine with Nelder-Mead (gradient-free, can escape flat gradient regions)
                    match run_nelder_mead(objective.clone(), best_p.clone(), maxiter / 2) {
                        Ok((nm_p, nm_c, nm_n, nm_conv)) if nm_c < best_c => {
                            (nm_p, nm_c, best_n + nm_n, nm_conv, "lbfgs+nm".to_string())
                        }
                        _ => (best_p, best_c, best_n, true, "lbfgs".to_string()),
                    }
                }
                None => {
                    // All L-BFGS attempts failed, fallback to Nelder-Mead
                    let (p, c, n, conv) = run_nelder_mead(objective.clone(), unconstrained_start, maxiter)
                        .map_err(|e| SarimaxError::OptimizationFailed(e))?;
                    (p, c, n, conv, "nelder-mead (fallback)".to_string())
                }
            }
        }
        _ => {
            return Err(SarimaxError::OptimizationFailed(format!(
                "unknown method: '{}'. Use 'lbfgs' or 'nelder-mead'",
                method
            )));
        }
    };

    // 4. Transform back to constrained space
    let final_constrained = transform_params(&best_unconstrained, config);

    // 5. Evaluate final log-likelihood
    let final_params = SarimaxParams::from_flat(&final_constrained, config)?;
    let ss = StateSpace::new(config, &final_params, endog, None)?;
    let init = KalmanInit::from_config(&ss, config, KalmanInit::default_kappa());
    let output = kalman_loglike(endog, &ss, &init, config.concentrate_scale)?;

    let n_params = SarimaxParams::n_estimated_params(config);

    let result = FitResult {
        params: final_constrained,
        loglike: output.loglike,
        scale: output.scale,
        n_obs: endog.len(),
        n_params,
        n_iter,
        converged,
        method: used_method,
        aic: 0.0,
        bic: 0.0,
    }
    .with_information_criteria();

    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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

    #[test]
    fn test_transform_untransform_roundtrip() {
        let config = make_config(2, 0, 1, true, true);
        let original = vec![0.5, -0.3, 0.2]; // ar(2), ma(1)
        let unconstrained = untransform_params(&original, &config).unwrap();
        let recovered = transform_params(&unconstrained, &config);
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "roundtrip failed: {} vs {}",
                a, b
            );
        }
    }

    #[test]
    fn test_transform_passthrough_no_enforce() {
        let config = make_config(1, 0, 1, false, false);
        let original = vec![0.7, -0.3];
        let unconstrained = untransform_params(&original, &config).unwrap();
        assert_eq!(original, unconstrained);
    }

    #[test]
    fn test_objective_finite() {
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap())
            .collect();

        let config = make_config(1, 0, 0, false, false);
        let obj = SarimaxObjective {
            endog: data,
            config,
        };

        let cost = obj.cost(&vec![0.5]).unwrap();
        assert!(cost.is_finite(), "cost should be finite: {}", cost);
    }

    #[test]
    fn test_gradient_finite() {
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap())
            .collect();

        let config = make_config(1, 0, 0, false, false);
        let obj = SarimaxObjective {
            endog: data,
            config,
        };

        let grad = obj.gradient(&vec![0.5]).unwrap();
        assert_eq!(grad.len(), 1);
        assert!(grad[0].is_finite(), "gradient should be finite: {}", grad[0]);
    }

    #[test]
    fn test_fit_ar1() {
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap())
            .collect();
        let expected_params: Vec<f64> = case["params"]
            .as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap())
            .collect();
        let expected_loglike = case["loglike"].as_f64().unwrap();

        // Fixture was generated with approximate_diffuse init, so use enforce=false
        let config = make_config(1, 0, 0, false, false);
        let result = fit(&data, &config, None, Some("lbfgs"), Some(500)).unwrap();

        assert!(result.converged, "AR(1) fit should converge");
        let param_err = (result.params[0] - expected_params[0]).abs();
        assert!(
            param_err < 1e-4,
            "AR(1) param error too large: {} (got {}, expected {})",
            param_err, result.params[0], expected_params[0]
        );
        let ll_err = (result.loglike - expected_loglike).abs();
        assert!(
            ll_err < 1e-2,
            "AR(1) loglike error: {} (got {}, expected {})",
            ll_err, result.loglike, expected_loglike
        );
    }

    #[test]
    fn test_fit_arma11() {
        let fixtures = load_fixtures();
        let case = &fixtures["arma11"];
        let data: Vec<f64> = case["data"]
            .as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap())
            .collect();
        let expected_params: Vec<f64> = case["params"]
            .as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap())
            .collect();

        // Fixture was generated with approximate_diffuse init
        let config = make_config(1, 0, 1, false, false);
        let result = fit(&data, &config, None, Some("lbfgs"), Some(500)).unwrap();

        for (i, (got, exp)) in result.params.iter().zip(expected_params.iter()).enumerate() {
            let err = (got - exp).abs();
            assert!(
                err < 1e-3,
                "ARMA(1,1) param[{}] error: {} (got {}, expected {})",
                i, err, got, exp
            );
        }
    }

    #[test]
    fn test_fit_arima111() {
        let fixtures = load_fixtures();
        let case = &fixtures["arima111"];
        let data: Vec<f64> = case["data"]
            .as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap())
            .collect();
        let expected_loglike = case["loglike"].as_f64().unwrap();

        // Fixture was generated with approximate_diffuse init
        let config = make_config(1, 1, 1, false, false);
        let result = fit(&data, &config, None, Some("lbfgs"), Some(500)).unwrap();

        let ll_err = (result.loglike - expected_loglike).abs();
        assert!(
            ll_err < 1.0,
            "ARIMA(1,1,1) loglike error: {} (got {}, expected {})",
            ll_err, result.loglike, expected_loglike
        );
    }

    #[test]
    fn test_fit_nelder_mead() {
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap())
            .collect();
        let expected_params: Vec<f64> = case["params"]
            .as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap())
            .collect();

        let config = make_config(1, 0, 0, false, false);
        let result = fit(&data, &config, None, Some("nelder-mead"), Some(1000)).unwrap();

        let param_err = (result.params[0] - expected_params[0]).abs();
        assert!(
            param_err < 1e-3,
            "NM AR(1) param error: {} (got {}, expected {})",
            param_err, result.params[0], expected_params[0]
        );
    }

    #[test]
    fn test_aic_bic() {
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap())
            .collect();

        let config = make_config(1, 0, 0, true, true);
        let result = fit(&data, &config, None, Some("lbfgs"), Some(500)).unwrap();

        // AIC = -2*loglike + 2*k, BIC = -2*loglike + k*ln(n)
        let k = result.n_params as f64;
        let n = result.n_obs as f64;
        let expected_aic = -2.0 * result.loglike + 2.0 * k;
        let expected_bic = -2.0 * result.loglike + k * n.ln();

        assert!(
            (result.aic - expected_aic).abs() < 1e-10,
            "AIC mismatch: got {}, expected {}",
            result.aic, expected_aic
        );
        assert!(
            (result.bic - expected_bic).abs() < 1e-10,
            "BIC mismatch: got {}, expected {}",
            result.bic, expected_bic
        );
    }

    #[test]
    fn test_fit_with_custom_start_params() {
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap())
            .collect();

        let config = make_config(1, 0, 0, false, false);
        let start = vec![0.5];
        let result = fit(&data, &config, Some(&start), Some("lbfgs"), Some(500)).unwrap();

        assert!(result.loglike.is_finite());
        assert!(result.params[0].is_finite());
    }
}
