//! SARIMAX parameter optimization via L-BFGS with Nelder-Mead fallback.
//!
//! This module provides:
//! - Parameter space transformations (constrained ↔ unconstrained)
//! - Negative log-likelihood objective function for argmin
//! - `fit()` function: the main entry point for model fitting

use argmin::core::{CostFunction, Executor, Gradient, State, TerminationReason};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::neldermead::NelderMead;
use argmin::solver::quasinewton::LBFGS;

use std::cell::RefCell;

use crate::error::{Result, SarimaxError};
use crate::initialization::KalmanInit;
use crate::kalman::kalman_loglike;
use crate::params::{self, SarimaxParams};
use crate::score;
use crate::start_params::compute_start_params;
use crate::state_space::StateSpace;
use crate::types::{FitResult, SarimaxConfig};

// ---------------------------------------------------------------------------
// Parameter transformations (constrained ↔ unconstrained)
// ---------------------------------------------------------------------------

/// Transform constrained parameters to unconstrained space for optimization.
///
/// Layout: `[trend | exog | ar(p) | ma(q) | sar(P) | sma(Q) | sigma2?]`
fn expected_param_len(config: &SarimaxConfig) -> usize {
    config.trend.k_trend()
        + config.n_exog
        + config.order.p
        + config.order.q
        + config.order.pp
        + config.order.qq
        + if config.concentrate_scale { 0 } else { 1 }
}

/// Transform constrained parameters to unconstrained space for optimization.
///
/// Layout: `[trend | exog | ar(p) | ma(q) | sar(P) | sma(Q) | sigma2?]`
pub fn untransform_params(constrained: &[f64], config: &SarimaxConfig) -> Result<Vec<f64>> {
    let expected = expected_param_len(config);
    if constrained.len() != expected {
        return Err(SarimaxError::ParamLengthMismatch {
            expected,
            got: constrained.len(),
        });
    }

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
pub fn transform_params(unconstrained: &[f64], config: &SarimaxConfig) -> Result<Vec<f64>> {
    let expected = expected_param_len(config);
    if unconstrained.len() != expected {
        return Err(SarimaxError::ParamLengthMismatch {
            expected,
            got: unconstrained.len(),
        });
    }

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

    Ok(out)
}

// ---------------------------------------------------------------------------
// Objective function for argmin
// ---------------------------------------------------------------------------

/// Cached fused evaluation result (cost + gradient at same params).
///
/// Used by L-BFGS path to avoid redundant StateSpace construction when
/// argmin calls `cost()` and `gradient()` at the same parameter point.
struct CachedEval {
    params: Vec<f64>,
    cost: f64,
    gradient: Vec<f64>,
}

/// Negative log-likelihood objective for optimizer.
struct SarimaxObjective {
    endog: Vec<f64>,
    config: SarimaxConfig,
    exog: Option<Vec<Vec<f64>>>,
    /// Single-entry cache: stores the last fused (cost, gradient) evaluation.
    /// Populated by `gradient()`, consumed by `cost()` at the same params.
    cache: RefCell<Option<CachedEval>>,
}

impl Clone for SarimaxObjective {
    fn clone(&self) -> Self {
        SarimaxObjective {
            endog: self.endog.clone(),
            config: self.config.clone(),
            exog: self.exog.clone(),
            cache: RefCell::new(None), // cloned objectives start with empty cache
        }
    }
}

impl SarimaxObjective {
    /// Evaluate negative log-likelihood for given unconstrained parameters.
    /// Used by L-BFGS-B which minimizes directly.
    fn eval_negloglike(&self, unconstrained: &[f64]) -> std::result::Result<f64, String> {
        self.eval_loglike(unconstrained).map(|ll| -ll)
    }

    /// Evaluate log-likelihood for given unconstrained parameters.
    fn eval_loglike(&self, unconstrained: &[f64]) -> std::result::Result<f64, String> {
        let constrained =
            transform_params(unconstrained, &self.config).map_err(|e| e.to_string())?;

        let sparams =
            SarimaxParams::from_flat(&constrained, &self.config).map_err(|e| e.to_string())?;

        let ss = StateSpace::new(&self.config, &sparams, &self.endog, self.exog.as_deref())
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

impl SarimaxObjective {
    /// Compute analytical gradient of negative log-likelihood in unconstrained space.
    ///
    /// Uses score() (tangent linear KF) in constrained space, then applies
    /// the chain rule via the Jacobian of transform_params.
    fn analytical_gradient_negloglike(
        &self,
        unconstrained: &[f64],
    ) -> std::result::Result<Vec<f64>, String> {
        let constrained =
            transform_params(unconstrained, &self.config).map_err(|e| e.to_string())?;

        let sparams =
            SarimaxParams::from_flat(&constrained, &self.config).map_err(|e| e.to_string())?;

        let ss = StateSpace::new(&self.config, &sparams, &self.endog, self.exog.as_deref())
            .map_err(|e| e.to_string())?;

        let init = KalmanInit::from_config(&ss, &self.config, KalmanInit::default_kappa());

        // Score in constrained space: ∂ll/∂θ_constrained
        let score_constrained = score::score(
            &self.endog,
            &ss,
            &init,
            &self.config,
            &sparams,
            self.config.concentrate_scale,
            self.exog.as_deref(),
        )
        .map_err(|e| e.to_string())?;

        // Chain rule: ∂(-ll)/∂u = -J' · ∂ll/∂θ
        // where J[j,i] = ∂θ_j / ∂u_i (Jacobian of transform_params)
        let grad = apply_transform_jacobian(&score_constrained, unconstrained, &self.config)?;

        // Return negative gradient (minimizing -loglike)
        Ok(grad.iter().map(|&g| -g).collect())
    }

    /// Fused function + gradient evaluation.
    ///
    /// Builds StateSpace and KalmanInit ONCE and computes both the negative
    /// log-likelihood and its analytical gradient. This is ~40% faster than
    /// calling eval_negloglike + analytical_gradient_negloglike separately.
    fn eval_negloglike_with_gradient(
        &self,
        unconstrained: &[f64],
    ) -> std::result::Result<(f64, Vec<f64>), String> {
        let constrained =
            transform_params(unconstrained, &self.config).map_err(|e| e.to_string())?;

        let sparams =
            SarimaxParams::from_flat(&constrained, &self.config).map_err(|e| e.to_string())?;

        let ss = StateSpace::new(&self.config, &sparams, &self.endog, self.exog.as_deref())
            .map_err(|e| e.to_string())?;

        let init = KalmanInit::from_config(&ss, &self.config, KalmanInit::default_kappa());

        // 1. Log-likelihood (forward KF)
        let output = kalman_loglike(&self.endog, &ss, &init, self.config.concentrate_scale)
            .map_err(|e| e.to_string())?;

        if !output.loglike.is_finite() {
            return Err("non-finite log-likelihood".to_string());
        }
        let negll = -output.loglike;

        // 2. Score (tangent linear KF, reuses ss and init)
        let score_constrained = score::score(
            &self.endog,
            &ss,
            &init,
            &self.config,
            &sparams,
            self.config.concentrate_scale,
            self.exog.as_deref(),
        )
        .map_err(|e| e.to_string())?;

        // 3. Chain rule: ∂(-ll)/∂u = -J' · ∂ll/∂θ
        let grad = apply_transform_jacobian(&score_constrained, unconstrained, &self.config)?;
        let neg_grad: Vec<f64> = grad.iter().map(|&g| -g).collect();

        Ok((negll, neg_grad))
    }
}

/// Apply the chain rule: grad_unconstrained = J' · grad_constrained.
///
/// Uses numerical Jacobian of transform_params (cheap: only n_params transform evaluations).
fn apply_transform_jacobian(
    score_constrained: &[f64],
    unconstrained: &[f64],
    config: &SarimaxConfig,
) -> std::result::Result<Vec<f64>, String> {
    let n = unconstrained.len();
    let eps = 1e-7;
    let c_base = transform_params(unconstrained, config).map_err(|e| e.to_string())?;
    let mut grad = vec![0.0; n];

    // Reuse buffer across iterations instead of allocating per-parameter
    let mut u_pert = unconstrained.to_vec();

    for i in 0..n {
        let orig = u_pert[i];
        u_pert[i] = orig + eps;
        let c_pert = transform_params(&u_pert, config).map_err(|e| e.to_string())?;
        u_pert[i] = orig; // reset for next iteration

        // grad[i] = Σ_j score[j] * ∂c_j/∂u_i
        for j in 0..n {
            grad[i] += score_constrained[j] * (c_pert[j] - c_base[j]) / eps;
        }
    }
    Ok(grad)
}

impl CostFunction for SarimaxObjective {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Vec<f64>) -> std::result::Result<f64, argmin::core::Error> {
        // Check cache (populated by gradient() via fused eval)
        if let Some(ref cached) = *self.cache.borrow() {
            if cached.params == *param {
                return Ok(cached.cost);
            }
        }

        match self.eval_loglike(param) {
            Ok(ll) => Ok(-ll),            // minimize negative log-likelihood
            Err(_) => Ok(f64::MAX / 2.0), // penalty for invalid parameters
        }
    }
}

impl Gradient for SarimaxObjective {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, param: &Vec<f64>) -> std::result::Result<Vec<f64>, argmin::core::Error> {
        // Check cache (populated by a previous fused eval at same params)
        if let Some(ref cached) = *self.cache.borrow() {
            if cached.params == *param {
                return Ok(cached.gradient.clone());
            }
        }

        // Try fused eval: builds StateSpace once for both cost and gradient.
        // Cache the result so a subsequent cost() call at the same params is free.
        if let Ok((negll, grad)) = self.eval_negloglike_with_gradient(param) {
            if negll.is_finite() && grad.iter().all(|g| g.is_finite()) {
                *self.cache.borrow_mut() = Some(CachedEval {
                    params: param.clone(),
                    cost: negll,
                    gradient: grad.clone(),
                });
                return Ok(grad);
            }
        }

        // Fallback: analytical gradient only (no fused eval)
        if let Ok(grad) = self.analytical_gradient_negloglike(param) {
            if grad.iter().all(|g| g.is_finite()) {
                return Ok(grad);
            }
        }

        // Fallback: numerical forward-diff (n+1 KF evaluations)
        let n = param.len();
        let mut grad = vec![0.0; n];
        let eps = f64::EPSILON.sqrt();

        let f0 = self.cost(param)?;
        let mut p_work = param.clone();

        for i in 0..n {
            let orig = p_work[i];
            p_work[i] = orig + eps;
            let f_plus = self.cost(&p_work)?;
            p_work[i] = orig;

            grad[i] = (f_plus - f0) / eps;

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
    let solver = LBFGS::new(linesearch, 10) // memory=10 (scipy default)
        .with_tolerance_grad(1e-5) // match scipy pgtol default
        .map_err(|e| e.to_string())?
        .with_tolerance_cost(1e-9)
        .map_err(|e| e.to_string())?;

    let result = Executor::new(objective, solver)
        .configure(
            |state: argmin::core::IterState<Vec<f64>, Vec<f64>, (), (), (), f64>| {
                state.param(init_params).max_iters(maxiter)
            },
        )
        .run()
        .map_err(|e| format!("L-BFGS failed: {}", e))?;

    let state = result.state();
    let best_param = state
        .get_best_param()
        .ok_or("L-BFGS: no best parameter found")?
        .clone();
    let best_cost = state.get_best_cost();
    let n_iter = state.get_iter();
    let term_reason = state.get_termination_reason();
    let converged = term_reason == Some(&TerminationReason::SolverConverged)
        || term_reason == Some(&TerminationReason::TargetCostReached);

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
        .configure(
            |state: argmin::core::IterState<Vec<f64>, (), (), (), (), f64>| {
                state.max_iters(maxiter)
            },
        )
        .run()
        .map_err(|e| format!("Nelder-Mead failed: {}", e))?;

    let state = result.state();
    let best_param = state
        .get_best_param()
        .ok_or("Nelder-Mead: no best parameter found")?
        .clone();
    let best_cost = state.get_best_cost();
    let n_iter = state.get_iter();
    let term_reason = state.get_termination_reason();
    let converged = term_reason == Some(&TerminationReason::SolverConverged)
        || term_reason == Some(&TerminationReason::TargetCostReached);

    Ok((best_param, best_cost, n_iter, converged))
}

// ---------------------------------------------------------------------------
// L-BFGS-B optimization (box-constrained)
// ---------------------------------------------------------------------------

/// Compute box bounds for each parameter based on config.
///
/// Layout: `[trend | exog | ar(p) | ma(q) | sar(P) | sma(Q) | sigma2?]`
fn compute_bounds(config: &SarimaxConfig) -> Vec<(Option<f64>, Option<f64>)> {
    let kt = config.trend.k_trend();
    let n_exog = config.n_exog;
    let mut bounds = Vec::new();

    // trend + exog: unbounded
    for _ in 0..(kt + n_exog) {
        bounds.push((None, None));
    }

    // AR coefficients: unbounded when enforce_stationarity (Monahan/Jones transform
    // maps any real to stationary roots). This matches statsmodels which passes no
    // bounds to scipy L-BFGS-B when enforce_stationarity=True.
    for _ in 0..config.order.p {
        if config.enforce_stationarity {
            bounds.push((None, None));
        } else {
            bounds.push((Some(-0.999), Some(0.999)));
        }
    }

    // MA coefficients: unbounded when enforce_invertibility (same reasoning)
    for _ in 0..config.order.q {
        if config.enforce_invertibility {
            bounds.push((None, None));
        } else {
            bounds.push((Some(-0.999), Some(0.999)));
        }
    }

    // Seasonal AR: unbounded when enforce_stationarity
    for _ in 0..config.order.pp {
        if config.enforce_stationarity {
            bounds.push((None, None));
        } else {
            bounds.push((Some(-0.999), Some(0.999)));
        }
    }

    // Seasonal MA: unbounded when enforce_invertibility
    for _ in 0..config.order.qq {
        if config.enforce_invertibility {
            bounds.push((None, None));
        } else {
            bounds.push((Some(-0.999), Some(0.999)));
        }
    }

    // sigma2 (unconstrained space: exp/log transform, so any real maps to positive σ²)
    // Lower bound -50.0 maps to σ² ≈ 1.9e-22, preventing extreme values
    if !config.concentrate_scale {
        bounds.push((Some(-50.0), None));
    }

    bounds
}

fn run_lbfgsb(
    objective: &SarimaxObjective,
    init_params: Vec<f64>,
    bounds_vec: Vec<(Option<f64>, Option<f64>)>,
    maxiter: u64,
) -> std::result::Result<(Vec<f64>, f64, u64, bool), String> {
    let n = init_params.len();
    let obj = objective.clone();
    let eval_count = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let eval_count_inner = eval_count.clone();
    let hit_limit = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let hit_limit_inner = hit_limit.clone();

    let evaluate = move |x: &[f64], g: &mut [f64]| -> anyhow::Result<f64> {
        let count = eval_count_inner.load(std::sync::atomic::Ordering::Relaxed);

        // Enforce maxiter: the lbfgsb crate doesn't support it natively,
        // so we stop producing useful gradients after the limit is reached,
        // which causes the optimizer to terminate.
        if count >= maxiter {
            hit_limit_inner.store(true, std::sync::atomic::Ordering::Relaxed);
            for g_i in g.iter_mut() {
                *g_i = 0.0;
            }
            // Return current cost with zero gradient to trigger convergence
            return match obj.eval_negloglike(x) {
                Ok(c) if c.is_finite() => Ok(c),
                _ => Ok(f64::MAX / 2.0),
            };
        }
        eval_count_inner.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Fused function + gradient: builds StateSpace & KalmanInit only once
        if let Ok((cost, ag)) = obj.eval_negloglike_with_gradient(x) {
            if cost.is_finite() && ag.iter().all(|v| v.is_finite()) {
                g[..n].copy_from_slice(&ag);
                return Ok(cost);
            }
        }

        // Fallback: separate function + numerical gradient
        let cost = match obj.eval_negloglike(x) {
            Ok(c) if c.is_finite() => c,
            _ => {
                for g_i in g.iter_mut() {
                    *g_i = 0.0;
                }
                return Ok(f64::MAX / 2.0);
            }
        };

        let eps = f64::EPSILON.sqrt();
        let mut x_work = x.to_vec();
        for i in 0..n {
            let orig = x_work[i];
            x_work[i] = orig + eps;
            let f_plus = match obj.eval_negloglike(&x_work) {
                Ok(c) if c.is_finite() => c,
                _ => cost,
            };
            x_work[i] = orig;
            g[i] = (f_plus - cost) / eps;
            if !g[i].is_finite() {
                g[i] = 0.0;
            }
        }
        Ok(cost)
    };

    let param = lbfgsb::LbfgsbParameter {
        m: 10,       // memory size (scipy default: 10)
        factr: 1e7,  // cost tolerance: factr * eps_mach ≈ 1e-9 (scipy default)
        pgtol: 1e-5, // projected gradient tolerance (scipy default)
        iprint: -1,  // silent
    };

    let mut problem = lbfgsb::LbfgsbProblem::build(init_params, evaluate);
    problem.set_bounds(bounds_vec);

    let mut state = lbfgsb::LbfgsbState::new(problem, param);
    state
        .minimize()
        .map_err(|e| format!("L-BFGS-B failed: {}", e))?;

    let x = state.x().to_vec();
    let cost = state.fx();
    let n_eval = eval_count.load(std::sync::atomic::Ordering::Relaxed);

    // Determine convergence: minimize() returning Ok means the Fortran code
    // terminated normally (either via gradient tolerance pgtol OR function
    // value tolerance factr). If we hit our eval limit, report not converged.
    let converged = !hit_limit.load(std::sync::atomic::Ordering::Relaxed);

    Ok((x, cost, n_eval, converged))
}

fn consume_budget(remaining: &mut u64, total_work: &mut u64, n: u64) {
    let used = n.min(*remaining);
    *total_work = total_work.saturating_add(used);
    *remaining = remaining.saturating_sub(used);
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
/// * `method` — "lbfgsb" (default, single run), "lbfgsb-multi" (multi-start), "lbfgsb-strict", "lbfgs", or "nelder-mead"
/// * `maxiter` — Maximum iterations (default: 500)
/// * `exog` — Optional exogenous variables, column-major: exog[j][t]
pub fn fit(
    endog: &[f64],
    config: &SarimaxConfig,
    start_params: Option<&[f64]>,
    method: Option<&str>,
    maxiter: Option<u64>,
    exog: Option<&[Vec<f64>]>,
) -> Result<FitResult> {
    let maxiter = maxiter.unwrap_or(500);
    let method = method.unwrap_or("lbfgsb");
    let min_obs = expected_param_len(config).max(config.order.k_states().saturating_add(1));
    if endog.len() <= min_obs {
        return Err(SarimaxError::DataError(format!(
            "Not enough observations: n={} <= minimum required {} for model order",
            endog.len(),
            min_obs
        )));
    }

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
        None => compute_start_params(endog, config, exog)?,
    };

    // 1b. Pure AR fast path: for high-order non-seasonal pure AR models with
    //     concentrated scale, Burg AR coefficients are asymptotically
    //     MLE-equivalent. Skip the optimizer and return directly.
    //     Evidence: PERF_DIAGNOSIS.md Phase 2 shows AR(5) 4.5x and AR(8) 5.0x
    //     excess iterations — per-eval is 10-12x faster but wasted by iterations.
    //     Restrictions:
    //     - p >= 3 (AR(1)/AR(2) already fast; avoids n_iter=0 for simple models)
    //     - No MA (q=0, Q=0): MA estimation requires optimizer
    //     - No seasonal AR (P=0): seasonal AR interaction needs optimization
    //     - concentrate_scale=true: sigma2 concentrated out, no need to estimate
    //     - No user-provided start_params: trust our Burg estimates
    //     - No trend, no exog: these require optimization
    let is_pure_ar_fast = config.order.p >= 3
        && config.order.q == 0
        && config.order.qq == 0
        && config.order.pp == 0
        && config.trend.k_trend() == 0
        && config.n_exog == 0
        && config.concentrate_scale
        && start_params.is_none();

    if is_pure_ar_fast && method == "lbfgsb" {
        let test_params = SarimaxParams::from_flat(&constrained_start, config)?;
        let test_ss = StateSpace::new(config, &test_params, endog, exog)?;
        let test_init = KalmanInit::from_config(&test_ss, config, KalmanInit::default_kappa());
        let test_output = kalman_loglike(endog, &test_ss, &test_init, config.concentrate_scale)?;

        if test_output.loglike.is_finite() {
            let n_params = SarimaxParams::n_estimated_params(config);
            return Ok(FitResult {
                params: constrained_start,
                loglike: test_output.loglike,
                scale: test_output.scale,
                n_obs: endog.len(),
                n_params,
                n_iter: 0,
                converged: true,
                method: "burg-direct".to_string(),
                aic: 0.0,
                bic: 0.0,
            }
            .with_information_criteria());
        }
        // If loglike is not finite, fall through to optimizer
    }

    // 1c. Early return for maxiter=0: no optimization, return start params as-is
    if maxiter == 0 {
        let sp = SarimaxParams::from_flat(&constrained_start, config)?;
        let ss = StateSpace::new(config, &sp, endog, exog)?;
        let init = KalmanInit::from_config(&ss, config, KalmanInit::default_kappa());
        let output = kalman_loglike(endog, &ss, &init, config.concentrate_scale)?;
        let n_params = SarimaxParams::n_estimated_params(config);
        return Ok(FitResult {
            params: constrained_start,
            loglike: output.loglike,
            scale: output.scale,
            n_obs: endog.len(),
            n_params,
            n_iter: 0,
            converged: false,
            method: method.to_string(),
            aic: 0.0,
            bic: 0.0,
        }
        .with_information_criteria());
    }

    // 2. Transform to unconstrained space
    let unconstrained_start = untransform_params(&constrained_start, config)?;

    let objective = SarimaxObjective {
        endog: endog.to_vec(),
        config: config.clone(),
        exog: exog.map(|e| e.to_vec()),
        cache: RefCell::new(None),
    };

    // Determine number of restarts based on model complexity
    let n_params_total = unconstrained_start.len();
    let has_seasonal = config.order.pp > 0 || config.order.qq > 0;
    let n_restarts = if n_params_total >= 4 {
        3
    } else if n_params_total >= 3 || has_seasonal {
        2
    } else if n_params_total >= 2 {
        1
    } else {
        0
    };

    // 3. Run optimization
    let (best_unconstrained, _best_cost, n_iter, converged, used_method) = match method {
        "nelder-mead" | "nm" => {
            let (p, c, n, conv) = run_nelder_mead(objective.clone(), unconstrained_start, maxiter)
                .map_err(|e| SarimaxError::OptimizationFailed(e))?;
            (p, c, n, conv, "nelder-mead".to_string())
        }
        "lbfgsb-strict" | "lbfgsb_single" => {
            let bounds = compute_bounds(config);
            let (p, c, n, conv) = run_lbfgsb(&objective, unconstrained_start, bounds, maxiter)
                .map_err(|e| SarimaxError::OptimizationFailed(e))?;
            (p, c, n, conv, "lbfgsb-strict".to_string())
        }
        "lbfgsb" => {
            // Single L-BFGS-B run matching statsmodels/scipy behavior.
            // NM fallback only if L-BFGS-B fails entirely.
            let bounds = compute_bounds(config);
            match run_lbfgsb(&objective, unconstrained_start.clone(), bounds, maxiter) {
                Ok((p, c, n, conv)) => (p, c, n, conv, "lbfgsb".to_string()),
                Err(_) => {
                    // L-BFGS-B failed → fallback to Nelder-Mead
                    let (p, c, n, conv) =
                        run_nelder_mead(objective.clone(), unconstrained_start, maxiter)
                            .map_err(|e| SarimaxError::OptimizationFailed(e))?;
                    (p, c, n, conv, "nelder-mead (fallback)".to_string())
                }
            }
        }
        "lbfgsb-multi" => {
            // Multi-start L-BFGS-B with grid search and NM refinement.
            // More robust but slower — use when accuracy matters more than speed.
            let bounds = compute_bounds(config);
            let mut remaining = maxiter;
            let mut total_work: u64 = 0;

            let initial_result = match run_lbfgsb(
                &objective,
                unconstrained_start.clone(),
                bounds.clone(),
                remaining,
            ) {
                Ok((p, c, n, conv)) => {
                    consume_budget(&mut remaining, &mut total_work, n);
                    Some((p, c, conv, "lbfgsb-multi".to_string()))
                }
                Err(_) => None,
            };

            let mut best = initial_result;

            let mut try_update = |p: Vec<f64>, c: f64, conv: bool, method_name: &str| match &best {
                Some((_, best_cost, _, _)) if c < *best_cost => {
                    best = Some((p, c, conv, method_name.to_string()));
                }
                None => {
                    best = Some((p, c, conv, method_name.to_string()));
                }
                _ => {}
            };

            if n_restarts > 0 && remaining > 0 {
                // 1. Zero-start
                let zeros = vec![0.0; n_params_total];
                if let Ok((p, c, n, conv)) =
                    run_lbfgsb(&objective, zeros, bounds.clone(), remaining)
                {
                    consume_budget(&mut remaining, &mut total_work, n);
                    try_update(p, c, conv, "lbfgsb-multi");
                }

                // 2. Seasonal MA grid (NM, gradient-free for boundary avoidance)
                if config.enforce_invertibility && config.order.qq > 0 && remaining > 0 {
                    let kt = config.trend.k_trend();
                    let n_exog = config.n_exog;
                    let ma_start = kt + n_exog + config.order.p;
                    let sma_start = ma_start + config.order.q + config.order.pp;

                    let grid_vals = [-0.3, -0.6, -0.9];
                    for &ma_val in &grid_vals {
                        if remaining == 0 {
                            break;
                        }
                        let mut grid_constrained = vec![0.0; n_params_total];
                        for i in 0..config.order.q {
                            grid_constrained[ma_start + i] = ma_val;
                        }
                        for i in 0..config.order.qq {
                            grid_constrained[sma_start + i] = ma_val;
                        }
                        if let Ok(grid_uncons) = untransform_params(&grid_constrained, config) {
                            if let Ok((p, c, n, conv)) =
                                run_nelder_mead(objective.clone(), grid_uncons, remaining)
                            {
                                consume_budget(&mut remaining, &mut total_work, n);
                                try_update(p, c, conv, "lbfgsb-multi+nm");
                            }
                        }
                    }
                }

                // 3. LCG perturbations
                let mut rng_state: u64 = 12345;
                for _ in 0..n_restarts {
                    if remaining == 0 {
                        break;
                    }
                    let mut perturbed = unconstrained_start.clone();
                    for v in perturbed.iter_mut() {
                        rng_state = rng_state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        let u = ((rng_state >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
                        let scale = if v.abs() > 0.1 { v.abs() * 0.5 } else { 0.1 };
                        *v += u * scale;
                    }
                    if let Ok((p, c, n, conv)) =
                        run_lbfgsb(&objective, perturbed, bounds.clone(), remaining)
                    {
                        consume_budget(&mut remaining, &mut total_work, n);
                        try_update(p, c, conv, "lbfgsb-multi");
                    }
                }
            }

            match best {
                Some((best_p, best_c, best_conv, method_name)) => {
                    if n_params_total >= 2 && remaining > 0 {
                        match run_nelder_mead(objective.clone(), best_p.clone(), remaining) {
                            Ok((nm_p, nm_c, nm_n, nm_conv)) if nm_c < best_c => {
                                consume_budget(&mut remaining, &mut total_work, nm_n);
                                (
                                    nm_p,
                                    nm_c,
                                    total_work,
                                    nm_conv,
                                    format!("{}+nm", method_name),
                                )
                            }
                            Ok((_, _, nm_n, _)) => {
                                consume_budget(&mut remaining, &mut total_work, nm_n);
                                (best_p, best_c, total_work, best_conv, method_name)
                            }
                            Err(_) => (best_p, best_c, total_work, best_conv, method_name),
                        }
                    } else {
                        (best_p, best_c, total_work, best_conv, method_name)
                    }
                }
                None => {
                    let (p, c, n, conv) =
                        run_nelder_mead(objective.clone(), unconstrained_start, remaining)
                            .map_err(|e| SarimaxError::OptimizationFailed(e))?;
                    consume_budget(&mut remaining, &mut total_work, n);
                    (p, c, total_work, conv, "nelder-mead (fallback)".to_string())
                }
            }
        }
        "lbfgs" => {
            let mut remaining = maxiter;
            let mut total_work: u64 = 0;

            // Initial L-BFGS run
            let initial_result =
                match run_lbfgs(objective.clone(), unconstrained_start.clone(), remaining) {
                    Ok((p, c, n, conv)) => {
                        consume_budget(&mut remaining, &mut total_work, n);
                        Some((p, c, conv, "lbfgs".to_string()))
                    }
                    Err(_) => None,
                };

            // Multi-start: try perturbed starting points for complex models
            let mut best = initial_result;

            // Helper to update best with a new result
            let mut try_update = |p: Vec<f64>, c: f64, conv: bool| match &best {
                Some((_, best_cost, _, _)) if c < *best_cost => {
                    best = Some((p, c, conv, "lbfgs".to_string()));
                }
                None => {
                    best = Some((p, c, conv, "lbfgs".to_string()));
                }
                _ => {}
            };

            if n_restarts > 0 && remaining > 0 {
                // 1. Try starting from zeros in unconstrained space
                let zeros = vec![0.0; n_params_total];
                if let Ok((p, c, n, conv)) = run_lbfgs(objective.clone(), zeros, remaining) {
                    consume_budget(&mut remaining, &mut total_work, n);
                    try_update(p, c, conv);
                }

                // 2. For seasonal MA models with enforced invertibility, try Nelder-Mead from grid starts
                if config.enforce_invertibility && config.order.qq > 0 && remaining > 0 {
                    let kt = config.trend.k_trend();
                    let n_exog = config.n_exog;
                    let ma_start = kt + n_exog + config.order.p;
                    let sma_start = ma_start + config.order.q + config.order.pp;

                    // NM from grid of constrained MA/SMA starts (gradient-free avoids boundary traps)
                    let grid_vals = [-0.3, -0.6, -0.9];
                    for &ma_val in &grid_vals {
                        if remaining == 0 {
                            break;
                        }
                        let mut grid_constrained = vec![0.0; n_params_total];
                        for i in 0..config.order.q {
                            grid_constrained[ma_start + i] = ma_val;
                        }
                        for i in 0..config.order.qq {
                            grid_constrained[sma_start + i] = ma_val;
                        }
                        if let Ok(grid_uncons) = untransform_params(&grid_constrained, config) {
                            if let Ok((p, c, n, conv)) =
                                run_nelder_mead(objective.clone(), grid_uncons, remaining)
                            {
                                consume_budget(&mut remaining, &mut total_work, n);
                                try_update(p, c, conv);
                            }
                        }
                    }
                }

                // 3. Deterministic LCG perturbations
                let mut rng_state: u64 = 12345;
                for _ in 0..n_restarts {
                    if remaining == 0 {
                        break;
                    }
                    let mut perturbed = unconstrained_start.clone();
                    for v in perturbed.iter_mut() {
                        rng_state = rng_state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        let u = ((rng_state >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
                        let scale = if v.abs() > 0.1 { v.abs() * 0.5 } else { 0.1 };
                        *v += u * scale;
                    }
                    if let Ok((p, c, n, conv)) = run_lbfgs(objective.clone(), perturbed, remaining)
                    {
                        consume_budget(&mut remaining, &mut total_work, n);
                        try_update(p, c, conv);
                    }
                }
            }

            match best {
                Some((best_p, best_c, best_conv, _)) => {
                    // NM refinement for models with 2+ params
                    if n_params_total >= 2 && remaining > 0 {
                        match run_nelder_mead(objective.clone(), best_p.clone(), remaining) {
                            Ok((nm_p, nm_c, nm_n, nm_conv)) if nm_c < best_c => {
                                consume_budget(&mut remaining, &mut total_work, nm_n);
                                (nm_p, nm_c, total_work, nm_conv, "lbfgs+nm".to_string())
                            }
                            Ok((_, _, nm_n, _)) => {
                                consume_budget(&mut remaining, &mut total_work, nm_n);
                                (best_p, best_c, total_work, best_conv, "lbfgs".to_string())
                            }
                            Err(_) => (best_p, best_c, total_work, best_conv, "lbfgs".to_string()),
                        }
                    } else {
                        (best_p, best_c, total_work, best_conv, "lbfgs".to_string())
                    }
                }
                None => {
                    // All L-BFGS attempts failed, fallback to Nelder-Mead
                    let (p, c, n, conv) =
                        run_nelder_mead(objective.clone(), unconstrained_start, remaining)
                            .map_err(|e| SarimaxError::OptimizationFailed(e))?;
                    consume_budget(&mut remaining, &mut total_work, n);
                    (p, c, total_work, conv, "nelder-mead (fallback)".to_string())
                }
            }
        }
        _ => {
            return Err(SarimaxError::OptimizationFailed(format!(
                "unknown method: '{}'. Use 'lbfgsb', 'lbfgsb-multi', 'lbfgsb-strict', 'lbfgs', or 'nelder-mead'",
                method
            )));
        }
    };

    // 4. Transform back to constrained space
    let final_constrained = transform_params(&best_unconstrained, config)?;

    // 5. Evaluate final log-likelihood
    let final_params = SarimaxParams::from_flat(&final_constrained, config)?;
    let ss = StateSpace::new(config, &final_params, endog, exog)?;
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
        p: usize,
        d: usize,
        q: usize,
        enforce_stat: bool,
        enforce_inv: bool,
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
        let recovered = transform_params(&unconstrained, &config).unwrap();
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-10, "roundtrip failed: {} vs {}", a, b);
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
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        let config = make_config(1, 0, 0, false, false);
        let obj = SarimaxObjective {
            endog: data,
            config,
            exog: None,
            cache: RefCell::new(None),
        };

        let cost = obj.cost(&vec![0.5]).unwrap();
        assert!(cost.is_finite(), "cost should be finite: {}", cost);
    }

    #[test]
    fn test_gradient_finite() {
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        let config = make_config(1, 0, 0, false, false);
        let obj = SarimaxObjective {
            endog: data,
            config,
            exog: None,
            cache: RefCell::new(None),
        };

        let grad = obj.gradient(&vec![0.5]).unwrap();
        assert_eq!(grad.len(), 1);
        assert!(
            grad[0].is_finite(),
            "gradient should be finite: {}",
            grad[0]
        );
    }

    #[test]
    fn test_fit_ar1() {
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let expected_params: Vec<f64> = case["params"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let expected_loglike = case["loglike"].as_f64().unwrap();

        // Fixture was generated with approximate_diffuse init, so use enforce=false
        let config = make_config(1, 0, 0, false, false);
        let result = fit(&data, &config, None, Some("lbfgs"), Some(500), None).unwrap();

        assert!(result.converged, "AR(1) fit should converge");
        let param_err = (result.params[0] - expected_params[0]).abs();
        assert!(
            param_err < 1e-4,
            "AR(1) param error too large: {} (got {}, expected {})",
            param_err,
            result.params[0],
            expected_params[0]
        );
        let ll_err = (result.loglike - expected_loglike).abs();
        assert!(
            ll_err < 1e-2,
            "AR(1) loglike error: {} (got {}, expected {})",
            ll_err,
            result.loglike,
            expected_loglike
        );
    }

    #[test]
    fn test_fit_arma11() {
        let fixtures = load_fixtures();
        let case = &fixtures["arma11"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let expected_params: Vec<f64> = case["params"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        // Fixture was generated with approximate_diffuse init
        let config = make_config(1, 0, 1, false, false);
        let result = fit(&data, &config, None, Some("lbfgs"), Some(500), None).unwrap();

        for (i, (got, exp)) in result.params.iter().zip(expected_params.iter()).enumerate() {
            let err = (got - exp).abs();
            assert!(
                err < 1e-3,
                "ARMA(1,1) param[{}] error: {} (got {}, expected {})",
                i,
                err,
                got,
                exp
            );
        }
    }

    #[test]
    fn test_fit_arima111() {
        let fixtures = load_fixtures();
        let case = &fixtures["arima111"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let expected_loglike = case["loglike"].as_f64().unwrap();

        // Fixture was generated with approximate_diffuse init
        let config = make_config(1, 1, 1, false, false);
        let result = fit(&data, &config, None, Some("lbfgs"), Some(500), None).unwrap();

        let ll_err = (result.loglike - expected_loglike).abs();
        assert!(
            ll_err < 1.0,
            "ARIMA(1,1,1) loglike error: {} (got {}, expected {})",
            ll_err,
            result.loglike,
            expected_loglike
        );
    }

    #[test]
    fn test_fit_nelder_mead() {
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let expected_params: Vec<f64> = case["params"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        let config = make_config(1, 0, 0, false, false);
        let result = fit(&data, &config, None, Some("nelder-mead"), Some(1000), None).unwrap();

        let param_err = (result.params[0] - expected_params[0]).abs();
        assert!(
            param_err < 1e-3,
            "NM AR(1) param error: {} (got {}, expected {})",
            param_err,
            result.params[0],
            expected_params[0]
        );
    }

    #[test]
    fn test_aic_bic() {
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        let config = make_config(1, 0, 0, true, true);
        let result = fit(&data, &config, None, Some("lbfgs"), Some(500), None).unwrap();

        // AIC = -2*loglike + 2*k, BIC = -2*loglike + k*ln(n)
        let k = result.n_params as f64;
        let n = result.n_obs as f64;
        let expected_aic = -2.0 * result.loglike + 2.0 * k;
        let expected_bic = -2.0 * result.loglike + k * n.ln();

        assert!(
            (result.aic - expected_aic).abs() < 1e-10,
            "AIC mismatch: got {}, expected {}",
            result.aic,
            expected_aic
        );
        assert!(
            (result.bic - expected_bic).abs() < 1e-10,
            "BIC mismatch: got {}, expected {}",
            result.bic,
            expected_bic
        );
    }

    #[test]
    fn test_fit_with_custom_start_params() {
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        let config = make_config(1, 0, 0, false, false);
        let start = vec![0.5];
        let result = fit(&data, &config, Some(&start), Some("lbfgs"), Some(500), None).unwrap();

        assert!(result.loglike.is_finite());
        assert!(result.params[0].is_finite());
    }

    #[test]
    fn test_zero_maxiter_not_converged_for_lbfgs_and_nm() {
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        let config = make_config(1, 0, 0, false, false);

        let lbfgs = fit(&data, &config, None, Some("lbfgs"), Some(0), None).unwrap();
        assert_eq!(lbfgs.n_iter, 0, "lbfgs with maxiter=0 should not run");
        assert!(
            !lbfgs.converged,
            "lbfgs with maxiter=0 must report not converged"
        );

        let nm = fit(&data, &config, None, Some("nelder-mead"), Some(0), None).unwrap();
        assert_eq!(nm.n_iter, 0, "nelder-mead with maxiter=0 should not run");
        assert!(
            !nm.converged,
            "nelder-mead with maxiter=0 must report not converged"
        );
    }

    #[test]
    fn test_zero_maxiter_not_converged_for_lbfgsb_multi() {
        let fixtures = load_fixtures();
        let case = &fixtures["arma11"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        let config = make_config(1, 0, 1, false, false);
        let result = fit(&data, &config, None, Some("lbfgsb-multi"), Some(0), None).unwrap();

        assert_eq!(
            result.n_iter, 0,
            "lbfgsb-multi with maxiter=0 should not consume budget"
        );
        assert!(
            !result.converged,
            "lbfgsb-multi with maxiter=0 must report not converged"
        );
    }

    #[test]
    fn test_zero_maxiter_not_converged_for_lbfgsb_single() {
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        let config = make_config(1, 0, 0, false, false);
        let result = fit(&data, &config, None, Some("lbfgsb"), Some(0), None).unwrap();

        eprintln!(
            "lbfgsb single maxiter=0: n_iter={}, converged={}, method={}",
            result.n_iter, result.converged, result.method
        );
        assert_eq!(
            result.n_iter, 0,
            "lbfgsb with maxiter=0 should report n_iter=0, got {}",
            result.n_iter
        );
        assert!(
            !result.converged,
            "lbfgsb with maxiter=0 must report not converged"
        );
    }

    #[test]
    fn test_small_maxiter_lbfgs_not_converged() {
        // With maxiter=1, L-BFGS should NOT converge (not enough iterations)
        let fixtures = load_fixtures();
        let case = &fixtures["arma11"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        let config = make_config(1, 0, 1, false, false);
        let result = fit(&data, &config, None, Some("lbfgs"), Some(1), None).unwrap();
        eprintln!(
            "lbfgs maxiter=1: n_iter={}, converged={}, method={}",
            result.n_iter, result.converged, result.method
        );
        // With only 1 iteration, ARMA(1,1) cannot converge
        assert!(
            !result.converged,
            "lbfgs with maxiter=1 on ARMA(1,1) should not converge"
        );
    }

    #[test]
    fn test_small_maxiter_nm_not_converged() {
        // With maxiter=1, NM should NOT converge
        let fixtures = load_fixtures();
        let case = &fixtures["arma11"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        let config = make_config(1, 0, 1, false, false);
        let result = fit(&data, &config, None, Some("nelder-mead"), Some(1), None).unwrap();
        eprintln!(
            "nm maxiter=1: n_iter={}, converged={}, method={}",
            result.n_iter, result.converged, result.method
        );
        assert!(
            !result.converged,
            "nelder-mead with maxiter=1 on ARMA(1,1) should not converge"
        );
    }

    #[test]
    fn test_multistart_respects_global_maxiter_budget() {
        let fixtures = load_fixtures();
        let case = &fixtures["arma11"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        let config = make_config(1, 0, 1, false, false);
        let maxiter = 5_u64;

        let lbfgs = fit(&data, &config, None, Some("lbfgs"), Some(maxiter), None).unwrap();
        assert!(
            lbfgs.n_iter <= maxiter,
            "lbfgs n_iter={} exceeds maxiter={}",
            lbfgs.n_iter,
            maxiter
        );

        let lbfgsb_multi = fit(
            &data,
            &config,
            None,
            Some("lbfgsb-multi"),
            Some(maxiter),
            None,
        )
        .unwrap();
        assert!(
            lbfgsb_multi.n_iter <= maxiter,
            "lbfgsb-multi n_iter={} exceeds maxiter={}",
            lbfgsb_multi.n_iter,
            maxiter
        );
    }
}
