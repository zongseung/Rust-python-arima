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

use nalgebra::DMatrix;
use rayon::prelude::*;

use crate::css;
use crate::error::{Result, SarimaxError};
use crate::pipeline;
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

/// Direction of parameter transformation.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TransformDirection {
    /// Constrained → unconstrained (for optimization)
    ToUnconstrained,
    /// Unconstrained → constrained (recover model params)
    ToConstrained,
}

/// Shared implementation for both transform and untransform.
///
/// Layout: `[trend | exog | ar(p) | ma(q) | sar(P) | sma(Q) | sigma2?]`
///
/// The only difference between the two directions is which `params::` helper
/// is called for each AR/MA block and for variance.
fn transform_params_inner(
    input: &[f64],
    config: &SarimaxConfig,
    direction: TransformDirection,
) -> Result<Vec<f64>> {
    let expected = expected_param_len(config);
    if input.len() != expected {
        return Err(SarimaxError::ParamLengthMismatch {
            expected,
            got: input.len(),
        });
    }

    let kt = config.trend.k_trend();
    let n_exog = config.n_exog;
    let p = config.order.p;
    let q = config.order.q;
    let pp = config.order.pp;
    let qq = config.order.qq;

    let mut out = Vec::with_capacity(input.len());
    let mut i = 0;

    // Trend + exog: pass through
    out.extend_from_slice(&input[i..i + kt + n_exog]);
    i += kt + n_exog;

    // AR coefficients
    if config.enforce_stationarity && p > 0 {
        out.extend(match direction {
            TransformDirection::ToUnconstrained => params::unconstrain_stationary(&input[i..i + p]),
            TransformDirection::ToConstrained => params::constrain_stationary(&input[i..i + p]),
        });
    } else {
        out.extend_from_slice(&input[i..i + p]);
    }
    i += p;

    // MA coefficients
    if config.enforce_invertibility && q > 0 {
        out.extend(match direction {
            TransformDirection::ToUnconstrained => params::unconstrain_invertible(&input[i..i + q]),
            TransformDirection::ToConstrained => params::constrain_invertible(&input[i..i + q]),
        });
    } else {
        out.extend_from_slice(&input[i..i + q]);
    }
    i += q;

    // Seasonal AR
    if config.enforce_stationarity && pp > 0 {
        out.extend(match direction {
            TransformDirection::ToUnconstrained => {
                params::unconstrain_stationary(&input[i..i + pp])
            }
            TransformDirection::ToConstrained => params::constrain_stationary(&input[i..i + pp]),
        });
    } else {
        out.extend_from_slice(&input[i..i + pp]);
    }
    i += pp;

    // Seasonal MA
    if config.enforce_invertibility && qq > 0 {
        out.extend(match direction {
            TransformDirection::ToUnconstrained => {
                params::unconstrain_invertible(&input[i..i + qq])
            }
            TransformDirection::ToConstrained => params::constrain_invertible(&input[i..i + qq]),
        });
    } else {
        out.extend_from_slice(&input[i..i + qq]);
    }
    i += qq;

    // sigma2
    if !config.concentrate_scale && i < input.len() {
        out.push(match direction {
            TransformDirection::ToUnconstrained => params::unconstrain_variance(input[i])?,
            TransformDirection::ToConstrained => params::constrain_variance(input[i]),
        });
    }

    Ok(out)
}

/// Transform constrained parameters to unconstrained space for optimization.
///
/// Layout: `[trend | exog | ar(p) | ma(q) | sar(P) | sma(Q) | sigma2?]`
pub fn untransform_params(constrained: &[f64], config: &SarimaxConfig) -> Result<Vec<f64>> {
    transform_params_inner(constrained, config, TransformDirection::ToUnconstrained)
}

/// Transform unconstrained parameters back to constrained space.
pub fn transform_params(unconstrained: &[f64], config: &SarimaxConfig) -> Result<Vec<f64>> {
    transform_params_inner(unconstrained, config, TransformDirection::ToConstrained)
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
    /// Cached StateSpace: reused across optimizer iterations via in-place
    /// update_params() to avoid reallocating k×k matrices every evaluation.
    ss_cache: RefCell<Option<StateSpace>>,
}

impl Clone for SarimaxObjective {
    fn clone(&self) -> Self {
        SarimaxObjective {
            endog: self.endog.clone(),
            config: self.config.clone(),
            exog: self.exog.clone(),
            cache: RefCell::new(None), // cloned objectives start with empty cache
            ss_cache: RefCell::new(None), // cloned objectives build fresh SS
        }
    }
}

impl SarimaxObjective {
    /// Take cached StateSpace (updating in-place) or build a new one.
    fn take_or_build_ss(
        &self,
        sparams: &SarimaxParams,
    ) -> std::result::Result<StateSpace, String> {
        match self.ss_cache.borrow_mut().take() {
            Some(mut ss) => {
                ss.update_params(&self.config, sparams, &self.endog, self.exog.as_deref());
                Ok(ss)
            }
            None => StateSpace::new(&self.config, sparams, &self.endog, self.exog.as_deref())
                .map_err(|e| e.to_string()),
        }
    }

    /// Return StateSpace to cache for reuse by the next evaluation.
    fn return_ss(&self, ss: StateSpace) {
        *self.ss_cache.borrow_mut() = Some(ss);
    }

    /// Transform unconstrained params → constrained → SarimaxParams → StateSpace → KalmanInit.
    ///
    /// Common pipeline for eval_loglike, analytical_gradient_negloglike, and
    /// eval_negloglike_with_gradient. StateSpace is taken from cache when available.
    /// Caller MUST return the StateSpace to cache via `return_ss()`.
    fn eval_pipeline(
        &self,
        unconstrained: &[f64],
    ) -> std::result::Result<(SarimaxParams, StateSpace, KalmanInit), String> {
        let constrained =
            transform_params(unconstrained, &self.config).map_err(|e| e.to_string())?;
        let sparams =
            SarimaxParams::from_flat(&constrained, &self.config).map_err(|e| e.to_string())?;
        let ss = self.take_or_build_ss(&sparams)?;
        let init = KalmanInit::from_config_default(&ss, &self.config);
        Ok((sparams, ss, init))
    }

    /// Evaluate negative log-likelihood for given unconstrained parameters.
    /// Used by L-BFGS-B which minimizes directly.
    fn eval_negloglike(&self, unconstrained: &[f64]) -> std::result::Result<f64, String> {
        self.eval_loglike(unconstrained).map(|ll| -ll)
    }

    /// Evaluate log-likelihood for given unconstrained parameters.
    fn eval_loglike(&self, unconstrained: &[f64]) -> std::result::Result<f64, String> {
        let (_sparams, ss, init) = self.eval_pipeline(unconstrained)?;

        let result = kalman_loglike(&self.endog, &ss, &init, self.config.concentrate_scale);
        self.return_ss(ss);

        let output = result.map_err(|e| e.to_string())?;

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
        let (sparams, ss, init) = self.eval_pipeline(unconstrained)?;

        // Score in constrained space: ∂ll/∂θ_constrained
        let score_result = score::score(
            &self.endog,
            &ss,
            &init,
            &self.config,
            &sparams,
            self.config.concentrate_scale,
            self.exog.as_deref(),
        );
        self.return_ss(ss);

        let score_constrained = score_result.map_err(|e| e.to_string())?;

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
        let (sparams, ss, init) = self.eval_pipeline(unconstrained)?;

        // 1. Log-likelihood (forward KF)
        let kf_result = kalman_loglike(&self.endog, &ss, &init, self.config.concentrate_scale);

        let output = match kf_result {
            Ok(o) => o,
            Err(e) => {
                self.return_ss(ss);
                return Err(e.to_string());
            }
        };

        if !output.loglike.is_finite() {
            self.return_ss(ss);
            return Err("non-finite log-likelihood".to_string());
        }
        let negll = -output.loglike;

        // 2. Score (tangent linear KF, reuses ss and init)
        let score_result = score::score(
            &self.endog,
            &ss,
            &init,
            &self.config,
            &sparams,
            self.config.concentrate_scale,
            self.exog.as_deref(),
        );
        self.return_ss(ss);

        let score_constrained = score_result.map_err(|e| e.to_string())?;

        // 3. Chain rule: ∂(-ll)/∂u = -J' · ∂ll/∂θ
        let grad = apply_transform_jacobian(&score_constrained, unconstrained, &self.config)?;
        let neg_grad: Vec<f64> = grad.iter().map(|&g| -g).collect();

        Ok((negll, neg_grad))
    }
}

/// Apply the chain rule: grad_unconstrained = J' · grad_constrained.
///
/// Delegates to [`crate::params::transform_jacobian_t_vec`].
fn apply_transform_jacobian(
    score_constrained: &[f64],
    unconstrained: &[f64],
    config: &SarimaxConfig,
) -> std::result::Result<Vec<f64>, String> {
    crate::params::transform_jacobian_t_vec(unconstrained, config, score_constrained)
        .map_err(|e| e.to_string())
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
// CSS pre-optimization objective
// ---------------------------------------------------------------------------

/// CSS-based objective for pre-optimization.
///
/// CSS is O(n·(p'+q')) per evaluation vs O(n·k³) for KF.
/// Used as a pre-optimization step to find better MLE starting parameters.
/// Supports exogenous variables: subtracts X·β from differenced endog.
struct CssObjective {
    endog: Vec<f64>,
    config: SarimaxConfig,
    exog: Option<Vec<Vec<f64>>>,
}

impl CostFunction for CssObjective {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, unconstrained: &Vec<f64>) -> std::result::Result<f64, argmin::core::Error> {
        match transform_params(unconstrained, &self.config) {
            Ok(constrained) => match SarimaxParams::from_flat(&constrained, &self.config) {
                Ok(sparams) => {
                    let ll = css::css_loglike_with_exog(
                        &self.endog, &self.config, &sparams,
                        self.exog.as_deref(),
                    );
                    Ok(if ll.is_finite() { -ll } else { f64::MAX / 2.0 })
                }
                Err(_) => Ok(f64::MAX / 2.0),
            },
            Err(_) => Ok(f64::MAX / 2.0),
        }
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
// Multi-start helpers (shared by lbfgsb-multi and lbfgs methods)
// ---------------------------------------------------------------------------

/// Seasonal MA grid initialization via Nelder-Mead.
///
/// For models with enforced invertibility and seasonal MA terms, tries
/// NM optimization from a grid of constrained MA/SMA starting points.
/// Gradient-free NM avoids boundary traps near invertibility constraints.
///
/// Returns updated `(best, remaining, total_work)` via mutable references.
fn grid_ma_initialization(
    objective: &SarimaxObjective,
    config: &SarimaxConfig,
    n_params_total: usize,
    remaining: &mut u64,
    total_work: &mut u64,
    best: &mut Option<(Vec<f64>, f64, bool, String)>,
    method_label: &str,
) {
    if !config.enforce_invertibility || config.order.qq == 0 || *remaining == 0 {
        return;
    }

    let kt = config.trend.k_trend();
    let n_exog = config.n_exog;
    let ma_start = kt + n_exog + config.order.p;
    let sma_start = ma_start + config.order.q + config.order.pp;

    let grid_vals = [-0.3, -0.6, -0.9];
    for &ma_val in &grid_vals {
        if *remaining == 0 {
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
                run_nelder_mead(objective.clone(), grid_uncons, *remaining)
            {
                consume_budget(remaining, total_work, n);
                try_update_best(best, p, c, conv, method_label);
            }
        }
    }
}

/// Generate LCG-perturbed starting points and optimize each.
///
/// Uses a deterministic linear congruential generator (LCG) seeded at 12345
/// for reproducible multi-start perturbations of the unconstrained start.
fn lcg_perturbed_starts(
    unconstrained_start: &[f64],
    n_restarts: usize,
    remaining: &u64,
) -> Vec<Vec<f64>> {
    let mut starts = Vec::with_capacity(n_restarts);
    let mut rng_state: u64 = 12345;

    for _ in 0..n_restarts {
        if *remaining == 0 {
            break;
        }
        let mut perturbed = unconstrained_start.to_vec();
        for v in perturbed.iter_mut() {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((rng_state >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
            let scale = if v.abs() > 0.1 { v.abs() * 0.5 } else { 0.1 };
            *v += u * scale;
        }
        starts.push(perturbed);
    }
    starts
}

/// Update `best` if `(p, c)` is better (lower cost) than the current best.
fn try_update_best(
    best: &mut Option<(Vec<f64>, f64, bool, String)>,
    p: Vec<f64>,
    c: f64,
    conv: bool,
    method_name: &str,
) {
    match best {
        Some((_, best_cost, _, _)) if c < *best_cost => {
            *best = Some((p, c, conv, method_name.to_string()));
        }
        None => {
            *best = Some((p, c, conv, method_name.to_string()));
        }
        _ => {}
    }
}

/// NM refinement: try polishing the best solution with Nelder-Mead.
///
/// Returns the final `(params, cost, total_work, converged, method_name)` tuple.
fn nm_refinement(
    objective: &SarimaxObjective,
    best_p: Vec<f64>,
    best_c: f64,
    best_conv: bool,
    method_name: String,
    n_params_total: usize,
    remaining: &mut u64,
    total_work: &mut u64,
) -> (Vec<f64>, f64, u64, bool, String) {
    if n_params_total >= 2 && *remaining > 0 {
        match run_nelder_mead(objective.clone(), best_p.clone(), *remaining) {
            Ok((nm_p, nm_c, nm_n, nm_conv)) if nm_c < best_c => {
                consume_budget(remaining, total_work, nm_n);
                (nm_p, nm_c, *total_work, nm_conv, format!("{}+nm", method_name))
            }
            Ok((_, _, nm_n, _)) => {
                consume_budget(remaining, total_work, nm_n);
                (best_p, best_c, *total_work, best_conv, method_name)
            }
            Err(_) => (best_p, best_c, *total_work, best_conv, method_name),
        }
    } else {
        (best_p, best_c, *total_work, best_conv, method_name)
    }
}

// ---------------------------------------------------------------------------
// Method-specific dispatch functions
// ---------------------------------------------------------------------------

/// Determine number of multi-start restarts based on model complexity.
fn compute_n_restarts(n_params_total: usize, config: &SarimaxConfig) -> usize {
    let has_seasonal = config.order.pp > 0 || config.order.qq > 0;
    if n_params_total >= 4 {
        3
    } else if n_params_total >= 3 || has_seasonal {
        2
    } else if n_params_total >= 2 {
        1
    } else {
        0
    }
}

/// Generic multi-start optimizer with grid search and NM refinement.
///
/// Implements the shared orchestration logic for both L-BFGS-B and L-BFGS
/// multi-start strategies. The `runner` closure encapsulates the specific
/// optimizer (strategy pattern), taking `(start_params, maxiter)` and
/// returning `(best_params, cost, n_evals, converged)`.
fn fit_multistart<F>(
    objective: &SarimaxObjective,
    unconstrained_start: &[f64],
    config: &SarimaxConfig,
    maxiter: u64,
    n_restarts: usize,
    method_label: &str,
    runner: F,
) -> std::result::Result<(Vec<f64>, f64, u64, bool, String), SarimaxError>
where
    F: Fn(Vec<f64>, u64) -> std::result::Result<(Vec<f64>, f64, u64, bool), String>,
{
    let n_params_total = unconstrained_start.len();
    let mut remaining = maxiter;
    let mut total_work: u64 = 0;

    // Initial optimizer run
    let mut best: Option<(Vec<f64>, f64, bool, String)> =
        match runner(unconstrained_start.to_vec(), remaining) {
            Ok((p, c, n, conv)) => {
                consume_budget(&mut remaining, &mut total_work, n);
                Some((p, c, conv, method_label.to_string()))
            }
            Err(_) => None,
        };

    if n_restarts > 0 && remaining > 0 {
        // 1. Zero-start
        let zeros = vec![0.0; n_params_total];
        if let Ok((p, c, n, conv)) = runner(zeros, remaining) {
            consume_budget(&mut remaining, &mut total_work, n);
            try_update_best(&mut best, p, c, conv, method_label);
        }

        // 2. Seasonal MA grid (NM, gradient-free for boundary avoidance)
        let grid_label = if method_label.contains("lbfgsb") {
            format!("{}+nm", method_label)
        } else {
            method_label.to_string()
        };
        grid_ma_initialization(
            objective,
            config,
            n_params_total,
            &mut remaining,
            &mut total_work,
            &mut best,
            &grid_label,
        );

        // 3. LCG perturbations (with P6 near-cancellation filter, α=0.01)
        let perturbations =
            lcg_perturbed_starts(unconstrained_start, n_restarts, &remaining);
        for perturbed in perturbations {
            if remaining == 0 {
                break;
            }
            if !passes_cancellation_filter(&perturbed, config) {
                continue; // reject near-cancellation start point
            }
            if let Ok((p, c, n, conv)) = runner(perturbed, remaining) {
                consume_budget(&mut remaining, &mut total_work, n);
                try_update_best(&mut best, p, c, conv, method_label);
            }
        }
    }

    match best {
        Some((best_p, best_c, best_conv, method_name)) => Ok(nm_refinement(
            objective,
            best_p,
            best_c,
            best_conv,
            method_name,
            n_params_total,
            &mut remaining,
            &mut total_work,
        )),
        None => {
            // All optimizer attempts failed, fallback to Nelder-Mead
            let (p, c, n, conv) =
                run_nelder_mead(objective.clone(), unconstrained_start.to_vec(), remaining)
                    .map_err(|e| SarimaxError::OptimizationFailed(e))?;
            consume_budget(&mut remaining, &mut total_work, n);
            Ok((p, c, total_work, conv, "nelder-mead (fallback)".to_string()))
        }
    }
}

/// Multi-start L-BFGS-B with grid search, parallel restarts, and NM refinement.
///
/// Uses rayon to parallelize LCG perturbation restarts. Each thread gets its
/// own SarimaxObjective with independent StateSpace cache.
fn fit_lbfgsb_multi(
    objective: &SarimaxObjective,
    unconstrained_start: &[f64],
    config: &SarimaxConfig,
    maxiter: u64,
    n_restarts: usize,
) -> std::result::Result<(Vec<f64>, f64, u64, bool, String), SarimaxError> {
    let bounds = compute_bounds(config);
    let n_params_total = unconstrained_start.len();
    let mut remaining = maxiter;
    let mut total_work: u64 = 0;

    // Initial run (sequential)
    let mut best: Option<(Vec<f64>, f64, bool, String)> =
        match run_lbfgsb(objective, unconstrained_start.to_vec(), bounds.clone(), remaining) {
            Ok((p, c, n, conv)) => {
                consume_budget(&mut remaining, &mut total_work, n);
                Some((p, c, conv, "lbfgsb-multi".to_string()))
            }
            Err(_) => None,
        };

    if n_restarts > 0 && remaining > 0 {
        // 1. Zero-start (sequential)
        let zeros = vec![0.0; n_params_total];
        if let Ok((p, c, n, conv)) = run_lbfgsb(objective, zeros, bounds.clone(), remaining) {
            consume_budget(&mut remaining, &mut total_work, n);
            try_update_best(&mut best, p, c, conv, "lbfgsb-multi");
        }

        // 2. Grid MA initialization (sequential NM, gradient-free)
        grid_ma_initialization(
            objective,
            config,
            n_params_total,
            &mut remaining,
            &mut total_work,
            &mut best,
            "lbfgsb-multi+nm",
        );

        // 3. LCG perturbations — PARALLEL via rayon (with P6 near-cancellation filter, α=0.01)
        let perturbations: Vec<_> = lcg_perturbed_starts(unconstrained_start, n_restarts, &remaining)
            .into_iter()
            .filter(|start| passes_cancellation_filter(start, config))
            .collect();
        if perturbations.len() >= 2 && remaining > 0 {
            let per_start_budget = remaining / perturbations.len() as u64;
            if per_start_budget > 0 {
                // Pre-clone data for thread-safe parallel execution.
                // Each rayon thread gets its own SarimaxObjective (with fresh caches).
                let endog_shared = &objective.endog;
                let config_shared = &objective.config;
                let exog_shared = &objective.exog;
                let bounds_shared = &bounds;

                let results: Vec<_> = perturbations
                    .into_par_iter()
                    .filter_map(|start| {
                        let obj = SarimaxObjective {
                            endog: endog_shared.clone(),
                            config: config_shared.clone(),
                            exog: exog_shared.clone(),
                            cache: RefCell::new(None),
                            ss_cache: RefCell::new(None),
                        };
                        run_lbfgsb(&obj, start, bounds_shared.clone(), per_start_budget).ok()
                    })
                    .collect();

                let par_work: u64 = results.iter().map(|(_, _, n, _)| n).sum();
                total_work = total_work.saturating_add(par_work);
                remaining = remaining.saturating_sub(par_work);

                for (p, c, _, conv) in results {
                    try_update_best(&mut best, p, c, conv, "lbfgsb-multi");
                }
            }
        } else {
            // Single perturbation or no budget: run sequentially
            for perturbed in perturbations {
                if remaining == 0 {
                    break;
                }
                if let Ok((p, c, n, conv)) =
                    run_lbfgsb(objective, perturbed, bounds.clone(), remaining)
                {
                    consume_budget(&mut remaining, &mut total_work, n);
                    try_update_best(&mut best, p, c, conv, "lbfgsb-multi");
                }
            }
        }
    }

    // NM refinement
    match best {
        Some((best_p, best_c, best_conv, method_name)) => Ok(nm_refinement(
            objective,
            best_p,
            best_c,
            best_conv,
            method_name,
            n_params_total,
            &mut remaining,
            &mut total_work,
        )),
        None => {
            let (p, c, n, conv) =
                run_nelder_mead(objective.clone(), unconstrained_start.to_vec(), remaining)
                    .map_err(|e| SarimaxError::OptimizationFailed(e))?;
            consume_budget(&mut remaining, &mut total_work, n);
            Ok((p, c, total_work, conv, "nelder-mead (fallback)".to_string()))
        }
    }
}

/// Multi-start L-BFGS (argmin) with grid search and NM refinement.
fn fit_lbfgs_argmin(
    objective: &SarimaxObjective,
    unconstrained_start: &[f64],
    config: &SarimaxConfig,
    maxiter: u64,
    n_restarts: usize,
) -> std::result::Result<(Vec<f64>, f64, u64, bool, String), SarimaxError> {
    let obj = objective.clone();
    fit_multistart(
        objective,
        unconstrained_start,
        config,
        maxiter,
        n_restarts,
        "lbfgs",
        |start, budget| run_lbfgs(obj.clone(), start, budget),
    )
}

// ---------------------------------------------------------------------------
// Fast paths (pure AR, maxiter=0)
// ---------------------------------------------------------------------------

/// Try pure AR fast path: for high-order non-seasonal pure AR models with
/// concentrated scale, Burg AR coefficients are asymptotically MLE-equivalent.
fn try_ar_fast_path(
    endog: &[f64],
    config: &SarimaxConfig,
    constrained_start: &[f64],
    exog: Option<&[Vec<f64>]>,
    method: &str,
    start_params_provided: bool,
) -> Result<Option<FitResult>> {
    let is_pure_ar_fast = config.order.p >= 3
        && config.order.q == 0
        && config.order.qq == 0
        && config.order.pp == 0
        && config.trend.k_trend() == 0
        && config.n_exog == 0
        && config.concentrate_scale
        && !start_params_provided;

    if !is_pure_ar_fast || method != "lbfgsb" {
        return Ok(None);
    }

    let test_params = SarimaxParams::from_flat(constrained_start, config)?;
    let test_ss = StateSpace::new(config, &test_params, endog, exog)?;
    let test_init = KalmanInit::from_config_default(&test_ss, config);
    let test_output = kalman_loglike(endog, &test_ss, &test_init, config.concentrate_scale)?;

    if test_output.loglike.is_finite() {
        let n_params = SarimaxParams::n_estimated_params(config);
        Ok(Some(
            FitResult {
                params: constrained_start.to_vec(),
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
            .with_information_criteria(),
        ))
    } else {
        Ok(None)
    }
}

/// Build result for maxiter=0: no optimization, return start params as-is.
fn build_zero_iter_result(
    endog: &[f64],
    config: &SarimaxConfig,
    constrained_start: &[f64],
    exog: Option<&[Vec<f64>]>,
    method: &str,
) -> Result<FitResult> {
    let sp = SarimaxParams::from_flat(constrained_start, config)?;
    let ss = StateSpace::new(config, &sp, endog, exog)?;
    let init = KalmanInit::from_config_default(&ss, config);
    let output = kalman_loglike(endog, &ss, &init, config.concentrate_scale)?;
    let n_params = SarimaxParams::n_estimated_params(config);
    Ok(FitResult {
        params: constrained_start.to_vec(),
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
    .with_information_criteria())
}

/// Build final FitResult from optimized unconstrained parameters.
fn build_fit_result(
    endog: &[f64],
    config: &SarimaxConfig,
    best_unconstrained: &[f64],
    n_iter: u64,
    converged: bool,
    used_method: String,
    exog: Option<&[Vec<f64>]>,
) -> Result<FitResult> {
    let final_constrained = transform_params(best_unconstrained, config)?;
    let final_params = SarimaxParams::from_flat(&final_constrained, config)?;

    // A-2: Warn if final result has near-cancellation (α=0.05, VER5.2 P6)
    if !validate_no_near_cancellation(&final_params, config, 0.05) {
        eprintln!(
            "[sarimax_rs] WARNING: near-cancellation detected in fitted ARMA({},{}) \
             parameters (min inverted-root distance < 0.05). \
             Model may be non-identifiable.",
            config.order.p, config.order.q
        );
    }

    let ss = StateSpace::new(config, &final_params, endog, exog)?;
    let init = KalmanInit::from_config_default(&ss, config);
    let output = kalman_loglike(endog, &ss, &init, config.concentrate_scale)?;
    let n_params = SarimaxParams::n_estimated_params(config);

    Ok(FitResult {
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
    .with_information_criteria())
}

// ---------------------------------------------------------------------------
// Validation + start param helpers
// ---------------------------------------------------------------------------

/// Validate observations length and resolve starting parameters.
///
/// Returns constrained start parameters, either from the caller or computed
/// via Hannan-Rissanen / Burg initialization.
fn validate_and_get_start_params(
    endog: &[f64],
    config: &SarimaxConfig,
    start_params: Option<&[f64]>,
    exog: Option<&[Vec<f64>]>,
) -> Result<Vec<f64>> {
    let min_obs = expected_param_len(config).max(config.order.k_states().saturating_add(1));
    if endog.len() <= min_obs {
        return Err(SarimaxError::DataError(format!(
            "Not enough observations: n={} <= minimum required {} for model order",
            endog.len(),
            min_obs
        )));
    }

    match start_params {
        Some(sp) => {
            let expected_len = expected_param_len(config);
            if sp.len() != expected_len {
                return Err(SarimaxError::ParamLengthMismatch {
                    expected: expected_len,
                    got: sp.len(),
                });
            }
            Ok(sp.to_vec())
        }
        None => compute_start_params(endog, config, exog),
    }
}

/// Single-run L-BFGS-B with Nelder-Mead fallback.
fn fit_lbfgsb_single(
    objective: &SarimaxObjective,
    unconstrained_start: Vec<f64>,
    config: &SarimaxConfig,
    maxiter: u64,
) -> std::result::Result<(Vec<f64>, f64, u64, bool, String), SarimaxError> {
    let bounds = compute_bounds(config);
    match run_lbfgsb(objective, unconstrained_start.clone(), bounds, maxiter) {
        Ok((p, c, n, conv)) => Ok((p, c, n, conv, "lbfgsb".to_string())),
        Err(_) => {
            let (p, c, n, conv) =
                run_nelder_mead(objective.clone(), unconstrained_start, maxiter)
                    .map_err(|e| SarimaxError::OptimizationFailed(e))?;
            Ok((p, c, n, conv, "nelder-mead (fallback)".to_string()))
        }
    }
}

// ---------------------------------------------------------------------------
// CSS pre-optimization helpers
// ---------------------------------------------------------------------------

/// Evaluate KF log-likelihood at constrained parameters. Returns NEG_INFINITY on error.
fn eval_kf_loglike_constrained(
    endog: &[f64],
    config: &SarimaxConfig,
    constrained: &[f64],
    exog: Option<&[Vec<f64>]>,
) -> f64 {
    match pipeline::kalman_eval_constrained(endog, constrained, config, exog) {
        Ok(out) if out.loglike.is_finite() => out.loglike,
        _ => f64::NEG_INFINITY,
    }
}

// ---------------------------------------------------------------------------
// A-2: Near-cancellation detection (VER5.2 P6)
// ---------------------------------------------------------------------------

/// Compute inverted roots of an AR or MA polynomial via companion matrix eigenvalues.
///
/// `coeffs` = [φ₁, φ₂, …, φ_p] (AR) or [θ₁, θ₂, …, θ_q] (MA).
/// Returns eigenvalues of the companion matrix as (real, imag) pairs.
/// These are the inverted polynomial roots (= 1 / z_i).
///
/// Uses nalgebra real Schur decomposition to handle complex conjugate pairs.
/// Matching arima2 convention: `inv_ar_roots = 1/polyroot(c(1, -ar_pars))`.
fn polynomial_roots(coeffs: &[f64]) -> Vec<(f64, f64)> {
    let p = coeffs.len();
    if p == 0 {
        return vec![];
    }
    if p == 1 {
        return vec![(coeffs[0], 0.0)];
    }

    // Companion matrix: first row = coefficients, sub-diagonal = 1
    let mut companion = DMatrix::zeros(p, p);
    for i in 0..p {
        companion[(0, i)] = coeffs[i];
    }
    for i in 1..p {
        companion[(i, i - 1)] = 1.0;
    }

    // Real Schur decomposition → quasi-upper triangular T
    let schur = nalgebra::Schur::new(companion);
    let (_, t) = schur.unpack();

    // Extract eigenvalues from diagonal blocks of T
    let mut eigenvalues = Vec::with_capacity(p);
    let mut k = 0_usize;
    while k < p {
        if k + 1 < p && t[(k + 1, k)].abs() > 1e-12 {
            // 2×2 block → complex conjugate pair
            let a = t[(k, k)];
            let b = t[(k, k + 1)];
            let c = t[(k + 1, k)];
            let d = t[(k + 1, k + 1)];
            let center = (a + d) / 2.0;
            let disc = (a - d) * (a - d) / 4.0 + b * c;
            // disc < 0 for genuine complex eigenvalues; clamp to 0 for safety
            let imag = (-disc).max(0.0).sqrt();
            eigenvalues.push((center, imag));
            eigenvalues.push((center, -imag));
            k += 2;
        } else {
            eigenvalues.push((t[(k, k)], 0.0));
            k += 1;
        }
    }
    eigenvalues
}

/// Minimum distance between AR and MA inverted roots.
///
/// Near-cancellation: AR and MA polynomials share a common root → the ARMA
/// representation is non-identifiable (Fisher information matrix singular).
///
/// Threshold guidance (arima2 convention):
///   α = 0.01 — filter random restart starting points
///   α = 0.05 — warn about final parameter estimates
fn min_root_distance(ar_coeffs: &[f64], ma_coeffs: &[f64]) -> f64 {
    let ar_roots = polynomial_roots(ar_coeffs);
    let ma_roots = polynomial_roots(ma_coeffs);

    let mut min_dist = f64::INFINITY;
    for &(ar_re, ar_im) in &ar_roots {
        for &(ma_re, ma_im) in &ma_roots {
            let dist = ((ar_re - ma_re).powi(2) + (ar_im - ma_im).powi(2)).sqrt();
            min_dist = min_dist.min(dist);
        }
    }
    min_dist
}

/// Returns true when ARMA parameters have no near-cancellation (dist > threshold).
///
/// AR-only or MA-only models always return true (no cancellation possible).
/// Spec (VER5.2 P6): threshold=0.01 for restart filtering, 0.05 for warnings.
fn validate_no_near_cancellation(
    sparams: &SarimaxParams,
    config: &SarimaxConfig,
    threshold: f64,
) -> bool {
    if config.order.p == 0 || config.order.q == 0 {
        return true;
    }
    min_root_distance(&sparams.ar_coeffs, &sparams.ma_coeffs) > threshold
}

/// Check whether unconstrained params exhibit near-cancellation.
///
/// Returns true if the params pass (no near-cancellation), false if they
/// should be rejected. Used to filter random restart starting points (α=0.01).
fn passes_cancellation_filter(unconstrained: &[f64], config: &SarimaxConfig) -> bool {
    // Skip check for AR-only or MA-only models
    if config.order.p == 0 || config.order.q == 0 {
        return true;
    }
    let constrained = match transform_params(unconstrained, config) {
        Ok(c) => c,
        Err(_) => return false, // invalid params → reject
    };
    let sparams = match SarimaxParams::from_flat(&constrained, config) {
        Ok(p) => p,
        Err(_) => return false,
    };
    validate_no_near_cancellation(&sparams, config, 0.01)
}

/// Run CSS pre-optimization: L-BFGS-B first, NM fallback (R-style CSS-ML).
///
/// CSS is O(n·(p'+q')) per evaluation, ~100x faster than KF for large k.
/// L-BFGS-B with finite-difference gradient on CSS matches R's optim(BFGS)
/// approach for the CSS stage.
/// Returns updated constrained parameters if optimization succeeds.
fn run_css_optimization(
    endog: &[f64],
    config: &SarimaxConfig,
    constrained_start: &[f64],
    maxiter: u64,
    exog: Option<&[Vec<f64>]>,
) -> Option<Vec<f64>> {
    let unconstrained_start = untransform_params(constrained_start, config).ok()?;
    let n = unconstrained_start.len();
    if n == 0 {
        return None;
    }

    // 1st: L-BFGS-B on CSS (R-style gradient-based optimization)
    if let Some(result) = run_lbfgsb_css(endog, config, &unconstrained_start, maxiter, exog) {
        if let Ok(constrained) = transform_params(&result, config) {
            return Some(constrained);
        }
    }

    // 2nd: NM fallback (gradient-free, more robust for ill-conditioned cases)
    run_nm_css(endog, config, &unconstrained_start, maxiter, exog)
}

/// L-BFGS-B optimization on CSS objective with finite-difference gradient.
fn run_lbfgsb_css(
    endog: &[f64],
    config: &SarimaxConfig,
    unconstrained_start: &[f64],
    maxiter: u64,
    exog: Option<&[Vec<f64>]>,
) -> Option<Vec<f64>> {
    let n = unconstrained_start.len();
    let endog_owned = endog.to_vec();
    let config_owned = config.clone();
    let exog_owned: Option<Vec<Vec<f64>>> = exog.map(|e| e.to_vec());
    let eval_count = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let eval_count_inner = eval_count.clone();
    let hit_limit = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let hit_limit_inner = hit_limit.clone();

    let css_eval = |unconstrained: &[f64]| -> f64 {
        match transform_params(&unconstrained.to_vec(), &config_owned) {
            Ok(constrained) => match SarimaxParams::from_flat(&constrained, &config_owned) {
                Ok(sparams) => {
                    let ll = css::css_loglike_with_exog(
                        &endog_owned, &config_owned, &sparams,
                        exog_owned.as_deref(),
                    );
                    if ll.is_finite() { -ll } else { f64::MAX / 2.0 }
                }
                Err(_) => f64::MAX / 2.0,
            },
            Err(_) => f64::MAX / 2.0,
        }
    };

    let evaluate = move |x: &[f64], g: &mut [f64]| -> anyhow::Result<f64> {
        let count = eval_count_inner.load(std::sync::atomic::Ordering::Relaxed);
        if count >= maxiter {
            hit_limit_inner.store(true, std::sync::atomic::Ordering::Relaxed);
            for g_i in g.iter_mut() {
                *g_i = 0.0;
            }
            return Ok(css_eval(x));
        }
        eval_count_inner.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let cost = css_eval(x);
        if !cost.is_finite() || cost >= f64::MAX / 4.0 {
            for g_i in g.iter_mut() {
                *g_i = 0.0;
            }
            return Ok(f64::MAX / 2.0);
        }

        // Central finite-difference gradient (eps=1e-7)
        let eps = 1e-7;
        let mut x_work = x.to_vec();
        for i in 0..n {
            let orig = x_work[i];
            x_work[i] = orig + eps;
            let f_fwd = css_eval(&x_work);
            x_work[i] = orig - eps;
            let f_bwd = css_eval(&x_work);
            x_work[i] = orig;
            g[i] = (f_fwd - f_bwd) / (2.0 * eps);
            if !g[i].is_finite() {
                g[i] = 0.0;
            }
        }
        Ok(cost)
    };

    // Box bounds: [-10, 10] for all params (unconstrained space)
    let bounds_vec: Vec<(Option<f64>, Option<f64>)> =
        vec![(Some(-10.0), Some(10.0)); n];

    let param = lbfgsb::LbfgsbParameter {
        m: 10,
        factr: 1e7,
        pgtol: 1e-5,
        iprint: -1,
    };

    let mut problem = lbfgsb::LbfgsbProblem::build(unconstrained_start.to_vec(), evaluate);
    problem.set_bounds(bounds_vec);

    let mut state = lbfgsb::LbfgsbState::new(problem, param);
    state.minimize().ok()?;

    Some(state.x().to_vec())
}

/// NM fallback for CSS optimization.
fn run_nm_css(
    endog: &[f64],
    config: &SarimaxConfig,
    unconstrained_start: &[f64],
    maxiter: u64,
    exog: Option<&[Vec<f64>]>,
) -> Option<Vec<f64>> {
    let n = unconstrained_start.len();
    let obj = CssObjective {
        endog: endog.to_vec(),
        config: config.clone(),
        exog: exog.map(|e| e.to_vec()),
    };

    // Build simplex: initial point + n perturbations
    let mut simplex = vec![unconstrained_start.to_vec()];
    for i in 0..n {
        let mut vertex = unconstrained_start.to_vec();
        let delta = if vertex[i].abs() > 0.1 {
            vertex[i] * 0.05
        } else {
            0.025
        };
        vertex[i] += delta;
        simplex.push(vertex);
    }

    let solver = NelderMead::new(simplex)
        .with_sd_tolerance(1e-4)
        .ok()?;

    let result = Executor::new(obj, solver)
        .configure(
            |state: argmin::core::IterState<Vec<f64>, (), (), (), (), f64>| {
                state.max_iters(maxiter)
            },
        )
        .run()
        .ok()?;

    let best_unconstrained = result.state().get_best_param()?.clone();
    transform_params(&best_unconstrained, config).ok()
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

    // simple_differencing: pre-difference the data so the Kalman filter sees
    // a stationary ARMA-only series. n_obs = eff_endog.len() for AIC/BIC.
    let (eff_endog, eff_exog_owned) = pipeline::prepare_endog(endog, config, exog);
    let eff_endog: &[f64] = &eff_endog;
    let eff_exog: Option<&[Vec<f64>]> = if config.simple_differencing {
        eff_exog_owned.as_deref()
    } else {
        exog
    };

    // 1. Validate + compute start params (always use original endog: CSS init
    //    applies differencing internally via apply_differencing)
    let mut constrained_start = validate_and_get_start_params(endog, config, start_params, exog)?;

    // 2. Fast paths (AR-only, maxiter=0) — use eff_endog
    if let Some(r) = try_ar_fast_path(eff_endog, config, &constrained_start, eff_exog, method, start_params.is_some())? {
        return Ok(r);
    }
    if maxiter == 0 {
        return build_zero_iter_result(eff_endog, config, &constrained_start, eff_exog, method);
    }

    // 2.5. CSS-ML 2-stage pre-optimization (R-style):
    // R's stats::arima(method="CSS-ML") runs CSS → ML for ALL ARMA models.
    // CSS is O(n·(p'+q')) per eval vs O(n·k³) for KF, so even with L-BFGS-B
    // gradient (2n+1 evals), CSS is much cheaper than a single KF evaluation.
    //
    // Strategy:
    //   - Seasonal models: REPLACE when CSS gives any KF improvement (R-matching)
    //   - Non-seasonal models: REPLACE only when improvement > 2.0 loglike units
    //     (avoids near-cancellation basin-trapping for borderline ARIMA cases)
    if start_params.is_none() {
        let n_arma = config.order.p + config.order.q + config.order.pp + config.order.qq;
        let is_seasonal = config.order.s >= 2
            && (config.order.pp > 0 || config.order.qq > 0);
        let benefit_from_css = n_arma >= 1 && (is_seasonal || n_arma >= 2);
        if benefit_from_css {
            // CSS optimization uses original endog (CSS applies differencing internally)
            if let Some(css_params) = run_css_optimization(endog, config, &constrained_start, 300, exog) {
                // KF evaluation uses eff_endog (already differenced)
                let css_kf_ll = eval_kf_loglike_constrained(eff_endog, config, &css_params, eff_exog);
                let orig_kf_ll =
                    eval_kf_loglike_constrained(eff_endog, config, &constrained_start, eff_exog);
                let threshold = if is_seasonal { 0.0 } else { 2.0 };
                if css_kf_ll > orig_kf_ll + threshold {
                    constrained_start = css_params;
                }
            }
        }
    }

    // 3. Transform to unconstrained space + build objective (with eff_endog)
    let unconstrained_start = untransform_params(&constrained_start, config)?;
    let objective = SarimaxObjective {
        endog: eff_endog.to_vec(),
        config: config.clone(),
        exog: eff_exog.map(|e| e.to_vec()),
        cache: RefCell::new(None),
        ss_cache: RefCell::new(None),
    };
    let n_restarts = compute_n_restarts(unconstrained_start.len(), config);

    // 4. Match method → dispatch to extracted function
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
        "lbfgsb" => fit_lbfgsb_single(&objective, unconstrained_start, config, maxiter)?,
        "lbfgsb-multi" => fit_lbfgsb_multi(
            &objective,
            &unconstrained_start,
            config,
            maxiter,
            n_restarts,
        )?,
        "lbfgs" => {
            fit_lbfgs_argmin(&objective, &unconstrained_start, config, maxiter, n_restarts)?
        }
        _ => {
            return Err(SarimaxError::OptimizationFailed(format!(
                "unknown method: '{}'. Use 'lbfgsb', 'lbfgsb-multi', 'lbfgsb-strict', 'lbfgs', or 'nelder-mead'",
                method
            )));
        }
    };

    // 5. Build result (n_obs = eff_endog.len() for correct AIC/BIC)
    build_fit_result(eff_endog, config, &best_unconstrained, n_iter, converged, used_method, eff_exog)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{load_fixtures, make_config_with_enforcement};

    #[test]
    fn test_transform_untransform_roundtrip() {
        let config = make_config_with_enforcement(2, 0, 1, true, true);
        let original = vec![0.5, -0.3, 0.2]; // ar(2), ma(1)
        let unconstrained = untransform_params(&original, &config).unwrap();
        let recovered = transform_params(&unconstrained, &config).unwrap();
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-10, "roundtrip failed: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_transform_passthrough_no_enforce() {
        let config = make_config_with_enforcement(1, 0, 1, false, false);
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

        let config = make_config_with_enforcement(1, 0, 0, false, false);
        let obj = SarimaxObjective {
            endog: data,
            config,
            exog: None,
            cache: RefCell::new(None),
            ss_cache: RefCell::new(None),
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

        let config = make_config_with_enforcement(1, 0, 0, false, false);
        let obj = SarimaxObjective {
            endog: data,
            config,
            exog: None,
            cache: RefCell::new(None),
            ss_cache: RefCell::new(None),
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
    fn test_fit_ar1_lbfgsb_convergence() {
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        // Test with enforce=false
        let config_noforce = make_config_with_enforcement(1, 0, 0, false, false);
        let r1 = fit(&data, &config_noforce, None, Some("lbfgsb"), Some(500), None).unwrap();
        assert!(r1.converged, "AR(1) lbfgsb enforce=false should converge");

        // Test with enforce=true (Python default)
        let config_force = make_config_with_enforcement(1, 0, 0, true, true);
        let r2 = fit(&data, &config_force, None, Some("lbfgsb"), Some(500), None).unwrap();
        assert!(r2.converged, "AR(1) lbfgsb enforce=true should converge");
    }

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
        let config = make_config_with_enforcement(1, 0, 0, false, false);
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
        let config = make_config_with_enforcement(1, 0, 1, false, false);
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
        let config = make_config_with_enforcement(1, 1, 1, false, false);
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

        let config = make_config_with_enforcement(1, 0, 0, false, false);
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

        let config = make_config_with_enforcement(1, 0, 0, true, true);
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

        let config = make_config_with_enforcement(1, 0, 0, false, false);
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

        let config = make_config_with_enforcement(1, 0, 0, false, false);

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

        let config = make_config_with_enforcement(1, 0, 1, false, false);
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

        let config = make_config_with_enforcement(1, 0, 0, false, false);
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

        let config = make_config_with_enforcement(1, 0, 1, false, false);
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

        let config = make_config_with_enforcement(1, 0, 1, false, false);
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

        let config = make_config_with_enforcement(1, 0, 1, false, false);
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

    // -------------------------------------------------------------------------
    // A-2: Near-cancellation detection tests (VER5.2 P6)
    // -------------------------------------------------------------------------

    #[test]
    fn test_polynomial_roots_ar1() {
        // AR(1) with φ=0.5: companion eigenvalue = 0.5 (inverted root)
        let roots = polynomial_roots(&[0.5]);
        assert_eq!(roots.len(), 1);
        assert!((roots[0].0 - 0.5).abs() < 1e-10, "real part: {}", roots[0].0);
        assert!(roots[0].1.abs() < 1e-10, "imag part: {}", roots[0].1);
    }

    #[test]
    fn test_polynomial_roots_ar2_real() {
        // AR(2) with real roots: φ₁=0.8, φ₂=-0.15 → roots near 0.3 and 0.5
        let roots = polynomial_roots(&[0.8, -0.15]);
        assert_eq!(roots.len(), 2);
        // Both roots should be real (imag ≈ 0)
        for (re, im) in &roots {
            assert!(im.abs() < 1e-8, "expected real root, got imag={}", im);
            assert!(re.abs() < 1.0 + 1e-6, "inverted root should be inside unit circle: {}", re);
        }
    }

    #[test]
    fn test_polynomial_roots_ar2_complex() {
        // AR(2) with complex roots: φ₁=0.0, φ₂=-0.5
        // Companion eigenvalues: λ² - 0·λ - (-0.5) = λ²+0.5 = 0 → λ=±i·√0.5
        // discriminant = φ₁² + 4φ₂ = 0 + 4·(-0.5) = -2 < 0 → complex
        let roots = polynomial_roots(&[0.0, -0.5]);
        assert_eq!(roots.len(), 2);
        // Should have a complex conjugate pair
        let has_complex = roots.iter().any(|(_, im)| im.abs() > 1e-8);
        assert!(has_complex, "expected complex roots for AR(2) with phi2=-0.5");
        // Conjugate: real parts equal, imag parts opposite
        assert!((roots[0].0 - roots[1].0).abs() < 1e-10);
        assert!((roots[0].1 + roots[1].1).abs() < 1e-10);
    }

    #[test]
    fn test_min_root_distance_far() {
        // AR root at 0.9 (real), MA root at 0.1 (real) → distance = 0.8
        let dist = min_root_distance(&[0.9], &[0.1]);
        assert!((dist - 0.8).abs() < 1e-10, "expected 0.8, got {}", dist);
    }

    #[test]
    fn test_min_root_distance_near_cancellation() {
        // AR root ≈ MA root: φ=0.9, θ=0.89 → inverted roots 0.9 vs 0.89 → dist=0.01
        let dist = min_root_distance(&[0.9], &[0.89]);
        assert!(dist < 0.02, "expected near-cancellation (dist < 0.02), got {}", dist);
        assert!(dist > 0.005, "dist should be ~0.01, got {}", dist);
    }

    #[test]
    fn test_validate_no_near_cancellation_ar_only() {
        // Pure AR: no MA → always valid
        let config = make_config_with_enforcement(2, 0, 0, false, false);
        let sp = SarimaxParams {
            ar_coeffs: vec![0.5, -0.2],
            ma_coeffs: vec![],
            sar_coeffs: vec![],
            sma_coeffs: vec![],
            sigma2: None,
            exog_coeffs: vec![],
            trend_coeffs: vec![],
        };
        assert!(validate_no_near_cancellation(&sp, &config, 0.05));
    }

    #[test]
    fn test_validate_no_near_cancellation_arma_far() {
        // ARMA(1,1) with distant roots: valid
        let config = make_config_with_enforcement(1, 0, 1, false, false);
        let sp = SarimaxParams {
            ar_coeffs: vec![0.8],
            ma_coeffs: vec![0.2],
            sar_coeffs: vec![],
            sma_coeffs: vec![],
            sigma2: None,
            exog_coeffs: vec![],
            trend_coeffs: vec![],
        };
        assert!(validate_no_near_cancellation(&sp, &config, 0.05));
    }

    #[test]
    fn test_validate_no_near_cancellation_arma_near() {
        // ARMA(1,1) with near-cancellation: φ≈θ → should fail α=0.05 check
        let config = make_config_with_enforcement(1, 0, 1, false, false);
        let sp = SarimaxParams {
            ar_coeffs: vec![0.9],
            ma_coeffs: vec![0.89],
            sar_coeffs: vec![],
            sma_coeffs: vec![],
            sigma2: None,
            exog_coeffs: vec![],
            trend_coeffs: vec![],
        };
        assert!(!validate_no_near_cancellation(&sp, &config, 0.05));
    }

    #[test]
    fn test_passes_cancellation_filter_ar_only() {
        // Pure AR: always passes (no MA to cancel with)
        let config = make_config_with_enforcement(2, 0, 0, false, false);
        let params = vec![0.5, -0.2]; // unconstrained = constrained when !enforce
        assert!(passes_cancellation_filter(&params, &config));
    }

    #[test]
    fn test_passes_cancellation_filter_arma_far() {
        // ARMA(1,1) with distant roots: should pass
        let config = make_config_with_enforcement(1, 0, 1, false, false);
        let params = vec![0.8, 0.2]; // ar=0.8, ma=0.2 → roots far apart
        assert!(passes_cancellation_filter(&params, &config));
    }

    #[test]
    fn test_passes_cancellation_filter_arma_near() {
        // ARMA(1,1) with near-cancellation: should fail α=0.01 check
        let config = make_config_with_enforcement(1, 0, 1, false, false);
        let params = vec![0.9, 0.895]; // ar=0.9, ma=0.895 → dist=0.005 < 0.01
        assert!(!passes_cancellation_filter(&params, &config));
    }
}
