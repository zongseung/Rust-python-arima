//! Optimizer runners (BFGS, L-BFGS, L-BFGS-B, Nelder-Mead) and gradient helpers.

use argmin::core::{Executor, State, TerminationReason};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::neldermead::NelderMead;
use argmin::solver::quasinewton::{LBFGS, BFGS};

use crate::error::SarimaxError;
use crate::types::SarimaxConfig;

use super::DIAG_PRECOND_SCALE;
use super::objective::{ProfiledSarimaxObjective, SarimaxObjective};

// ---------------------------------------------------------------------------
// L-BFGS optimization
// ---------------------------------------------------------------------------

/// Run argmin BFGS (full-Hessian) with MoreThuente line search.
///
/// Outcome of a single optimizer run: `(params, cost, n_iter, converged)`.
pub(super) type RunOutcome = std::result::Result<(Vec<f64>, f64, u64, bool), String>;

/// A `RunOutcome` tagged with the method string that produced it:
/// `(params, cost, n_iter, converged, method)`.
pub(super) type MethodOutcome =
    std::result::Result<(Vec<f64>, f64, u64, bool, String), SarimaxError>;

/// Compared to L-BFGS: keeps full n×n Hessian approximation (no limited memory)
/// → first-iteration step is curvature-aware as the BFGS update accumulates,
/// reducing the chance of large overshoots in high-magnitude dimensions like
/// exog coefficients. Closer in behavior to R's `optim(method='BFGS')`.
pub(super) fn run_bfgs(
    objective: SarimaxObjective,
    init_params: Vec<f64>,
    maxiter: u64,
) -> RunOutcome {
    let n = init_params.len();
    let linesearch = MoreThuenteLineSearch::new();
    let solver = BFGS::new(linesearch)
        .with_tolerance_grad(1e-5)
        .map_err(|e| e.to_string())?
        .with_tolerance_cost(1e-9)
        .map_err(|e| e.to_string())?;

    // Diagonal preconditioning: inv_H_0 = c · diag(1 / |∇f_i|).
    //
    // Without preconditioning, BFGS starts with inv_H = I and the line search
    // picks α based on the largest-gradient dimension. For SARIMAX, the ARMA
    // coefficients can have |∇f| ~ 25,000 while exog coefficients have |∇f| < 5.
    // The resulting α is "right" for ARMA dims (Monahan/Jones transform clamps
    // any overshoot) but completely wrong for unbounded exog dims, which drift
    // far from their start values in a single line-search probe and never
    // recover (the basin we land in has the wrong ta sign — see analysis at
    // https://en.wikipedia.org/wiki/Limited-memory_BFGS#Scaling).
    //
    // Preconditioning with `1/|∇f_i|` per dim makes the first step
    // approximately ±c in every dimension simultaneously, regardless of
    // gradient magnitude. The line search then chooses α relative to *all*
    // dims, not just the steepest. Choice of c=0.1 is conservative: small
    // enough to keep first-step exog moves bounded, large enough for ARMA
    // dims to make meaningful progress before BFGS curvature kicks in.
    // Reference: Nocedal & Wright (2006) §6.4 — Numerical Optimization.
    let grad0 = objective
        .analytical_gradient_negloglike(&init_params)
        .unwrap_or_else(|_| vec![1.0; n]);
    let c_precond = DIAG_PRECOND_SCALE;
    let init_hessian: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if i == j {
                        c_precond / grad0[i].abs().max(1e-3)
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();

    let result = Executor::new(objective, solver)
        .configure(
            |state: argmin::core::IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, (), f64>| {
                state.param(init_params).inv_hessian(init_hessian).max_iters(maxiter)
            },
        )
        .run()
        .map_err(|e| format!("BFGS failed: {}", e))?;

    let state = result.state();
    let best_param = state
        .get_best_param()
        .ok_or("BFGS: no best parameter found")?
        .clone();
    let best_cost = state.get_best_cost();
    let n_iter = state.get_iter();
    let term_reason = state.get_termination_reason();
    let converged = term_reason == Some(&TerminationReason::SolverConverged)
        || term_reason == Some(&TerminationReason::TargetCostReached);

    Ok((best_param, best_cost, n_iter, converged))
}

/// Central finite-difference gradient with relative step h = eps * (1 + |x_i|).
///
/// `f` must return the objective value to differentiate (already sign-adjusted
/// by the caller). Errors from `f` propagate.
fn central_diff_grad<F>(
    x: &[f64],
    eps: f64,
    mut f: F,
) -> std::result::Result<Vec<f64>, String>
where
    F: FnMut(&[f64]) -> std::result::Result<f64, String>,
{
    let n = x.len();
    let mut grad = vec![0.0; n];
    let mut xw = x.to_vec();
    for i in 0..n {
        let h = eps * (1.0 + x[i].abs());
        let orig = xw[i];
        xw[i] = orig + h;
        let fp = f(&xw)?;
        xw[i] = orig - h;
        let fm = f(&xw)?;
        xw[i] = orig;
        grad[i] = (fp - fm) / (2.0 * h);
    }
    Ok(grad)
}

/// Central finite-difference gradient of -LL in unconstrained space.
/// Used by trust-region BFGS — analytical gradient (via score()) returns
/// near-zero values for unbounded exog dimensions on this codebase, so we
/// fall back to finite-diff which empirically matches the true loss
/// landscape (verified against statsmodels at the same parameters).
pub(super) fn finite_diff_grad_negll(
    objective: &SarimaxObjective,
    x: &[f64],
    eps: f64,
) -> std::result::Result<Vec<f64>, String> {
    central_diff_grad(x, eps, |xi| objective.eval_loglike(xi).map(|ll| -ll))
}

pub(super) fn finite_diff_grad_profile_negll(
    objective: &ProfiledSarimaxObjective,
    x: &[f64],
    eps: f64,
) -> std::result::Result<Vec<f64>, String> {
    central_diff_grad(x, eps, |xi| objective.eval_negloglike(xi))
}

pub(super) fn run_lbfgs(
    objective: SarimaxObjective,
    init_params: Vec<f64>,
    maxiter: u64,
) -> RunOutcome {
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

pub(super) fn run_nelder_mead(
    objective: SarimaxObjective,
    init_params: Vec<f64>,
    maxiter: u64,
) -> RunOutcome {
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
pub(super) fn compute_bounds(config: &SarimaxConfig) -> Vec<(Option<f64>, Option<f64>)> {
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

pub(super) fn run_lbfgsb(
    objective: &SarimaxObjective,
    init_params: Vec<f64>,
    bounds_vec: Vec<(Option<f64>, Option<f64>)>,
    maxiter: u64,
) -> RunOutcome {
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

        let mut x_work = x.to_vec();
        for i in 0..n {
            let eps_i = (1.0 + x_work[i].abs()) * 1e-7;
            let orig = x_work[i];
            x_work[i] = orig + eps_i;
            let f_plus = match obj.eval_negloglike(&x_work) {
                Ok(c) if c.is_finite() => c,
                _ => cost,
            };
            x_work[i] = orig;
            g[i] = (f_plus - cost) / eps_i;
            if !g[i].is_finite() {
                g[i] = 0.0;
            }
        }
        Ok(cost)
    };

    let param = crate::lbfgsb_wrapper::LbfgsbParameter {
        m: 10,       // memory size (scipy default: 10)
        factr: 1e7,  // cost tolerance: factr * eps_mach ≈ 1e-9 (scipy default)
        pgtol: 1e-5, // projected gradient tolerance (scipy default)
        iprint: -1,  // silent
    };

    let mut problem = crate::lbfgsb_wrapper::LbfgsbProblem::build(init_params, evaluate);
    problem.set_bounds(bounds_vec);

    let mut state = crate::lbfgsb_wrapper::LbfgsbState::new(problem, param);
    let final_task = state
        .minimize()
        .map_err(|e| format!("L-BFGS-B failed: {}", e))?;

    let x = state.x().to_vec();
    let cost = state.fx();
    let n_eval = eval_count.load(std::sync::atomic::Ordering::Relaxed);

    // Determine convergence from Fortran task code:
    // - CONVERGENCE (20-25): solver converged via pgtol or factr
    // - STOP/WARNING: solver stopped but not converged
    // Additionally, if we hit our eval limit, report not converged.
    let converged_by_solver = crate::lbfgsb_ffi::is_converged(final_task);
    let converged = converged_by_solver && !hit_limit.load(std::sync::atomic::Ordering::Relaxed);

    Ok((x, cost, n_eval, converged))
}

pub(super) fn consume_budget(remaining: &mut u64, total_work: &mut u64, n: u64) {
    let used = n.min(*remaining);
    *total_work = total_work.saturating_add(used);
    *remaining = remaining.saturating_sub(used);
}
