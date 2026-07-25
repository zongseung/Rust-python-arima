//! SARIMAX parameter optimization via L-BFGS with Nelder-Mead fallback.
//!
//! This module provides:
//! - Parameter space transformations (constrained ↔ unconstrained)
//! - Negative log-likelihood objective function for argmin
//! - `fit()` function: the main entry point for model fitting

use argmin::core::{Executor, State};
use argmin::solver::neldermead::NelderMead;

use std::cell::RefCell;

use nalgebra::DMatrix;

use crate::css;
use crate::error::{Result, SarimaxError};
use crate::pipeline;
use crate::initialization::KalmanInit;
use crate::kalman::kalman_loglike;
use crate::params::SarimaxParams;
use crate::start_params::compute_start_params;
use crate::state_space::StateSpace;
use crate::types::{FitResult, SarimaxConfig};

mod transforms;
mod objective;
mod runners;
mod trust_region;
mod multistart;
#[cfg(test)]
mod tests;

pub use transforms::{transform_params, untransform_params};
use transforms::expected_param_len;
use objective::{CssObjective, SarimaxObjective};
use runners::{compute_bounds, run_lbfgsb, run_nelder_mead};
use multistart::{
    compute_n_restarts, fit_bfgs_single, fit_lbfgs_argmin, fit_lbfgsb_adaptive_restart,
    fit_lbfgsb_hybrid, fit_lbfgsb_multi, fit_lbfgsb_single,
};
use trust_region::{fit_profile_trust_region, fit_trust_region_single};

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

/// Diagonal preconditioning scale `c` for BFGS / trust-region: the initial
/// inverse Hessian is `c · diag(1 / |∇f_i|)`. Smaller values make the first
/// step more conservative (good for ill-conditioned problems with mixed-scale
/// gradients across dimensions); larger values make the first step bolder.
/// 0.1 chosen by experiment so that the first-step displacement in each
/// dimension is ≈ 0.1 in unconstrained space — small enough to keep exog
/// coefficients near their CSS seed, large enough for ARMA dims to begin
/// curvature accumulation.
const DIAG_PRECOND_SCALE: f64 = 0.1;

/// Penalty variance used when the Kalman innovation variance F_t is non-
/// positive at observation t (a sign the model is at a near-non-stationary
/// boundary, e.g. AR roots ≈ unit circle). Instead of aborting, we contribute
/// `log F_safe` to the log-likelihood for that observation, yielding a smooth,
/// finite-but-bad cost the optimizer can back away from. Value ≈ 1e10 means
/// each offending observation costs ~23 log-likelihood units — small enough
/// to be meaningful, large enough to dominate any real innovation magnitude.
pub(crate) const KF_FT_FALLBACK_VARIANCE: f64 = 1.0e10;

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
                warnings: vec![],
            }
            .with_information_criteria(),
        ))
    } else {
        Ok(None)
    }
}

/// Build result for maxiter=0 or zero-parameter models: no optimization needed.
///
/// `converged` is `true` when `n_params == 0` (trivially converged — nothing to
/// optimize), `false` when the caller explicitly requested `maxiter=0` but there
/// are free parameters remaining.
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
    let k_free = expected_param_len(config);
    let converged = k_free == 0;
    Ok(FitResult {
        params: constrained_start.to_vec(),
        loglike: output.loglike,
        scale: output.scale,
        n_obs: endog.len(),
        n_params,
        n_iter: 0,
        converged,
        method: method.to_string(),
        aic: 0.0,
        bic: 0.0,
        warnings: vec![],
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

    // A-2: Collect near-cancellation warning into result (VER5.2 P6, V8.5 P2-1)
    let mut warnings = Vec::new();
    if !validate_no_near_cancellation(&final_params, config, 0.05) {
        warnings.push(format!(
            "near-cancellation detected in fitted ARMA({},{}) \
             parameters (min inverted-root distance < 0.05). \
             Model may be non-identifiable.",
            config.order.p, config.order.q
        ));
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
        warnings,
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

/// For high-dimensional state-space models (k_states >= 40), the standard
/// optimization landscape is treacherous — many local minima.  Pre-fitting
/// with `simple_differencing=true` reduces the state dimension (removes
/// diff states), giving a smoother landscape and better convergence.
/// The resulting params are then used as warm-start for the full model.
fn run_sd_warm_start(
    endog: &[f64],
    config: &SarimaxConfig,
    exog: Option<&[Vec<f64>]>,
) -> Option<Vec<f64>> {
    let mut sd_config = config.clone();
    sd_config.simple_differencing = true;

    let (sd_endog, sd_exog_owned) = pipeline::prepare_endog(endog, &sd_config, exog).ok()?;
    let sd_exog: Option<&[Vec<f64>]> = sd_exog_owned.as_deref();

    // Get CSS-based start params for SD model
    let mut sd_start = validate_and_get_start_params(endog, &sd_config, None, exog).ok()?;

    // CSS pre-optimization on SD path
    let n_arma = sd_config.order.p + sd_config.order.q + sd_config.order.pp + sd_config.order.qq;
    if n_arma >= 1 {
        if let Some(css_p) = run_css_optimization(endog, &sd_config, &sd_start, 200, exog) {
            let css_ll = eval_kf_loglike_constrained(&sd_endog, &sd_config, &css_p, sd_exog);
            let orig_ll = eval_kf_loglike_constrained(&sd_endog, &sd_config, &sd_start, sd_exog);
            if css_ll > orig_ll {
                sd_start = css_p;
            }
        }
    }

    // Quick L-BFGS-B on SD model (200 iterations — enough for warm-start)
    let sd_unc = untransform_params(&sd_start, &sd_config).ok()?;
    let sd_obj = SarimaxObjective {
        endog: sd_endog.to_vec(),
        config: sd_config.clone(),
        exog: sd_exog.map(|e| e.to_vec()),
        cache: RefCell::new(None),
        ss_cache: RefCell::new(None),
    };

    let (best_unc, _, _, _, _) = fit_lbfgsb_single(&sd_obj, sd_unc, &sd_config, 200).ok()?;
    transform_params(&best_unc, &sd_config).ok()
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

    // Real Schur decomposition → quasi-upper triangular T.
    //
    // MUST be bounded: `Schur::new` iterates the QR algorithm without an
    // iteration cap and can spin forever on ill-conditioned companion
    // matrices (observed: the degree-10 companion of a (10,0,10) fit hung
    // indefinitely while holding the GIL). The roots feed only the
    // near-cancellation warning/filter, so on non-convergence we degrade
    // gracefully by returning no roots (min_root_distance → INFINITY →
    // "no near-cancellation" → the warning is simply skipped).
    let max_niter = 100 * p.max(10);
    let Some(schur) = nalgebra::Schur::try_new(companion, 1e-12, max_niter) else {
        return vec![];
    };
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
    // AR is 1 - φ₁z - … so the companion takes +φ, but MA is 1 + θ₁z + …
    // (arima2: `1/polyroot(c(1, ma_pars))`), so its companion needs -θ.
    // Passing +θ compares against the roots of the sign-flipped polynomial
    // and misses genuine cancellation entirely (DIAGNOSIS_V9 follow-up N3).
    let neg_ma: Vec<f64> = ma_coeffs.iter().map(|v| -v).collect();
    let ma_roots = polynomial_roots(&neg_ma);

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
        match transform_params(unconstrained, &config_owned) {
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

    let param = crate::lbfgsb_wrapper::LbfgsbParameter {
        m: 10,
        factr: 1e7,
        pgtol: 1e-5,
        iprint: -1,
    };

    let mut problem = crate::lbfgsb_wrapper::LbfgsbProblem::build(unconstrained_start.to_vec(), evaluate);
    problem.set_bounds(bounds_vec);

    let mut state = crate::lbfgsb_wrapper::LbfgsbState::new(problem, param);
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
/// * `method` — "lbfgsb" (default), "lbfgsb-multi", "trust-region",
///   "profile-trust-region", "lbfgs", or "nelder-mead"
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

    // --- Early validation (before any fast-path) ---
    // (a) Method whitelist: reject unknown methods immediately.
    const VALID_METHODS: &[&str] = &[
        "lbfgsb", "lbfgsb-multi", "lbfgsb-adaptive", "lbfgsb-hybrid",
        "lbfgsb-strict", "lbfgsb_single",
        "lbfgs", "bfgs", "trust-region", "profile-trust-region",
        "nelder-mead", "nm",
    ];
    if !VALID_METHODS.contains(&method) {
        return Err(SarimaxError::OptimizationFailed(format!(
            "unknown method: '{}'. Valid methods: {}",
            method,
            VALID_METHODS.join(", ")
        )));
    }

    // (b) start_params length: must match expected free param count if provided.
    if let Some(sp) = start_params {
        let expected = expected_param_len(config);
        if sp.len() != expected {
            return Err(SarimaxError::InvalidInput(format!(
                "start_params length mismatch: expected {}, got {}",
                expected, sp.len()
            )));
        }
    }

    // simple_differencing: pre-difference the data so the Kalman filter sees
    // a stationary ARMA-only series. n_obs = eff_endog.len() for AIC/BIC.
    let (eff_endog, eff_exog_owned) = pipeline::prepare_endog(endog, config, exog)?;
    let eff_endog: &[f64] = &eff_endog;
    let eff_exog: Option<&[Vec<f64>]> = if config.simple_differencing {
        eff_exog_owned.as_deref()
    } else {
        exog
    };

    // 0. Zero-parameter fast path: models like (0,d,0) with concentrate_scale
    //    have no free parameters to optimize. Return closed-form result directly
    //    to avoid entering L-BFGS-B FFI with n=0.
    {
        let k_params = config.trend.k_trend()
            + config.n_exog
            + config.order.p
            + config.order.q
            + config.order.pp
            + config.order.qq
            + if config.concentrate_scale { 0 } else { 1 };
        if k_params == 0 {
            return build_zero_iter_result(eff_endog, config, &[], eff_exog, method);
        }
    }

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
    //
    // Trust-region method intentionally skips CSS pre-opt: its whole point is
    // to keep the optimizer close to a carefully-computed seed (CSS 2-stage
    // start_params) by capping per-step radius. CSS pre-opt would discard that
    // seed and start the trust-region search from a basin determined by CSS,
    // defeating the purpose of preventing first-iteration drift.
    let skip_css_preopt = method == "trust-region";
    if start_params.is_none() && !skip_css_preopt {
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

    // 2.7. High-dimensional warm-start via simple_differencing.
    // For k_states >= 40 (high-order seasonal models), the full state-space
    // optimization landscape has many local minima.  Pre-fitting with
    // simple_differencing=true gives a much smoother landscape, yielding
    // params close to the global optimum that then serve as warm-start.
    if start_params.is_none() && !config.simple_differencing && !skip_css_preopt {
        let k_states = config.order.k_states();
        if k_states >= 40 {
            if let Some(sd_params) = run_sd_warm_start(endog, config, exog) {
                let sd_ll = eval_kf_loglike_constrained(eff_endog, config, &sd_params, eff_exog);
                let orig_ll =
                    eval_kf_loglike_constrained(eff_endog, config, &constrained_start, eff_exog);
                if sd_ll > orig_ll + 1.0 {
                    constrained_start = sd_params;
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
                .map_err(SarimaxError::OptimizationFailed)?;
            (p, c, n, conv, "nelder-mead".to_string())
        }
        "lbfgsb-strict" | "lbfgsb_single" => {
            let bounds = compute_bounds(config);
            let (p, c, n, conv) = run_lbfgsb(&objective, unconstrained_start, bounds, maxiter)
                .map_err(SarimaxError::OptimizationFailed)?;
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
        "lbfgsb-adaptive" => fit_lbfgsb_adaptive_restart(
            &objective,
            unconstrained_start,
            config,
            maxiter,
            n_restarts,
        )?,
        "lbfgsb-hybrid" => fit_lbfgsb_hybrid(
            &objective,
            &unconstrained_start,
            config,
            maxiter,
            n_restarts,
        )?,
        "lbfgs" => {
            fit_lbfgs_argmin(&objective, &unconstrained_start, config, maxiter, n_restarts)?
        }
        "bfgs" => fit_bfgs_single(&objective, unconstrained_start, config, maxiter)?,
        "trust-region" => fit_trust_region_single(&objective, unconstrained_start, config, maxiter)?,
        "profile-trust-region" => {
            fit_profile_trust_region(
                &objective,
                unconstrained_start,
                config,
                maxiter,
                start_params.is_some(),
            )?
        }
        // Unreachable in practice: the VALID_METHODS whitelist above rejects
        // unknown methods before dispatch. Kept as an error (not a panic) so a
        // future whitelist/match drift degrades gracefully.
        _ => {
            return Err(SarimaxError::OptimizationFailed(format!(
                "unknown method: '{}'. Valid methods: {}",
                method,
                VALID_METHODS.join(", ")
            )));
        }
    };

    // 5. Build result (n_obs = eff_endog.len() for correct AIC/BIC)
    build_fit_result(eff_endog, config, &best_unconstrained, n_iter, converged, used_method, eff_exog)
}
