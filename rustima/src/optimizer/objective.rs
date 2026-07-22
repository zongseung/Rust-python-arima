//! Negative log-likelihood objective functions for the optimizer.

use std::cell::RefCell;

use nalgebra::{DMatrix, DVector};

use argmin::core::{CostFunction, Gradient};

use crate::css;
use crate::initialization::KalmanInit;
use crate::kalman::{kalman_filter_batched, kalman_loglike};
use crate::params::SarimaxParams;
use crate::score;
use crate::state_space::StateSpace;
use crate::types::SarimaxConfig;

use super::KF_FT_FALLBACK_VARIANCE;
use super::transforms::{apply_transform_jacobian, expected_param_len, transform_params};

// ---------------------------------------------------------------------------
// Objective function for argmin
// ---------------------------------------------------------------------------

/// Cached fused evaluation result (cost + gradient at same params).
///
/// Used by L-BFGS path to avoid redundant StateSpace construction when
/// argmin calls `cost()` and `gradient()` at the same parameter point.
pub(super) struct CachedEval {
    params: Vec<f64>,
    cost: f64,
    gradient: Vec<f64>,
}

/// Negative log-likelihood objective for optimizer.
pub(super) struct SarimaxObjective {
    pub(super) endog: Vec<f64>,
    pub(super) config: SarimaxConfig,
    pub(super) exog: Option<Vec<Vec<f64>>>,
    /// Single-entry cache: stores the last fused (cost, gradient) evaluation.
    /// Populated by `gradient()`, consumed by `cost()` at the same params.
    pub(super) cache: RefCell<Option<CachedEval>>,
    /// Cached StateSpace: reused across optimizer iterations via in-place
    /// update_params() to avoid reallocating k×k matrices every evaluation.
    pub(super) ss_cache: RefCell<Option<StateSpace>>,
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
    pub(super) fn eval_negloglike(&self, unconstrained: &[f64]) -> std::result::Result<f64, String> {
        self.eval_loglike(unconstrained).map(|ll| -ll)
    }

    /// Evaluate log-likelihood for given unconstrained parameters.
    pub(super) fn eval_loglike(&self, unconstrained: &[f64]) -> std::result::Result<f64, String> {
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
    pub(super) fn analytical_gradient_negloglike(
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
    pub(super) fn eval_negloglike_with_gradient(
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

/// Parameter length for the profiled objective.
///
/// Layout is the standard unconstrained layout with the exog beta block removed:
/// `[trend | ar(p) | ma(q) | sar(P) | sma(Q) | sigma2?]`.
fn profiled_param_len(config: &SarimaxConfig) -> usize {
    expected_param_len(config).saturating_sub(config.n_exog)
}

pub(super) fn remove_exog_block(flat: &[f64], config: &SarimaxConfig) -> Vec<f64> {
    let kt = config.trend.k_trend();
    let n_exog = config.n_exog;
    let mut out = Vec::with_capacity(flat.len().saturating_sub(n_exog));
    out.extend_from_slice(&flat[..kt]);
    out.extend_from_slice(&flat[kt + n_exog..]);
    out
}

fn insert_exog_block(profiled: &[f64], exog_beta: &[f64], config: &SarimaxConfig) -> Vec<f64> {
    let kt = config.trend.k_trend();
    let mut out = Vec::with_capacity(profiled.len() + exog_beta.len());
    out.extend_from_slice(&profiled[..kt]);
    out.extend_from_slice(exog_beta);
    out.extend_from_slice(&profiled[kt..]);
    out
}

/// Objective value and recovered full parameterization for the profiled method.
pub(super) struct ProfiledEval {
    negll: f64,
    pub(super) full_unconstrained: Vec<f64>,
    full_params: SarimaxParams,
}

/// SARIMAX objective that profiles out exogenous regression coefficients.
///
/// For fixed nonlinear parameters, beta is estimated by GLS in Kalman
/// innovation space. The optimizer only sees the non-exog coordinates.
#[derive(Clone)]
pub(super) struct ProfiledSarimaxObjective {
    pub(super) endog: Vec<f64>,
    pub(super) config: SarimaxConfig,
    pub(super) exog: Vec<Vec<f64>>,
}

impl ProfiledSarimaxObjective {
    pub(super) fn eval_profiled(&self, profiled_unconstrained: &[f64]) -> std::result::Result<ProfiledEval, String> {
        if profiled_unconstrained.len() != profiled_param_len(&self.config) {
            return Err(format!(
                "profiled parameter length mismatch: expected {}, got {}",
                profiled_param_len(&self.config),
                profiled_unconstrained.len()
            ));
        }

        let zero_beta = vec![0.0; self.config.n_exog];
        let full_zero_unconstrained =
            insert_exog_block(profiled_unconstrained, &zero_beta, &self.config);
        let full_zero_constrained =
            transform_params(&full_zero_unconstrained, &self.config).map_err(|e| e.to_string())?;
        let zero_params =
            SarimaxParams::from_flat(&full_zero_constrained, &self.config).map_err(|e| e.to_string())?;

        let (beta_hat, loglike) = self.profile_beta_and_loglike(&zero_params)?;
        let full_unconstrained =
            insert_exog_block(profiled_unconstrained, &beta_hat, &self.config);
        let full_constrained =
            transform_params(&full_unconstrained, &self.config).map_err(|e| e.to_string())?;
        let full_params =
            SarimaxParams::from_flat(&full_constrained, &self.config).map_err(|e| e.to_string())?;

        Ok(ProfiledEval {
            negll: -loglike,
            full_unconstrained,
            full_params,
        })
    }

    fn profile_beta_and_loglike(
        &self,
        zero_params: &SarimaxParams,
    ) -> std::result::Result<(Vec<f64>, f64), String> {
        let n_exog = self.config.n_exog;
        if n_exog == 0 {
            let ss = StateSpace::new(&self.config, zero_params, &self.endog, None)
                .map_err(|e| e.to_string())?;
            let init = KalmanInit::from_config_default(&ss, &self.config);
            let output =
                kalman_loglike(&self.endog, &ss, &init, self.config.concentrate_scale)
                    .map_err(|e| e.to_string())?;
            return Ok((Vec::new(), output.loglike));
        }

        // Build the per-evaluation state-space ONCE. With `exog=None` and the
        // trend/exog coefficients in `zero_params` set to zero, the matrices
        // T, Z, R, Q and intercepts c_t, d_t depend only on the nonlinear
        // (ARMA) parameters — they are identical whether we filter `y` or any
        // `x_j`. The batched filter exploits this: covariance prediction/
        // update (the expensive O(k^2) per-step work) happens once and is
        // shared across all 1 + n_exog series, while the per-series state
        // mean recursion (O(k)) is repeated.
        let mut x_params = zero_params.clone();
        for coeff in x_params.trend_coeffs.iter_mut() {
            *coeff = 0.0;
        }
        for coeff in x_params.exog_coeffs.iter_mut() {
            *coeff = 0.0;
        }

        let ss = StateSpace::new(&self.config, &x_params, &self.endog, None)
            .map_err(|e| e.to_string())?;
        let init = KalmanInit::from_config_default(&ss, &self.config);

        let mut obs_refs: Vec<&[f64]> = Vec::with_capacity(1 + n_exog);
        obs_refs.push(self.endog.as_slice());
        for j in 0..n_exog {
            let col = self
                .exog
                .get(j)
                .ok_or_else(|| format!("missing exog column {}", j))?;
            obs_refs.push(col.as_slice());
        }

        let batched =
            kalman_filter_batched(&obs_refs, &ss, &init, self.config.concentrate_scale)
                .map_err(|e| e.to_string())?;

        let n_eff = batched.n_obs_effective;
        if n_eff == 0 {
            return Err("profiled objective has no effective observations".to_string());
        }
        let n_total = batched.innovation_vars.len();
        let burn = n_total.saturating_sub(n_eff);

        let y_innovations = &batched.innovations[0];
        let x_innovations: &[Vec<f64>] = &batched.innovations[1..];
        let innovation_vars = &batched.innovation_vars;

        let mut xtwx = DMatrix::<f64>::zeros(n_exog, n_exog);
        let mut xtwy = DVector::<f64>::zeros(n_exog);

        for t in burn..n_total {
            let f_t = innovation_vars
                .get(t)
                .copied()
                .unwrap_or(KF_FT_FALLBACK_VARIANCE);
            let f_safe = if f_t > 0.0 && f_t.is_finite() {
                f_t
            } else {
                KF_FT_FALLBACK_VARIANCE
            };
            let w = 1.0 / f_safe;
            let vy = y_innovations[t];
            for i in 0..n_exog {
                let xi = x_innovations[i][t];
                xtwy[i] += xi * vy * w;
                for j in 0..n_exog {
                    xtwx[(i, j)] += xi * x_innovations[j][t] * w;
                }
            }
        }

        let beta_vec = xtwx
            .svd(true, true)
            .solve(&xtwy, 1e-12)
            .map_err(|e| format!("profiled exog GLS solve failed: {}", e))?;
        let beta_hat: Vec<f64> = beta_vec.iter().copied().collect();

        let mut sum_log_f = 0.0;
        let mut sum_v2_f = 0.0;
        for t in burn..n_total {
            let f_t = innovation_vars
                .get(t)
                .copied()
                .unwrap_or(KF_FT_FALLBACK_VARIANCE);
            let f_safe = if f_t > 0.0 && f_t.is_finite() {
                f_t
            } else {
                KF_FT_FALLBACK_VARIANCE
            };
            let mut resid = y_innovations[t];
            for j in 0..n_exog {
                resid -= beta_hat[j] * x_innovations[j][t];
            }
            sum_log_f += f_safe.ln();
            sum_v2_f += resid * resid / f_safe;
        }

        if !sum_log_f.is_finite() || !sum_v2_f.is_finite() {
            return Err("profiled objective produced non-finite Kalman sums".to_string());
        }

        let loglike = if self.config.concentrate_scale {
            let sigma2_hat = (sum_v2_f / n_eff as f64).max(1e-300);
            -0.5 * (n_eff as f64) * (2.0 * std::f64::consts::PI).ln()
                - 0.5 * (n_eff as f64) * sigma2_hat.ln()
                - 0.5 * (n_eff as f64)
                - 0.5 * sum_log_f
        } else {
            -0.5 * (n_eff as f64) * (2.0 * std::f64::consts::PI).ln()
                - 0.5 * sum_log_f
                - 0.5 * sum_v2_f
        };

        if loglike.is_finite() {
            Ok((beta_hat, loglike))
        } else {
            Err("profiled objective produced non-finite loglike".to_string())
        }
    }

    pub(super) fn eval_negloglike(&self, profiled_unconstrained: &[f64]) -> std::result::Result<f64, String> {
        self.eval_profiled(profiled_unconstrained).map(|e| e.negll)
    }

    pub(super) fn analytical_gradient_negloglike(
        &self,
        profiled_unconstrained: &[f64],
    ) -> std::result::Result<Vec<f64>, String> {
        let eval = self.eval_profiled(profiled_unconstrained)?;
        let ss = StateSpace::new(
            &self.config,
            &eval.full_params,
            &self.endog,
            Some(&self.exog),
        )
        .map_err(|e| e.to_string())?;
        let init = KalmanInit::from_config_default(&ss, &self.config);
        let score_constrained = score::score(
            &self.endog,
            &ss,
            &init,
            &self.config,
            &eval.full_params,
            self.config.concentrate_scale,
            Some(&self.exog),
        )
        .map_err(|e| e.to_string())?;
        let full_grad =
            apply_transform_jacobian(&score_constrained, &eval.full_unconstrained, &self.config)?;
        let profiled_grad = remove_exog_block(&full_grad, &self.config);
        Ok(profiled_grad.iter().map(|&g| -g).collect())
    }
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

        let f0 = self.cost(param)?;
        let mut p_work = param.clone();

        for i in 0..n {
            let eps_i = (1.0 + p_work[i].abs()) * 1e-7;
            let orig = p_work[i];
            p_work[i] = orig + eps_i;
            let f_plus = self.cost(&p_work)?;
            p_work[i] = orig;

            grad[i] = (f_plus - f0) / eps_i;

            if !grad[i].is_finite() {
                p_work[i] = orig + eps_i;
                let fp = self.cost(&p_work)?;
                p_work[i] = orig - eps_i;
                let fm = self.cost(&p_work)?;
                p_work[i] = orig;
                grad[i] = (fp - fm) / (2.0 * eps_i);
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
pub(super) struct CssObjective {
    pub(super) endog: Vec<f64>,
    pub(super) config: SarimaxConfig,
    pub(super) exog: Option<Vec<Vec<f64>>>,
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
