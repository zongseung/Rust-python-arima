use nalgebra::{DMatrix, DVector};

use crate::error::{Result, SarimaxError};
use crate::initialization::{KalmanInit, KalmanSteadyState};
use crate::state_space::StateSpace;

/// Output of the Kalman filter loglikelihood computation.
#[derive(Debug, Clone)]
pub struct KalmanOutput {
    /// Log-likelihood value.
    pub loglike: f64,
    /// Estimated (concentrated) scale: sigma2_hat.
    pub scale: f64,
    /// Innovation sequence v_t.
    pub innovations: Vec<f64>,
    /// Effective number of observations (n - burn).
    pub n_obs_effective: usize,
}

/// Full Kalman filter output with final state information for forecasting.
#[derive(Debug, Clone)]
pub struct KalmanFilterOutput {
    /// Log-likelihood value.
    pub loglike: f64,
    /// Estimated (concentrated) scale: sigma2_hat.
    pub scale: f64,
    /// Innovation sequence v_t.
    pub innovations: Vec<f64>,
    /// Innovation variances F_t (before scaling by sigma2).
    pub innovation_vars: Vec<f64>,
    /// Effective number of observations (n - burn).
    pub n_obs_effective: usize,
    /// Final filtered state a_{n|n}.
    pub filtered_state: DVector<f64>,
    /// Final filtered covariance P_{n|n}.
    pub filtered_cov: DMatrix<f64>,
    /// Final predicted state a_{n+1|n}.
    pub predicted_state: DVector<f64>,
    /// Final predicted covariance P_{n+1|n}.
    pub predicted_cov: DMatrix<f64>,
}

/// Steady-state convergence tolerance for the gain vector pz = P*Z.
///
/// We check pz convergence (k-vector) because F_t = Z'*pz (scalar) can
/// converge while pz changes in directions orthogonal to Z, leading to
/// incorrect cached K_inf = T*pz/F. Error analysis shows the loglike
/// error from a converged-but-imprecise K_inf is O(eps * n^2) due to
/// compounding through the state recursion (unit-root states integrate
/// the error), so eps must be tight.
///
/// At 1e-9, loglike error stays below ~1e-5 for n=1000.
/// For models where pz converges slowly, steady-state won't trigger
/// and the sparse T*P*T' path provides the primary speedup instead.
pub(crate) const STEADY_STATE_TOL: f64 = 1e-9;

/// Minimum steps past burn-in before checking for convergence.
pub(crate) const STEADY_STATE_MIN_STEPS: usize = 5;

/// Number of consecutive converged pz values required.
pub(crate) const STEADY_STATE_CONSEC: usize = 3;

// ---------------------------------------------------------------------------
// Sparse T representation for optimized matrix operations
// ---------------------------------------------------------------------------

/// Sparse representation of the transition matrix T.
///
/// SARIMA companion matrices are very sparse (e.g. 31/729 = 4% for k=27).
/// This enables sparse T*a (O(nnz)), T*P (O(nnz*k)), and T*P*T' (O(nnz*k)).
pub(crate) struct SparseT {
    pub(crate) entries: Vec<(usize, usize, f64)>,
    pub(crate) k: usize,
}

impl SparseT {
    /// Build sparse T from a dense matrix. Always succeeds (used for steady-state
    /// which always needs sparse entries regardless of density).
    pub(crate) fn from_dense(t: &DMatrix<f64>, k: usize) -> Self {
        let mut entries = Vec::new();
        for i in 0..k {
            for j in 0..k {
                let v = t[(i, j)];
                if v != 0.0 {
                    entries.push((i, j, v));
                }
            }
        }
        SparseT { entries, k }
    }

    /// Check if T is sparse enough to warrant sparse operations (< 50% density).
    pub(crate) fn is_sparse(&self) -> bool {
        self.entries.len() < self.k * self.k / 2
    }

    /// Sparse matrix-vector multiply: result = T * v  (O(nnz))
    #[inline]
    pub(crate) fn t_mul_vec_into(&self, v: &DVector<f64>, result: &mut DVector<f64>) {
        let v_s = v.as_slice();
        let r_s = result.as_mut_slice();
        for val in r_s.iter_mut() {
            *val = 0.0;
        }
        for &(i, j, t_val) in &self.entries {
            r_s[i] += t_val * v_s[j];
        }
    }

    /// Sparse T*P into temp buffer: temp = T * P  (O(nnz * k))
    #[inline]
    pub(crate) fn t_mul_p_into(&self, p: &DMatrix<f64>, temp: &mut DMatrix<f64>) {
        let k = self.k;
        let p_data = p.as_slice(); // column-major
        let tmp = temp.as_mut_slice();
        for v in tmp.iter_mut() {
            *v = 0.0;
        }
        // temp[i, j] += T[i,l] * P[l,j]
        // In column-major: temp[i + j*k] += val * p_data[l + j*k]
        for &(i, l, val) in &self.entries {
            for j in 0..k {
                tmp[i + j * k] += val * p_data[l + j * k];
            }
        }
    }

    /// Sparse temp * T' + rqr into P: P = temp * T' + rqr  (O(nnz * k))
    #[inline]
    pub(crate) fn temp_tt_plus_rqr_into(
        &self,
        temp: &DMatrix<f64>,
        rqr: &DMatrix<f64>,
        p: &mut DMatrix<f64>,
    ) {
        let k = self.k;
        let tmp = temp.as_slice();
        let p_data = p.as_mut_slice();
        let rqr_data = rqr.as_slice();
        // Start with RQR'
        p_data.copy_from_slice(rqr_data);
        // Accumulate: P[i,j] += temp[i,l] * T[j,l]
        // Iterate over sparse T entries as (j, l, val)
        for &(j, l, val) in &self.entries {
            // Add val * temp[:, l] to P[:, j]
            let col_l = l * k;
            let col_j = j * k;
            for i in 0..k {
                p_data[col_j + i] += val * tmp[col_l + i];
            }
        }
    }
}

/// Sparse Z representation for O(nnz_z) dot product operations.
pub(crate) struct SparseZ {
    pub(crate) entries: Vec<(usize, f64)>,
}

impl SparseZ {
    pub(crate) fn from_dense(z: &DVector<f64>, k: usize) -> Self {
        let mut entries = Vec::new();
        for i in 0..k {
            let v = z[i];
            if v != 0.0 {
                entries.push((i, v));
            }
        }
        SparseZ { entries }
    }

    /// Sparse dot product: Z' * a  (O(nnz_z))
    #[inline]
    pub(crate) fn dot(&self, a: &DVector<f64>) -> f64 {
        let a_s = a.as_slice();
        self.entries.iter().map(|&(i, v)| v * a_s[i]).sum()
    }

    /// Sparse P * Z using column access: pz = P * Z  (O(nnz_z * k))
    #[inline]
    pub(crate) fn p_mul_z_into(&self, p: &DMatrix<f64>, pz: &mut DVector<f64>, k: usize) {
        let pz_s = pz.as_mut_slice();
        let p_data = p.as_slice(); // column-major
        for v in pz_s.iter_mut() {
            *v = 0.0;
        }
        for &(zi, zv) in &self.entries {
            // P[:, zi] * zv -> accumulate into pz
            let col_start = zi * k;
            for r in 0..k {
                pz_s[r] += zv * p_data[col_start + r];
            }
        }
    }

    /// Sparse F_t = Z' * pz  (O(nnz_z))
    #[inline]
    pub(crate) fn z_dot_pz(&self, pz: &DVector<f64>) -> f64 {
        self.entries.iter().map(|&(i, v)| v * pz[i]).sum()
    }
}

// ---------------------------------------------------------------------------
// Kalman filter execution strategy (enum dispatch = zero-cost)
// ---------------------------------------------------------------------------

/// Kalman filter execution strategy for the non-converged phase.
///
/// Uses enum dispatch for zero-cost abstraction. Selects between:
/// - Dense: Standard nalgebra gemv/gemm for small state dimensions or dense T.
/// - Sparse: Sparse T*P*T' computation when T density < 50%.
///
/// The steady-state (converged) phase is handled separately in the main loop
/// since it has fundamentally different control flow (no filter update, frozen P).
enum KalmanStrategy {
    /// Standard dense matrix operations.
    Dense,
    /// Sparse T matrix optimization (T density < 50%).
    Sparse,
}

impl KalmanStrategy {
    /// Compute pz = P*Z and F_t = Z'*pz.
    #[inline]
    fn compute_pz_and_f(
        &self,
        p: &DMatrix<f64>,
        z: &DVector<f64>,
        sparse_z: &SparseZ,
        sparse_t: &SparseT,
        pz: &mut DVector<f64>,
    ) -> f64 {
        match self {
            Self::Sparse => {
                sparse_z.p_mul_z_into(p, pz, sparse_t.k);
                sparse_z.z_dot_pz(pz)
            }
            Self::Dense => {
                pz.gemv(1.0, p, z, 0.0);
                z.dot(pz)
            }
        }
    }

    /// Predict state: a_next = T * a_updated + c_t.
    #[inline]
    fn predict_state(
        &self,
        t_mat: &DMatrix<f64>,
        sparse_t: &SparseT,
        state: &DVector<f64>,
        a_next: &mut DVector<f64>,
        state_intercept: &[f64],
        t: usize,
        k: usize,
        has_state_intercept: bool,
    ) {
        match self {
            Self::Sparse => {
                sparse_t.t_mul_vec_into(state, a_next);
            }
            Self::Dense => {
                a_next.gemv(1.0, t_mat, state, 0.0);
            }
        }
        if has_state_intercept {
            let an_s = a_next.as_mut_slice();
            let base = t * k;
            for i in 0..k {
                an_s[i] += state_intercept[base + i];
            }
        }
    }

    /// Predict covariance: P = T * P_updated * T' + RQR'.
    #[inline]
    fn predict_cov(
        &self,
        t_mat: &DMatrix<f64>,
        t_mat_t: &DMatrix<f64>,
        sparse_t: &SparseT,
        p: &mut DMatrix<f64>,
        rqr: &DMatrix<f64>,
        temp_kk: &mut DMatrix<f64>,
    ) {
        match self {
            Self::Sparse => {
                sparse_t.t_mul_p_into(p, temp_kk);
                sparse_t.temp_tt_plus_rqr_into(temp_kk, rqr, p);
            }
            Self::Dense => {
                temp_kk.gemm(1.0, t_mat, p, 0.0);
                p.gemm(1.0, temp_kk, t_mat_t, 0.0);
                *p += rqr;
            }
        }
    }

    /// Compute K_inf = T * pz for steady-state transition.
    #[inline]
    fn compute_k_gain(
        &self,
        t_mat: &DMatrix<f64>,
        sparse_t: &SparseT,
        pz: &DVector<f64>,
        k_gain: &mut DVector<f64>,
    ) {
        match self {
            Self::Sparse => {
                sparse_t.t_mul_vec_into(pz, k_gain);
            }
            Self::Dense => {
                k_gain.gemv(1.0, t_mat, pz, 0.0);
            }
        }
    }
}

/// Cached steady-state quantities, frozen once P converges.
struct SteadyStateCache {
    /// Cached Kalman gain K_inf = T * pz_inf.
    k_gain: DVector<f64>,
    /// Cached innovation variance F_inf = Z' * P_inf * Z.
    f_steady: f64,
    /// Cached ln(F_inf).
    log_f_steady: f64,
    /// Cached pz_inf = P_inf * Z for filtered state computation.
    pz_inf: DVector<f64>,
}

/// Check pz-vector based convergence for steady-state detection.
///
/// Returns true if convergence criterion is met for STEADY_STATE_CONSEC
/// consecutive steps.
#[inline]
pub(crate) fn check_convergence(
    pz: &DVector<f64>,
    pz_prev: &DVector<f64>,
    consec_count: &mut usize,
) -> bool {
    let pz_diff_sq: f64 = pz
        .iter()
        .zip(pz_prev.iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum();
    let pz_norm_sq: f64 = pz_prev.iter().map(|v| v * v).sum();
    let pz_norm = pz_norm_sq.sqrt().max(1e-15);

    if pz_diff_sq.sqrt() / pz_norm < STEADY_STATE_TOL {
        *consec_count += 1;
        *consec_count >= STEADY_STATE_CONSEC
    } else {
        *consec_count = 0;
        false
    }
}

// ---------------------------------------------------------------------------
// Core Kalman filter with strategy-based dispatch
// ---------------------------------------------------------------------------

/// Compute the scalar innovation: v_t = y_t - Z'*a_t - d_t.
///
/// Uses the sparse Z representation for O(nnz_z) instead of O(k).
#[inline(always)]
fn compute_innovation(endog_t: f64, sparse_z: &SparseZ, a: &DVector<f64>, d_t: f64) -> f64 {
    endog_t - sparse_z.dot(a) - d_t
}

/// Unified Kalman filter core with strategy-based dispatch.
///
/// When `store_full=false`, skips storing innovation_vars and filtered state
/// (used by `kalman_loglike`). When `store_full=true`, stores everything
/// needed for forecasting and diagnostics (used by `kalman_filter`).
///
/// **Strategy selection**: The filter automatically selects the best execution
/// strategy based on the structure of the transition matrix T:
/// - Dense: Standard O(k^3) operations for small or dense T.
/// - Sparse: O(nnz*k) operations when T density < 50%.
/// - SteadyState: O(nnz) per step once P converges (skips covariance updates).
///
/// **Note on state_intercept (c_t)**: The steady-state optimization is valid
/// even when state_intercept varies over time (e.g. trend models). Covariance
/// convergence depends only on T, Z, R, Q which are time-invariant in SARIMAX.
/// The time-varying c_t affects only the state mean recursion, not P, and is
/// correctly applied at each step in the steady-state path.
fn kalman_core(
    endog: &[f64],
    ss: &StateSpace,
    init: &KalmanInit,
    concentrate_scale: bool,
    store_full: bool,
) -> Result<KalmanFilterOutput> {
    let n = endog.len();
    let k = ss.k_states;
    let burn = init.loglikelihood_burn;

    if n <= burn {
        return Err(SarimaxError::DataError(format!(
            "Not enough observations: n={} <= burn={}",
            n, burn
        )));
    }

    let n_eff = n - burn;

    let mut a = init.initial_state.clone();
    let mut p = init.initial_state_cov.clone();

    let t_mat = &ss.transition;
    let z = &ss.design;
    let r_mat = &ss.selection;
    let q_mat = &ss.state_cov;

    // Precompute time-invariant matrices
    let rqr = r_mat * q_mat * r_mat.transpose();
    let t_mat_t = t_mat.transpose();
    let has_state_intercept = ss.state_intercept.len() == n * k;

    // Build sparse representations
    let sparse_z = SparseZ::from_dense(z, k);
    let sparse_t = SparseT::from_dense(t_mat, k);

    // Select strategy based on T density
    let strategy = if sparse_t.is_sparse() {
        KalmanStrategy::Sparse
    } else {
        KalmanStrategy::Dense
    };

    let mut sum_log_f = 0.0;
    let mut sum_v2_f = 0.0;
    let mut innovations = Vec::with_capacity(n);
    let mut innovation_vars = if store_full {
        Vec::with_capacity(n)
    } else {
        Vec::new()
    };

    // Filtered state tracking (only for store_full)
    let mut a_filtered = DVector::<f64>::zeros(k);
    let mut p_filtered = DMatrix::<f64>::zeros(k, k);

    // Pre-allocate work buffers
    let mut pz = DVector::<f64>::zeros(k);
    let mut a_next = DVector::<f64>::zeros(k);
    let mut temp_kk = DMatrix::<f64>::zeros(k, k);

    // Steady-state detection.
    // If the initializer pre-computed steady-state quantities (DARE with sd=0),
    // activate the cache immediately — P_0 = P_∞ so no transient phase exists.
    let mut ss_cache: Option<SteadyStateCache> = init
        .steady_state
        .as_ref()
        .filter(|_| burn == 0)
        .map(|KalmanSteadyState { k_gain, f_steady, log_f_steady, pz_inf }| {
            SteadyStateCache {
                k_gain: k_gain.clone(),
                f_steady: *f_steady,
                log_f_steady: *log_f_steady,
                pz_inf: pz_inf.clone(),
            }
        });
    let mut pz_prev = DVector::<f64>::zeros(k);
    let mut consec_count = 0_usize;

    for t in 0..n {
        // --- Observation intercept ---
        let d_t = if t < ss.obs_intercept.len() {
            ss.obs_intercept[t]
        } else {
            0.0
        };

        if let Some(ref cache) = ss_cache {
            // ---- STEADY-STATE PATH: O(nnz) per step ----
            // P has converged to P_inf, so K_inf and F_inf are constant.
            // Only the state mean recursion is computed; covariance is frozen.
            debug_assert!(
                cache.f_steady > 0.0 && cache.f_steady.is_finite(),
                "steady-state F must be positive and finite, got {}",
                cache.f_steady
            );
            let v_t = compute_innovation(endog[t], &sparse_z, &a, d_t);
            innovations.push(v_t);
            if store_full {
                innovation_vars.push(cache.f_steady);
            }

            // Predict: a_next = T*a + c_t + K_inf * (v_t / F_inf)
            strategy.predict_state(
                t_mat, &sparse_t, &a, &mut a_next,
                &ss.state_intercept, t, k, has_state_intercept,
            );
            a_next.axpy(v_t / cache.f_steady, &cache.k_gain, 1.0);

            if store_full {
                // Filtered state: a_{t|t} = a_{t|t-1} + (v_t / F_inf) * pz_inf
                a_filtered.copy_from(&a);
                a_filtered.axpy(v_t / cache.f_steady, &cache.pz_inf, 1.0);
            }

            std::mem::swap(&mut a, &mut a_next);

            if t >= burn {
                sum_log_f += cache.log_f_steady;
                sum_v2_f += v_t * v_t / cache.f_steady;
            }
        } else {
            // ---- NON-CONVERGED PATH: Dense or Sparse strategy ----
            let v_t = compute_innovation(endog[t], &sparse_z, &a, d_t);
            innovations.push(v_t);

            let f_t = strategy.compute_pz_and_f(&p, z, &sparse_z, &sparse_t, &mut pz);
            if store_full {
                innovation_vars.push(f_t);
            }

            if f_t > 0.0 {
                let f_inv = 1.0 / f_t;

                // State update: a = a + (v_t / F_t) * pz
                a.axpy(v_t * f_inv, &pz, 1.0);

                // Covariance update: P = P - (1/F_t) * pz * pz'  (Joseph form)
                p.ger(-f_inv, &pz, &pz, 1.0);

                if store_full {
                    a_filtered.copy_from(&a);
                    p_filtered.copy_from(&p);
                }

                // Predict state: a_next = T * a_updated + c_t
                strategy.predict_state(
                    t_mat,
                    &sparse_t,
                    &a,
                    &mut a_next,
                    &ss.state_intercept,
                    t,
                    k,
                    has_state_intercept,
                );

                // Covariance prediction: T*P*T' + RQR' (sparse path when T is sparse).
                strategy.predict_cov(t_mat, &t_mat_t, &sparse_t, &mut p, &rqr, &mut temp_kk);

                std::mem::swap(&mut a, &mut a_next);

                if t >= burn {
                    sum_log_f += f_t.ln();
                    sum_v2_f += v_t * v_t * f_inv;
                }

                // --- Steady-state convergence check (pz-vector based) ---
                if t >= burn + STEADY_STATE_MIN_STEPS {
                    // Compute pz from the PREDICTED P (already updated above)
                    pz.gemv(1.0, &p, z, 0.0);

                    if check_convergence(&pz, &pz_prev, &mut consec_count) {
                        let f_steady = z.dot(&pz);
                        // Guard: only adopt steady-state if F_inf is positive and finite
                        if f_steady > 0.0 && f_steady.is_finite() {
                            let log_f_steady = f_steady.ln();
                            let pz_inf = pz.clone();

                            // K_inf = T * pz_inf
                            let mut k_gain = DVector::<f64>::zeros(k);
                            strategy.compute_k_gain(t_mat, &sparse_t, &pz, &mut k_gain);

                            ss_cache = Some(SteadyStateCache {
                                k_gain,
                                f_steady,
                                log_f_steady,
                                pz_inf,
                            });
                        }
                        // else: pz converged but F_inf invalid — stay on non-converged path
                    }
                    pz_prev.copy_from(&pz);
                }
            } else if t >= burn {
                // SAFEGUARD (numerical wall at parameter boundary).
                //
                // F_t <= 0 means the predicted innovation variance went non-
                // positive — typically because the model is at a near-non-
                // stationary boundary (AR roots ≈ unit circle). Aborting here
                // hides the entire likelihood surface beyond the boundary from
                // the optimizer: L-BFGS-B sees only a hard wall (NaN return)
                // and backs off, never reaching the genuine local maxima that
                // statsmodels finds at e.g. ar1 ≈ 1.04.
                //
                // Instead, treat the offending observation as having huge
                // variance: contribute a large `log F_t` and tiny `v²/F_t` to
                // the likelihood. The resulting LL is bad (so the optimizer
                // gets a clear "worse" signal and shrinks its step), but is
                // finite and smooth in the parameters, so gradient descent can
                // still steer back into a stable basin.
                if store_full {
                    a_filtered.copy_from(&a);
                    p_filtered.copy_from(&p);
                }
                let f_safe: f64 = crate::optimizer::KF_FT_FALLBACK_VARIANCE;
                sum_log_f += f_safe.ln();
                sum_v2_f += v_t * v_t / f_safe;

                // Skip the Kalman state/cov update (we have no valid F_t to
                // weight the innovation by) and just predict forward.
                strategy.predict_state(
                    t_mat,
                    &sparse_t,
                    &a,
                    &mut a_next,
                    &ss.state_intercept,
                    t,
                    k,
                    has_state_intercept,
                );
                strategy.predict_cov(t_mat, &t_mat_t, &sparse_t, &mut p, &rqr, &mut temp_kk);
                std::mem::swap(&mut a, &mut a_next);
            } else {
                // F_t <= 0 during burn-in: skip update, predict from current state
                if store_full {
                    a_filtered.copy_from(&a);
                    p_filtered.copy_from(&p);
                }

                strategy.predict_state(
                    t_mat,
                    &sparse_t,
                    &a,
                    &mut a_next,
                    &ss.state_intercept,
                    t,
                    k,
                    has_state_intercept,
                );
                strategy.predict_cov(t_mat, &t_mat_t, &sparse_t, &mut p, &rqr, &mut temp_kk);
                std::mem::swap(&mut a, &mut a_next);
            }
        }
    }

    // Compute log-likelihood
    if !sum_log_f.is_finite() || !sum_v2_f.is_finite() {
        return Err(SarimaxError::DataError(
            "non-finite Kalman statistics encountered (possible NaN/Inf in inputs)".into(),
        ));
    }

    if n_eff == 0 {
        return Err(SarimaxError::DataError(
            "no effective observations after burn-in exclusion (n_eff=0)".into(),
        ));
    }

    let (loglike, scale) = if concentrate_scale {
        let sigma2_hat = sum_v2_f / n_eff as f64;
        let sigma2_safe = sigma2_hat.max(1e-300);
        let ll = -0.5 * (n_eff as f64) * (2.0 * std::f64::consts::PI).ln()
            - 0.5 * (n_eff as f64) * sigma2_safe.ln()
            - 0.5 * (n_eff as f64)
            - 0.5 * sum_log_f;
        (ll, sigma2_hat)
    } else {
        let ll = -0.5 * (n_eff as f64) * (2.0 * std::f64::consts::PI).ln()
            - 0.5 * sum_log_f
            - 0.5 * sum_v2_f;
        let sigma2 = ss.state_cov[(0, 0)];
        (ll, sigma2)
    };
    if !loglike.is_finite() || !scale.is_finite() {
        return Err(SarimaxError::DataError(
            "non-finite loglike/scale produced by Kalman filter".into(),
        ));
    }

    Ok(KalmanFilterOutput {
        loglike,
        scale,
        innovations,
        innovation_vars,
        n_obs_effective: n_eff,
        filtered_state: a_filtered,
        filtered_cov: p_filtered,
        predicted_state: a,
        predicted_cov: p,
    })
}

// ---------------------------------------------------------------------------
// Public API (unchanged signatures)
// ---------------------------------------------------------------------------

/// Compute the (optionally concentrated) log-likelihood via the Kalman filter.
///
/// Uses steady-state acceleration: once the predicted covariance P converges,
/// the Kalman gain is cached and O(k^3) covariance updates are skipped.
pub fn kalman_loglike(
    endog: &[f64],
    ss: &StateSpace,
    init: &KalmanInit,
    concentrate_scale: bool,
) -> Result<KalmanOutput> {
    let fo = kalman_core(endog, ss, init, concentrate_scale, false)?;
    Ok(KalmanOutput {
        loglike: fo.loglike,
        scale: fo.scale,
        innovations: fo.innovations,
        n_obs_effective: fo.n_obs_effective,
    })
}

/// Run the Kalman filter and return full output including final state.
///
/// Uses steady-state acceleration for performance. Returns innovation
/// variances and final filtered/predicted state for forecasting.
pub fn kalman_filter(
    endog: &[f64],
    ss: &StateSpace,
    init: &KalmanInit,
    concentrate_scale: bool,
) -> Result<KalmanFilterOutput> {
    kalman_core(endog, ss, init, concentrate_scale, true)
}

// ---------------------------------------------------------------------------
// Batched Kalman filter
// ---------------------------------------------------------------------------

/// Output of a batched Kalman filter pass over several observation series
/// that share the same state-space matrices T, Z, R, Q (and intercepts).
///
/// The predicted covariance P, the innovation variance F_t, and the
/// (implicit) Kalman gain are identical across series and are therefore
/// computed only once. Only the per-series state mean recursion and
/// innovations differ.
#[derive(Debug, Clone)]
pub struct BatchedKalmanOutput {
    /// Innovation sequence v_t for each series: `innovations[s][t]`.
    pub innovations: Vec<Vec<f64>>,
    /// Innovation variances F_t (shared across all series, before scaling
    /// by sigma2). Same semantics as `KalmanFilterOutput::innovation_vars`.
    pub innovation_vars: Vec<f64>,
    /// Effective number of observations (n - burn). Identical per series.
    pub n_obs_effective: usize,
}

/// Batched Kalman filter for several observation series that share the
/// same state-space matrices.
///
/// This is mathematically equivalent to calling `kalman_filter` once per
/// series, but the covariance prediction/update (the expensive O(k^2)
/// per-step work) is performed only once and amortised across all series.
/// Only the cheap O(k) state-mean recursion is repeated per series.
///
/// # Precondition
///
/// All observation slices in `observations` must have the same length and
/// must be compatible with `ss` (i.e. equivalent to what would be passed
/// individually to `kalman_filter` with this same `ss`). The state-space
/// matrices `T, Z, R, Q` and intercepts `c_t, d_t` must be identical for
/// every series — which is exactly the case in profile-likelihood code
/// where `ss` is built from the same nonlinear parameters with `exog=None`
/// and zeroed trend coefficients.
///
/// # Returns
///
/// `BatchedKalmanOutput` with per-series innovations and a single shared
/// innovation-variance vector. `n_obs_effective` matches the single-series
/// implementation because the burn-in count is covariance-driven.
pub fn kalman_filter_batched(
    observations: &[&[f64]],
    ss: &StateSpace,
    init: &KalmanInit,
    _concentrate_scale: bool,
) -> Result<BatchedKalmanOutput> {
    if observations.is_empty() {
        return Err(SarimaxError::DataError(
            "kalman_filter_batched: observations must contain at least one series".into(),
        ));
    }
    let n = observations[0].len();
    for (s, obs) in observations.iter().enumerate() {
        if obs.len() != n {
            return Err(SarimaxError::DataError(format!(
                "kalman_filter_batched: series {} has length {} but series 0 has length {}",
                s,
                obs.len(),
                n
            )));
        }
    }

    let k = ss.k_states;
    let burn = init.loglikelihood_burn;
    if n <= burn {
        return Err(SarimaxError::DataError(format!(
            "Not enough observations: n={} <= burn={}",
            n, burn
        )));
    }
    let n_eff = n - burn;

    let n_series = observations.len();

    // Per-series predicted state means a^{(s)}_{t|t-1}.
    let mut a_states: Vec<DVector<f64>> = (0..n_series)
        .map(|_| init.initial_state.clone())
        .collect();
    // Shared predicted covariance P_{t|t-1}.
    let mut p = init.initial_state_cov.clone();

    let t_mat = &ss.transition;
    let z = &ss.design;
    let r_mat = &ss.selection;
    let q_mat = &ss.state_cov;

    let rqr = r_mat * q_mat * r_mat.transpose();
    let t_mat_t = t_mat.transpose();
    let has_state_intercept = ss.state_intercept.len() == n * k;

    let sparse_z = SparseZ::from_dense(z, k);
    let sparse_t = SparseT::from_dense(t_mat, k);

    let strategy = if sparse_t.is_sparse() {
        KalmanStrategy::Sparse
    } else {
        KalmanStrategy::Dense
    };

    // Per-series innovation buffers.
    let mut innovations: Vec<Vec<f64>> = (0..n_series).map(|_| Vec::with_capacity(n)).collect();
    let mut innovation_vars: Vec<f64> = Vec::with_capacity(n);

    // Scratch buffers (shared across series at each time step).
    let mut pz = DVector::<f64>::zeros(k);
    let mut a_next = DVector::<f64>::zeros(k);
    let mut temp_kk = DMatrix::<f64>::zeros(k, k);

    // Steady-state cache (identical for every series because it depends
    // only on T, Z, R, Q, P_0).
    let mut ss_cache: Option<SteadyStateCache> = init
        .steady_state
        .as_ref()
        .filter(|_| burn == 0)
        .map(|KalmanSteadyState { k_gain, f_steady, log_f_steady, pz_inf }| {
            SteadyStateCache {
                k_gain: k_gain.clone(),
                f_steady: *f_steady,
                log_f_steady: *log_f_steady,
                pz_inf: pz_inf.clone(),
            }
        });
    let mut pz_prev = DVector::<f64>::zeros(k);
    let mut consec_count = 0_usize;

    for t in 0..n {
        let d_t = if t < ss.obs_intercept.len() {
            ss.obs_intercept[t]
        } else {
            0.0
        };

        if let Some(ref cache) = ss_cache {
            // ---- STEADY-STATE PATH: shared cache, per-series mean update ----
            debug_assert!(
                cache.f_steady > 0.0 && cache.f_steady.is_finite(),
                "steady-state F must be positive and finite, got {}",
                cache.f_steady
            );
            innovation_vars.push(cache.f_steady);

            for s in 0..n_series {
                let v_t = compute_innovation(observations[s][t], &sparse_z, &a_states[s], d_t);
                innovations[s].push(v_t);

                // a_next = T * a + c_t
                strategy.predict_state(
                    t_mat, &sparse_t, &a_states[s], &mut a_next,
                    &ss.state_intercept, t, k, has_state_intercept,
                );
                // a_next += (v_t / F_inf) * K_inf
                a_next.axpy(v_t / cache.f_steady, &cache.k_gain, 1.0);

                std::mem::swap(&mut a_states[s], &mut a_next);
            }
            // Covariance is frozen; nothing more to do.
        } else {
            // ---- NON-CONVERGED PATH: compute shared F_t / pz once ----
            let f_t = strategy.compute_pz_and_f(&p, z, &sparse_z, &sparse_t, &mut pz);
            innovation_vars.push(f_t);

            if f_t > 0.0 {
                let f_inv = 1.0 / f_t;

                // Per-series mean update + predict.
                for s in 0..n_series {
                    let v_t = compute_innovation(observations[s][t], &sparse_z, &a_states[s], d_t);
                    innovations[s].push(v_t);

                    // a = a + (v_t / F_t) * pz
                    a_states[s].axpy(v_t * f_inv, &pz, 1.0);
                    // a_next = T * a + c_t
                    strategy.predict_state(
                        t_mat, &sparse_t, &a_states[s], &mut a_next,
                        &ss.state_intercept, t, k, has_state_intercept,
                    );
                    std::mem::swap(&mut a_states[s], &mut a_next);
                }

                // Shared covariance update (Joseph form) and prediction.
                p.ger(-f_inv, &pz, &pz, 1.0);
                strategy.predict_cov(t_mat, &t_mat_t, &sparse_t, &mut p, &rqr, &mut temp_kk);

                // Steady-state convergence check (covariance only).
                if t >= burn + STEADY_STATE_MIN_STEPS {
                    pz.gemv(1.0, &p, z, 0.0);
                    if check_convergence(&pz, &pz_prev, &mut consec_count) {
                        let f_steady = z.dot(&pz);
                        if f_steady > 0.0 && f_steady.is_finite() {
                            let log_f_steady = f_steady.ln();
                            let pz_inf = pz.clone();
                            let mut k_gain = DVector::<f64>::zeros(k);
                            strategy.compute_k_gain(t_mat, &sparse_t, &pz, &mut k_gain);
                            ss_cache = Some(SteadyStateCache {
                                k_gain,
                                f_steady,
                                log_f_steady,
                                pz_inf,
                            });
                        }
                    }
                    pz_prev.copy_from(&pz);
                }
            } else {
                // F_t <= 0: replicate single-series behaviour — skip the
                // Kalman update and just predict forward. Per-series state
                // means still need to be recorded so downstream GLS sees
                // innovations of identical length across t and s.
                for s in 0..n_series {
                    let v_t = compute_innovation(observations[s][t], &sparse_z, &a_states[s], d_t);
                    innovations[s].push(v_t);

                    strategy.predict_state(
                        t_mat, &sparse_t, &a_states[s], &mut a_next,
                        &ss.state_intercept, t, k, has_state_intercept,
                    );
                    std::mem::swap(&mut a_states[s], &mut a_next);
                }
                // Predict shared covariance forward.
                strategy.predict_cov(t_mat, &t_mat_t, &sparse_t, &mut p, &rqr, &mut temp_kk);
            }
        }
    }

    Ok(BatchedKalmanOutput {
        innovations,
        innovation_vars,
        n_obs_effective: n_eff,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::initialization::KalmanInit;
    use crate::params::SarimaxParams;
    use crate::state_space::StateSpace;
    use crate::test_helpers::{load_fixtures, make_config, make_params};
    use crate::types::{SarimaxConfig, SarimaxOrder, Trend};

    fn run_kalman_test(fixture_key: &str, p: usize, d: usize, q: usize, tol: f64) {
        let fixtures = load_fixtures();
        let case = &fixtures[fixture_key];

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
        let expected_loglike = case["loglike"].as_f64().unwrap();
        let expected_scale = case["scale"].as_f64().unwrap();

        let config = make_config(p, d, q);
        let ar_coeffs = &params_vec[..p];
        let ma_coeffs = &params_vec[p..p + q];
        let params = make_params(ar_coeffs, ma_coeffs);

        let ss = StateSpace::new(&config, &params, &data, None).unwrap();
        let init = KalmanInit::approximate_diffuse(ss.k_states, KalmanInit::default_kappa());

        let output = kalman_loglike(&data, &ss, &init, true).unwrap();

        let loglike_err = (output.loglike - expected_loglike).abs();
        assert!(
            loglike_err < tol,
            "{}: loglike mismatch: got {}, expected {}, err={}",
            fixture_key,
            output.loglike,
            expected_loglike,
            loglike_err
        );

        let scale_err = (output.scale - expected_scale).abs();
        assert!(
            scale_err < tol,
            "{}: scale mismatch: got {}, expected {}, err={}",
            fixture_key,
            output.scale,
            expected_scale,
            scale_err
        );
    }

    #[test]
    fn test_ar1_loglike_vs_statsmodels() {
        run_kalman_test("ar1", 1, 0, 0, 1e-6);
    }

    #[test]
    fn test_arma11_loglike_vs_statsmodels() {
        run_kalman_test("arma11", 1, 0, 1, 1e-6);
    }

    #[test]
    fn test_arima111_loglike_vs_statsmodels() {
        run_kalman_test("arima111", 1, 1, 1, 1e-6);
    }

    #[test]
    fn test_innovations_length() {
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let n = data.len();

        let config = make_config(1, 0, 0);
        let params = make_params(&[0.6527425084139002], &[]);
        let ss = StateSpace::new(&config, &params, &data, None).unwrap();
        let init = KalmanInit::approximate_diffuse(ss.k_states, 1e6);

        let output = kalman_loglike(&data, &ss, &init, true).unwrap();
        assert_eq!(output.innovations.len(), n);
        assert_eq!(output.n_obs_effective, n - 1); // burn = k_states = 1
    }

    // ---- Seasonal Kalman tests ----

    fn run_seasonal_kalman_test(
        fixture_key: &str,
        p: usize,
        d: usize,
        q: usize,
        pp: usize,
        dd: usize,
        qq: usize,
        s: usize,
        tol: f64,
    ) {
        let fixtures = load_fixtures();
        let case = &fixtures[fixture_key];

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
        let expected_loglike = case["loglike"].as_f64().unwrap();
        let expected_scale = case["scale"].as_f64().unwrap();

        let config = SarimaxConfig {
            order: SarimaxOrder::new(p, d, q, pp, dd, qq, s),
            n_exog: 0,
            trend: Trend::None,
            enforce_stationarity: false,
            enforce_invertibility: false,
            concentrate_scale: true,
            simple_differencing: false,
            measurement_error: false,
        };

        let mut i = 0;
        let ar = &params_vec[i..i + p];
        i += p;
        let ma = &params_vec[i..i + q];
        i += q;
        let sar = &params_vec[i..i + pp];
        i += pp;
        let sma = &params_vec[i..i + qq];

        let params = SarimaxParams {
            trend_coeffs: vec![],
            exog_coeffs: vec![],
            ar_coeffs: ar.to_vec(),
            ma_coeffs: ma.to_vec(),
            sar_coeffs: sar.to_vec(),
            sma_coeffs: sma.to_vec(),
            sigma2: None,
        };

        let ss = StateSpace::new(&config, &params, &data, None).unwrap();
        let init = KalmanInit::approximate_diffuse(ss.k_states, KalmanInit::default_kappa());

        let output = kalman_loglike(&data, &ss, &init, true).unwrap();

        let loglike_err = (output.loglike - expected_loglike).abs();
        assert!(
            loglike_err < tol,
            "{}: loglike mismatch: got {}, expected {}, err={}",
            fixture_key,
            output.loglike,
            expected_loglike,
            loglike_err
        );

        let scale_err = (output.scale - expected_scale).abs();
        assert!(
            scale_err < tol,
            "{}: scale mismatch: got {}, expected {}, err={}",
            fixture_key,
            output.scale,
            expected_scale,
            scale_err
        );
    }

    #[test]
    fn test_sarima_100_100_4_loglike() {
        run_seasonal_kalman_test("sarima_100_100_4", 1, 0, 0, 1, 0, 0, 4, 1e-6);
    }

    #[test]
    fn test_sarima_111_111_12_loglike() {
        run_seasonal_kalman_test("sarima_111_111_12", 1, 1, 1, 1, 1, 1, 12, 1e-6);
    }

    // ---- kalman_filter tests ----

    #[test]
    fn test_kalman_filter_matches_loglike() {
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
        let ss = StateSpace::new(&config, &params, &data, None).unwrap();
        let init = KalmanInit::approximate_diffuse(ss.k_states, 1e6);

        let lo = kalman_loglike(&data, &ss, &init, true).unwrap();
        let fo = kalman_filter(&data, &ss, &init, true).unwrap();

        assert!(
            (lo.loglike - fo.loglike).abs() < 1e-12,
            "loglike mismatch: {} vs {}",
            lo.loglike,
            fo.loglike
        );
        assert!((lo.scale - fo.scale).abs() < 1e-12);
        assert_eq!(lo.innovations.len(), fo.innovations.len());
        for (a, b) in lo.innovations.iter().zip(fo.innovations.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_kalman_filter_innovation_vars_positive() {
        let fixtures = load_fixtures();
        let case = &fixtures["arma11"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        let config = make_config(1, 0, 1);
        let params_vec: Vec<f64> = case["params"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let params = make_params(&params_vec[..1], &params_vec[1..2]);
        let ss = StateSpace::new(&config, &params, &data, None).unwrap();
        let init = KalmanInit::approximate_diffuse(ss.k_states, 1e6);

        let fo = kalman_filter(&data, &ss, &init, true).unwrap();
        assert_eq!(fo.innovation_vars.len(), data.len());
        for &f in &fo.innovation_vars[fo.n_obs_effective..] {
            assert!(f >= 0.0, "F_t should be non-negative, got {}", f);
        }
    }

    #[test]
    fn test_kalman_filter_state_dimensions() {
        let fixtures = load_fixtures();
        let case = &fixtures["arma11"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        let config = make_config(1, 0, 1);
        let params_vec: Vec<f64> = case["params"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let params = make_params(&params_vec[..1], &params_vec[1..2]);
        let ss = StateSpace::new(&config, &params, &data, None).unwrap();
        let init = KalmanInit::approximate_diffuse(ss.k_states, 1e6);

        let fo = kalman_filter(&data, &ss, &init, true).unwrap();
        assert_eq!(fo.filtered_state.len(), ss.k_states);
        assert_eq!(fo.predicted_state.len(), ss.k_states);
        assert_eq!(fo.filtered_cov.nrows(), ss.k_states);
        assert_eq!(fo.filtered_cov.ncols(), ss.k_states);
        assert_eq!(fo.predicted_cov.nrows(), ss.k_states);
        assert_eq!(fo.predicted_cov.ncols(), ss.k_states);
    }

    // ---- kalman_filter_batched tests ----

    /// Generate small synthetic (y, x_1, x_2) test data.
    fn make_batched_fixture() -> (Vec<f64>, Vec<Vec<f64>>) {
        let n = 80usize;
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                10.0 + (t * 0.41).sin() + 0.05 * t + 0.3 * (t * 0.13).cos()
            })
            .collect();
        let x1: Vec<f64> = (0..n)
            .map(|i| 3.0 + (i as f64 * 0.22).cos() + 0.02 * i as f64)
            .collect();
        let x2: Vec<f64> = (0..n)
            .map(|i| -1.0 + (i as f64 * 0.31).sin() - 0.01 * i as f64)
            .collect();
        (y, vec![x1, x2])
    }

    #[test]
    fn test_batched_matches_per_series_filter() {
        // SARIMA(1,0,1): k_states=2 with simple params, exog handled externally.
        let config = make_config(1, 0, 1);
        let params = make_params(&[0.5], &[0.3]);

        let (y, exogs) = make_batched_fixture();

        let ss = StateSpace::new(&config, &params, &y, None).unwrap();
        let init = KalmanInit::from_config_default(&ss, &config);

        // Single-series ground-truth: y and each x_j.
        let y_single = kalman_filter(&y, &ss, &init, false).unwrap();
        let x_singles: Vec<KalmanFilterOutput> = exogs
            .iter()
            .map(|col| kalman_filter(col, &ss, &init, false).unwrap())
            .collect();

        // Batched call: [y, x_1, x_2].
        let mut obs_refs: Vec<&[f64]> = Vec::with_capacity(1 + exogs.len());
        obs_refs.push(y.as_slice());
        for col in &exogs {
            obs_refs.push(col.as_slice());
        }
        let batched = kalman_filter_batched(&obs_refs, &ss, &init, false).unwrap();

        // n_obs_effective must match.
        assert_eq!(batched.n_obs_effective, y_single.n_obs_effective);

        // innovation_vars (shared) must match the single-series F_t.
        assert_eq!(batched.innovation_vars.len(), y_single.innovation_vars.len());
        for (t, (b, y)) in batched
            .innovation_vars
            .iter()
            .zip(y_single.innovation_vars.iter())
            .enumerate()
        {
            assert!(
                (b - y).abs() < 1e-10,
                "F_t mismatch at t={}: batched={}, single={}",
                t, b, y
            );
        }

        // innovations[0] must match y's single-series innovations.
        assert_eq!(batched.innovations[0].len(), y_single.innovations.len());
        for (t, (b, y)) in batched.innovations[0]
            .iter()
            .zip(y_single.innovations.iter())
            .enumerate()
        {
            assert!(
                (b - y).abs() < 1e-10,
                "y innovation mismatch at t={}: batched={}, single={}",
                t, b, y
            );
        }

        // innovations[s+1] must match x_s single-series innovations.
        for (j, xs) in x_singles.iter().enumerate() {
            let b_inn = &batched.innovations[j + 1];
            assert_eq!(b_inn.len(), xs.innovations.len());
            for (t, (b, x)) in b_inn.iter().zip(xs.innovations.iter()).enumerate() {
                assert!(
                    (b - x).abs() < 1e-10,
                    "x_{} innovation mismatch at t={}: batched={}, single={}",
                    j, t, b, x
                );
            }
        }
    }

    #[test]
    fn test_batched_length_mismatch_errors() {
        let config = make_config(1, 0, 0);
        let params = make_params(&[0.4], &[]);
        let (y, _exogs) = make_batched_fixture();
        let ss = StateSpace::new(&config, &params, &y, None).unwrap();
        let init = KalmanInit::from_config_default(&ss, &config);

        let short: Vec<f64> = vec![0.0; y.len() - 1];
        let obs_refs: Vec<&[f64]> = vec![y.as_slice(), short.as_slice()];
        assert!(kalman_filter_batched(&obs_refs, &ss, &init, false).is_err());
    }
}
