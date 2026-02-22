use nalgebra::{DMatrix, DVector};

use crate::error::{Result, SarimaxError};
use crate::initialization::KalmanInit;
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
/// incorrect cached K_∞ = T*pz/F. Error analysis shows the loglike
/// error from a converged-but-imprecise K_∞ is O(ε * n²) due to
/// compounding through the state recursion (unit-root states integrate
/// the error), so ε must be tight.
///
/// At 1e-9, loglike error stays below ~1e-5 for n=1000.
/// For models where pz converges slowly, steady-state won't trigger
/// and the sparse T·P·T' path provides the primary speedup instead.
const STEADY_STATE_TOL: f64 = 1e-9;

/// Minimum steps past burn-in before checking for convergence.
const STEADY_STATE_MIN_STEPS: usize = 5;

/// Number of consecutive converged pz values required.
const STEADY_STATE_CONSEC: usize = 3;

// ---------------------------------------------------------------------------
// Core Kalman filter with steady-state acceleration
// ---------------------------------------------------------------------------

/// Unified Kalman filter core with optional steady-state detection.
///
/// When `store_full=false`, skips storing innovation_vars and filtered state
/// (used by `kalman_loglike`). When `store_full=true`, stores everything
/// needed for forecasting and diagnostics (used by `kalman_filter`).
///
/// **Steady-state optimization**: For time-invariant systems, the predicted
/// covariance P_{t+1|t} converges to a steady-state P_∞. Once detected,
/// the Kalman gain K_∞ is cached and the expensive O(k³) covariance
/// prediction step is skipped for all remaining time steps.
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

    // Sparse representation of T for O(nnz) operations.
    // SARIMA companion matrices are very sparse (e.g. 31/729 = 4% for k=27).
    // This enables sparse T·a (O(nnz)), T·P (O(nnz×k)), and T·P·T' (O(nnz×k)).
    let sparse_t: Vec<(usize, usize, f64)> = {
        let mut entries = Vec::new();
        for i in 0..k {
            for j in 0..k {
                let v = t_mat[(i, j)];
                if v != 0.0 {
                    entries.push((i, j, v));
                }
            }
        }
        entries
    };
    // Use sparse path when T is significantly sparse (< 50% density).
    let use_sparse = sparse_t.len() < k * k / 2;

    // Sparse Z for O(nnz_z) dot product (e.g. 3 non-zeros for SARIMA k=27).
    let sparse_z: Vec<(usize, f64)> = {
        let mut entries = Vec::new();
        for i in 0..k {
            let v = z[i];
            if v != 0.0 {
                entries.push((i, v));
            }
        }
        entries
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

    // Steady-state detection buffers (pz-vector based)
    let mut converged = false;
    let mut k_gain = DVector::<f64>::zeros(k); // K_∞ = T * P_∞ * Z
    let mut f_steady = 0.0;
    let mut log_f_steady = 0.0;
    let mut pz_prev = DVector::<f64>::zeros(k);
    let mut pz_inf = DVector::<f64>::zeros(k); // cached pz_∞ = P_∞ * Z at convergence
    let mut consec_count = 0_usize;

    for t in 0..n {
        // --- Innovation ---
        let d_t = if t < ss.obs_intercept.len() {
            ss.obs_intercept[t]
        } else {
            0.0
        };

        if converged {
            // ---- PATH 1: Steady-state — O(nnz) per step ----
            // P has converged to P_∞, so K_∞ and F_∞ are constant.
            // Only the state mean recursion is computed; covariance is frozen.
            let a_slice = a.as_slice();
            let za: f64 = sparse_z.iter().map(|&(i, v)| v * a_slice[i]).sum();
            let v_t = endog[t] - za - d_t;
            innovations.push(v_t);
            if store_full {
                innovation_vars.push(f_steady);
            }

            // Predict state: a_next = T_sparse * a + K_∞ * (v_t / F_∞)
            // Using sparse T: O(nnz) instead of dense gemv O(k²)
            let a_slice = a.as_slice();
            let a_next_slice = a_next.as_mut_slice();
            for v in a_next_slice.iter_mut() {
                *v = 0.0;
            }
            for &(i, j, val) in &sparse_t {
                a_next_slice[i] += val * a_slice[j];
            }
            let scale_v = v_t / f_steady;
            let k_slice = k_gain.as_slice();
            for i in 0..k {
                a_next_slice[i] += scale_v * k_slice[i];
            }
            if has_state_intercept {
                for i in 0..k {
                    a_next[i] += ss.state_intercept[t * k + i];
                }
            }

            if store_full {
                // Filtered state: a_{t|t} = a_{t|t-1} + (v_t / F_∞) * pz_∞
                a_filtered.copy_from(&a);
                let scale_v = v_t / f_steady;
                let af_s = a_filtered.as_mut_slice();
                let pz_s = pz_inf.as_slice();
                for i in 0..k {
                    af_s[i] += scale_v * pz_s[i];
                }
            }

            std::mem::swap(&mut a, &mut a_next);

            if t >= burn {
                sum_log_f += log_f_steady;
                sum_v2_f += v_t * v_t / f_steady;
            }
        } else if use_sparse {
            // ---- PATH 2: Sparse Kalman — O(nnz×k) per step ----
            // Used when T density < 50%. For SARIMA(1,1,1)(1,1,1,12) with k=27,
            // T has ~31/729 = 4% non-zeros, giving ~23× speedup over dense gemm.
            let v_t = endog[t] - z.dot(&a) - d_t;
            innovations.push(v_t);

            // pz = P * z (sparse Z: O(nnz_z × k) instead of O(k²))
            {
                let pz_s = pz.as_mut_slice();
                let p_data = p.as_slice(); // column-major
                for v in pz_s.iter_mut() {
                    *v = 0.0;
                }
                for &(zi, zv) in &sparse_z {
                    // P[:, zi] * zv → accumulate into pz
                    let col_start = zi * k;
                    for r in 0..k {
                        pz_s[r] += zv * p_data[col_start + r];
                    }
                }
            }
            let f_t: f64 = sparse_z.iter().map(|&(i, v)| v * pz[i]).sum();
            if store_full {
                innovation_vars.push(f_t);
            }

            if f_t > 0.0 {
                let f_inv = 1.0 / f_t;

                // State update: a = a + (v_t / F_t) * pz
                a.axpy(v_t * f_inv, &pz, 1.0);

                // Covariance update: P = P - (1/F_t) * pz * pz'
                p.ger(-f_inv, &pz, &pz, 1.0);

                if store_full {
                    a_filtered.copy_from(&a);
                    p_filtered.copy_from(&p);
                }

                // Predict state: a_next = T_sparse * a (O(nnz))
                {
                    let a_s = a.as_slice();
                    let an_s = a_next.as_mut_slice();
                    for v in an_s.iter_mut() {
                        *v = 0.0;
                    }
                    for &(i, j, val) in &sparse_t {
                        an_s[i] += val * a_s[j];
                    }
                }
                if has_state_intercept {
                    let an_s = a_next.as_mut_slice();
                    let base = t * k;
                    for i in 0..k {
                        an_s[i] += ss.state_intercept[base + i];
                    }
                }

                // Predict covariance: P = T_sparse * P * T_sparse' + RQR'
                // Step 1: temp_kk = T_sparse * P  — O(nnz × k)
                {
                    let p_data = p.as_slice(); // column-major
                    let tmp = temp_kk.as_mut_slice();
                    for v in tmp.iter_mut() {
                        *v = 0.0;
                    }
                    // temp_kk[i, j] += T[i,l] * P[l,j]
                    // In column-major: temp_kk[i + j*k] += val * p_data[l + j*k]
                    for &(i, l, val) in &sparse_t {
                        for j in 0..k {
                            tmp[i + j * k] += val * p_data[l + j * k];
                        }
                    }
                }
                // Step 2: P = temp_kk * T' + RQR'  — O(nnz × k)
                // P[i,j] = sum_l temp_kk[i,l] * T[j,l]  (since T'[l,j] = T[j,l])
                {
                    let tmp = temp_kk.as_slice();
                    let p_data = p.as_mut_slice();
                    let rqr_data = rqr.as_slice();
                    // Start with RQR'
                    p_data.copy_from_slice(rqr_data);
                    // Accumulate: P[i,j] += temp_kk[i,l] * T[j,l]
                    // Iterate over sparse T entries as (j, l, val)
                    for &(j, l, val) in &sparse_t {
                        // Add val * temp_kk[:, l] to P[:, j]
                        let col_l = l * k;
                        let col_j = j * k;
                        for i in 0..k {
                            p_data[col_j + i] += val * tmp[col_l + i];
                        }
                    }
                }

                std::mem::swap(&mut a, &mut a_next);

                if t >= burn {
                    sum_log_f += f_t.ln();
                    sum_v2_f += v_t * v_t * f_inv;
                }

                // --- Steady-state convergence check (pz-vector based) ---
                // We check pz = P*Z convergence instead of scalar F_t = Z'*pz
                // because F_t can converge while pz still changes in directions
                // orthogonal to Z, causing incorrect cached K_∞.
                if t >= burn + STEADY_STATE_MIN_STEPS {
                    // Compute pz from the PREDICTED P (already updated above)
                    pz.gemv(1.0, &p, z, 0.0);
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
                        consec_count += 1;
                        if consec_count >= STEADY_STATE_CONSEC {
                            converged = true;
                            f_steady = z.dot(&pz);
                            log_f_steady = f_steady.ln();
                            pz_inf.copy_from(&pz); // cache pz_∞ for filtered state
                                                   // K_∞ = T * pz_∞
                            {
                                let pz_s = pz.as_slice();
                                let kg_s = k_gain.as_mut_slice();
                                for v in kg_s.iter_mut() {
                                    *v = 0.0;
                                }
                                for &(i, j, val) in &sparse_t {
                                    kg_s[i] += val * pz_s[j];
                                }
                            }
                        }
                    } else {
                        consec_count = 0;
                    }
                    pz_prev.copy_from(&pz);
                }
            } else if t >= burn {
                return Err(SarimaxError::DataError(format!(
                    "innovation variance F_t <= 0 at t={} (F_t={}); \
                     model parameters may be numerically unstable",
                    t, f_t
                )));
            } else {
                // F_t <= 0 during burn-in: skip update, predict from current state
                if store_full {
                    a_filtered.copy_from(&a);
                    p_filtered.copy_from(&p);
                }

                {
                    let a_s = a.as_slice();
                    let an_s = a_next.as_mut_slice();
                    for v in an_s.iter_mut() {
                        *v = 0.0;
                    }
                    for &(i, j, val) in &sparse_t {
                        an_s[i] += val * a_s[j];
                    }
                }
                if has_state_intercept {
                    let an_s = a_next.as_mut_slice();
                    let base = t * k;
                    for i in 0..k {
                        an_s[i] += ss.state_intercept[base + i];
                    }
                }
                {
                    let p_data = p.as_slice();
                    let tmp = temp_kk.as_mut_slice();
                    for v in tmp.iter_mut() {
                        *v = 0.0;
                    }
                    for &(i, l, val) in &sparse_t {
                        for j in 0..k {
                            tmp[i + j * k] += val * p_data[l + j * k];
                        }
                    }
                }
                {
                    let tmp = temp_kk.as_slice();
                    let p_data = p.as_mut_slice();
                    let rqr_data = rqr.as_slice();
                    p_data.copy_from_slice(rqr_data);
                    for &(j, l, val) in &sparse_t {
                        let col_l = l * k;
                        let col_j = j * k;
                        for i in 0..k {
                            p_data[col_j + i] += val * tmp[col_l + i];
                        }
                    }
                }
                std::mem::swap(&mut a, &mut a_next);
            }
        } else {
            // ---- PATH 3: Dense Kalman — O(k³) per step ----
            // Standard nalgebra gemm for small state dimensions or dense T.
            // This is the textbook Kalman filter implementation.
            let v_t = endog[t] - z.dot(&a) - d_t;
            innovations.push(v_t);

            // pz = P * z
            pz.gemv(1.0, &p, z, 0.0);
            let f_t: f64 = z.dot(&pz);
            if store_full {
                innovation_vars.push(f_t);
            }

            if f_t > 0.0 {
                let f_inv = 1.0 / f_t;

                // State update: a = a + (v_t / F_t) * pz
                a.axpy(v_t * f_inv, &pz, 1.0);

                // Covariance update: P = P - (1/F_t) * pz * pz'
                p.ger(-f_inv, &pz, &pz, 1.0);

                if store_full {
                    a_filtered.copy_from(&a);
                    p_filtered.copy_from(&p);
                }

                // Predict state: a_next = T * a_updated
                a_next.gemv(1.0, t_mat, &a, 0.0);
                if has_state_intercept {
                    for i in 0..k {
                        a_next[i] += ss.state_intercept[t * k + i];
                    }
                }

                // Predict covariance: P = T * P_updated * T' + RQR'
                temp_kk.gemm(1.0, t_mat, &p, 0.0);
                p.gemm(1.0, &temp_kk, &t_mat_t, 0.0);
                p += &rqr;

                std::mem::swap(&mut a, &mut a_next);

                if t >= burn {
                    sum_log_f += f_t.ln();
                    sum_v2_f += v_t * v_t * f_inv;
                }

                // --- Steady-state convergence check (pz-vector based) ---
                if t >= burn + STEADY_STATE_MIN_STEPS {
                    pz.gemv(1.0, &p, z, 0.0);
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
                        consec_count += 1;
                        if consec_count >= STEADY_STATE_CONSEC {
                            converged = true;
                            f_steady = z.dot(&pz);
                            log_f_steady = f_steady.ln();
                            pz_inf.copy_from(&pz); // cache pz_∞ for filtered state
                            k_gain.gemv(1.0, t_mat, &pz, 0.0);
                        }
                    } else {
                        consec_count = 0;
                    }
                    pz_prev.copy_from(&pz);
                }
            } else if t >= burn {
                return Err(SarimaxError::DataError(format!(
                    "innovation variance F_t <= 0 at t={} (F_t={}); \
                     model parameters may be numerically unstable",
                    t, f_t
                )));
            } else {
                // F_t <= 0 during burn-in: skip update, predict from current state
                if store_full {
                    a_filtered.copy_from(&a);
                    p_filtered.copy_from(&p);
                }

                a_next.gemv(1.0, t_mat, &a, 0.0);
                if has_state_intercept {
                    for i in 0..k {
                        a_next[i] += ss.state_intercept[t * k + i];
                    }
                }
                temp_kk.gemm(1.0, t_mat, &p, 0.0);
                p.gemm(1.0, &temp_kk, &t_mat_t, 0.0);
                p += &rqr;
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
/// the Kalman gain is cached and O(k³) covariance updates are skipped.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::initialization::KalmanInit;
    use crate::params::SarimaxParams;
    use crate::state_space::StateSpace;
    use crate::types::{SarimaxConfig, SarimaxOrder, Trend};

    fn load_fixtures() -> serde_json::Value {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/statsmodels_reference.json"
        );
        let data = std::fs::read_to_string(path).expect("fixtures file not found");
        serde_json::from_str(&data).expect("invalid JSON")
    }

    fn make_config(p: usize, d: usize, q: usize) -> SarimaxConfig {
        SarimaxConfig {
            order: SarimaxOrder::new(p, d, q, 0, 0, 0, 0),
            n_exog: 0,
            trend: Trend::None,
            enforce_stationarity: false,
            enforce_invertibility: false,
            concentrate_scale: true,
            simple_differencing: false,
            measurement_error: false,
        }
    }

    fn make_params(ar: &[f64], ma: &[f64]) -> SarimaxParams {
        SarimaxParams {
            trend_coeffs: vec![],
            exog_coeffs: vec![],
            ar_coeffs: ar.to_vec(),
            ma_coeffs: ma.to_vec(),
            sar_coeffs: vec![],
            sma_coeffs: vec![],
            sigma2: None,
        }
    }

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
}
