use nalgebra::{DMatrix, DVector};

use crate::state_space::StateSpace;
use crate::types::SarimaxConfig;

/// Pre-computed steady-state Kalman quantities derived from the DARE solution.
///
/// Populated only when the initial covariance is the exact steady-state P_∞
/// (i.e. DARE succeeded and there are no diffuse states, sd=0). Allows
/// `kalman_core` to activate the steady-state cache immediately at t=0,
/// skipping the O(nnz·k) transient convergence phase entirely.
pub struct KalmanSteadyState {
    /// Cached Kalman gain K_inf = T * P_∞ * z.
    pub k_gain: DVector<f64>,
    /// Innovation variance F_inf = z' * P_∞ * z.
    pub f_steady: f64,
    /// ln(F_inf), pre-computed to avoid repeated log calls.
    pub log_f_steady: f64,
    /// pz_inf = P_∞ * z used in filtered-state correction.
    pub pz_inf: DVector<f64>,
}

/// Kalman filter initial state and covariance.
pub struct KalmanInit {
    /// Initial state vector a_0 (zeros).
    pub initial_state: DVector<f64>,
    /// Initial state covariance P_0.
    pub initial_state_cov: DMatrix<f64>,
    /// Number of initial observations to skip in loglikelihood (burn-in).
    pub loglikelihood_burn: usize,
    /// Pre-computed steady-state (Some only for DARE init with sd=0).
    pub steady_state: Option<KalmanSteadyState>,
}

impl KalmanInit {
    /// Approximate diffuse initialization.
    ///
    /// Used when `enforce_stationarity=false`.
    /// - a_0 = 0
    /// - P_0 = kappa * I_{k_states}
    /// - burn = k_states
    pub fn approximate_diffuse(k_states: usize, kappa: f64) -> Self {
        Self {
            initial_state: DVector::zeros(k_states),
            initial_state_cov: DMatrix::identity(k_states, k_states) * kappa,
            loglikelihood_burn: k_states,
            steady_state: None,
        }
    }

    /// Mixed initialization matching statsmodels when `enforce_stationarity=true`.
    ///
    /// - Diffuse states (d + s*D): P_0 = kappa * I, burn = k_states_diff
    /// - ARMA states: P_0 = P_∞ from discrete Lyapunov equation, no burn
    ///
    /// Falls back to approximate diffuse if Lyapunov solve fails.
    pub fn mixed(ss: &StateSpace, config: &SarimaxConfig, kappa: f64) -> Self {
        let k_states = ss.k_states;
        let sd = config.effective_sd();
        let ko = config.order.k_order();

        if ko == 0 || sd >= k_states {
            return Self::approximate_diffuse(k_states, kappa);
        }

        // Extract ARMA block of transition matrix T[sd..sd+ko, sd..sd+ko]
        let t_arma = ss.transition.view((sd, sd), (ko, ko)).into_owned();

        // Build RQR for ARMA block
        // R_arma = selection[sd..sd+ko, :], Q = state_cov
        let r_arma = ss.selection.view((sd, 0), (ko, ss.k_posdef)).into_owned();
        let rqr_arma = &r_arma * &ss.state_cov * r_arma.transpose();

        // Solve discrete Lyapunov: P_∞ = T * P_∞ * T' + RQR
        match solve_discrete_lyapunov(&t_arma, &rqr_arma) {
            Some(p_inf) => {
                let mut p0 = DMatrix::<f64>::zeros(k_states, k_states);

                // Diffuse part: kappa * I for diff states
                for i in 0..sd {
                    p0[(i, i)] = kappa;
                }

                // Stationary part: P_∞ for ARMA states
                for i in 0..ko {
                    for j in 0..ko {
                        p0[(sd + i, sd + j)] = p_inf[(i, j)];
                    }
                }

                Self {
                    initial_state: DVector::zeros(k_states),
                    initial_state_cov: p0,
                    loglikelihood_burn: sd,
                    steady_state: None,
                }
            }
            None => {
                // Lyapunov solve failed → fall back to approximate diffuse
                Self::approximate_diffuse(k_states, kappa)
            }
        }
    }

    /// DARE-based initialization using the Discrete Algebraic Riccati Equation.
    ///
    /// When `enforce_stationarity=true` AND `enforce_invertibility=true`, the
    /// steady-state predicted covariance P_inf can be computed directly by
    /// iteratively solving the DARE:
    ///
    /// ```text
    /// P = T * P * T' + RQR' - (T * P * Z) * (Z' * P * Z)^{-1} * (Z' * P * T')
    /// ```
    ///
    /// This gives:
    /// - Diffuse states (d + s*D): P_0 = kappa * I, burn = k_states_diff
    /// - ARMA states: P_0 = P_inf from DARE (the converged predicted covariance)
    ///
    /// The DARE solution provides a tighter P_0 than the Lyapunov solution
    /// (which gives P = T*P*T' + RQR' without the Kalman gain correction),
    /// allowing the filter to converge faster.
    ///
    /// Falls back to `mixed()` (Lyapunov) if DARE iteration fails.
    ///
    /// When sd=0 (pure stationary ARMA), also pre-computes steady-state Kalman
    /// quantities (K_inf, F_inf, pz_inf) so that `kalman_core` can activate the
    /// steady-state cache immediately at t=0, skipping the transient phase.
    pub fn dare(ss: &StateSpace, config: &SarimaxConfig, kappa: f64) -> Self {
        let k_states = ss.k_states;
        let sd = config.effective_sd();
        let ko = config.order.k_order();

        if ko == 0 || sd >= k_states {
            return Self::approximate_diffuse(k_states, kappa);
        }

        // Extract ARMA block: T[sd..sd+ko, sd..sd+ko]
        let t_arma = ss.transition.view((sd, sd), (ko, ko)).into_owned();

        // Z for ARMA block: design[sd..sd+ko]
        let z_arma = DVector::from_iterator(ko, (sd..sd + ko).map(|i| ss.design[i]));

        // RQR for ARMA block
        let r_arma = ss.selection.view((sd, 0), (ko, ss.k_posdef)).into_owned();
        let rqr_arma = &r_arma * &ss.state_cov * r_arma.transpose();

        // Try DARE first (200 iters for high-dimensional k>=50; tol 1e-11 for faster convergence)
        let dare_result = solve_dare_iterative(&t_arma, &z_arma, &rqr_arma, 200, 1e-11);

        // Quality gate: validate DARE solution before adopting
        let dare_ok = dare_result
            .as_ref()
            .map(dare_quality_gate)
            .unwrap_or(false);

        if let (Some(p_inf), true) = (dare_result, dare_ok) {
            let mut p0 = DMatrix::<f64>::zeros(k_states, k_states);

            // Diffuse part: kappa * I for diff states
            for i in 0..sd {
                p0[(i, i)] = kappa;
            }

            // Stationary part: P_inf for ARMA states
            for i in 0..ko {
                for j in 0..ko {
                    p0[(sd + i, sd + j)] = p_inf[(i, j)];
                }
            }

            // P1 effective: when sd=0, P_0 = P_∞ for the full state.
            // Pre-compute steady-state quantities so kalman_core can skip
            // the transient convergence phase entirely.
            let steady_state = if sd == 0 {
                let z = &ss.design;
                let t = &ss.transition;
                let pz_inf = &p_inf * z;
                let f_steady = z.dot(&pz_inf);
                if f_steady > 0.0 && f_steady.is_finite() {
                    let mut k_gain = DVector::zeros(k_states);
                    k_gain.gemv(1.0, t, &pz_inf, 0.0);
                    Some(KalmanSteadyState {
                        k_gain,
                        f_steady,
                        log_f_steady: f_steady.ln(),
                        pz_inf,
                    })
                } else {
                    None
                }
            } else {
                None
            };

            Self {
                initial_state: DVector::zeros(k_states),
                initial_state_cov: p0,
                loglikelihood_burn: sd,
                steady_state,
            }
        } else {
            // DARE failed or quality gate rejected — fall back to mixed (Lyapunov)
            Self::mixed(ss, config, kappa)
        }
    }

    /// Choose initialization based on config.
    ///
    /// - `enforce_stationarity=true`:
    ///   - sd=0 (pure stationary ARMA): DARE for tighter P_0 + pre-computed
    ///     steady-state Kalman quantities for immediate ss_cache activation.
    ///   - sd>0 (models with differencing): Lyapunov (`mixed()`) for
    ///     statsmodels compatibility.
    /// - `enforce_stationarity=false`: approximate diffuse for all states
    pub fn from_config(ss: &StateSpace, config: &SarimaxConfig, kappa: f64) -> Self {
        if config.enforce_stationarity {
            let sd = config.effective_sd();
            if sd == 0 {
                // Pure stationary: DARE gives tighter P_0 + steady-state shortcut
                Self::dare(ss, config, kappa)
            } else {
                // Differencing present: Lyapunov matches statsmodels behavior
                Self::mixed(ss, config, kappa)
            }
        } else {
            Self::approximate_diffuse(ss.k_states, kappa)
        }
    }

    /// Default kappa value matching statsmodels.
    pub fn default_kappa() -> f64 {
        1e6
    }

    /// Convenience constructor using the default kappa value (1e6).
    pub fn from_config_default(ss: &StateSpace, config: &SarimaxConfig) -> Self {
        Self::from_config(ss, config, Self::default_kappa())
    }
}

/// Solve the discrete Lyapunov equation: P = T * P * T' + Q
///
/// Uses iterative doubling (Smith's method):
///   A_0 = T,  Q_0 = Q
///   Q_{i+1} = Q_i + A_i * Q_i * A_i'
///   A_{i+1} = A_i * A_i
///
/// Converges in O(log(k)) iterations, each O(k³), total O(k³ log k).
/// Much faster than the Kronecker approach O(k⁶) for large k.
fn solve_discrete_lyapunov(t: &DMatrix<f64>, q: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let k = t.nrows();
    if k == 0 {
        return Some(DMatrix::zeros(0, 0));
    }

    let mut a_i = t.clone();
    let mut q_i = q.clone();
    let mut temp_aq = DMatrix::<f64>::zeros(k, k);
    let mut temp_aa = DMatrix::<f64>::zeros(k, k);

    for _ in 0..100 {
        // temp_aq = A_i * Q_i
        temp_aq.gemm(1.0, &a_i, &q_i, 0.0);

        // q_next = Q_i + temp_aq * A_i'
        let mut q_next = q_i.clone();
        q_next.gemm(1.0, &temp_aq, &a_i.transpose(), 1.0);

        // A_{i+1} = A_i * A_i
        temp_aa.gemm(1.0, &a_i, &a_i, 0.0);

        // Convergence check
        let mut diff_sq = 0.0;
        let mut norm_sq = 0.0;
        for (new, old) in q_next.iter().zip(q_i.iter()) {
            let d = new - old;
            diff_sq += d * d;
            norm_sq += old * old;
        }
        let norm = norm_sq.sqrt().max(1.0);

        q_i = q_next;
        a_i.copy_from(&temp_aa);

        if diff_sq.sqrt() < 1e-14 * norm {
            // Validate result: check finiteness
            if q_i.iter().any(|v| !v.is_finite()) {
                return None;
            }
            // Validate positive-definiteness via Cholesky decomposition.
            //
            // Finite-order MA state covariances can be numerically close to
            // singular, especially after seasonal expansion. Treat that as a
            // conditioning problem, not an initialization failure: symmetrize
            // and add a scale-aware ridge before falling back to diffuse init.
            q_i = (&q_i + q_i.transpose()) * 0.5;
            if q_i.clone().cholesky().is_some() {
                return Some(q_i);
            }

            let max_diag = (0..k)
                .map(|i| q_i[(i, i)].abs())
                .fold(1.0_f64, f64::max);
            let mut stabilized = q_i.clone();
            for attempt in 0..8 {
                let ridge = max_diag * 1e-12 * 10_f64.powi(attempt);
                for i in 0..k {
                    stabilized[(i, i)] += ridge;
                }
                if stabilized.clone().cholesky().is_some() {
                    return Some(stabilized);
                }
            }

            return None;
        }
    }
    None
}

/// Solve the Discrete Algebraic Riccati Equation (DARE) iteratively:
///
/// Validate DARE solution quality: symmetry and approximate PSD.
///
/// Returns `true` if the solution passes both checks, `false` otherwise.
/// - Symmetry: max |P - P'| < sym_tol * max(||P||_F, 1)
/// - PSD: min diagonal >= -psd_tol (lightweight proxy for eigenvalue check)
fn dare_quality_gate(p: &DMatrix<f64>) -> bool {
    let k = p.nrows();
    if k == 0 {
        return true;
    }

    // 1) Symmetry check
    let p_norm = p.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0);
    let sym_tol = 1e-8 * p_norm;
    for i in 0..k {
        for j in (i + 1)..k {
            if (p[(i, j)] - p[(j, i)]).abs() > sym_tol {
                return false;
            }
        }
    }

    // 2) PSD proxy: all diagonal elements must be non-negative (within tolerance)
    let psd_tol = 1e-8 * p_norm;
    for i in 0..k {
        if p[(i, i)] < -psd_tol {
            return false;
        }
    }

    true
}

/// P = T * P * T' + RQR' - (T * P * Z) * (Z' * P * Z)^{-1} * (Z' * P * T')
///
/// Uses fixed-point iteration starting from P_0 = RQR'.
/// Converges for stable (T, Z) pairs with all eigenvalues of T inside the unit circle.
///
/// Returns Some(P_inf) if converged within `max_iter` iterations, None otherwise.
fn solve_dare_iterative(
    t: &DMatrix<f64>,
    z: &DVector<f64>,
    rqr: &DMatrix<f64>,
    max_iter: usize,
    tol: f64,
) -> Option<DMatrix<f64>> {
    let k = t.nrows();
    if k == 0 {
        return Some(DMatrix::zeros(0, 0));
    }

    let t_t = t.transpose();
    let mut p = rqr.clone(); // P_0 = RQR'

    for _ in 0..max_iter {
        // pz = P * Z
        let pz = &p * z;
        // f = Z' * P * Z (scalar)
        let f = z.dot(&pz);
        if f.abs() < 1e-300 {
            return None; // degenerate
        }
        // tpz = T * P * Z
        let tpz = t * &pz;
        // correction = tpz * tpz' / f
        let correction = &tpz * tpz.transpose() / f;
        // tpt = T * P * T'
        let tpt = t * &p * &t_t;
        // p_new = tpt + rqr - correction
        let p_new = &tpt + rqr - &correction;

        // Convergence check
        let mut diff_sq = 0.0;
        let mut norm_sq = 0.0;
        for (new, old) in p_new.iter().zip(p.iter()) {
            let d = new - old;
            diff_sq += d * d;
            norm_sq += old * old;
        }
        let norm = norm_sq.sqrt().max(1.0);

        p = p_new;

        if diff_sq.sqrt() < tol * norm {
            // Validate finiteness
            if p.iter().any(|v| !v.is_finite()) {
                return None;
            }
            return Some(p);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approximate_diffuse_basic() {
        let init = KalmanInit::approximate_diffuse(2, 1e6);

        assert_eq!(init.initial_state.len(), 2);
        assert!((init.initial_state[0]).abs() < 1e-15);
        assert!((init.initial_state[1]).abs() < 1e-15);

        assert_eq!(init.initial_state_cov.nrows(), 2);
        assert_eq!(init.initial_state_cov.ncols(), 2);
        assert!((init.initial_state_cov[(0, 0)] - 1e6).abs() < 1e-4);
        assert!((init.initial_state_cov[(0, 1)]).abs() < 1e-15);
        assert!((init.initial_state_cov[(1, 0)]).abs() < 1e-15);
        assert!((init.initial_state_cov[(1, 1)] - 1e6).abs() < 1e-4);

        assert_eq!(init.loglikelihood_burn, 2);
    }

    #[test]
    fn test_approximate_diffuse_no_differencing() {
        let init = KalmanInit::approximate_diffuse(3, 1e6);
        assert_eq!(init.loglikelihood_burn, 3);
        assert_eq!(init.initial_state.len(), 3);
        assert_eq!(init.initial_state_cov.nrows(), 3);
    }

    #[test]
    fn test_default_kappa() {
        assert!((KalmanInit::default_kappa() - 1e6).abs() < 1e-10);
    }

    #[test]
    fn test_lyapunov_ar1() {
        // AR(1) with phi=0.5: P_∞ = sigma² / (1 - phi²) = 1.0 / 0.75 = 1.333...
        let t = DMatrix::from_row_slice(1, 1, &[0.5]);
        let q = DMatrix::from_row_slice(1, 1, &[1.0]);
        let p = solve_discrete_lyapunov(&t, &q).unwrap();
        let expected = 1.0 / (1.0 - 0.25); // 1.333...
        assert!(
            (p[(0, 0)] - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            p[(0, 0)]
        );
    }

    #[test]
    fn test_lyapunov_ar2() {
        // AR(2) companion: T = [[phi1, 1], [phi2, 0]], Q = [[1, 0], [0, 0]]
        let t = DMatrix::from_row_slice(2, 2, &[0.5, 1.0, -0.3, 0.0]);
        let q = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]);
        let p = solve_discrete_lyapunov(&t, &q).unwrap();

        // Verify: P should satisfy P = T * P * T' + Q
        let t_t = t.transpose();
        let residual = &t * &p * &t_t + &q - &p;
        let max_err = residual.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        assert!(max_err < 1e-10, "Lyapunov residual too large: {}", max_err);

        // P should be symmetric positive definite
        assert!(p[(0, 0)] > 0.0);
        assert!(p[(1, 1)] > 0.0);
        assert!((p[(0, 1)] - p[(1, 0)]).abs() < 1e-10);
    }

    #[test]
    fn test_lyapunov_empty() {
        let t = DMatrix::zeros(0, 0);
        let q = DMatrix::zeros(0, 0);
        let p = solve_discrete_lyapunov(&t, &q).unwrap();
        assert_eq!(p.nrows(), 0);
    }

    // ---- DARE tests ----

    #[test]
    fn test_dare_ar1() {
        // AR(1) with phi=0.5: DARE should converge
        let t = DMatrix::from_row_slice(1, 1, &[0.5]);
        let z = DVector::from_row_slice(&[1.0]);
        let rqr = DMatrix::from_row_slice(1, 1, &[1.0]);

        let p = solve_dare_iterative(&t, &z, &rqr, 100, 1e-12).unwrap();

        // Verify DARE equation: P = T*P*T' + RQR' - T*P*Z*(Z'*P*Z)^{-1}*Z'*P*T'
        let t_t = t.transpose();
        let tpt = &t * &p * &t_t;
        let pz = &p * &z;
        let f = z.dot(&pz);
        let tpz = &t * &pz;
        let correction = &tpz * tpz.transpose() / f;
        let residual = &tpt + &rqr - &correction - &p;
        let max_err = residual.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        assert!(max_err < 1e-9, "DARE residual too large: {}", max_err);

        // P should be positive
        assert!(p[(0, 0)] > 0.0);
    }

    #[test]
    fn test_dare_ar2() {
        // AR(2) companion: T = [[0.5, 1], [-0.3, 0]], Z = [1, 0], RQR = [[1,0],[0,0]]
        let t = DMatrix::from_row_slice(2, 2, &[0.5, 1.0, -0.3, 0.0]);
        let z = DVector::from_row_slice(&[1.0, 0.0]);
        let rqr = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]);

        let p = solve_dare_iterative(&t, &z, &rqr, 100, 1e-12).unwrap();

        // Verify DARE equation
        let t_t = t.transpose();
        let tpt = &t * &p * &t_t;
        let pz = &p * &z;
        let f = z.dot(&pz);
        let tpz = &t * &pz;
        let correction = &tpz * tpz.transpose() / f;
        let residual = &tpt + &rqr - &correction - &p;
        let max_err = residual.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        assert!(max_err < 1e-9, "DARE residual too large: {}", max_err);

        // P should be symmetric; P(0,0) must be positive; P(1,1) >= 0
        // (noise only enters through first state in companion form)
        assert!(p[(0, 0)] > 0.0);
        assert!(p[(1, 1)] >= 0.0);
        assert!((p[(0, 1)] - p[(1, 0)]).abs() < 1e-10);
    }

    #[test]
    fn test_dare_empty() {
        let t = DMatrix::zeros(0, 0);
        let z = DVector::from_row_slice(&[] as &[f64]);
        let rqr = DMatrix::zeros(0, 0);
        let p = solve_dare_iterative(&t, &z, &rqr, 100, 1e-12).unwrap();
        assert_eq!(p.nrows(), 0);
    }

    #[test]
    fn test_dare_vs_lyapunov_consistency() {
        // DARE solution should have smaller or equal diagonal than Lyapunov
        // (DARE subtracts the Kalman gain correction term)
        let t = DMatrix::from_row_slice(1, 1, &[0.8]);
        let z = DVector::from_row_slice(&[1.0]);
        let rqr = DMatrix::from_row_slice(1, 1, &[1.0]);

        let p_lyap = solve_discrete_lyapunov(&t, &rqr).unwrap();
        let p_dare = solve_dare_iterative(&t, &z, &rqr, 100, 1e-12).unwrap();

        assert!(
            p_dare[(0, 0)] <= p_lyap[(0, 0)] + 1e-10,
            "DARE P ({}) should be <= Lyapunov P ({})",
            p_dare[(0, 0)],
            p_lyap[(0, 0)]
        );
        assert!(p_dare[(0, 0)] > 0.0);
    }
}
