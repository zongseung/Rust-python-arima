use nalgebra::{DMatrix, DVector};

use crate::state_space::StateSpace;
use crate::types::SarimaxConfig;

/// Kalman filter initial state and covariance.
pub struct KalmanInit {
    /// Initial state vector a_0 (zeros).
    pub initial_state: DVector<f64>,
    /// Initial state covariance P_0.
    pub initial_state_cov: DMatrix<f64>,
    /// Number of initial observations to skip in loglikelihood (burn-in).
    pub loglikelihood_burn: usize,
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
        let sd = config.order.k_states_diff();
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
                }
            }
            None => {
                // Lyapunov solve failed → fall back to approximate diffuse
                Self::approximate_diffuse(k_states, kappa)
            }
        }
    }

    /// Choose initialization based on config.
    ///
    /// - `enforce_stationarity=true`: mixed (stationary ARMA + diffuse diff)
    /// - `enforce_stationarity=false`: approximate diffuse for all states
    pub fn from_config(ss: &StateSpace, config: &SarimaxConfig, kappa: f64) -> Self {
        if config.enforce_stationarity {
            Self::mixed(ss, config, kappa)
        } else {
            Self::approximate_diffuse(ss.k_states, kappa)
        }
    }

    /// Default kappa value matching statsmodels.
    pub fn default_kappa() -> f64 {
        1e6
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
            // Validate result
            if q_i.iter().any(|v| !v.is_finite()) {
                return None;
            }
            if (0..k).any(|i| q_i[(i, i)] < -1e-10) {
                return None;
            }
            return Some(q_i);
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
        assert!(
            max_err < 1e-10,
            "Lyapunov residual too large: {}",
            max_err
        );

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
}
