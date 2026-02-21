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
/// Uses the Kronecker product approach:
///   vec(P) = (I - T ⊗ T)^{-1} * vec(Q)
///
/// Returns None if the system is singular (T has eigenvalue on unit circle).
fn solve_discrete_lyapunov(t: &DMatrix<f64>, q: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let k = t.nrows();
    if k == 0 {
        return Some(DMatrix::zeros(0, 0));
    }

    let kk = k * k;

    // Build I_{k²} - T ⊗ T
    let mut lhs = DMatrix::<f64>::identity(kk, kk);
    for i in 0..k {
        for j in 0..k {
            let t_ij = t[(i, j)];
            for p in 0..k {
                for q_idx in 0..k {
                    // (T ⊗ T)[(i*k+p), (j*k+q)] = T[i,j] * T[p,q]
                    lhs[(i * k + p, j * k + q_idx)] -= t_ij * t[(p, q_idx)];
                }
            }
        }
    }

    // Vectorize Q (column-major order for nalgebra)
    let rhs = DVector::from_iterator(kk, q.iter().copied());

    // Solve the linear system
    let p_vec = lhs.lu().solve(&rhs)?;

    // Unvectorize back to matrix (column-major)
    let p = DMatrix::from_iterator(k, k, p_vec.iter().copied());

    // Validate: P should be symmetric positive semi-definite
    // Check for NaN/Inf
    if p.iter().any(|v| !v.is_finite()) {
        return None;
    }

    // Check diagonal is positive
    if (0..k).any(|i| p[(i, i)] < -1e-10) {
        return None;
    }

    Some(p)
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
