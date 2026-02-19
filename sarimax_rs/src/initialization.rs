use nalgebra::{DMatrix, DVector};

/// Kalman filter initial state and covariance.
pub struct KalmanInit {
    /// Initial state vector a_0 (zeros).
    pub initial_state: DVector<f64>,
    /// Initial state covariance P_0 (kappa * I).
    pub initial_state_cov: DMatrix<f64>,
    /// Number of initial observations to skip in loglikelihood (burn-in).
    pub loglikelihood_burn: usize,
}

impl KalmanInit {
    /// Approximate diffuse initialization.
    ///
    /// This is the default initialization when `enforce_stationarity=false`.
    /// - a_0 = 0
    /// - P_0 = kappa * I_{k_states}
    /// - burn = k_states (skip all diffuse-affected observations, matching statsmodels)
    pub fn approximate_diffuse(k_states: usize, kappa: f64) -> Self {
        Self {
            initial_state: DVector::zeros(k_states),
            initial_state_cov: DMatrix::identity(k_states, k_states) * kappa,
            loglikelihood_burn: k_states,
        }
    }

    /// Default kappa value matching statsmodels.
    pub fn default_kappa() -> f64 {
        1e6
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approximate_diffuse_basic() {
        let init = KalmanInit::approximate_diffuse(2, 1e6);

        // a_0 = [0, 0]
        assert_eq!(init.initial_state.len(), 2);
        assert!((init.initial_state[0]).abs() < 1e-15);
        assert!((init.initial_state[1]).abs() < 1e-15);

        // P_0 = [[1e6, 0], [0, 1e6]]
        assert_eq!(init.initial_state_cov.nrows(), 2);
        assert_eq!(init.initial_state_cov.ncols(), 2);
        assert!((init.initial_state_cov[(0, 0)] - 1e6).abs() < 1e-4);
        assert!((init.initial_state_cov[(0, 1)]).abs() < 1e-15);
        assert!((init.initial_state_cov[(1, 0)]).abs() < 1e-15);
        assert!((init.initial_state_cov[(1, 1)] - 1e6).abs() < 1e-4);

        // burn = k_states = 2
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
}
