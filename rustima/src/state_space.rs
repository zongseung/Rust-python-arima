use nalgebra::{DMatrix, DVector};

use crate::error::{Result, SarimaxError};
use crate::params::SarimaxParams;
use crate::polynomial::{reduced_ar, reduced_ma};
use crate::types::SarimaxConfig;

/// Harvey-representation state space for SARIMAX.
///
/// State equation:  alpha_{t+1} = T * alpha_t + c_t + R * eta_t
/// Observation:     y_t          = Z' * alpha_t + d_t + eps_t
///
/// where eta_t ~ N(0, Q), eps_t ~ N(0, H).
/// H = 0 (no measurement error) for standard ARIMA.
pub struct StateSpace {
    pub k_states: usize,
    pub k_states_diff: usize,
    pub k_posdef: usize,           // always 1 for univariate ARIMA
    pub transition: DMatrix<f64>,  // T: k_states × k_states
    pub design: DVector<f64>,      // Z: k_states (single observation row)
    pub selection: DMatrix<f64>,   // R: k_states × k_posdef
    pub state_cov: DMatrix<f64>,   // Q: k_posdef × k_posdef
    pub obs_intercept: Vec<f64>,   // d_t: exog contribution per time step
    pub state_intercept: Vec<f64>, // c_t: trend contribution per time step
}

impl StateSpace {
    /// Construct the Harvey representation for a SARIMA model.
    ///
    /// Supports SARIMA(p,d,q)(P,D,Q,s) with D <= 1.
    pub fn new(
        config: &SarimaxConfig,
        params: &SarimaxParams,
        endog: &[f64],
        exog: Option<&[Vec<f64>]>,
    ) -> Result<Self> {
        let order = &config.order;

        if order.dd > 1 {
            return Err(SarimaxError::StateSpaceError(
                "Seasonal differencing D > 1 is not yet supported".into(),
            ));
        }

        if order.dd > 0 && order.s < 2 {
            return Err(SarimaxError::StateSpaceError(format!(
                "Seasonal differencing D={} requires seasonal period s >= 2, got s={}",
                order.dd, order.s
            )));
        }

        if config.measurement_error {
            return Err(SarimaxError::StateSpaceError(
                "measurement_error is not yet supported".into(),
            ));
        }

        // simple_differencing: diff states are removed from the state vector.
        // The data will be pre-differenced externally; the SS only models the ARMA part.
        let k_states = if config.simple_differencing { order.k_order() } else { order.k_states() };
        let k_states_diff = if config.simple_differencing { 0 } else { order.k_states_diff() };
        let _k_order = order.k_order();
        let k_posdef = 1;
        let n = endog.len();

        // V-3: Validate exog column lengths match number of observations
        if let Some(x) = exog {
            for (j, col) in x.iter().enumerate() {
                if col.len() != n {
                    return Err(SarimaxError::DataError(format!(
                        "exog column {} has {} rows but y has {} observations",
                        j,
                        col.len(),
                        n
                    )));
                }
            }
        }

        // Build matrices (sd=0 when simple_differencing=true → ARMA-only)
        let transition = Self::build_transition_sd(config, params, k_states, k_states_diff)?;
        let design = Self::build_design_sd(config, k_states, k_states_diff);
        let selection = Self::build_selection_sd(config, params, k_states, k_states_diff)?;
        let state_cov = Self::build_state_cov(config, params);

        // Observation intercept: d_t = exog * beta
        let obs_intercept = Self::build_obs_intercept(n, params, exog);

        // State intercept: c_t (trend contribution)
        let state_intercept =
            Self::build_state_intercept(n, k_states, k_states_diff, config, params);

        if transition.nrows() != k_states || transition.ncols() != k_states {
            return Err(SarimaxError::StateSpaceError(format!(
                "T matrix dimension mismatch: expected {}×{}, got {}×{}",
                k_states,
                k_states,
                transition.nrows(),
                transition.ncols()
            )));
        }

        Ok(Self {
            k_states,
            k_states_diff,
            k_posdef,
            transition,
            design,
            selection,
            state_cov,
            obs_intercept,
            state_intercept,
        })
    }

    /// Update parameter-dependent matrices in-place.
    ///
    /// During optimization, only the ARMA parameters change between iterations.
    /// This method updates T (companion column), R (MA coefficients), Q (sigma2),
    /// and intercepts in-place, avoiding the full matrix reconstruction cost of
    /// `StateSpace::new()`.
    ///
    /// Config-dependent structure (diff blocks, seasonal shifts, connections,
    /// superdiagonal ones) remains unchanged.
    pub fn update_params(
        &mut self,
        config: &SarimaxConfig,
        params: &SarimaxParams,
        endog: &[f64],
        exog: Option<&[Vec<f64>]>,
    ) {
        let order = &config.order;
        let sd = self.k_states_diff; // already 0 when simple_differencing
        let ko = order.k_order();
        let n = endog.len();

        // Defensive: exog columns must match endog length (callers should pre-validate).
        debug_assert!(
            exog.is_none_or(|cols| cols.iter().all(|c| c.len() == n)),
            "update_params: exog column length != endog length ({})",
            n
        );

        // 1. Update ARMA companion first column in T: T[sd+i, sd] = -reduced_ar[i+1]
        let red_ar = reduced_ar(params, order);
        for i in 0..ko {
            let idx = i + 1;
            self.transition[(sd + i, sd)] = if idx < red_ar.len() {
                -red_ar[idx]
            } else {
                0.0
            };
        }

        // 2. Update MA coefficients in R: R[sd+i, 0] = reduced_ma[i]
        let red_ma = reduced_ma(params, order);
        for i in 1..ko {
            self.selection[(sd + i, 0)] = if i < red_ma.len() {
                red_ma[i]
            } else {
                0.0
            };
        }

        // 3. Update Q: sigma2
        self.state_cov[(0, 0)] = if config.concentrate_scale {
            1.0
        } else {
            params.sigma2.unwrap_or(1.0)
        };

        // 4. Update obs_intercept (d_t = exog * beta)
        self.obs_intercept = Self::build_obs_intercept(n, params, exog);

        // 5. Update state_intercept (trend contribution)
        self.state_intercept = Self::build_state_intercept(
            n,
            self.k_states,
            self.k_states_diff,
            config,
            params,
        );
    }

    /// Build the transition matrix T (sd-aware version).
    ///
    /// When `sd == 0` (simple_differencing=true): pure ARMA companion of size k_order × k_order.
    /// When `sd > 0`: full Harvey representation with diff blocks.
    fn build_transition_sd(
        config: &SarimaxConfig,
        params: &SarimaxParams,
        k_states: usize,
        sd: usize,
    ) -> Result<DMatrix<f64>> {
        let order = &config.order;
        let d = order.d;
        let dd = order.dd;
        let s = order.s;
        let ko = order.k_order();

        let mut t = DMatrix::<f64>::zeros(k_states, k_states);

        if sd > 0 {
            // 1. Regular differencing block [0..d, 0..d]: upper triangular ones
            for i in 0..d {
                for j in i..d {
                    t[(i, j)] = 1.0;
                }
            }

            // 2. Seasonal differencing: cyclic shift blocks
            for layer in 0..dd {
                let base = d + layer * s;
                // Wrap: first row of block → last column of block
                t[(base, base + s - 1)] = 1.0;
                // Shift down
                for i in 0..(s - 1) {
                    t[(base + i + 1, base + i)] = 1.0;
                }
            }

            // 3. Cross-diff: regular diff states → last seasonal state
            if dd > 0 {
                let last_seasonal = d + s * dd - 1;
                for i in 0..d {
                    t[(i, last_seasonal)] = 1.0;
                }
            }

            // 4. Diff → ARMA connections
            // Regular diff → first ARMA state
            for i in 0..d {
                t[(i, sd)] = 1.0;
            }
            // First seasonal state of each layer → first ARMA state
            for layer in 0..dd {
                t[(d + layer * s, sd)] = 1.0;
            }
        }

        // 5. ARMA companion matrix [sd..sd+ko, sd..sd+ko]
        let red_ar = reduced_ar(params, order);
        for i in 0..ko {
            let idx = i + 1;
            if idx < red_ar.len() {
                t[(sd + i, sd)] = -red_ar[idx];
            }
        }
        // Superdiagonal ones
        for i in 0..(ko.saturating_sub(1)) {
            t[(sd + i, sd + i + 1)] = 1.0;
        }

        Ok(t)
    }

    /// Build the design vector Z (sd-aware version).
    ///
    /// When sd == 0 (simple_differencing): Z[0] = 1.0 only (first ARMA state).
    /// When sd > 0: full Harvey Z with diff states + ARMA state.
    fn build_design_sd(config: &SarimaxConfig, k_states: usize, sd: usize) -> DVector<f64> {
        let order = &config.order;
        let d = order.d;
        let dd = order.dd;
        let s = order.s;

        let mut z = DVector::<f64>::zeros(k_states);

        if sd > 0 {
            // Regular diff states
            for i in 0..d {
                z[i] = 1.0;
            }

            // Last state of each seasonal layer
            for layer in 0..dd {
                z[d + (layer + 1) * s - 1] = 1.0;
            }
        }

        // First ARMA state (always present; when sd=0, this is index 0)
        if sd < k_states {
            z[sd] = 1.0;
        }

        z
    }

    /// Build the selection matrix R (sd-aware version).
    ///
    /// When sd == 0 (simple_differencing): R[0,0]=1, R[i,0]=ma[i] for i>=1.
    /// When sd > 0: R[sd,0]=1, R[sd+i,0]=ma[i] for i>=1.
    fn build_selection_sd(
        config: &SarimaxConfig,
        params: &SarimaxParams,
        k_states: usize,
        sd: usize,
    ) -> Result<DMatrix<f64>> {
        let order = &config.order;
        let ko = order.k_order();

        let mut r = DMatrix::<f64>::zeros(k_states, 1);

        let red_ma = reduced_ma(params, order);

        // R[sd, 0] = 1 (corresponds to reduced_ma[0] which is always 1)
        r[(sd, 0)] = 1.0;

        // R[sd+i, 0] = reduced_ma[i] for i >= 1
        for i in 1..ko {
            if i < red_ma.len() {
                r[(sd + i, 0)] = red_ma[i];
            }
        }

        Ok(r)
    }

    /// Build the state covariance Q.
    ///
    /// For concentrate_scale: Q = [[1.0]]
    /// Otherwise: Q = [[sigma2]]
    fn build_state_cov(config: &SarimaxConfig, params: &SarimaxParams) -> DMatrix<f64> {
        let sigma2 = if config.concentrate_scale {
            1.0
        } else {
            params.sigma2.unwrap_or(1.0)
        };
        DMatrix::from_element(1, 1, sigma2)
    }

    /// Build observation intercept d_t = exog_t * beta_exog.
    fn build_obs_intercept(
        n: usize,
        params: &SarimaxParams,
        exog: Option<&[Vec<f64>]>,
    ) -> Vec<f64> {
        match exog {
            Some(x) if !params.exog_coeffs.is_empty() => (0..n)
                .map(|t| {
                    x.iter()
                        .zip(params.exog_coeffs.iter())
                        .map(|(col, &b)| col[t] * b)
                        .sum()
                })
                .collect(),
            _ => vec![0.0; n],
        }
    }

    /// Build state intercept c_t (trend contribution).
    ///
    /// For trend='c': c_t[d] = const  (injected into ARMA first state)
    /// For trend='t': c_t[d] = beta * t
    /// For trend='ct': c_t[d] = const + beta * t
    fn build_state_intercept(
        n: usize,
        k_states: usize,
        k_states_diff: usize,
        config: &SarimaxConfig,
        params: &SarimaxParams,
    ) -> Vec<f64> {
        use crate::types::Trend;

        if config.trend == Trend::None || params.trend_coeffs.is_empty() {
            return vec![0.0; n * k_states];
        }

        // State intercept is a flat vec: [c_0[0..k], c_1[0..k], ...]
        // Trend contribution goes to state index k_states_diff (first ARMA state)
        let mut c = vec![0.0; n * k_states];
        let inject_idx = k_states_diff; // where trend enters the state

        for t in 0..n {
            let val = match config.trend {
                Trend::Constant => params.trend_coeffs[0],
                Trend::Linear => params.trend_coeffs[0] * (t as f64),
                Trend::Both => params.trend_coeffs[0] + params.trend_coeffs[1] * (t as f64),
                Trend::None => 0.0,
            };
            c[t * k_states + inject_idx] = val;
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{make_config, make_seasonal_config, make_params, make_seasonal_params};
    use crate::types::{SarimaxConfig, SarimaxOrder};

    /// Appendix verification: structural sparsity of the Harvey transition
    /// matrix across SARIMA specifications. Counts the nonzero entries of T
    /// built with generic (non-cancelling) parameter values.
    /// Run with:  cargo test report_transition_sparsity -- --nocapture
    #[test]
    fn report_transition_sparsity() {
        // (p, d, q, P, D, Q, s, label)
        let specs: &[(usize, usize, usize, usize, usize, usize, usize, &str)] = &[
            (1, 1, 1, 0, 0, 0, 0, "ARIMA(1,1,1)"),
            (2, 1, 2, 0, 0, 0, 0, "ARIMA(2,1,2)"),
            (1, 1, 1, 1, 1, 1, 7, "SARIMA(1,1,1)(1,1,1)_7"),
            (1, 1, 1, 1, 1, 1, 12, "SARIMA(1,1,1)(1,1,1)_12"),
            (1, 1, 1, 1, 1, 1, 24, "SARIMA(1,1,1)(1,1,1)_24"),
            (2, 1, 2, 1, 1, 1, 24, "SARIMA(2,1,2)(1,1,1)_24"),
            (2, 0, 1, 1, 0, 1, 24, "SARIMA(2,0,1)(1,0,1)_24"),
            (3, 1, 3, 2, 1, 2, 24, "SARIMA(3,1,3)(2,1,2)_24"),
        ];
        let ar_pool = [0.5, 0.3, 0.2];
        let ma_pool = [0.4, 0.25, 0.15];
        let sar_pool = [0.45, 0.22];
        let sma_pool = [0.35, 0.18];
        let endog = vec![0.0; 300];

        println!();
        println!(
            "{:<26} {:>4} {:>7} {:>8} {:>9} {:>7}",
            "Model", "k", "nnz(T)", "k^2", "density", "nnz(Z)"
        );
        println!("{}", "-".repeat(66));
        for &(p, d, q, pp, dd, qq, s, label) in specs {
            let config = make_seasonal_config(p, d, q, pp, dd, qq, s);
            let params = make_seasonal_params(
                &ar_pool[..p],
                &ma_pool[..q],
                &sar_pool[..pp],
                &sma_pool[..qq],
            );
            let ss = StateSpace::new(&config, &params, &endog, None).unwrap();
            let k = ss.k_states;
            let nnz_t = ss.transition.iter().filter(|&&x| x != 0.0).count();
            let nnz_z = ss.design.iter().filter(|&&x| x != 0.0).count();
            let k2 = k * k;
            let density = 100.0 * nnz_t as f64 / k2 as f64;
            assert!(nnz_t > 0, "transition matrix must have nonzero entries");
            println!(
                "{:<26} {:>4} {:>7} {:>8} {:>8.2}% {:>7}",
                label, k, nnz_t, k2, density, nnz_z
            );
        }
        println!();
    }

    #[test]
    fn test_ar1_transition() {
        // AR(1) with phi=0.6527: k_states=1, T=[[phi]]
        let config = make_config(1, 0, 0);
        let params = make_params(&[0.6527425084139002], &[]);
        let endog = vec![0.0; 10]; // dummy
        let ss = StateSpace::new(&config, &params, &endog, None).unwrap();

        assert_eq!(ss.k_states, 1);
        assert_eq!(ss.k_states_diff, 0);
        assert!((ss.transition[(0, 0)] - 0.6527425084139002).abs() < 1e-10);
    }

    #[test]
    fn test_ar1_design_selection() {
        let config = make_config(1, 0, 0);
        let params = make_params(&[0.65], &[]);
        let endog = vec![0.0; 10];
        let ss = StateSpace::new(&config, &params, &endog, None).unwrap();

        // Z = [1.0]
        assert_eq!(ss.design.len(), 1);
        assert!((ss.design[0] - 1.0).abs() < 1e-10);

        // R = [[1.0]]
        assert!((ss.selection[(0, 0)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_arma11_transition() {
        // ARMA(1,1) with phi=0.4139, theta=0.336
        // k_order=2, k_states=2, k_states_diff=0
        // T = [[phi, 1], [0, 0]]
        let config = make_config(1, 0, 1);
        let params = make_params(&[0.41390307727487496], &[0.33603638737455516]);
        let endog = vec![0.0; 10];
        let ss = StateSpace::new(&config, &params, &endog, None).unwrap();

        assert_eq!(ss.k_states, 2);
        assert_eq!(ss.k_states_diff, 0);

        // T[0,0] = phi
        assert!((ss.transition[(0, 0)] - 0.41390307727487496).abs() < 1e-10);
        // T[0,1] = 1 (superdiagonal)
        assert!((ss.transition[(0, 1)] - 1.0).abs() < 1e-10);
        // T[1,0] = 0
        assert!((ss.transition[(1, 0)]).abs() < 1e-10);
        // T[1,1] = 0
        assert!((ss.transition[(1, 1)]).abs() < 1e-10);
    }

    #[test]
    fn test_arma11_selection() {
        // ARMA(1,1) with theta=0.336
        // R = [[1.0], [theta]]
        let config = make_config(1, 0, 1);
        let params = make_params(&[0.4139], &[0.33603638737455516]);
        let endog = vec![0.0; 10];
        let ss = StateSpace::new(&config, &params, &endog, None).unwrap();

        assert!((ss.selection[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((ss.selection[(1, 0)] - 0.33603638737455516).abs() < 1e-10);
    }

    #[test]
    fn test_arma11_design() {
        // Z = [1, 0]
        let config = make_config(1, 0, 1);
        let params = make_params(&[0.4139], &[0.336]);
        let endog = vec![0.0; 10];
        let ss = StateSpace::new(&config, &params, &endog, None).unwrap();

        assert!((ss.design[0] - 1.0).abs() < 1e-10);
        assert!((ss.design[1]).abs() < 1e-10);
    }

    #[test]
    fn test_arima111_transition() {
        // ARIMA(1,1,1) with phi=-0.6441, theta=0.7
        // k_states=3, k_states_diff=1, k_order=2
        // T = [[1, 1, 0], [0, phi, 1], [0, 0, 0]]
        let config = make_config(1, 1, 1);
        let params = make_params(&[-0.6441303822894944], &[0.7000629128883827]);
        let endog = vec![0.0; 10];
        let ss = StateSpace::new(&config, &params, &endog, None).unwrap();

        assert_eq!(ss.k_states, 3);
        assert_eq!(ss.k_states_diff, 1);

        // Diff block
        assert!((ss.transition[(0, 0)] - 1.0).abs() < 1e-10);
        // Connection
        assert!((ss.transition[(0, 1)] - 1.0).abs() < 1e-10);
        assert!((ss.transition[(0, 2)]).abs() < 1e-10);
        // ARMA block
        assert!((ss.transition[(1, 0)]).abs() < 1e-10);
        assert!((ss.transition[(1, 1)] - (-0.6441303822894944)).abs() < 1e-10);
        assert!((ss.transition[(1, 2)] - 1.0).abs() < 1e-10);
        assert!((ss.transition[(2, 0)]).abs() < 1e-10);
        assert!((ss.transition[(2, 1)]).abs() < 1e-10);
        assert!((ss.transition[(2, 2)]).abs() < 1e-10);
    }

    #[test]
    fn test_arima111_design() {
        // Z = [1, 1, 0]
        let config = make_config(1, 1, 1);
        let params = make_params(&[-0.6441], &[0.7]);
        let endog = vec![0.0; 10];
        let ss = StateSpace::new(&config, &params, &endog, None).unwrap();

        assert!((ss.design[0] - 1.0).abs() < 1e-10);
        assert!((ss.design[1] - 1.0).abs() < 1e-10);
        assert!((ss.design[2]).abs() < 1e-10);
    }

    #[test]
    fn test_arima111_selection() {
        // R = [[0], [1], [theta]]
        let config = make_config(1, 1, 1);
        let params = make_params(&[-0.6441], &[0.7000629128883827]);
        let endog = vec![0.0; 10];
        let ss = StateSpace::new(&config, &params, &endog, None).unwrap();

        assert!((ss.selection[(0, 0)]).abs() < 1e-10);
        assert!((ss.selection[(1, 0)] - 1.0).abs() < 1e-10);
        assert!((ss.selection[(2, 0)] - 0.7000629128883827).abs() < 1e-10);
    }

    #[test]
    fn test_state_cov_concentrated() {
        let config = make_config(1, 0, 0);
        let params = make_params(&[0.5], &[]);
        let endog = vec![0.0; 10];
        let ss = StateSpace::new(&config, &params, &endog, None).unwrap();

        assert!((ss.state_cov[(0, 0)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_update_params_ar1() {
        // Build SS with one set of params, then update in-place
        let config = make_config(1, 0, 0);
        let params1 = make_params(&[0.5], &[]);
        let endog = vec![0.0; 10];
        let mut ss = StateSpace::new(&config, &params1, &endog, None).unwrap();
        assert!((ss.transition[(0, 0)] - 0.5).abs() < 1e-10);

        // Update to new AR coefficient
        let params2 = make_params(&[0.8], &[]);
        ss.update_params(&config, &params2, &endog, None);
        assert!((ss.transition[(0, 0)] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_update_params_arma11() {
        let config = make_config(1, 0, 1);
        let params1 = make_params(&[0.4], &[0.3]);
        let endog = vec![0.0; 10];
        let mut ss = StateSpace::new(&config, &params1, &endog, None).unwrap();

        // Verify initial values
        assert!((ss.transition[(0, 0)] - 0.4).abs() < 1e-10);
        assert!((ss.selection[(1, 0)] - 0.3).abs() < 1e-10);

        // Update to new params
        let params2 = make_params(&[0.9], &[-0.5]);
        ss.update_params(&config, &params2, &endog, None);
        assert!((ss.transition[(0, 0)] - 0.9).abs() < 1e-10);
        assert!((ss.selection[(1, 0)] - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_update_params_matches_new() {
        // Verify that update_params produces the same result as building from scratch
        let config = make_seasonal_config(1, 1, 1, 1, 0, 1, 12);
        let params1 = make_seasonal_params(&[0.5], &[0.3], &[0.2], &[-0.6]);
        let endog = vec![0.0; 300];
        let mut ss = StateSpace::new(&config, &params1, &endog, None).unwrap();

        let params2 = make_seasonal_params(&[0.9], &[-0.4], &[0.1], &[-0.8]);
        ss.update_params(&config, &params2, &endog, None);

        // Build fresh SS with params2 for comparison
        let ss_fresh = StateSpace::new(&config, &params2, &endog, None).unwrap();

        // Compare T, R, Q
        let t_diff = (&ss.transition - &ss_fresh.transition).norm();
        let r_diff = (&ss.selection - &ss_fresh.selection).norm();
        let q_diff = (&ss.state_cov - &ss_fresh.state_cov).norm();
        assert!(t_diff < 1e-12, "T mismatch after update_params: {}", t_diff);
        assert!(r_diff < 1e-12, "R mismatch after update_params: {}", r_diff);
        assert!(q_diff < 1e-12, "Q mismatch after update_params: {}", q_diff);
    }

    #[test]
    fn test_seasonal_d2_rejected() {
        // D > 1 is not yet supported
        let config = SarimaxConfig {
            order: SarimaxOrder::new(1, 0, 0, 0, 2, 0, 12),
            ..make_config(1, 0, 0)
        };
        let params = make_params(&[0.5], &[]);
        let endog = vec![0.0; 10];
        assert!(StateSpace::new(&config, &params, &endog, None).is_err());
    }

    #[test]
    fn test_ar2_companion() {
        // AR(2) with phi1=0.5, phi2=-0.3
        // k_order=2, k_states=2
        // make_ar_poly([0.5, -0.3], 2) = [1, -0.5, 0.3]
        // -reduced_ar[1:] = [0.5, -0.3]
        // T = [[0.5, 1], [-0.3, 0]]
        let config = make_config(2, 0, 0);
        let params = make_params(&[0.5, -0.3], &[]);
        let endog = vec![0.0; 10];
        let ss = StateSpace::new(&config, &params, &endog, None).unwrap();

        assert_eq!(ss.k_states, 2);
        assert!((ss.transition[(0, 0)] - 0.5).abs() < 1e-10);
        assert!((ss.transition[(0, 1)] - 1.0).abs() < 1e-10);
        assert!((ss.transition[(1, 0)] - (-0.3)).abs() < 1e-10);
        assert!((ss.transition[(1, 1)]).abs() < 1e-10);
    }

    // ---- Seasonal tests ----

    #[test]
    fn test_sarima_100_100_4_transition() {
        // SARIMA(1,0,0)(1,0,0,4): k_states=5, no diff
        // reduced_ar = polymul([1,-0.7672], [1,0,0,0,-0.2322])
        //            = [1, -0.7672, 0, 0, -0.2322, 0.17815]
        // ARMA companion first col = [0.7672, 0, 0, 0.2322, -0.17815]
        let config = make_seasonal_config(1, 0, 0, 1, 0, 0, 4);
        let params = make_seasonal_params(&[0.7671699347442852], &[], &[0.2322174491752982], &[]);
        let endog = vec![0.0; 200];
        let ss = StateSpace::new(&config, &params, &endog, None).unwrap();

        assert_eq!(ss.k_states, 5);
        assert_eq!(ss.k_states_diff, 0);

        // First column of companion
        assert!((ss.transition[(0, 0)] - 0.7671699347442852).abs() < 1e-10);
        assert!((ss.transition[(1, 0)]).abs() < 1e-10);
        assert!((ss.transition[(2, 0)]).abs() < 1e-10);
        assert!((ss.transition[(3, 0)] - 0.2322174491752982).abs() < 1e-6);
        let cross = 0.7671699347442852 * 0.2322174491752982;
        assert!((ss.transition[(4, 0)] - (-cross)).abs() < 1e-6);

        // Superdiagonal
        assert!((ss.transition[(0, 1)] - 1.0).abs() < 1e-10);
        assert!((ss.transition[(1, 2)] - 1.0).abs() < 1e-10);
        assert!((ss.transition[(2, 3)] - 1.0).abs() < 1e-10);
        assert!((ss.transition[(3, 4)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sarima_100_100_4_design_selection() {
        let config = make_seasonal_config(1, 0, 0, 1, 0, 0, 4);
        let params = make_seasonal_params(&[0.77], &[], &[0.23], &[]);
        let endog = vec![0.0; 200];
        let ss = StateSpace::new(&config, &params, &endog, None).unwrap();

        // Z = [1, 0, 0, 0, 0]
        assert!((ss.design[0] - 1.0).abs() < 1e-10);
        for i in 1..5 {
            assert!(ss.design[i].abs() < 1e-10);
        }

        // R = [[1], [0], [0], [0], [0]] (no MA)
        assert!((ss.selection[(0, 0)] - 1.0).abs() < 1e-10);
        for i in 1..5 {
            assert!(ss.selection[(i, 0)].abs() < 1e-10);
        }
    }

    #[test]
    fn test_sarima_111_111_12_dimensions() {
        // SARIMA(1,1,1)(1,1,1,12): k_states=27
        let config = make_seasonal_config(1, 1, 1, 1, 1, 1, 12);
        let params = make_seasonal_params(&[0.9903], &[0.0660], &[0.0007], &[-1.0664]);
        let endog = vec![0.0; 300];
        let ss = StateSpace::new(&config, &params, &endog, None).unwrap();

        assert_eq!(ss.k_states, 27);
        assert_eq!(ss.k_states_diff, 13);
        assert_eq!(ss.transition.nrows(), 27);
        assert_eq!(ss.transition.ncols(), 27);
        assert_eq!(ss.design.len(), 27);
        assert_eq!(ss.selection.nrows(), 27);
    }

    #[test]
    fn test_sarima_111_111_12_transition() {
        let config = make_seasonal_config(1, 1, 1, 1, 1, 1, 12);
        let params = make_seasonal_params(
            &[0.9903479224371599],
            &[0.0659541127042639],
            &[0.0007132203797734934],
            &[-1.0663518115052784],
        );
        let endog = vec![0.0; 300];
        let ss = StateSpace::new(&config, &params, &endog, None).unwrap();

        // Regular diff: T[0,0] = 1
        assert!((ss.transition[(0, 0)] - 1.0).abs() < 1e-10);

        // Cross-diff: T[0,12] = 1
        assert!((ss.transition[(0, 12)] - 1.0).abs() < 1e-10);

        // Diff → ARMA: T[0,13] = 1, T[1,13] = 1
        assert!((ss.transition[(0, 13)] - 1.0).abs() < 1e-10);
        assert!((ss.transition[(1, 13)] - 1.0).abs() < 1e-10);

        // Seasonal cyclic shift: T[1,12]=1 (wrap), T[i+1,i]=1 for i=1..11
        assert!((ss.transition[(1, 12)] - 1.0).abs() < 1e-10);
        for i in 1..12 {
            assert!(
                (ss.transition[(i + 1, i)] - 1.0).abs() < 1e-10,
                "T[{}, {}] should be 1, got {}",
                i + 1,
                i,
                ss.transition[(i + 1, i)]
            );
        }

        // ARMA companion first entry: -reduced_ar[1]
        assert!((ss.transition[(13, 13)] - 0.9903479224371599).abs() < 1e-6);

        // Superdiagonal in ARMA block
        for i in 0..13 {
            assert!(
                (ss.transition[(13 + i, 14 + i)] - 1.0).abs() < 1e-10,
                "Superdiag T[{}, {}] should be 1",
                13 + i,
                14 + i
            );
        }
    }

    #[test]
    fn test_sarima_111_111_12_design() {
        let config = make_seasonal_config(1, 1, 1, 1, 1, 1, 12);
        let params = make_seasonal_params(&[0.99], &[0.07], &[0.001], &[-1.07]);
        let endog = vec![0.0; 300];
        let ss = StateSpace::new(&config, &params, &endog, None).unwrap();

        // Z[0] = 1 (regular diff)
        assert!((ss.design[0] - 1.0).abs() < 1e-10);
        // Z[1..12] = 0
        for i in 1..12 {
            assert!(ss.design[i].abs() < 1e-10, "Z[{}] should be 0", i);
        }
        // Z[12] = 1 (last seasonal state)
        assert!((ss.design[12] - 1.0).abs() < 1e-10);
        // Z[13] = 1 (first ARMA state)
        assert!((ss.design[13] - 1.0).abs() < 1e-10);
        // Z[14..27] = 0
        for i in 14..27 {
            assert!(ss.design[i].abs() < 1e-10, "Z[{}] should be 0", i);
        }
    }

    // ---- simple_differencing tests ----

    fn make_sd_config(p: usize, d: usize, q: usize) -> SarimaxConfig {
        SarimaxConfig {
            order: SarimaxOrder::new(p, d, q, 0, 0, 0, 0),
            simple_differencing: true,
            ..SarimaxConfig::default()
        }
    }

    fn make_sd_seasonal_config(
        p: usize, d: usize, q: usize,
        pp: usize, dd: usize, qq: usize, s: usize,
    ) -> SarimaxConfig {
        SarimaxConfig {
            order: SarimaxOrder::new(p, d, q, pp, dd, qq, s),
            simple_differencing: true,
            ..SarimaxConfig::default()
        }
    }

    #[test]
    fn test_sd_arima111_k_states() {
        // ARIMA(1,1,1) with simple_differencing: state = ARMA only
        // k_order = max(p, q+1) = max(1, 2) = 2
        // k_states = 2, k_states_diff = 0
        let config = make_sd_config(1, 1, 1);
        let params = make_params(&[-0.6441], &[0.7]);
        let endog = vec![0.0; 10];
        let ss = StateSpace::new(&config, &params, &endog, None).unwrap();

        assert_eq!(ss.k_states, 2, "k_states should be k_order=2 (no diff states)");
        assert_eq!(ss.k_states_diff, 0, "k_states_diff should be 0 (pre-differenced)");
        assert_eq!(ss.transition.nrows(), 2);
        assert_eq!(ss.transition.ncols(), 2);
    }

    #[test]
    fn test_sd_arima111_matrices() {
        // ARIMA(1,1,1) with simple_differencing: T should be pure ARMA companion
        // phi=-0.6441, theta=0.7
        // T = [[phi, 1], [0, 0]]
        // Z = [1, 0]
        // R = [[1], [theta]]
        let config = make_sd_config(1, 1, 1);
        let params = make_params(&[-0.6441], &[0.7]);
        let endog = vec![0.0; 10];
        let ss = StateSpace::new(&config, &params, &endog, None).unwrap();

        // T[0,0] = phi = -0.6441
        assert!((ss.transition[(0, 0)] - (-0.6441)).abs() < 1e-10, "T[0,0] = phi");
        // T[0,1] = 1 (superdiagonal)
        assert!((ss.transition[(0, 1)] - 1.0).abs() < 1e-10, "T[0,1] = 1");
        // T[1,0] = T[1,1] = 0
        assert!(ss.transition[(1, 0)].abs() < 1e-10, "T[1,0] = 0");
        assert!(ss.transition[(1, 1)].abs() < 1e-10, "T[1,1] = 0");

        // Z = [1, 0]
        assert!((ss.design[0] - 1.0).abs() < 1e-10, "Z[0] = 1");
        assert!(ss.design[1].abs() < 1e-10, "Z[1] = 0");

        // R = [[1], [theta]]
        assert!((ss.selection[(0, 0)] - 1.0).abs() < 1e-10, "R[0,0] = 1");
        assert!((ss.selection[(1, 0)] - 0.7).abs() < 1e-10, "R[1,0] = theta");
    }

    #[test]
    fn test_sd_sarima_111_111_12_k_states() {
        // SARIMA(1,1,1)(1,1,1,12) with simple_differencing:
        // k_order = max(1+12, 1+12+1) = max(13, 14) = 14
        // Without SD: k_states = 14 + 1 + 12 = 27
        // With SD: k_states = 14, k_states_diff = 0
        let config = make_sd_seasonal_config(1, 1, 1, 1, 1, 1, 12);
        let params = make_seasonal_params(&[0.9903], &[0.0660], &[0.0007], &[-1.0664]);
        let endog = vec![0.0; 300];
        let ss = StateSpace::new(&config, &params, &endog, None).unwrap();

        assert_eq!(ss.k_states, 14, "k_states = k_order = 14 (no diff states)");
        assert_eq!(ss.k_states_diff, 0, "k_states_diff = 0 (pre-differenced)");
        assert_eq!(ss.transition.nrows(), 14);
        assert_eq!(ss.design.len(), 14);
        assert_eq!(ss.selection.nrows(), 14);
    }

    #[test]
    fn test_sd_arima_d1_t_is_arma() {
        // ARIMA(1,1,0) with simple_differencing: T = [[phi]] (scalar)
        // k_order = max(1, 0+1) = 1, k_states = 1
        let config = make_sd_config(1, 1, 0);
        let params = make_params(&[0.5], &[]);
        let endog = vec![0.0; 10];
        let ss = StateSpace::new(&config, &params, &endog, None).unwrap();

        assert_eq!(ss.k_states, 1);
        assert_eq!(ss.k_states_diff, 0);
        // T = [[0.5]]
        assert!((ss.transition[(0, 0)] - 0.5).abs() < 1e-10, "T[0,0] = phi");
        // Z = [1]
        assert!((ss.design[0] - 1.0).abs() < 1e-10, "Z[0] = 1");
        // R = [[1]]
        assert!((ss.selection[(0, 0)] - 1.0).abs() < 1e-10, "R[0,0] = 1");
    }

    #[test]
    fn test_sd_vs_normal_same_arma_block() {
        // For ARIMA(1,1,1), the ARMA block in normal SS (rows/cols [1..3, 1..3])
        // should match the full T in SD mode (rows/cols [0..2, 0..2]).
        let phi = -0.6441;
        let theta = 0.7;

        let config_normal = make_config(1, 1, 1);
        let config_sd = make_sd_config(1, 1, 1);
        let params = make_params(&[phi], &[theta]);
        let endog = vec![0.0; 10];

        let ss_normal = StateSpace::new(&config_normal, &params, &endog, None).unwrap();
        let ss_sd = StateSpace::new(&config_sd, &params, &endog, None).unwrap();

        // Normal: k_states=3, k_states_diff=1
        // SD: k_states=2, k_states_diff=0
        // Normal ARMA block: T[1..3, 1..3]
        // SD full T: T[0..2, 0..2]
        for i in 0..2 {
            for j in 0..2 {
                let normal_val = ss_normal.transition[(1 + i, 1 + j)];
                let sd_val = ss_sd.transition[(i, j)];
                assert!(
                    (normal_val - sd_val).abs() < 1e-10,
                    "ARMA block mismatch at ({},{}) normal={} sd={}",
                    i, j, normal_val, sd_val
                );
            }
        }
    }

    #[test]
    fn test_sarima_111_111_12_selection() {
        let config = make_seasonal_config(1, 1, 1, 1, 1, 1, 12);
        let params = make_seasonal_params(
            &[0.9903479224371599],
            &[0.0659541127042639],
            &[0.0007132203797734934],
            &[-1.0663518115052784],
        );
        let endog = vec![0.0; 300];
        let ss = StateSpace::new(&config, &params, &endog, None).unwrap();

        // R[0..13, 0] = 0 (diff states)
        for i in 0..13 {
            assert!(ss.selection[(i, 0)].abs() < 1e-10);
        }
        // R[13, 0] = 1 (reduced_ma[0])
        assert!((ss.selection[(13, 0)] - 1.0).abs() < 1e-10);
        // R[14, 0] = reduced_ma[1] = ma_coeff = 0.0660
        assert!((ss.selection[(14, 0)] - 0.0659541127042639).abs() < 1e-6);
        // R[25, 0] = reduced_ma[12] = sma_coeff = -1.0664
        assert!((ss.selection[(25, 0)] - (-1.0663518115052784)).abs() < 1e-6);
        // R[26, 0] = reduced_ma[13] = ma*sma cross term
        let cross_ma = 0.0659541127042639 * (-1.0663518115052784);
        assert!((ss.selection[(26, 0)] - cross_ma).abs() < 1e-6);
    }
}
