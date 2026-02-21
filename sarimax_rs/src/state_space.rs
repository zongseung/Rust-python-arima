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
    pub k_posdef: usize, // always 1 for univariate ARIMA
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

        if config.simple_differencing {
            return Err(SarimaxError::StateSpaceError(
                "simple_differencing is not yet supported".into(),
            ));
        }

        if config.measurement_error {
            return Err(SarimaxError::StateSpaceError(
                "measurement_error is not yet supported".into(),
            ));
        }

        let k_states = order.k_states();
        let k_states_diff = order.k_states_diff();
        let _k_order = order.k_order();
        let k_posdef = 1;
        let n = endog.len();

        // V-3: Validate exog column lengths match number of observations
        if let Some(ref x) = exog {
            for (j, col) in x.iter().enumerate() {
                if col.len() != n {
                    return Err(SarimaxError::DataError(format!(
                        "exog column {} has {} rows but y has {} observations",
                        j, col.len(), n
                    )));
                }
            }
        }

        // Build matrices
        let transition = Self::build_transition(config, params)?;
        let design = Self::build_design(config);
        let selection = Self::build_selection(config, params)?;
        let state_cov = Self::build_state_cov(config, params);

        // Observation intercept: d_t = exog * beta
        let obs_intercept = Self::build_obs_intercept(n, params, exog);

        // State intercept: c_t (trend contribution)
        let state_intercept = Self::build_state_intercept(n, k_states, k_states_diff, config, params);

        if transition.nrows() != k_states || transition.ncols() != k_states {
            return Err(SarimaxError::StateSpaceError(format!(
                "T matrix dimension mismatch: expected {}×{}, got {}×{}",
                k_states, k_states, transition.nrows(), transition.ncols()
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

    /// Build the transition matrix T.
    ///
    /// Structure for SARIMA(p,d,q)(P,D,Q,s):
    /// 1. Regular diff block [0..d, 0..d]: upper triangular ones
    /// 2. Seasonal diff blocks: D layers of s×s cyclic shift
    /// 3. Cross-diff: regular diff states → last seasonal state
    /// 4. Diff → ARMA: regular diff + first seasonal of each layer → ARMA
    /// 5. ARMA companion [sd..sd+ko, sd..sd+ko]
    fn build_transition(
        config: &SarimaxConfig,
        params: &SarimaxParams,
    ) -> Result<DMatrix<f64>> {
        let order = &config.order;
        let k_states = order.k_states();
        let d = order.d;
        let dd = order.dd;
        let s = order.s;
        let sd = order.k_states_diff();
        let ko = order.k_order();

        let mut t = DMatrix::<f64>::zeros(k_states, k_states);

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

    /// Build the design vector Z.
    ///
    /// Z[i] = 1 for i in 0..d (regular diff states)
    /// Z[d + (layer+1)*s - 1] = 1 for each seasonal layer (last state of layer)
    /// Z[sd] = 1 (first ARMA state)
    fn build_design(config: &SarimaxConfig) -> DVector<f64> {
        let order = &config.order;
        let k_states = order.k_states();
        let d = order.d;
        let dd = order.dd;
        let s = order.s;
        let sd = order.k_states_diff();

        let mut z = DVector::<f64>::zeros(k_states);

        // Regular diff states
        for i in 0..d {
            z[i] = 1.0;
        }

        // Last state of each seasonal layer
        for layer in 0..dd {
            z[d + (layer + 1) * s - 1] = 1.0;
        }

        // First ARMA state
        if sd < k_states {
            z[sd] = 1.0;
        }

        z
    }

    /// Build the selection matrix R (k_states × k_posdef).
    ///
    /// R[d, 0] = 1
    /// R[d+i, 0] = reduced_ma[i] for i >= 1
    fn build_selection(
        config: &SarimaxConfig,
        params: &SarimaxParams,
    ) -> Result<DMatrix<f64>> {
        let order = &config.order;
        let k_states = order.k_states();
        let sd = order.k_states_diff();
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
            Some(x) if !params.exog_coeffs.is_empty() => {
                (0..n)
                    .map(|t| {
                        x.iter()
                            .zip(params.exog_coeffs.iter())
                            .map(|(col, &b)| col[t] * b)
                            .sum()
                    })
                    .collect()
            }
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
                Trend::Both => {
                    params.trend_coeffs[0] + params.trend_coeffs[1] * (t as f64)
                }
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
    use crate::params::SarimaxParams;
    use crate::types::{SarimaxConfig, SarimaxOrder, Trend};

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

    fn make_seasonal_config(
        p: usize, d: usize, q: usize,
        pp: usize, dd: usize, qq: usize, s: usize,
    ) -> SarimaxConfig {
        SarimaxConfig {
            order: SarimaxOrder::new(p, d, q, pp, dd, qq, s),
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

    fn make_seasonal_params(
        ar: &[f64], ma: &[f64],
        sar: &[f64], sma: &[f64],
    ) -> SarimaxParams {
        SarimaxParams {
            trend_coeffs: vec![],
            exog_coeffs: vec![],
            ar_coeffs: ar.to_vec(),
            ma_coeffs: ma.to_vec(),
            sar_coeffs: sar.to_vec(),
            sma_coeffs: sma.to_vec(),
            sigma2: None,
        }
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
        let params = make_seasonal_params(
            &[0.7671699347442852], &[],
            &[0.2322174491752982], &[],
        );
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
        let params = make_seasonal_params(
            &[0.9903], &[0.0660],
            &[0.0007], &[-1.0664],
        );
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
            &[0.9903479224371599], &[0.0659541127042639],
            &[0.0007132203797734934], &[-1.0663518115052784],
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
                i + 1, i, ss.transition[(i + 1, i)]
            );
        }

        // ARMA companion first entry: -reduced_ar[1]
        assert!((ss.transition[(13, 13)] - 0.9903479224371599).abs() < 1e-6);

        // Superdiagonal in ARMA block
        for i in 0..13 {
            assert!(
                (ss.transition[(13 + i, 14 + i)] - 1.0).abs() < 1e-10,
                "Superdiag T[{}, {}] should be 1",
                13 + i, 14 + i
            );
        }
    }

    #[test]
    fn test_sarima_111_111_12_design() {
        let config = make_seasonal_config(1, 1, 1, 1, 1, 1, 12);
        let params = make_seasonal_params(
            &[0.99], &[0.07], &[0.001], &[-1.07],
        );
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

    #[test]
    fn test_sarima_111_111_12_selection() {
        let config = make_seasonal_config(1, 1, 1, 1, 1, 1, 12);
        let params = make_seasonal_params(
            &[0.9903479224371599], &[0.0659541127042639],
            &[0.0007132203797734934], &[-1.0663518115052784],
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
