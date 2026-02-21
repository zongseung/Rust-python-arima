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

/// Compute the (optionally concentrated) log-likelihood via the Kalman filter.
///
/// Uses the standard Harvey-form Kalman filter:
///   - a_{t|t-1}, P_{t|t-1} are the predicted state/cov at time t
///   - Innovation: v_t = y_t - Z' * a_{t|t-1}
///   - Update: a_{t|t} = a_{t|t-1} + K * v_t
///   - Predict: a_{t+1|t} = T * a_{t|t} + c_t
///
/// For concentrated scale:
///   sigma2_hat = (1/n_eff) * sum(v_t^2 / F_t)
///   loglike = -n_eff/2 * ln(2pi) - n_eff/2 * ln(sigma2_hat) - n_eff/2 - 0.5 * sum(ln F_t)
pub fn kalman_loglike(
    endog: &[f64],
    ss: &StateSpace,
    init: &KalmanInit,
    concentrate_scale: bool,
) -> Result<KalmanOutput> {
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

    // Initialize: a = a_{0|-1}, P = P_{0|-1}
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

    let mut sum_log_f = 0.0;
    let mut sum_v2_f = 0.0;
    let mut innovations = Vec::with_capacity(n);

    // Pre-allocate work buffers (zero heap allocation in the loop)
    let mut pz = DVector::<f64>::zeros(k);
    let mut a_next = DVector::<f64>::zeros(k);
    let mut temp_kk = DMatrix::<f64>::zeros(k, k);

    for t in 0..n {
        // --- Innovation ---
        let d_t = if t < ss.obs_intercept.len() {
            ss.obs_intercept[t]
        } else {
            0.0
        };
        let v_t = endog[t] - z.dot(&a) - d_t;
        innovations.push(v_t);

        // pz = P * z (in-place)
        pz.gemv(1.0, &p, z, 0.0);
        let f_t: f64 = z.dot(&pz);

        // --- Update & Predict ---
        if f_t > 0.0 {
            let f_inv = 1.0 / f_t;

            // State update: a = a + (v_t / F_t) * pz
            a.axpy(v_t * f_inv, &pz, 1.0);

            // Covariance update: P = P - (1/F_t) * pz * pz' (rank-1 downdate)
            p.ger(-f_inv, &pz, &pz, 1.0);

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
        } else if t >= burn {
            return Err(SarimaxError::DataError(format!(
                "innovation variance F_t <= 0 at t={} (F_t={}); \
                 model parameters may be numerically unstable",
                t, f_t
            )));
        } else {
            // F_t <= 0 during burn-in: skip update, predict from current state
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

    // Compute log-likelihood
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

    Ok(KalmanOutput {
        loglike,
        scale,
        innovations,
        n_obs_effective: n_eff,
    })
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

/// Run the Kalman filter and return full output including final state.
///
/// Unlike `kalman_loglike()`, this function stores innovation variances and
/// the final filtered/predicted state and covariance for use in forecasting
/// and residual diagnostics.
pub fn kalman_filter(
    endog: &[f64],
    ss: &StateSpace,
    init: &KalmanInit,
    concentrate_scale: bool,
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

    let mut sum_log_f = 0.0;
    let mut sum_v2_f = 0.0;
    let mut innovations = Vec::with_capacity(n);
    let mut innovation_vars = Vec::with_capacity(n);

    // Track the last filtered state/cov
    let mut a_filtered = DVector::<f64>::zeros(k);
    let mut p_filtered = DMatrix::<f64>::zeros(k, k);

    // Pre-allocate work buffers
    let mut pz = DVector::<f64>::zeros(k);
    let mut a_next = DVector::<f64>::zeros(k);
    let mut temp_kk = DMatrix::<f64>::zeros(k, k);

    for t in 0..n {
        // --- Innovation ---
        let d_t = if t < ss.obs_intercept.len() {
            ss.obs_intercept[t]
        } else {
            0.0
        };
        let v_t = endog[t] - z.dot(&a) - d_t;
        innovations.push(v_t);

        // pz = P * z (in-place)
        pz.gemv(1.0, &p, z, 0.0);
        let f_t: f64 = z.dot(&pz);
        innovation_vars.push(f_t);

        // --- Update & Predict ---
        if f_t > 0.0 {
            let f_inv = 1.0 / f_t;

            // State update: a = a + (v_t / F_t) * pz
            a.axpy(v_t * f_inv, &pz, 1.0);

            // Covariance update: P = P - (1/F_t) * pz * pz' (rank-1 downdate)
            p.ger(-f_inv, &pz, &pz, 1.0);

            // Store filtered state/cov (reuse allocations)
            a_filtered.copy_from(&a);
            p_filtered.copy_from(&p);

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
        } else if t >= burn {
            return Err(SarimaxError::DataError(format!(
                "innovation variance F_t <= 0 at t={} (F_t={}); \
                 model parameters may be numerically unstable",
                t, f_t
            )));
        } else {
            // F_t <= 0 during burn-in: skip update, use current state as filtered
            a_filtered.copy_from(&a);
            p_filtered.copy_from(&p);

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

    fn run_kalman_test(
        fixture_key: &str,
        p: usize,
        d: usize,
        q: usize,
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

        let config = make_config(p, d, q);
        let ar_coeffs = &params_vec[..p];
        let ma_coeffs = &params_vec[p..p + q];
        let params = make_params(ar_coeffs, ma_coeffs);

        let ss = StateSpace::new(&config, &params, &data, None).unwrap();
        let init = KalmanInit::approximate_diffuse(
            ss.k_states,
            KalmanInit::default_kappa(),
        );

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
        p: usize, d: usize, q: usize,
        pp: usize, dd: usize, qq: usize, s: usize,
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

        // Parse params: [ar(p), ma(q), sar(P), sma(Q)]
        let mut i = 0;
        let ar = &params_vec[i..i + p]; i += p;
        let ma = &params_vec[i..i + q]; i += q;
        let sar = &params_vec[i..i + pp]; i += pp;
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
        let init = KalmanInit::approximate_diffuse(
            ss.k_states,
            KalmanInit::default_kappa(),
        );

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
        run_seasonal_kalman_test(
            "sarima_100_100_4",
            1, 0, 0,  // p, d, q
            1, 0, 0, 4,  // P, D, Q, s
            1e-6,
        );
    }

    #[test]
    fn test_sarima_111_111_12_loglike() {
        run_seasonal_kalman_test(
            "sarima_111_111_12",
            1, 1, 1,  // p, d, q
            1, 1, 1, 12,  // P, D, Q, s
            1e-6,
        );
    }

    // ---- kalman_filter tests ----

    #[test]
    fn test_kalman_filter_matches_loglike() {
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();

        let config = make_config(1, 0, 0);
        let params = make_params(&[0.6527425084139002], &[]);
        let ss = StateSpace::new(&config, &params, &data, None).unwrap();
        let init = KalmanInit::approximate_diffuse(ss.k_states, 1e6);

        let lo = kalman_loglike(&data, &ss, &init, true).unwrap();
        let fo = kalman_filter(&data, &ss, &init, true).unwrap();

        assert!((lo.loglike - fo.loglike).abs() < 1e-12,
            "loglike mismatch: {} vs {}", lo.loglike, fo.loglike);
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
            .as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();

        let config = make_config(1, 0, 1);
        let params_vec: Vec<f64> = case["params"]
            .as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let params = make_params(&params_vec[..1], &params_vec[1..2]);
        let ss = StateSpace::new(&config, &params, &data, None).unwrap();
        let init = KalmanInit::approximate_diffuse(ss.k_states, 1e6);

        let fo = kalman_filter(&data, &ss, &init, true).unwrap();
        assert_eq!(fo.innovation_vars.len(), data.len());
        // After burn-in, all F_t should be positive
        for &f in &fo.innovation_vars[fo.n_obs_effective..] {
            assert!(f >= 0.0, "F_t should be non-negative, got {}", f);
        }
    }

    #[test]
    fn test_kalman_filter_state_dimensions() {
        let fixtures = load_fixtures();
        let case = &fixtures["arma11"];
        let data: Vec<f64> = case["data"]
            .as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();

        let config = make_config(1, 0, 1);
        let params_vec: Vec<f64> = case["params"]
            .as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
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
