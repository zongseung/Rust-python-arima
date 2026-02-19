use nalgebra::DMatrix;

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

    // Precompute R*Q*R' (time-invariant)
    let rqr = r_mat * q_mat * r_mat.transpose();

    let mut sum_log_f = 0.0;
    let mut sum_v2_f = 0.0;
    let mut innovations = Vec::with_capacity(n);

    let eye = DMatrix::<f64>::identity(k, k);

    for t in 0..n {
        // --- Innovation ---
        // a is a_{t|t-1} (predicted state for time t)
        // v_t = y_t - Z' * a_{t|t-1} - d_t
        let d_t = if t < ss.obs_intercept.len() {
            ss.obs_intercept[t]
        } else {
            0.0
        };
        let v_t = endog[t] - z.dot(&a) - d_t;
        innovations.push(v_t);

        // F_t = Z' * P_{t|t-1} * Z (scalar, univariate)
        let p_z = &p * z;
        let f_t: f64 = z.dot(&p_z);

        // --- Update & Predict ---
        if f_t > 0.0 {
            // Kalman gain: K = P_{t|t-1} * Z / F_t
            let k_gain = &p_z / f_t;

            // Update: a_{t|t} = a_{t|t-1} + K * v_t
            let a_updated = &a + &k_gain * v_t;

            // Joseph form: P_{t|t} = (I - K*Z') * P_{t|t-1} * (I - K*Z')'
            let k_z_t = &k_gain * z.transpose();
            let i_kz = &eye - &k_z_t;
            let p_updated = &i_kz * &p * i_kz.transpose();

            // Predict: a_{t+1|t} = T * a_{t|t} + c_t
            a = t_mat * &a_updated;
            // P_{t+1|t} = T * P_{t|t} * T' + R * Q * R'
            p = t_mat * &p_updated * t_mat.transpose() + &rqr;

            // Add state intercept c_t to predicted state
            if ss.state_intercept.len() == n * k {
                for i in 0..k {
                    a[i] += ss.state_intercept[t * k + i];
                }
            }

            // Accumulate log-likelihood terms for t >= burn
            if t >= burn {
                sum_log_f += f_t.ln();
                sum_v2_f += v_t * v_t / f_t;
            }
        } else {
            // F_t <= 0: skip update, predict from current state
            a = t_mat * &a;
            p = t_mat * &p * t_mat.transpose() + &rqr;
            if ss.state_intercept.len() == n * k {
                for i in 0..k {
                    a[i] += ss.state_intercept[t * k + i];
                }
            }
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
        // Non-concentrated: sigma2 is in Q already
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
}
