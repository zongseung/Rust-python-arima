//! Conditional Sum of Squares (CSS) objective for fast approximate optimization.
//!
//! CSS computes ARMA prediction errors via direct recursion without Kalman filter,
//! achieving O(n) per evaluation vs O(n × k³) for KF. Used as a pre-optimization
//! step to find good starting parameters for MLE (R's CSS-ML 2-stage approach).
//!
//! Reference: R `stats::arima()` source code — CSS-ML 2-stage optimization.

use crate::params::SarimaxParams;
use crate::polynomial::{reduced_ar, reduced_ma};
use crate::types::SarimaxConfig;

/// Apply non-seasonal and seasonal differencing.
///
/// Order: seasonal differencing first (D times with period s), then
/// non-seasonal differencing (d times). This matches the convention used
/// in `start_params.rs` and R's `stats::arima()`.
pub fn apply_differencing(y: &[f64], config: &SarimaxConfig) -> Vec<f64> {
    let mut result = y.to_vec();
    let s = config.order.s;

    // Seasonal differencing (D times)
    for _ in 0..config.order.dd {
        if s == 0 || result.len() <= s {
            break;
        }
        let prev = result;
        result = (s..prev.len()).map(|t| prev[t] - prev[t - s]).collect();
    }

    // Non-seasonal differencing (d times)
    for _ in 0..config.order.d {
        if result.len() <= 1 {
            break;
        }
        let prev = result;
        result = (1..prev.len()).map(|t| prev[t] - prev[t - 1]).collect();
    }

    result
}

/// Conditional Sum of Squares log-likelihood.
///
/// Uses the expanded (reduced) AR/MA polynomials from `polynomial.rs` to compute
/// one-step prediction errors via ARMA recursion on the differenced series.
///
/// The reduced AR polynomial is `[1, -α₁, -α₂, ...]` where αⱼ are the expanded
/// AR coefficients (product of non-seasonal and seasonal polynomials). Similarly,
/// the reduced MA polynomial is `[1, β₁, β₂, ...]`.
///
/// When `exog` is provided, the exogenous contribution (X·β) is subtracted from
/// the differenced series before the ARMA recursion, matching R's approach of
/// fitting ARMA on y - X·β.
///
/// Cost: O(n × (p' + q')) where p', q' are expanded polynomial orders.
/// For SARIMA(1,1,1)(1,1,1,24): p' = 25, q' = 25, so ~50 ops/observation.
/// Compare to Kalman filter O(n × k³) where k = 51: ~132,651 ops/observation.
pub fn css_loglike(endog: &[f64], config: &SarimaxConfig, params: &SarimaxParams) -> f64 {
    css_loglike_with_exog(endog, config, params, None)
}

/// CSS log-likelihood with optional exogenous variables.
///
/// `exog`: column-major `exog[j][t]` where j is regressor index, t is time.
pub fn css_loglike_with_exog(
    endog: &[f64],
    config: &SarimaxConfig,
    params: &SarimaxParams,
    exog: Option<&[Vec<f64>]>,
) -> f64 {
    let ar_poly = reduced_ar(params, &config.order);
    let ma_poly = reduced_ma(params, &config.order);

    let diffed = apply_differencing(endog, config);

    // If exog provided, difference each regressor and subtract X_diff · β
    let diffed = if let Some(exog_cols) = exog {
        if !params.exog_coeffs.is_empty() && exog_cols.len() == params.exog_coeffs.len() {
            let mut residual = diffed;
            for (j, col) in exog_cols.iter().enumerate() {
                let diffed_xj = apply_differencing(col, config);
                let n = residual.len().min(diffed_xj.len());
                // Align from the end (differencing may produce different lengths)
                let offset_r = residual.len().saturating_sub(n);
                let offset_x = diffed_xj.len().saturating_sub(n);
                for t in 0..n {
                    residual[offset_r + t] -= params.exog_coeffs[j] * diffed_xj[offset_x + t];
                }
            }
            residual
        } else {
            diffed
        }
    } else {
        diffed
    };

    // Expanded polynomial orders (excluding leading 1.0)
    let p = ar_poly.len().saturating_sub(1);
    let q = ma_poly.len().saturating_sub(1);
    let r = p.max(q);

    if r >= diffed.len() || diffed.is_empty() {
        return f64::NEG_INFINITY;
    }

    let n = diffed.len();
    let mut e = vec![0.0; n];
    let mut css = 0.0;
    let mut n_eff = 0usize;

    for t in r..n {
        let mut pred = 0.0;

        // AR part: Φ(L)y_t where Φ = [1, -α₁, -α₂, ...]
        // y_hat_t = α₁·y_{t-1} + α₂·y_{t-2} + ... and α_j = -ar_poly[j+1]
        for j in 0..p {
            pred += (-ar_poly[j + 1]) * diffed[t - 1 - j];
        }

        // MA part: Θ(L)e_t where Θ = [1, β₁, β₂, ...]
        // y_hat_t += β₁·e_{t-1} + β₂·e_{t-2} + ...
        for j in 0..q {
            pred += ma_poly[j + 1] * e[t - 1 - j];
        }

        e[t] = diffed[t] - pred;
        css += e[t] * e[t];
        n_eff += 1;
    }

    if n_eff == 0 || css <= 0.0 || !css.is_finite() {
        return f64::NEG_INFINITY;
    }

    // Concentrated CSS loglike: -n/2 · ln(σ²) - n/2
    // where σ² = CSS / n_eff
    let sigma2 = css / n_eff as f64;
    -0.5 * n_eff as f64 * sigma2.ln() - 0.5 * n_eff as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::make_seasonal_config;

    #[test]
    fn test_apply_differencing_d1() {
        let config = make_seasonal_config(1, 1, 0, 0, 0, 0, 0);
        let y = vec![1.0, 3.0, 6.0, 10.0];
        let d = apply_differencing(&y, &config);
        assert_eq!(d, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_apply_differencing_seasonal() {
        let config = make_seasonal_config(0, 0, 0, 0, 1, 0, 4);
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let d = apply_differencing(&y, &config);
        assert_eq!(d, vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_apply_differencing_d1_dd1() {
        let config = make_seasonal_config(0, 1, 0, 0, 1, 0, 4);
        let y: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let d = apply_differencing(&y, &config);
        // Seasonal diff: y[t] - y[t-4] = 4 for all t, then regular diff: 0
        assert!(d.iter().all(|&v| v.abs() < 1e-10));
    }

    #[test]
    fn test_css_loglike_ar1() {
        // Generate AR(1) process: y[t] = 0.7 * y[t-1] + e[t]
        let n = 200;
        let mut y = vec![0.0; n];
        let mut rng_state: u64 = 42;
        for t in 1..n {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
            y[t] = 0.7 * y[t - 1] + u;
        }

        let config = make_seasonal_config(1, 0, 0, 0, 0, 0, 0);

        let params_good = SarimaxParams {
            trend_coeffs: vec![],
            exog_coeffs: vec![],
            ar_coeffs: vec![0.7],
            ma_coeffs: vec![],
            sar_coeffs: vec![],
            sma_coeffs: vec![],
            sigma2: None,
        };
        let params_zero = SarimaxParams {
            trend_coeffs: vec![],
            exog_coeffs: vec![],
            ar_coeffs: vec![0.0],
            ma_coeffs: vec![],
            sar_coeffs: vec![],
            sma_coeffs: vec![],
            sigma2: None,
        };

        let ll_good = css_loglike(&y, &config, &params_good);
        let ll_zero = css_loglike(&y, &config, &params_zero);

        assert!(ll_good.is_finite());
        assert!(ll_zero.is_finite());
        assert!(
            ll_good > ll_zero,
            "CSS at true params should be better: {} vs {}",
            ll_good,
            ll_zero
        );
    }

    #[test]
    fn test_css_loglike_arima111() {
        let n = 200;
        let mut y = vec![0.0; n];
        let mut e = vec![0.0; n];
        let mut rng_state: u64 = 42;
        for t in 1..n {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
            e[t] = u;
            let yt = 0.5 * y[t - 1] + e[t] + 0.3 * e[t - 1];
            y[t] = y[t - 1] + yt; // integrate (d=1)
        }

        let config = make_seasonal_config(1, 1, 1, 0, 0, 0, 0);
        let params = SarimaxParams {
            trend_coeffs: vec![],
            exog_coeffs: vec![],
            ar_coeffs: vec![0.5],
            ma_coeffs: vec![0.3],
            sar_coeffs: vec![],
            sma_coeffs: vec![],
            sigma2: None,
        };

        let ll = css_loglike(&y, &config, &params);
        assert!(ll.is_finite(), "CSS loglike should be finite: {}", ll);
    }

    #[test]
    fn test_css_loglike_sarima() {
        let n = 300;
        let s = 12;
        let mut y = vec![0.0; n];
        let mut e = vec![0.0; n];
        let mut rng_state: u64 = 42;
        for t in 1..n {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
            e[t] = u;
            y[t] = e[t];
            if t >= 1 {
                y[t] += 0.5 * y[t - 1];
            }
            if t >= s {
                y[t] += 0.3 * y[t - s];
            }
            if t >= 1 {
                y[t] += 0.2 * e[t - 1];
            }
            if t >= s {
                y[t] += -0.4 * e[t - s];
            }
        }

        let config = make_seasonal_config(1, 0, 1, 1, 0, 1, 12);
        let params = SarimaxParams {
            trend_coeffs: vec![],
            exog_coeffs: vec![],
            ar_coeffs: vec![0.5],
            ma_coeffs: vec![0.2],
            sar_coeffs: vec![0.3],
            sma_coeffs: vec![-0.4],
            sigma2: None,
        };

        let ll = css_loglike(&y, &config, &params);
        assert!(ll.is_finite(), "SARIMA CSS loglike should be finite: {}", ll);
    }

    #[test]
    fn test_css_loglike_empty_series() {
        let config = make_seasonal_config(1, 0, 0, 0, 0, 0, 0);
        let params = SarimaxParams {
            trend_coeffs: vec![],
            exog_coeffs: vec![],
            ar_coeffs: vec![0.5],
            ma_coeffs: vec![],
            sar_coeffs: vec![],
            sma_coeffs: vec![],
            sigma2: None,
        };

        let ll = css_loglike(&[], &config, &params);
        assert_eq!(ll, f64::NEG_INFINITY);
    }

    #[test]
    fn test_css_loglike_short_series() {
        let config = make_seasonal_config(1, 1, 1, 1, 1, 1, 12);
        let y: Vec<f64> = (0..5).map(|i| i as f64).collect();
        let params = SarimaxParams {
            trend_coeffs: vec![],
            exog_coeffs: vec![],
            ar_coeffs: vec![0.5],
            ma_coeffs: vec![0.3],
            sar_coeffs: vec![0.2],
            sma_coeffs: vec![-0.4],
            sigma2: None,
        };

        let ll = css_loglike(&y, &config, &params);
        assert_eq!(ll, f64::NEG_INFINITY);
    }
}
