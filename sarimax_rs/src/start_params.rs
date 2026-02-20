//! Initial parameter estimation via CSS (Conditional Sum of Squares).
//!
//! Provides reasonable starting values for the optimizer by:
//! 1. Differencing the series (regular + seasonal)
//! 2. Estimating AR coefficients via Yule-Walker equations
//! 3. Estimating MA coefficients from AR residuals
//! 4. Falling back to zeros on failure

use crate::error::Result;
use crate::types::SarimaxConfig;

/// Apply regular differencing d times.
fn difference(y: &[f64], d: usize) -> Vec<f64> {
    let mut out = y.to_vec();
    for _ in 0..d {
        let prev = out.clone();
        out = prev.windows(2).map(|w| w[1] - w[0]).collect();
    }
    out
}

/// Apply seasonal differencing D times with period s.
fn seasonal_difference(y: &[f64], d: usize, s: usize) -> Vec<f64> {
    if s == 0 {
        return y.to_vec();
    }
    let mut out = y.to_vec();
    for _ in 0..d {
        if out.len() <= s {
            return vec![];
        }
        let prev = out.clone();
        out = (s..prev.len()).map(|i| prev[i] - prev[i - s]).collect();
    }
    out
}

/// Compute sample autocovariance at lag k.
fn autocovariance(y: &[f64], k: usize) -> f64 {
    let n = y.len();
    if k >= n {
        return 0.0;
    }
    let mean: f64 = y.iter().sum::<f64>() / n as f64;
    let mut sum = 0.0;
    for i in 0..n - k {
        sum += (y[i] - mean) * (y[i + k] - mean);
    }
    sum / n as f64
}

/// Estimate AR coefficients via Yule-Walker equations.
///
/// Solves the Yule-Walker system: R * phi = r
/// where R[i,j] = gamma(|i-j|) and r[i] = gamma(i+1).
fn yule_walker(y: &[f64], p: usize) -> Option<Vec<f64>> {
    if p == 0 || y.len() <= p {
        return Some(vec![]);
    }

    let gamma0 = autocovariance(y, 0);
    if gamma0.abs() < 1e-15 {
        return None;
    }

    // Build Toeplitz matrix R and vector r
    let gammas: Vec<f64> = (0..=p).map(|k| autocovariance(y, k)).collect();

    // Levinson-Durbin recursion for efficient Toeplitz solve
    let mut phi = vec![0.0; p];
    let mut phi_prev = vec![0.0; p];
    let mut var = gammas[0];

    for k in 0..p {
        // Compute reflection coefficient
        let mut num = gammas[k + 1];
        for j in 0..k {
            num -= phi[j] * gammas[k - j];
        }
        if var.abs() < 1e-15 {
            return None;
        }
        let lambda = num / var;

        // Update coefficients
        phi_prev[..p].copy_from_slice(&phi[..p]);
        phi[k] = lambda;
        for j in 0..k {
            phi[j] = phi_prev[j] - lambda * phi_prev[k - 1 - j];
        }
        var *= 1.0 - lambda * lambda;
    }

    Some(phi)
}

/// Estimate MA coefficients from AR residuals using innovation algorithm.
fn estimate_ma_from_residuals(residuals: &[f64], q: usize) -> Vec<f64> {
    if q == 0 || residuals.len() <= q {
        return vec![0.0; q];
    }

    let gamma0 = autocovariance(residuals, 0);
    if gamma0.abs() < 1e-15 {
        return vec![0.0; q];
    }

    // Simple method: use autocorrelations of residuals as MA coefficients
    // This is a rough estimate but sufficient for starting values
    let mut ma = Vec::with_capacity(q);
    for k in 1..=q {
        let rho = autocovariance(residuals, k) / gamma0;
        // Clamp to prevent extreme values
        ma.push(rho.clamp(-0.9, 0.9));
    }
    ma
}

/// Compute AR residuals given coefficients.
fn ar_residuals(y: &[f64], ar: &[f64]) -> Vec<f64> {
    let p = ar.len();
    if p == 0 {
        return y.to_vec();
    }
    let n = y.len();
    let mut resid = Vec::with_capacity(n.saturating_sub(p));
    for t in p..n {
        let mut pred = 0.0;
        for j in 0..p {
            pred += ar[j] * y[t - 1 - j];
        }
        resid.push(y[t] - pred);
    }
    resid
}

/// Compute starting parameters for SARIMAX model.
///
/// Returns a flat parameter vector in the layout expected by `SarimaxParams::from_flat`:
/// `[trend | exog | ar(p) | ma(q) | sar(P) | sma(Q)]`
/// (sigma2 omitted when `concentrate_scale=true`)
pub fn compute_start_params(endog: &[f64], config: &SarimaxConfig) -> Result<Vec<f64>> {
    let order = &config.order;
    let p = order.p;
    let q = order.q;
    let pp = order.pp;
    let qq = order.qq;
    let s = order.s;

    let n_params = config.trend.k_trend()
        + config.n_exog
        + p + q + pp + qq
        + if config.concentrate_scale { 0 } else { 1 };

    // Apply differencing
    let diffed = difference(endog, order.d);
    let diffed = seasonal_difference(&diffed, order.dd, s);

    if diffed.len() < 3 {
        return Ok(vec![0.0; n_params]);
    }

    // Trend coefficients (zeros)
    let kt = config.trend.k_trend();
    let mut params = vec![0.0; kt + config.n_exog];

    // AR coefficients
    let ar = yule_walker(&diffed, p).unwrap_or_else(|| vec![0.0; p]);
    params.extend_from_slice(&ar);

    // MA coefficients from AR residuals
    let residuals = ar_residuals(&diffed, &ar);
    let ma = estimate_ma_from_residuals(&residuals, q);
    params.extend_from_slice(&ma);

    // Seasonal AR coefficients
    if pp > 0 && s > 0 {
        // Subsample at seasonal lag
        let seasonal_data: Vec<f64> = diffed.iter().step_by(s).copied().collect();
        let sar = if seasonal_data.len() > pp {
            yule_walker(&seasonal_data, pp).unwrap_or_else(|| vec![0.0; pp])
        } else {
            vec![0.0; pp]
        };
        params.extend_from_slice(&sar);
    } else {
        params.extend(vec![0.0; pp]);
    }

    // Seasonal MA coefficients
    if qq > 0 && s > 0 {
        let sar_coeffs = if pp > 0 && s > 0 {
            let seasonal_data: Vec<f64> = diffed.iter().step_by(s).copied().collect();
            yule_walker(&seasonal_data, pp).unwrap_or_else(|| vec![0.0; pp])
        } else {
            vec![]
        };
        let seasonal_resid: Vec<f64> = diffed.iter().step_by(s).copied().collect();
        let seasonal_resid = ar_residuals(&seasonal_resid, &sar_coeffs);
        let sma = estimate_ma_from_residuals(&seasonal_resid, qq);
        params.extend_from_slice(&sma);
    } else {
        params.extend(vec![0.0; qq]);
    }

    // sigma2 if not concentrated
    if !config.concentrate_scale {
        let var = autocovariance(&diffed, 0).max(1e-6);
        params.push(var);
    }

    assert_eq!(params.len(), n_params);
    Ok(params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SarimaxConfig, SarimaxOrder, Trend};

    fn make_config(
        p: usize, d: usize, q: usize,
        pp: usize, dd: usize, qq: usize, s: usize,
        concentrate: bool,
    ) -> SarimaxConfig {
        SarimaxConfig {
            order: SarimaxOrder::new(p, d, q, pp, dd, qq, s),
            n_exog: 0,
            trend: Trend::None,
            enforce_stationarity: false,
            enforce_invertibility: false,
            concentrate_scale: concentrate,
            simple_differencing: false,
            measurement_error: false,
        }
    }

    #[test]
    fn test_difference_once() {
        let y = vec![1.0, 3.0, 6.0, 10.0];
        let d = difference(&y, 1);
        assert_eq!(d, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_difference_twice() {
        let y = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        let d = difference(&y, 2);
        assert_eq!(d, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_seasonal_difference() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let d = seasonal_difference(&y, 1, 4);
        assert_eq!(d, vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_yule_walker_ar1() {
        // Generate AR(1) process with phi=0.7
        let n = 500;
        let mut y = vec![0.0; n];
        // Use a simple LCG for deterministic pseudo-random
        let mut rng_state: u64 = 42;
        for t in 1..n {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
            y[t] = 0.7 * y[t - 1] + u;
        }
        let ar = yule_walker(&y, 1).unwrap();
        assert_eq!(ar.len(), 1);
        // Should be roughly near 0.7 (not exact due to finite sample)
        assert!((ar[0] - 0.7).abs() < 0.15, "AR(1) estimate too far: {}", ar[0]);
    }

    #[test]
    fn test_start_params_length_ar1() {
        let config = make_config(1, 0, 0, 0, 0, 0, 0, true);
        let y: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        let params = compute_start_params(&y, &config).unwrap();
        assert_eq!(params.len(), 1); // ar(1)
    }

    #[test]
    fn test_start_params_length_arma11() {
        let config = make_config(1, 0, 1, 0, 0, 0, 0, true);
        let y: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        let params = compute_start_params(&y, &config).unwrap();
        assert_eq!(params.len(), 2); // ar(1) + ma(1)
    }

    #[test]
    fn test_start_params_length_sarima() {
        let config = make_config(1, 1, 1, 1, 1, 1, 12, true);
        let y: Vec<f64> = (0..300).map(|i| (i as f64 * 0.1).sin() + (i as f64 * 0.01).cos()).collect();
        let params = compute_start_params(&y, &config).unwrap();
        assert_eq!(params.len(), 4); // ar(1) + ma(1) + sar(1) + sma(1)
    }

    #[test]
    fn test_start_params_with_sigma2() {
        let config = make_config(1, 0, 0, 0, 0, 0, 0, false);
        let y: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        let params = compute_start_params(&y, &config).unwrap();
        assert_eq!(params.len(), 2); // ar(1) + sigma2
        assert!(params[1] > 0.0); // sigma2 should be positive
    }

    #[test]
    fn test_fallback_short_series() {
        let config = make_config(1, 1, 1, 0, 0, 0, 0, true);
        let y = vec![1.0, 2.0]; // Too short after differencing
        let params = compute_start_params(&y, &config).unwrap();
        assert_eq!(params.len(), 2); // ar(1) + ma(1)
        assert!(params.iter().all(|&x| x == 0.0)); // Should be zeros
    }

    #[test]
    fn test_start_params_finite() {
        let config = make_config(2, 1, 1, 0, 0, 0, 0, true);
        let y: Vec<f64> = (0..200).map(|i| (i as f64 * 0.05).sin()).collect();
        let params = compute_start_params(&y, &config).unwrap();
        assert!(params.iter().all(|x| x.is_finite()), "Non-finite start params: {:?}", params);
    }
}
