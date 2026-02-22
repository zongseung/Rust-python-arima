//! Initial parameter estimation via CSS (Conditional Sum of Squares).
//!
//! Provides reasonable starting values for the optimizer by:
//! 1. Differencing the series (regular + seasonal)
//! 2. Estimating AR coefficients via Yule-Walker equations
//! 3. Estimating MA coefficients from AR residuals
//! 4. Falling back to zeros on failure

use crate::error::{Result, SarimaxError};
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

/// Estimate AR coefficients via Burg's maximum entropy method.
///
/// Burg's method has less finite-sample bias than Yule-Walker for high-order AR,
/// because it minimizes the sum of forward and backward prediction errors directly
/// from the data rather than going through autocovariance estimates. It also
/// guarantees a stable (stationary) AR model at each stage.
fn burg_ar(y: &[f64], p: usize) -> Option<Vec<f64>> {
    if p == 0 {
        return Some(vec![]);
    }
    let n = y.len();
    if n <= p {
        return None;
    }

    let mean: f64 = y.iter().sum::<f64>() / n as f64;

    // Initialize forward and backward prediction errors
    let mut ef: Vec<f64> = y.iter().map(|&v| v - mean).collect();
    let mut eb: Vec<f64> = ef.clone();

    let mut a = vec![0.0; p];

    for k in 0..p {
        // Compute reflection coefficient
        let mut num = 0.0;
        let mut den = 0.0;
        for t in (k + 1)..n {
            num += ef[t] * eb[t - 1];
            den += ef[t] * ef[t] + eb[t - 1] * eb[t - 1];
        }
        if den.abs() < 1e-15 {
            return None;
        }
        let kk = 2.0 * num / den;

        // Stability check: |kk| < 1 (should always hold for Burg, but be safe)
        if kk.abs() >= 1.0 {
            return None;
        }

        // Update AR coefficients via Levinson recursion
        let a_prev: Vec<f64> = a[..k].to_vec();
        a[k] = kk;
        for j in 0..k {
            a[j] = a_prev[j] - kk * a_prev[k - 1 - j];
        }

        // Update prediction errors (reverse iteration to avoid overwriting
        // eb[t-1] before it's read at the next t)
        for t in ((k + 1)..n).rev() {
            let ef_t = ef[t];
            ef[t] = ef_t - kk * eb[t - 1];
            eb[t] = eb[t - 1] - kk * ef_t;
        }
    }

    Some(a)
}

/// Estimate AR coefficients via Yule-Walker equations.
///
/// Solves the Yule-Walker system: R * phi = r
/// where R[i,j] = gamma(|i-j|) and r[i] = gamma(i+1).
fn yule_walker(y: &[f64], p: usize) -> Option<Vec<f64>> {
    if p == 0 {
        return Some(vec![]);
    }
    if y.len() <= p {
        return None;
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

/// Solve Yule-Walker equations from a pre-computed autocovariance sequence.
///
/// `gammas` should contain gamma(0), gamma(1), ..., gamma(p) where these may
/// be at seasonal lags (i.e. gamma[k] = autocovariance at lag k*s).
/// Returns AR coefficients [phi_1, ..., phi_p].
fn yule_walker_from_acov(gammas: &[f64], p: usize) -> Option<Vec<f64>> {
    if p == 0 || gammas.len() <= p {
        return Some(vec![]);
    }

    if gammas[0].abs() < 1e-15 {
        return None;
    }

    // Levinson-Durbin recursion for efficient Toeplitz solve
    let mut phi = vec![0.0; p];
    let mut phi_prev = vec![0.0; p];
    let mut var = gammas[0];

    for k in 0..p {
        let mut num = gammas[k + 1];
        for j in 0..k {
            num -= phi[j] * gammas[k - j];
        }
        if var.abs() < 1e-15 {
            return None;
        }
        let lambda = num / var;

        phi_prev[..p].copy_from_slice(&phi[..p]);
        phi[k] = lambda;
        for j in 0..k {
            phi[j] = phi_prev[j] - lambda * phi_prev[k - 1 - j];
        }
        var *= 1.0 - lambda * lambda;
    }

    Some(phi)
}

/// Estimate MA coefficients via the innovation algorithm (Brockwell & Davis).
///
/// More accurate than raw autocorrelation because the innovation algorithm
/// recovers the MA structure from the autocovariance sequence directly.
fn estimate_ma_from_residuals(residuals: &[f64], q: usize) -> Vec<f64> {
    if q == 0 || residuals.len() <= q {
        return vec![0.0; q];
    }

    // Compute autocovariances gamma(0..q)
    let gamma: Vec<f64> = (0..=q).map(|k| autocovariance(residuals, k)).collect();

    if gamma[0].abs() < 1e-15 {
        return vec![0.0; q];
    }

    // Innovation algorithm (Brockwell & Davis, Sec 5.2)
    // Computes theta[i][j] and v[i] iteratively
    let m = q;
    let mut theta = vec![vec![0.0; m]; m + 1]; // theta[i][j], 0-indexed
    let mut v = vec![0.0; m + 1];
    v[0] = gamma[0];

    for i in 1..=m {
        // Compute theta[i][i-1-k] for k = 0..i
        for k in 0..i {
            let mut sum = gamma[i - k];
            for j in 0..k {
                sum -= theta[k][k - 1 - j] * theta[i][i - 1 - j] * v[j];
            }
            theta[i][i - 1 - k] = if v[k].abs() > 1e-15 { sum / v[k] } else { 0.0 };
        }
        // Update v[i]
        v[i] = gamma[0];
        for j in 0..i {
            v[i] -= theta[i][i - 1 - j].powi(2) * v[j];
        }
        v[i] = v[i].max(1e-15);
    }

    // Extract MA(q) coefficients from theta[m][0..q]
    (0..q).map(|k| theta[m][k].clamp(-0.99, 0.99)).collect()
}

/// Estimate seasonal MA coefficients from autocovariances at seasonal lags.
///
/// Instead of subsampling every s-th observation (which discards most data),
/// computes autocovariances at lags 0, s, 2s, ..., Q*s from the full series
/// and applies the innovation algorithm on these seasonal autocovariances.
fn estimate_seasonal_ma(residuals: &[f64], qq: usize, s: usize) -> Vec<f64> {
    if qq == 0 || s == 0 || residuals.len() <= qq * s {
        return vec![0.0; qq];
    }

    // Compute autocovariances at seasonal lags: γ(0), γ(s), γ(2s), ..., γ(Q*s)
    let gamma: Vec<f64> = (0..=qq).map(|k| autocovariance(residuals, k * s)).collect();

    if gamma[0].abs() < 1e-15 {
        return vec![0.0; qq];
    }

    // Innovation algorithm treating seasonal lags as consecutive
    let m = qq;
    let mut theta = vec![vec![0.0; m]; m + 1];
    let mut v = vec![0.0; m + 1];
    v[0] = gamma[0];

    for i in 1..=m {
        for k in 0..i {
            let mut sum = gamma[i - k];
            for j in 0..k {
                sum -= theta[k][k - 1 - j] * theta[i][i - 1 - j] * v[j];
            }
            theta[i][i - 1 - k] = if v[k].abs() > 1e-15 { sum / v[k] } else { 0.0 };
        }
        v[i] = gamma[0];
        for j in 0..i {
            v[i] -= theta[i][i - 1 - j].powi(2) * v[j];
        }
        v[i] = v[i].max(1e-15);
    }

    (0..qq).map(|k| theta[m][k].clamp(-0.99, 0.99)).collect()
}

/// Compute AR residuals given coefficients (lag-1 recursion).
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

/// Compute seasonal AR residuals at seasonal lags (lag s, 2s, ..., P*s).
///
/// y[t] - sar[0]*y[t-s] - sar[1]*y[t-2s] - ... - sar[P-1]*y[t-P*s]
fn seasonal_ar_residuals(y: &[f64], sar: &[f64], s: usize) -> Vec<f64> {
    let pp = sar.len();
    if pp == 0 || s == 0 {
        return y.to_vec();
    }
    let n = y.len();
    let start = pp * s;
    if n <= start {
        return vec![];
    }
    let mut resid = Vec::with_capacity(n - start);
    for t in start..n {
        let mut pred = 0.0;
        for j in 0..pp {
            pred += sar[j] * y[t - (j + 1) * s];
        }
        resid.push(y[t] - pred);
    }
    resid
}

/// Estimate exogenous variable coefficients via simple regression.
///
/// For each exog variable j, computes β_j = cov(y, x_j) / var(x_j).
fn estimate_exog_coeffs(endog: &[f64], exog: &[Vec<f64>]) -> Vec<f64> {
    let n = endog.len();
    if n == 0 {
        return vec![0.0; exog.len()];
    }
    let y_mean: f64 = endog.iter().sum::<f64>() / n as f64;

    exog.iter()
        .map(|col| {
            let x_mean: f64 = col.iter().sum::<f64>() / n as f64;
            let mut cov = 0.0;
            let mut var_x = 0.0;
            for t in 0..n.min(col.len()) {
                let dx = col[t] - x_mean;
                cov += (endog[t] - y_mean) * dx;
                var_x += dx * dx;
            }
            if var_x.abs() < 1e-15 {
                0.0
            } else {
                cov / var_x
            }
        })
        .collect()
}

/// Compute starting parameters for SARIMAX model.
///
/// Returns a flat parameter vector in the layout expected by `SarimaxParams::from_flat`:
/// `[trend | exog | ar(p) | ma(q) | sar(P) | sma(Q)]`
/// (sigma2 omitted when `concentrate_scale=true`)
pub fn compute_start_params(
    endog: &[f64],
    config: &SarimaxConfig,
    exog: Option<&[Vec<f64>]>,
) -> Result<Vec<f64>> {
    let order = &config.order;
    let p = order.p;
    let q = order.q;
    let pp = order.pp;
    let qq = order.qq;
    let s = order.s;

    let n_params = config.trend.k_trend()
        + config.n_exog
        + p
        + q
        + pp
        + qq
        + if config.concentrate_scale { 0 } else { 1 };

    // Apply differencing
    let diffed = difference(endog, order.d);
    let diffed = seasonal_difference(&diffed, order.dd, s);

    if diffed.len() < 3 {
        return Ok(vec![0.0; n_params]);
    }

    // Trend coefficients (zeros)
    let kt = config.trend.k_trend();
    let mut params = vec![0.0; kt];

    // Exog coefficients: OLS estimates or zeros
    if config.n_exog > 0 {
        if let Some(exog_cols) = exog {
            let exog_betas = estimate_exog_coeffs(&diffed, exog_cols);
            params.extend_from_slice(&exog_betas);
        } else {
            params.extend(vec![0.0; config.n_exog]);
        }
    }

    // AR coefficients (Burg primary, Yule-Walker fallback)
    let ar = burg_ar(&diffed, p)
        .or_else(|| yule_walker(&diffed, p))
        .unwrap_or_else(|| vec![0.0; p]);
    params.extend_from_slice(&ar);

    // MA coefficients from AR residuals
    let residuals = ar_residuals(&diffed, &ar);
    let ma = estimate_ma_from_residuals(&residuals, q);
    params.extend_from_slice(&ma);

    // Seasonal AR coefficients via Yule-Walker on seasonal autocovariances.
    // Instead of subsampling every s-th observation (which discards most data),
    // compute autocovariances at lags 0, s, 2s, ..., P*s from ALL observations
    // and solve Yule-Walker equations on those.
    if pp > 0 && s > 0 && diffed.len() > pp * s {
        let seasonal_gammas: Vec<f64> = (0..=pp).map(|k| autocovariance(&diffed, k * s)).collect();
        let sar = yule_walker_from_acov(&seasonal_gammas, pp).unwrap_or_else(|| vec![0.0; pp]);
        params.extend_from_slice(&sar);
    } else {
        params.extend(vec![0.0; pp]);
    }

    // Seasonal MA coefficients
    if qq > 0 && s > 0 {
        // Filter through seasonal AR at lags s, 2s, ..., P*s to get proper residuals
        let sar_resid = if pp > 0 {
            seasonal_ar_residuals(
                &diffed,
                &params[kt + config.n_exog + p + q..kt + config.n_exog + p + q + pp],
                s,
            )
        } else {
            residuals.clone()
        };
        let sma = estimate_seasonal_ma(&sar_resid, qq, s);
        params.extend_from_slice(&sma);
    } else {
        params.extend(vec![0.0; qq]);
    }

    // sigma2 if not concentrated
    if !config.concentrate_scale {
        let var = autocovariance(&diffed, 0).max(1e-6);
        params.push(var);
    }

    if params.len() != n_params {
        return Err(SarimaxError::DataError(format!(
            "failed to build start params: expected length {}, got {}",
            n_params,
            params.len()
        )));
    }
    Ok(params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SarimaxConfig, SarimaxOrder, Trend};

    fn make_config(
        p: usize,
        d: usize,
        q: usize,
        pp: usize,
        dd: usize,
        qq: usize,
        s: usize,
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
    fn test_burg_ar1() {
        // Generate AR(1) process with phi=0.7
        let n = 500;
        let mut y = vec![0.0; n];
        let mut rng_state: u64 = 42;
        for t in 1..n {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
            y[t] = 0.7 * y[t - 1] + u;
        }
        let ar = burg_ar(&y, 1).unwrap();
        assert_eq!(ar.len(), 1);
        assert!(
            (ar[0] - 0.7).abs() < 0.15,
            "Burg AR(1) estimate too far: {}",
            ar[0]
        );
    }

    #[test]
    fn test_burg_ar5() {
        // Generate AR(5) process
        let n = 1000;
        let phi = [0.3, -0.2, 0.15, -0.1, 0.05];
        let mut y = vec![0.0; n];
        let mut rng_state: u64 = 42;
        for t in 5..n {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
            y[t] = phi[0] * y[t - 1]
                + phi[1] * y[t - 2]
                + phi[2] * y[t - 3]
                + phi[3] * y[t - 4]
                + phi[4] * y[t - 5]
                + u;
        }
        let ar = burg_ar(&y, 5).unwrap();
        assert_eq!(ar.len(), 5);
        // Check first coefficient is roughly in the right direction
        assert!(
            (ar[0] - phi[0]).abs() < 0.2,
            "Burg AR(5)[0] estimate too far: {} (expected {})",
            ar[0],
            phi[0]
        );
    }

    #[test]
    fn test_burg_stability() {
        // Burg should always produce stable (stationary) coefficients
        let n = 200;
        let mut y = vec![0.0; n];
        let mut rng_state: u64 = 123;
        for t in 1..n {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
            y[t] = 0.95 * y[t - 1] + u;
        }
        // High-order Burg should still produce a result
        let ar = burg_ar(&y, 8);
        assert!(ar.is_some(), "Burg should succeed for AR(8) estimation");
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
        assert!(
            (ar[0] - 0.7).abs() < 0.15,
            "AR(1) estimate too far: {}",
            ar[0]
        );
    }

    #[test]
    fn test_start_params_length_ar1() {
        let config = make_config(1, 0, 0, 0, 0, 0, 0, true);
        let y: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        let params = compute_start_params(&y, &config, None).unwrap();
        assert_eq!(params.len(), 1); // ar(1)
    }

    #[test]
    fn test_start_params_length_arma11() {
        let config = make_config(1, 0, 1, 0, 0, 0, 0, true);
        let y: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        let params = compute_start_params(&y, &config, None).unwrap();
        assert_eq!(params.len(), 2); // ar(1) + ma(1)
    }

    #[test]
    fn test_start_params_length_sarima() {
        let config = make_config(1, 1, 1, 1, 1, 1, 12, true);
        let y: Vec<f64> = (0..300)
            .map(|i| (i as f64 * 0.1).sin() + (i as f64 * 0.01).cos())
            .collect();
        let params = compute_start_params(&y, &config, None).unwrap();
        assert_eq!(params.len(), 4); // ar(1) + ma(1) + sar(1) + sma(1)
    }

    #[test]
    fn test_start_params_with_sigma2() {
        let config = make_config(1, 0, 0, 0, 0, 0, 0, false);
        let y: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        let params = compute_start_params(&y, &config, None).unwrap();
        assert_eq!(params.len(), 2); // ar(1) + sigma2
        assert!(params[1] > 0.0); // sigma2 should be positive
    }

    #[test]
    fn test_fallback_short_series() {
        let config = make_config(1, 1, 1, 0, 0, 0, 0, true);
        let y = vec![1.0, 2.0]; // Too short after differencing
        let params = compute_start_params(&y, &config, None).unwrap();
        assert_eq!(params.len(), 2); // ar(1) + ma(1)
        assert!(params.iter().all(|&x| x == 0.0)); // Should be zeros
    }

    #[test]
    fn test_start_params_finite() {
        let config = make_config(2, 1, 1, 0, 0, 0, 0, true);
        let y: Vec<f64> = (0..200).map(|i| (i as f64 * 0.05).sin()).collect();
        let params = compute_start_params(&y, &config, None).unwrap();
        assert!(
            params.iter().all(|x| x.is_finite()),
            "Non-finite start params: {:?}",
            params
        );
    }
}
