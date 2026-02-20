use crate::error::{Result, SarimaxError};
use crate::types::SarimaxConfig;

/// Unpacked SARIMAX parameters.
///
/// Layout (flat vector order, matching statsmodels):
/// `[trend(k_trend) | exog(k_exog) | ar(p) | ma(q) | sar(P) | sma(Q) | sigma2?]`
///
/// When `concentrate_scale=true`, sigma2 is omitted from the optimization
/// vector but is still counted in AIC's k.
#[derive(Debug, Clone)]
pub struct SarimaxParams {
    pub trend_coeffs: Vec<f64>,
    pub exog_coeffs: Vec<f64>,
    pub ar_coeffs: Vec<f64>,
    pub ma_coeffs: Vec<f64>,
    pub sar_coeffs: Vec<f64>,
    pub sma_coeffs: Vec<f64>,
    pub sigma2: Option<f64>,
}

impl SarimaxParams {
    /// Unpack a flat parameter vector into structured fields.
    pub fn from_flat(flat: &[f64], config: &SarimaxConfig) -> Result<Self> {
        let kt = config.trend.k_trend();
        let n_exog = config.n_exog;
        let p = config.order.p;
        let q = config.order.q;
        let pp = config.order.pp;
        let qq = config.order.qq;

        let expected = kt + n_exog + p + q + pp + qq
            + if config.concentrate_scale { 0 } else { 1 };

        if flat.len() != expected {
            return Err(SarimaxError::ParamLengthMismatch {
                expected,
                got: flat.len(),
            });
        }

        let mut i = 0;
        let trend_coeffs = flat[i..i + kt].to_vec();
        i += kt;
        let exog_coeffs = flat[i..i + n_exog].to_vec();
        i += n_exog;
        let ar_coeffs = flat[i..i + p].to_vec();
        i += p;
        let ma_coeffs = flat[i..i + q].to_vec();
        i += q;
        let sar_coeffs = flat[i..i + pp].to_vec();
        i += pp;
        let sma_coeffs = flat[i..i + qq].to_vec();
        i += qq;
        let sigma2 = if !config.concentrate_scale {
            Some(flat[i])
        } else {
            None
        };

        Ok(Self {
            trend_coeffs,
            exog_coeffs,
            ar_coeffs,
            ma_coeffs,
            sar_coeffs,
            sma_coeffs,
            sigma2,
        })
    }

    /// Pack structured fields back into a flat vector.
    pub fn to_flat(&self) -> Vec<f64> {
        let mut v = Vec::new();
        v.extend(&self.trend_coeffs);
        v.extend(&self.exog_coeffs);
        v.extend(&self.ar_coeffs);
        v.extend(&self.ma_coeffs);
        v.extend(&self.sar_coeffs);
        v.extend(&self.sma_coeffs);
        if let Some(s) = self.sigma2 {
            v.push(s);
        }
        v
    }

    /// Number of estimated parameters for AIC/BIC.
    /// sigma2 is always counted even when concentrated.
    pub fn n_estimated_params(config: &SarimaxConfig) -> usize {
        config.trend.k_trend()
            + config.n_exog
            + config.order.p
            + config.order.q
            + config.order.pp
            + config.order.qq
            + 1 // sigma2 always counted
    }
}

// ---------------------------------------------------------------------------
// Monahan (1984) / Jones (1980) parameter transformations
// ---------------------------------------------------------------------------

/// Transform unconstrained parameters to stationary AR coefficients.
///
/// Algorithm:
/// 1. Map each x[k] to PACF via `r[k] = x[k] / sqrt(1 + x[k]^2)`
/// 2. Apply Levinson-Durbin recursion to get AR coefficients
/// 3. Negate final row: `constrained = -y[n-1][:]`
pub fn constrain_stationary(unconstrained: &[f64]) -> Vec<f64> {
    let n = unconstrained.len();
    if n == 0 {
        return vec![];
    }

    // Step 1: unconstrained → PACF
    let pacf: Vec<f64> = unconstrained
        .iter()
        .map(|&x| x / (1.0 + x * x).sqrt())
        .collect();

    // Step 2: Levinson-Durbin recursion
    let mut y = vec![vec![0.0; n]; n];
    for k in 0..n {
        for i in 0..k {
            y[k][i] = y[k - 1][i] + pacf[k] * y[k - 1][k - i - 1];
        }
        y[k][k] = pacf[k];
    }

    // Step 3: negate
    y[n - 1].iter().map(|&v| -v).collect()
}

/// Inverse transform: stationary AR coefficients → unconstrained parameters.
pub fn unconstrain_stationary(constrained: &[f64]) -> Vec<f64> {
    let n = constrained.len();
    if n == 0 {
        return vec![];
    }

    let mut y = vec![vec![0.0; n]; n];
    // Initialize last row from constrained (negate back)
    for i in 0..n {
        y[n - 1][i] = -constrained[i];
    }

    // Reverse Levinson-Durbin
    for k in (1..n).rev() {
        let rk = y[k][k];
        let denom = (1.0 - rk * rk).max(1e-15);
        for i in 0..k {
            y[k - 1][i] = (y[k][i] - rk * y[k][k - i - 1]) / denom;
        }
    }

    // PACF → unconstrained
    (0..n)
        .map(|k| {
            let r = y[k][k];
            r / (1.0 - r * r).max(1e-15).sqrt()
        })
        .collect()
}

/// Transform unconstrained parameters to invertible MA coefficients.
/// Same as stationary transform but with sign flip.
pub fn constrain_invertible(unconstrained: &[f64]) -> Vec<f64> {
    constrain_stationary(unconstrained)
        .iter()
        .map(|&x| -x)
        .collect()
}

/// Inverse: invertible MA coefficients → unconstrained parameters.
pub fn unconstrain_invertible(constrained: &[f64]) -> Vec<f64> {
    let negated: Vec<f64> = constrained.iter().map(|&x| -x).collect();
    unconstrain_stationary(&negated)
}

/// Constrain variance: unconstrained → positive (x^2).
pub fn constrain_variance(x: f64) -> f64 {
    x * x
}

/// Unconstrain variance: positive → unconstrained (sqrt).
/// Returns error if s <= 0.
pub fn unconstrain_variance(s: f64) -> Result<f64> {
    if s <= 0.0 {
        return Err(SarimaxError::DataError(format!(
            "variance sigma2 must be positive, got {}",
            s
        )));
    }
    Ok(s.sqrt())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SarimaxOrder, Trend};

    fn make_config(p: usize, q: usize, pp: usize, qq: usize, concentrate: bool) -> SarimaxConfig {
        SarimaxConfig {
            order: SarimaxOrder::new(p, 0, q, pp, 0, qq, 12),
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
    fn test_from_flat_to_flat_roundtrip() {
        let config = make_config(2, 1, 1, 1, true);
        // flat: ar(2) + ma(1) + sar(1) + sma(1) = 5 params
        let flat = vec![0.5, -0.3, 0.2, 0.4, -0.1];
        let params = SarimaxParams::from_flat(&flat, &config).unwrap();
        assert_eq!(params.ar_coeffs, vec![0.5, -0.3]);
        assert_eq!(params.ma_coeffs, vec![0.2]);
        assert_eq!(params.sar_coeffs, vec![0.4]);
        assert_eq!(params.sma_coeffs, vec![-0.1]);
        assert!(params.sigma2.is_none());
        assert_eq!(params.to_flat(), flat);
    }

    #[test]
    fn test_from_flat_with_sigma2() {
        let config = make_config(1, 0, 0, 0, false);
        // flat: ar(1) + sigma2 = 2 params
        let flat = vec![0.7, 1.5];
        let params = SarimaxParams::from_flat(&flat, &config).unwrap();
        assert_eq!(params.ar_coeffs, vec![0.7]);
        assert_eq!(params.sigma2, Some(1.5));
        assert_eq!(params.to_flat(), flat);
    }

    #[test]
    fn test_from_flat_with_trend_and_exog() {
        let config = SarimaxConfig {
            order: SarimaxOrder::new(1, 0, 1, 0, 0, 0, 0),
            n_exog: 2,
            trend: Trend::Constant,
            concentrate_scale: true,
            ..Default::default()
        };
        // flat: trend(1) + exog(2) + ar(1) + ma(1) = 5 params
        let flat = vec![0.1, 0.2, 0.3, 0.5, -0.3];
        let params = SarimaxParams::from_flat(&flat, &config).unwrap();
        assert_eq!(params.trend_coeffs, vec![0.1]);
        assert_eq!(params.exog_coeffs, vec![0.2, 0.3]);
        assert_eq!(params.ar_coeffs, vec![0.5]);
        assert_eq!(params.ma_coeffs, vec![-0.3]);
        assert_eq!(params.to_flat(), flat);
    }

    #[test]
    fn test_from_flat_length_mismatch() {
        let config = make_config(1, 0, 0, 0, true);
        let flat = vec![0.5, 0.3]; // too many
        assert!(SarimaxParams::from_flat(&flat, &config).is_err());
    }

    #[test]
    fn test_n_estimated_params() {
        // SARIMA(1,1,1)(1,1,1,12) with trend='c': 1+0+1+1+1+1+1 = 6
        let config = SarimaxConfig {
            order: SarimaxOrder::new(1, 1, 1, 1, 1, 1, 12),
            n_exog: 0,
            trend: Trend::Constant,
            concentrate_scale: true,
            ..Default::default()
        };
        assert_eq!(SarimaxParams::n_estimated_params(&config), 6);
    }

    #[test]
    fn test_monahan_roundtrip_ar1() {
        let original = vec![0.5];
        let constrained = constrain_stationary(&original);
        let unconstrained = unconstrain_stationary(&constrained);
        assert!((original[0] - unconstrained[0]).abs() < 1e-10);
    }

    #[test]
    fn test_monahan_roundtrip_ar2() {
        let original = vec![0.5, -0.3];
        let constrained = constrain_stationary(&original);
        let unconstrained = unconstrain_stationary(&constrained);
        for (a, b) in original.iter().zip(unconstrained.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "roundtrip failed: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_monahan_roundtrip_ar3() {
        let original = vec![1.0, -0.5, 0.2];
        let constrained = constrain_stationary(&original);
        let unconstrained = unconstrain_stationary(&constrained);
        for (a, b) in original.iter().zip(unconstrained.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "roundtrip failed: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_constrain_stationary_empty() {
        let empty: Vec<f64> = vec![];
        assert_eq!(constrain_stationary(&[]), empty);
        assert_eq!(unconstrain_stationary(&[]), empty);
    }

    #[test]
    fn test_invertible_roundtrip() {
        let original = vec![0.4, -0.2];
        let constrained = constrain_invertible(&original);
        let unconstrained = unconstrain_invertible(&constrained);
        for (a, b) in original.iter().zip(unconstrained.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_variance_roundtrip() {
        let x = 2.5;
        let s = constrain_variance(x);
        assert!((s - 6.25).abs() < 1e-10);
        let x2 = unconstrain_variance(s).unwrap();
        assert!((x2 - x.abs()).abs() < 1e-10);
    }
}
