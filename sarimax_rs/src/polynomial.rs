use crate::params::SarimaxParams;
use crate::types::SarimaxOrder;

/// Polynomial multiplication (convolution): c[k] = sum_i a[i]*b[k-i].
pub fn polymul(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let mut r = vec![0.0; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            r[i + j] += ai * bj;
        }
    }
    r
}

/// AR polynomial: 1 - phi_1*L - phi_2*L^2 - ...
/// `coeffs` = [phi_1, phi_2, ...], `max_lag` = p.
pub fn make_ar_poly(coeffs: &[f64], max_lag: usize) -> Vec<f64> {
    let mut p = vec![0.0; max_lag + 1];
    p[0] = 1.0;
    for (i, &c) in coeffs.iter().enumerate() {
        if i + 1 <= max_lag {
            p[i + 1] = -c;
        }
    }
    p
}

/// Seasonal AR polynomial: 1 - Phi_1*L^s - Phi_2*L^(2s) - ...
pub fn make_seasonal_ar_poly(coeffs: &[f64], s: usize) -> Vec<f64> {
    if coeffs.is_empty() {
        return vec![1.0];
    }
    let mut p = vec![0.0; coeffs.len() * s + 1];
    p[0] = 1.0;
    for (i, &c) in coeffs.iter().enumerate() {
        p[(i + 1) * s] = -c;
    }
    p
}

/// MA polynomial: 1 + theta_1*L + theta_2*L^2 + ...
/// `coeffs` = [theta_1, theta_2, ...], `max_lag` = q.
pub fn make_ma_poly(coeffs: &[f64], max_lag: usize) -> Vec<f64> {
    let mut p = vec![0.0; max_lag + 1];
    p[0] = 1.0;
    for (i, &c) in coeffs.iter().enumerate() {
        if i + 1 <= max_lag {
            p[i + 1] = c;
        }
    }
    p
}

/// Seasonal MA polynomial: 1 + Theta_1*L^s + Theta_2*L^(2s) + ...
pub fn make_seasonal_ma_poly(coeffs: &[f64], s: usize) -> Vec<f64> {
    if coeffs.is_empty() {
        return vec![1.0];
    }
    let mut p = vec![0.0; coeffs.len() * s + 1];
    p[0] = 1.0;
    for (i, &c) in coeffs.iter().enumerate() {
        p[(i + 1) * s] = c;
    }
    p
}

/// Reduced (expanded) AR polynomial = polymul(non-seasonal AR, seasonal AR).
pub fn reduced_ar(params: &SarimaxParams, order: &SarimaxOrder) -> Vec<f64> {
    polymul(
        &make_ar_poly(&params.ar_coeffs, order.p),
        &make_seasonal_ar_poly(&params.sar_coeffs, order.s),
    )
}

/// Reduced (expanded) MA polynomial = polymul(non-seasonal MA, seasonal MA).
pub fn reduced_ma(params: &SarimaxParams, order: &SarimaxOrder) -> Vec<f64> {
    polymul(
        &make_ma_poly(&params.ma_coeffs, order.q),
        &make_seasonal_ma_poly(&params.sma_coeffs, order.s),
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polymul_basic() {
        // (1 + 2x)(1 + 3x) = 1 + 5x + 6x^2
        let r = polymul(&[1.0, 2.0], &[1.0, 3.0]);
        assert_eq!(r.len(), 3);
        assert!((r[0] - 1.0).abs() < 1e-10);
        assert!((r[1] - 5.0).abs() < 1e-10);
        assert!((r[2] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_polymul_identity() {
        // a * [1] = a
        let a = vec![1.0, -0.5, 0.3];
        let r = polymul(&a, &[1.0]);
        assert_eq!(r.len(), a.len());
        for (x, y) in r.iter().zip(a.iter()) {
            assert!((x - y).abs() < 1e-10);
        }
    }

    #[test]
    fn test_polymul_empty() {
        let empty: Vec<f64> = vec![];
        assert_eq!(polymul(&[], &[1.0, 2.0]), empty);
        assert_eq!(polymul(&[1.0], &[]), empty);
    }

    #[test]
    fn test_make_ar_poly() {
        // AR(2): phi=[0.5, -0.3] → [1, -0.5, 0.3]
        let p = make_ar_poly(&[0.5, -0.3], 2);
        assert_eq!(p.len(), 3);
        assert!((p[0] - 1.0).abs() < 1e-10);
        assert!((p[1] - (-0.5)).abs() < 1e-10);
        assert!((p[2] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_make_ma_poly() {
        // MA(1): theta=[0.3] → [1, 0.3]
        let p = make_ma_poly(&[0.3], 1);
        assert_eq!(p.len(), 2);
        assert!((p[0] - 1.0).abs() < 1e-10);
        assert!((p[1] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_seasonal_ar_poly() {
        // SAR(1) with s=12, Phi=[0.3] → [1, 0, ..., -0.3] (length 13)
        let p = make_seasonal_ar_poly(&[0.3], 12);
        assert_eq!(p.len(), 13);
        assert!((p[0] - 1.0).abs() < 1e-10);
        for i in 1..12 {
            assert!((p[i]).abs() < 1e-10, "p[{}] should be 0", i);
        }
        assert!((p[12] - (-0.3)).abs() < 1e-10);
    }

    #[test]
    fn test_seasonal_ar_poly_empty() {
        let p = make_seasonal_ar_poly(&[], 12);
        assert_eq!(p, vec![1.0]);
    }

    #[test]
    fn test_seasonal_ma_poly() {
        // SMA(1) with s=12, Theta=[0.4] → [1, 0, ..., 0.4]
        let p = make_seasonal_ma_poly(&[0.4], 12);
        assert_eq!(p.len(), 13);
        assert!((p[0] - 1.0).abs() < 1e-10);
        assert!((p[12] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_reduced_ar_sarima() {
        // SARIMA(1,0,0)(1,0,0,12): AR=[0.5], SAR=[0.3]
        // (1 - 0.5L)(1 - 0.3L^12)
        // = 1 - 0.5L + 0*L^2 + ... - 0.3L^12 + 0.15L^13
        let params = SarimaxParams {
            trend_coeffs: vec![],
            exog_coeffs: vec![],
            ar_coeffs: vec![0.5],
            ma_coeffs: vec![],
            sar_coeffs: vec![0.3],
            sma_coeffs: vec![],
            sigma2: None,
        };
        let order = SarimaxOrder::new(1, 0, 0, 1, 0, 0, 12);
        let r = reduced_ar(&params, &order);

        assert_eq!(r.len(), 14); // degree 13 → 14 elements
        assert!((r[0] - 1.0).abs() < 1e-10);
        assert!((r[1] - (-0.5)).abs() < 1e-10);
        for i in 2..12 {
            assert!((r[i]).abs() < 1e-10, "r[{}] = {} should be 0", i, r[i]);
        }
        assert!((r[12] - (-0.3)).abs() < 1e-10);
        assert!((r[13] - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_reduced_ma_sarima() {
        // SARIMA(0,0,1)(0,0,1,12): MA=[0.2], SMA=[0.4]
        // (1 + 0.2L)(1 + 0.4L^12)
        // = 1 + 0.2L + 0*L^2 + ... + 0.4L^12 + 0.08L^13
        let params = SarimaxParams {
            trend_coeffs: vec![],
            exog_coeffs: vec![],
            ar_coeffs: vec![],
            ma_coeffs: vec![0.2],
            sar_coeffs: vec![],
            sma_coeffs: vec![0.4],
            sigma2: None,
        };
        let order = SarimaxOrder::new(0, 0, 1, 0, 0, 1, 12);
        let r = reduced_ma(&params, &order);

        assert_eq!(r.len(), 14);
        assert!((r[0] - 1.0).abs() < 1e-10);
        assert!((r[1] - 0.2).abs() < 1e-10);
        assert!((r[12] - 0.4).abs() < 1e-10);
        assert!((r[13] - 0.08).abs() < 1e-10);
    }
}
