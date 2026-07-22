use super::*;
use super::objective::SarimaxObjective;
use super::trust_region::refine_exog_brent;
use argmin::core::{CostFunction, Gradient};
use crate::kalman::kalman_filter;
use crate::params::SarimaxParams;
use crate::test_helpers::{load_fixtures, make_config_with_enforcement};
use crate::types::SarimaxConfig;
use std::cell::RefCell;

#[test]
fn test_transform_untransform_roundtrip() {
    let config = make_config_with_enforcement(2, 0, 1, true, true);
    let original = vec![0.5, -0.3, 0.2, 1.5]; // ar(2), ma(1), sigma2
    let unconstrained = untransform_params(&original, &config).unwrap();
    let recovered = transform_params(&unconstrained, &config).unwrap();
    for (a, b) in original.iter().zip(recovered.iter()) {
        assert!((a - b).abs() < 1e-10, "roundtrip failed: {} vs {}", a, b);
    }
}

#[test]
fn test_transform_passthrough_no_enforce() {
    let config = make_config_with_enforcement(1, 0, 1, false, false);
    // ar(1), ma(1), sigma2; only sigma2 is transformed (ln) without enforcement
    let original = vec![0.7, -0.3, 1.0];
    let unconstrained = untransform_params(&original, &config).unwrap();
    assert_eq!(original[..2], unconstrained[..2]);
    assert!((unconstrained[2] - 1.0f64.ln()).abs() < 1e-15);
}

#[test]
fn test_objective_finite() {
    let fixtures = load_fixtures();
    let case = &fixtures["ar1"];
    let data: Vec<f64> = case["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let config = make_config_with_enforcement(1, 0, 0, false, false);
    let obj = SarimaxObjective {
        endog: data,
        config,
        exog: None,
        cache: RefCell::new(None),
        ss_cache: RefCell::new(None),
    };

    let cost = obj.cost(&vec![0.5]).unwrap();
    assert!(cost.is_finite(), "cost should be finite: {}", cost);
}

#[test]
fn test_exog_coordinate_polish_improves_bad_beta_start() {
    use crate::types::{SarimaxOrder, Trend};

    let n = 180;
    let x: Vec<f64> = (0..n)
        .map(|t| ((t as f64) * 0.17).sin() + 0.5 * ((t as f64) * 0.07).cos())
        .collect();
    let mut y = vec![0.0; n];
    for t in 1..n {
        let noise = 0.05 * ((t as f64) * 0.31).sin();
        y[t] = 2.0 * x[t] + 0.4 * y[t - 1] + noise;
    }
    let exog = vec![x];
    let config = SarimaxConfig {
        order: SarimaxOrder::new(1, 0, 0, 0, 0, 0, 0),
        n_exog: 1,
        trend: Trend::None,
        enforce_stationarity: false,
        enforce_invertibility: false,
        concentrate_scale: true,
        simple_differencing: false,
        measurement_error: false,
    };
    let obj = SarimaxObjective {
        endog: y,
        config: config.clone(),
        exog: Some(exog),
        cache: RefCell::new(None),
        ss_cache: RefCell::new(None),
    };
    let start = vec![0.0, 0.4];
    let start_cost = obj.eval_negloglike(&start).unwrap();

    let (polished, polished_cost) = refine_exog_brent(&obj, &config, start, start_cost);

    assert!(
        polished_cost < start_cost - 1.0,
        "polish should improve cost: start={} polished={}",
        start_cost,
        polished_cost
    );
    assert!(
        polished[0] > 0.5,
        "exog beta should move toward the positive signal: {}",
        polished[0]
    );
}

#[test]
fn test_profile_trust_region_with_exog_returns_profiled_beta() {
    use crate::types::{SarimaxOrder, Trend};

    let n = 220;
    let x: Vec<f64> = (0..n)
        .map(|t| ((t as f64) * 0.13).sin() + 0.25 * ((t as f64) * 0.03).cos())
        .collect();
    let mut y = vec![0.0; n];
    for t in 1..n {
        let noise = 0.03 * ((t as f64) * 0.37).sin();
        y[t] = 1.7 * x[t] + 0.35 * y[t - 1] + noise;
    }
    let exog = vec![x];
    let config = SarimaxConfig {
        order: SarimaxOrder::new(1, 0, 0, 0, 0, 0, 0),
        n_exog: 1,
        trend: Trend::None,
        enforce_stationarity: false,
        enforce_invertibility: false,
        concentrate_scale: true,
        simple_differencing: false,
        measurement_error: false,
    };

    let result = fit(
        &y,
        &config,
        None,
        Some("profile-trust-region"),
        Some(80),
        Some(&exog),
    )
    .unwrap();

    assert!(result.loglike.is_finite());
    assert_eq!(result.params.len(), 2);
    assert!(
        result.params[0] > 0.5,
        "profiled beta should move toward the positive signal: {}",
        result.params[0]
    );
    assert_eq!(result.method, "profile-trust-region");
}

#[test]
fn test_kalman_innovation_linearity_for_exog_intercept() {
    use crate::types::{SarimaxOrder, Trend};

    let config = SarimaxConfig {
        order: SarimaxOrder::new(0, 0, 0, 0, 1, 1, 4),
        n_exog: 1,
        trend: Trend::None,
        enforce_stationarity: false,
        enforce_invertibility: false,
        concentrate_scale: false,
        simple_differencing: false,
        measurement_error: false,
    };

    let y: Vec<f64> = (0..40)
        .map(|i| 10.0 + (i as f64 * 0.7).sin() + i as f64 * 0.2)
        .collect();
    let x = vec![(0..40)
        .map(|i| 3.0 + (i as f64 * 0.3).cos() + i as f64 * 0.05)
        .collect::<Vec<f64>>()];

    let zero = SarimaxParams::from_flat(&[0.0, 0.25, 2.0], &config).unwrap();
    let full = SarimaxParams::from_flat(&[1.7, 0.25, 2.0], &config).unwrap();

    let ss_y = StateSpace::new(&config, &zero, &y, None).unwrap();
    let init_y = KalmanInit::from_config_default(&ss_y, &config);
    let y_filter = kalman_filter(&y, &ss_y, &init_y, false).unwrap();

    let ss_x = StateSpace::new(&config, &zero, &x[0], None).unwrap();
    let init_x = KalmanInit::from_config_default(&ss_x, &config);
    let x_filter = kalman_filter(&x[0], &ss_x, &init_x, false).unwrap();

    let ss_full = StateSpace::new(&config, &full, &y, Some(&x)).unwrap();
    let init_full = KalmanInit::from_config_default(&ss_full, &config);
    let full_filter = kalman_filter(&y, &ss_full, &init_full, false).unwrap();

    for t in 0..y.len() {
        let expected = y_filter.innovations[t] - 1.7 * x_filter.innovations[t];
        assert!(
            (full_filter.innovations[t] - expected).abs() < 1e-8,
            "innovation mismatch at t={}: got {}, expected {}",
            t,
            full_filter.innovations[t],
            expected
        );
    }
}

#[test]
fn test_gradient_finite() {
    let fixtures = load_fixtures();
    let case = &fixtures["ar1"];
    let data: Vec<f64> = case["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let config = make_config_with_enforcement(1, 0, 0, false, false);
    let obj = SarimaxObjective {
        endog: data,
        config,
        exog: None,
        cache: RefCell::new(None),
        ss_cache: RefCell::new(None),
    };

    let grad = obj.gradient(&vec![0.5]).unwrap();
    assert_eq!(grad.len(), 1);
    assert!(
        grad[0].is_finite(),
        "gradient should be finite: {}",
        grad[0]
    );
}

#[test]
fn test_fit_ar1_lbfgsb_convergence() {
    let fixtures = load_fixtures();
    let case = &fixtures["ar1"];
    let data: Vec<f64> = case["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    // Test with enforce=false
    let config_noforce = make_config_with_enforcement(1, 0, 0, false, false);
    let r1 = fit(&data, &config_noforce, None, Some("lbfgsb"), Some(500), None).unwrap();
    assert!(r1.converged, "AR(1) lbfgsb enforce=false should converge");

    // Test with enforce=true (Python default)
    let config_force = make_config_with_enforcement(1, 0, 0, true, true);
    let r2 = fit(&data, &config_force, None, Some("lbfgsb"), Some(500), None).unwrap();
    assert!(r2.converged, "AR(1) lbfgsb enforce=true should converge");
}


#[test]
fn test_fit_arma11() {
    let fixtures = load_fixtures();
    let case = &fixtures["arma11"];
    let data: Vec<f64> = case["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let expected_params: Vec<f64> = case["params"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    // Fixture was generated with approximate_diffuse init
    let config = make_config_with_enforcement(1, 0, 1, false, false);
    let result = fit(&data, &config, None, Some("lbfgs"), Some(500), None).unwrap();

    for (i, (got, exp)) in result.params.iter().zip(expected_params.iter()).enumerate() {
        let err = (got - exp).abs();
        assert!(
            err < 1e-3,
            "ARMA(1,1) param[{}] error: {} (got {}, expected {})",
            i,
            err,
            got,
            exp
        );
    }
}

#[test]
fn test_fit_arima111() {
    let fixtures = load_fixtures();
    let case = &fixtures["arima111"];
    let data: Vec<f64> = case["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let expected_loglike = case["loglike"].as_f64().unwrap();

    // Fixture was generated with approximate_diffuse init
    let config = make_config_with_enforcement(1, 1, 1, false, false);
    let result = fit(&data, &config, None, Some("lbfgs"), Some(500), None).unwrap();

    let ll_err = (result.loglike - expected_loglike).abs();
    assert!(
        ll_err < 1.0,
        "ARIMA(1,1,1) loglike error: {} (got {}, expected {})",
        ll_err,
        result.loglike,
        expected_loglike
    );
}

#[test]
fn test_fit_nelder_mead() {
    let fixtures = load_fixtures();
    let case = &fixtures["ar1"];
    let data: Vec<f64> = case["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let expected_params: Vec<f64> = case["params"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let config = make_config_with_enforcement(1, 0, 0, false, false);
    let result = fit(&data, &config, None, Some("nelder-mead"), Some(1000), None).unwrap();

    let param_err = (result.params[0] - expected_params[0]).abs();
    assert!(
        param_err < 1e-3,
        "NM AR(1) param error: {} (got {}, expected {})",
        param_err,
        result.params[0],
        expected_params[0]
    );
}

#[test]
fn test_aic_bic() {
    let fixtures = load_fixtures();
    let case = &fixtures["ar1"];
    let data: Vec<f64> = case["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let config = make_config_with_enforcement(1, 0, 0, true, true);
    let result = fit(&data, &config, None, Some("lbfgs"), Some(500), None).unwrap();

    // AIC = -2*loglike + 2*k, BIC = -2*loglike + k*ln(n)
    let k = result.n_params as f64;
    let n = result.n_obs as f64;
    let expected_aic = -2.0 * result.loglike + 2.0 * k;
    let expected_bic = -2.0 * result.loglike + k * n.ln();

    assert!(
        (result.aic - expected_aic).abs() < 1e-10,
        "AIC mismatch: got {}, expected {}",
        result.aic,
        expected_aic
    );
    assert!(
        (result.bic - expected_bic).abs() < 1e-10,
        "BIC mismatch: got {}, expected {}",
        result.bic,
        expected_bic
    );
}

#[test]
fn test_fit_with_custom_start_params() {
    let fixtures = load_fixtures();
    let case = &fixtures["ar1"];
    let data: Vec<f64> = case["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let config = make_config_with_enforcement(1, 0, 0, false, false);
    let start = vec![0.5, 1.0]; // ar(1), sigma2
    let result = fit(&data, &config, Some(&start), Some("lbfgs"), Some(500), None).unwrap();

    assert!(result.loglike.is_finite());
    assert!(result.params[0].is_finite());
}

#[test]
fn test_zero_maxiter_not_converged_for_lbfgs_and_nm() {
    let fixtures = load_fixtures();
    let case = &fixtures["ar1"];
    let data: Vec<f64> = case["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let config = make_config_with_enforcement(1, 0, 0, false, false);

    let lbfgs = fit(&data, &config, None, Some("lbfgs"), Some(0), None).unwrap();
    assert_eq!(lbfgs.n_iter, 0, "lbfgs with maxiter=0 should not run");
    assert!(
        !lbfgs.converged,
        "lbfgs with maxiter=0 must report not converged"
    );

    let nm = fit(&data, &config, None, Some("nelder-mead"), Some(0), None).unwrap();
    assert_eq!(nm.n_iter, 0, "nelder-mead with maxiter=0 should not run");
    assert!(
        !nm.converged,
        "nelder-mead with maxiter=0 must report not converged"
    );
}

#[test]
fn test_zero_maxiter_not_converged_for_lbfgsb_multi() {
    let fixtures = load_fixtures();
    let case = &fixtures["arma11"];
    let data: Vec<f64> = case["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let config = make_config_with_enforcement(1, 0, 1, false, false);
    let result = fit(&data, &config, None, Some("lbfgsb-multi"), Some(0), None).unwrap();

    assert_eq!(
        result.n_iter, 0,
        "lbfgsb-multi with maxiter=0 should not consume budget"
    );
    assert!(
        !result.converged,
        "lbfgsb-multi with maxiter=0 must report not converged"
    );
}

#[test]
fn test_zero_maxiter_not_converged_for_lbfgsb_single() {
    let fixtures = load_fixtures();
    let case = &fixtures["ar1"];
    let data: Vec<f64> = case["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let config = make_config_with_enforcement(1, 0, 0, false, false);
    let result = fit(&data, &config, None, Some("lbfgsb"), Some(0), None).unwrap();

    eprintln!(
        "lbfgsb single maxiter=0: n_iter={}, converged={}, method={}",
        result.n_iter, result.converged, result.method
    );
    assert_eq!(
        result.n_iter, 0,
        "lbfgsb with maxiter=0 should report n_iter=0, got {}",
        result.n_iter
    );
    assert!(
        !result.converged,
        "lbfgsb with maxiter=0 must report not converged"
    );
}

#[test]
fn test_small_maxiter_lbfgs_not_converged() {
    // With maxiter=1, L-BFGS should NOT converge (not enough iterations)
    let fixtures = load_fixtures();
    let case = &fixtures["arma11"];
    let data: Vec<f64> = case["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let config = make_config_with_enforcement(1, 0, 1, false, false);
    let result = fit(&data, &config, None, Some("lbfgs"), Some(1), None).unwrap();
    eprintln!(
        "lbfgs maxiter=1: n_iter={}, converged={}, method={}",
        result.n_iter, result.converged, result.method
    );
    // With only 1 iteration, ARMA(1,1) cannot converge
    assert!(
        !result.converged,
        "lbfgs with maxiter=1 on ARMA(1,1) should not converge"
    );
}

#[test]
fn test_small_maxiter_nm_not_converged() {
    // With maxiter=1, NM should NOT converge
    let fixtures = load_fixtures();
    let case = &fixtures["arma11"];
    let data: Vec<f64> = case["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let config = make_config_with_enforcement(1, 0, 1, false, false);
    let result = fit(&data, &config, None, Some("nelder-mead"), Some(1), None).unwrap();
    eprintln!(
        "nm maxiter=1: n_iter={}, converged={}, method={}",
        result.n_iter, result.converged, result.method
    );
    assert!(
        !result.converged,
        "nelder-mead with maxiter=1 on ARMA(1,1) should not converge"
    );
}

#[test]
fn test_multistart_respects_global_maxiter_budget() {
    let fixtures = load_fixtures();
    let case = &fixtures["arma11"];
    let data: Vec<f64> = case["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let config = make_config_with_enforcement(1, 0, 1, false, false);
    let maxiter = 5_u64;

    let lbfgs = fit(&data, &config, None, Some("lbfgs"), Some(maxiter), None).unwrap();
    assert!(
        lbfgs.n_iter <= maxiter,
        "lbfgs n_iter={} exceeds maxiter={}",
        lbfgs.n_iter,
        maxiter
    );

    let lbfgsb_multi = fit(
        &data,
        &config,
        None,
        Some("lbfgsb-multi"),
        Some(maxiter),
        None,
    )
    .unwrap();
    assert!(
        lbfgsb_multi.n_iter <= maxiter,
        "lbfgsb-multi n_iter={} exceeds maxiter={}",
        lbfgsb_multi.n_iter,
        maxiter
    );
}

// -------------------------------------------------------------------------
// A-2: Near-cancellation detection tests (VER5.2 P6)
// -------------------------------------------------------------------------

#[test]
fn test_polynomial_roots_ar1() {
    // AR(1) with φ=0.5: companion eigenvalue = 0.5 (inverted root)
    let roots = polynomial_roots(&[0.5]);
    assert_eq!(roots.len(), 1);
    assert!((roots[0].0 - 0.5).abs() < 1e-10, "real part: {}", roots[0].0);
    assert!(roots[0].1.abs() < 1e-10, "imag part: {}", roots[0].1);
}

#[test]
fn test_polynomial_roots_ar2_real() {
    // AR(2) with real roots: φ₁=0.8, φ₂=-0.15 → roots near 0.3 and 0.5
    let roots = polynomial_roots(&[0.8, -0.15]);
    assert_eq!(roots.len(), 2);
    // Both roots should be real (imag ≈ 0)
    for (re, im) in &roots {
        assert!(im.abs() < 1e-8, "expected real root, got imag={}", im);
        assert!(re.abs() < 1.0 + 1e-6, "inverted root should be inside unit circle: {}", re);
    }
}

#[test]
fn test_polynomial_roots_ar2_complex() {
    // AR(2) with complex roots: φ₁=0.0, φ₂=-0.5
    // Companion eigenvalues: λ² - 0·λ - (-0.5) = λ²+0.5 = 0 → λ=±i·√0.5
    // discriminant = φ₁² + 4φ₂ = 0 + 4·(-0.5) = -2 < 0 → complex
    let roots = polynomial_roots(&[0.0, -0.5]);
    assert_eq!(roots.len(), 2);
    // Should have a complex conjugate pair
    let has_complex = roots.iter().any(|(_, im)| im.abs() > 1e-8);
    assert!(has_complex, "expected complex roots for AR(2) with phi2=-0.5");
    // Conjugate: real parts equal, imag parts opposite
    assert!((roots[0].0 - roots[1].0).abs() < 1e-10);
    assert!((roots[0].1 + roots[1].1).abs() < 1e-10);
}

#[test]
fn test_min_root_distance_far() {
    // AR root at 0.9 (real), MA root at 0.1 (real) → distance = 0.8
    let dist = min_root_distance(&[0.9], &[0.1]);
    assert!((dist - 0.8).abs() < 1e-10, "expected 0.8, got {}", dist);
}

#[test]
fn test_min_root_distance_near_cancellation() {
    // AR root ≈ MA root: φ=0.9, θ=0.89 → inverted roots 0.9 vs 0.89 → dist=0.01
    let dist = min_root_distance(&[0.9], &[0.89]);
    assert!(dist < 0.02, "expected near-cancellation (dist < 0.02), got {}", dist);
    assert!(dist > 0.005, "dist should be ~0.01, got {}", dist);
}

#[test]
fn test_validate_no_near_cancellation_ar_only() {
    // Pure AR: no MA → always valid
    let config = make_config_with_enforcement(2, 0, 0, false, false);
    let sp = SarimaxParams {
        ar_coeffs: vec![0.5, -0.2],
        ma_coeffs: vec![],
        sar_coeffs: vec![],
        sma_coeffs: vec![],
        sigma2: None,
        exog_coeffs: vec![],
        trend_coeffs: vec![],
    };
    assert!(validate_no_near_cancellation(&sp, &config, 0.05));
}

#[test]
fn test_validate_no_near_cancellation_arma_far() {
    // ARMA(1,1) with distant roots: valid
    let config = make_config_with_enforcement(1, 0, 1, false, false);
    let sp = SarimaxParams {
        ar_coeffs: vec![0.8],
        ma_coeffs: vec![0.2],
        sar_coeffs: vec![],
        sma_coeffs: vec![],
        sigma2: None,
        exog_coeffs: vec![],
        trend_coeffs: vec![],
    };
    assert!(validate_no_near_cancellation(&sp, &config, 0.05));
}

#[test]
fn test_validate_no_near_cancellation_arma_near() {
    // ARMA(1,1) with near-cancellation: φ≈θ → should fail α=0.05 check
    let config = make_config_with_enforcement(1, 0, 1, false, false);
    let sp = SarimaxParams {
        ar_coeffs: vec![0.9],
        ma_coeffs: vec![0.89],
        sar_coeffs: vec![],
        sma_coeffs: vec![],
        sigma2: None,
        exog_coeffs: vec![],
        trend_coeffs: vec![],
    };
    assert!(!validate_no_near_cancellation(&sp, &config, 0.05));
}

#[test]
fn test_passes_cancellation_filter_ar_only() {
    // Pure AR: always passes (no MA to cancel with)
    let config = make_config_with_enforcement(2, 0, 0, false, false);
    let params = vec![0.5, -0.2]; // unconstrained = constrained when !enforce
    assert!(passes_cancellation_filter(&params, &config));
}

#[test]
fn test_passes_cancellation_filter_arma_far() {
    // ARMA(1,1) with distant roots: should pass.
    // Non-concentrated layout: [ar, ma, sigma2].
    let config = make_config_with_enforcement(1, 0, 1, false, false);
    let params = vec![0.8, 0.2, 1.0]; // ar=0.8, ma=0.2 → roots far apart
    assert!(passes_cancellation_filter(&params, &config));
}

#[test]
fn test_passes_cancellation_filter_arma_near() {
    // ARMA(1,1) with near-cancellation: should fail α=0.01 check.
    // Non-concentrated layout: [ar, ma, sigma2] — previously this vector
    // omitted sigma2 and "failed" only because transform_params errored.
    let config = make_config_with_enforcement(1, 0, 1, false, false);
    let params = vec![0.9, 0.895, 1.0]; // ar=0.9, ma=0.895 → dist=0.005 < 0.01
    assert!(!passes_cancellation_filter(&params, &config));
}
