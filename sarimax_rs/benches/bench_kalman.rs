use criterion::{criterion_group, criterion_main, Criterion};
use sarimax_rs::initialization::KalmanInit;
use sarimax_rs::kalman::kalman_loglike;
use sarimax_rs::params::SarimaxParams;
use sarimax_rs::state_space::StateSpace;
use sarimax_rs::types::{SarimaxConfig, SarimaxOrder, Trend};

/// Deterministic LCG data generator (fixed seed → reproducible data).
fn generate_ar1_data(n: usize, phi: f64, seed: u64) -> Vec<f64> {
    let mut y = vec![0.0; n];
    let mut state: u64 = seed;
    for t in 1..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((state >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0;
        y[t] = phi * y[t - 1] + u;
    }
    y
}

fn bench_kalman_ar1(c: &mut Criterion) {
    let data = generate_ar1_data(500, 0.65, 42);
    let order = SarimaxOrder::new(1, 0, 0, 0, 0, 0, 0);
    let config = SarimaxConfig {
        order,
        n_exog: 0,
        trend: Trend::None,
        enforce_stationarity: false,
        enforce_invertibility: false,
        concentrate_scale: true,
        simple_differencing: false,
        measurement_error: false,
    };
    let params = SarimaxParams {
        trend_coeffs: vec![],
        exog_coeffs: vec![],
        ar_coeffs: vec![0.65],
        ma_coeffs: vec![],
        sar_coeffs: vec![],
        sma_coeffs: vec![],
        sigma2: None,
    };
    let ss = StateSpace::new(&config, &params, &data, None).unwrap();
    let init = KalmanInit::approximate_diffuse(ss.k_states, KalmanInit::default_kappa());

    c.bench_function("kalman_ar1_n500", |b| {
        b.iter(|| kalman_loglike(&data, &ss, &init, true).unwrap())
    });
}

fn bench_kalman_arima111(c: &mut Criterion) {
    let data = generate_ar1_data(500, 0.3, 42);
    let order = SarimaxOrder::new(1, 1, 1, 0, 0, 0, 0);
    let config = SarimaxConfig {
        order,
        n_exog: 0,
        trend: Trend::None,
        enforce_stationarity: false,
        enforce_invertibility: false,
        concentrate_scale: true,
        simple_differencing: false,
        measurement_error: false,
    };
    let params = SarimaxParams {
        trend_coeffs: vec![],
        exog_coeffs: vec![],
        ar_coeffs: vec![0.3],
        ma_coeffs: vec![0.4],
        sar_coeffs: vec![],
        sma_coeffs: vec![],
        sigma2: None,
    };
    let ss = StateSpace::new(&config, &params, &data, None).unwrap();
    let init = KalmanInit::approximate_diffuse(ss.k_states, KalmanInit::default_kappa());

    c.bench_function("kalman_arima111_n500", |b| {
        b.iter(|| kalman_loglike(&data, &ss, &init, true).unwrap())
    });
}

criterion_group!(benches, bench_kalman_ar1, bench_kalman_arima111);
criterion_main!(benches);
