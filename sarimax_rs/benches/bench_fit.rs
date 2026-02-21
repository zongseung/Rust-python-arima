use criterion::{criterion_group, criterion_main, Criterion};
use sarimax_rs::optimizer;
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

fn bench_fit_ar1(c: &mut Criterion) {
    let data = generate_ar1_data(500, 0.65, 42);
    let config = SarimaxConfig {
        order: SarimaxOrder::new(1, 0, 0, 0, 0, 0, 0),
        n_exog: 0,
        trend: Trend::None,
        enforce_stationarity: true,
        enforce_invertibility: true,
        concentrate_scale: true,
        simple_differencing: false,
        measurement_error: false,
    };

    c.bench_function("fit_ar1_n500", |b| {
        b.iter(|| optimizer::fit(&data, &config, None, Some("lbfgs"), None, None).unwrap())
    });
}

fn bench_fit_arima111(c: &mut Criterion) {
    let data = generate_ar1_data(500, 0.3, 42);
    let config = SarimaxConfig {
        order: SarimaxOrder::new(1, 1, 1, 0, 0, 0, 0),
        n_exog: 0,
        trend: Trend::None,
        enforce_stationarity: true,
        enforce_invertibility: true,
        concentrate_scale: true,
        simple_differencing: false,
        measurement_error: false,
    };

    c.bench_function("fit_arima111_n500", |b| {
        b.iter(|| optimizer::fit(&data, &config, None, Some("lbfgs"), None, None).unwrap())
    });
}

fn bench_fit_sarima_111_111_12(c: &mut Criterion) {
    // Cumulative sum of LCG noise to simulate integrated series
    let mut data = generate_ar1_data(500, 0.0, 42);
    for t in 1..data.len() {
        data[t] += data[t - 1];
    }
    let config = SarimaxConfig {
        order: SarimaxOrder::new(1, 1, 1, 1, 1, 1, 12),
        n_exog: 0,
        trend: Trend::None,
        enforce_stationarity: false,
        enforce_invertibility: false,
        concentrate_scale: true,
        simple_differencing: false,
        measurement_error: false,
    };

    c.bench_function("fit_sarima111_111_12_n500", |b| {
        b.iter(|| optimizer::fit(&data, &config, None, Some("lbfgsb"), None, None).unwrap())
    });
}

criterion_group!(benches, bench_fit_ar1, bench_fit_arima111, bench_fit_sarima_111_111_12);
criterion_main!(benches);
