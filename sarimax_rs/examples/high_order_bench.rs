use sarimax_rs::optimizer;
use sarimax_rs::types::{SarimaxConfig, SarimaxOrder, Trend};
use std::time::Instant;

fn generate_data(n: usize, phi: f64, seed: u64) -> Vec<f64> {
    let mut y = vec![0.0; n];
    let mut state: u64 = seed;
    for t in 1..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((state >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0;
        y[t] = phi * y[t - 1] + u;
    }
    for t in 1..n {
        y[t] += y[t - 1];
    }
    y
}

fn run_test(label: &str, order: SarimaxOrder, n: usize) {
    let data = generate_data(n, 0.3, 42);
    let config = SarimaxConfig {
        order,
        n_exog: 0,
        trend: Trend::None,
        enforce_stationarity: true,
        enforce_invertibility: true,
        concentrate_scale: true,
        simple_differencing: false,
        measurement_error: false,
    };

    let start = Instant::now();
    match optimizer::fit(&data, &config, None, Some("lbfgsb"), Some(500), None) {
        Ok(r) => {
            let elapsed = start.elapsed();
            eprintln!(
                "{:<45} | {:>9.1}ms | conv={:<5} | ll={:>12.3} | iter={:>4} | k={:>3} | {}",
                label,
                elapsed.as_secs_f64() * 1000.0,
                r.converged,
                r.loglike,
                r.n_iter,
                config.order.k_states(),
                r.method
            );
        }
        Err(e) => {
            let elapsed = start.elapsed();
            eprintln!(
                "{:<45} | {:>9.1}ms | ERROR: {}",
                label,
                elapsed.as_secs_f64() * 1000.0,
                e
            );
        }
    }
}

fn main() {
    eprintln!("\n{:=<110}", "= HIGH-ORDER SARIMA BENCHMARK ");
    eprintln!(
        "{:<45} | {:>9} | {:>8} | {:>12} | {:>6} | {:>3} | {}",
        "Model", "Time", "Conv", "LogLike", "Iter", "k", "Method"
    );
    eprintln!("{:-<110}", "");

    // Non-seasonal ARMA/ARIMA
    run_test("ARMA(2,2) n=500",          SarimaxOrder::new(2,0,2,0,0,0,0), 500);
    run_test("ARMA(4,2) n=500",          SarimaxOrder::new(4,0,2,0,0,0,0), 500);
    run_test("ARIMA(2,1,2) n=500",       SarimaxOrder::new(2,1,2,0,0,0,0), 500);
    run_test("ARIMA(4,1,4) n=500",       SarimaxOrder::new(4,1,4,0,0,0,0), 500);
    run_test("ARIMA(4,2,3) n=500",       SarimaxOrder::new(4,2,3,0,0,0,0), 500);

    eprintln!("{:-<110}", "");

    // Seasonal SARIMA (s=12)
    run_test("SARIMA(1,1,1)(1,1,1,12) n=500",   SarimaxOrder::new(1,1,1,1,1,1,12), 500);
    run_test("SARIMA(2,1,1)(1,1,1,12) n=500",   SarimaxOrder::new(2,1,1,1,1,1,12), 500);
    run_test("SARIMA(2,1,2)(1,1,1,12) n=500",   SarimaxOrder::new(2,1,2,1,1,1,12), 500);
    run_test("SARIMA(3,1,2)(1,1,1,12) n=500",   SarimaxOrder::new(3,1,2,1,1,1,12), 500);
    run_test("SARIMA(2,1,2)(2,1,1,12) n=500",   SarimaxOrder::new(2,1,2,2,1,1,12), 500);
    run_test("SARIMA(3,1,3)(2,1,2,12) n=500",   SarimaxOrder::new(3,1,3,2,1,2,12), 500);

    eprintln!("{:-<110}", "");

    // Large n
    run_test("SARIMA(1,1,1)(1,1,1,12) n=2000",  SarimaxOrder::new(1,1,1,1,1,1,12), 2000);
    run_test("SARIMA(2,1,2)(1,1,1,12) n=2000",  SarimaxOrder::new(2,1,2,1,1,1,12), 2000);
    run_test("SARIMA(3,1,3)(2,1,2,12) n=2000",  SarimaxOrder::new(3,1,3,2,1,2,12), 2000);

    eprintln!("{:-<110}", "");

    // Very large n
    run_test("SARIMA(1,1,1)(1,1,1,12) n=10000", SarimaxOrder::new(1,1,1,1,1,1,12), 10000);
    run_test("ARIMA(4,1,4) n=10000",            SarimaxOrder::new(4,1,4,0,0,0,0), 10000);

    eprintln!("{:=<110}\n", "");
}
