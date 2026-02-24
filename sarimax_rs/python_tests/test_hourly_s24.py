"""Test: hourly data with s=24 SARIMA fit.

Generates synthetic hourly data with daily seasonality (s=24),
fits SARIMA models, and compares with statsmodels where possible.
"""

import time
import numpy as np

import sarimax_rs

np.random.seed(2024)


def generate_hourly_data(n_days=30):
    """Generate synthetic hourly data with daily seasonality.

    y_t = trend + seasonal_24h + AR(1) noise
    """
    n = n_days * 24
    t = np.arange(n)

    # slow linear trend
    trend = 50.0 + 0.005 * t

    # daily seasonal pattern (24h cycle)
    seasonal = 10.0 * np.sin(2 * np.pi * t / 24) + 3.0 * np.cos(2 * np.pi * t / 12)

    # AR(1) noise
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.5 * noise[i - 1] + np.random.normal(0, 1.5)

    return trend + seasonal + noise


def test_sarima_110_100_24():
    """SARIMA(1,1,0)(1,0,0,24) — simplest seasonal AR model."""
    y = generate_hourly_data(14)  # 2 weeks = 336 points

    t0 = time.perf_counter()
    result = sarimax_rs.sarimax_fit(
        y,
        (1, 1, 0),          # (p, d, q)
        (1, 0, 0, 24),      # (P, D, Q, s)
    )
    elapsed = time.perf_counter() - t0

    print(f"\n--- SARIMA(1,1,0)(1,0,0,24) | n={len(y)} ---")
    print(f"  converged : {result['converged']}")
    print(f"  method    : {result['method']}")
    print(f"  params    : {[f'{p:.4f}' for p in result['params']]}")
    print(f"  loglike   : {result['loglike']:.4f}")
    print(f"  AIC       : {result['aic']:.4f}")
    print(f"  BIC       : {result['bic']:.4f}")
    print(f"  n_iter    : {result['n_iter']}")
    print(f"  time      : {elapsed:.3f}s")

    if not result["converged"]:
        print(f"  WARNING: did not converge (method={result['method']})")
    assert np.isfinite(result["loglike"]), "loglike should be finite"


def test_sarima_111_111_24():
    """SARIMA(1,1,1)(1,1,1,24) — full seasonal model, k_states=51."""
    y = generate_hourly_data(14)  # 2 weeks = 336 points

    t0 = time.perf_counter()
    result = sarimax_rs.sarimax_fit(
        y,
        (1, 1, 1),          # (p, d, q)
        (1, 1, 1, 24),      # (P, D, Q, s)
    )
    elapsed = time.perf_counter() - t0

    print(f"\n--- SARIMA(1,1,1)(1,1,1,24) | n={len(y)}, k_states=51 ---")
    print(f"  converged : {result['converged']}")
    print(f"  method    : {result['method']}")
    print(f"  params    : {[f'{p:.4f}' for p in result['params']]}")
    print(f"  loglike   : {result['loglike']:.4f}")
    print(f"  AIC       : {result['aic']:.4f}")
    print(f"  BIC       : {result['bic']:.4f}")
    print(f"  n_iter    : {result['n_iter']}")
    print(f"  time      : {elapsed:.3f}s")

    if not result["converged"]:
        print(f"  WARNING: did not converge")
    assert np.isfinite(result["loglike"]), "loglike should be finite"


def test_sarima_111_111_24_longer():
    """SARIMA(1,1,1)(1,1,1,24) with 30 days (720 points)."""
    y = generate_hourly_data(30)

    t0 = time.perf_counter()
    result = sarimax_rs.sarimax_fit(
        y,
        (1, 1, 1),
        (1, 1, 1, 24),
    )
    elapsed = time.perf_counter() - t0

    print(f"\n--- SARIMA(1,1,1)(1,1,1,24) | n={len(y)}, 30days ---")
    print(f"  converged : {result['converged']}")
    print(f"  method    : {result['method']}")
    print(f"  params    : {[f'{p:.4f}' for p in result['params']]}")
    print(f"  loglike   : {result['loglike']:.4f}")
    print(f"  AIC       : {result['aic']:.4f}")
    print(f"  BIC       : {result['bic']:.4f}")
    print(f"  n_iter    : {result['n_iter']}")
    print(f"  time      : {elapsed:.3f}s")

    if not result["converged"]:
        print(f"  WARNING: did not converge")
    assert np.isfinite(result["loglike"]), "loglike should be finite"


def test_sarima_forecast_s24():
    """SARIMA(1,1,1)(1,1,1,24) fit + 48h forecast."""
    y = generate_hourly_data(14)

    t0 = time.perf_counter()
    result = sarimax_rs.sarimax_fit(y, (1, 1, 1), (1, 1, 1, 24))
    fit_time = time.perf_counter() - t0

    if not result["converged"]:
        print(f"\n  WARNING: fit did not converge, forecasting with best params anyway")

    t0 = time.perf_counter()
    params = np.array(result["params"])
    fc = sarimax_rs.sarimax_forecast(
        y,
        (1, 1, 1), (1, 1, 1, 24),
        params,
        48,  # 48 hours ahead
        0.05,  # alpha for 95% CI
    )
    fc_time = time.perf_counter() - t0

    print(f"\n--- SARIMA(1,1,1)(1,1,1,24) forecast 48h ---")
    print(f"  fit time  : {fit_time:.3f}s")
    print(f"  fc time   : {fc_time:.3f}s")
    print(f"  forecast  : {[f'{v:.2f}' for v in fc['mean'][:6]]} ... ({len(fc['mean'])} steps)")
    print(f"  CI lower  : {[f'{v:.2f}' for v in fc['ci_lower'][:6]]}")
    print(f"  CI upper  : {[f'{v:.2f}' for v in fc['ci_upper'][:6]]}")

    assert len(fc["mean"]) == 48
    assert all(np.isfinite(fc["mean"])), "forecast values should be finite"


def test_statsmodels_comparison():
    """Compare SARIMA(1,1,0)(1,0,0,24) with statsmodels (if available)."""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        print("\n--- statsmodels not installed, skipping comparison ---")
        return

    y = generate_hourly_data(14)

    # statsmodels
    t0 = time.perf_counter()
    sm_model = SARIMAX(y, order=(1, 1, 0), seasonal_order=(1, 0, 0, 24),
                       enforce_stationarity=False, enforce_invertibility=False)
    sm_result = sm_model.fit(disp=False)
    sm_time = time.perf_counter() - t0

    # sarimax_rs
    t0 = time.perf_counter()
    rs_result = sarimax_rs.sarimax_fit(y, (1, 1, 0), (1, 0, 0, 24))
    rs_time = time.perf_counter() - t0

    print(f"\n--- SARIMA(1,1,0)(1,0,0,24) statsmodels comparison ---")
    print(f"  statsmodels: loglike={sm_result.llf:.4f}, params={sm_result.params.tolist()}, time={sm_time:.3f}s")
    print(f"  sarimax_rs : loglike={rs_result['loglike']:.4f}, params={rs_result['params']}, time={rs_time:.3f}s")
    print(f"  speedup    : {sm_time / rs_time:.1f}x")
    print(f"  loglike gap: {abs(rs_result['loglike'] - sm_result.llf):.4f}")


if __name__ == "__main__":
    test_sarima_110_100_24()
    test_sarima_111_111_24()
    test_sarima_111_111_24_longer()
    test_sarima_forecast_s24()
    test_statsmodels_comparison()
    print("\n=== All hourly s=24 tests passed ===")
