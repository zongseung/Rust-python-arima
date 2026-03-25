"""Benchmark comparison: rustima vs statsmodels.

Usage:
    uv run python python_tests/bench_comparison.py
"""

import platform
import sys
import time

import numpy as np
import statsmodels.api as sm

import rustima

COMPARE_MAXITER = 200
RUST_METHOD = "lbfgsb-strict"


def generate_ar1_data(n=200, seed=42):
    np.random.seed(seed)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.7 * y[t - 1] + np.random.randn()
    return y


def generate_arima111_data(n=300, seed=42):
    np.random.seed(seed)
    return np.cumsum(np.random.randn(n))


def generate_sarima_data(n=300, s=12, seed=42):
    np.random.seed(seed)
    y = np.zeros(n)
    for t in range(s, n):
        y[t] = 0.5 * y[t - 1] + 0.3 * y[t - s] + np.random.randn()
    return y


def time_fn(fn, n_repeat=3):
    """Time a function, return best of n_repeat runs in milliseconds."""
    times = []
    for _ in range(n_repeat):
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return min(times)


def bench_single_fit(name, data, order, seasonal_order, n_repeat=3):
    """Compare single fit: rustima vs statsmodels."""
    # rustima — also capture iteration count
    rs_result = None
    def rs_fit():
        nonlocal rs_result
        rs_result = rustima.sarimax_fit(
            data,
            order,
            seasonal_order,
            method=RUST_METHOD,
            maxiter=COMPARE_MAXITER,
            enforce_stationarity=True,
            enforce_invertibility=True,
        )

    rs_time = time_fn(rs_fit, n_repeat=n_repeat)

    # statsmodels
    sm_result = None
    def sm_fit():
        nonlocal sm_result
        model = sm.tsa.SARIMAX(
            data,
            order=order,
            seasonal_order=seasonal_order,
            trend="n",
            enforce_stationarity=True,
            enforce_invertibility=True,
            concentrate_scale=True,
        )
        sm_result = model.fit(disp=False)

    sm_time = time_fn(sm_fit, n_repeat=n_repeat)

    speedup = sm_time / rs_time if rs_time > 0 else float("inf")
    rs_iters = rs_result.get("n_iter", "?") if rs_result else "?"
    sm_iters = getattr(sm_result, "mle_retvals", {}).get("iterations", "?") if sm_result else "?"
    sm_funcalls = getattr(sm_result, "mle_retvals", {}).get("fcalls", "?") if sm_result else "?"

    return {
        "name": name,
        "rust_ms": round(rs_time, 1),
        "statsmodels_ms": round(sm_time, 1),
        "speedup": round(speedup, 1),
        "rs_iters": rs_iters,
        "sm_iters": sm_iters,
        "sm_funcalls": sm_funcalls,
    }


def bench_batch_fit(name, series_list, order, seasonal_order, n_repeat=3):
    """Compare batch fit: rustima (parallel) vs statsmodels (sequential)."""
    # rustima batch
    rs_time = time_fn(
        lambda: rustima.sarimax_batch_fit(
            series_list,
            order,
            seasonal_order,
            method=RUST_METHOD,
            maxiter=COMPARE_MAXITER,
            enforce_stationarity=True,
            enforce_invertibility=True,
        ),
        n_repeat=n_repeat,
    )

    # statsmodels sequential
    def sm_batch():
        for data in series_list:
            model = sm.tsa.SARIMAX(
                data,
                order=order,
                seasonal_order=seasonal_order,
                trend="n",
                enforce_stationarity=True,
                enforce_invertibility=True,
                concentrate_scale=True,
            )
            model.fit(disp=False, maxiter=COMPARE_MAXITER)

    sm_time = time_fn(sm_batch, n_repeat=n_repeat)

    speedup = sm_time / rs_time if rs_time > 0 else float("inf")
    return {
        "name": name,
        "rust_ms": round(rs_time, 1),
        "statsmodels_ms": round(sm_time, 1),
        "speedup": round(speedup, 1),
    }


def main():
    print("=" * 70)
    print("rustima vs statsmodels — Performance Comparison")
    print("=" * 70)
    print(f"Platform: {platform.platform()}")
    print(f"CPU: {platform.processor()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"rustima: {rustima.version()}")
    print(f"Compare mode: rust_method={RUST_METHOD}, maxiter={COMPARE_MAXITER}, enforce=True")
    print()

    results = []

    # --- Single fit benchmarks ---
    print("Running single fit benchmarks...")

    y_ar1 = generate_ar1_data(200)
    results.append(bench_single_fit(
        "AR(1) single fit (n=200)", y_ar1, (1, 0, 0), (0, 0, 0, 0),
    ))

    y_arima = generate_arima111_data(300)
    results.append(bench_single_fit(
        "ARIMA(1,1,1) single fit (n=300)", y_arima, (1, 1, 1), (0, 0, 0, 0),
    ))

    y_sarima = generate_sarima_data(300, 12)
    results.append(bench_single_fit(
        "SARIMA(1,1,1)(1,1,1,12) single fit (n=300)",
        y_sarima, (1, 1, 1), (1, 1, 1, 12),
    ))

    # --- Batch fit benchmarks ---
    print("Running batch fit benchmarks...")

    series_100 = [generate_ar1_data(200, seed=i) for i in range(100)]
    results.append(bench_batch_fit(
        "AR(1) batch fit (100 x n=200)",
        series_100, (1, 0, 0), (0, 0, 0, 0),
        n_repeat=2,
    ))

    # --- Print results ---
    print()
    print(f"{'Scenario':<45} {'Rust (ms)':>10} {'SM (ms)':>10} {'Speedup':>10} {'RS it/eval':>10} {'SM iter':>8} {'SM fcall':>9}")
    print("-" * 105)
    for r in results:
        rs_it = str(r.get("rs_iters", ""))
        sm_it = str(r.get("sm_iters", ""))
        sm_fc = str(r.get("sm_funcalls", ""))
        print(
            f"{r['name']:<45} {r['rust_ms']:>10.1f} {r['statsmodels_ms']:>10.1f} {r['speedup']:>9.1f}x {rs_it:>10} {sm_it:>8} {sm_fc:>9}"
        )
    print()


if __name__ == "__main__":
    main()
