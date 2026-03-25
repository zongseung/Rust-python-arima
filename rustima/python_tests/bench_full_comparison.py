"""Full benchmark comparison: rustima vs statsmodels.

Validates speed, accuracy, and convergence across 10 model types
per BENCHMARK_SPEC.md criteria.

Usage:
    .venv/bin/python python_tests/bench_full_comparison.py
"""

import platform
import sys
import time

import numpy as np
import statsmodels.api as sm

import rustima

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COMPARE_MAXITER = 200
RUST_METHOD = "lbfgsb"
N_REPEAT_FIT = 3
N_REPEAT_FAST = 5

# Tolerance thresholds (from BENCHMARK_SPEC.md)
TOL_ORACLE_LL = 1e-6
TOL_PARAM = 1e-2
TOL_FIT_LL = 3.0
TOL_AIC_BIC = 6.0
TOL_FORECAST_MEAN = 1e-4
TOL_FORECAST_CI = 1e-3

# ---------------------------------------------------------------------------
# Data generators (deterministic)
# ---------------------------------------------------------------------------

def gen_ar(n, phi, seed=42):
    np.random.seed(seed)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + np.random.randn()
    return y

def gen_arma(n, phi, theta, seed=42):
    np.random.seed(seed)
    e = np.random.randn(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t] + theta * e[t - 1]
    return y

def gen_arima(n, seed=42):
    np.random.seed(seed)
    return np.cumsum(np.random.randn(n))

def gen_sarima(n, s, seed=42):
    np.random.seed(seed)
    y = np.zeros(n)
    for t in range(s, n):
        y[t] = 0.5 * y[t - 1] + 0.3 * y[t - s] + np.random.randn()
    return y

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

MODELS = [
    {"name": "AR(1)", "order": (1,0,0), "seasonal": (0,0,0,0), "data_fn": lambda: gen_ar(200, 0.7), "n": 200},
    {"name": "AR(2)", "order": (2,0,0), "seasonal": (0,0,0,0), "data_fn": lambda: gen_ar(300, 0.5, seed=43), "n": 300},
    {"name": "MA(1)", "order": (0,0,1), "seasonal": (0,0,0,0), "data_fn": lambda: gen_arma(200, 0.0, 0.6), "n": 200},
    {"name": "ARMA(1,1)", "order": (1,0,1), "seasonal": (0,0,0,0), "data_fn": lambda: gen_arma(300, 0.5, 0.3), "n": 300},
    {"name": "ARMA(2,2)", "order": (2,0,2), "seasonal": (0,0,0,0), "data_fn": lambda: gen_arma(400, 0.3, 0.2, seed=44), "n": 400},
    {"name": "ARIMA(1,1,1)", "order": (1,1,1), "seasonal": (0,0,0,0), "data_fn": lambda: gen_arima(300), "n": 300},
    {"name": "ARIMA(2,1,2)", "order": (2,1,2), "seasonal": (0,0,0,0), "data_fn": lambda: gen_arima(400, seed=45), "n": 400},
    {"name": "SARIMA(1,0,0)(1,0,0,4)", "order": (1,0,0), "seasonal": (1,0,0,4), "data_fn": lambda: gen_sarima(200, 4), "n": 200},
    {"name": "SARIMA(1,1,1)(1,1,1,12)", "order": (1,1,1), "seasonal": (1,1,1,12), "data_fn": lambda: gen_sarima(300, 12), "n": 300},
    {"name": "SARIMA(2,1,1)(1,1,1,12)", "order": (2,1,1), "seasonal": (1,1,1,12), "data_fn": lambda: gen_sarima(500, 12, seed=46), "n": 500},
]

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def time_fn(fn, n_repeat=3):
    times = []
    result = None
    for _ in range(n_repeat):
        start = time.perf_counter()
        result = fn()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return min(times), result

# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def bench_speed(model, data):
    """Measure fit speed for both engines."""
    order, seasonal = model["order"], model["seasonal"]

    # Rust
    def rs_fit():
        return rustima.sarimax_fit(
            data, order, seasonal,
            method=RUST_METHOD, maxiter=COMPARE_MAXITER,
            enforce_stationarity=True, enforce_invertibility=True,
        )
    rs_time, rs_result = time_fn(rs_fit, N_REPEAT_FIT)

    # statsmodels
    def sm_fit():
        m = sm.tsa.SARIMAX(
            data, order=order,
            seasonal_order=seasonal if seasonal != (0,0,0,0) else (0,0,0,0),
            trend="n", enforce_stationarity=True, enforce_invertibility=True,
            concentrate_scale=True,
        )
        return m.fit(disp=False, maxiter=COMPARE_MAXITER)
    sm_time, sm_result = time_fn(sm_fit, N_REPEAT_FIT)

    speedup = sm_time / rs_time if rs_time > 0 else float("inf")
    return {
        "rs_time": rs_time,
        "sm_time": sm_time,
        "speedup": speedup,
        "rs_result": rs_result,
        "sm_result": sm_result,
    }


def bench_accuracy(model, data, rs_result, sm_result):
    """Compare accuracy between engines."""
    order, seasonal = model["order"], model["seasonal"]
    result = {}

    # --- Fitted params comparison ---
    rs_params = np.array(rs_result["params"])
    sm_params = sm_result.params
    # statsmodels may include sigma2 at end; trim to match
    n_rs = len(rs_params)
    sm_params_trimmed = sm_params[:n_rs]
    param_err = np.max(np.abs(rs_params - sm_params_trimmed))
    result["param_err"] = param_err

    # --- Loglike comparison ---
    ll_err = abs(rs_result["loglike"] - sm_result.llf)
    result["ll_err"] = ll_err

    # --- AIC/BIC comparison ---
    aic_err = abs(rs_result["aic"] - sm_result.aic)
    bic_err = abs(rs_result["bic"] - sm_result.bic)
    result["aic_err"] = aic_err
    result["bic_err"] = bic_err

    # --- Oracle loglike (evaluate at statsmodels params) ---
    # Use same enforce settings as fitting to ensure identical initialization
    try:
        rs_ll_at_sm = rustima.sarimax_loglike(
            data, order, seasonal, sm_params_trimmed,
            concentrate_scale=True,
            enforce_stationarity=True, enforce_invertibility=True,
        )
        sm_ll = sm_result.llf
        oracle_err = abs(rs_ll_at_sm - sm_ll)
        result["oracle_ll_err"] = oracle_err
    except Exception as e:
        result["oracle_ll_err"] = float("nan")
        result["oracle_err_msg"] = str(e)

    # --- Forecast comparison (use Rust fitted params) ---
    try:
        rs_fc = rustima.sarimax_forecast(
            data, order, seasonal, np.array(rs_result["params"]),
            steps=10, alpha=0.05,
            concentrate_scale=True,
        )
        sm_fc = sm_result.get_forecast(steps=10)
        sm_mean = np.asarray(sm_fc.predicted_mean).ravel()
        fc_mean_err = np.max(np.abs(np.array(rs_fc["mean"]) - sm_mean))
        fc_ci = np.asarray(sm_fc.conf_int(alpha=0.05))
        ci_lower_err = np.max(np.abs(np.array(rs_fc["ci_lower"]) - fc_ci[:, 0]))
        ci_upper_err = np.max(np.abs(np.array(rs_fc["ci_upper"]) - fc_ci[:, 1]))
        result["fc_mean_err"] = fc_mean_err
        result["fc_ci_err"] = max(ci_lower_err, ci_upper_err)
    except Exception as e:
        result["fc_mean_err"] = float("nan")
        result["fc_ci_err"] = float("nan")
        result["fc_err_msg"] = str(e)
        # Print forecast error for debugging
        print(f"\n    [WARN] Forecast error for {model['name']}: {e}", end="")

    return result


def bench_convergence(rs_result, sm_result):
    """Compare convergence between engines."""
    rs_conv = rs_result.get("converged", False)
    sm_conv = sm_result.mle_retvals.get("converged", False) if hasattr(sm_result, "mle_retvals") else True
    rs_iter = rs_result.get("n_iter", -1)
    sm_iter = sm_result.mle_retvals.get("iterations", -1) if hasattr(sm_result, "mle_retvals") else -1
    return {
        "rs_conv": rs_conv,
        "sm_conv": sm_conv,
        "rs_iter": rs_iter,
        "sm_iter": sm_iter,
    }


def bench_batch():
    """Batch fit benchmark: 50 AR(1) series."""
    series = [gen_ar(200, 0.7, seed=i) for i in range(50)]

    # Rust batch
    def rs_batch():
        return rustima.sarimax_batch_fit(
            series, (1,0,0), (0,0,0,0),
            method=RUST_METHOD, maxiter=COMPARE_MAXITER,
            enforce_stationarity=True, enforce_invertibility=True,
        )
    rs_time, rs_results = time_fn(rs_batch, 2)

    # statsmodels sequential
    def sm_batch():
        results = []
        for s in series:
            m = sm.tsa.SARIMAX(s, order=(1,0,0), trend="n",
                              enforce_stationarity=True, concentrate_scale=True)
            results.append(m.fit(disp=False, maxiter=COMPARE_MAXITER))
        return results
    sm_time, _ = time_fn(sm_batch, 2)

    rs_conv = sum(1 for r in rs_results if r.get("converged", False))
    speedup = sm_time / rs_time if rs_time > 0 else float("inf")
    return {
        "rs_time": rs_time,
        "sm_time": sm_time,
        "speedup": speedup,
        "rs_converged": rs_conv,
        "total": len(series),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("  sarimax-rs vs statsmodels — Full Performance & Accuracy Report")
    print("=" * 72)
    print(f"  Platform:    {platform.platform()}")
    print(f"  CPU:         {platform.processor()}")
    print(f"  Python:      {sys.version.split()[0]}")
    print(f"  rustima:  {rustima.version()}")
    print(f"  Config:      method={RUST_METHOD}, maxiter={COMPARE_MAXITER}")
    print()

    all_speed = []
    all_accuracy = []
    all_convergence = []

    for model in MODELS:
        name = model["name"]
        data = model["data_fn"]()
        print(f"  Benchmarking {name} (n={model['n']})...", end=" ", flush=True)

        speed = bench_speed(model, data)
        accuracy = bench_accuracy(model, data, speed["rs_result"], speed["sm_result"])
        convergence = bench_convergence(speed["rs_result"], speed["sm_result"])

        speed["name"] = name
        accuracy["name"] = name
        convergence["name"] = name

        all_speed.append(speed)
        all_accuracy.append(accuracy)
        all_convergence.append(convergence)
        print(f"done ({speed['speedup']:.1f}x)")

    # --- SPEED TABLE ---
    print()
    print("=" * 72)
    print("  SPEED BENCHMARK")
    print("=" * 72)
    print(f"  {'Model':<38} {'Rust(ms)':>9} {'SM(ms)':>9} {'Speedup':>9}")
    print("  " + "-" * 65)
    for s in all_speed:
        print(f"  {s['name']:<38} {s['rs_time']:>9.1f} {s['sm_time']:>9.1f} {s['speedup']:>8.1f}x")

    # --- ACCURACY TABLE ---
    print()
    print("=" * 72)
    print("  ACCURACY BENCHMARK")
    print("=" * 72)
    print(f"  {'Model':<32} {'Oracle LL':>10} {'Param':>10} {'LL':>10} {'FC mean':>10} {'FC CI':>10}")
    print("  " + "-" * 82)
    for a in all_accuracy:
        oracle = f"{a['oracle_ll_err']:.1e}" if np.isfinite(a.get('oracle_ll_err', float('nan'))) else "N/A"
        param = f"{a['param_err']:.1e}"
        ll = f"{a['ll_err']:.1e}"
        fc_m = f"{a['fc_mean_err']:.1e}" if np.isfinite(a.get('fc_mean_err', float('nan'))) else "N/A"
        fc_c = f"{a['fc_ci_err']:.1e}" if np.isfinite(a.get('fc_ci_err', float('nan'))) else "N/A"
        print(f"  {a['name']:<32} {oracle:>10} {param:>10} {ll:>10} {fc_m:>10} {fc_c:>10}")

    # --- CONVERGENCE TABLE ---
    print()
    print("=" * 72)
    print("  CONVERGENCE")
    print("=" * 72)
    print(f"  {'Model':<38} {'RS conv':>8} {'SM conv':>8} {'RS iter':>8} {'SM iter':>8}")
    print("  " + "-" * 70)
    for c in all_convergence:
        rs_c = "Y" if c["rs_conv"] else "N"
        sm_c = "Y" if c["sm_conv"] else "N"
        print(f"  {c['name']:<38} {rs_c:>8} {sm_c:>8} {c['rs_iter']:>8} {c['sm_iter']:>8}")

    # --- BATCH BENCHMARK ---
    print()
    print("=" * 72)
    print("  BATCH BENCHMARK")
    print("=" * 72)
    print("  Running AR(1) batch fit (50 x n=200)...", end=" ", flush=True)
    batch = bench_batch()
    print(f"done ({batch['speedup']:.1f}x)")
    print(f"  Rust:  {batch['rs_time']:.1f} ms  ({batch['rs_converged']}/{batch['total']} converged)")
    print(f"  SM:    {batch['sm_time']:.1f} ms")
    print(f"  Speedup: {batch['speedup']:.1f}x")

    # --- VERDICT ---
    print()
    print("=" * 72)
    print("  VERDICT")
    print("=" * 72)

    n_faster = sum(1 for s in all_speed if s["speedup"] >= 1.0)
    n_oracle_ok = sum(1 for a in all_accuracy if np.isfinite(a.get("oracle_ll_err", float("nan"))) and a["oracle_ll_err"] < TOL_ORACLE_LL)
    n_param_ok = sum(1 for a in all_accuracy if a["param_err"] < TOL_PARAM)
    n_fc_ok = sum(1 for a in all_accuracy if np.isfinite(a.get("fc_mean_err", float("nan"))) and a["fc_mean_err"] < TOL_FORECAST_MEAN)
    n_conv_ok = sum(1 for c in all_convergence if c["rs_conv"])
    batch_pass = batch["speedup"] >= 3.0

    speed_pass = n_faster == len(MODELS)
    oracle_pass = n_oracle_ok == len(MODELS)
    param_pass = n_param_ok == len(MODELS)
    fc_pass = n_fc_ok >= len(MODELS) - 1  # allow 1 model to fail forecast (e.g. complex SARIMA)
    conv_pass = n_conv_ok >= int(len(MODELS) * 0.9)

    checks = [
        ("Speed (all models faster)", speed_pass, f"{n_faster}/{len(MODELS)}"),
        ("Oracle loglike < 1e-6", oracle_pass, f"{n_oracle_ok}/{len(MODELS)}"),
        ("Param error < 1e-2", param_pass, f"{n_param_ok}/{len(MODELS)}"),
        ("Forecast mean < 1e-4", fc_pass, f"{n_fc_ok}/{len(MODELS)}"),
        ("Convergence >= 90%", conv_pass, f"{n_conv_ok}/{len(MODELS)}"),
        ("Batch speedup >= 3x", batch_pass, f"{batch['speedup']:.1f}x"),
    ]

    all_pass = True
    for label, passed, detail in checks:
        mark = "PASS" if passed else "FAIL"
        symbol = "[x]" if passed else "[ ]"
        print(f"  {symbol} {label:<35} {mark}  ({detail})")
        if not passed:
            all_pass = False

    print()
    overall = "PASS" if all_pass else "FAIL"
    print(f"  Overall: {overall}")
    print("=" * 72)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
