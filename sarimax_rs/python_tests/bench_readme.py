"""Comprehensive benchmark for README: accuracy + speed comparison.

Runs sarimax_rs vs statsmodels on multiple model configurations,
comparing both numerical accuracy and computation speed.

Usage:
    .venv/bin/python python_tests/bench_readme.py
"""

import platform
import sys
import time
import warnings

import numpy as np
import statsmodels.api as sm

import sarimax_rs

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def gen_ar1(n=200, seed=42):
    np.random.seed(seed)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.7 * y[t - 1] + np.random.randn()
    return y

def gen_ar2(n=300, seed=42):
    np.random.seed(seed)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.5 * y[t - 1] - 0.3 * y[t - 2] + np.random.randn()
    return y

def gen_ma1(n=200, seed=42):
    np.random.seed(seed)
    e = np.random.randn(n + 1)
    y = np.array([e[t] + 0.6 * e[t - 1] for t in range(1, n + 1)])
    return y

def gen_arima111(n=300, seed=42):
    np.random.seed(seed)
    return np.cumsum(np.random.randn(n))

def gen_arima211(n=400, seed=42):
    np.random.seed(seed)
    y = np.cumsum(np.random.randn(n))
    return y

def gen_sarima_100_100_4(n=200, seed=42):
    np.random.seed(seed)
    y = np.zeros(n)
    for t in range(4, n):
        y[t] = 0.5 * y[t - 1] + 0.3 * y[t - 4] + np.random.randn()
    return y

def gen_sarima_011_011_12(n=300, seed=42):
    np.random.seed(seed)
    return np.cumsum(np.random.randn(n)) + 0.3 * np.sin(np.arange(n) * 2 * np.pi / 12)

def gen_sarima_111_111_12(n=300, seed=42):
    np.random.seed(seed)
    y = np.zeros(n)
    for t in range(13, n):
        y[t] = 0.5 * y[t - 1] + 0.3 * y[t - 12] + np.random.randn()
    return np.cumsum(y)

# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------

def time_fn(fn, n_repeat=5, warmup=1):
    """Time a function, return (best_ms, median_ms, result_from_last_call)."""
    result = None
    for _ in range(warmup):
        result = fn()
    times = []
    for _ in range(n_repeat):
        start = time.perf_counter()
        result = fn()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    times.sort()
    return times[0], times[len(times) // 2], result

# ---------------------------------------------------------------------------
# Accuracy comparison
# ---------------------------------------------------------------------------

MODELS = [
    # (name, data_fn, order, seasonal_order)
    ("AR(1)", gen_ar1, (1, 0, 0), (0, 0, 0, 0)),
    ("AR(2)", gen_ar2, (2, 0, 0), (0, 0, 0, 0)),
    ("MA(1)", gen_ma1, (0, 0, 1), (0, 0, 0, 0)),
    ("ARMA(1,1)", gen_arima111, (1, 0, 1), (0, 0, 0, 0)),
    ("ARIMA(1,1,1)", gen_arima111, (1, 1, 1), (0, 0, 0, 0)),
    ("ARIMA(2,1,1)", gen_arima211, (2, 1, 1), (0, 0, 0, 0)),
    ("SARIMA(1,0,0)(1,0,0,4)", gen_sarima_100_100_4, (1, 0, 0), (1, 0, 0, 4)),
    ("SARIMA(0,1,1)(0,1,1,12)", gen_sarima_011_011_12, (0, 1, 1), (0, 1, 1, 12)),
    ("SARIMA(1,1,1)(1,1,1,12)", gen_sarima_111_111_12, (1, 1, 1), (1, 1, 1, 12)),
]

def run_accuracy():
    """Compare parameter estimates and loglike between sarimax_rs and statsmodels."""
    print("\n" + "=" * 80)
    print("ACCURACY COMPARISON: sarimax_rs vs statsmodels")
    print("=" * 80)

    rows = []
    for name, data_fn, order, seasonal in MODELS:
        y = data_fn()
        n_obs = len(y)

        # --- sarimax_rs ---
        rs = sarimax_rs.sarimax_fit(
            y, order, seasonal,
            enforce_stationarity=True,
            enforce_invertibility=True,
            method="lbfgsb",
        )

        # --- statsmodels ---
        try:
            sm_model = sm.tsa.SARIMAX(
                y, order=order, seasonal_order=seasonal,
                trend="n",
                enforce_stationarity=True,
                enforce_invertibility=True,
                concentrate_scale=True,
            )
            sm_res = sm_model.fit(disp=False, maxiter=500)
        except Exception as e:
            print(f"  [{name}] statsmodels failed: {e}")
            continue

        # Compare
        rs_params = np.array(rs["params"])
        sm_params = sm_res.params  # statsmodels may have sigma2 at end when concentrate_scale=True... check
        # statsmodels with concentrate_scale=True should not include sigma2
        n_rs = len(rs_params)
        sm_params_cmp = sm_params[:n_rs]  # match length

        param_err = np.max(np.abs(rs_params - sm_params_cmp)) if len(rs_params) == len(sm_params_cmp) else float("nan")
        ll_err = abs(rs["loglike"] - sm_res.llf)
        aic_err = abs(rs["aic"] - sm_res.aic)
        bic_err = abs(rs["bic"] - sm_res.bic)

        rows.append({
            "name": name,
            "n_obs": n_obs,
            "n_params": n_rs,
            "rs_ll": rs["loglike"],
            "sm_ll": sm_res.llf,
            "param_err": param_err,
            "ll_err": ll_err,
            "aic_err": aic_err,
            "bic_err": bic_err,
            "rs_conv": rs["converged"],
            "sm_conv": not sm_res.mle_retvals.get("warnflag", 1),
        })

    # Print table
    print(f"\n{'Model':<30} {'n':>4} {'k':>2} {'|Δparam|':>10} {'|Δloglike|':>10} {'|ΔAIC|':>8} {'|ΔBIC|':>8} {'RS conv':>8} {'SM conv':>8}")
    print("-" * 110)
    for r in rows:
        print(f"{r['name']:<30} {r['n_obs']:>4} {r['n_params']:>2} {r['param_err']:>10.6f} {r['ll_err']:>10.4f} {r['aic_err']:>8.4f} {r['bic_err']:>8.4f} {'✓' if r['rs_conv'] else '✗':>8} {'✓' if r['sm_conv'] else '✗':>8}")

    # Summary
    max_param = max(r["param_err"] for r in rows if not np.isnan(r["param_err"]))
    max_ll = max(r["ll_err"] for r in rows)
    max_aic = max(r["aic_err"] for r in rows)
    max_bic = max(r["bic_err"] for r in rows)
    print(f"\n{'MAX ERROR':<30} {'':>4} {'':>2} {max_param:>10.6f} {max_ll:>10.4f} {max_aic:>8.4f} {max_bic:>8.4f}")

    return rows

# ---------------------------------------------------------------------------
# Speed comparison
# ---------------------------------------------------------------------------

def run_speed():
    """Compare computation speed across model configurations."""
    print("\n" + "=" * 80)
    print("SPEED COMPARISON: sarimax_rs vs statsmodels")
    print("=" * 80)

    configs = [
        # (name, data_fn, order, seasonal, n_repeat)
        ("AR(1) n=200", gen_ar1, (1, 0, 0), (0, 0, 0, 0), 10),
        ("AR(2) n=300", gen_ar2, (2, 0, 0), (0, 0, 0, 0), 10),
        ("MA(1) n=200", gen_ma1, (0, 0, 1), (0, 0, 0, 0), 10),
        ("ARMA(1,1) n=300", gen_arima111, (1, 0, 1), (0, 0, 0, 0), 5),
        ("ARIMA(1,1,1) n=300", gen_arima111, (1, 1, 1), (0, 0, 0, 0), 5),
        ("ARIMA(2,1,1) n=400", gen_arima211, (2, 1, 1), (0, 0, 0, 0), 5),
        ("SARIMA(1,0,0)(1,0,0,4) n=200", gen_sarima_100_100_4, (1, 0, 0), (1, 0, 0, 4), 5),
        ("SARIMA(0,1,1)(0,1,1,12) n=300", gen_sarima_011_011_12, (0, 1, 1), (0, 1, 1, 12), 3),
        ("SARIMA(1,1,1)(1,1,1,12) n=300", gen_sarima_111_111_12, (1, 1, 1), (1, 1, 1, 12), 3),
    ]

    rows = []
    for name, data_fn, order, seasonal, n_rep in configs:
        y = data_fn()

        # sarimax_rs
        def rs_fit():
            return sarimax_rs.sarimax_fit(
                y, order, seasonal,
                enforce_stationarity=True,
                enforce_invertibility=True,
                method="lbfgsb",
            )

        rs_best, rs_med, rs_res = time_fn(rs_fit, n_repeat=n_rep)

        # statsmodels
        def sm_fit():
            model = sm.tsa.SARIMAX(
                y, order=order, seasonal_order=seasonal,
                trend="n",
                enforce_stationarity=True,
                enforce_invertibility=True,
                concentrate_scale=True,
            )
            return model.fit(disp=False, maxiter=500)

        sm_best, sm_med, sm_res = time_fn(sm_fit, n_repeat=n_rep)

        speedup = sm_best / rs_best if rs_best > 0 else float("inf")
        rows.append({
            "name": name,
            "rs_best": rs_best,
            "rs_med": rs_med,
            "sm_best": sm_best,
            "sm_med": sm_med,
            "speedup": speedup,
        })

    print(f"\n{'Model':<38} {'Rust best':>10} {'Rust med':>10} {'SM best':>10} {'SM med':>10} {'Speedup':>10}")
    print("-" * 98)
    for r in rows:
        sp_str = f"{r['speedup']:.1f}x"
        print(f"{r['name']:<38} {r['rs_best']:>9.1f}ms {r['rs_med']:>9.1f}ms {r['sm_best']:>9.1f}ms {r['sm_med']:>9.1f}ms {sp_str:>10}")

    # --- Batch comparison ---
    print("\n--- Batch Fit (parallel) ---")
    batch_sizes = [10, 100, 500]
    for bs in batch_sizes:
        series = [gen_ar1(200, seed=i) for i in range(bs)]

        def rs_batch():
            return sarimax_rs.sarimax_batch_fit(
                series, (1, 0, 0), (0, 0, 0, 0),
                enforce_stationarity=True,
                enforce_invertibility=True,
                method="lbfgsb",
            )

        def sm_batch():
            results = []
            for s in series:
                model = sm.tsa.SARIMAX(s, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0),
                    trend="n", enforce_stationarity=True, enforce_invertibility=True,
                    concentrate_scale=True)
                results.append(model.fit(disp=False, maxiter=500))
            return results

        n_rep = 3 if bs <= 100 else 2
        rs_best, _, _ = time_fn(rs_batch, n_repeat=n_rep)
        sm_best, _, _ = time_fn(sm_batch, n_repeat=n_rep)
        speedup = sm_best / rs_best if rs_best > 0 else float("inf")

        print(f"  AR(1) batch {bs:>3} series:  Rust {rs_best:>8.1f}ms  SM {sm_best:>9.1f}ms  → {speedup:.1f}x")

    # --- Forecast comparison ---
    print("\n--- Forecast Speed (10 steps, after fit) ---")
    y = gen_arima111()
    rs_res = sarimax_rs.sarimax_fit(y, (1, 1, 1), (0, 0, 0, 0))
    rs_params = np.array(rs_res["params"])

    sm_model = sm.tsa.SARIMAX(y, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
        trend="n", enforce_stationarity=True, enforce_invertibility=True, concentrate_scale=True)
    sm_res = sm_model.fit(disp=False)

    def rs_fc():
        return sarimax_rs.sarimax_forecast(y, (1, 1, 1), (0, 0, 0, 0), rs_params, steps=10)

    def sm_fc():
        return sm_res.get_forecast(steps=10)

    rs_fc_best, _, _ = time_fn(rs_fc, n_repeat=20)
    sm_fc_best, _, _ = time_fn(sm_fc, n_repeat=20)
    fc_speedup = sm_fc_best / rs_fc_best if rs_fc_best > 0 else float("inf")
    print(f"  ARIMA(1,1,1) forecast:  Rust {rs_fc_best:.2f}ms  SM {sm_fc_best:.2f}ms  → {fc_speedup:.1f}x")

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("sarimax_rs vs statsmodels — Comprehensive Comparison")
    print(f"Platform: {platform.platform()}")
    print(f"CPU: {platform.processor() or platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"sarimax_rs: {sarimax_rs.version()}")
    print(f"statsmodels: {sm.__version__}")

    accuracy_rows = run_accuracy()
    speed_rows = run_speed()

    # --- Markdown table for README ---
    print("\n" + "=" * 80)
    print("MARKDOWN FOR README (copy-paste)")
    print("=" * 80)

    print("\n### Accuracy\n")
    print("| Model | n | k | Max |Δparam| | |Δloglike| | |ΔAIC| |")
    print("|-------|:-:|:-:|:-----------:|:----------:|:------:|")
    for r in accuracy_rows:
        print(f"| {r['name']} | {r['n_obs']} | {r['n_params']} | {r['param_err']:.6f} | {r['ll_err']:.4f} | {r['aic_err']:.4f} |")

    print("\n### Speed (single fit)\n")
    print("| Model | sarimax_rs | statsmodels | Speedup |")
    print("|-------|:----------:|:-----------:|:-------:|")
    for r in speed_rows:
        sp = f"**{r['speedup']:.1f}x**" if r['speedup'] > 1.0 else f"{r['speedup']:.1f}x"
        print(f"| {r['name']} | {r['rs_best']:.1f} ms | {r['sm_best']:.1f} ms | {sp} |")


if __name__ == "__main__":
    main()
