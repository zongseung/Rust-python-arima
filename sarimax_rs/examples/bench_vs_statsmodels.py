"""
Direct benchmark: sarimax_rs (Rust) vs statsmodels (Python)
Includes s=12 and s=24 seasonal models.
"""
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sarimax_rs import sarimax_fit

# ──────────────────────────────────────────────────────────────────
# Data generator (deterministic, reproducible)
# ──────────────────────────────────────────────────────────────────
def generate_data(n, phi=0.3, seed=42):
    rng = np.random.RandomState(seed)
    eps = rng.randn(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t-1] + eps[t]
    y = np.cumsum(y)  # integrated-like
    return y


# ──────────────────────────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────────────────────────
def run_statsmodels(data, order, seasonal_order, maxiter=500):
    t0 = time.perf_counter()
    try:
        model = SARIMAX(data, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=True, enforce_invertibility=True)
        result = model.fit(maxiter=maxiter, disp=False)
        elapsed = time.perf_counter() - t0
        return elapsed, result.llf, result.mle_retvals.get('converged', True)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return elapsed, None, str(e)[:40]


def run_rust(data, order, seasonal_order, maxiter=500):
    t0 = time.perf_counter()
    try:
        result = sarimax_fit(
            data,
            order=order,
            seasonal=seasonal_order,
            concentrate_scale=True,
            enforce_stationarity=True,
            enforce_invertibility=True,
            method="lbfgsb",
            maxiter=maxiter,
        )
        elapsed = time.perf_counter() - t0
        return elapsed, result["loglike"], result["converged"]
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return elapsed, None, str(e)[:60]


# ──────────────────────────────────────────────────────────────────
# Benchmark cases
# ──────────────────────────────────────────────────────────────────
CASES = [
    # label,                    order,    seasonal_order, n
    # --- Non-seasonal ---
    ("ARMA(2,2) n=500",         (2,0,2), (0,0,0,0),    500),
    ("ARMA(4,2) n=500",         (4,0,2), (0,0,0,0),    500),
    ("ARIMA(2,1,2) n=500",      (2,1,2), (0,0,0,0),    500),
    ("ARIMA(4,1,4) n=500",      (4,1,4), (0,0,0,0),    500),
    ("ARIMA(4,2,3) n=500",      (4,2,3), (0,0,0,0),    500),

    # --- Seasonal s=12 ---
    ("SARIMA(1,1,1)(1,1,1,12) n=500",  (1,1,1), (1,1,1,12), 500),
    ("SARIMA(2,1,2)(1,1,1,12) n=500",  (2,1,2), (1,1,1,12), 500),
    ("SARIMA(3,1,2)(1,1,1,12) n=500",  (3,1,2), (1,1,1,12), 500),
    ("SARIMA(2,1,2)(2,1,1,12) n=500",  (2,1,2), (2,1,1,12), 500),
    ("SARIMA(3,1,3)(2,1,2,12) n=500",  (3,1,3), (2,1,2,12), 500),

    # --- Seasonal s=24 ---
    ("SARIMA(1,1,1)(1,1,1,24) n=500",  (1,1,1), (1,1,1,24), 500),
    ("SARIMA(2,1,2)(1,1,1,24) n=500",  (2,1,2), (1,1,1,24), 500),
    ("SARIMA(3,1,2)(1,1,1,24) n=500",  (3,1,2), (1,1,1,24), 500),
    ("SARIMA(2,1,2)(2,1,1,24) n=500",  (2,1,2), (2,1,1,24), 500),

    # --- Large n ---
    ("SARIMA(1,1,1)(1,1,1,12) n=2000", (1,1,1), (1,1,1,12), 2000),
    ("SARIMA(2,1,2)(1,1,1,12) n=2000", (2,1,2), (1,1,1,12), 2000),
    ("SARIMA(1,1,1)(1,1,1,24) n=2000", (1,1,1), (1,1,1,24), 2000),

    # --- Very large n ---
    ("ARIMA(4,1,4) n=10000",           (4,1,4), (0,0,0,0),  10000),
    ("SARIMA(1,1,1)(1,1,1,12) n=10000",(1,1,1), (1,1,1,12), 10000),
]


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print()
    header = f"{'Model':<45} | {'statsmodels':>12} | {'Rust':>12} | {'Speedup':>8} | {'LL match':>8}"
    print("=" * len(header))
    print("  SARIMAX_RS (Rust) vs STATSMODELS (Python)  — Direct Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    prev_section = None
    for label, order, seasonal, n in CASES:
        # Section separator
        section = label.split("(")[0].strip()
        if "s=12" in label or "s=24" in label:
            p,d,q = order
            P,D,Q,s = seasonal
            section = f"s={s}"
        elif "n=2000" in label or "n=10000" in label:
            section = "large-n"
        if section != prev_section and prev_section is not None:
            print("-" * len(header))
        prev_section = section

        data = generate_data(n)

        # Run statsmodels
        sm_time, sm_ll, sm_conv = run_statsmodels(data, order, seasonal)

        # Run Rust
        rs_time, rs_ll, rs_conv = run_rust(data, order, seasonal)

        # Speedup
        if sm_time > 0 and rs_time > 0:
            speedup = sm_time / rs_time
            speedup_str = f"{speedup:.1f}x"
        else:
            speedup_str = "N/A"

        # LogLike match
        if sm_ll is not None and rs_ll is not None:
            ll_diff = abs(sm_ll - rs_ll)
            ll_str = f"{ll_diff:.2f}" if ll_diff < 100 else f"DIFF={ll_diff:.0f}"
        else:
            ll_str = "ERR"

        sm_time_str = f"{sm_time*1000:.1f}ms"
        rs_time_str = f"{rs_time*1000:.1f}ms"

        print(f"{label:<45} | {sm_time_str:>12} | {rs_time_str:>12} | {speedup_str:>8} | {ll_str:>8}")

    print("=" * len(header))
    print()
    print("LL match = |loglike_statsmodels - loglike_rust| (< 3.0 is acceptable)")
    print()
