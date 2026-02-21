"""
Higher-order SARIMAX benchmark: test p,q,P,Q > 1 configurations.
Cross-loglike validation + speed comparison vs statsmodels.
"""
import time
import numpy as np
import sarimax_rs
import statsmodels.api as sm

np.random.seed(42)
n = 500
y = np.cumsum(np.random.standard_normal(n))
exog = np.column_stack([
    np.sin(2 * np.pi * np.arange(n) / 12),
    np.cos(2 * np.pi * np.arange(n) / 12),
])
exog3 = np.column_stack([
    exog,
    np.random.standard_normal(n),
])

# ─── Model configurations ───
MODELS = [
    # Higher AR orders
    ("ARIMA(3,1,1)",              (3,1,1), (0,0,0,0), None),
    ("ARIMA(4,1,1)",              (4,1,1), (0,0,0,0), None),
    ("ARIMA(5,1,1)",              (5,1,1), (0,0,0,0), None),

    # Higher MA orders
    ("ARIMA(1,1,2)",              (1,1,2), (0,0,0,0), None),
    ("ARIMA(1,1,3)",              (1,1,3), (0,0,0,0), None),

    # Higher AR+MA orders
    ("ARIMA(2,1,2)",              (2,1,2), (0,0,0,0), None),
    ("ARIMA(3,1,2)",              (3,1,2), (0,0,0,0), None),
    ("ARIMA(3,1,3)",              (3,1,3), (0,0,0,0), None),

    # Higher differencing
    ("ARIMA(1,2,1)",              (1,2,1), (0,0,0,0), None),
    ("ARIMA(2,2,1)",              (2,2,1), (0,0,0,0), None),

    # SARIMA with higher seasonal orders
    ("SARIMA(1,1,1)(2,1,0,12)",   (1,1,1), (2,1,0,12), None),
    ("SARIMA(1,1,1)(0,1,2,12)",   (1,1,1), (0,1,2,12), None),
    ("SARIMA(1,1,1)(2,1,1,12)",   (1,1,1), (2,1,1,12), None),
    ("SARIMA(1,1,1)(1,1,2,12)",   (1,1,1), (1,1,2,12), None),
    ("SARIMA(2,1,1)(2,1,1,12)",   (2,1,1), (2,1,1,12), None),

    # SARIMA with higher non-seasonal + seasonal
    ("SARIMA(2,1,2)(1,1,1,12)",   (2,1,2), (1,1,1,12), None),
    ("SARIMA(3,1,1)(1,1,1,12)",   (3,1,1), (1,1,1,12), None),

    # Different seasonal periods
    ("SARIMA(1,1,1)(1,1,1,4)",    (1,1,1), (1,1,1,4), None),
    ("SARIMA(2,1,1)(1,1,1,4)",    (2,1,1), (1,1,1,4), None),

    # SARIMAX with exog (2 exog vars)
    ("SARIMAX(1,1,1)+exog",       (1,1,1), (0,0,0,0), exog),
    ("SARIMAX(2,1,1)+exog",       (2,1,1), (0,0,0,0), exog),
    ("SARIMAX(2,1,2)+exog",       (2,1,2), (0,0,0,0), exog),
    ("SARIMAX(1,1,1)(1,1,1,4)+e", (1,1,1), (1,1,1,4), exog),
    ("SARIMAX(2,1,1)(1,1,1,4)+e", (2,1,1), (1,1,1,4), exog),
    ("SARIMAX(1,1,1)(1,1,1,12)+e",(1,1,1), (1,1,1,12), exog),
    ("SARIMAX(2,1,1)(1,1,1,12)+e",(2,1,1), (1,1,1,12), exog),
    ("SARIMAX(2,1,2)(1,1,1,12)+e",(2,1,2), (1,1,1,12), exog),

    # SARIMAX with 3 exog vars
    ("SARIMAX(1,1,1)+3exog",      (1,1,1), (0,0,0,0), exog3),
    ("SARIMAX(1,1,1)(1,1,1,12)+3e",(1,1,1),(1,1,1,12), exog3),
]

# ─── Part 1: Fit comparison ───
print(f"{'Model':<42} {'rs(ms)':>8} {'sm(ms)':>8} {'ratio':>7} "
      f"{'Δparam':>8} {'Δll':>8} {'rs_ll':>12} {'sm_ll':>12}")
print("─" * 120)

fit_results = {}
for label, order, seasonal, exog_arg in MODELS:
    try:
        # statsmodels fit
        t0 = time.perf_counter()
        sm_model = sm.tsa.SARIMAX(
            y, order=order,
            seasonal_order=seasonal,
            exog=exog_arg,
            enforce_stationarity=False,
            enforce_invertibility=False,
            concentrate_scale=True,
        )
        sm_res = sm_model.fit(disp=False, maxiter=500)
        t_sm = (time.perf_counter() - t0) * 1000

        # sarimax_rs fit
        t0 = time.perf_counter()
        rs_result = sarimax_rs.sarimax_fit(
            y, order=order, seasonal=seasonal,
            exog=exog_arg,
            enforce_stationarity=False,
            enforce_invertibility=False,
            concentrate_scale=True,
        )
        t_rs = (time.perf_counter() - t0) * 1000

        sm_params = np.array(sm_res.params)
        rs_params = np.array(rs_result["params"])
        n_params = min(len(sm_params), len(rs_params))

        delta_param = np.max(np.abs(sm_params[:n_params] - rs_params[:n_params]))
        delta_ll = abs(rs_result["loglike"] - sm_res.llf)
        ratio = t_rs / t_sm if t_sm > 0 else float('inf')

        print(f"{label:<42} {t_rs:8.1f} {t_sm:8.1f} {ratio:7.2f}x "
              f"{delta_param:8.4f} {delta_ll:8.3f} {rs_result['loglike']:12.3f} {sm_res.llf:12.3f}")

        fit_results[label] = (sm_model, sm_res)

    except Exception as e:
        print(f"{label:<42} ERROR: {e}")

# ─── Part 2: Cross-loglike validation (most important!) ───
print(f"\n\n{'='*100}")
print(f"  Cross-Loglike Validation (same params → same loglike?)")
print(f"{'='*100}")
print(f"{'Model':<42} {'sm_ll':>18} {'rs_ll':>18} {'diff':>12} {'status':>10}")
print("─" * 100)

pass_count = 0
fail_count = 0
error_count = 0

for label, order, seasonal, exog_arg in MODELS:
    try:
        sm_model = sm.tsa.SARIMAX(
            y, order=order,
            seasonal_order=seasonal,
            exog=exog_arg,
            enforce_stationarity=False,
            enforce_invertibility=False,
            concentrate_scale=True,
        )
        sm_res = sm_model.fit(disp=False, maxiter=500)
        sm_params = np.array(sm_res.params)

        # Check for extreme params
        if np.any(np.abs(sm_params) > 1e6):
            print(f"{label:<42} {'SKIP':>18} {'extreme sm_params':>18} {'':>12} {'SKIP':>10}")
            continue

        sm_ll = sm_model.loglike(sm_params)
        rs_ll = sarimax_rs.sarimax_loglike(
            y, order=order, seasonal=seasonal,
            params=sm_params, exog=exog_arg,
            enforce_stationarity=False,
            enforce_invertibility=False,
            concentrate_scale=True,
        )
        diff = abs(sm_ll - rs_ll)
        ok = diff < 1e-4
        status = "OK" if ok else "MISMATCH"
        if ok:
            pass_count += 1
        else:
            fail_count += 1

        print(f"{label:<42} {sm_ll:18.8f} {rs_ll:18.8f} {diff:12.2e} [{status:>8}]")

    except Exception as e:
        error_count += 1
        err_str = str(e)[:50]
        print(f"{label:<42} {'ERROR':>18} {err_str:>18} {'':>12} {'ERROR':>10}")

print(f"\n{'─'*100}")
print(f"Results: {pass_count} OK, {fail_count} MISMATCH, {error_count} ERROR")
print(f"Cross-loglike threshold: 1e-4")
if fail_count == 0:
    print("ALL MODELS PASS cross-loglike validation!")
else:
    print(f"WARNING: {fail_count} models have loglike mismatches — investigate!")
