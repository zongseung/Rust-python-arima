"""
Comprehensive benchmark: sarimax_rs vs statsmodels
Multiple ARIMA/SARIMA/SARIMAX orders with accuracy and speed comparison.
"""
import time
import numpy as np
import sarimax_rs
import statsmodels.api as sm

np.random.seed(42)
n = 500
y = np.cumsum(np.random.standard_normal(n))
# Exogenous variable for SARIMAX tests
exog = np.column_stack([
    np.sin(2 * np.pi * np.arange(n) / 12),
    np.cos(2 * np.pi * np.arange(n) / 12),
])

# ─── Model configurations ───
MODELS = [
    # (label, order, seasonal, exog_flag, enforce_stat, enforce_inv)
    ("AR(1)",              (1,0,0), (0,0,0,0), False, False, False),
    ("AR(2)",              (2,0,0), (0,0,0,0), False, False, False),
    ("AR(3)",              (3,0,0), (0,0,0,0), False, False, False),
    ("MA(1)",              (0,0,1), (0,0,0,0), False, False, False),
    ("MA(2)",              (0,0,2), (0,0,0,0), False, False, False),
    ("ARMA(1,1)",          (1,0,1), (0,0,0,0), False, False, False),
    ("ARMA(2,1)",          (2,0,1), (0,0,0,0), False, False, False),
    ("ARMA(2,2)",          (2,0,2), (0,0,0,0), False, False, False),
    ("ARIMA(1,1,1)",       (1,1,1), (0,0,0,0), False, False, False),
    ("ARIMA(2,1,1)",       (2,1,1), (0,0,0,0), False, False, False),
    ("ARIMA(2,1,2)",       (2,1,2), (0,0,0,0), False, False, False),
    ("ARIMA(3,1,1)",       (3,1,1), (0,0,0,0), False, False, False),
    # Seasonal
    ("SAR(1,0,0)(1,0,0,4)",   (1,0,0), (1,0,0,4), False, False, False),
    ("SAR(1,0,0)(1,0,0,12)",  (1,0,0), (1,0,0,12), False, False, False),
    ("SARIMA(1,1,1)(1,1,1,4)", (1,1,1), (1,1,1,4), False, False, False),
    ("SARIMA(1,1,1)(1,1,1,12)", (1,1,1), (1,1,1,12), False, False, False),
    ("SARIMA(2,1,1)(1,1,1,12)", (2,1,1), (1,1,1,12), False, False, False),
    # With enforcement
    ("SARIMA(1,1,1)(1,1,1,12) [stat+inv]", (1,1,1), (1,1,1,12), False, True, True),
    # SARIMAX with exog
    ("SARIMAX(1,1,1)+exog",   (1,1,1), (0,0,0,0), True, False, False),
    ("SARIMAX(1,1,1)(1,1,1,12)+exog", (1,1,1), (1,1,1,12), True, False, False),
]

print(f"{'Model':<42} {'rs(ms)':>8} {'sm(ms)':>8} {'ratio':>7} "
      f"{'Δparam':>8} {'Δll':>8} {'rs_ll':>12} {'sm_ll':>12}")
print("─" * 120)

for label, order, seasonal, use_exog, enforce_stat, enforce_inv in MODELS:
    exog_arg = exog if use_exog else None
    s = seasonal[3] if len(seasonal) == 4 else 0

    try:
        # ── statsmodels fit ──
        t0 = time.perf_counter()
        sm_model = sm.tsa.SARIMAX(
            y, order=order,
            seasonal_order=seasonal if s > 0 else (0,0,0,0),
            exog=exog_arg,
            enforce_stationarity=enforce_stat,
            enforce_invertibility=enforce_inv,
            concentrate_scale=True,
        )
        sm_res = sm_model.fit(disp=False, maxiter=500)
        t_sm = (time.perf_counter() - t0) * 1000

        # ── sarimax_rs fit ──
        seasonal_arg = seasonal if s > 0 else (0,0,0,0)
        t0 = time.perf_counter()
        rs_result = sarimax_rs.sarimax_fit(
            y, order=order, seasonal=seasonal_arg,
            exog=exog_arg,
            enforce_stationarity=enforce_stat,
            enforce_invertibility=enforce_inv,
            concentrate_scale=True,
        )
        t_rs = (time.perf_counter() - t0) * 1000

        # ── Compare ──
        sm_params = np.array(sm_res.params)
        rs_params = np.array(rs_result["params"])
        n_params = min(len(sm_params), len(rs_params))

        # param diff (max abs)
        delta_param = np.max(np.abs(sm_params[:n_params] - rs_params[:n_params]))
        # loglike diff
        delta_ll = abs(rs_result["loglike"] - sm_res.llf)
        ratio = t_rs / t_sm if t_sm > 0 else float('inf')

        print(f"{label:<42} {t_rs:8.1f} {t_sm:8.1f} {ratio:7.2f}x "
              f"{delta_param:8.4f} {delta_ll:8.3f} {rs_result['loglike']:12.3f} {sm_res.llf:12.3f}")

    except Exception as e:
        print(f"{label:<42} ERROR: {e}")

print()
print("─" * 120)
print("ratio = rs/sm (< 1 means rs is faster)")
print("Δparam = max |param_rs - param_sm|")
print("Δll = |loglike_rs - loglike_sm|")

# ─── Cross-loglike validation (most important test) ───
print("\n\n=== Cross-Loglike Validation (same params → same loglike?) ===")
print(f"{'Model':<42} {'sm_ll(sm_params)':>18} {'rs_ll(sm_params)':>18} {'diff':>10}")
print("─" * 100)

CROSS_MODELS = [
    ("ARMA(1,1)",          (1,0,1), (0,0,0,0), False, False, False),
    ("ARIMA(1,1,1)",       (1,1,1), (0,0,0,0), False, False, False),
    ("ARIMA(2,1,2)",       (2,1,2), (0,0,0,0), False, False, False),
    ("SARIMA(1,1,1)(1,1,1,4)", (1,1,1), (1,1,1,4), False, False, False),
    ("SARIMA(1,1,1)(1,1,1,12)", (1,1,1), (1,1,1,12), False, False, False),
    ("SARIMA(2,1,1)(1,1,1,12)", (2,1,1), (1,1,1,12), False, False, False),
    ("SARIMAX(1,1,1)(1,1,1,12)+exog", (1,1,1), (1,1,1,12), True, False, False),
]

for label, order, seasonal, use_exog, enforce_stat, enforce_inv in CROSS_MODELS:
    exog_arg = exog if use_exog else None
    s = seasonal[3] if len(seasonal) == 4 else 0

    try:
        sm_model = sm.tsa.SARIMAX(
            y, order=order,
            seasonal_order=seasonal if s > 0 else (0,0,0,0),
            exog=exog_arg,
            enforce_stationarity=enforce_stat,
            enforce_invertibility=enforce_inv,
            concentrate_scale=True,
        )
        sm_res = sm_model.fit(disp=False, maxiter=500)
        sm_params = np.array(sm_res.params)

        sm_ll = sm_model.loglike(sm_params)
        rs_ll = sarimax_rs.sarimax_loglike(
            y, order=order, seasonal=seasonal if s > 0 else (0,0,0,0),
            params=sm_params, exog=exog_arg,
            enforce_stationarity=enforce_stat,
            enforce_invertibility=enforce_inv,
            concentrate_scale=True,
        )
        diff = abs(sm_ll - rs_ll)
        ok = "OK" if diff < 1e-4 else "MISMATCH"
        print(f"{label:<42} {sm_ll:18.8f} {rs_ll:18.8f} {diff:10.2e}  [{ok}]")
    except Exception as e:
        print(f"{label:<42} ERROR: {e}")

print()
print("If 'diff' is < 1e-4 for all models, loglike computation is correct.")
print("Param/loglike differences in the fit comparison above indicate")
print("optimizer convergence differences, not computational bugs.")
