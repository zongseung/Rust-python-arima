#!/usr/bin/env python3
"""
Comprehensive comparison of sarimax_rs vs statsmodels SARIMAX
for ALL model orders up to 4.
"""

import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import statsmodels.api as sm
import sarimax_rs

# -- Data generation -----------------------------------------------------------
np.random.seed(42)
data = np.cumsum(np.random.randn(300))

# -- Model definitions ---------------------------------------------------------

# Non-seasonal: all (p,d,q) with p in {1..4}, d in {0,1}, q in {0..4}
non_seasonal = []
for p in range(1, 5):
    for d in [0, 1]:
        for q in range(0, 5):
            non_seasonal.append(((p, d, q), (0, 0, 0, 0)))

# Seasonal models
seasonal = [
    # s=12
    ((1, 1, 1), (1, 1, 1, 12)),
    ((1, 1, 1), (2, 1, 1, 12)),
    ((1, 1, 1), (1, 1, 2, 12)),
    ((1, 1, 1), (2, 1, 2, 12)),
    ((2, 1, 1), (1, 1, 1, 12)),
    ((1, 1, 2), (1, 1, 1, 12)),
    # s=24
    ((1, 1, 1), (1, 1, 1, 24)),
    ((1, 1, 4), (1, 1, 1, 24)),
    ((3, 1, 3), (1, 1, 1, 24)),
]

all_models = non_seasonal + seasonal

# -- Helpers -------------------------------------------------------------------

def classify(ll_diff, rs_converged, sm_converged):
    if not rs_converged or not sm_converged:
        return "FAIL"
    if abs(ll_diff) < 0.01:
        return "EXACT"
    if abs(ll_diff) < 1.0:
        return "CLOSE"
    if abs(ll_diff) < 5.0:
        return "OK"
    return "WARN"

def format_params(params, max_show=4):
    if params is None:
        return "N/A"
    arr = np.array(params)
    if len(arr) <= max_show:
        return "[" + ",".join(f"{v:.4f}" for v in arr) + "]"
    return "[" + ",".join(f"{v:.4f}" for v in arr[:max_show]) + ",...]"

def model_label(order, seasonal_order):
    p, d, q = order
    P, D, Q, s = seasonal_order
    if s == 0:
        return f"ARIMA({p},{d},{q})"
    return f"SARIMA({p},{d},{q})({P},{D},{Q},{s})"


# -- Header --------------------------------------------------------------------
print("=" * 190)
print(f"{'Model':<28} | {'SM_LL':>10} | {'RS_LL':>10} | {'LL_diff':>9} | {'SM_params':<28} | {'RS_params':<28} | {'P_maxdiff':>9} | {'SM_ms':>8} | {'RS_ms':>8} | {'Speedup':>7} | {'Status':<6}")
print("-" * 190)

total = 0
counts = {"EXACT": 0, "CLOSE": 0, "OK": 0, "WARN": 0, "FAIL": 0, "ERROR": 0}
speedups = []
warn_fail_details = []

for order, seasonal_order in all_models:
    label = model_label(order, seasonal_order)
    total += 1

    p, d, q = order
    P, D, Q, s = seasonal_order

    sm_ll = None
    rs_ll = None
    sm_params = None
    rs_params = None
    sm_time = None
    rs_time = None
    sm_converged = False
    rs_converged = False
    param_max_diff = None
    status = "ERROR"
    speedup_val = None
    error_msg = ""

    # -- statsmodels fit -------------------------------------------------------
    try:
        t0 = time.perf_counter()
        if s == 0:
            sm_model = sm.tsa.SARIMAX(data, order=(p, d, q), trend='n',
                                       enforce_stationarity=True,
                                       enforce_invertibility=True,
                                       concentrate_scale=True)
        else:
            sm_model = sm.tsa.SARIMAX(data, order=(p, d, q),
                                       seasonal_order=(P, D, Q, s),
                                       trend='n',
                                       enforce_stationarity=True,
                                       enforce_invertibility=True,
                                       concentrate_scale=True)
        sm_result = sm_model.fit(method='lbfgs', maxiter=500, disp=False)
        t1 = time.perf_counter()
        sm_time = (t1 - t0) * 1000
        sm_ll = sm_result.llf
        sm_params = sm_result.params
        sm_converged = True
    except Exception as e:
        sm_time = 0
        sm_converged = False
        error_msg += f"SM:{str(e)[:60]} "

    # -- sarimax_rs fit --------------------------------------------------------
    try:
        t0 = time.perf_counter()
        rs_result = sarimax_rs.sarimax_fit(
            data,
            order=(p, d, q),
            seasonal=(P, D, Q, s),
            concentrate_scale=True,
            enforce_stationarity=True,
            enforce_invertibility=True,
            maxiter=500,
        )
        t1 = time.perf_counter()
        rs_time = (t1 - t0) * 1000
        rs_ll = rs_result["loglike"]
        rs_params = np.array(rs_result["params"])
        rs_converged = rs_result.get("converged", True)
    except Exception as e:
        rs_time = 0
        rs_converged = False
        error_msg += f"RS:{str(e)[:60]} "

    # -- Compare ---------------------------------------------------------------
    ll_diff = None
    if sm_ll is not None and rs_ll is not None:
        ll_diff = rs_ll - sm_ll
        status = classify(ll_diff, rs_converged, sm_converged)

        if sm_params is not None and rs_params is not None:
            min_len = min(len(sm_params), len(rs_params))
            sm_p = sm_params[:min_len]
            rs_p = rs_params[:min_len]
            param_max_diff = float(np.max(np.abs(sm_p - rs_p)))
    elif sm_ll is None or rs_ll is None:
        status = "FAIL"

    if sm_time and rs_time and rs_time > 0 and sm_time > 0:
        speedup_val = sm_time / rs_time
        speedups.append(speedup_val)

    counts[status] = counts.get(status, 0) + 1

    if status in ("WARN", "FAIL", "ERROR"):
        detail = {
            "model": label,
            "status": status,
            "sm_ll": sm_ll,
            "rs_ll": rs_ll,
            "ll_diff": ll_diff if ll_diff is not None else "N/A",
            "param_max_diff": param_max_diff,
            "error": error_msg,
        }
        warn_fail_details.append(detail)

    # -- Print row -------------------------------------------------------------
    sm_ll_str = f"{sm_ll:.4f}" if sm_ll is not None else "N/A"
    rs_ll_str = f"{rs_ll:.4f}" if rs_ll is not None else "N/A"
    ll_diff_str = f"{ll_diff:+.4f}" if ll_diff is not None else "N/A"
    pdiff_str = f"{param_max_diff:.6f}" if param_max_diff is not None else "N/A"
    sm_t_str = f"{sm_time:.1f}" if sm_time is not None else "N/A"
    rs_t_str = f"{rs_time:.1f}" if rs_time is not None else "N/A"
    spd_str = f"{speedup_val:.1f}x" if speedup_val is not None else "N/A"
    sm_p_str = format_params(sm_params)
    rs_p_str = format_params(rs_params)

    print(f"{label:<28} | {sm_ll_str:>10} | {rs_ll_str:>10} | {ll_diff_str:>9} | {sm_p_str:<28} | {rs_p_str:<28} | {pdiff_str:>9} | {sm_t_str:>8} | {rs_t_str:>8} | {spd_str:>7} | {status:<6}")

# -- Summary -------------------------------------------------------------------
print("=" * 190)
print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total models tested: {total}")
print()
print("Status breakdown:")
for s_key in ["EXACT", "CLOSE", "OK", "WARN", "FAIL", "ERROR"]:
    c = counts.get(s_key, 0)
    pct = c / total * 100 if total > 0 else 0
    bar = "#" * int(pct / 2)
    print(f"  {s_key:<6}: {c:>3} ({pct:5.1f}%)  {bar}")

print()
if speedups:
    print(f"Speedup statistics (sarimax_rs vs statsmodels):")
    print(f"  Average: {np.mean(speedups):.1f}x")
    print(f"  Median:  {np.median(speedups):.1f}x")
    print(f"  Min:     {np.min(speedups):.1f}x")
    print(f"  Max:     {np.max(speedups):.1f}x")
else:
    print("No speedup data available.")

print()
if warn_fail_details:
    print("WARN/FAIL/ERROR details:")
    print("-" * 80)
    for d in warn_fail_details:
        print(f"  Model: {d['model']}")
        print(f"    Status:         {d['status']}")
        print(f"    SM loglike:     {d['sm_ll']}")
        print(f"    RS loglike:     {d['rs_ll']}")
        print(f"    LL diff:        {d['ll_diff']}")
        print(f"    Param max diff: {d['param_max_diff']}")
        if d.get('error'):
            print(f"    Error:          {d['error']}")
        print()
else:
    print("No WARN/FAIL/ERROR models -- all comparisons passed!")

print("=" * 80)
