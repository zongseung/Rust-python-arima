"""s=24 (hourly) high-order SARIMA benchmark.

Models:
  SARIMA(1,1,0)(1,0,0,24)   — simplest seasonal AR
  SARIMA(1,1,1)(0,1,1,24)   — airline-type hourly
  SARIMA(1,1,1)(1,1,1,24)   — full seasonal, k_states=51
  SARIMA(2,1,2)(1,1,1,24)   — higher-order ARMA
  SARIMA(2,1,1)(2,1,1,24)   — high seasonal AR
  SARIMA(3,1,1)(1,1,1,24)   — AR(3) seasonal
  SARIMA(2,1,2)(2,1,1,24)   — high-order, k_states~75

Data lengths:
  2 weeks  (336  obs)
  1 month  (720  obs)
  3 months (2160 obs)
  6 months (4320 obs)
"""

import time
import numpy as np

import sarimax_rs
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX as SM_SARIMAX
    HAS_SM = True
except ImportError:
    HAS_SM = False

np.random.seed(2024)


# ---------------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------------

def make_hourly(n_days=30, trend_slope=0.005, noise_ar=0.5, noise_std=1.5):
    """Synthetic hourly data: trend + daily seasonality + AR(1) noise."""
    n = n_days * 24
    t = np.arange(n, dtype=float)
    trend    = 50.0 + trend_slope * t
    seasonal = 10.0 * np.sin(2 * np.pi * t / 24) + 3.0 * np.cos(2 * np.pi * t / 12)
    noise    = np.zeros(n)
    for i in range(1, n):
        noise[i] = noise_ar * noise[i - 1] + np.random.normal(0, noise_std)
    return trend + seasonal + noise


# ---------------------------------------------------------------------------
# Single-model benchmark
# ---------------------------------------------------------------------------

def bench_model(label, y, order, seasonal, n_reps=3, compare_sm=True):
    """Fit model n_reps times, return best time + fit stats."""
    best_time = float("inf")
    result = None
    for _ in range(n_reps):
        t0 = time.perf_counter()
        r = sarimax_rs.sarimax_fit(y, order, seasonal)
        elapsed = time.perf_counter() - t0
        if elapsed < best_time:
            best_time = elapsed
            result = r

    sm_time = None
    sm_ll   = None
    sm_converged = None
    if compare_sm and HAS_SM:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t0 = time.perf_counter()
                sm_mod = SM_SARIMAX(y, order=order, seasonal_order=seasonal,
                                    enforce_stationarity=True,
                                    enforce_invertibility=True)
                sm_res = sm_mod.fit(disp=False)
                sm_time = time.perf_counter() - t0
                sm_ll   = sm_res.llf
                sm_converged = True
        except Exception as e:
            sm_time = None
            sm_ll   = None
            sm_converged = False

    return dict(
        label=label,
        n=len(y),
        order=order,
        seasonal=seasonal,
        rs_time=best_time,
        rs_ll=result["loglike"],
        rs_aic=result["aic"],
        rs_bic=result["bic"],
        rs_converged=result["converged"],
        rs_method=result["method"],
        rs_niter=result["n_iter"],
        sm_time=sm_time,
        sm_ll=sm_ll,
        sm_converged=sm_converged,
    )


# ---------------------------------------------------------------------------
# Forecast benchmark
# ---------------------------------------------------------------------------

def bench_forecast(label, y, order, seasonal, steps=48):
    """Fit + forecast, return timings."""
    t0 = time.perf_counter()
    fit = sarimax_rs.sarimax_fit(y, order, seasonal)
    fit_time = time.perf_counter() - t0

    params = np.array(fit["params"])
    t0 = time.perf_counter()
    fc = sarimax_rs.sarimax_forecast(y, order, seasonal, params, steps=steps)
    fc_time = time.perf_counter() - t0

    return dict(
        label=label,
        n=len(y),
        steps=steps,
        fit_time=fit_time,
        fc_time=fc_time,
        converged=fit["converged"],
        fc_mean_first6=[round(v, 2) for v in fc["mean"][:6]],
        fc_finite=all(np.isfinite(fc["mean"])),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("sarimax-rs  s=24 (hourly) 고차 모델 벤치마크")
    print("=" * 70)

    rows = []  # for summary table

    # ------------------------------------------------------------------
    # Section 1: model family × data length
    # ------------------------------------------------------------------
    models = [
        ("SARIMA(1,1,0)(1,0,0,24)", (1,1,0), (1,0,0,24)),
        ("SARIMA(1,1,1)(0,1,1,24)", (1,1,1), (0,1,1,24)),
        ("SARIMA(1,1,1)(1,1,1,24)", (1,1,1), (1,1,1,24)),
        ("SARIMA(2,1,2)(1,1,1,24)", (2,1,2), (1,1,1,24)),
        ("SARIMA(2,1,1)(2,1,1,24)", (2,1,1), (2,1,1,24)),
        ("SARIMA(3,1,1)(1,1,1,24)", (3,1,1), (1,1,1,24)),
        ("SARIMA(2,1,2)(2,1,1,24)", (2,1,2), (2,1,1,24)),
    ]

    n_days_list = [14, 30, 90]  # 336, 720, 2160 obs

    print("\n[1/3] 모델별 × 데이터 길이별  속도 & 우도 비교")
    print("-" * 70)

    for n_days in n_days_list:
        y = make_hourly(n_days)
        n = len(y)
        print(f"\n  ▶ n={n} ({n_days}일, {n}시간)")
        print(f"  {'모델':<36} {'rs(ms)':>8}  {'sm(ms)':>8}  {'speedup':>8}  {'rs_LL':>10}  {'ΔLL':>8}  {'수렴':>5}")
        print(f"  {'-'*36} {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*5}")

        for label, order, seasonal in models:
            r = bench_model(label, y, order, seasonal, n_reps=2)
            rs_ms = r["rs_time"] * 1000
            sm_ms = r["sm_time"] * 1000 if r["sm_time"] else None
            speedup = (r["sm_time"] / r["rs_time"]) if r["sm_time"] else None
            dll = abs(r["rs_ll"] - r["sm_ll"]) if r["sm_ll"] else None

            sm_str  = f"{sm_ms:8.1f}" if sm_ms else "    N/A "
            spd_str = f"{speedup:8.1f}x" if speedup else "    N/A "
            dll_str = f"{dll:8.4f}" if dll else "    N/A "
            conv    = "✓" if r["rs_converged"] else "✗"

            print(f"  {label:<36} {rs_ms:8.1f}  {sm_str}  {spd_str}  {r['rs_ll']:10.3f}  {dll_str}  {conv:>5}")

            rows.append(dict(
                model=label, n_obs=n,
                rs_time=r["rs_time"], rs_ll=r["rs_ll"],
                rs_aic=r["rs_aic"], rs_converged=r["rs_converged"],
                sm_time=r["sm_time"], sm_ll=r["sm_ll"],
                dll=dll, speedup=speedup,
            ))

    # ------------------------------------------------------------------
    # Section 2: 6-month data (4320 obs) — only sarimax_rs
    # ------------------------------------------------------------------
    print("\n\n[2/3] 6개월 (4320시간) — sarimax_rs 단독 (statsmodels 제외: 너무 느림)")
    print("-" * 70)
    y_long = make_hourly(180)
    n = len(y_long)
    print(f"\n  ▶ n={n} (180일, {n}시간)")
    print(f"  {'모델':<36} {'rs(ms)':>8}  {'AIC':>10}  {'BIC':>10}  {'수렴':>5}  {'반복':>6}")
    print(f"  {'-'*36} {'-'*8}  {'-'*10}  {'-'*10}  {'-'*5}  {'-'*6}")

    for label, order, seasonal in models:
        t0 = time.perf_counter()
        r = sarimax_rs.sarimax_fit(y_long, order, seasonal)
        elapsed = (time.perf_counter() - t0) * 1000
        conv = "✓" if r["converged"] else "✗"
        print(f"  {label:<36} {elapsed:8.1f}  {r['aic']:10.3f}  {r['bic']:10.3f}  {conv:>5}  {r['n_iter']:>6}")

    # ------------------------------------------------------------------
    # Section 3: Forecast test
    # ------------------------------------------------------------------
    print("\n\n[3/3] 예측 (48h forecast) — SARIMA(1,1,1)(1,1,1,24)")
    print("-" * 70)
    for n_days in [14, 30, 90]:
        y = make_hourly(n_days)
        r = bench_forecast("SARIMA(1,1,1)(1,1,1,24)", y, (1,1,1), (1,1,1,24), steps=48)
        print(f"  n={len(y):5d} ({n_days:3d}일): fit={r['fit_time']*1000:7.1f}ms  "
              f"fc={r['fc_time']*1000:6.2f}ms  "
              f"finite={'✓' if r['fc_finite'] else '✗'}  "
              f"mean[0:3]={r['fc_mean_first6'][:3]}")

    # ------------------------------------------------------------------
    # Summary for README
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("README 삽입용 요약 테이블")
    print("=" * 70)

    # state dimension helper
    def k_states(order, seasonal):
        p, d, q = order
        P, D, Q, s = seasonal
        k_diff = d + s * D
        k_ord  = max(p + s * P, q + s * Q + 1)
        return k_diff + k_ord

    # Print grouped by n
    for n_days in n_days_list:
        n = n_days * 24
        y = make_hourly(n_days)
        print(f"\n### n={n} ({n_days}일)")
        print(f"| 모델 | k_states | rs(ms) | sm(ms) | 속도비 | rs LL | sm LL | ΔLL | 수렴 |")
        print(f"|------|:--------:|-------:|-------:|-------:|------:|------:|----:|:---:|")
        for label, order, seasonal in models:
            ks = k_states(order, seasonal)
            r  = bench_model(label, y, order, seasonal, n_reps=2)
            rs_ms = r["rs_time"] * 1000
            sm_ms = r["sm_time"] * 1000 if r["sm_time"] else None
            speedup = (r["sm_time"] / r["rs_time"]) if r["sm_time"] else None
            dll = abs(r["rs_ll"] - r["sm_ll"]) if r["sm_ll"] else None

            sm_str  = f"{sm_ms:.1f}" if sm_ms else "N/A"
            spd_str = f"{speedup:.1f}x" if speedup else "N/A"
            dll_str = f"{dll:.4f}" if dll else "N/A"
            conv    = "✓" if r["rs_converged"] else "✗"

            sm_ll_str = f"{r['sm_ll']:.3f}" if r['sm_ll'] is not None else "N/A"
            print(f"| {label} | {ks} | {rs_ms:.1f} | {sm_str} | {spd_str} | {r['rs_ll']:.3f} | {sm_ll_str} | {dll_str} | {conv} |")


if __name__ == "__main__":
    main()
