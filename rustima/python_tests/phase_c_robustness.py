#!/usr/bin/env python3
"""
Phase C — robustness 검증 실험.
=============================
20개 무작위 SARIMAX 데이터셋에서 다음을 비교:
  - rustima method='lbfgsb' (baseline single-start)
  - rustima method='lbfgsb-multi' (uniform multi-start)
  - rustima method='lbfgsb-adaptive' (Phase B: gradient-informed basin hopping)
  - statsmodels SARIMAX (sm benchmark)

각 데이터셋은:
  - SARIMAX(2,0,2)(1,1,1)[12] 구조
  - 1년치 (n=720), exog 2개 (랜덤 walk + sine + noise)
  - 진짜 파라미터로 시뮬레이트 후 노이즈 추가

측정:
  - 도달 LL
  - 시간
  - ΔLL vs statsmodels
  - 수렴 성공/실패

산출:
  - paper/adaptive_restart_results.csv
"""
import os, sys, time, warnings, json
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(PROJECT_ROOT, "..", "paper")
os.makedirs(OUTDIR, exist_ok=True)
CSV_PATH = os.path.join(OUTDIR, "adaptive_restart_results.csv")

N_DATASETS = 20
N = 720  # 1y of hourly-ish data (12-periodic)
S = 12
ORDER = (2, 0, 2)
SEASONAL = (1, 1, 1, S)


def gen_dataset(seed: int):
    """Generate a SARIMAX-like time series with exog effects."""
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    # 2 exog: AR(1) + sine, both with random scale
    exog = np.column_stack([
        np.cumsum(rng.normal(0, 1.0, N)),                    # random walk
        20 * np.sin(2 * np.pi * t / S) + rng.normal(0, 1.0, N),  # seasonal sine
    ])
    # True params
    true_beta = rng.uniform(-50, 50, 2)
    # Build series: trend + exog effect + seasonal + AR-ish + noise
    base = 1000.0
    seasonal = 200.0 * np.sin(2 * np.pi * t / S)
    exog_eff = exog @ true_beta
    noise = rng.normal(0, 50.0, N)
    arma = np.zeros(N)
    for i in range(2, N):
        arma[i] = 0.7 * arma[i-1] - 0.2 * arma[i-2] + 0.3 * noise[i-1] + 0.1 * noise[i-2]
    y = base + seasonal + exog_eff + arma + noise
    return y, exog, true_beta


def fit_rustima(y, exog, method):
    from rustima.model import SARIMAXModel
    t0 = time.perf_counter()
    try:
        m = SARIMAXModel(y, exog=exog, order=ORDER, seasonal_order=SEASONAL,
                         enforce_stationarity=False, enforce_invertibility=False)
        res = m.fit(method=method)
        t = time.perf_counter() - t0
        return {"ll": float(res.llf), "time_s": t, "ok": True, "ta": float(res.params[0]),
                "aic": float(res.aic)}
    except Exception as e:
        return {"ll": None, "time_s": time.perf_counter() - t0, "ok": False,
                "err": f"{type(e).__name__}: {str(e)[:120]}", "ta": None, "aic": None}


def fit_sm(y, exog):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    t0 = time.perf_counter()
    try:
        m = SARIMAX(y, exog=exog, order=ORDER, seasonal_order=SEASONAL,
                    enforce_stationarity=False, enforce_invertibility=False)
        r = m.fit(disp=False, maxiter=300, method="lbfgs")
        t = time.perf_counter() - t0
        return {"ll": float(r.llf), "time_s": t, "ok": True, "ta": float(r.params[0]),
                "aic": float(r.aic)}
    except Exception as e:
        return {"ll": None, "time_s": time.perf_counter() - t0, "ok": False,
                "err": f"{type(e).__name__}: {str(e)[:120]}", "ta": None, "aic": None}


def main():
    methods = ["lbfgsb", "lbfgsb-multi", "lbfgsb-adaptive"]
    rows = []
    print(f"=== Phase C: robustness over {N_DATASETS} synthetic SARIMAX datasets ===")
    print(f"    n={N}, order={ORDER}, seasonal={SEASONAL}")
    print()
    for k in range(N_DATASETS):
        seed = 100 + k
        y, exog, true_beta = gen_dataset(seed)
        print(f"[{k+1:2d}/{N_DATASETS}] seed={seed}  true_beta={true_beta.round(2).tolist()}")

        sm_r = fit_sm(y, exog)
        sm_ll = sm_r["ll"] if sm_r["ok"] else None
        for method in methods:
            r = fit_rustima(y, exog, method)
            delta = (r["ll"] - sm_ll) if (r["ok"] and sm_ll is not None) else None
            rows.append({
                "seed": seed, "engine": "rustima", "method": method,
                "ll": r["ll"], "time_s": r["time_s"], "ta": r["ta"], "aic": r["aic"],
                "ok": r["ok"], "delta_ll_vs_sm": delta,
                "sm_ll": sm_ll, "true_beta_0": float(true_beta[0]),
                "true_beta_1": float(true_beta[1]),
            })
            ll_str = f"{r['ll']:+.1f}" if r["ll"] is not None else "FAIL"
            d_str = f"{delta:+.2f}" if delta is not None else "—"
            print(f"      rustima {method:>18s}  LL={ll_str:>10s}  ΔLL={d_str:>8s}  t={r['time_s']:5.1f}s")
        rows.append({
            "seed": seed, "engine": "statsmodels", "method": "lbfgs",
            "ll": sm_r["ll"], "time_s": sm_r["time_s"], "ta": sm_r["ta"], "aic": sm_r["aic"],
            "ok": sm_r["ok"], "delta_ll_vs_sm": 0.0 if sm_r["ok"] else None,
            "sm_ll": sm_r["ll"], "true_beta_0": float(true_beta[0]),
            "true_beta_1": float(true_beta[1]),
        })
        sm_str = f"{sm_r['ll']:+.1f}" if sm_r["ll"] is not None else "FAIL"
        print(f"      statsmodels       lbfgs           LL={sm_str:>10s}                  t={sm_r['time_s']:5.1f}s")
        print()

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"\n[CSV] saved → {CSV_PATH}")

    # Summary stats
    print("\n=== Summary ===")
    grp = df.groupby(["engine", "method"]).agg(
        n_ok=("ok", "sum"),
        mean_ll=("ll", "mean"),
        mean_delta=("delta_ll_vs_sm", "mean"),
        median_delta=("delta_ll_vs_sm", "median"),
        n_match_sm=("delta_ll_vs_sm", lambda s: int((s.fillna(-9999) > -0.5).sum())),
        mean_time=("time_s", "mean"),
    )
    print(grp.to_string())


if __name__ == "__main__":
    main()
