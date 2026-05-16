#!/usr/bin/env python3
"""
SARIMA vs SARIMAX(ta+hm) 동치성 검증.
- 동일 차수 (3,0,3)(1,1,1)[24] 를 exog 유무로 fit
- exog 계수 크기 / t-stat, LL, AIC 비교
"""
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJECT_ROOT, "..", "power_demand_final.csv")


def load_1y():
    df = pd.read_csv(DATA)
    df["일시"] = pd.to_datetime(df["일시"])
    df = df[(df["일시"] >= "2019-01-01") & (df["일시"] < "2020-01-01")]
    y = df["power demand(MW)"].values.astype(np.float64)
    ex = df[["ta", "hm"]].values.astype(np.float64)
    if np.isnan(y).any():
        y = np.where(np.isnan(y), np.nanmean(y), y)
    if np.isnan(ex).any():
        for j in range(ex.shape[1]):
            ex[:, j] = np.where(np.isnan(ex[:, j]), np.nanmean(ex[:, j]), ex[:, j])
    return y, ex


def fit_rustima(y, exog, order=(3, 0, 3), seasonal=(1, 1, 1, 24)):
    from rustima.model import SARIMAXModel
    m = SARIMAXModel(y, exog=exog, order=order, seasonal_order=seasonal)
    res = m.fit()
    return res


def fit_statsmodels(y, exog, order=(3, 0, 3), seasonal=(1, 1, 1, 24)):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    m = SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal,
                enforce_stationarity=False, enforce_invertibility=False)
    res = m.fit(disp=False, maxiter=300, method="lbfgs")
    return res


def main():
    y, ex = load_1y()
    print(f"data: n={len(y)}, exog shape={ex.shape}")
    print(f"  y range:  [{y.min():.0f}, {y.max():.0f}]  mean={y.mean():.0f}")
    print(f"  ta range: [{ex[:,0].min():.1f}, {ex[:,0].max():.1f}]  mean={ex[:,0].mean():.1f}")
    print(f"  hm range: [{ex[:,1].min():.1f}, {ex[:,1].max():.1f}]  mean={ex[:,1].mean():.1f}")

    print("\n--- rustima (3,0,3)(1,1,1)[24] ---")
    print("[NO  exog]")
    r0 = fit_rustima(y, exog=None)
    print(f"  LL={r0.llf:.2f}  AIC={r0.aic:.2f}  k={len(r0.params)}")
    print(f"  params: {[f'{p:.4f}' for p in r0.params]}")

    print("[WITH exog (ta, hm)]")
    r1 = fit_rustima(y, exog=ex)
    print(f"  LL={r1.llf:.2f}  AIC={r1.aic:.2f}  k={len(r1.params)}")
    print(f"  params: {[f'{p:.4f}' for p in r1.params]}")
    # exog coeffs are first 2 in layout [exog | ar | ma | sar | sma]
    print(f"  → exog coeffs: ta={r1.params[0]:.4f}, hm={r1.params[1]:.4f}")

    print(f"\nΔ(WITH - NO):  ΔLL = {r1.llf - r0.llf:+.2f}   "
          f"ΔAIC = {r1.aic - r0.aic:+.2f}   (k diff = +2)")

    # cross-check with statsmodels
    try:
        print("\n--- statsmodels SARIMAX (3,0,3)(1,1,1)[24] ---  [교차검증]")
        s0 = fit_statsmodels(y, exog=None)
        s1 = fit_statsmodels(y, exog=ex)
        print(f"  no exog:  LL={s0.llf:.2f}  AIC={s0.aic:.2f}")
        print(f"  w/ exog:  LL={s1.llf:.2f}  AIC={s1.aic:.2f}")
        print(f"  exog coeffs: ta={s1.params[0]:.4f}, hm={s1.params[1]:.4f}")
        if hasattr(s1, 'tvalues'):
            print(f"  t-stats:  ta={s1.tvalues[0]:.2f}, hm={s1.tvalues[1]:.2f}")
        print(f"  ΔLL = {s1.llf - s0.llf:+.2f}   ΔAIC = {s1.aic - s0.aic:+.2f}")
    except Exception as e:
        print(f"  statsmodels 비교 실패: {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
