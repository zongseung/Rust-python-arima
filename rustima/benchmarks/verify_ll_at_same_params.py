#!/usr/bin/env python3
"""
동일 파라미터에서 rustima vs statsmodels LL 직접 비교.
- 차이가 거의 0이면 → 옵티마이저 문제
- 차이가 크면 → KF 또는 initialization 차이
"""
import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJECT_ROOT, "..", "power_demand_final.csv")
ORDER = (3, 0, 3)
SEASONAL = (1, 1, 1, 24)


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


def main():
    y, ex = load_1y()

    from statsmodels.tsa.statespace.sarimax import SARIMAX as SMSARIMAX
    sm = SMSARIMAX(y, exog=ex, order=ORDER, seasonal_order=SEASONAL,
                   enforce_stationarity=False, enforce_invertibility=False)
    sm_res = sm.fit(disp=False, maxiter=300, method="lbfgs")
    sm_params_full = np.array(sm_res.params)  # includes sigma2 at end
    sm_params_no_sigma = sm_params_full[:-1]
    sigma2_sm = sm_params_full[-1]
    print(f"[statsmodels @ its own optimum]  LL = {sm_res.llf:.4f}")
    print(f"  sigma2 = {sigma2_sm:.2f}")

    from rustima import sarimax_loglike
    p, d, q = ORDER
    P, D, Q, s = SEASONAL

    # 1) concentrated (no sigma2 in params)
    ll_rs_concentrated = sarimax_loglike(
        y, (p, d, q), (P, D, Q, s),
        np.asarray(sm_params_no_sigma, dtype=np.float64),
        exog=ex,
        concentrate_scale=True,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    print(f"\n[rustima @ sm params, concentrate=True]   LL = {ll_rs_concentrated:.4f}")
    print(f"  ΔLL (rustima - sm) = {ll_rs_concentrated - sm_res.llf:+.4f}")

    # 2) full (sigma2 included)
    try:
        ll_rs_full = sarimax_loglike(
            y, (p, d, q), (P, D, Q, s),
            np.asarray(sm_params_full, dtype=np.float64),
            exog=ex,
            concentrate_scale=False,
            enforce_stationarity=False, enforce_invertibility=False,
        )
        print(f"\n[rustima @ sm params, concentrate=False]  LL = {ll_rs_full:.4f}")
        print(f"  ΔLL (rustima - sm) = {ll_rs_full - sm_res.llf:+.4f}")
    except Exception as e:
        print(f"\n[rustima @ sm params, concentrate=False]  실패: {e}")

    # 3) sm loglike at sm params (sanity check)
    sm_ll_recomputed = sm.loglike(sm_params_full)
    print(f"\n[statsmodels.loglike() @ same params]  LL = {sm_ll_recomputed:.4f}")
    print(f"  (sm_res.llf의 정확성 확인용; 일치해야 함)")

    # 4) statsmodels with simple_differencing=True (rustima와 동일 처리)
    sm_sd = SMSARIMAX(y, exog=ex, order=ORDER, seasonal_order=SEASONAL,
                      enforce_stationarity=False, enforce_invertibility=False,
                      simple_differencing=True)
    sm_sd_res = sm_sd.fit(disp=False, maxiter=300, method="lbfgs")
    print(f"\n[statsmodels simple_differencing=True @ its own optimum]  "
          f"LL = {sm_sd_res.llf:.4f}")
    print(f"  exog: ta={sm_sd_res.params[0]:.4f}, hm={sm_sd_res.params[1]:.4f}")
    print(f"  n_obs = {sm_sd_res.nobs}")


if __name__ == "__main__":
    main()
