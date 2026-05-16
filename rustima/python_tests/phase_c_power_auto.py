#!/usr/bin/env python3
"""
Phase C-real — power demand auto_arima with fit method comparison.

전력수요 1y 데이터에 대해:
  - rustima auto_arima (각 fit method: lbfgsb / lbfgsb-multi / lbfgsb-adaptive)
  - statsmodels SARIMAX (rustima가 선택한 order에서 fit, 정답 기준)
  - pmdarima auto_arima 1y (이미 측정된 값 인용)

비교:
  - 선택된 order
  - LL / AIC
  - 시간
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJECT_ROOT, "..", "power_demand_final.csv")
OUTDIR = os.path.join(PROJECT_ROOT, "..", "paper")
os.makedirs(OUTDIR, exist_ok=True)


def load_1y():
    df = pd.read_csv(DATA)
    df["일시"] = pd.to_datetime(df["일시"])
    df = df[(df["일시"] >= "2019-01-01") & (df["일시"] < "2020-01-01")].reset_index(drop=True)
    y = df["power demand(MW)"].values.astype(np.float64)
    ex = df[["ta", "hm"]].values.astype(np.float64)
    if np.isnan(y).any(): y = np.where(np.isnan(y), np.nanmean(y), y)
    for j in range(ex.shape[1]):
        col = ex[:, j]
        ex[:, j] = np.where(np.isnan(col), np.nanmean(col), col)
    return y, ex


def run_auto(y, ex, method):
    from rustima.auto import auto_arima
    t0 = time.perf_counter()
    res = auto_arima(
        y, exog=ex, s=24,
        max_p=3, max_q=3, max_P=1, max_Q=1, max_d=1, max_D=1,
        stepwise=True, criterion="aic",
        method=method,
    )
    t = time.perf_counter() - t0
    return res, t


def run_sm_fit(y, ex, order, seasonal):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    t0 = time.perf_counter()
    m = SARIMAX(y, exog=ex, order=order, seasonal_order=seasonal,
                enforce_stationarity=False, enforce_invertibility=False)
    r = m.fit(disp=False, maxiter=300, method="lbfgs")
    return r, time.perf_counter() - t0


def main():
    y, ex = load_1y()
    print(f"=== Power demand 1y (n={len(y)}, exog=(ta, hm)) ===\n")

    rows = []
    for method in ["lbfgsb", "lbfgsb-multi", "lbfgsb-adaptive"]:
        print(f"--- rustima auto_arima method={method} ---")
        try:
            res, t = run_auto(y, ex, method)
            order = tuple(res.order)
            seasonal = tuple(res.seasonal_order)
            print(f"  order = {order}  seasonal = {seasonal}")
            print(f"  AIC = {res.best_ic:.2f}  n_models = {len(res.history)}  time = {t:.1f}s")

            # statsmodels 동일 차수 비교
            sm_res, t_sm = run_sm_fit(y, ex, order, seasonal)
            print(f"  vs sm @ same order: AIC = {sm_res.aic:.2f}  LL = {sm_res.llf:.2f}  time = {t_sm:.1f}s")

            # rustima에서 동일 차수 LL 직접 계산
            ll_rustima = None
            try:
                # AIC = 2k - 2LL  →  LL = (2k - AIC) / 2 — but we need exact LL
                # use the result params loglike directly via Rust if possible
                from rustima.model import SARIMAXModel
                m2 = SARIMAXModel(y, exog=ex, order=order, seasonal_order=seasonal,
                                  enforce_stationarity=False, enforce_invertibility=False)
                fit_r = m2.fit(method=method)
                ll_rustima = float(fit_r.llf)
                print(f"  rustima LL @ chosen order (re-fit): {ll_rustima:.2f}")
            except Exception as e:
                print(f"  rustima LL re-fit failed: {e}")

            rows.append({
                "engine": "rustima",
                "fit_method": method,
                "order": str(order),
                "seasonal": str(seasonal),
                "rustima_aic": float(res.best_ic),
                "rustima_ll": ll_rustima,
                "sm_aic_same_order": float(sm_res.aic),
                "sm_ll_same_order": float(sm_res.llf),
                "delta_ll_vs_sm": (ll_rustima - float(sm_res.llf)) if ll_rustima is not None else None,
                "delta_aic_vs_sm": float(res.best_ic) - float(sm_res.aic),
                "n_models_tried": len(res.history),
                "time_s": t,
            })
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            rows.append({"engine": "rustima", "fit_method": method, "error": str(e)})
        print()

    # pmdarima 1y 결과 인용 (이미 측정됨)
    rows.append({
        "engine": "pmdarima",
        "fit_method": "default (stepwise)",
        "order": "(0, 1, 2)",
        "seasonal": "(1, 0, 1, 24)",
        "rustima_aic": None,
        "rustima_ll": None,
        "sm_aic_same_order": 144723.09,  # from compare_pmdarima_1y.csv
        "sm_ll_same_order": None,
        "delta_ll_vs_sm": None,
        "delta_aic_vs_sm": None,
        "n_models_tried": None,
        "time_s": 404.0,  # from compare_pmdarima_1y.csv
        "note": "Already-measured: 404s, AIC=144,723.09",
    })

    df = pd.DataFrame(rows)
    out = os.path.join(OUTDIR, "power_auto_method_compare.csv")
    df.to_csv(out, index=False)
    print(f"\n[CSV] {out}")
    print("\n=== Summary ===")
    cols = ["engine", "fit_method", "order", "seasonal", "rustima_aic", "sm_aic_same_order",
            "delta_aic_vs_sm", "time_s"]
    print(df[[c for c in cols if c in df.columns]].to_string())


if __name__ == "__main__":
    main()
