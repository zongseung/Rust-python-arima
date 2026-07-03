#!/usr/bin/env python3
"""
각 method × mode의 auto_arima 결과 coef 출력.
이전 매트릭스 (bench_matrix_results.csv) 결과 차수에 맞춰 다시 fit.
"""
import os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJECT_ROOT, "..", "power_demand_final.csv")

# auto_arima가 선택한 best 차수 (bench_matrix_results.csv에서)
CASES = [
    ("sarima",  "lbfgsb",          (2,0,2), (1,1,0,24)),
    ("sarima",  "lbfgsb-multi",    (2,0,2), (1,1,0,24)),
    ("sarima",  "lbfgsb-adaptive", (2,0,2), (1,1,0,24)),
    ("sarima",  "lbfgsb-hybrid",   (2,0,2), (1,1,0,24)),
    ("sarima",  "trust-region",    (2,0,2), (0,1,1,24)),
    ("sarimax", "lbfgsb",          (3,0,3), (0,1,1,24)),
    ("sarimax", "lbfgsb-multi",    (3,0,3), (0,1,1,24)),
    ("sarimax", "lbfgsb-adaptive", (3,0,3), (1,1,1,24)),
    ("sarimax", "lbfgsb-hybrid",   (3,0,3), (0,1,1,24)),
    ("sarimax", "trust-region",    (2,0,2), (0,1,1,24)),
]


def load_1y():
    df = pd.read_csv(DATA)
    df["일시"] = pd.to_datetime(df["일시"])
    df = df[(df["일시"] >= "2019-01-01") & (df["일시"] < "2020-01-01")].reset_index(drop=True)
    y = df["power demand(MW)"].values.astype(np.float64)
    ex = df[["ta", "hm"]].values.astype(np.float64)
    if np.isnan(y).any(): y = np.where(np.isnan(y), np.nanmean(y), y)
    for j in range(ex.shape[1]):
        c = ex[:, j]; ex[:, j] = np.where(np.isnan(c), np.nanmean(c), c)
    return y, ex


def name_params(order, seasonal, has_exog):
    p, d, q = order
    P, D, Q, s = seasonal
    names = []
    if has_exog:
        names += ["ta", "hm"]
    names += [f"ar.L{i+1}" for i in range(p)]
    names += [f"ma.L{i+1}" for i in range(q)]
    names += [f"ar.S.L{(i+1)*s}" for i in range(P)]
    names += [f"ma.S.L{(i+1)*s}" for i in range(Q)]
    names += ["sigma2"]
    return names


def main():
    y, ex = load_1y()
    from rustima.model import SARIMAXModel

    print("="*100)
    for mode, method, order, seasonal in CASES:
        exog = ex if mode == "sarimax" else None
        m = SARIMAXModel(y, exog=exog, order=order, seasonal_order=seasonal)
        try:
            res = m.fit(method=method)
            names = name_params(order, seasonal, mode == "sarimax")
            params = list(res.params)
            # pad/trim
            if len(names) < len(params):
                names = names + [f"p{i}" for i in range(len(names), len(params))]
            print(f"\n--- {mode:>7s} / {method:>16s}  order={order}{seasonal}  "
                  f"LL={res.llf:.2f}  AIC={res.aic:.2f} ---")
            for n, p in zip(names, params):
                print(f"    {n:>10s} = {p:+15.4f}")
        except Exception as e:
            print(f"\n--- {mode}/{method}: ERROR {e}")


if __name__ == "__main__":
    main()
