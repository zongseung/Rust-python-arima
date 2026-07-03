"""Compare SARIMAX(3,0,3)(1,1,1)[24] with exog=[hm, ta] across methods.

Rust profile-trust-region (simple_diff / exact) and statsmodels (simple_diff
/ exact). Emits full parameter values (β_hm, β_ta, ar1..ar3, ma1..ma3, sar1,
sma1, sigma2), LL, AIC, n_obs, runtime to a CSV plus a pretty Markdown table.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from rustima.model import SARIMAXModel  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "power_demand_final.csv"
OUT_CSV = Path(__file__).resolve().parents[1] / "docs" / "param_compare_2019.csv"


def load_2019():
    df = pd.read_csv(DATA, parse_dates=["일시"])
    df = df[(df["일시"] >= "2019-01-01") & (df["일시"] < "2020-01-01")].reset_index(drop=True)
    y = df["power demand(MW)"].astype(float).to_numpy()
    X = df[["hm", "ta"]].astype(float).to_numpy()
    return y, X


def fit_rust(y, X, simple_diff: bool):
    t0 = time.time()
    model = SARIMAXModel(
        endog=y,
        order=(3, 0, 3),
        seasonal_order=(1, 1, 1, 24),
        exog=X,
        trend="n",
        simple_differencing=simple_diff,
    )
    res = model.fit(method="profile-trust-region", maxiter=500)
    dt = time.time() - t0
    return {
        "method": f"rust_profile_tr_{'sd' if simple_diff else 'exact'}",
        "params": np.asarray(res.params, dtype=float),
        "names": list(res.param_names),
        "loglike": float(res.llf),
        "aic": float(res.aic),
        "n_obs": int(res.nobs),
        "converged": bool(res.converged),
        "runtime_s": dt,
    }


def fit_statsmodels(y, X, simple_diff: bool):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    t0 = time.time()
    sm = SARIMAX(
        endog=y,
        exog=X,
        order=(3, 0, 3),
        seasonal_order=(1, 1, 1, 24),
        trend="n",
        simple_differencing=simple_diff,
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    res = sm.fit(disp=False, maxiter=500, method="lbfgs")
    dt = time.time() - t0
    return {
        "method": f"statsmodels_{'sd' if simple_diff else 'exact'}",
        "params": np.asarray(res.params, dtype=float),
        "names": list(res.param_names),
        "loglike": float(res.llf),
        "aic": float(res.aic),
        "n_obs": int(res.nobs),
        "converged": bool(res.mle_retvals.get("converged", False)),
        "runtime_s": dt,
    }


def main():
    y, X = load_2019()
    print(f"data: n={len(y)}, n_exog={X.shape[1]}")

    results = []
    for fn, label in [
        (lambda: fit_rust(y, X, True), "rust_profile_tr_sd"),
        (lambda: fit_rust(y, X, False), "rust_profile_tr_exact"),
        (lambda: fit_statsmodels(y, X, True), "statsmodels_sd"),
        (lambda: fit_statsmodels(y, X, False), "statsmodels_exact"),
    ]:
        print(f"\n=== fitting {label} ===")
        try:
            r = fn()
            results.append(r)
            print(f"  LL={r['loglike']:.4f}  AIC={r['aic']:.4f}  "
                  f"converged={r['converged']}  runtime={r['runtime_s']:.1f}s")
            for nm, v in zip(r["names"], r["params"]):
                print(f"    {nm:>20s} = {v:+.6f}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            results.append({"method": label, "error": str(e)})

    # Aggregate to a wide DataFrame: rows=param_name, cols=method
    all_names = []
    for r in results:
        if "names" in r:
            for nm in r["names"]:
                if nm not in all_names:
                    all_names.append(nm)
    summary = pd.DataFrame(index=all_names)
    meta_rows = ["loglike", "aic", "n_obs", "converged", "runtime_s"]
    # Build meta with object dtype from a dict to avoid pandas float-upcast errors
    meta_data = {}
    for r in results:
        col = r["method"]
        meta_data[col] = [r.get(k, np.nan) for k in meta_rows]
    meta = pd.DataFrame(meta_data, index=meta_rows, dtype=object)
    for r in results:
        col = r["method"]
        if "params" in r:
            for nm, v in zip(r["names"], r["params"]):
                summary.loc[nm, col] = v

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([summary, meta])
    combined.to_csv(OUT_CSV)
    print(f"\nwrote {OUT_CSV}")
    print("\n--- summary ---")
    print(combined.round(4).to_string())


if __name__ == "__main__":
    main()
