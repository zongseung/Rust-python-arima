"""rustima auto_arima on 1yr/2yr/3yr power demand vs R forecast::auto.arima.

Mirrors compare_profile_methods_R_auto.R: xreg=[hm, ta], s=24, stepwise,
ic=aic, trend='n'. Writes per-horizon summary to docs/auto_arima_rs.csv.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rustima" / "python"))

from rustima.auto import auto_arima  # noqa: E402

DATA = ROOT / "power_demand_final.csv"
OUT_CSV = ROOT / "rustima" / "docs" / "auto_arima_rs.csv"


def load_window(start: str, end: str):
    df = pd.read_csv(DATA, parse_dates=["일시"])
    df = df[(df["일시"] >= start) & (df["일시"] < end)].reset_index(drop=True)
    y = df["power demand(MW)"].astype(float).to_numpy()
    X = df[["hm", "ta"]].astype(float).to_numpy()
    return y, X


def fit_horizon(label: str, start: str, end: str, simple_diff: bool):
    y, X = load_window(start, end)
    tag = f"{label}-sd{int(simple_diff)}"
    print(f"\n=== [{tag}] {start} .. {end}  n_obs={len(y)}  "
          f"simple_diff={simple_diff} ===", flush=True)
    t0 = time.time()
    res = auto_arima(
        endog=y,
        exog=X,
        s=24,
        max_p=5, max_q=5,
        max_P=2, max_Q=2,
        max_D=1,
        trend="n",
        criterion="aic",
        stepwise=True,
        simple_differencing=simple_diff,
        trace=False,
    )
    dt = time.time() - t0
    fit = res.result
    order = res.order
    sord = res.seasonal_order
    sigma2 = getattr(fit, "sigma2", None)
    if sigma2 is None:
        sigma2 = float(getattr(fit, "scale", float("nan")))
    print(
        f"[{tag}] selected: {order}{sord}  "
        f"LL={fit.llf:.4f}  AIC={fit.aic:.4f}  BIC={fit.bic:.4f}  "
        f"sigma2={sigma2:.6f}  "
        f"runtime={dt:.2f}s  evals={len(res.history)}",
        flush=True,
    )
    return {
        "horizon": label,
        "simple_diff": simple_diff,
        "n_obs": len(y),
        "order": f"({order[0]},{order[1]},{order[2]})",
        "seasonal": f"({sord[0]},{sord[1]},{sord[2]})[{sord[3]}]",
        "loglik": float(fit.llf),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "sigma2": float(sigma2),
        "runtime_s": dt,
        "n_models_evaluated": len(res.history),
        "status": "ok",
    }


def main():
    horizons = [
        ("1yr", "2019-01-01", "2020-01-01"),
        ("2yr", "2019-01-01", "2021-01-01"),
        ("3yr", "2019-01-01", "2022-01-01"),
    ]
    rows = []
    # Order: simple_diff=True first (cheaper, matches R), then False
    for simple_diff in (True, False):
        for label, start, end in horizons:
            try:
                rows.append(fit_horizon(label, start, end, simple_diff))
            except Exception as e:
                print(f"[{label}-sd{int(simple_diff)}] FAILED: "
                      f"{type(e).__name__}: {e}", flush=True)
                rows.append({"horizon": label, "simple_diff": simple_diff,
                             "status": f"failed: {e}"})
            OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
            print(f"wrote partial {OUT_CSV}", flush=True)

    print("\n--- summary ---")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
