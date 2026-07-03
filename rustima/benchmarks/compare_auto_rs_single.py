"""rustima auto_arima on a single horizon (for clean per-horizon peak RSS).

Usage:
    python compare_auto_rs_single.py LABEL START END

Mirrors compare_auto_rs_2021.py settings: xreg=[hm, ta], s=24, stepwise,
ic=aic, simple_differencing=True. One horizon per process so /usr/bin/time -l
captures a true per-horizon peak RSS.
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


def load_window(start: str, end: str):
    df = pd.read_csv(DATA, parse_dates=["일시"])
    df = df[(df["일시"] >= start) & (df["일시"] < end)].reset_index(drop=True)
    y = df["power demand(MW)"].astype(float).to_numpy()
    X = df[["hm", "ta"]].astype(float).to_numpy()
    return y, X


def main():
    if len(sys.argv) != 4:
        print("usage: compare_auto_rs_single.py LABEL START END", file=sys.stderr)
        sys.exit(2)
    label, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
    y, X = load_window(start, end)
    print(f"\n=== [{label}] {start} .. {end}  n_obs={len(y)} ===", flush=True)
    t0 = time.time()
    res = auto_arima(
        endog=y, exog=X, s=24,
        max_p=5, max_q=5, max_P=2, max_Q=2, max_D=1,
        trend="n", criterion="aic",
        stepwise=True, simple_differencing=True, trace=False,
    )
    dt = time.time() - t0
    fit = res.result
    order = res.order
    sord = res.seasonal_order
    sigma2 = getattr(fit, "sigma2", None)
    if sigma2 is None:
        sigma2 = float(getattr(fit, "scale", float("nan")))
    print(
        f"[{label}] selected: {order}{sord}  "
        f"LL={fit.llf:.4f}  AIC={fit.aic:.4f}  BIC={fit.bic:.4f}  "
        f"sigma2={sigma2:.6f}  runtime={dt:.2f}s  evals={len(res.history)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
