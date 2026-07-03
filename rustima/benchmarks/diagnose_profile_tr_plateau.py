"""Diagnose why Rust profile-trust-region (simple_diff) plateaus below R/sm.

Runs profile-TR with three different start_params:
  (a) default (CSS-derived inside Rust)
  (b) statsmodels-sd converged params
  (c) R CSS-ML converged params

If LL stays at -69618 in all three, the plateau is the optimizer giving up
near the start. If LL jumps to -69443 / -69427 when given those starts, the
optimizer's warm-start path is the bottleneck (not the profiled objective).
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from rustima.model import SARIMAXModel

DATA = Path(__file__).resolve().parents[2] / "power_demand_final.csv"


def load_2019():
    df = pd.read_csv(DATA, parse_dates=["일시"])
    df = df[(df["일시"] >= "2019-01-01") & (df["일시"] < "2020-01-01")].reset_index(drop=True)
    y = df["power demand(MW)"].astype(float).to_numpy()
    X = df[["hm", "ta"]].astype(float).to_numpy()
    return y, X


def run(y, X, label, start=None, method="profile-trust-region"):
    t0 = time.time()
    m = SARIMAXModel(y, order=(3, 0, 3), seasonal_order=(1, 1, 1, 24),
                     exog=X, trend="n", simple_differencing=True)
    r = m.fit(method=method, maxiter=500, start_params=start)
    dt = time.time() - t0
    print(f"\n[{label}]  method={method}")
    print(f"  LL={r.llf:.4f}  AIC={r.aic:.4f}  converged={r.converged}  runtime={dt:.1f}s")
    for nm, v in zip(r.param_names, r.params):
        print(f"    {nm:>20s} = {v:+.6f}")
    return np.asarray(r.params, dtype=float), r.llf


# Reference results (from prior run, same data, n=8736 differenced)
SM_SD_PARAMS = [
    -5.116607,    # hm (x1)
    -268.364192,  # ta (x2)
    +0.853262,    # ar.L1
    -0.007422,    # ar.L2
    +0.095569,    # ar.L3
    +0.833954,    # ma.L1
    +0.720936,    # ma.L2
    +0.463877,    # ma.L3
    +0.245082,    # ar.S.L24
    -0.930502,    # ma.S.L24
    +467207.659315,  # sigma2
]
R_CSS_ML_PARAMS = [
    +1.722429,    # hm
    -122.797820,  # ta
    +0.855892,    # ar.L1
    -0.010334,    # ar.L2
    +0.095316,    # ar.L3
    +0.834931,    # ma.L1
    +0.723408,    # ma.L2
    +0.465348,    # ma.L3
    +0.247099,    # ar.S.L24
    -0.929212,    # ma.S.L24
    +466027.886998,  # sigma2
]
# Rust param order: [exog β | ar | ma | sar | sma | sigma2]
# statsmodels order: same (hm, ta, ar*, ma*, sar*, sma*, sigma2). Confirmed by names.


def main():
    y, X = load_2019()
    print(f"data: n={len(y)}, n_exog={X.shape[1]}")

    p_default, ll_default = run(y, X, "default-start (CSS inside Rust)")
    p_sm, ll_sm = run(y, X, "start=statsmodels-sd", start=SM_SD_PARAMS)
    p_r, ll_r = run(y, X, "start=R-CSS-ML", start=R_CSS_ML_PARAMS)

    print("\n=== summary ===")
    print(f"  default      LL = {ll_default:.4f}")
    print(f"  from-sm-sd   LL = {ll_sm:.4f}")
    print(f"  from-R       LL = {ll_r:.4f}")
    print(f"  sm-sd ref    LL = -69443.4903")
    print(f"  R-CSS-ML ref LL = -69427.7143")


if __name__ == "__main__":
    main()
