"""JSS §4 single-engine fit runner (Python: rustima | statsmodels).

Run as a subprocess from exp_jss_fixed_order.py so that wall time and
peak RSS are measured by the parent's OOM watchdog. Writes a JSON
result file on success; non-zero exit on failure.

Usage:
    python fit_py.py --engine rustima      --order 2,0,0 --seasonal-order 1,0,1,24 --out result.json
    python fit_py.py --engine statsmodels  --order 2,1,2 --seasonal-order 1,1,1,24 --out result.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))
from jss_common import load_power_3y  # noqa: E402

warnings.filterwarnings("ignore")


def _parse_tuple(s: str, n: int) -> tuple[int, ...]:
    vals = tuple(int(x) for x in s.split(","))
    if len(vals) != n:
        raise SystemExit(f"expected {n}-tuple, got {vals!r}")
    return vals


def fit_rustima(y, exog, order, seasonal_order, trend):
    from rustima.model import SARIMAXModel
    t0 = time.perf_counter()
    # enforce_stationarity=False so seasonal AR can sit at the |φ|≈1 boundary
    # (R/statsmodels default; required for parity on highly seasonal series).
    mod = SARIMAXModel(
        y,
        order=order,
        seasonal_order=seasonal_order,
        exog=exog,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
        simple_differencing=False,
    )
    res = mod.fit()
    elapsed = time.perf_counter() - t0
    params = np.asarray(res.params, dtype=float)
    return {
        "params": params.tolist(),
        "loglike": float(res.llf),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "scale": float(res.scale),
        "n_obs": int(res.nobs),
        "runtime_inner_s": elapsed,
    }


def fit_statsmodels(y, exog, order, seasonal_order, trend):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    t0 = time.perf_counter()
    mod = SARIMAX(
        y,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        simple_differencing=False,
        concentrate_scale=True,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    # Stationary-init Lyapunov solve fails (LU decomposition error) on highly
    # seasonal large-n data; fall back to approximate diffuse so the run can
    # proceed. Asymptotically the initialization choice is washed out for
    # n=26,280.
    try:
        res = mod.fit(disp=False, maxiter=200)
    except np.linalg.LinAlgError:
        mod.initialize_approximate_diffuse()
        res = mod.fit(disp=False, maxiter=200)
    elapsed = time.perf_counter() - t0
    params = np.asarray(res.params, dtype=float)
    return {
        "params": params.tolist(),
        "loglike": float(res.llf),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "scale": float(res.scale),
        "n_obs": int(res.nobs),
        "runtime_inner_s": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["rustima", "statsmodels"])
    ap.add_argument("--order", required=True, help="p,d,q")
    ap.add_argument("--seasonal-order", required=True, help="P,D,Q,s")
    ap.add_argument("--trend", default="n", choices=["n", "c", "t", "ct"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    order = _parse_tuple(args.order, 3)
    sorder = _parse_tuple(args.seasonal_order, 4)

    y, exog, _ = load_power_3y()
    print(f"[runner:{args.engine}] n={y.size} order={order} sorder={sorder}", flush=True)

    if args.engine == "rustima":
        out = fit_rustima(y, exog, order, sorder, args.trend)
    else:
        out = fit_statsmodels(y, exog, order, sorder, args.trend)

    out["engine"] = args.engine
    out["order"] = list(order)
    out["seasonal_order"] = list(sorder)
    out["trend"] = args.trend

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(
        f"[runner:{args.engine}] OK ll={out['loglike']:.3f} aic={out['aic']:.3f} "
        f"bic={out['bic']:.3f} inner={out['runtime_inner_s']:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
