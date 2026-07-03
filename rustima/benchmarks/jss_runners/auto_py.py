"""JSS §4.2 single-engine auto_arima runner (Python: rustima | pmdarima).

Run as a subprocess from exp_jss_auto_arima.py so wall time and peak RSS
are measured by the parent's OOM watchdog. Writes a JSON result file on
success; non-zero exit on failure.

Usage:
    python auto_py.py --engine rustima  --out r1.json
    python auto_py.py --engine pmdarima --out r2.json
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

# Common search space (must match auto_r.R).
SEARCH = dict(
    s=24,
    max_p=3,
    max_q=3,
    max_P=1,
    max_Q=1,
    max_d=1,
    max_D=1,
    stepwise=True,
    criterion="aic",
)


def fit_rustima(y, exog):
    from rustima.auto import auto_arima
    t0 = time.perf_counter()
    res = auto_arima(
        y,
        exog=exog,
        s=SEARCH["s"],
        max_p=SEARCH["max_p"],
        max_q=SEARCH["max_q"],
        max_P=SEARCH["max_P"],
        max_Q=SEARCH["max_Q"],
        max_d=SEARCH["max_d"],
        max_D=SEARCH["max_D"],
        stepwise=SEARCH["stepwise"],
        criterion=SEARCH["criterion"],
    )
    elapsed = time.perf_counter() - t0
    return {
        "order": list(res.order),
        "seasonal_order": list(res.seasonal_order),
        "aic": float(res.best_ic),
        "n_models": int(len(res.history)),
        "runtime_inner_s": elapsed,
    }


def fit_pmdarima(y, exog):
    import pmdarima as pm
    t0 = time.perf_counter()
    res = pm.auto_arima(
        y,
        exogenous=exog,
        m=SEARCH["s"],
        max_p=SEARCH["max_p"],
        max_q=SEARCH["max_q"],
        max_P=SEARCH["max_P"],
        max_Q=SEARCH["max_Q"],
        max_d=SEARCH["max_d"],
        max_D=SEARCH["max_D"],
        seasonal=True,
        stepwise=SEARCH["stepwise"],
        information_criterion=SEARCH["criterion"],
        suppress_warnings=True,
        error_action="ignore",
    )
    elapsed = time.perf_counter() - t0
    order = list(res.order)
    sorder = list(res.seasonal_order)
    aic = float(res.aic())
    return {
        "order": order,
        "seasonal_order": sorder,
        "aic": aic,
        "n_models": None,  # pmdarima doesn't expose explicit count
        "runtime_inner_s": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["rustima", "pmdarima"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--years", type=int, default=3, choices=[1, 2, 3],
                    help="training-window length in years anchored at 2021-01-01")
    args = ap.parse_args()

    end_year = 2021 + args.years
    y, exog, _ = load_power_3y(start="2021-01-01", end=f"{end_year}-01-01")
    print(f"[runner:auto:{args.engine}] years={args.years} n={y.size} "
          f"exog={exog.shape} s={SEARCH['s']}", flush=True)

    if args.engine == "rustima":
        out = fit_rustima(y, exog)
    else:
        out = fit_pmdarima(y, exog)

    out["engine"] = args.engine
    out["search"] = SEARCH

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(
        f"[runner:auto:{args.engine}] OK order={out['order']} "
        f"sorder={out['seasonal_order']} aic={out['aic']:.3f} "
        f"inner={out['runtime_inner_s']:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
