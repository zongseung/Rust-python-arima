"""Matched parallel grid-search runner: rustima (Rayon) vs pmdarima (joblib).

Fixed d=0, D=1, s=24 on the 1-year demand window so both engines evaluate the
same 64-model grid (p,q in 0..3; P,Q in 0..1) with stepwise disabled.
rustima parallelizes candidates with native threads (RAYON_NUM_THREADS set by
the parent); pmdarima with n_jobs processes. Writes JSON to --out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import load_power_3y  # noqa: E402

warnings.filterwarnings("ignore")

SEARCH = dict(s=24, d=0, D=1, max_p=3, max_q=3, max_P=1, max_Q=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["rustima", "pmdarima"])
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    y, exog, _ = load_power_3y(start="2021-01-01", end="2022-01-01")
    print(
        f"[runner:grid:{args.engine}] n={y.size} d={SEARCH['d']} D={SEARCH['D']} "
        f"grid=64 n_jobs={args.n_jobs}",
        flush=True,
    )

    t0 = time.perf_counter()
    if args.engine == "rustima":
        from rustima.auto import auto_arima

        res = auto_arima(
            y, exog=exog, s=SEARCH["s"], d=SEARCH["d"], D=SEARCH["D"],
            max_p=SEARCH["max_p"], max_q=SEARCH["max_q"],
            max_P=SEARCH["max_P"], max_Q=SEARCH["max_Q"],
            stepwise=False, criterion="aic",
        )
        order = list(res.order)
        sorder = list(res.seasonal_order)
        aic = float(res.best_ic)
        n_models = int(len(res.history))
    else:
        import pmdarima as pm

        res = pm.auto_arima(
            y, X=exog, m=SEARCH["s"], d=SEARCH["d"], D=SEARCH["D"],
            max_p=SEARCH["max_p"], max_q=SEARCH["max_q"],
            max_P=SEARCH["max_P"], max_Q=SEARCH["max_Q"],
            seasonal=True, stepwise=False, n_jobs=args.n_jobs,
            information_criterion="aic",
            suppress_warnings=True, error_action="ignore",
        )
        order = list(res.order)
        sorder = list(res.seasonal_order)
        aic = float(res.aic())
        n_models = None
    elapsed = time.perf_counter() - t0

    out = {
        "engine": args.engine,
        "n_jobs": args.n_jobs,
        "order": order,
        "seasonal_order": sorder,
        "aic": aic,
        "n_models": n_models,
        "runtime_inner_s": elapsed,
        "search": SEARCH,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(
        f"[runner:grid:{args.engine}] OK order={order} sorder={sorder} "
        f"aic={aic:.3f} inner={elapsed:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
