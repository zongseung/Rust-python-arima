"""StatsForecast AutoARIMA runner on the hourly demand window.

Search space mirrors jss_runners/auto_py.py and auto_r.R:
s=24, max_p/q=3, max_P/Q=1, max_d/D=1, stepwise, AIC, no drift/mean,
exog = [ta, hm]. Run as a measured subprocess; writes JSON to --out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import load_power_3y  # noqa: E402

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--years", type=int, default=3, choices=[1, 2, 3])
    args = ap.parse_args()

    end_year = 2021 + args.years
    y, exog, _ = load_power_3y(start="2021-01-01", end=f"{end_year}-01-01")
    print(f"[runner:auto:statsforecast] n={y.size} exog={exog.shape} s=24", flush=True)

    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    df = pd.DataFrame(
        {
            "unique_id": "power",
            "ds": np.arange(y.size),
            "y": y,
            "ta": exog[:, 0],
            "hm": exog[:, 1],
        }
    )
    model = AutoARIMA(
        season_length=24,
        max_p=3, max_q=3, max_P=1, max_Q=1, max_d=1, max_D=1,
        start_p=1, start_q=1, start_P=1, start_Q=1,
        seasonal=True, stepwise=True, ic="aic", approximation=False,
        allowdrift=False, allowmean=False,
    )
    sf = StatsForecast(models=[model], freq=1, n_jobs=1)

    t0 = time.perf_counter()
    sf.fit(df)
    elapsed = time.perf_counter() - t0

    fitted = sf.fitted_[0][0].model_
    p, q, P, Q, m, d, D = fitted["arma"]
    out = {
        "engine": "statsforecast",
        "order": [int(p), int(d), int(q)],
        "seasonal_order": [int(P), int(D), int(Q), int(m)],
        "aic": float(fitted["aic"]),
        "n_models": None,
        "runtime_inner_s": elapsed,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(
        f"[runner:auto:statsforecast] OK order={out['order']} "
        f"sorder={out['seasonal_order']} aic={out['aic']:.3f} "
        f"inner={elapsed:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
