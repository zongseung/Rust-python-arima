"""Extended-period application: South Korean hourly demand, 2019--2023 train.

Replaces the single-month (n=744) hourly application with a five-year
training window (2019-01-01 .. 2023-12-31, n=43,824) and a 48-hour holdout
starting 2024-01-01. Each (specification, engine) cell runs in a
watchdog-measured subprocess reporting log-likelihood, AIC, fit time,
convergence, peak RSS, and holdout MAPE. SARIMAX rows use [ta, hm] with
observed 2024 values supplied for the forecast horizon.

Usage:
  python demand_extended.py [--smoke] [--timeout SECONDS]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import (  # noqa: E402
    PY,
    RAW_DIR,
    data_csv_path,
    ensure_dirs,
    run_with_oom_watchdog,
    worker_env,
    write_rows_csv,
)

TRAIN_START = "2019-01-01"
TRAIN_END = "2024-01-01"  # exclusive
HORIZON = 48

# (label, order, seasonal_order, use_exog)
SPECS = [
    ("SARIMA(1,0,1)(1,0,1)_24", (1, 0, 1), (1, 0, 1, 24), False),
    ("SARIMA(2,1,1)(1,0,0)_24", (2, 1, 1), (1, 0, 0, 24), False),
    ("SARIMAX(1,1,1)(1,0,1)_24 + [ta,hm]", (1, 1, 1), (1, 0, 1, 24), True),
    ("SARIMAX(1,1,1)(1,1,1)_24 + [ta,hm]", (1, 1, 1), (1, 1, 1, 24), True),
]
ENGINES = ["rustima", "statsmodels"]


def load_window(train_start, train_end, horizon):
    import pandas as pd

    df = pd.read_csv(data_csv_path())
    df["일시"] = pd.to_datetime(df["일시"])
    df = df.sort_values("일시").reset_index(drop=True)

    tr = df[(df["일시"] >= train_start) & (df["일시"] < train_end)]
    te = df[df["일시"] >= train_end].head(horizon)
    assert len(te) == horizon, f"holdout has {len(te)} < {horizon} rows"

    def cols(d):
        y = d["power demand(MW)"].to_numpy(dtype=np.float64)
        x = d[["ta", "hm"]].to_numpy(dtype=np.float64)
        y = np.where(np.isnan(y), np.nanmean(y), y)
        for j in range(x.shape[1]):
            c = x[:, j]
            x[:, j] = np.where(np.isnan(c), np.nanmean(c), c)
        return y, x

    y_tr, x_tr = cols(tr)
    y_te, x_te = cols(te)
    return y_tr, x_tr, y_te, x_te


def run_worker(args) -> None:
    spec = SPECS[args.spec_idx]
    _, order, seasonal, use_exog = spec
    y_tr, x_tr, y_te, x_te = load_window(args.train_start, args.train_end, HORIZON)
    exog_tr = x_tr if use_exog else None
    exog_te = x_te if use_exog else None

    t0 = time.perf_counter()
    if args.engine == "rustima":
        from rustima import SARIMAXModel

        res = SARIMAXModel(
            y_tr, order=order, seasonal_order=seasonal, exog=exog_tr
        ).fit()
        fit_s = time.perf_counter() - t0
        fc = res.forecast(steps=HORIZON, exog=exog_te)
        mean = np.asarray(fc.predicted_mean)
        out = {
            "loglike": float(res.llf),
            "aic": float(res.aic),
            "converged": bool(res.converged),
        }
    else:
        import statsmodels.api as sm

        res = sm.tsa.SARIMAX(
            y_tr, exog=exog_tr, order=order, seasonal_order=seasonal,
            enforce_stationarity=True, enforce_invertibility=True,
        ).fit(disp=0)
        fit_s = time.perf_counter() - t0
        mean = np.asarray(res.get_forecast(steps=HORIZON, exog=exog_te).predicted_mean)
        out = {
            "loglike": float(res.llf),
            "aic": float(res.aic),
            "converged": bool(res.mle_retvals.get("converged", True)),
        }

    out["fit_time_s"] = fit_s
    out["mape_48h"] = float(np.mean(np.abs((y_te - mean) / y_te)) * 100.0)
    with open(args.out, "w") as f:
        json.dump(out, f)


def run_driver(args) -> None:
    ensure_dirs()
    train_start = "2023-01-01" if args.smoke else TRAIN_START
    rows = []
    for idx, (label, order, seasonal, use_exog) in enumerate(SPECS):
        for engine in ENGINES:
            fd, out_path = tempfile.mkstemp(suffix=".json", prefix="app_")
            os.close(fd)
            cmd = [
                PY, os.path.abspath(__file__), "--worker",
                "--engine", engine, "--spec-idx", str(idx),
                "--train-start", train_start, "--train-end", TRAIN_END,
                "--out", out_path,
            ]
            print(f"[run] {label:38s} {engine}", flush=True)
            r = run_with_oom_watchdog(
                cmd, timeout_s=args.timeout, env=worker_env(rayon_threads=8),
            )
            data = None
            if r.status == "ok" and os.path.exists(out_path):
                try:
                    with open(out_path) as f:
                        data = json.load(f)
                except Exception:
                    pass
            try:
                os.unlink(out_path)
            except OSError:
                pass

            row = {
                "spec": label,
                "order": str(order),
                "seasonal_order": str(seasonal),
                "exog": use_exog,
                "engine": engine,
                "status": r.status,
                "wall_time_s": round(r.wall_time_s, 2),
                "fit_time_s": round(data["fit_time_s"], 3) if data else None,
                "loglike": round(data["loglike"], 1) if data else None,
                "aic": round(data["aic"], 1) if data else None,
                "converged": data["converged"] if data else None,
                "mape_48h": round(data["mape_48h"], 3) if data else None,
                "peak_rss_gb": round(r.peak_rss_gb, 3),
            }
            if r.status != "ok":
                row["stderr_tail"] = (r.stderr or "")[-400:]
            print(
                f"  -> {row['status']} fit={row['fit_time_s']}s "
                f"aic={row['aic']} mape={row['mape_48h']}%",
                flush=True,
            )
            rows.append(row)

    suffix = "_smoke" if args.smoke else ""
    write_rows_csv(rows, os.path.join(RAW_DIR, f"application_extended{suffix}.csv"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--engine")
    ap.add_argument("--spec-idx", type=int)
    ap.add_argument("--train-start", default=TRAIN_START)
    ap.add_argument("--train-end", default=TRAIN_END)
    ap.add_argument("--out")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--timeout", type=float, default=3600.0)
    args = ap.parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_driver(args)


if __name__ == "__main__":
    main()
