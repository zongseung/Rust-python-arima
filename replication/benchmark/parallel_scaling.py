"""Fair parallel batch-fitting comparison (time + peak RSS).

Matched conditions over identical simulated series, every condition in its
own subprocess with BLAS/Numba threading pinned to 1:

  * rustima            native Rayon threads      1, 2, 4, 8
  * statsmodels        sequential loop           1
  * statsmodels+joblib loky processes            2, 4, 8
  * StatsForecast      fixed-order ARIMA, n_jobs 1, 8

Workloads:
  * ar1 — 200 series, n=200, AR(1); interface/overhead-bound
  * s24 — 48 series,  n=1000, SARMA(1,0,1)(1,0,1)_24; kernel-bound

The single-fit estimators differ across packages (rustima/statsmodels:
state-space MLE; StatsForecast: CSS-ML via its own C kernel), so ratios
measure end-to-end batch throughput of each package's native workflow, not
identical numerical work. Peak RSS covers the whole process tree.

Usage:
  python parallel_scaling.py [--smoke] [--reps N]
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
from common import PY, RAW_DIR, ensure_dirs, run_with_oom_watchdog, worker_env, write_rows_csv  # noqa: E402

WORKLOADS = {
    # name: (n_series, n_obs, order, seasonal_order)
    "ar1": (200, 200, (1, 0, 0), (0, 0, 0, 0)),
    "s24": (48, 1000, (1, 0, 1), (1, 0, 1, 24)),
}
SMOKE_WORKLOADS = {
    "ar1": (8, 100, (1, 0, 0), (0, 0, 0, 0)),
    "s24": (4, 300, (1, 0, 1), (1, 0, 1, 24)),
}

CONDITIONS = [
    ("rustima", 1),
    ("rustima", 2),
    ("rustima", 4),
    ("rustima", 8),
    ("statsmodels_seq", 1),
    ("statsmodels_joblib", 2),
    ("statsmodels_joblib", 4),
    ("statsmodels_joblib", 8),
    ("statsforecast", 1),
    ("statsforecast", 8),
]


def simulate_sarma(n_series: int, n_obs: int, s: int, seed: int = 42) -> list[np.ndarray]:
    """Multiplicative SARMA(1,0,1)(1,0,1)_s simulation via direct recursion.

    With s == 0 this degenerates to AR(1) with phi=0.7.
    """
    rng = np.random.default_rng(seed)
    burn = 200 + 2 * max(s, 1)
    total = n_obs + burn
    out = []
    for _ in range(n_series):
        e = rng.standard_normal(total)
        y = np.zeros(total)
        if s == 0:
            for t in range(1, total):
                y[t] = 0.7 * y[t - 1] + e[t]
        else:
            phi, theta, sphi, stheta = 0.6, 0.3, 0.5, 0.3
            for t in range(s + 1, total):
                y[t] = (
                    phi * y[t - 1] + sphi * y[t - s] - phi * sphi * y[t - s - 1]
                    + e[t] + theta * e[t - 1] + stheta * e[t - s]
                    + theta * stheta * e[t - s - 1]
                )
        out.append(y[burn:].copy())
    return out


def _fit_batch_rustima(series, order, seasonal):
    import rustima

    return rustima.sarimax_batch_fit(series, order, seasonal)


def _fit_one_sm(y, order, seasonal):
    import statsmodels.api as sm

    m = sm.tsa.SARIMAX(
        y, order=order, seasonal_order=seasonal,
        enforce_stationarity=True, enforce_invertibility=True,
    )
    return float(m.fit(disp=0).llf)


def run_worker(args) -> None:
    n_series, n_obs, order, seasonal = json.loads(args.workload_spec)
    order, seasonal = tuple(order), tuple(seasonal)
    s = seasonal[3]
    series = simulate_sarma(n_series, n_obs, s)

    inner_times = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        if args.engine == "rustima":
            _fit_batch_rustima(series, order, seasonal)
        elif args.engine == "statsmodels_seq":
            for y in series:
                _fit_one_sm(y, order, seasonal)
        elif args.engine == "statsmodels_joblib":
            from joblib import Parallel, delayed

            Parallel(n_jobs=args.n_jobs, backend="loky")(
                delayed(_fit_one_sm)(y, order, seasonal) for y in series
            )
        elif args.engine == "statsforecast":
            import pandas as pd
            from statsforecast import StatsForecast
            from statsforecast.models import ARIMA

            df = pd.concat(
                [
                    pd.DataFrame(
                        {"unique_id": i, "ds": np.arange(len(y)), "y": y}
                    )
                    for i, y in enumerate(series)
                ],
                ignore_index=True,
            )
            model = ARIMA(order=order, seasonal_order=seasonal[:3], season_length=max(s, 1))
            sf = StatsForecast(models=[model], freq=1, n_jobs=args.n_jobs)
            sf.fit(df)
        else:
            raise SystemExit(f"unknown engine {args.engine}")
        inner_times.append(time.perf_counter() - t0)

    with open(args.out, "w") as f:
        json.dump({"inner_times_s": inner_times}, f)


def run_driver(args) -> None:
    ensure_dirs()
    workloads = SMOKE_WORKLOADS if args.smoke else WORKLOADS
    rows = []
    for wname, spec in workloads.items():
        for engine, n_jobs in CONDITIONS:
            fd, out_path = tempfile.mkstemp(suffix=".json", prefix="par_")
            os.close(fd)
            cmd = [
                PY, os.path.abspath(__file__), "--worker",
                "--engine", engine,
                "--n-jobs", str(n_jobs),
                "--reps", str(args.reps),
                "--workload-spec", json.dumps(spec),
                "--out", out_path,
            ]
            env = worker_env(rayon_threads=n_jobs if engine == "rustima" else 1)
            print(f"[run] {wname:4s} {engine:18s} jobs={n_jobs}", flush=True)
            r = run_with_oom_watchdog(cmd, timeout_s=args.timeout, env=env)

            inner = None
            if r.status == "ok" and os.path.exists(out_path):
                try:
                    with open(out_path) as f:
                        inner = json.load(f)["inner_times_s"]
                except Exception:
                    pass
            try:
                os.unlink(out_path)
            except OSError:
                pass

            row = {
                "workload": wname,
                "n_series": spec[0],
                "n_obs": spec[1],
                "engine": engine,
                "n_jobs": n_jobs,
                "status": r.status,
                "wall_time_s": round(r.wall_time_s, 3),
                "inner_min_s": round(min(inner), 4) if inner else None,
                "inner_median_s": round(float(np.median(inner)), 4) if inner else None,
                "peak_rss_gb": round(r.peak_rss_gb, 3),
            }
            if r.status != "ok":
                row["stderr_tail"] = (r.stderr or "")[-600:]
                print(f"  -> {r.status}: {row['stderr_tail'][-200:]}", flush=True)
            else:
                print(
                    f"  -> ok inner_min={row['inner_min_s']}s "
                    f"rss={row['peak_rss_gb']}GB",
                    flush=True,
                )
            rows.append(row)

    suffix = "_smoke" if args.smoke else ""
    write_rows_csv(rows, os.path.join(RAW_DIR, f"parallel_scaling{suffix}.csv"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--engine")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--workload-spec")
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
