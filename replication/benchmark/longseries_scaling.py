"""Long-series / high-resolution scaling benchmark (time + peak RSS + DNF).

Fits SARIMA(1,0,1)(1,0,1)_s to simulated multiplicative-SARMA data over

    s in {24, 168}   x   n in {10000, 50000, 100000}

with rustima and statsmodels, one subprocess per cell, watchdog-limited by
wall clock and peak RSS. Cells that exceed the limits are reported as
timeout/oom rather than being silently dropped — the point is to map the
largest problem size each engine completes on a 24 GB host. State dimension
is s+2 for this specification (26 for s=24, 170 for s=168).

Usage:
  python longseries_scaling.py [--smoke] [--timeout SECONDS]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from common import PY, RAW_DIR, ensure_dirs, run_with_oom_watchdog, worker_env, write_rows_csv  # noqa: E402
from parallel_scaling import simulate_sarma  # noqa: E402

ORDER = (1, 0, 1)
GRID_S = [24, 168]
GRID_N = [10_000, 50_000, 100_000]
ENGINES = ["rustima", "statsmodels"]


def run_worker(args) -> None:
    y = simulate_sarma(1, args.n, args.s, seed=7)[0]
    seasonal = (1, 0, 1, args.s)

    t0 = time.perf_counter()
    if args.engine == "rustima":
        from rustima import SARIMAXModel

        res = SARIMAXModel(y, order=ORDER, seasonal_order=seasonal).fit()
        loglike, converged = float(res.llf), bool(res.converged)
    else:
        import statsmodels.api as sm

        res = sm.tsa.SARIMAX(
            y, order=ORDER, seasonal_order=seasonal,
            enforce_stationarity=True, enforce_invertibility=True,
        ).fit(disp=0)
        loglike = float(res.llf)
        converged = bool(res.mle_retvals.get("converged", True))
    inner = time.perf_counter() - t0

    with open(args.out, "w") as f:
        json.dump({"inner_s": inner, "loglike": loglike, "converged": converged}, f)


def run_driver(args) -> None:
    ensure_dirs()
    grid_s = [24] if args.smoke else GRID_S
    grid_n = [2000] if args.smoke else GRID_N

    rows = []
    for s in grid_s:
        for n in grid_n:
            for engine in ENGINES:
                fd, out_path = tempfile.mkstemp(suffix=".json", prefix="long_")
                os.close(fd)
                cmd = [
                    PY, os.path.abspath(__file__), "--worker",
                    "--engine", engine, "--n", str(n), "--s", str(s),
                    "--out", out_path,
                ]
                print(f"[run] s={s:<3d} n={n:<6d} {engine}", flush=True)
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
                    "s": s,
                    "n": n,
                    "k_states": s + 2,
                    "engine": engine,
                    "status": r.status,
                    "wall_time_s": round(r.wall_time_s, 2),
                    "fit_time_s": round(data["inner_s"], 3) if data else None,
                    "loglike": data["loglike"] if data else None,
                    "converged": data["converged"] if data else None,
                    "peak_rss_gb": round(r.peak_rss_gb, 3),
                }
                if r.status != "ok":
                    row["stderr_tail"] = (r.stderr or "")[-400:]
                print(
                    f"  -> {row['status']} fit={row['fit_time_s']}s "
                    f"rss={row['peak_rss_gb']}GB",
                    flush=True,
                )
                rows.append(row)

    suffix = "_smoke" if args.smoke else ""
    write_rows_csv(rows, os.path.join(RAW_DIR, f"longseries_scaling{suffix}.csv"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--engine")
    ap.add_argument("--n", type=int)
    ap.add_argument("--s", type=int)
    ap.add_argument("--out")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_driver(args)


if __name__ == "__main__":
    main()
