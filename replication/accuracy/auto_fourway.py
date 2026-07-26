"""Four-way automatic order selection on hourly Korean electricity demand.

Part A — stepwise search, 3-year window (2021--2023, n=26,280, exog=[ta,hm]):
  rustima, pmdarima, StatsForecast AutoARIMA, R forecast::auto.arima,
  each in a watchdog-measured subprocess (wall time, peak RSS, OOM/timeout).
  Selected orders are additionally re-fit under a single engine (rustima) so
  the information criteria are compared on one likelihood implementation.

Part B — matched parallel grid search, 1-year window, d=0, D=1, 64 models:
  rustima auto_arima(stepwise=False) on 8 Rayon threads vs
  pmdarima auto_arima(stepwise=False, n_jobs=8) process pool.

Usage:
  python auto_fourway.py [--smoke] [--skip-full] [--skip-grid]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import (  # noqa: E402
    JSS_RUNNERS_DIR,
    PY,
    RAW_DIR,
    ensure_dirs,
    load_power_3y,
    run_with_oom_watchdog,
    worker_env,
    write_rows_csv,
)

AUTO_PY = os.path.join(JSS_RUNNERS_DIR, "auto_py.py")
AUTO_R = os.path.join(JSS_RUNNERS_DIR, "auto_r.R")
SF_WORKER = os.path.join(HERE, "sf_auto_worker.py")
GRID_WORKER = os.path.join(HERE, "grid_auto_worker.py")


def _run_engine(name, cmd, timeout_s, rayon_threads=1):
    fd, out_path = tempfile.mkstemp(suffix=".json", prefix="auto_")
    os.close(fd)
    r = run_with_oom_watchdog(
        cmd + ["--out", out_path],
        timeout_s=timeout_s,
        env=worker_env(rayon_threads=rayon_threads),
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
        "engine": name,
        "status": r.status,
        "wall_time_s": round(r.wall_time_s, 2),
        "peak_rss_gb": round(r.peak_rss_gb, 3),
        "peak_swap_delta_gb": round(r.peak_swap_delta_gb, 3),
    }
    if data:
        row.update(
            {
                "order": str(tuple(data["order"])),
                "seasonal_order": str(tuple(data["seasonal_order"])),
                "aic_native": data["aic"],
                "runtime_inner_s": round(data["runtime_inner_s"], 2),
                "_order_raw": data["order"],
                "_sorder_raw": data["seasonal_order"],
            }
        )
    else:
        row["stderr_tail"] = (r.stderr or "")[-800:]
    print(f"  -> {row['status']} wall={row['wall_time_s']}s rss={row['peak_rss_gb']}GB", flush=True)
    return row


def _refit_aic_rustima(rows, years):
    """Re-fit every engine's selected order under rustima on the same window."""
    from rustima import SARIMAXModel

    end_year = 2021 + years
    y, exog, _ = load_power_3y(start="2021-01-01", end=f"{end_year}-01-01")
    for row in rows:
        order = row.pop("_order_raw", None)
        sorder = row.pop("_sorder_raw", None)
        if order is None:
            row["aic_refit_rustima"] = None
            continue
        try:
            res = SARIMAXModel(
                y, order=tuple(order), seasonal_order=tuple(sorder), exog=exog
            ).fit()
            row["aic_refit_rustima"] = round(float(res.aic), 3)
        except Exception as e:
            row["aic_refit_rustima"] = f"error: {e}"


def run_full(args):
    years = 1 if args.smoke else 3
    timeout = 600.0 if args.smoke else 7200.0
    engines = [
        ("rustima", [PY, AUTO_PY, "--engine", "rustima", "--years", str(years)], 8),
        ("pmdarima", [PY, AUTO_PY, "--engine", "pmdarima", "--years", str(years)], 1),
        ("statsforecast", [PY, SF_WORKER, "--years", str(years)], 1),
        ("r_forecast", ["Rscript", AUTO_R], 1),
    ]
    if args.smoke:
        # auto_r.R hardcodes the 3-year window; skip R in smoke mode.
        engines = [e for e in engines if e[0] != "r_forecast"]

    rows = []
    for name, cmd, rayon in engines:
        print(f"[full] engine={name}", flush=True)
        rows.append(_run_engine(name, cmd, timeout, rayon_threads=rayon))

    _refit_aic_rustima(rows, years)
    suffix = "_smoke" if args.smoke else ""
    write_rows_csv(rows, os.path.join(RAW_DIR, f"auto_fourway{suffix}.csv"))


def run_grid(args):
    timeout = 900.0 if args.smoke else 7200.0
    rows = []
    for name, cmd, rayon in [
        ("rustima_grid8", [PY, GRID_WORKER, "--engine", "rustima"], 8),
        ("pmdarima_grid8", [PY, GRID_WORKER, "--engine", "pmdarima", "--n-jobs", "8"], 1),
    ]:
        print(f"[grid] engine={name}", flush=True)
        rows.append(_run_engine(name, cmd, timeout, rayon_threads=rayon))
    for row in rows:
        row.pop("_order_raw", None)
        row.pop("_sorder_raw", None)
    suffix = "_smoke" if args.smoke else ""
    write_rows_csv(rows, os.path.join(RAW_DIR, f"auto_grid{suffix}.csv"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--skip-grid", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    if not args.skip_full:
        run_full(args)
    if not args.skip_grid:
        run_grid(args)


if __name__ == "__main__":
    main()
