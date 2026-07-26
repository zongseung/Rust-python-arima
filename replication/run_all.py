"""Regenerate every benchmark table in the paper from one entry point.

  python run_all.py            # full suite (several hours; needs ~24 GB host)
  python run_all.py --smoke    # reduced-scale wiring check (a few minutes)
  python run_all.py --only parallel,longseries

Stages:
  data        stage + validate the demand dataset
  parallel    fair parallel batch scaling (time + peak RSS)
  longseries  long-series / s in {24,168} scaling with DNF reporting
  auto        four-way auto-ARIMA + matched parallel grid search
  application extended-period demand application
  tables      render outputs/raw/*.csv -> outputs/tables/*.tex
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import PY, ensure_dirs, save_manifest  # noqa: E402

STAGES = [
    ("data", [PY, os.path.join(HERE, "data", "prepare.py")]),
    ("parallel", [PY, os.path.join(HERE, "benchmark", "parallel_scaling.py")]),
    ("auto", [PY, os.path.join(HERE, "accuracy", "auto_fourway.py")]),
    ("application", [PY, os.path.join(HERE, "application", "demand_extended.py")]),
    ("longseries", [PY, os.path.join(HERE, "benchmark", "longseries_scaling.py")]),
    ("tables", [PY, os.path.join(HERE, "gen_tables.py")]),
]
SMOKE_AWARE = {"parallel", "longseries", "auto", "application"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", help="comma-separated stage names")
    args = ap.parse_args()

    ensure_dirs()
    selected = set(args.only.split(",")) if args.only else None
    t_start = time.perf_counter()
    stage_status = {}

    for name, cmd in STAGES:
        if selected is not None and name not in selected:
            continue
        full_cmd = list(cmd)
        if args.smoke and name in SMOKE_AWARE:
            full_cmd.append("--smoke")
        print(f"\n===== stage: {name} =====", flush=True)
        t0 = time.perf_counter()
        rc = subprocess.call(full_cmd)
        dt = time.perf_counter() - t0
        stage_status[name] = {"returncode": rc, "seconds": round(dt, 1)}
        print(f"===== stage {name}: rc={rc} ({dt:,.1f}s) =====", flush=True)
        if rc != 0 and name == "data":
            sys.exit("data stage failed; aborting")

    save_manifest(
        {
            "smoke": args.smoke,
            "stages": stage_status,
            "total_seconds": round(time.perf_counter() - t_start, 1),
        }
    )
    failed = [n for n, s in stage_status.items() if s["returncode"] != 0]
    if failed:
        sys.exit(f"stages with non-zero exit: {failed}")


if __name__ == "__main__":
    main()
