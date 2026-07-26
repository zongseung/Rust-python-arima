"""Wiring check for the replication archive.

Runs the parallel and longseries stages at smoke scale end-to-end (real
subprocesses, real watchdog, all engines) and asserts the result CSVs have
the expected schema with at least one successful row per engine family.

  python test_smoke.py
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import PY, RAW_DIR  # noqa: E402


def main() -> None:
    rc = subprocess.call(
        [PY, os.path.join(HERE, "run_all.py"), "--smoke",
         "--only", "data,parallel,longseries"]
    )
    assert rc == 0, f"run_all --smoke failed with rc={rc}"

    import pandas as pd

    par = pd.read_csv(os.path.join(RAW_DIR, "parallel_scaling_smoke.csv"))
    for col in ("workload", "engine", "n_jobs", "status", "inner_min_s", "peak_rss_gb"):
        assert col in par.columns, f"parallel csv missing {col}"
    for engine in ("rustima", "statsmodels_seq", "statsmodels_joblib", "statsforecast"):
        ok = par[(par["engine"] == engine) & (par["status"] == "ok")]
        assert len(ok) > 0, f"no successful {engine} rows:\n{par}"
        assert ok["inner_min_s"].notna().all(), f"{engine} missing timings"
        assert (ok["peak_rss_gb"] > 0).all(), f"{engine} missing RSS"

    lng = pd.read_csv(os.path.join(RAW_DIR, "longseries_scaling_smoke.csv"))
    for engine in ("rustima", "statsmodels"):
        ok = lng[(lng["engine"] == engine) & (lng["status"] == "ok")]
        assert len(ok) > 0, f"no successful longseries {engine} rows:\n{lng}"

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
