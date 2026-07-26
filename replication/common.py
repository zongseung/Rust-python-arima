"""Shared utilities for the JSS replication archive.

Reuses the measured-subprocess harness (wall time + peak RSS incl.
descendants + swap tripwires + timeout) from rustima/benchmarks/jss_common.py
rather than duplicating it. All experiment drivers launch each measured
condition in a fresh subprocess with BLAS/OpenMP/Numba threading pinned to 1
so that only the task-level parallelism under study varies.
"""
from __future__ import annotations

import json
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
RUSTIMA_DIR = os.path.join(REPO_ROOT, "rustima")
BENCH_DIR = os.path.join(RUSTIMA_DIR, "benchmarks")
JSS_RUNNERS_DIR = os.path.join(BENCH_DIR, "jss_runners")

sys.path.insert(0, BENCH_DIR)
from jss_common import (  # noqa: E402,F401
    RunResult,
    load_power_3y,
    run_with_oom_watchdog,
)

PY = sys.executable

OUT_DIR = os.path.join(HERE, "outputs")
RAW_DIR = os.path.join(OUT_DIR, "raw")
TABLES_DIR = os.path.join(OUT_DIR, "tables")
FIGURES_DIR = os.path.join(OUT_DIR, "figures")

DATA_CSV_LOCAL = os.path.join(HERE, "data", "power_demand_final.csv")
DATA_CSV_ROOT = os.path.join(REPO_ROOT, "power_demand_final.csv")


def data_csv_path() -> str:
    return DATA_CSV_LOCAL if os.path.exists(DATA_CSV_LOCAL) else DATA_CSV_ROOT


def ensure_dirs() -> None:
    for d in (OUT_DIR, RAW_DIR, TABLES_DIR, FIGURES_DIR):
        os.makedirs(d, exist_ok=True)


def worker_env(rayon_threads: int | None = None, **extra: str) -> dict:
    """Environment for a measured worker: intra-op threading pinned to 1.

    Only the task-level parallelism under study (Rayon threads, joblib
    processes, StatsForecast n_jobs) is allowed to vary between conditions.
    """
    env = os.environ.copy()
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        env[var] = "1"
    if rayon_threads is not None:
        env["RAYON_NUM_THREADS"] = str(rayon_threads)
    env.update(extra)
    return env


def write_rows_csv(rows: list[dict], path: str) -> None:
    import pandas as pd

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[out] {path}")


def environment_manifest() -> dict:
    import numpy
    import pandas

    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
    }
    for mod in ("rustima", "statsmodels", "pmdarima", "statsforecast", "joblib"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            info[mod] = "not installed"
    return info


def save_manifest(extra: dict | None = None) -> None:
    ensure_dirs()
    manifest = environment_manifest()
    if extra:
        manifest.update(extra)
    path = os.path.join(OUT_DIR, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[out] {path}")
