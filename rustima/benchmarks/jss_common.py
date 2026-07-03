"""JSS §4 benchmark common utilities.

Provides:
  * load_power_3y(): 2021-01-01 ~ 2023-12-31 hourly Korean power demand
    with exog=[ta, hm]. n ≈ 26,304, s = 24.
  * run_with_oom_watchdog(): subprocess runner that SIGKILLs the whole
    process group when peak RSS exceeds OOM_RSS_BYTES (default 16 GB),
    or when wall-clock exceeds timeout_s. Reports peak RSS regardless.

Designed to be imported by all §4 scripts AND invoked directly as a
sanity check (`python -m benchmarks.jss_common`).
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import psutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "..", "power_demand_final.csv")

OOM_RSS_BYTES = 16 * 1024 ** 3
# macOS pages out heavily under memory pressure, so process RSS alone misses
# the true working-set. Cap the system-wide swap delta as a second tripwire,
# and the absolute swap usage as a third one (delta misses cases where the
# baseline is already high).
SWAP_DELTA_BYTES = 3 * 1024 ** 3
MAX_SWAP_USED_BYTES = 0  # 0 disables; callers can set e.g. int(6.5 * 1024**3)
SAMPLE_INTERVAL_S = 0.05
DEFAULT_TIMEOUT_S = 3600.0

WINDOW_START = "2021-01-01"
WINDOW_END = "2024-01-01"
SEASONAL_PERIOD = 24


def load_power_3y(
    path: str = DATA_PATH,
    start: str = WINDOW_START,
    end: str = WINDOW_END,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Load hourly Korean power demand with exogenous regressors.

    Returns
    -------
    y : (n,) float64 — power demand in MW
    exog : (n, 2) float64 — columns [ta, hm]
    dates : pd.DatetimeIndex of length n
    """
    df = pd.read_csv(path)
    df["일시"] = pd.to_datetime(df["일시"])
    mask = (df["일시"] >= start) & (df["일시"] < end)
    df = df.loc[mask].sort_values("일시").reset_index(drop=True)

    if df.empty:
        raise RuntimeError(f"no rows in window [{start}, {end})")

    y = df["power demand(MW)"].to_numpy(dtype=np.float64)
    exog = df[["ta", "hm"]].to_numpy(dtype=np.float64)

    if np.isnan(y).any():
        y = np.where(np.isnan(y), float(np.nanmean(y)), y)
    if np.isnan(exog).any():
        for j in range(exog.shape[1]):
            col = exog[:, j]
            exog[:, j] = np.where(np.isnan(col), float(np.nanmean(col)), col)

    return y, exog, df["일시"]


@dataclass
class RunResult:
    """Outcome of a watchdog-monitored subprocess run.

    status:
        'ok'           — exit 0
        'oom'          — killed because peak process RSS exceeded oom_bytes
        'oom_swap'     — killed because system swap delta exceeded swap_delta_bytes
        'oom_swap_abs' — killed because absolute system swap usage exceeded
                          max_swap_used_bytes
        'timeout'      — killed because wall-clock exceeded the threshold
        'error'        — exited with non-zero status (or pre-flight failure)
    """
    status: str
    wall_time_s: float
    peak_rss_gb: float
    peak_swap_delta_gb: float
    returncode: Optional[int]
    stdout: str
    stderr: str

    @property
    def killed(self) -> bool:
        return self.status in ("oom", "oom_swap", "oom_swap_abs", "timeout")


def _sum_rss(proc: psutil.Process) -> int:
    total = 0
    try:
        total += proc.memory_info().rss
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0
    try:
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return total


def _system_swap_used() -> int:
    try:
        return int(psutil.swap_memory().used)
    except Exception:
        return 0


def run_with_oom_watchdog(
    cmd: Sequence[str],
    oom_bytes: int = OOM_RSS_BYTES,
    swap_delta_bytes: int = SWAP_DELTA_BYTES,
    max_swap_used_bytes: int = MAX_SWAP_USED_BYTES,
    sample_interval_s: float = SAMPLE_INTERVAL_S,
    timeout_s: Optional[float] = DEFAULT_TIMEOUT_S,
    env: Optional[dict] = None,
    cwd: Optional[str] = None,
) -> RunResult:
    """Run `cmd` as a subprocess; SIGKILL the process group on OOM, swap
    blow-up, or timeout.

    Three memory tripwires:
      * peak process RSS (incl. all descendants) > oom_bytes
      * system swap usage rises by more than swap_delta_bytes above the
        baseline measured just before launch (catches macOS's case where
        the kernel pages out heavily and RSS undercounts the working set)
      * absolute system swap usage > max_swap_used_bytes (catches the case
        where the baseline is already high — delta would never trip)
    """
    swap_baseline = _system_swap_used()
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=cwd,
        start_new_session=True,
    )

    try:
        ps = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        out, err = proc.communicate()
        return RunResult(
            status="error",
            wall_time_s=time.perf_counter() - t0,
            peak_rss_gb=0.0,
            peak_swap_delta_gb=0.0,
            returncode=proc.returncode,
            stdout=out,
            stderr=err,
        )

    peak_rss = 0
    peak_swap_delta = 0
    kill_reason: Optional[str] = None

    while True:
        if proc.poll() is not None:
            break

        rss = _sum_rss(ps)
        if rss > peak_rss:
            peak_rss = rss

        swap_delta = max(0, _system_swap_used() - swap_baseline)
        if swap_delta > peak_swap_delta:
            peak_swap_delta = swap_delta

        if rss > oom_bytes:
            kill_reason = "oom"
            break
        if swap_delta > swap_delta_bytes:
            kill_reason = "oom_swap"
            break
        if max_swap_used_bytes > 0 and _system_swap_used() > max_swap_used_bytes:
            kill_reason = "oom_swap_abs"
            break
        if timeout_s is not None and (time.perf_counter() - t0) > timeout_s:
            kill_reason = "timeout"
            break

        time.sleep(sample_interval_s)

    if kill_reason is not None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            out, err = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = "", ""
        return RunResult(
            status=kill_reason,
            wall_time_s=time.perf_counter() - t0,
            peak_rss_gb=peak_rss / (1024 ** 3),
            peak_swap_delta_gb=peak_swap_delta / (1024 ** 3),
            returncode=proc.returncode,
            stdout=out or "",
            stderr=err or "",
        )

    out, err = proc.communicate()
    return RunResult(
        status="ok" if proc.returncode == 0 else "error",
        wall_time_s=time.perf_counter() - t0,
        peak_rss_gb=peak_rss / (1024 ** 3),
        peak_swap_delta_gb=peak_swap_delta / (1024 ** 3),
        returncode=proc.returncode,
        stdout=out,
        stderr=err,
    )


def _self_test() -> None:
    y, exog, dates = load_power_3y()
    print(f"[data] n={y.size:,}  exog.shape={exog.shape}  "
          f"range={dates.iloc[0]} ~ {dates.iloc[-1]}")
    print(f"[data] y nan={int(np.isnan(y).sum())}  "
          f"exog nan={int(np.isnan(exog).sum())}")
    print(f"[data] y mean={y.mean():,.1f} MW  std={y.std():,.1f} MW")
    expected = 3 * 365 * 24  # 26,280; leap day in 2024 not included
    print(f"[data] expected ≈ {expected:,}  actual = {y.size:,}")

    print("\n[watchdog] running 'python -c \"import time; time.sleep(0.5)\"' ...")
    r = run_with_oom_watchdog(
        ["python", "-c", "import time; time.sleep(0.5)"],
        timeout_s=10.0,
    )
    print(f"  status={r.status}  wall={r.wall_time_s:.2f}s  "
          f"peak_rss={r.peak_rss_gb*1024:.1f} MB  "
          f"peak_swap_delta={r.peak_swap_delta_gb*1024:.1f} MB  "
          f"rc={r.returncode}")


if __name__ == "__main__":
    _self_test()
