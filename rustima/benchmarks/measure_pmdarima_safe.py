"""Measure pmdarima auto_arima at 1 and 2 years on the 2021-anchored window,
with a strict swap watchdog that SIGKILLs the worker (and its process group)
before the host can thrash. Reuses bench_sarima_worker_exog.py as the worker.

Stop conditions (any one triggers SIGKILL, recorded as KILLED_SWAP):
  * swap.used rises >= KILL_DELTA_MB above the per-run baseline
  * swap.used reaches KILL_ABS_MB absolute
  * RAM available drops below MIN_AVAIL_MB
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "bench_sarima_worker_exog.py")

KILL_DELTA_MB = 600       # swap growth vs baseline -> kill
KILL_ABS_MB = 6800        # absolute swap.used -> kill (7GB total)
MIN_AVAIL_MB = 400        # RAM available floor -> kill
POLL_S = 0.15
TIMEOUT_S = 1500.0


def mb(x):
    return x / (1024 ** 2)


def run_one(years: int):
    base_swap = psutil.swap_memory().used
    print(f"\n=== pmdarima years={years} | baseline swap={mb(base_swap):,.0f}MB ===",
          flush=True)
    proc = subprocess.Popen(
        ["uv", "run", "python", WORKER, "pmdarima", str(years)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, start_new_session=True,
    )
    peak_rss = 0.0
    peak_swap_delta = 0.0
    t0 = time.perf_counter()
    killed_reason = None
    try:
        p = psutil.Process(proc.pid)
        while proc.poll() is None:
            # peak RSS (worker + children)
            try:
                rss = p.memory_info().rss
                for ch in p.children(recursive=True):
                    try:
                        rss += ch.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak_rss = max(peak_rss, rss)
            except psutil.NoSuchProcess:
                break
            sw = psutil.swap_memory().used
            vm_avail = psutil.virtual_memory().available
            delta = sw - base_swap
            peak_swap_delta = max(peak_swap_delta, delta)
            if delta >= KILL_DELTA_MB * 1024 ** 2:
                killed_reason = f"swap delta {mb(delta):,.0f}MB >= {KILL_DELTA_MB}MB"
            elif sw >= KILL_ABS_MB * 1024 ** 2:
                killed_reason = f"swap abs {mb(sw):,.0f}MB >= {KILL_ABS_MB}MB"
            elif vm_avail <= MIN_AVAIL_MB * 1024 ** 2:
                killed_reason = f"RAM avail {mb(vm_avail):,.0f}MB <= {MIN_AVAIL_MB}MB"
            if killed_reason or (time.perf_counter() - t0) > TIMEOUT_S:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                if not killed_reason:
                    killed_reason = "timeout"
                break
            time.sleep(POLL_S)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    elapsed = time.perf_counter() - t0

    if killed_reason:
        print(f"  KILLED -> {killed_reason} | peak RSS={mb(peak_rss):,.0f}MB "
              f"peak swapΔ={mb(peak_swap_delta):,.0f}MB elapsed={elapsed:.0f}s",
              flush=True)
        return {"years": years, "status": "KILLED_SWAP",
                "peak_rss_mb": mb(peak_rss), "peak_swap_delta_mb": mb(peak_swap_delta),
                "time_s": elapsed}

    out, _ = proc.communicate()
    res = None
    for line in (out or "").splitlines():
        if line.startswith("__RESULT__"):
            res = json.loads(line[len("__RESULT__"):])
    if res is None:
        print(f"  ERROR: no __RESULT__ (peak RSS={mb(peak_rss):,.0f}MB)", flush=True)
        return {"years": years, "status": "ERROR", "peak_rss_mb": mb(peak_rss)}
    res["status"] = "OK"
    res["peak_rss_mb"] = max(res.get("peak_rss_mb", 0), mb(peak_rss))
    res["peak_swap_delta_mb"] = mb(peak_swap_delta)
    print(f"  OK -> time={res['time_s']:.0f}s RSS={res['peak_rss_mb']:,.0f}MB "
          f"order={tuple(res['order'])}{tuple(res['seasonal'])} AIC={res['aic']:.1f}",
          flush=True)
    return res


def main():
    results = []
    for years in (1, 2):
        r = run_one(years)
        results.append(r)
        if r["status"] != "OK":
            print(f"\n  years={years} did not complete; skipping larger windows.",
                  flush=True)
            break
    print("\n__SUMMARY__" + json.dumps(results))


if __name__ == "__main__":
    main()
