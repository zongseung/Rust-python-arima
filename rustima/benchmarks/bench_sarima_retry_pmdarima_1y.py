#!/usr/bin/env python3
"""
pmdarima 1y SARIMA(s=24) 단독 재시도 (swap 임계치 완화).
- Δswap ≥ 4096 MB 까지는 허용 (macOS는 동적 swap 확장).
- 성공/킬 결과를 sarima_scaling_results.csv 의 해당 행에 머지.
"""
import json
import os
import subprocess
import sys
import time

import pandas as pd
import psutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(PROJECT_ROOT, "benchmarks", "bench_sarima_worker.py")
CSV_PATH = os.path.join(PROJECT_ROOT, "..", "paper", "bench_2019_2023",
                        "sarima_scaling_results.csv")

SWAP_DELTA_KILL_MB = 4096       # relaxed (was 500)
HARD_TIMEOUT_S = 1800
POLL_S = 0.2
PYTHON = sys.executable


def _swap_used_mb():
    return psutil.swap_memory().used / (1024 ** 2)


def _rss_mb(pid):
    try:
        p = psutil.Process(pid)
        rss = p.memory_info().rss
        for ch in p.children(recursive=True):
            try: rss += ch.memory_info().rss
            except psutil.NoSuchProcess: pass
        return rss / (1024 ** 2)
    except psutil.NoSuchProcess:
        return 0.0


def main():
    baseline = _swap_used_mb()
    print(f"[retry] pmdarima 1y SARIMA(s=24)  "
          f"(swap baseline={baseline:,.0f} MB, kill if Δswap ≥ "
          f"{SWAP_DELTA_KILL_MB} MB)")

    proc = subprocess.Popen(
        [PYTHON, WORKER, "pmdarima", "1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    t0 = time.perf_counter()
    peak_rss = 0.0
    peak_swap = 0.0
    killed = False

    while proc.poll() is None:
        rss = _rss_mb(proc.pid)
        if rss > peak_rss: peak_rss = rss
        delta = max(0.0, _swap_used_mb() - baseline)
        if delta > peak_swap: peak_swap = delta
        elapsed = time.perf_counter() - t0

        if delta >= SWAP_DELTA_KILL_MB:
            print(f"  !! Δswap {delta:,.0f} MB ≥ {SWAP_DELTA_KILL_MB} → SIGKILL")
            killed = True
            try:
                p = psutil.Process(proc.pid)
                for ch in p.children(recursive=True):
                    try: ch.kill()
                    except psutil.NoSuchProcess: pass
                p.kill()
            except psutil.NoSuchProcess:
                pass
            break
        if elapsed > HARD_TIMEOUT_S:
            print(f"  !! timeout {HARD_TIMEOUT_S}s")
            killed = True
            try: psutil.Process(proc.pid).kill()
            except psutil.NoSuchProcess: pass
            break
        if int(elapsed) and int(elapsed) % 60 == 0:
            print(f"  .. t={elapsed:6.1f}s  RSS={rss:7.0f} MB  "
                  f"Δswap={delta:6.0f} MB")
            sys.stdout.flush()
            time.sleep(0.5)
        time.sleep(POLL_S)

    stdout, stderr = proc.communicate(timeout=5)
    elapsed = time.perf_counter() - t0

    row = {
        "engine": "pmdarima", "years": 1, "n_obs": 0,
        "time_s": elapsed, "peak_rss_mb": peak_rss,
        "peak_swap_delta_mb": peak_swap,
        "order": "", "seasonal": "", "aic": "", "n_models": "",
        "status": "KILLED_SWAP" if killed else "OK",
        "error": (f"swap delta ≥ {SWAP_DELTA_KILL_MB} MB" if killed else ""),
    }
    if not killed:
        for line in stdout.splitlines():
            if line.startswith("__RESULT__"):
                data = json.loads(line[len("__RESULT__"):])
                row["n_obs"] = data["n_obs"]
                row["time_s"] = data["time_s"]
                row["peak_rss_mb"] = max(peak_rss, data["peak_rss_mb"])
                row["order"] = str(tuple(data["order"]))
                row["seasonal"] = str(tuple(data["seasonal"]))
                row["aic"] = data["aic"]
                break

    print(f"\n[result] status={row['status']}  time={row['time_s']:.1f}s  "
          f"peak_RSS={row['peak_rss_mb']:.0f} MB  "
          f"peak_Δswap={row['peak_swap_delta_mb']:.0f} MB")

    # merge into CSV (replace pmdarima/years=1 row)
    df = pd.read_csv(CSV_PATH)
    mask = (df["engine"] == "pmdarima") & (df["years"] == 1)
    for k, v in row.items():
        df.loc[mask, k] = v
    df.to_csv(CSV_PATH, index=False)
    print(f"[csv] updated → {CSV_PATH}")


if __name__ == "__main__":
    main()
