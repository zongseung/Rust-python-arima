#!/usr/bin/env python3
"""
SARIMA(s=24) 스케일링 비교: rustima vs pmdarima (exog 없음).
==================================================================
이미지의 SARIMAX(exog=ta+hm) 비교를 SARIMA(no exog)로 재현한다.

핵심 안전장치:
    pmdarima 자식 프로세스의 swap 사용량이 500MB(델타) 이상 증가하면
    즉시 SIGKILL → 'KILLED' 로 기록. 부모/시스템 보호.

실행:
    cd rustima/
    .venv/bin/python benchmarks/bench_sarima_scaling.py
"""
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import pandas as pd
import psutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(PROJECT_ROOT, "benchmarks", "bench_sarima_worker_exog.py")
OUTDIR = os.path.join(PROJECT_ROOT, "..", "paper", "bench_2019_2023")
os.makedirs(OUTDIR, exist_ok=True)

# Hard guards (보수적 — 시스템에 이미 swap 사용 중)
SWAP_DELTA_KILL_MB = 300          # swap delta vs baseline ≥ this → SIGKILL (more aggressive)
HARD_TIMEOUT_S_RUSTIMA = 1800     # 30 min
HARD_TIMEOUT_S_PMDARIMA = 1800    # 30 min
POLL_INTERVAL_S = 0.2             # swap/RSS 폴링 간격

PYTHON = sys.executable


@dataclass
class Run:
    engine: str
    years: int
    n_obs: int = 0
    time_s: float = 0.0
    peak_rss_mb: float = 0.0
    peak_swap_delta_mb: float = 0.0
    order: Optional[list] = None
    seasonal: Optional[list] = None
    aic: Optional[float] = None
    n_models: Optional[int] = None
    status: str = "OK"           # OK | KILLED_SWAP | TIMEOUT | ERROR
    error: Optional[str] = None


def _read_swap_used_mb() -> float:
    return psutil.swap_memory().used / (1024 ** 2)


def _try_rss_mb(pid: int) -> float:
    try:
        p = psutil.Process(pid)
        # include children if any
        rss = p.memory_info().rss
        for ch in p.children(recursive=True):
            try:
                rss += ch.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return rss / (1024 ** 2)
    except psutil.NoSuchProcess:
        return 0.0


def run_worker(engine: str, years: int) -> Run:
    """Spawn worker subprocess; monitor swap; SIGKILL if swap delta crosses limit."""
    swap_baseline_mb = _read_swap_used_mb()

    print(f"\n  [{engine} y={years}] swap baseline = {swap_baseline_mb:,.0f} MB"
          f"  (kill if Δswap ≥ {SWAP_DELTA_KILL_MB} MB)")
    sys.stdout.flush()

    proc = subprocess.Popen(
        [PYTHON, WORKER, engine, str(years)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    t0 = time.perf_counter()
    peak_rss = 0.0
    peak_swap_delta = 0.0
    killed = False
    timed_out = False
    timeout_s = HARD_TIMEOUT_S_PMDARIMA if engine == "pmdarima" else HARD_TIMEOUT_S_RUSTIMA

    while True:
        if proc.poll() is not None:
            break
        elapsed = time.perf_counter() - t0
        rss = _try_rss_mb(proc.pid)
        if rss > peak_rss:
            peak_rss = rss
        cur_swap_delta = max(0.0, _read_swap_used_mb() - swap_baseline_mb)
        if cur_swap_delta > peak_swap_delta:
            peak_swap_delta = cur_swap_delta

        # SWAP KILL GUARD ─────────────────────────────────────────────
        if cur_swap_delta >= SWAP_DELTA_KILL_MB:
            print(f"    !! swap delta {cur_swap_delta:,.0f} MB ≥ "
                  f"{SWAP_DELTA_KILL_MB} MB → SIGKILL (RSS={rss:,.0f} MB, "
                  f"t={elapsed:.1f}s)")
            sys.stdout.flush()
            killed = True
            try:
                # kill whole process group / tree
                parent = psutil.Process(proc.pid)
                for ch in parent.children(recursive=True):
                    try: ch.kill()
                    except psutil.NoSuchProcess: pass
                parent.kill()
            except psutil.NoSuchProcess:
                pass
            break

        # Timeout guard
        if elapsed > timeout_s:
            print(f"    !! timeout {timeout_s}s → SIGKILL")
            sys.stdout.flush()
            timed_out = True
            try: psutil.Process(proc.pid).kill()
            except psutil.NoSuchProcess: pass
            break

        # periodic heartbeat
        if int(elapsed) and int(elapsed) % 30 == 0:
            print(f"    .. t={elapsed:6.1f}s  RSS={rss:7.0f} MB  "
                  f"Δswap={cur_swap_delta:6.0f} MB")
            sys.stdout.flush()
            time.sleep(0.5)  # avoid duplicate heartbeats within the same second

        time.sleep(POLL_INTERVAL_S)

    stdout, stderr = proc.communicate(timeout=5)
    elapsed = time.perf_counter() - t0

    run = Run(engine=engine, years=years, time_s=elapsed,
              peak_rss_mb=peak_rss, peak_swap_delta_mb=peak_swap_delta)

    if killed:
        run.status = "KILLED_SWAP"
        run.error = f"swap delta ≥ {SWAP_DELTA_KILL_MB} MB"
        print(f"    → KILLED_SWAP after {elapsed:.1f}s "
              f"(peak RSS={peak_rss:,.0f} MB, peak Δswap={peak_swap_delta:,.0f} MB)")
        return run
    if timed_out:
        run.status = "TIMEOUT"
        run.error = f"timeout {timeout_s}s"
        return run

    # parse JSON result
    result_line = None
    for line in stdout.splitlines():
        if line.startswith("__RESULT__"):
            result_line = line[len("__RESULT__"):]
            break
    if result_line is None:
        run.status = "ERROR"
        run.error = (stderr or stdout)[-300:]
        return run
    try:
        data = json.loads(result_line)
        run.n_obs = data["n_obs"]
        run.time_s = data["time_s"]           # worker가 측정한 순수 시간 사용
        run.peak_rss_mb = max(run.peak_rss_mb, data["peak_rss_mb"])
        run.order = data["order"]
        run.seasonal = data["seasonal"]
        run.aic = data["aic"]
        run.n_models = data.get("n_models")
    except Exception as e:
        run.status = "ERROR"
        run.error = f"json parse: {e}; raw: {result_line[:200]}"
        return run

    print(f"    ✓ {run.time_s:.1f}s  RSS={run.peak_rss_mb:.0f} MB  "
          f"order={tuple(run.order)} seasonal={tuple(run.seasonal)}  "
          f"AIC={run.aic:.1f}")
    return run


def main():
    print("=" * 78)
    print("SARIMA(s=24) 스케일링 비교: rustima vs pmdarima  [exog 없음]")
    print(f"swap kill threshold: Δswap ≥ {SWAP_DELTA_KILL_MB} MB")
    print("=" * 78)

    all_runs: list[Run] = []
    for years in (1, 2, 3):
        print(f"\n── years = {years} ──────────────────────────────────────────")
        # rustima 먼저
        r = run_worker("rustima", years)
        all_runs.append(r)
        # 그 다음 pmdarima
        p = run_worker("pmdarima", years)
        all_runs.append(p)
        # 스왑 시 다음 연도도 더 큰 데이터라 어차피 못 끝남 → 빨리 종료
        if p.status == "KILLED_SWAP":
            print(f"\n  pmdarima y={years}에서 swap 트리거. "
                  f"다음 연도(y={years+1})는 더 큰 데이터 → pmdarima 스킵 결정.")
            # rustima는 계속 측정해야 함 — 그래프에서 5y까지 라인 필요
            # 단, pmdarima 추가 시도는 막는다.
            remaining = [y for y in (1, 2, 3) if y > years]
            for ry in remaining:
                r2 = run_worker("rustima", ry)
                all_runs.append(r2)
                # pmdarima는 KILLED_SWAP 더미로 기록
                all_runs.append(Run(
                    engine="pmdarima", years=ry,
                    status="KILLED_SWAP",
                    error=f"skipped: y={years} already triggered swap",
                ))
            break

    # CSV
    rows = []
    for r in all_runs:
        d = asdict(r)
        if d.get("order") is not None:
            d["order"] = str(tuple(d["order"]))
        if d.get("seasonal") is not None:
            d["seasonal"] = str(tuple(d["seasonal"]))
        rows.append(d)
    csv_path = os.path.join(OUTDIR, "sarima_scaling_results_exog.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n[CSV]  {csv_path}")

    # plot
    try:
        from bench_sarima_plot import save_plot
        plot_path = os.path.join(OUTDIR, "sarima_scaling_with_pmdarima.png")
        save_plot(all_runs, plot_path)
        print(f"[PNG]  {plot_path}")
    except Exception as e:
        print(f"[plot] 실패: {e}  — CSV는 저장됨, 별도 실행: bench_sarima_plot.py")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "benchmarks"))
    main()
