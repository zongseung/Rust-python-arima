#!/usr/bin/env python3
"""
Parent: orchestrates the SARIMA/SARIMAX × {fit_method} × pmdarima matrix
across years. Monitors swap; SIGKILL pmdarima if swap delta ≥ threshold.

Each (engine, mode, fit_method, years) runs as a subprocess via
bench_matrix_worker.py. Outputs:
    paper/bench_matrix_results.csv
"""
import os, sys, json, time, subprocess, threading
from dataclasses import dataclass, asdict
from typing import Optional
import pandas as pd, psutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(PROJECT_ROOT, "python_tests", "bench_matrix_worker.py")
OUTDIR = os.path.join(PROJECT_ROOT, "..", "paper")
os.makedirs(OUTDIR, exist_ok=True)
CSV_PATH = os.path.join(OUTDIR, "bench_matrix_results.csv")

PYTHON = sys.executable
SWAP_KILL_MB = 4096           # SIGKILL if pmdarima Δswap >= this
HARD_TIMEOUT_S = 1800
POLL_S = 0.2

# Test plan
YEARS_LIST = [1]  # focus on 1y first; extend later if needed
MODES = ["sarima", "sarimax"]
RUSTIMA_METHODS = ["lbfgsb", "lbfgsb-multi", "lbfgsb-adaptive", "lbfgsb-hybrid", "trust-region"]
PMDARIMA_METHODS = ["default"]  # pmdarima has no method choice


@dataclass
class Run:
    engine: str
    mode: str
    fit_method: str
    years: int
    n_obs: int = 0
    time_s: float = 0.0
    peak_rss_mb: float = 0.0
    peak_swap_delta_mb: float = 0.0
    order: Optional[str] = None
    seasonal: Optional[str] = None
    aic: Optional[float] = None
    n_models: Optional[int] = None
    status: str = "OK"
    error: Optional[str] = None


def _swap_used_mb(): return psutil.swap_memory().used / (1024 ** 2)
def _rss_mb(pid):
    try:
        p = psutil.Process(pid); rss = p.memory_info().rss
        for ch in p.children(recursive=True):
            try: rss += ch.memory_info().rss
            except psutil.NoSuchProcess: pass
        return rss / (1024 ** 2)
    except psutil.NoSuchProcess: return 0.0


def run_combo(engine, mode, fit_method, years) -> Run:
    swap_baseline = _swap_used_mb()
    label = f"{engine:>8s}/{mode:>7s}/{fit_method:>16s}/y={years}"
    print(f"\n  [{label}]  swap_base={swap_baseline:.0f}MB  (kill if Δ≥{SWAP_KILL_MB}MB)")
    sys.stdout.flush()

    proc = subprocess.Popen(
        [PYTHON, WORKER, engine, mode, fit_method, str(years)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    t0 = time.perf_counter()
    peak_rss = 0.0
    peak_swap = 0.0
    killed = False
    timed_out = False
    last_heartbeat = 0

    while True:
        if proc.poll() is not None: break
        rss = _rss_mb(proc.pid)
        if rss > peak_rss: peak_rss = rss
        d_swap = max(0.0, _swap_used_mb() - swap_baseline)
        if d_swap > peak_swap: peak_swap = d_swap
        elapsed = time.perf_counter() - t0

        if d_swap >= SWAP_KILL_MB:
            print(f"    !! Δswap {d_swap:.0f}MB ≥ {SWAP_KILL_MB} → SIGKILL")
            try:
                p = psutil.Process(proc.pid)
                for ch in p.children(recursive=True):
                    try: ch.kill()
                    except psutil.NoSuchProcess: pass
                p.kill()
            except psutil.NoSuchProcess: pass
            killed = True
            break
        if elapsed > HARD_TIMEOUT_S:
            print(f"    !! timeout {HARD_TIMEOUT_S}s")
            try: psutil.Process(proc.pid).kill()
            except psutil.NoSuchProcess: pass
            timed_out = True; break
        # heartbeat every 60s
        cur_min = int(elapsed) // 60
        if cur_min > last_heartbeat and int(elapsed) % 60 < 1:
            last_heartbeat = cur_min
            print(f"    .. t={elapsed:6.1f}s RSS={rss:6.0f}MB Δswap={d_swap:.0f}MB")
            sys.stdout.flush()
            time.sleep(0.5)
        time.sleep(POLL_S)

    stdout, stderr = proc.communicate(timeout=5)
    elapsed = time.perf_counter() - t0
    run = Run(engine=engine, mode=mode, fit_method=fit_method, years=years,
              time_s=elapsed, peak_rss_mb=peak_rss, peak_swap_delta_mb=peak_swap)
    if killed:
        run.status = "KILLED_SWAP"; run.error = f"Δswap ≥ {SWAP_KILL_MB}MB"
        print(f"    → KILLED_SWAP at {elapsed:.1f}s (peak RSS {peak_rss:.0f}MB)")
        return run
    if timed_out:
        run.status = "TIMEOUT"; run.error = f"timeout {HARD_TIMEOUT_S}s"
        return run

    rl = None
    for line in stdout.splitlines():
        if line.startswith("__RESULT__"):
            rl = line[len("__RESULT__"):]; break
    if rl is None:
        run.status = "ERROR"; run.error = (stderr or stdout)[-300:]
        print(f"    → ERROR (no result line)\n    stderr: {(stderr or '')[:200]}")
        return run
    try:
        d = json.loads(rl)
        run.n_obs = d["n_obs"]; run.time_s = d["time_s"]
        run.peak_rss_mb = max(run.peak_rss_mb, d["peak_rss_mb"])
        run.order = str(tuple(d["order"])); run.seasonal = str(tuple(d["seasonal"]))
        run.aic = d["aic"]; run.n_models = d.get("n_models")
    except Exception as e:
        run.status = "ERROR"; run.error = f"json: {e}"
        return run

    print(f"    ✓ AIC={run.aic:.1f}  order={run.order}{run.seasonal}  "
          f"t={run.time_s:.1f}s  RSS={run.peak_rss_mb:.0f}MB"
          + (f"  n_models={run.n_models}" if run.n_models else ""))
    return run


def main():
    print("=" * 80)
    print("Power demand SARIMA/SARIMAX × fit_method × engine matrix")
    print(f"swap kill: Δ≥{SWAP_KILL_MB}MB,  years={YEARS_LIST}")
    print("=" * 80)

    runs: list[Run] = []
    pmdarima_killed = {"sarima": False, "sarimax": False}

    for years in YEARS_LIST:
        for mode in MODES:
            # rustima with all 4 methods
            for method in RUSTIMA_METHODS:
                runs.append(run_combo("rustima", mode, method, years))
            # pmdarima (skip if already killed for this mode in a previous year)
            if not pmdarima_killed[mode]:
                r = run_combo("pmdarima", mode, "default", years)
                runs.append(r)
                if r.status == "KILLED_SWAP":
                    pmdarima_killed[mode] = True
                    print(f"  → pmdarima/{mode} killed at y={years}; "
                          f"skipping pmdarima for y>{years} in mode={mode}")
            else:
                runs.append(Run(engine="pmdarima", mode=mode, fit_method="default",
                                years=years, status="KILLED_SWAP",
                                error=f"skipped: pmdarima/{mode} killed in earlier year"))

    # save
    rows = [asdict(r) for r in runs]
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
    print(f"\n[CSV]  {CSV_PATH}")

    # quick summary
    print("\n=== Summary ===")
    df = pd.DataFrame(rows)
    df["aic_fmt"] = df["aic"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
    df["t_fmt"] = df["time_s"].apply(lambda x: f"{x:.1f}s")
    print(df[["engine", "mode", "fit_method", "years", "order", "seasonal",
              "aic_fmt", "t_fmt", "peak_rss_mb", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
