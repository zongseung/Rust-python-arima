#!/usr/bin/env python3
"""
SARIMA(s=24) 단일 측정 워커 — exog 없음, pmdarima 또는 rustima 하나만 실행.

부모 프로세스가 swap 진입 시 SIGKILL 할 수 있도록 자식 프로세스에서 실행된다.

사용:
    python bench_sarima_worker.py <engine> <years>
        engine: 'rustima' | 'pmdarima'
        years : 1~5

stdout 마지막 줄에 JSON 한 줄을 출력:
    {"engine":..., "years":..., "n_obs":..., "time_s":..., "peak_rss_mb":...,
     "order":[p,d,q], "seasonal":[P,D,Q,s], "aic":..., "n_models":...}
"""
import gc
import json
import os
import sys
import threading
import time
import warnings

import numpy as np
import pandas as pd
import psutil

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "..", "power_demand_final.csv")


class RSSMonitor(threading.Thread):
    def __init__(self, interval=0.1):
        super().__init__(daemon=True)
        self.interval = interval
        self.proc = psutil.Process(os.getpid())
        self.peak = self.proc.memory_info().rss
        self.baseline = self.peak
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            rss = self.proc.memory_info().rss
            if rss > self.peak:
                self.peak = rss
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        self.join()


def load_series(years: int):
    df = pd.read_csv(DATA_PATH)
    df["일시"] = pd.to_datetime(df["일시"])
    df = df[(df["일시"] >= "2019-01-01") & (df["일시"] < "2024-01-01")]
    df = df.sort_values("일시").reset_index(drop=True)
    y = df["power demand(MW)"].values.astype(np.float64)
    n = min(years * 24 * 365, len(y))
    y = y[:n]
    if np.isnan(y).any():
        y = np.where(np.isnan(y), np.nanmean(y), y)
    return y


def run_rustima(y, s=24):
    from rustima.auto import auto_arima
    res = auto_arima(
        y, s=s,
        max_p=3, max_q=3, max_P=1, max_Q=1, max_d=1, max_D=1,
        stepwise=True, criterion="aic",
    )
    return {
        "order": list(res.order),
        "seasonal": list(res.seasonal_order),
        "aic": float(res.best_ic),
        "n_models": len(res.history),
    }


def run_pmdarima(y, s=24):
    import pmdarima as pm
    res = pm.auto_arima(
        y, m=s,
        start_p=0, start_q=0, start_P=0, start_Q=0,
        max_p=3, max_q=3, max_P=1, max_Q=1, max_d=1, max_D=1,
        stepwise=True, information_criterion="aic",
        error_action="ignore", suppress_warnings=True, trace=False,
    )
    return {
        "order": list(res.order),
        "seasonal": list(res.seasonal_order),
        "aic": float(res.aic()),
        "n_models": None,
    }


def main():
    if len(sys.argv) != 3:
        print("usage: bench_sarima_worker.py <rustima|pmdarima> <years>",
              file=sys.stderr)
        sys.exit(2)
    engine = sys.argv[1]
    years = int(sys.argv[2])

    y = load_series(years)
    gc.collect()
    mon = RSSMonitor(interval=0.1)
    mon.start()
    t0 = time.perf_counter()
    if engine == "rustima":
        fit = run_rustima(y)
    elif engine == "pmdarima":
        fit = run_pmdarima(y)
    else:
        print(f"unknown engine: {engine}", file=sys.stderr)
        sys.exit(2)
    elapsed = time.perf_counter() - t0
    mon.stop()

    out = {
        "engine": engine,
        "years": years,
        "n_obs": int(len(y)),
        "time_s": elapsed,
        "peak_rss_mb": mon.peak / (1024 ** 2),
        **fit,
    }
    # JSON on the LAST line so the parent can parse it reliably
    print("__RESULT__" + json.dumps(out))


if __name__ == "__main__":
    main()
