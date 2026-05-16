#!/usr/bin/env python3
"""
Worker: runs ONE (engine, mode, fit_method) combo on power demand and emits JSON.

Args:
    engine     : rustima | pmdarima
    mode       : sarima  | sarimax     (sarima = no exog)
    fit_method : (rustima) lbfgsb | lbfgsb-multi | lbfgsb-adaptive | lbfgsb-hybrid
                 (pmdarima) ignored — uses pmdarima default
    years      : 1..5  (data length)

Output (last stdout line):
    __RESULT__<json>
"""
import os, sys, json, time, warnings, gc, threading
import numpy as np, pandas as pd
import psutil
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJECT_ROOT, "..", "power_demand_final.csv")


class RSSMonitor(threading.Thread):
    def __init__(self, interval=0.1):
        super().__init__(daemon=True)
        self.interval = interval
        self.proc = psutil.Process(os.getpid())
        self.peak = self.proc.memory_info().rss
        self._stop = threading.Event()
    def run(self):
        while not self._stop.is_set():
            rss = self.proc.memory_info().rss
            if rss > self.peak: self.peak = rss
            self._stop.wait(self.interval)
    def stop(self):
        self._stop.set(); self.join()


def load_data(years):
    df = pd.read_csv(DATA)
    df["일시"] = pd.to_datetime(df["일시"])
    df = df[(df["일시"] >= "2019-01-01") & (df["일시"] < "2024-01-01")].sort_values("일시").reset_index(drop=True)
    y = df["power demand(MW)"].values.astype(np.float64)
    ex = df[["ta", "hm"]].values.astype(np.float64)
    n = min(years * 24 * 365, len(y))
    y = y[:n]; ex = ex[:n]
    if np.isnan(y).any(): y = np.where(np.isnan(y), np.nanmean(y), y)
    for j in range(ex.shape[1]):
        c = ex[:, j]; ex[:, j] = np.where(np.isnan(c), np.nanmean(c), c)
    return y, ex


def fit_rustima(y, exog, fit_method):
    from rustima.auto import auto_arima
    res = auto_arima(
        y, exog=exog, s=24,
        max_p=3, max_q=3, max_P=1, max_Q=1, max_d=1, max_D=1,
        stepwise=True, criterion="aic", method=fit_method,
    )
    return {
        "order": list(res.order),
        "seasonal": list(res.seasonal_order),
        "aic": float(res.best_ic),
        "n_models": len(res.history),
    }


def fit_pmdarima(y, exog):
    import pmdarima as pm
    res = pm.auto_arima(
        y, X=exog, m=24,
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
    if len(sys.argv) != 5:
        print("usage: bench_matrix_worker.py <engine> <mode> <fit_method> <years>", file=sys.stderr)
        sys.exit(2)
    engine, mode, fit_method, years = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

    y, ex = load_data(years)
    exog = ex if mode == "sarimax" else None

    gc.collect()
    mon = RSSMonitor(interval=0.1); mon.start()
    t0 = time.perf_counter()
    try:
        if engine == "rustima":
            fit = fit_rustima(y, exog, fit_method)
        elif engine == "pmdarima":
            fit = fit_pmdarima(y, exog)
        else:
            raise ValueError(f"unknown engine: {engine}")
    finally:
        elapsed = time.perf_counter() - t0
        mon.stop()

    out = {
        "engine": engine, "mode": mode, "fit_method": fit_method, "years": years,
        "n_obs": int(len(y)),
        "time_s": elapsed, "peak_rss_mb": mon.peak / (1024 ** 2),
        **fit,
    }
    print("__RESULT__" + json.dumps(out))


if __name__ == "__main__":
    main()
