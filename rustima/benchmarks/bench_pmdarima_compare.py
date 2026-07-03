#!/usr/bin/env python3
"""
pmdarima vs rustima auto_arima 비교 (1년치, 동일 조건)
====================================================
- 데이터: 2019년 한국 전력수요 (시간단위, n=8,760)
- exog: ta, hm
- s = 24
- 하드 타임아웃: 600s (워치독 스레드)
- 메모리: psutil RSS sampling
"""
import gc
import os
import signal
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
HARD_TIMEOUT_S = 600  # pmdarima만 적용


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
            if rss > self.peak:
                self.peak = rss
            self._stop.wait(self.interval)
    def stop(self):
        self._stop.set(); self.join()


class TimeoutError(Exception):
    pass


def watchdog(seconds):
    """SIGALRM-based hard timeout (POSIX only)."""
    def handler(signum, frame):
        raise TimeoutError(f"hard timeout after {seconds}s")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)


def load_1y():
    df = pd.read_csv(DATA_PATH)
    df['일시'] = pd.to_datetime(df['일시'])
    df = df[(df['일시'] >= '2019-01-01') & (df['일시'] < '2020-01-01')].reset_index(drop=True)
    y = df['power demand(MW)'].values.astype(np.float64)
    ex = df[['ta', 'hm']].values.astype(np.float64)
    if np.isnan(y).any() or np.isnan(ex).any():
        y = np.where(np.isnan(y), np.nanmean(y), y)
        for j in range(ex.shape[1]):
            col = ex[:, j]
            ex[:, j] = np.where(np.isnan(col), np.nanmean(col), col)
    print(f"[데이터] 2019년 1년치  n={len(y):,}  exog={ex.shape}")
    return y, ex


def run_rustima(y, ex):
    from rustima.auto import auto_arima
    print("\n--- rustima ---")
    gc.collect()
    mon = RSSMonitor(); mon.start()
    t0 = time.perf_counter()
    res = auto_arima(y, exog=ex, s=24, max_p=3, max_q=3, max_P=1, max_Q=1,
                     max_d=1, max_D=1, stepwise=True, criterion='aic')
    elapsed = time.perf_counter() - t0
    mon.stop()
    print(f"  order={res.order}  seasonal={res.seasonal_order}")
    print(f"  AIC={res.best_ic:.1f}  n_models={len(res.history)}")
    print(f"  time={elapsed:.2f}s  peak_RSS={mon.peak/(1024**2):.1f} MB")
    return {
        "engine": "rustima",
        "time_s": elapsed, "peak_rss_mb": mon.peak/(1024**2),
        "order": res.order, "seasonal": res.seasonal_order,
        "aic": res.best_ic, "n_models": len(res.history),
    }


def run_pmdarima(y, ex):
    import pmdarima as pm
    print(f"\n--- pmdarima (timeout={HARD_TIMEOUT_S}s) ---")
    gc.collect()
    mon = RSSMonitor(); mon.start()
    t0 = time.perf_counter()
    err = None
    res = None
    try:
        watchdog(HARD_TIMEOUT_S)
        res = pm.auto_arima(
            y, X=ex, m=24,
            start_p=0, start_q=0, start_P=0, start_Q=0,
            max_p=3, max_q=3, max_P=1, max_Q=1, max_d=1, max_D=1,
            stepwise=True, information_criterion='aic',
            error_action='ignore', suppress_warnings=True, trace=False,
        )
        signal.alarm(0)
    except TimeoutError as e:
        err = f"TIMEOUT: {e}"
        try: signal.alarm(0)
        except Exception: pass
    except KeyboardInterrupt:
        err = "KeyboardInterrupt"
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:200]}"
    elapsed = time.perf_counter() - t0
    mon.stop()
    out = {
        "engine": "pmdarima",
        "time_s": elapsed, "peak_rss_mb": mon.peak/(1024**2),
        "error": err,
    }
    if res is not None:
        out.update({
            "order": res.order, "seasonal": res.seasonal_order,
            "aic": float(res.aic()), "n_models": None,
        })
        print(f"  order={res.order}  seasonal={res.seasonal_order}")
        print(f"  AIC={res.aic():.1f}")
    else:
        print(f"  ERROR: {err}")
    print(f"  time={elapsed:.2f}s  peak_RSS={mon.peak/(1024**2):.1f} MB")
    return out


def main():
    y, ex = load_1y()
    rs = run_rustima(y, ex)
    pm_out = run_pmdarima(y, ex)

    print("\n" + "="*60)
    print(" 비교 요약 (n=8,760, exog=ta+hm, s=24)")
    print("="*60)
    print(f"  rustima  : {rs['time_s']:.1f}s   peak_RSS={rs['peak_rss_mb']:.0f} MB")
    print(f"  pmdarima : {pm_out['time_s']:.1f}s   peak_RSS={pm_out['peak_rss_mb']:.0f} MB"
          + (f"   ({pm_out['error']})" if pm_out.get('error') else ''))
    if pm_out.get('error') is None and rs['time_s'] > 0:
        print(f"  speedup  : {pm_out['time_s']/rs['time_s']:.1f}x")
        print(f"  mem ratio: {pm_out['peak_rss_mb']/rs['peak_rss_mb']:.2f}x")

    # CSV 저장
    out_dir = os.path.join(PROJECT_ROOT, "..", "paper", "bench_2019_2023")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([rs, pm_out]).to_csv(
        os.path.join(out_dir, "compare_pmdarima_1y.csv"), index=False)
    print(f"\nCSV: {out_dir}/compare_pmdarima_1y.csv")


if __name__ == "__main__":
    main()
