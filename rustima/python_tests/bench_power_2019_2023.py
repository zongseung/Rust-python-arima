#!/usr/bin/env python3
"""
rustima auto_arima 속도·메모리 측정 (2019-2023 전력수요, exog=ta+hm)
=================================================================
시간단위 데이터를 1y/2y/3y/4y/5y로 늘려가며 auto_arima(s=24) 실행 시간과
peak RSS를 측정하고, 결과를 CSV + 2-panel 그래프로 저장한다.

실행 (rustima/ 디렉토리에서):
    .venv/bin/python python_tests/bench_power_2019_2023.py
"""
import gc
import os
import sys
import time
import threading
import warnings
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import psutil

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "..", "power_demand_final.csv")
OUTDIR = os.path.join(PROJECT_ROOT, "..", "paper", "bench_2019_2023")
os.makedirs(OUTDIR, exist_ok=True)


@dataclass
class Run:
    years: int
    n_obs: int
    time_s: float
    peak_rss_mb: float
    delta_rss_mb: float
    order: tuple
    seasonal_order: tuple
    aic: float
    n_models: int


# ── memory monitor ────────────────────────────────────────────────────────
class RSSMonitor(threading.Thread):
    """Sample process RSS at fixed interval; report peak."""
    def __init__(self, interval=0.05):
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


# ── data ───────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['일시'] = pd.to_datetime(df['일시'])
    mask = (df['일시'] >= '2019-01-01') & (df['일시'] < '2024-01-01')
    df = df.loc[mask].sort_values('일시').reset_index(drop=True)

    y = df['power demand(MW)'].values.astype(np.float64)
    exog = df[['ta', 'hm']].values.astype(np.float64)
    print(f"[데이터] 2019-01 ~ 2023-12  n={len(y):,}  exog=(ta, hm)  shape={exog.shape}")

    # NaN check
    if np.isnan(y).any() or np.isnan(exog).any():
        n_y = int(np.isnan(y).sum())
        n_e = int(np.isnan(exog).sum())
        print(f"[경고] NaN 발견: y={n_y}, exog={n_e} → 평균값으로 대체")
        if n_y:
            y = np.where(np.isnan(y), np.nanmean(y), y)
        if n_e:
            for j in range(exog.shape[1]):
                col = exog[:, j]
                exog[:, j] = np.where(np.isnan(col), np.nanmean(col), col)
    return df, y, exog


# ── single run ─────────────────────────────────────────────────────────────
def measure(years: int, y: np.ndarray, exog: np.ndarray, s: int = 24) -> Run:
    from rustima.auto import auto_arima

    n = years * 24 * 365  # rough hourly count (leap years not exact)
    n = min(n, len(y))
    yi, xi = y[:n], exog[:n]

    gc.collect()
    mon = RSSMonitor(interval=0.05)
    mon.start()
    baseline = mon.baseline

    t0 = time.perf_counter()
    res = auto_arima(
        yi, exog=xi, s=s,
        max_p=3, max_q=3, max_P=1, max_Q=1, max_d=1, max_D=1,
        stepwise=True, criterion='aic',
    )
    elapsed = time.perf_counter() - t0

    mon.stop()
    peak_mb = mon.peak / (1024 ** 2)
    delta_mb = (mon.peak - baseline) / (1024 ** 2)

    return Run(
        years=years, n_obs=n, time_s=elapsed,
        peak_rss_mb=peak_mb, delta_rss_mb=delta_mb,
        order=tuple(res.order),
        seasonal_order=tuple(res.seasonal_order),
        aic=float(res.best_ic),
        n_models=len(res.history),
    )


# ── plotting ───────────────────────────────────────────────────────────────
def save_plot(runs, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['axes.unicode_minus'] = False
    try:
        plt.rcParams['font.family'] = 'Apple SD Gothic Neo'
    except Exception:
        pass

    years = [r.years for r in runs]
    n_obs = [r.n_obs for r in runs]
    times = [r.time_s for r in runs]
    mems = [r.peak_rss_mb for r in runs]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Time
    ax = axes[0]
    ax.plot(years, times, 'o-', color='#DE6B35', lw=2.5, ms=10)
    for x, t, n in zip(years, times, n_obs):
        ax.annotate(f'{t:.1f}s\n(n={n:,})', xy=(x, t), xytext=(0, 12),
                    textcoords='offset points', ha='center', fontsize=9)
    ax.set_xlabel('Years of data (2019 → 2019+k)', fontweight='bold')
    ax.set_ylabel('auto_arima wall time (s)', fontweight='bold')
    ax.set_title('Wall time vs. data size', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xticks(years)

    # Memory
    ax = axes[1]
    ax.plot(years, mems, 's-', color='#306EA8', lw=2.5, ms=10)
    for x, m, n in zip(years, mems, n_obs):
        ax.annotate(f'{m:.0f} MB', xy=(x, m), xytext=(0, 12),
                    textcoords='offset points', ha='center', fontsize=9)
    ax.set_xlabel('Years of data', fontweight='bold')
    ax.set_ylabel('peak RSS (MB)', fontweight='bold')
    ax.set_title('Peak resident memory vs. data size', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xticks(years)

    plt.suptitle('rustima auto_arima — Korean power demand (hourly, exog=ta+hm, s=24)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[그래프] {path}")


# ── main ───────────────────────────────────────────────────────────────────
def main():
    df, y, exog = load_data()

    runs = []
    print(f"\n{'='*78}")
    print(f"{'years':>5s} {'n_obs':>8s} {'time(s)':>9s} {'ΔRSS(MB)':>10s} "
          f"{'order':>14s} {'seasonal':>14s} {'AIC':>12s} {'fits':>5s}")
    print('-' * 78)

    for k in (1, 2, 3, 4, 5):
        try:
            r = measure(k, y, exog, s=24)
        except Exception as e:
            print(f"  [{k}y] ERROR: {e}")
            continue
        runs.append(r)
        print(f"{r.years:5d} {r.n_obs:8d} {r.time_s:9.2f} {r.delta_rss_mb:10.1f} "
              f"{str(r.order):>14s} {str(r.seasonal_order):>14s} "
              f"{r.aic:12.1f} {r.n_models:5d}")
        sys.stdout.flush()

    if not runs:
        print("실행된 결과가 없습니다.")
        return

    # CSV
    rows = []
    for r in runs:
        d = asdict(r)
        d['order'] = str(r.order)
        d['seasonal_order'] = str(r.seasonal_order)
        rows.append(d)
    csv_path = os.path.join(OUTDIR, 'bench_results.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n[CSV]  {csv_path}")

    # Plot
    plot_path = os.path.join(OUTDIR, 'auto_arima_scaling.png')
    save_plot(runs, plot_path)


if __name__ == '__main__':
    main()
