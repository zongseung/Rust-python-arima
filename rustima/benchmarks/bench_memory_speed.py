#!/usr/bin/env python3
"""
Rustima vs statsmodels 메모리·속도 벤치마크
===========================================
2024 전력수요 데이터(시간별, s=24) 기반 측정.

사용법 (sarimax_rs/ 디렉토리에서):
  .venv/bin/python benchmarks/bench_memory_speed.py
  .venv/bin/python benchmarks/bench_memory_speed.py --with-pmdarima
  .venv/bin/python benchmarks/bench_memory_speed.py --no-chart
"""

import argparse
import os
import sys
import time
import gc
import tracemalloc
import pickle
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "..", "power_demand_final.csv")
CHART_DIR = os.path.join(PROJECT_ROOT, "..", "paper")


# ─────────────────────────────────────────────────────────────
# 결과 컨테이너
# ─────────────────────────────────────────────────────────────

@dataclass
class BatchResult:
    n_months: int
    n_obs: int
    rs_mem_mb: float = 0.0
    rs_time_s: float = 0.0
    rs_converged: int = 0
    sm_mem_mb: float = 0.0
    sm_time_s: float = 0.0
    sm_mp_mem_mb: float = 0.0
    mem_ratio: float = 0.0
    speed_ratio: float = 0.0


@dataclass
class AutoResult:
    engine: str = ""
    order: tuple = ()
    seasonal_order: tuple = ()
    aic: float = np.nan
    time_s: float = 0.0
    n_models: int = 0
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# 데이터
# ─────────────────────────────────────────────────────────────

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df['일시'] = pd.to_datetime(df['일시'])
    df24 = df[df['일시'].dt.year == 2024].copy().reset_index(drop=True)
    y = df24['power demand(MW)'].values.astype(np.float64)

    df24['month'] = df24['일시'].dt.month
    monthly = [df24[df24['month'] == m]['power demand(MW)'].values.astype(np.float64)
               for m in sorted(df24['month'].unique())]

    print(f"[데이터] 2024 전력수요: {len(y):,}obs, {len(monthly)}개월")
    return y, monthly


# ─────────────────────────────────────────────────────────────
# 1. 배치 피팅 벤치마크
# ─────────────────────────────────────────────────────────────

def run_batch(monthly, order=(1, 0, 1), seasonal=(1, 0, 1, 24)):
    import rustima
    from statsmodels.tsa.statespace.sarimax import SARIMAX as SM

    configs = [1, 2, 4, 6, 8, 10]
    n_workers = os.cpu_count() or 4
    results = []

    print(f"\n{'='*70}")
    print(f" 배치 피팅 벤치마크  SARIMA{order}{seasonal}  (CPU {n_workers}코어)")
    print(f"{'='*70}")
    print(f"{'월':>3s} {'obs':>6s} | {'rs MB':>7s} {'rs초':>6s} {'수렴':>5s} | {'sm MB':>7s} {'sm초':>6s} | {'mp MB':>7s} | {'메모리':>6s} {'속도':>5s}")
    print("─" * 72)

    for nm in configs:
        if nm > len(monthly):
            break
        data = monthly[:nm]
        nobs = sum(len(s) for s in data)
        r = BatchResult(n_months=nm, n_obs=nobs)

        # rustima
        gc.collect(); tracemalloc.start()
        t0 = time.perf_counter()
        rr = rustima.sarimax_batch_fit(data, order, seasonal,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        r.rs_time_s = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        dm = sum(a.nbytes for a in data) / (1024**2)
        r.rs_mem_mb = dm + peak / (1024**2)
        r.rs_converged = sum(1 for x in rr if x['converged'])

        # statsmodels
        gc.collect(); tracemalloc.start()
        t0 = time.perf_counter()
        for y in data:
            SM(y, order=order, seasonal_order=seasonal,
               enforce_stationarity=False, enforce_invertibility=False,
               concentrate_scale=True).fit(disp=False, maxiter=200)
        r.sm_time_s = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        r.sm_mem_mb = dm + peak / (1024**2)

        # mp 추정
        po = sum(len(pickle.dumps(s)) for s in data) / (1024**2)
        r.sm_mp_mem_mb = r.sm_mem_mb + po + 30 * min(n_workers, nm)

        r.mem_ratio = r.sm_mp_mem_mb / r.rs_mem_mb if r.rs_mem_mb > 0 else 0
        r.speed_ratio = r.sm_time_s / r.rs_time_s if r.rs_time_s > 0 else 0

        print(f"{nm:3d} {nobs:6d} | {r.rs_mem_mb:7.2f} {r.rs_time_s:5.3f}s "
              f"{r.rs_converged:3d}/{nm} | {r.sm_mem_mb:6.1f} {r.sm_time_s:5.2f}s | "
              f"{r.sm_mp_mem_mb:6.0f} | {r.mem_ratio:5.0f}x {r.speed_ratio:4.0f}x")
        results.append(r)
        del data, rr; gc.collect()

    return results


# ─────────────────────────────────────────────────────────────
# 2. auto_arima
# ─────────────────────────────────────────────────────────────

def run_auto(y, s=24, with_pmdarima=False):
    from rustima.auto import auto_arima as rs_auto
    results = []

    print(f"\n{'='*70}")
    print(f" auto_arima  (n={len(y):,}, s={s})")
    print(f"{'='*70}")

    # rustima
    ar = AutoResult(engine="rustima")
    try:
        t0 = time.perf_counter()
        res = rs_auto(y, s=s, max_p=3, max_q=3, max_P=2, max_Q=2,
                      max_d=2, max_D=1, stepwise=True, criterion='aic')
        ar.time_s = time.perf_counter() - t0
        ar.order, ar.seasonal_order = res.order, res.seasonal_order
        ar.aic, ar.n_models = res.best_ic, len(res.history)
        print(f"  rustima  : SARIMA{ar.order}{ar.seasonal_order}")
        print(f"             AIC={ar.aic:,.2f}  {ar.time_s:.1f}초  {ar.n_models}개 모델")
    except Exception as e:
        ar.error = str(e)[:200]
        print(f"  rustima  : ERROR - {ar.error}")
    results.append(ar)

    # pmdarima
    if with_pmdarima:
        pr = AutoResult(engine="pmdarima")
        try:
            import pmdarima as pm
            print(f"  pmdarima : 실행 중 (스왑 메모리 주의)...")
            t0 = time.perf_counter()
            pm_res = pm.auto_arima(y, m=s, start_p=0, start_q=0, start_P=0, start_Q=0,
                                    max_p=3, max_q=3, max_P=2, max_Q=2,
                                    max_d=2, max_D=1, stepwise=True,
                                    information_criterion='aic',
                                    error_action='ignore', suppress_warnings=True)
            pr.time_s = time.perf_counter() - t0
            pr.order, pr.seasonal_order = pm_res.order, pm_res.seasonal_order
            pr.aic = pm_res.aic()
            print(f"  pmdarima : SARIMA{pr.order}{pr.seasonal_order}")
            print(f"             AIC={pr.aic:,.2f}  {pr.time_s:.1f}초")
        except ImportError:
            pr.error = "미설치"
            print(f"  pmdarima : 미설치 (pip install pmdarima)")
        except Exception as e:
            pr.error = str(e)[:200]
            print(f"  pmdarima : ERROR - {pr.error}")
        results.append(pr)
    else:
        print(f"  pmdarima : 스킵 (--with-pmdarima 로 활성화)")
        print(f"             ※ s=24, n≥7000에서 스왑 메모리 → 시스템 정지 가능")

    return results


# ─────────────────────────────────────────────────────────────
# 3. 그래프
# ─────────────────────────────────────────────────────────────

def save_charts(br, ar, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    plt.rcParams["font.family"] = "Apple SD Gothic Neo"
    plt.rcParams["axes.unicode_minus"] = False
    os.makedirs(outdir, exist_ok=True)

    configs = [r.n_months for r in br]
    rs_m = [r.rs_mem_mb for r in br]
    sm_m = [r.sm_mem_mb for r in br]
    mp_m = [r.sm_mp_mem_mb for r in br]
    rs_t = [r.rs_time_s for r in br]
    sm_t = [r.sm_time_s for r in br]
    nw = os.cpu_count() or 4
    x = np.arange(len(configs))
    xl = [f"{b}개월" for b in configs]
    w = 0.25

    # 개별 1: 메모리
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(x-w, rs_m, w, label='rustima (Rayon)', color='#2E75B6', zorder=3)
    ax.bar(x, sm_m, w, label='statsmodels (seq)', color='#FFA500', zorder=3)
    ax.bar(x+w, mp_m, w, label=f'statsmodels (mp, {nw}workers)', color='#C00000', zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(xl)
    ax.set_ylabel('메모리 사용량 (MB)', fontsize=12, fontweight='bold')
    ax.set_title('메모리 사용량 비교\n(2024 전력수요, SARIMA(1,0,1)(1,0,1)₂₄)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log'); ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ratio = mp_m[-1]/rs_m[-1] if rs_m[-1]>0 else 0
    ax.annotate(f'rustima: {rs_m[-1]:.2f} MB\nstatsmodels(mp): {mp_m[-1]:.0f} MB\n→ {ratio:,.0f}배 절감',
                xy=(x[-1]+w, mp_m[-1]), xytext=(x[-3], mp_m[-1]*2),
                fontsize=10, fontweight='bold', color='#C00000',
                arrowprops=dict(arrowstyle='->', color='#C00000', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF0F0', edgecolor='#C00000', alpha=0.9))
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "rustima_benchmark_memory.png"), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # 개별 2: 속도
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(configs, rs_t, 'o-', color='#2E75B6', lw=2.5, ms=8, label='rustima (Rayon)')
    ax.plot(configs, sm_t, '^-', color='#C00000', lw=2.5, ms=8, label='statsmodels (seq)')
    for i, bc in enumerate(configs):
        spd = sm_t[i]/rs_t[i] if rs_t[i]>0 else 0
        ax.annotate(f'{spd:.0f}x', xy=(bc, rs_t[i]), xytext=(0,-18),
                    textcoords='offset points', fontsize=10, fontweight='bold', color='#2E75B6', ha='center')
    ax.fill_between(configs, rs_t, sm_t, alpha=0.08, color='red')
    ax.set_xlabel('배치 크기 (개월 수)', fontsize=12, fontweight='bold')
    ax.set_ylabel('피팅 시간 (초)', fontsize=12, fontweight='bold')
    ax.set_title('배치 피팅 속도 비교\n(2024 전력수요, 월별 분할)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "rustima_benchmark_speed.png"), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # 합본 2-in-1 (메모리 + 속도)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.bar(x-w, rs_m, w, label='rustima (Rayon)', color='#2E75B6', zorder=3)
    ax.bar(x, sm_m, w, label='statsmodels (seq)', color='#FFA500', zorder=3)
    ax.bar(x+w, mp_m, w, label=f'statsmodels (mp, {nw}workers)', color='#C00000', zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(xl, fontsize=10)
    ax.set_ylabel('메모리 사용량 (MB)', fontsize=12, fontweight='bold')
    ax.set_title('메모리 사용량 비교\n(2024 전력수요, SARIMA(1,0,1)(1,0,1)₂₄)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left'); ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log'); ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ratio = mp_m[-1]/rs_m[-1] if rs_m[-1]>0 else 0
    ax.annotate(f'rustima: {rs_m[-1]:.2f} MB\nstatsmodels(mp): {mp_m[-1]:.0f} MB\n→ {ratio:,.0f}배 절감',
                xy=(x[-1]+w, mp_m[-1]), xytext=(x[-3], mp_m[-1]*2),
                fontsize=10, fontweight='bold', color='#C00000',
                arrowprops=dict(arrowstyle='->', color='#C00000', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF0F0', edgecolor='#C00000', alpha=0.9))

    ax = axes[1]
    ax.plot(configs, rs_t, 'o-', color='#2E75B6', lw=2.5, ms=8, label='rustima (Rayon)')
    ax.plot(configs, sm_t, '^-', color='#C00000', lw=2.5, ms=8, label='statsmodels (seq)')
    for i, bc in enumerate(configs):
        spd = sm_t[i]/rs_t[i] if rs_t[i]>0 else 0
        ax.annotate(f'{spd:.0f}x', xy=(bc, rs_t[i]), xytext=(0,-18),
                    textcoords='offset points', fontsize=10, fontweight='bold', color='#2E75B6', ha='center')
    ax.fill_between(configs, rs_t, sm_t, alpha=0.08, color='red')
    ax.set_xlabel('배치 크기 (개월 수)', fontsize=12, fontweight='bold')
    ax.set_ylabel('피팅 시간 (초)', fontsize=12, fontweight='bold')
    ax.set_title('배치 피팅 속도 비교\n(2024 전력수요, 월별 분할)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left'); ax.grid(True, alpha=0.3)

    plt.suptitle('Rustima 성능 벤치마크 — 2024 전력수요 (시간별, 7,320 obs)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout(pad=2.0)
    fig.savefig(os.path.join(outdir, "rustima_power_demand_benchmark.png"), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"\n[그래프 저장 완료] {outdir}/")
    print(f"  - rustima_benchmark_memory.png")
    print(f"  - rustima_benchmark_speed.png")
    print(f"  - rustima_power_demand_benchmark.png (합본)")


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rustima 메모리·속도 벤치마크")
    parser.add_argument("--data", default=DATA_PATH, help="전력수요 CSV 경로")
    parser.add_argument("--outdir", default=CHART_DIR, help="그래프 저장 경로")
    parser.add_argument("--with-pmdarima", action="store_true",
                        help="pmdarima 비교 포함 (s=24 스왑 메모리 주의)")
    parser.add_argument("--no-chart", action="store_true", help="그래프 생성 건너뛰기")
    args = parser.parse_args()

    y, monthly = load_data(args.data)
    br = run_batch(monthly)
    ar = run_auto(y, s=24, with_pmdarima=args.with_pmdarima)

    # 요약
    last = br[-1]
    print(f"\n{'='*70}")
    print(f" 최종 요약")
    print(f"{'='*70}")
    print(f"  메모리 : rustima {last.rs_mem_mb:.2f}MB vs sm(mp) {last.sm_mp_mem_mb:.0f}MB → {last.mem_ratio:,.0f}배 절감")
    print(f"  속도   : rustima {last.rs_time_s:.3f}초 vs sm {last.sm_time_s:.2f}초 → {last.speed_ratio:.0f}배")

    if not args.no_chart:
        save_charts(br, ar, args.outdir)

    print("\n완료!")


if __name__ == "__main__":
    main()
