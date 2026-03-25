"""
bench_v5_report.py — v5 종합 벤치마크
측정 항목:
  1. 단일 모델 속도 및 우도 비교 (rustima vs statsmodels)
  2. 배치 처리 속도
  3. auto_arima 벤치마크 (일단위 s=7, 시간단위 s=24)
결과는 Markdown 파일로 저장.
"""

import math
import sys
import time
from datetime import datetime

import numpy as np

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX as SM_SARIMAX
    HAS_SM = True
except ImportError:
    HAS_SM = False

# rustima path
sys.path.insert(0, "python")
import rustima
from sarimax_py import SARIMAXModel, auto_arima

# ─────────────────────────────────────────
# 데이터 생성 유틸
# ─────────────────────────────────────────

def _gen(n, seed=42):
    """기본 AR(1) 잡음"""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.6 * y[i-1] + rng.normal()
    return y

def gen_arima(n=300, seed=42):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    eps = rng.normal(size=n)
    for i in range(2, n):
        y[i] = y[i-1] + 0.4*(y[i-1]-y[i-2]) - 0.3*eps[i-1] + eps[i]
    return y

def gen_seasonal(n, s, seed=42):
    """계절성 있는 데이터"""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    seasonal = 2.0 * np.sin(2 * math.pi * t / s)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.5 * y[i-1] + seasonal[i] + rng.normal(scale=0.5)
    return y

def gen_daily(n=365*2, seed=42):
    """일단위 (주간 계절성 s=7)"""
    return gen_seasonal(n, s=7, seed=seed)

def gen_hourly(n=24*90, seed=42):
    """시간단위 (일별 계절성 s=24) — 90일치"""
    return gen_seasonal(n, s=24, seed=seed)

def gen_sarimax(n=300, seed=42, n_exog=2):
    """SARIMAX 데이터 생성 (exog 포함) — ARIMA(1,1,1) 과정 + n_exog 외생변수"""
    rng  = np.random.default_rng(seed)
    exog = rng.normal(size=(n, n_exog))
    betas = np.array([1.5, -0.8])[:n_exog]
    eps  = rng.normal(size=n)
    y    = np.zeros(n)
    for i in range(2, n):
        y[i] = (y[i-1] + 0.4*(y[i-1]-y[i-2])
                - 0.3*eps[i-1] + eps[i]
                + exog[i] @ betas)
    return y, exog

# ─────────────────────────────────────────
# 타이머 헬퍼
# ─────────────────────────────────────────

def timeit(fn, repeat=3):
    times = []
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return result, min(times)

# ─────────────────────────────────────────
# 1. 단일 모델 속도 & 우도 비교
# ─────────────────────────────────────────

_sarimax_111_2exog = gen_sarimax(300, n_exog=2)
_sarimax_s12_1exog = gen_sarimax(300, seed=7, n_exog=1)

MODELS = [
    {"name": "AR(1)",             "data": _gen(200),      "order": (1,0,0), "seasonal": (0,0,0,0)},
    {"name": "ARMA(1,1)",         "data": _gen(300),      "order": (1,0,1), "seasonal": (0,0,0,0)},
    {"name": "ARIMA(1,1,1)",      "data": gen_arima(300), "order": (1,1,1), "seasonal": (0,0,0,0)},
    {"name": "ARIMA(2,1,2)",      "data": gen_arima(400), "order": (2,1,2), "seasonal": (0,0,0,0)},
    {"name": "SARIMA(1,1,1)(1,1,1)12", "data": gen_seasonal(300,12), "order": (1,1,1), "seasonal": (1,1,1,12)},
    {"name": "SARIMA(1,0,0)(1,0,0)7 daily", "data": gen_daily(365), "order": (1,0,0), "seasonal": (1,0,0,7)},
    {"name": "SARIMA(1,1,1)(1,1,1)24 hourly", "data": gen_hourly(24*90), "order": (1,1,1), "seasonal": (1,1,1,24)},
    # SARIMAX with exogenous regressors
    {"name": "SARIMAX(1,1,1)+2exog",
     "data": _sarimax_111_2exog[0], "exog": _sarimax_111_2exog[1],
     "order": (1,1,1), "seasonal": (0,0,0,0)},
    {"name": "SARIMAX(1,1,1)(1,1,1)12+1exog",
     "data": _sarimax_s12_1exog[0], "exog": _sarimax_s12_1exog[1],
     "order": (1,1,1), "seasonal": (1,1,1,12)},
]

def bench_single():
    rows = []
    for spec in MODELS:
        name = spec["name"]
        y    = spec["data"]
        ord_ = spec["order"]
        sea_ = spec["seasonal"]
        exog = spec.get("exog", None)

        # --- rustima ---
        def fit_rs(y=y, ord_=ord_, sea_=sea_, exog=exog):
            model = SARIMAXModel(y, order=ord_, seasonal_order=sea_, exog=exog)
            return model.fit()

        rs_res, rs_time = timeit(fit_rs)
        rs_ll   = rs_res.llf
        rs_aic  = rs_res.aic
        rs_bic  = rs_res.bic
        rs_conv = rs_res.converged

        # --- statsmodels ---
        sm_ll = sm_aic = sm_bic = None
        sm_time = None
        sm_conv = None
        ll_diff = None
        speedup = None

        if HAS_SM:
            try:
                def fit_sm(y=y, ord_=ord_, sea_=sea_, exog=exog):
                    m = SM_SARIMAX(y, exog=exog, order=ord_, seasonal_order=sea_,
                                   enforce_stationarity=True,
                                   enforce_invertibility=True)
                    return m.fit(disp=False)

                sm_res, sm_time = timeit(fit_sm)
                sm_ll   = sm_res.llf
                sm_aic  = sm_res.aic
                sm_bic  = sm_res.bic
                sm_conv = sm_res.mle_retvals.get("converged", True)
                ll_diff = abs(rs_ll - sm_ll)
                speedup = sm_time / rs_time if rs_time > 0 else None
            except Exception as e:
                sm_ll = sm_aic = sm_bic = None

        rows.append({
            "name": name,
            "n": len(y),
            "rs_time_ms":  rs_time * 1000,
            "sm_time_ms":  sm_time * 1000 if sm_time else None,
            "speedup":     speedup,
            "rs_ll":       rs_ll,
            "sm_ll":       sm_ll,
            "ll_diff":     ll_diff,
            "rs_aic":      rs_aic,
            "rs_bic":      rs_bic,
            "rs_conv":     rs_conv,
        })
        print(f"  [{name}] rs={rs_time*1000:.1f}ms  ll={rs_ll:.3f}"
              + (f"  sm={sm_time*1000:.1f}ms  Δll={ll_diff:.4f}" if sm_ll else ""))
    return rows

# ─────────────────────────────────────────
# 2. 배치 처리 속도
# ─────────────────────────────────────────

BATCH_MODELS = [
    {
        "name": "AR(1)",
        "order": (1,0,0), "seasonal": (0,0,0,0),
        "gen": lambda n,s: _gen(n,s),
        "n_obs": 200, "n_series_list": [10, 50, 100],
    },
    {
        "name": "ARIMA(1,1,1)",
        "order": (1,1,1), "seasonal": (0,0,0,0),
        "gen": lambda n,s: gen_arima(n,s),
        "n_obs": 200, "n_series_list": [10, 50, 100],
    },
    {
        "name": "SARIMA(1,1,1)(1,1,1)12",
        "order": (1,1,1), "seasonal": (1,1,1,12),
        "gen": lambda n,s: gen_seasonal(n,12,s),
        "n_obs": 200, "n_series_list": [10, 30, 50],
    },
    {
        "name": "SARIMA(1,1,1)(1,1,1)24",
        "order": (1,1,1), "seasonal": (1,1,1,24),
        "gen": lambda n,s: gen_seasonal(n,24,s),
        "n_obs": 240, "n_series_list": [5, 10, 20],  # heavy: 시리즈 수 축소
    },
    # SARIMAX with exogenous regressors
    {
        "name": "SARIMAX(1,1,1)+2exog",
        "order": (1,1,1), "seasonal": (0,0,0,0),
        "gen": lambda n,s: gen_sarimax(n, seed=s, n_exog=2),
        "n_obs": 200, "n_series_list": [10, 50, 100],
        "has_exog": True,
    },
]

def bench_batch():
    results = []

    for spec in BATCH_MODELS:
        name          = spec["name"]
        order         = spec["order"]
        seasonal      = spec["seasonal"]
        gen_fn        = spec["gen"]
        n_obs         = spec["n_obs"]
        n_series_list = spec["n_series_list"]

        has_exog = spec.get("has_exog", False)

        for n_series in n_series_list:
            if has_exog:
                raw = [gen_fn(n_obs, i) for i in range(n_series)]
                series    = [r[0] for r in raw]
                exog_list = [r[1] for r in raw]
            else:
                series    = [gen_fn(n_obs, i) for i in range(n_series)]
                exog_list = None

            # RS batch
            def fit_rs_batch(s=series, o=order, sea=seasonal, el=exog_list):
                return rustima.sarimax_batch_fit(s, o, sea, exog_list=el)
            _, rs_t = timeit(fit_rs_batch, repeat=3)

            # SM sequential
            sm_t = None
            if HAS_SM:
                def fit_sm_batch(s=series, o=order, sea=seasonal, el=exog_list):
                    out = []
                    for idx, y in enumerate(s):
                        ex = el[idx] if el is not None else None
                        m = SM_SARIMAX(y, exog=ex, order=o, seasonal_order=sea,
                                       enforce_stationarity=True,
                                       enforce_invertibility=True)
                        out.append(m.fit(disp=False))
                    return out
                _, sm_t = timeit(fit_sm_batch, repeat=2)

            speedup = (sm_t / rs_t) if sm_t else None
            results.append({
                "model":    name,
                "n_series": n_series,
                "n_obs":    n_obs,
                "rs_ms":    rs_t * 1000,
                "sm_ms":    sm_t * 1000 if sm_t else None,
                "speedup":  speedup,
            })
            print(f"  Batch [{name}] {n_series}x{n_obs}: rs={rs_t*1000:.1f}ms"
                  + (f"  sm={sm_t*1000:.1f}ms  x{speedup:.1f}" if speedup else ""))
    return results

# ─────────────────────────────────────────
# 3. auto_arima 벤치마크
# ─────────────────────────────────────────

def bench_auto_arima():
    scenarios = [
        {
            "label": "일단위 s=7 (2년, 730일) stepwise",
            "data": gen_daily(730),
            "kwargs": dict(max_p=3, max_q=3, s=7, stepwise=True, trace=False),
        },
        {
            "label": "일단위 s=7 (2년, 730일) grid",
            "data": gen_daily(730),
            "kwargs": dict(max_p=2, max_q=2, s=7, stepwise=False, trace=False),
        },
        {
            "label": "시간단위 s=24 (90일, 2160obs) stepwise",
            "data": gen_hourly(24*90),
            "kwargs": dict(max_p=2, max_q=2, max_P=1, max_Q=1, s=24, stepwise=True, trace=False),
        },
        {
            "label": "시간단위 s=24 (90일, 2160obs) grid",
            "data": gen_hourly(24*90),
            "kwargs": dict(max_p=1, max_q=1, max_P=1, max_Q=1, s=24, stepwise=False, trace=False),
        },
        {
            "label": "시간단위 s=24 (30일, 720obs) stepwise",
            "data": gen_hourly(24*30),
            "kwargs": dict(max_p=2, max_q=2, max_P=1, max_Q=1, s=24, stepwise=True, trace=False),
        },
        {
            "label": "시간단위 s=24 (30일, 720obs) grid",
            "data": gen_hourly(24*30),
            "kwargs": dict(max_p=1, max_q=1, max_P=1, max_Q=1, s=24, stepwise=False, trace=False),
        },
    ]

    rows = []
    for sc in scenarios:
        label = sc["label"]
        y     = sc["data"]
        kw    = sc["kwargs"]
        print(f"  [{label}] n={len(y)} ...", end=" ", flush=True)

        t0  = time.perf_counter()
        res = auto_arima(y, **kw)
        elapsed = time.perf_counter() - t0

        n_models = len(res.history)
        n_conv   = sum(1 for h in res.history if h["converged"])
        best_ic  = res.best_ic
        best_ord = res.order
        best_sea = res.seasonal_order

        print(f"{elapsed:.2f}s  best={best_ord}{best_sea}  AIC={best_ic:.2f}"
              f"  ({n_conv}/{n_models} converged)")

        rows.append({
            "label":     label,
            "n":         len(y),
            "mode":      "stepwise" if kw.get("stepwise", True) else "grid",
            "time_s":    elapsed,
            "n_models":  n_models,
            "n_conv":    n_conv,
            "best_order":    str(best_ord),
            "best_seasonal": str(best_sea),
            "best_aic":  best_ic,
        })
    return rows

# ─────────────────────────────────────────
# 4. grid_search 병렬 vs 순차 속도
# ─────────────────────────────────────────

def bench_grid_parallel():
    """sarimax_grid_search (Rayon) vs 순차 sarimax_fit 루프"""
    rows = []
    for n_obs, n_combos, label in [
        (200,  9,  "n=200,  9개 조합"),
        (500,  9,  "n=500,  9개 조합"),
        (500,  25, "n=500, 25개 조합"),
        (2000, 9,  "n=2000, 9개 조합"),
    ]:
        y     = _gen(n_obs)
        pq    = [(p, 0, q) for p in range(3) for q in range(3)][:n_combos]
        seas  = [(0,0,0,0)] * len(pq)

        # 병렬
        def par():
            return rustima.sarimax_grid_search(y, pq, seas)
        _, par_t = timeit(par, repeat=3)

        # 순차
        def seq():
            return [rustima.sarimax_fit(y, o, s) for o, s in zip(pq, seas)]
        _, seq_t = timeit(seq, repeat=3)

        speedup = seq_t / par_t if par_t > 0 else None
        rows.append({
            "label":   label,
            "combos":  len(pq),
            "par_ms":  par_t*1000,
            "seq_ms":  seq_t*1000,
            "speedup": speedup,
        })
        print(f"  {label}: 병렬={par_t*1000:.1f}ms  순차={seq_t*1000:.1f}ms"
              + (f"  x{speedup:.2f}" if speedup else ""))
    return rows

# ─────────────────────────────────────────
# Markdown 생성
# ─────────────────────────────────────────

def fmt(v, decimals=2):
    if v is None:
        return "N/A"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    if isinstance(v, float):
        if not math.isfinite(v):
            return "∞"
        return f"{v:.{decimals}f}"
    return str(v)

def build_md(single, batch, auto, grid_par):
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    sm_note = "통합 비교(statsmodels)" if HAS_SM else "statsmodels 미설치 — rustima 단독"

    lines += [
        f"# sarimax-rs v5 벤치마크 리포트",
        f"",
        f"> 측정일: {now}  ",
        f"> {sm_note}",
        f"",
        "---",
        "",
    ]

    # ── 1. 단일 모델 ──
    lines += [
        "## 1. 단일 모델 속도 & 우도 비교",
        "",
        "| 모델 | n | rs 시간(ms) | sm 시간(ms) | 속도비 | rs LL | sm LL | ΔLL | 수렴 |",
        "|------|---|------------|------------|-------|-------|-------|-----|-----|",
    ]
    for r in single:
        spd = f"{r['speedup']:.1f}x" if r['speedup'] else "N/A"
        lines.append(
            f"| {r['name']} | {r['n']} "
            f"| {fmt(r['rs_time_ms'],1)} | {fmt(r['sm_time_ms'],1)} | {spd} "
            f"| {fmt(r['rs_ll'],2)} | {fmt(r['sm_ll'],2)} | {fmt(r['ll_diff'],4)} "
            f"| {'✓' if r['rs_conv'] else '✗'} |"
        )

    if HAS_SM:
        valid_sp = [r['speedup'] for r in single if r['speedup']]
        if valid_sp:
            lines += [
                "",
                f"**평균 속도비**: {sum(valid_sp)/len(valid_sp):.1f}x  ",
                f"**최대 속도비**: {max(valid_sp):.1f}x  ",
                f"**최소 ΔLL**: {min(r['ll_diff'] for r in single if r['ll_diff'] is not None):.4f}  ",
                f"**최대 ΔLL**: {max(r['ll_diff'] for r in single if r['ll_diff'] is not None):.4f}",
            ]

    # ── 2. 배치 ──
    lines += [
        "",
        "---",
        "",
        "## 2. 배치 처리 속도 (모델별, Rayon 병렬 vs statsmodels 순차)",
        "",
        "| 모델 | 시계열 수 | n_obs | rs 시간(ms) | sm 시간(ms) | 속도비 |",
        "|-----|----------|-------|------------|------------|-------|",
    ]
    for r in batch:
        spd = f"{r['speedup']:.1f}x" if r['speedup'] else "N/A"
        lines.append(
            f"| {r['model']} | {r['n_series']} | {r['n_obs']} "
            f"| {fmt(r['rs_ms'],1)} | {fmt(r['sm_ms'],1)} | {spd} |"
        )

    # ── 3. grid_search 병렬 ──
    lines += [
        "",
        "---",
        "",
        "## 3. sarimax_grid_search 병렬 vs 순차 속도",
        "",
        "| 시나리오 | 조합 수 | 병렬(ms) | 순차(ms) | 속도비 |",
        "|---------|--------|---------|---------|-------|",
    ]
    for r in grid_par:
        spd = f"{r['speedup']:.2f}x" if r['speedup'] else "N/A"
        lines.append(
            f"| {r['label']} | {r['combos']} "
            f"| {fmt(r['par_ms'],1)} | {fmt(r['seq_ms'],1)} | {spd} |"
        )

    # ── 4. auto_arima ──
    lines += [
        "",
        "---",
        "",
        "## 4. auto_arima 벤치마크",
        "",
        "### 4-1. 일단위 데이터 (s=7, 주간 계절성)",
        "",
        "| 시나리오 | n | 모드 | 시간(s) | 탐색 모델 | 수렴 | 최적 order | AIC |",
        "|---------|---|-----|--------|----------|-----|-----------|-----|",
    ]
    daily = [r for r in auto if "일단위" in r["label"]]
    for r in daily:
        lines.append(
            f"| {r['label']} | {r['n']} | {r['mode']} "
            f"| {r['time_s']:.2f} | {r['n_models']} | {r['n_conv']}/{r['n_models']} "
            f"| {r['best_order']}{r['best_seasonal']} | {fmt(r['best_aic'],3)} |"
        )

    lines += [
        "",
        "### 4-2. 시간단위 데이터 (s=24, 일별 계절성)",
        "",
        "| 시나리오 | n | 모드 | 시간(s) | 탐색 모델 | 수렴 | 최적 order | AIC |",
        "|---------|---|-----|--------|----------|-----|-----------|-----|",
    ]
    hourly = [r for r in auto if "시간단위" in r["label"]]
    for r in hourly:
        lines.append(
            f"| {r['label']} | {r['n']} | {r['mode']} "
            f"| {r['time_s']:.2f} | {r['n_models']} | {r['n_conv']}/{r['n_models']} "
            f"| {r['best_order']}{r['best_seasonal']} | {fmt(r['best_aic'],3)} |"
        )

    # ── 분석 요약 ──
    lines += [
        "",
        "---",
        "",
        "## 5. 분석 요약",
        "",
    ]

    if HAS_SM:
        valid_ll = [(r['name'], r['ll_diff']) for r in single if r['ll_diff'] is not None]
        lines += [
            "### 우도 정확도",
            "",
            "| 평가 기준 | 값 |",
            "|----------|---|",
        ]
        for name, d in valid_ll:
            grade = "✓ 우수 (< 1e-2)" if d < 0.01 else ("✓ 양호 (< 1.0)" if d < 1.0 else "주의 (≥ 1.0)")
            lines.append(f"| {name} ΔLL | {d:.4f} — {grade} |")

    lines += [
        "",
        "### auto_arima 관찰 사항",
        "",
    ]

    for r in auto:
        stepwise_tag = "stepwise" if r["mode"] == "stepwise" else "grid(Rayon 병렬)"
        note = (
            f"- **{r['label']}**: {r['time_s']:.2f}s, "
            f"최적 ARIMA{r['best_order']}×{r['best_seasonal']}, "
            f"AIC={fmt(r['best_aic'],3)}, "
            f"수렴 {r['n_conv']}/{r['n_models']}"
        )
        lines.append(note)

    lines += [
        "",
        "### 결론",
        "",
        "| 항목 | 결과 |",
        "|-----|-----|",
    ]
    if HAS_SM:
        avg_sp = sum(r['speedup'] for r in single if r['speedup']) / max(1, sum(1 for r in single if r['speedup']))
        lines.append(f"| 단일 모델 평균 속도 향상 | {avg_sp:.1f}x |")
        avg_batch_sp = sum(r['speedup'] for r in batch if r['speedup']) / max(1, sum(1 for r in batch if r['speedup']))
        lines.append(f"| 배치 평균 속도 향상 | {avg_batch_sp:.1f}x |")
        max_ll_diff = max((r['ll_diff'] for r in single if r['ll_diff'] is not None), default=None)
        if max_ll_diff:
            lines.append(f"| 최대 ΔLogLikelihood | {max_ll_diff:.4f} |")

    par_speedups = [r['speedup'] for r in grid_par if r['speedup']]
    if par_speedups:
        lines.append(f"| grid_search 병렬 평균 속도비 | {sum(par_speedups)/len(par_speedups):.2f}x |")

    daily_times = [r['time_s'] for r in auto if "일단위" in r['label']]
    hourly_times = [r['time_s'] for r in auto if "시간단위" in r['label']]
    if daily_times:
        lines.append(f"| auto_arima 일단위(s=7) 시간 범위 | {min(daily_times):.2f}s ~ {max(daily_times):.2f}s |")
    if hourly_times:
        lines.append(f"| auto_arima 시간단위(s=24) 시간 범위 | {min(hourly_times):.2f}s ~ {max(hourly_times):.2f}s |")

    lines += ["", "---", "", "_sarimax-rs v5 by Rust + PyO3 + Rayon_"]
    return "\n".join(lines)

# ─────────────────────────────────────────
# main
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("sarimax-rs v5 벤치마크")
    print("=" * 60)

    print("\n[1/4] 단일 모델 속도 & 우도 비교 ...")
    single = bench_single()

    print("\n[2/4] 배치 처리 속도 ...")
    batch = bench_batch()

    print("\n[3/4] grid_search 병렬 vs 순차 ...")
    grid_par = bench_grid_parallel()

    print("\n[4/4] auto_arima (일단위 s=7 / 시간단위 s=24) ...")
    auto = bench_auto_arima()

    print("\n결과 Markdown 생성 중 ...")
    md = build_md(single, batch, auto, grid_par)

    out_path = "../ver5/BENCHMARK_v5.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n저장 완료 → {out_path}")
    print("\n" + "=" * 60)
    print(md[:2000])  # 미리보기
