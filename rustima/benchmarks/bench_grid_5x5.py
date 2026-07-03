"""
bench_grid_5x5.py — SARIMA 격자 탐색 벤치마크
  p, q, P, Q ∈ {0, 1, 2, 3, 4}  (5^4 = 625 조합)
  d=1, D=1, s=12  고정

측정:
  - rustima: 전체 625개 병렬 grid_search
  - statsmodels: AIC 상위 10개 비교 (ΔLL, 속도비)
  - p×q AIC 격자 (P=Q=0)
  - P×Q AIC 격자 (best p,q)

결과: ../ver5/result/bench_grid_5x5.md
"""

import math
import sys
import time
from datetime import datetime
from itertools import product

import numpy as np

sys.path.insert(0, "python")
import rustima
from rustima import SARIMAXModel

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX as SM_SARIMAX
    HAS_SM = True
except ImportError:
    HAS_SM = False

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
D_FIXED  = 1
D_SEA    = 1
S        = 12
N_OBS    = 300
TOP_N_SM = 10   # statsmodels 비교 대상 모델 수

# ─────────────────────────────────────────
# 데이터 생성
# ─────────────────────────────────────────
def gen_seasonal(n, s, seed=42):
    rng = np.random.default_rng(seed)
    t   = np.arange(n, dtype=float)
    seasonal = 2.0 * np.sin(2 * math.pi * t / s)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.5 * y[i-1] + seasonal[i] + rng.normal(scale=0.5)
    return y

y = gen_seasonal(N_OBS, S)

# ─────────────────────────────────────────
# 625개 조합 생성
# ─────────────────────────────────────────
orders    = []
seasonals = []
for p, q, P, Q in product(range(5), repeat=4):
    orders.append((p, D_FIXED, q))
    seasonals.append((P, D_SEA, Q, S))

n_total = len(orders)
print(f"총 {n_total}개 조합: SARIMA(p,{D_FIXED},q)(P,{D_SEA},Q,{S}), p/q/P/Q ∈ [0..4]")
print(f"데이터: n={N_OBS}, 계절 주기 s={S}")

# ─────────────────────────────────────────
# [1] rustima 전체 grid_search (Rayon 병렬)
# ─────────────────────────────────────────
print(f"\n[1/2] rustima grid_search ({n_total}개 병렬) ...")
t0 = time.perf_counter()
rs_raw = rustima.sarimax_grid_search(y, orders, seasonals)
rs_total_time = time.perf_counter() - t0
print(f"  완료: {rs_total_time:.2f}s  (평균 {rs_total_time/n_total*1000:.1f}ms/모델)")

# 결과 파싱
rows = []
for i, raw in enumerate(rs_raw):
    p, d, q  = orders[i]
    P, D, Q, s = seasonals[i]
    if "error" in raw:
        rows.append({
            "p": p, "q": q, "P": P, "Q": Q,
            "order": orders[i], "seasonal": seasonals[i],
            "rs_ll": None, "rs_aic": None, "rs_bic": None,
            "rs_nparams": None, "rs_conv": False,
            "error": raw["error"],
        })
    else:
        rows.append({
            "p": p, "q": q, "P": P, "Q": Q,
            "order": orders[i], "seasonal": seasonals[i],
            "rs_ll":      raw["loglike"],
            "rs_aic":     raw["aic"],
            "rs_bic":     raw["bic"],
            "rs_nparams": raw["n_params"],
            "rs_conv":    raw["converged"],
            "error": None,
        })

converged_rows = [
    r for r in rows
    if r["rs_conv"]
    and r["rs_aic"] is not None
    and math.isfinite(r["rs_aic"])
]
failed_rows = [r for r in rows if not r["rs_conv"] or r["rs_aic"] is None or not math.isfinite(r["rs_aic"])]
converged_rows.sort(key=lambda x: x["rs_aic"])

n_conv  = len(converged_rows)
n_fail  = len(failed_rows)
best    = converged_rows[0]
worst   = converged_rows[-1]

print(f"  수렴: {n_conv}/{n_total}  실패: {n_fail}")
print(f"  최적 AIC={best['rs_aic']:.2f}  SARIMA{best['order']}×{best['seasonal']}")
print(f"  최대 AIC={worst['rs_aic']:.2f}  SARIMA{worst['order']}×{worst['seasonal']}")

# ─────────────────────────────────────────
# [2] statsmodels 상위 TOP_N_SM 비교
# ─────────────────────────────────────────
sm_compare = []
if HAS_SM:
    print(f"\n[2/2] statsmodels 상위 {TOP_N_SM}개 비교 ...")
    for r in converged_rows[:TOP_N_SM]:
        try:
            t0 = time.perf_counter()
            m   = SM_SARIMAX(y, order=r["order"], seasonal_order=r["seasonal"],
                             enforce_stationarity=True, enforce_invertibility=True)
            res = m.fit(disp=False)
            sm_t = time.perf_counter() - t0
            dl = abs(r["rs_ll"] - res.llf)
            sm_compare.append({
                **r,
                "sm_ll":   res.llf,
                "sm_aic":  res.aic,
                "sm_bic":  res.bic,
                "sm_ms":   sm_t * 1000,
                "delta_ll": dl,
                "delta_aic": abs(r["rs_aic"] - res.aic),
            })
            print(f"  SARIMA{r['order']}×{r['seasonal']}: "
                  f"rs_aic={r['rs_aic']:.2f}  sm_aic={res.aic:.2f}  "
                  f"ΔLL={dl:.4f}  sm={sm_t*1000:.0f}ms")
        except Exception as e:
            print(f"  SARIMA{r['order']}×{r['seasonal']}: SM ERROR — {e}")
            sm_compare.append({**r, "sm_ll": None, "sm_aic": None, "sm_bic": None,
                               "sm_ms": None, "delta_ll": None, "delta_aic": None})

# ─────────────────────────────────────────
# 격자 테이블 빌드 헬퍼
# ─────────────────────────────────────────
def aic_grid(rows_all, fixed_P, fixed_Q):
    """p×q AIC 격자 (P,Q 고정)"""
    grid = {}
    for r in rows_all:
        if r["P"] == fixed_P and r["Q"] == fixed_Q and r["rs_aic"] is not None and math.isfinite(r["rs_aic"]):
            grid[(r["p"], r["q"])] = r["rs_aic"]
    return grid

def best_pq_grid(rows_all, P_val, Q_val):
    """특정 P,Q에서 최적 p,q"""
    sub = [r for r in rows_all if r["P"]==P_val and r["Q"]==Q_val and r["rs_aic"] is not None and math.isfinite(r["rs_aic"])]
    if not sub:
        return None
    return min(sub, key=lambda x: x["rs_aic"])

def PQ_grid(rows_all, fixed_p, fixed_q):
    """P×Q AIC 격자 (p,q 고정)"""
    grid = {}
    for r in rows_all:
        if r["p"] == fixed_p and r["q"] == fixed_q and r["rs_aic"] is not None and math.isfinite(r["rs_aic"]):
            grid[(r["P"], r["Q"])] = r["rs_aic"]
    return grid

def fmt_aic(v):
    if v is None:
        return "   —   "
    if not math.isfinite(v):
        return "  ∞  "
    return f"{v:7.1f}"

def build_grid_md(grid_fn, row_label, col_label, vals=range(5)):
    """범용 5×5 격자 Markdown 테이블"""
    lines = []
    header = f"| {row_label}\\{col_label} |" + "".join(f" {v}      |" for v in vals)
    sep    = "|" + "-----------|" * (len(list(vals)) + 1)
    lines.append(header)
    lines.append(sep)
    for rv in vals:
        row_str = f"| **{rv}** |"
        for cv in vals:
            v = grid_fn.get((rv, cv), None)
            row_str += f" {fmt_aic(v)} |"
        lines.append(row_str)
    return lines

# ─────────────────────────────────────────
# Markdown 생성
# ─────────────────────────────────────────
def fmt(v, d=2):
    if v is None:
        return "N/A"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    if isinstance(v, float):
        if not math.isfinite(v):
            return "∞"
        return f"{v:.{d}f}"
    return str(v)

lines = []
now   = datetime.now().strftime("%Y-%m-%d %H:%M")

lines += [
    f"# SARIMA 격자 탐색 벤치마크: p,q,P,Q ∈ {{0..4}}",
    f"",
    f"> 측정일: {now}",
    f"> 환경: Apple Silicon (macOS), Rust + PyO3 + Rayon, Python 3.14",
    f"> statsmodels {'0.14.6' if HAS_SM else '미설치'}",
    f"",
    "---",
    "",
    "## 설정",
    "",
    f"| 항목 | 값 |",
    f"|-----|---|",
    f"| 조합 공간 | p, q, P, Q ∈ {{0,1,2,3,4}} |",
    f"| 고정 차수 | d={D_FIXED}, D={D_SEA}, s={S} |",
    f"| 총 조합 수 | {n_total} |",
    f"| 데이터 | 계절성 시뮬레이션 (s={S}, n={N_OBS}) |",
    f"| rustima 방법 | L-BFGS-B + Nelder-Mead fallback (Rayon 병렬) |",
    f"",
    "---",
    "",
]

# ── 1. 수렴 현황 ──
lines += [
    "## 1. 전체 수렴 현황",
    "",
    f"| 항목 | 값 |",
    f"|-----|---|",
    f"| 총 조합 | {n_total} |",
    f"| 수렴 성공 | {n_conv} ({n_conv/n_total*100:.1f}%) |",
    f"| 수렴 실패 / 에러 | {n_fail} ({n_fail/n_total*100:.1f}%) |",
    f"| rustima 총 시간 | {rs_total_time:.2f}s |",
    f"| rustima 평균 시간/모델 | {rs_total_time/n_total*1000:.1f}ms |",
    f"| 최적 AIC | {fmt(best['rs_aic'],2)} — SARIMA{best['order']}×{best['seasonal']} |",
    f"| 최소 AIC | {fmt(converged_rows[-1]['rs_aic'],2)} |",
    f"| AIC 중앙값 | {fmt(sorted([r['rs_aic'] for r in converged_rows])[n_conv//2],2)} |",
    f"",
    "---",
    "",
]

# ── 2. 상위 30개 ──
TOP_SHOW = 30
lines += [
    f"## 2. 상위 {TOP_SHOW}개 모델 (AIC 기준, rustima)",
    "",
    "| 순위 | 모델 | n_params | LL | AIC | BIC | 수렴 |",
    "|:---:|------|:-------:|---:|----:|----:|:---:|",
]
for rank, r in enumerate(converged_rows[:TOP_SHOW], 1):
    model_str = f"SARIMA{r['order']}×{r['seasonal']}"
    lines.append(
        f"| {rank} | {model_str} | {r['rs_nparams']} "
        f"| {fmt(r['rs_ll'],2)} | {fmt(r['rs_aic'],2)} | {fmt(r['rs_bic'],2)} "
        f"| {'✓' if r['rs_conv'] else '✗'} |"
    )

lines += ["", "---", ""]

# ── 3. statsmodels 비교 ──
if sm_compare:
    valid_sm = [r for r in sm_compare if r["sm_ll"] is not None]
    lines += [
        f"## 3. statsmodels 상위 {TOP_N_SM}개 비교",
        "",
        "| 순위 | 모델 | rs AIC | sm AIC | ΔAIC | rs LL | sm LL | ΔLL | sm 시간(ms) |",
        "|:---:|------|-------:|-------:|-----:|------:|------:|----:|:----------:|",
    ]
    for rank, r in enumerate(sm_compare, 1):
        model_str = f"SARIMA{r['order']}×{r['seasonal']}"
        lines.append(
            f"| {rank} | {model_str} "
            f"| {fmt(r['rs_aic'],2)} | {fmt(r['sm_aic'],2)} | {fmt(r['delta_aic'],3)} "
            f"| {fmt(r['rs_ll'],2)} | {fmt(r['sm_ll'],2)} | {fmt(r['delta_ll'],4)} "
            f"| {fmt(r['sm_ms'],1)} |"
        )

    if valid_sm:
        avg_dll  = sum(r["delta_ll"]  for r in valid_sm) / len(valid_sm)
        avg_daic = sum(r["delta_aic"] for r in valid_sm) / len(valid_sm)
        max_dll  = max(r["delta_ll"]  for r in valid_sm)
        lines += [
            "",
            f"**평균 ΔLL**: {avg_dll:.4f}  ",
            f"**최대 ΔLL**: {max_dll:.4f}  ",
            f"**평균 ΔAIC**: {avg_daic:.4f}",
        ]

    lines += ["", "---", ""]

# ── 4. p×q AIC 격자 (P=0,Q=0) ──
grid_pq_00 = aic_grid(converged_rows, 0, 0)
lines += [
    "## 4. AIC 격자 (P=0, Q=0) — 순수 ARIMA",
    "",
    f"> SARIMA(p,{D_FIXED},q)(0,{D_SEA},0,{S}) 格子. 값: AIC",
    "",
]
lines += build_grid_md(grid_pq_00, "p", "q")

lines += ["", ""]

# ── 5. p×q AIC 격자 (P=1,Q=1) ──
grid_pq_11 = aic_grid(converged_rows, 1, 1)
lines += [
    "## 5. AIC 격자 (P=1, Q=1)",
    "",
    f"> SARIMA(p,{D_FIXED},q)(1,{D_SEA},1,{S}) 格子. 값: AIC",
    "",
]
lines += build_grid_md(grid_pq_11, "p", "q")

lines += ["", ""]

# ── 6. P×Q AIC 격자 (best p,q 고정) ──
best_p, best_q = best["p"], best["q"]
grid_PQ_best = PQ_grid(converged_rows, best_p, best_q)
lines += [
    f"## 6. AIC 격자 (p={best_p}, q={best_q}) — 계절 성분 탐색",
    "",
    f"> SARIMA({best_p},{D_FIXED},{best_q})(P,{D_SEA},Q,{S}) 格子. 값: AIC",
    "",
]
lines += build_grid_md(grid_PQ_best, "P", "Q")

lines += ["", ""]

# ── 7. 실패 모델 목록 ──
if failed_rows:
    lines += [
        "## 7. 수렴 실패 / 에러 모델",
        "",
        f"총 {len(failed_rows)}개 모델이 수렴에 실패하거나 에러 발생:",
        "",
        "| 모델 | 에러/사유 |",
        "|-----|---------|",
    ]
    for r in failed_rows[:30]:
        err = r.get("error") or "수렴 실패"
        # 에러 메시지 길이 제한
        err_short = err[:80] + ("..." if len(err) > 80 else "")
        lines.append(f"| SARIMA{r['order']}×{r['seasonal']} | {err_short} |")
    if len(failed_rows) > 30:
        lines.append(f"| ... | (총 {len(failed_rows)}개) |")
    lines += ["", "---", ""]

# ── 8. 모델 복잡도별 요약 ──
lines += [
    "## 8. 모델 복잡도별 AIC 요약",
    "",
    "| 모델 그룹 | 조합 수 | 수렴 수 | 최적 AIC | 최적 모델 |",
    "|----------|:-------:|:------:|--------:|----------|",
]
groups = [
    ("순수 ARIMA (P=Q=0)",       lambda r: r["P"]==0 and r["Q"]==0),
    ("계절 AR만 (P>0,Q=0)",      lambda r: r["P"]>0  and r["Q"]==0),
    ("계절 MA만 (P=0,Q>0)",      lambda r: r["P"]==0 and r["Q"]>0),
    ("계절 ARMA (P>0,Q>0)",      lambda r: r["P"]>0  and r["Q"]>0),
    ("소형 (p+q≤2, P+Q≤1)",     lambda r: r["p"]+r["q"]<=2 and r["P"]+r["Q"]<=1),
    ("대형 (p+q≥6, P+Q≥3)",     lambda r: r["p"]+r["q"]>=6 and r["P"]+r["Q"]>=3),
]
for label, fn in groups:
    sub  = [r for r in rows if fn(r)]
    conv = [r for r in sub if r["rs_aic"] is not None and math.isfinite(r["rs_aic"]) and r["rs_conv"]]
    if conv:
        best_g = min(conv, key=lambda x: x["rs_aic"])
        lines.append(
            f"| {label} | {len(sub)} | {len(conv)} "
            f"| {fmt(best_g['rs_aic'],2)} | SARIMA{best_g['order']}×{best_g['seasonal']} |"
        )
    else:
        lines.append(f"| {label} | {len(sub)} | 0 | — | — |")

lines += ["", "---", ""]

# ── 9. 결론 ──
lines += [
    "## 9. 결론",
    "",
    "| 항목 | 결과 |",
    "|-----|------|",
    f"| 탐색 조합 수 | {n_total}개 |",
    f"| rustima 총 시간 | {rs_total_time:.2f}s (Rayon 병렬) |",
    f"| 수렴률 | {n_conv/n_total*100:.1f}% |",
    f"| 최적 모델 | SARIMA{best['order']}×{best['seasonal']} |",
    f"| 최적 AIC | {fmt(best['rs_aic'],2)} |",
    f"| 최적 BIC | {fmt(best['rs_bic'],2)} |",
    f"| 최적 LL | {fmt(best['rs_ll'],2)} |",
]
if sm_compare and any(r["sm_ll"] is not None for r in sm_compare):
    valid_sm = [r for r in sm_compare if r["sm_ll"] is not None]
    avg_dll  = sum(r["delta_ll"] for r in valid_sm) / len(valid_sm)
    lines.append(f"| statsmodels 대비 평균 ΔLL (상위 {len(valid_sm)}개) | {avg_dll:.4f} |")
    avg_sm_t = sum(r["sm_ms"] for r in valid_sm) / len(valid_sm)
    rs_avg_t = rs_total_time / n_total * 1000
    lines.append(f"| rustima 평균 속도 (vs statsmodels 상위{len(valid_sm)}) | {avg_sm_t/rs_avg_t:.1f}x |")

lines += [
    "",
    "---",
    "",
    "_sarimax-rs by Rust + PyO3 + Rayon_",
]

md = "\n".join(lines)

# ─────────────────────────────────────────
# 저장
# ─────────────────────────────────────────
out_path = "../ver5/result/bench_grid_5x5.md"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(md)

print(f"\n저장 완료 → {out_path}")
print("\n" + "=" * 60)
print(md[:3000])
