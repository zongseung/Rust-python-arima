"""
bench_matrix_6x6_s7_s12_s24.py — 6×6 차수 매트릭스 벤치마크 (Section 2.2)

범위: p, q, P, Q ∈ {0,1,2,3,4,5}, d=1, D=1
주기: s=7 (주간), s=12 (월별), s=24 (시간별)
총 조합: 1296 × 3 = 3888개

결과: ../ver5/result/bench_matrix_6x6_s7_s12_s24.md
"""

import math
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, "python")
import rustima

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────

PERIODS = [
    {"s": 7,  "n_obs": 730,  "label": "s=7 (주간)"},
    {"s": 12, "n_obs": 300,  "label": "s=12 (월별)"},
    {"s": 24, "n_obs": 2160, "label": "s=24 (시간별)"},
]

PQ_RANGE = range(6)  # 0..5
D_FIXED  = 1
D_SEA    = 1
TOP_N    = 20  # 주기별 Top-N 표시

# ─────────────────────────────────────────
# 데이터 생성
# ─────────────────────────────────────────

def gen_seasonal(n, s, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    seasonal = 2.0 * np.sin(2 * math.pi * t / s)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.5 * y[i - 1] + seasonal[i] + rng.normal(scale=0.5)
    return y


# ─────────────────────────────────────────
# AIC 히트맵 (6×6 Markdown 표)
# ─────────────────────────────────────────

def aic_heatmap(results, row_key, col_key, row_range, col_range, fixed):
    """
    row_key, col_key: 'p','q','P','Q' 중 하나
    fixed: dict — 나머지 키 고정 값 (예: {"P": 0, "Q": 0})
    """
    key_map = {"p": 0, "q": 2, "P": 3, "Q": 5}  # order+seasonal 인덱스
    header = f"| {row_key}\\{col_key} |" + "".join(f" {c} |" for c in col_range)
    sep    = "|-----|" + "".join("------|" for _ in col_range)
    rows = [header, sep]

    for r_val in row_range:
        cells = []
        for c_val in col_range:
            # 조건에 맞는 결과 찾기
            match = None
            for res in results:
                o = res.get("order", (None, None, None))
                s = res.get("seasonal_order", (None, None, None, None))
                combo = {"p": o[0], "q": o[2], "P": s[0], "Q": s[2]}
                if combo.get(row_key) != r_val or combo.get(col_key) != c_val:
                    continue
                if all(combo.get(k) == v for k, v in fixed.items()):
                    match = res
                    break
            if match is None or "aic" not in match:
                cells.append("   -  ")
            else:
                cells.append(f"{match['aic']:6.0f}")
        rows.append(f"| {r_val}   |" + "|".join(cells) + "|")

    return "\n".join(rows)


# ─────────────────────────────────────────
# 단일 주기 실행
# ─────────────────────────────────────────

def run_period(y, s, label):
    """1296개 모델을 grid_search로 실행."""
    orders    = []
    seasonals = []
    for p, q, P, Q in product(PQ_RANGE, repeat=4):
        orders.append((p, D_FIXED, q))
        seasonals.append((P, D_SEA, Q, s))

    print(f"\n[{label}] {len(orders)} 조합 실행 중...")
    t0 = time.perf_counter()
    results = rustima.sarimax_grid_search(y, orders, seasonals)
    elapsed = time.perf_counter() - t0

    # 수렴 통계
    n_total = len(results)
    n_converged = sum(1 for r in results if r.get("converged", False))
    n_error = sum(1 for r in results if "error" in r)
    rate = n_converged / n_total * 100 if n_total > 0 else 0

    # AIC 기준 정렬 (수렴 모델만)
    converged = [r for r in results if r.get("converged", False) and "aic" in r]
    converged.sort(key=lambda r: r["aic"])

    print(f"  수렴: {n_converged}/{n_total} ({rate:.1f}%), 소요: {elapsed:.1f}s")

    return {
        "label": label,
        "s": s,
        "elapsed": elapsed,
        "n_total": n_total,
        "n_converged": n_converged,
        "n_error": n_error,
        "rate": rate,
        "results": results,
        "converged_sorted": converged,
    }


# ─────────────────────────────────────────
# Markdown 출력
# ─────────────────────────────────────────

def main():
    all_periods = []
    for cfg in PERIODS:
        y = gen_seasonal(cfg["n_obs"], cfg["s"], seed=42)
        info = run_period(y, cfg["s"], cfg["label"])
        all_periods.append(info)

    lines = []
    lines.append("# 6×6 차수 매트릭스 벤치마크\n")
    lines.append(f"> 생성: {datetime.now():%Y-%m-%d %H:%M}  ")
    lines.append(f"> rustima v{rustima.version()}  ")
    lines.append(f"> 범위: p,q,P,Q ∈ {{0..5}}, d=1, D=1\n")

    # ── 요약 ──
    lines.append("## 1. 주기별 요약\n")
    lines.append("| Period | Total | Converged | Rate | Errors | Time (s) | Best AIC | Best Model |")
    lines.append("|--------|------:|----------:|-----:|-------:|---------:|---------:|:-----------|")

    for info in all_periods:
        best = info["converged_sorted"][0] if info["converged_sorted"] else {}
        best_aic = f"{best.get('aic', float('nan')):.1f}"
        best_model = (
            f"SARIMA{best.get('order', '?')}{best.get('seasonal_order', '?')}"
            if best else "N/A"
        )
        lines.append(
            f"| {info['label']} | {info['n_total']} | {info['n_converged']} "
            f"| {info['rate']:.1f}% | {info['n_error']} | {info['elapsed']:.1f} "
            f"| {best_aic} | {best_model} |"
        )
    lines.append("")

    # ── Top-20 per period ──
    for info in all_periods:
        lines.append(f"## 2. Top-{TOP_N}: {info['label']}\n")
        lines.append("| Rank | Order | Seasonal | AIC | BIC | Loglike | #Params |")
        lines.append("|-----:|-------|----------|----:|----:|--------:|--------:|")
        for i, r in enumerate(info["converged_sorted"][:TOP_N]):
            lines.append(
                f"| {i+1} | {r['order']} | {r['seasonal_order']} "
                f"| {r['aic']:.1f} | {r['bic']:.1f} "
                f"| {r['loglike']:.2f} | {r['n_params']} |"
            )
        lines.append("")

    # ── AIC 히트맵 ──
    lines.append("## 3. AIC 히트맵 (p × q, P=0 Q=0 고정)\n")
    for info in all_periods:
        lines.append(f"### {info['label']}\n")
        hm = aic_heatmap(
            info["converged_sorted"], "p", "q", PQ_RANGE, PQ_RANGE,
            fixed={"P": 0, "Q": 0},
        )
        lines.append(hm)
        lines.append("")

    # ── 결론 ──
    lines.append("## 4. 결론\n")
    for info in all_periods:
        status = "PASS" if info["rate"] >= 95 else "FAIL"
        lines.append(f"- {info['label']}: 수렴률 {info['rate']:.1f}% → **{status}**")
    lines.append("")

    # 저장
    out_path = Path(__file__).resolve().parent.parent.parent / "ver5" / "result" / "bench_matrix_6x6_s7_s12_s24.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ 저장: {out_path}")


if __name__ == "__main__":
    main()
