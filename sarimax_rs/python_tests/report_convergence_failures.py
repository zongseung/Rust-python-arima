"""
report_convergence_failures.py — 수렴 실패 자동 분류 리포트 (Section 2.6)

분류 체계:
  - converged       : 정상 수렴
  - non_converged   : 최대 반복 도달, 수렴 미달
  - numerical_nan_inf: NaN/Inf 발생
  - invalid_params  : 파라미터 범위 위반
  - timeout         : 시간 초과 (단일 모델 >60s)
  - exception_other : 기타 예외

모델 범위:
  - 비계절: p,q ∈ [0..3], d ∈ [0,1]         → 32 조합
  - s=12:  p,q ∈ [0..2], P,Q ∈ [0..2], d=1, D=1 → 81 조합
  - s=24:  p,q ∈ [0..2], P,Q ∈ [0..1], d=1, D=1 → 36 조합

결과: ../ver5/result/convergence_report.md
"""

import math
import sys
import time
from collections import Counter
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, "python")
import rustima

TIMEOUT_SEC = 60  # 단일 모델 timeout 기준

# ─────────────────────────────────────────
# 데이터 생성
# ─────────────────────────────────────────

def gen_stationary(n=300, seed=42):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.5 * y[i - 1] + rng.normal()
    return y


def gen_random_walk(n=300, seed=42):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=n))


def gen_seasonal(n, s, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    seasonal = 2.0 * np.sin(2 * math.pi * t / s)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.5 * y[i - 1] + seasonal[i] + rng.normal(scale=0.5)
    return y


# ─────────────────────────────────────────
# 분류 함수
# ─────────────────────────────────────────

def classify_result(y, order, seasonal):
    """모델 하나를 fit하고 결과를 분류한다."""
    try:
        t0 = time.perf_counter()
        result = rustima.sarimax_fit(y, order, seasonal)
        elapsed = time.perf_counter() - t0

        if elapsed > TIMEOUT_SEC:
            return "timeout", elapsed, None

        if not result["converged"]:
            return "non_converged", elapsed, None

        params = result["params"]
        if any(not np.isfinite(p) for p in params):
            return "numerical_nan_inf", elapsed, None

        if not np.isfinite(result["loglike"]):
            return "numerical_nan_inf", elapsed, None

        return "converged", elapsed, result

    except Exception as e:
        err = str(e).lower()
        if "nan" in err or "inf" in err or "finite" in err:
            return "numerical_nan_inf", None, str(e)
        if "param" in err or "length" in err:
            return "invalid_params", None, str(e)
        return "exception_other", None, str(e)


# ─────────────────────────────────────────
# 모델 매트릭스 정의
# ─────────────────────────────────────────

def build_matrix():
    """3개 그룹에 대한 (order, seasonal, group_name) 리스트 반환."""
    configs = []

    # 비계절: p,q ∈ [0..3], d ∈ [0,1]
    for p, q, d in product(range(4), range(4), [0, 1]):
        if p == 0 and q == 0:
            continue  # ARIMA(0,d,0) — 의미 없음
        configs.append(((p, d, q), (0, 0, 0, 0), "non-seasonal"))

    # s=12: p,q ∈ [0..2], P,Q ∈ [0..2], d=1, D=1
    for p, q, P, Q in product(range(3), range(3), range(3), range(3)):
        if p == 0 and q == 0 and P == 0 and Q == 0:
            continue
        configs.append(((p, 1, q), (P, 1, Q, 12), "s=12"))

    # s=24: p,q ∈ [0..2], P,Q ∈ [0..1], d=1, D=1
    for p, q, P, Q in product(range(3), range(3), range(2), range(2)):
        if p == 0 and q == 0 and P == 0 and Q == 0:
            continue
        configs.append(((p, 1, q), (P, 1, Q, 24), "s=24"))

    return configs


# ─────────────────────────────────────────
# 히트맵 생성 (ASCII Markdown 표)
# ─────────────────────────────────────────

def build_heatmap(results, group, p_range, q_range, P_fixed=None, Q_fixed=None):
    """p × q 히트맵 (특정 P,Q 고정). ✓=수렴, ✗=실패, -=해당없음"""
    header = "| p\\q |" + "".join(f" {q} |" for q in q_range)
    sep    = "|-----|" + "".join("---|" for _ in q_range)
    rows = [header, sep]

    for p in p_range:
        cells = []
        for q in q_range:
            # 해당 (p,q) 에 맞는 결과 찾기
            matches = [
                r for r in results
                if r["group"] == group
                and r["order"][0] == p and r["order"][2] == q
                and (P_fixed is None or r["seasonal"][0] == P_fixed)
                and (Q_fixed is None or r["seasonal"][2] == Q_fixed)
            ]
            if not matches:
                cells.append(" - ")
            else:
                n_conv = sum(1 for m in matches if m["status"] == "converged")
                if n_conv == len(matches):
                    cells.append(" ✓ ")
                elif n_conv > 0:
                    cells.append(f"{n_conv}/{len(matches)}")
                else:
                    cells.append(" ✗ ")
        rows.append(f"| {p}   |" + "|".join(cells) + "|")

    return "\n".join(rows)


# ─────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────

def main():
    configs = build_matrix()
    print(f"총 {len(configs)} 모델 조합 실행...")

    # 데이터 (그룹별 하나씩)
    data = {
        "non-seasonal": gen_random_walk(300, seed=42),
        "s=12": gen_seasonal(300, 12, seed=42),
        "s=24": gen_seasonal(2160, 24, seed=42),
    }

    # 실행 + 분류
    results = []
    for i, (order, seasonal, group) in enumerate(configs):
        y = data[group]
        status, elapsed, _ = classify_result(y, order, seasonal)
        results.append({
            "order": order,
            "seasonal": seasonal,
            "group": group,
            "status": status,
            "elapsed": elapsed,
        })
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(configs)}...")

    # ── 통계 집계 ──
    groups = ["non-seasonal", "s=12", "s=24"]
    lines = []
    lines.append("# 수렴 실패 자동 분류 리포트\n")
    lines.append(f"> 생성: {datetime.now():%Y-%m-%d %H:%M}  ")
    lines.append(f"> rustima v{rustima.version()}\n")

    # 그룹별 요약
    lines.append("## 1. 그룹별 수렴률\n")
    lines.append("| Group | Total | Converged | Rate | Non-conv | NaN/Inf | Invalid | Timeout | Other |")
    lines.append("|-------|------:|----------:|-----:|---------:|--------:|--------:|--------:|------:|")

    total_all, conv_all = 0, 0
    for g in groups:
        grp = [r for r in results if r["group"] == g]
        counts = Counter(r["status"] for r in grp)
        n = len(grp)
        nc = counts.get("converged", 0)
        total_all += n
        conv_all += nc
        rate = nc / n * 100 if n > 0 else 0
        lines.append(
            f"| {g} | {n} | {nc} | {rate:.1f}% "
            f"| {counts.get('non_converged', 0)} "
            f"| {counts.get('numerical_nan_inf', 0)} "
            f"| {counts.get('invalid_params', 0)} "
            f"| {counts.get('timeout', 0)} "
            f"| {counts.get('exception_other', 0)} |"
        )

    overall_rate = conv_all / total_all * 100 if total_all > 0 else 0
    lines.append(f"| **전체** | **{total_all}** | **{conv_all}** | **{overall_rate:.1f}%** | | | | | |")
    lines.append("")

    # 히트맵
    lines.append("## 2. p × q 수렴 히트맵\n")

    lines.append("### 비계절 (d=1 기준)\n")
    nonseasonal_d1 = [r for r in results if r["group"] == "non-seasonal" and r["order"][1] == 1]
    if nonseasonal_d1:
        # 비계절 d=1 히트맵
        header = "| p\\q |" + "".join(f" {q} |" for q in range(4))
        sep    = "|-----|" + "".join("---|" for _ in range(4))
        rows_ns = [header, sep]
        for p in range(4):
            cells = []
            for q in range(4):
                matches = [r for r in nonseasonal_d1 if r["order"][0] == p and r["order"][2] == q]
                if not matches:
                    cells.append(" - ")
                elif matches[0]["status"] == "converged":
                    cells.append(" ✓ ")
                else:
                    cells.append(" ✗ ")
            rows_ns.append(f"| {p}   |" + "|".join(cells) + "|")
        lines.append("\n".join(rows_ns))
    lines.append("")

    lines.append("### s=12 (P=0, Q=0 고정)\n")
    lines.append(build_heatmap(results, "s=12", range(3), range(3), P_fixed=0, Q_fixed=0))
    lines.append("")

    lines.append("### s=24 (P=0, Q=0 고정)\n")
    lines.append(build_heatmap(results, "s=24", range(3), range(3), P_fixed=0, Q_fixed=0))
    lines.append("")

    # 실패 모델 상세
    failures = [r for r in results if r["status"] != "converged"]
    if failures:
        lines.append("## 3. 실패 모델 상세\n")
        lines.append("| Group | Order | Seasonal | Status | Elapsed (s) |")
        lines.append("|-------|-------|----------|--------|------------:|")
        for r in failures[:50]:  # 상위 50개만
            e_str = f"{r['elapsed']:.2f}" if r["elapsed"] is not None else "N/A"
            lines.append(
                f"| {r['group']} | {r['order']} | {r['seasonal']} "
                f"| {r['status']} | {e_str} |"
            )
        if len(failures) > 50:
            lines.append(f"\n... 외 {len(failures) - 50}건 생략")
    else:
        lines.append("## 3. 실패 모델 상세\n")
        lines.append("실패 모델 없음 — 전체 수렴 100%\n")

    # 타이밍 통계
    lines.append("")
    lines.append("## 4. 소요 시간 통계\n")
    for g in groups:
        times = [r["elapsed"] for r in results if r["group"] == g and r["elapsed"] is not None]
        if times:
            lines.append(f"- **{g}**: p50={np.median(times):.3f}s, p95={np.percentile(times, 95):.3f}s, max={max(times):.3f}s")
    lines.append("")

    # 결론
    lines.append("## 5. 결론\n")
    lines.append(f"- 전체 수렴률: **{overall_rate:.1f}%** ({conv_all}/{total_all})")
    lines.append(f"- 합격 기준 (≥95%): **{'PASS' if overall_rate >= 95 else 'FAIL'}**")
    lines.append("")

    # 저장
    out_path = Path(__file__).resolve().parent.parent.parent / "ver5" / "result" / "convergence_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ 저장: {out_path}")
    print(f"  수렴률: {overall_rate:.1f}% ({conv_all}/{total_all})")
    print(f"  합격: {'PASS' if overall_rate >= 95 else 'FAIL'}")


if __name__ == "__main__":
    main()
