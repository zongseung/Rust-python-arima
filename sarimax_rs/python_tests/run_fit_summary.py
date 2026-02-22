"""
sarimax_rs 모델 적합 요약 리포트 생성기
=========================================

다양한 ARIMA/SARIMA 모델을 적합하고 파라미터 추정치, 표준오차,
검정통계량, 정보기준(AIC/BIC), 수렴 상태를 한눈에 출력합니다.

실행 방법:
    .venv/bin/python python_tests/run_fit_summary.py

출력:
    - 콘솔: 모델별 파라미터 테이블
    - fit_summary_report.md: 마크다운 형식 리포트 (저장)
"""

import sys
import time
from pathlib import Path

import numpy as np

# sarimax_py 고수준 API 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from sarimax_py import SARIMAXModel


# ── 데이터 생성 ─────────────────────────────────────────────────────────────

def make_random_walk(n=300, seed=42):
    np.random.seed(seed)
    return np.cumsum(np.random.randn(n))

def make_stationary(n=300, seed=42):
    np.random.seed(seed)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.5 * y[t-1] + np.random.randn()
    return y

def make_seasonal(n=300, s=12, seed=42):
    np.random.seed(seed)
    y = np.zeros(n)
    for t in range(s, n):
        y[t] = 0.3 * y[t-1] + 0.4 * y[t-s] + np.random.randn()
    return np.cumsum(y)

def make_ar4(n=400, seed=42):
    np.random.seed(seed)
    eps = np.random.randn(n)
    dy = np.zeros(n)
    for t in range(4, n):
        dy[t] = 0.35*dy[t-1] + 0.20*dy[t-2] - 0.10*dy[t-3] + 0.05*dy[t-4] + eps[t]
    return np.cumsum(dy)

def make_hourly(n=600, s=24, seed=70):
    np.random.seed(seed)
    t = np.arange(n)
    seasonal = 8.0*np.sin(2*np.pi*t/s) + 3.0*np.sin(4*np.pi*t/s)
    eps = np.random.randn(n) * 0.5
    dy = np.zeros(n)
    for i in range(s, n):
        dy[i] = 0.25*dy[i-1] + 0.40*dy[i-s] + seasonal[i] + eps[i]
    return np.cumsum(dy)


# ── 포맷 유틸리티 ────────────────────────────────────────────────────────────

def star(p):
    """p-value → 유의성 별표"""
    if np.isnan(p): return "   "
    if p < 0.001:   return "***"
    if p < 0.01:    return " **"
    if p < 0.05:    return "  *"
    if p < 0.1:     return "  ."
    return "   "

def fmt(v, w=10, decimals=4):
    if np.isnan(v):
        return f"{'n/a':>{w}}"
    return f"{v:>{w}.{decimals}f}"

def sig_bar(p):
    """p-value 막대 시각화"""
    if np.isnan(p): return "░░░░░"
    if p < 0.001:   return "█████"
    if p < 0.01:    return "████░"
    if p < 0.05:    return "███░░"
    if p < 0.1:     return "██░░░"
    return "░░░░░"


# ── 핵심 리포트 함수 ─────────────────────────────────────────────────────────

def fit_and_report(label, y, order, seasonal_order=(0,0,0,0), inference="hessian"):
    """모델을 적합하고 요약 dict를 반환."""
    t0 = time.perf_counter()
    model = SARIMAXModel(
        y, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=True, enforce_invertibility=True,
    )
    result = model.fit()
    elapsed = (time.perf_counter() - t0) * 1000  # ms

    ps = result.parameter_summary(alpha=0.05, inference=inference)

    return {
        "label": label,
        "order": order,
        "seasonal_order": seasonal_order,
        "n": len(y),
        "k_states": result._raw.get("k_states", "?") if hasattr(result, "_raw") else "?",
        "loglike": result.llf,
        "aic": result.aic,
        "bic": result.bic,
        "scale": result.scale,
        "n_iter": result._raw["n_iter"] if hasattr(result, "_raw") else "?",
        "converged": result.converged,
        "method": result.method,
        "elapsed_ms": elapsed,
        "param_names": ps["name"],
        "coef":        ps["coef"],
        "std_err":     ps["std_err"],
        "z":           ps["z"],
        "p_value":     ps["p_value"],
        "ci_lower":    ps["ci_lower"],
        "ci_upper":    ps["ci_upper"],
        "inf_status":  ps.get("inference_status", "n/a"),
    }


def print_model_block(r, out=sys.stdout):
    """단일 모델 결과를 콘솔/파일에 출력."""
    p, d, q = r["order"]
    P, D, Q, s = r["seasonal_order"]
    if s > 0:
        model_str = f"SARIMA({p},{d},{q})({P},{D},{Q},{s})"
    else:
        model_str = f"ARIMA({p},{d},{q})"

    n_params = len(r["param_names"])
    conv_mark = "✓" if r["converged"] else "~"

    print(f"\n{'━'*72}", file=out)
    print(f"  {r['label']}  [{model_str}]", file=out)
    print(f"  n={r['n']}  params={n_params}  "
          f"loglike={r['loglike']:.3f}  AIC={r['aic']:.2f}  BIC={r['bic']:.2f}  "
          f"σ²={r['scale']:.5f}", file=out)
    print(f"  iter={r['n_iter']}  conv={conv_mark}  method={r['method']}  "
          f"fit={r['elapsed_ms']:.1f}ms  inference={r['inf_status']}", file=out)
    print(f"{'─'*72}", file=out)
    print(f"  {'Parameter':<18} {'Coef':>10} {'StdErr':>10} {'z':>8} "
          f"{'P>|z|':>8} {'[0.025':>9} {'0.975]':>9}  Sig  ▌", file=out)
    print(f"{'─'*72}", file=out)

    for i, name in enumerate(r["param_names"]):
        coef  = r["coef"][i]
        se    = r["std_err"][i]
        z     = r["z"][i]
        pv    = r["p_value"][i]
        cilo  = r["ci_lower"][i]
        cihi  = r["ci_upper"][i]
        print(
            f"  {name:<18}"
            f"{fmt(coef,10,4)}"
            f"{fmt(se,10,4)}"
            f"{fmt(z,8,3)}"
            f"{fmt(pv,8,4)}"
            f"{fmt(cilo,9,4)}"
            f"{fmt(cihi,9,4)}"
            f"  {star(pv)} {sig_bar(pv)}",
            file=out,
        )

    print(f"{'━'*72}", file=out)
    print(f"  Signif. codes: *** p<0.001  ** p<0.01  * p<0.05  . p<0.1", file=out)


def to_markdown_table(results):
    """모든 모델 결과를 마크다운 요약 테이블로 변환."""
    lines = []
    lines.append("## 전체 모델 요약 (fit overview)")
    lines.append("")
    lines.append("| 모델 | n | params | loglike | AIC | BIC | σ² | iter | conv | fit(ms) |")
    lines.append("|------|:-:|:------:|--------:|----:|----:|---:|:----:|:----:|--------:|")
    for r in results:
        p, d, q = r["order"]
        P, D, Q, s = r["seasonal_order"]
        if s > 0:
            m = f"SARIMA({p},{d},{q})({P},{D},{Q},{s})"
        else:
            m = f"ARIMA({p},{d},{q})"
        n_par = len(r["param_names"])
        conv = "✓" if r["converged"] else "~"
        lines.append(
            f"| {m} | {r['n']} | {n_par} | {r['loglike']:.2f} | "
            f"{r['aic']:.2f} | {r['bic']:.2f} | {r['scale']:.5f} | "
            f"{r['n_iter']} | {conv} | {r['elapsed_ms']:.1f} |"
        )
    return "\n".join(lines)


def to_markdown_detail(r):
    """단일 모델 파라미터 테이블 마크다운."""
    p, d, q = r["order"]
    P, D, Q, s = r["seasonal_order"]
    if s > 0:
        model_str = f"SARIMA({p},{d},{q})({P},{D},{Q},{s})"
    else:
        model_str = f"ARIMA({p},{d},{q})"

    lines = []
    lines.append(f"### {r['label']}  `{model_str}`")
    lines.append("")
    lines.append(f"| 항목 | 값 |")
    lines.append(f"|---|---|")
    lines.append(f"| n | {r['n']} |")
    lines.append(f"| loglike | {r['loglike']:.4f} |")
    lines.append(f"| AIC | {r['aic']:.2f} |")
    lines.append(f"| BIC | {r['bic']:.2f} |")
    lines.append(f"| σ² (scale) | {r['scale']:.6f} |")
    lines.append(f"| iteration | {r['n_iter']} |")
    lines.append(f"| converged | {'Yes' if r['converged'] else 'Near-converged'} |")
    lines.append(f"| method | {r['method']} |")
    lines.append(f"| fit time | {r['elapsed_ms']:.1f} ms |")
    lines.append(f"| inference | {r['inf_status']} |")
    lines.append("")
    lines.append("| Parameter | Coef | Std Err | z | P>\\|z\\| | [0.025 | 0.975] | Sig |")
    lines.append("|-----------|-----:|--------:|--:|---------:|-------:|-------:|:---:|")

    for i, name in enumerate(r["param_names"]):
        coef = r["coef"][i]
        se   = r["std_err"][i]
        z    = r["z"][i]
        pv   = r["p_value"][i]
        cilo = r["ci_lower"][i]
        cihi = r["ci_upper"][i]

        def fv(v, d=4):
            return f"{v:.{d}f}" if not np.isnan(v) else "n/a"

        lines.append(
            f"| `{name}` | {fv(coef)} | {fv(se)} | {fv(z,3)} | "
            f"{fv(pv)} | {fv(cilo)} | {fv(cihi)} | {star(pv).strip()} |"
        )
    lines.append("")
    return "\n".join(lines)


# ── 메인 ────────────────────────────────────────────────────────────────────

MODELS = [
    # (label, data_fn, order, seasonal_order)
    ("AR(1)",           make_stationary,  (1,0,0), (0,0,0,0)),
    ("AR(2)",           make_stationary,  (2,0,0), (0,0,0,0)),
    ("MA(1)",           make_stationary,  (0,0,1), (0,0,0,0)),
    ("ARMA(1,1)",       make_stationary,  (1,0,1), (0,0,0,0)),
    ("ARIMA(1,1,1)",    make_random_walk, (1,1,1), (0,0,0,0)),
    ("ARIMA(2,1,1)",    make_random_walk, (2,1,1), (0,0,0,0)),
    ("ARIMA(4,1,1)",    make_ar4,         (4,1,1), (0,0,0,0)),
    ("ARIMA(4,1,4)",    make_ar4,         (4,1,4), (0,0,0,0)),
    ("SARIMA(1,0,0)(1,0,0,4)",
                        lambda **kw: make_seasonal(s=4,  **kw),
                        (1,0,0), (1,0,0,4)),
    ("SARIMA(0,1,1)(0,1,1,12)",
                        lambda **kw: make_seasonal(s=12, **kw),
                        (0,1,1), (0,1,1,12)),
    ("SARIMA(1,1,1)(1,1,1,12)",
                        lambda **kw: make_seasonal(s=12, **kw),
                        (1,1,1), (1,1,1,12)),
    ("SARIMA(4,1,1)(2,1,1,12)",
                        lambda **kw: make_seasonal(s=12, **kw),
                        (4,1,1), (2,1,1,12)),
    ("SARIMA(2,1,1)(2,1,1,24)",
                        make_hourly,
                        (2,1,1), (2,1,1,24)),
]

DATA_KWARGS = {
    "make_stationary":  {"n": 300, "seed": 42},
    "make_random_walk": {"n": 300, "seed": 42},
    "make_ar4":         {"n": 400, "seed": 42},
    "<lambda>":         {"n": 300, "seed": 42},
    "make_hourly":      {"n": 600, "seed": 70},
}


def main():
    print("=" * 72)
    print("  sarimax-rs  Fitting Summary Report")
    print("  inference: Hessian-based (numerical, central difference)")
    print("=" * 72)

    results = []
    for label, data_fn, order, seas in MODELS:
        fn_name = data_fn.__name__ if hasattr(data_fn, "__name__") else "<lambda>"
        kw = DATA_KWARGS.get(fn_name, {"n": 300, "seed": 42})
        print(f"\n  Fitting {label} ...", end=" ", flush=True)
        try:
            y = data_fn(**kw)
            r = fit_and_report(label, y, order, seas, inference="hessian")
            results.append(r)
            print(f"done ({r['elapsed_ms']:.1f}ms, "
                  f"loglike={r['loglike']:.2f}, AIC={r['aic']:.2f})")
        except Exception as e:
            print(f"ERROR: {e}")

    # ── 콘솔 상세 출력 ──────────────────────────────────────────────────────
    print("\n")
    for r in results:
        print_model_block(r)

    # ── 마크다운 파일 저장 ───────────────────────────────────────────────────
    out_path = Path(__file__).resolve().parent.parent / "fit_summary_report.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# sarimax-rs Fitting Summary Report\n\n")
        f.write("> Inference: Hessian-based (numerical central difference)  \n")
        f.write("> Significance: `***` p<0.001  `**` p<0.01  `*` p<0.05  `.` p<0.1\n\n")
        f.write("---\n\n")
        f.write(to_markdown_table(results))
        f.write("\n\n---\n\n")
        f.write("## 모델별 파라미터 상세\n\n")
        for r in results:
            f.write(to_markdown_detail(r))
            f.write("\n---\n\n")

    print(f"\n\n  ✓ 마크다운 리포트 저장: {out_path}")
    print(f"  총 {len(results)}개 모델 적합 완료")


if __name__ == "__main__":
    main()
