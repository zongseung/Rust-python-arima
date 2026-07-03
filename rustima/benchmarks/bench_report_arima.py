"""
bench_report_arima.py — ARIMA vs statsmodels 종합 비교 리포트

측정 항목:
  - 수렴률, 우도(loglike), AIC, BIC
  - 파라미터 유사도 (MAE vs statsmodels)
  - 우도 유사도 (Δloglike)
  - 속도 비교 (rustima vs statsmodels)
  - h=1,6 예측 RMSE 비교

범위: p,q ∈ [0..5], d=1 (36 조합)
출력: ver5/result_v1/arima_report.md
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

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX as SM_SARIMAX
    HAS_SM = True
except ImportError:
    HAS_SM = False

OUT = Path(__file__).resolve().parent.parent.parent / "ver5" / "result_v1" / "arima_report.md"
OUT.parent.mkdir(parents=True, exist_ok=True)

N_OBS  = 500
STEPS  = 6
SEED   = 42
SM_TOL_PARAMS   = 0.10   # MAE 합격 기준
SM_TOL_LOGLIKE  = 5.0    # |Δloglike| 합격 기준


# ─── 데이터 생성 ──────────────────────────────────────────────────────────────

def gen_arima_data(n=N_OBS, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    e = rng.normal(size=n)
    for i in range(2, n):
        y[i] = y[i-1] + 0.4*(y[i-1]-y[i-2]) - 0.3*e[i-1] + e[i]
    return y


# ─── 단일 모델 실행 ───────────────────────────────────────────────────────────

def run_rs(y, order):
    t0 = time.perf_counter()
    r = rustima.sarimax_fit(y, order, (0,0,0,0))
    elapsed = time.perf_counter() - t0
    return r, elapsed


def run_sm(y, order):
    if not HAS_SM:
        return None, None
    try:
        t0 = time.perf_counter()
        m = SM_SARIMAX(y, order=order, seasonal_order=(0,0,0,0),
                       enforce_stationarity=False, enforce_invertibility=False)
        res = m.fit(disp=False, maxiter=200)
        elapsed = time.perf_counter() - t0
        return res, elapsed
    except Exception:
        return None, None


def forecast_rmse(y, order, params, h=STEPS):
    """Rolling origin forecast RMSE (3 origins)."""
    origins = [int(len(y)*0.7), int(len(y)*0.8), int(len(y)*0.9)]
    errs = []
    for orig in origins:
        train, actual = y[:orig], y[orig:orig+h]
        if len(actual) < h:
            continue
        r2 = rustima.sarimax_fit(train, order, (0,0,0,0))
        if not r2["converged"]:
            continue
        fc = rustima.sarimax_forecast(
            train, order, (0,0,0,0), np.array(r2["params"]), steps=h
        )
        errs.extend((np.array(fc["mean"]) - actual)**2)
    return math.sqrt(np.mean(errs)) if errs else float("nan")


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    y = gen_arima_data()
    combos = [(p, 1, q) for p, q in product(range(6), range(6))]

    rows = []
    for order in combos:
        p, d, q = order
        rs_r, rs_t  = run_rs(y, order)
        sm_r, sm_t  = run_sm(y, order)

        conv = rs_r.get("converged", False)
        ll_rs  = rs_r.get("loglike", float("nan")) if conv else float("nan")
        aic_rs = rs_r.get("aic",     float("nan")) if conv else float("nan")
        bic_rs = rs_r.get("bic",     float("nan")) if conv else float("nan")

        ll_sm   = float("nan")
        aic_sm  = float("nan")
        bic_sm  = float("nan")
        mae_par = float("nan")
        spd_r   = rs_t * 1000  # ms

        if sm_r is not None:
            try:
                ll_sm  = sm_r.llf
                aic_sm = sm_r.aic
                bic_sm = sm_r.bic
                # 파라미터 MAE (sigma2 제외)
                n_p = p + q
                if n_p > 0 and len(rs_r["params"]) == n_p:
                    sm_p = sm_r.params[:n_p]
                    mae_par = float(np.mean(np.abs(np.array(rs_r["params"]) - sm_p)))
            except Exception:
                pass

        delta_ll  = abs(ll_rs - ll_sm)  if (math.isfinite(ll_rs)  and math.isfinite(ll_sm))  else float("nan")
        delta_aic = abs(aic_rs - aic_sm) if (math.isfinite(aic_rs) and math.isfinite(aic_sm)) else float("nan")

        rows.append({
            "order": order, "conv": conv,
            "ll_rs": ll_rs, "ll_sm": ll_sm, "delta_ll": delta_ll,
            "aic_rs": aic_rs, "aic_sm": aic_sm, "delta_aic": delta_aic,
            "bic_rs": bic_rs, "bic_sm": bic_sm,
            "mae_par": mae_par,
            "spd_rs": spd_r, "spd_sm": (sm_t*1000 if sm_t else float("nan")),
        })

    # 수렴 모델만 추출
    conv_rows = [r for r in rows if r["conv"]]
    n_total   = len(rows)
    n_conv    = len(conv_rows)

    # RMSE (Top-5 AIC 모델)
    sorted_rows = sorted(conv_rows, key=lambda r: r["aic_rs"] if math.isfinite(r["aic_rs"]) else 1e18)
    best5 = sorted_rows[:5]
    rmse_rows = []
    for r in best5:
        if r["conv"]:
            rmse = forecast_rmse(y, r["order"], r["ll_rs"])
            rmse_rows.append((r["order"], rmse))

    # ── 출력 ──────────────────────────────────────────────────────────────────
    lines = []
    lines += [
        "# ARIMA 종합 비교 리포트\n",
        f"> 생성: {datetime.now():%Y-%m-%d %H:%M}  ",
        f"> rustima v{rustima.version()}  ",
        f"> statsmodels: {'설치됨' if HAS_SM else '미설치'}  ",
        f"> 데이터: n={N_OBS}, ARIMA(1,1,1) DGP\n",
    ]

    # 1. 요약
    lines += [
        "## 1. 수렴률 요약\n",
        f"| 항목 | 값 |",
        f"|------|-----|",
        f"| 전체 조합 (p,q ∈ [0..5], d=1) | {n_total} |",
        f"| 수렴 | {n_conv} ({n_conv/n_total*100:.1f}%) |",
        f"| 미수렴 | {n_total-n_conv} |",
        "",
    ]

    # 2. 속도 비교
    rs_times = [r["spd_rs"] for r in rows if math.isfinite(r["spd_rs"])]
    sm_times = [r["spd_sm"] for r in rows if math.isfinite(r["spd_sm"])]
    lines += [
        "## 2. 속도 비교 (ms/model)\n",
        "| Engine | p50 | p95 | max |",
        "|--------|----:|----:|----:|",
    ]
    if rs_times:
        lines.append(f"| rustima | {np.median(rs_times):.2f} | {np.percentile(rs_times,95):.2f} | {max(rs_times):.2f} |")
    if sm_times:
        lines.append(f"| statsmodels | {np.median(sm_times):.2f} | {np.percentile(sm_times,95):.2f} | {max(sm_times):.2f} |")
    if rs_times and sm_times:
        spd = np.median(sm_times) / np.median(rs_times)
        lines.append(f"\nrustima는 statsmodels 대비 **{spd:.1f}x** 빠름 (p50 기준)\n")
    lines.append("")

    # 3. 우도·AIC 유사도
    delta_lls  = [r["delta_ll"]  for r in conv_rows if math.isfinite(r["delta_ll"])]
    delta_aics = [r["delta_aic"] for r in conv_rows if math.isfinite(r["delta_aic"])]
    mae_pars   = [r["mae_par"]   for r in conv_rows if math.isfinite(r["mae_par"])]
    lines += [
        "## 3. statsmodels 유사도\n",
        "| 지표 | 평균 | 중앙값 | p95 | 합격률 |",
        "|------|-----:|------:|----:|------:|",
    ]
    if delta_lls:
        pass_rate = sum(1 for v in delta_lls if v < SM_TOL_LOGLIKE) / len(delta_lls) * 100
        lines.append(f"| Δloglike | {np.mean(delta_lls):.3f} | {np.median(delta_lls):.3f} | {np.percentile(delta_lls,95):.3f} | {pass_rate:.1f}% |")
    if delta_aics:
        lines.append(f"| ΔAIC | {np.mean(delta_aics):.3f} | {np.median(delta_aics):.3f} | {np.percentile(delta_aics,95):.3f} | - |")
    if mae_pars:
        pass_rate_p = sum(1 for v in mae_pars if v < SM_TOL_PARAMS) / len(mae_pars) * 100
        lines.append(f"| Param MAE | {np.mean(mae_pars):.4f} | {np.median(mae_pars):.4f} | {np.percentile(mae_pars,95):.4f} | {pass_rate_p:.1f}% |")
    lines.append("")

    # 4. Top-20 by AIC
    lines += [
        "## 4. Top-20 모델 (AIC 기준)\n",
        "| Rank | Order | loglike | AIC(rs) | AIC(sm) | ΔAIC | Param MAE | Speed(ms) |",
        "|-----:|-------|--------:|--------:|--------:|-----:|----------:|----------:|",
    ]
    for i, r in enumerate(sorted_rows[:20]):
        delta_aic_str = f"{r['delta_aic']:.2f}" if math.isfinite(r['delta_aic']) else "-"
        mae_str       = f"{r['mae_par']:.4f}"   if math.isfinite(r['mae_par'])   else "-"
        sm_aic_str    = f"{r['aic_sm']:.1f}"    if math.isfinite(r['aic_sm'])    else "-"
        lines.append(
            f"| {i+1} | ARIMA{r['order']} | {r['ll_rs']:.2f} | {r['aic_rs']:.1f} | {sm_aic_str} | {delta_aic_str} | {mae_str} | {r['spd_rs']:.2f} |"
        )
    lines.append("")

    # 5. 전수 결과 테이블
    lines += [
        "## 5. 전수 결과 (p,q ∈ [0..5], d=1)\n",
        "| Order | Conv | loglike(rs) | loglike(sm) | Δloglike | AIC(rs) | AIC(sm) | Param MAE | Speed_rs(ms) | Speed_sm(ms) |",
        "|-------|------|------------:|------------:|---------:|--------:|--------:|----------:|-------------:|-------------:|",
    ]
    for r in rows:
        conv_str = "✓" if r["conv"] else "✗"
        ll_rs_s  = f"{r['ll_rs']:.2f}"   if math.isfinite(r['ll_rs'])   else "-"
        ll_sm_s  = f"{r['ll_sm']:.2f}"   if math.isfinite(r['ll_sm'])   else "-"
        dll_s    = f"{r['delta_ll']:.3f}" if math.isfinite(r['delta_ll']) else "-"
        aic_rs_s = f"{r['aic_rs']:.1f}"  if math.isfinite(r['aic_rs'])  else "-"
        aic_sm_s = f"{r['aic_sm']:.1f}"  if math.isfinite(r['aic_sm'])  else "-"
        mae_s    = f"{r['mae_par']:.4f}"  if math.isfinite(r['mae_par']) else "-"
        spd_sm_s = f"{r['spd_sm']:.1f}"  if math.isfinite(r['spd_sm'])  else "-"
        lines.append(
            f"| ARIMA{r['order']} | {conv_str} | {ll_rs_s} | {ll_sm_s} | {dll_s} | {aic_rs_s} | {aic_sm_s} | {mae_s} | {r['spd_rs']:.2f} | {spd_sm_s} |"
        )
    lines.append("")

    # 6. Top-5 예측 RMSE
    if rmse_rows:
        lines += [
            f"## 6. 예측 RMSE (Top-5 AIC 모델, h={STEPS})\n",
            "| Order | RMSE |",
            "|-------|-----:|",
        ]
        for order, rmse in rmse_rows:
            lines.append(f"| ARIMA{order} | {rmse:.4f} |")
        lines.append("")

    # 7. 결론
    pass_ll = sum(1 for v in delta_lls if v < SM_TOL_LOGLIKE)
    lines += [
        "## 7. 결론\n",
        f"- 전체 수렴률: **{n_conv/n_total*100:.1f}%** ({n_conv}/{n_total})",
        f"- Δloglike < {SM_TOL_LOGLIKE} 합격: **{pass_ll}/{len(delta_lls)}** ({pass_ll/len(delta_lls)*100:.1f}%)" if delta_lls else "",
        f"- 최적 모델 (AIC): **ARIMA{sorted_rows[0]['order']}** (AIC={sorted_rows[0]['aic_rs']:.1f})" if sorted_rows else "",
        "",
    ]

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ 저장: {OUT}")
    print(f"  수렴: {n_conv}/{n_total} ({n_conv/n_total*100:.1f}%)")
    if delta_lls:
        print(f"  Δloglike 평균: {np.mean(delta_lls):.3f}")


if __name__ == "__main__":
    main()
