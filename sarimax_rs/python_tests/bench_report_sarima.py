"""
bench_report_sarima.py — SARIMA vs statsmodels 종합 비교 리포트

측정 항목:
  - 수렴률, 우도(loglike), AIC, BIC
  - 파라미터 유사도 (MAE vs statsmodels) [subset only]
  - 우도 유사도 (Δloglike)               [subset only]
  - 속도 비교 (sarimax_rs vs statsmodels)
  - h=6 예측 RMSE 비교

범위:
  - 전체 매트릭스: p,q,P,Q ∈ [0..5], d=1, D=1 per s=7,12,24
  - statsmodels 비교 서브셋: p,q,P,Q ∈ [0..2]

출력: ver5/result_v1/sarima_report.md  (주기별 섹션 포함)
"""

import math
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, "python")
import sarimax_rs

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX as SM_SARIMAX
    HAS_SM = True
except ImportError:
    HAS_SM = False

OUT = Path(__file__).resolve().parent.parent.parent / "ver5" / "result_v1" / "sarima_report.md"
OUT.parent.mkdir(parents=True, exist_ok=True)

N_OBS_MAP = {7: 500, 12: 300, 24: 500}
STEPS     = 6
SEED      = 42
SM_TOL_PARAMS   = 0.10
SM_TOL_LOGLIKE  = 5.0
SM_SUBSET_MAX   = 2   # statsmodels 비교는 p,q,P,Q ≤ 2 서브셋만
MAX_P = 5; MAX_Q = 5   # p, q ∈ [0..5]
MAX_PP = 4; MAX_QQ = 4  # P, Q ∈ [0..4] (engine hard limit)


# ─── 데이터 생성 ──────────────────────────────────────────────────────────────

def gen_seasonal(n, s, seed=SEED):
    """Simple seasonal ARIMA(1,1,1)(1,1,1)_s DGP."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n + s + 2)
    e = rng.normal(size=n + s + 2)
    for i in range(s + 2, n + s + 2):
        y[i] = (y[i-1] - y[i-s] + y[i-s-1]
                + 0.4 * (y[i-1] - y[i-2])
                - 0.3 * e[i-1]
                + 0.3 * (y[i-s] - y[i-s-1])
                + e[i])
    return y[s+2:]


# ─── 단일 모델 실행 ───────────────────────────────────────────────────────────

def run_rs(y, order, seasonal):
    t0 = time.perf_counter()
    try:
        r = sarimax_rs.sarimax_fit(y, order, seasonal)
    except Exception:
        r = {"converged": False, "loglike": float("nan"), "aic": float("nan"),
             "bic": float("nan"), "params": []}
    return r, time.perf_counter() - t0


def run_sm(y, order, seasonal):
    if not HAS_SM:
        return None, None
    try:
        t0  = time.perf_counter()
        m   = SM_SARIMAX(y, order=order, seasonal_order=seasonal,
                         enforce_stationarity=False, enforce_invertibility=False)
        res = m.fit(disp=False, maxiter=200)
        return res, time.perf_counter() - t0
    except Exception:
        return None, None


def forecast_rmse(y, order, seasonal, h=STEPS):
    origins = [int(len(y)*0.7), int(len(y)*0.8), int(len(y)*0.9)]
    errs = []
    for orig in origins:
        train, actual = y[:orig], y[orig:orig+h]
        if len(actual) < h:
            continue
        r2 = sarimax_rs.sarimax_fit(train, order, seasonal)
        if not r2["converged"]:
            continue
        fc = sarimax_rs.sarimax_forecast(
            train, order, seasonal, np.array(r2["params"]), steps=h
        )
        errs.extend((np.array(fc["mean"]) - actual)**2)
    return math.sqrt(np.mean(errs)) if errs else float("nan")


# ─── 주기별 벤치마크 ──────────────────────────────────────────────────────────

def bench_season(s):
    n_obs = N_OBS_MAP[s]
    y = gen_seasonal(n_obs, s)

    # 전체 매트릭스: p,q ∈ [0..5], P,Q ∈ [0..4]
    combos_full = [(p, 1, q, P, 1, Q, s)
                   for p, q, P, Q in product(range(MAX_P+1), range(MAX_Q+1),
                                             range(MAX_PP+1), range(MAX_QQ+1))]
    # statsmodels 서브셋: p,q,P,Q ∈ [0..2]
    combos_sm   = {(p, 1, q, P, 1, Q, s)
                   for p, q, P, Q in product(range(SM_SUBSET_MAX+1), repeat=4)}

    rows = []
    total = len(combos_full)
    print(f"  s={s}: {total} combos...", flush=True)

    for idx, combo in enumerate(combos_full):
        p, d, q, P, D, Q, ss = combo
        order    = (p, d, q)
        seasonal = (P, D, Q, ss)

        rs_r, rs_t = run_rs(y, order, seasonal)
        conv = rs_r.get("converged", False)
        ll_rs  = rs_r.get("loglike", float("nan")) if conv else float("nan")
        aic_rs = rs_r.get("aic",     float("nan")) if conv else float("nan")
        bic_rs = rs_r.get("bic",     float("nan")) if conv else float("nan")

        ll_sm = aic_sm = bic_sm = mae_par = float("nan")
        sm_t_val = float("nan")

        if combo in combos_sm:
            sm_r, sm_t2 = run_sm(y, order, seasonal)
            if sm_r is not None:
                try:
                    ll_sm  = sm_r.llf
                    aic_sm = sm_r.aic
                    bic_sm = sm_r.bic
                    sm_t_val = sm_t2 * 1000
                    n_p = p + q + P + Q
                    if n_p > 0 and conv and len(rs_r["params"]) == n_p:
                        sm_p = sm_r.params[:n_p]
                        mae_par = float(np.mean(np.abs(np.array(rs_r["params"]) - sm_p)))
                except Exception:
                    pass

        delta_ll  = abs(ll_rs - ll_sm)  if (math.isfinite(ll_rs)  and math.isfinite(ll_sm))  else float("nan")
        delta_aic = abs(aic_rs - aic_sm) if (math.isfinite(aic_rs) and math.isfinite(aic_sm)) else float("nan")

        rows.append({
            "order": order, "seasonal": seasonal, "conv": conv,
            "ll_rs": ll_rs, "ll_sm": ll_sm, "delta_ll": delta_ll,
            "aic_rs": aic_rs, "aic_sm": aic_sm, "delta_aic": delta_aic,
            "bic_rs": bic_rs, "bic_sm": bic_sm,
            "mae_par": mae_par,
            "spd_rs": rs_t * 1000,
            "spd_sm": sm_t_val,
        })

        if (idx + 1) % 200 == 0:
            print(f"    {idx+1}/{total} done", flush=True)

    return rows


# ─── 섹션 생성 ────────────────────────────────────────────────────────────────

def section_for_season(s, rows):
    conv_rows = [r for r in rows if r["conv"]]
    n_total   = len(rows)
    n_conv    = len(conv_rows)

    sorted_rows = sorted(conv_rows, key=lambda r: r["aic_rs"] if math.isfinite(r["aic_rs"]) else 1e18)
    best5       = sorted_rows[:5]

    # RMSE (best-5)
    rmse_rows = []
    for r in best5:
        rmse = forecast_rmse(
            gen_seasonal(N_OBS_MAP[s], s), r["order"], r["seasonal"]
        )
        rmse_rows.append((r["order"], r["seasonal"], rmse))

    rs_times  = [r["spd_rs"] for r in rows if math.isfinite(r["spd_rs"])]
    sm_times  = [r["spd_sm"] for r in rows if math.isfinite(r["spd_sm"])]
    delta_lls = [r["delta_ll"]  for r in conv_rows if math.isfinite(r["delta_ll"])]
    delta_aics= [r["delta_aic"] for r in conv_rows if math.isfinite(r["delta_aic"])]
    mae_pars  = [r["mae_par"]   for r in conv_rows if math.isfinite(r["mae_par"])]

    lines = []
    lines += [
        f"## s={s} 섹션 (n={N_OBS_MAP[s]})\n",
        f"### {s}-1. 수렴률\n",
        "| 항목 | 값 |",
        "|------|-----|",
        f"| 전체 조합 (p,q,P,Q ∈ [0..5], d=1, D=1) | {n_total} |",
        f"| 수렴 | {n_conv} ({n_conv/n_total*100:.1f}%) |",
        f"| 미수렴 | {n_total-n_conv} |",
        "",
    ]

    lines += [
        f"### {s}-2. 속도 비교 (ms/model)\n",
        "| Engine | p50 | p95 | max |",
        "|--------|----:|----:|----:|",
    ]
    if rs_times:
        lines.append(f"| sarimax_rs  | {np.median(rs_times):.2f} | {np.percentile(rs_times,95):.2f} | {max(rs_times):.2f} |")
    if sm_times:
        lines.append(f"| statsmodels | {np.median(sm_times):.2f} | {np.percentile(sm_times,95):.2f} | {max(sm_times):.2f} |")
    if rs_times and sm_times:
        spd = np.median(sm_times) / np.median(rs_times)
        lines.append(f"\nsarimax_rs는 statsmodels 대비 **{spd:.1f}x** 빠름 (p50 기준, subset only)\n")
    lines.append("")

    lines += [
        f"### {s}-3. statsmodels 유사도 (서브셋: p,q,P,Q ≤ {SM_SUBSET_MAX})\n",
        "| 지표 | 평균 | 중앙값 | p95 | 합격률 |",
        "|------|-----:|------:|----:|------:|",
    ]
    if delta_lls:
        pr = sum(1 for v in delta_lls if v < SM_TOL_LOGLIKE) / len(delta_lls) * 100
        lines.append(f"| Δloglike | {np.mean(delta_lls):.3f} | {np.median(delta_lls):.3f} | {np.percentile(delta_lls,95):.3f} | {pr:.1f}% |")
    if delta_aics:
        lines.append(f"| ΔAIC     | {np.mean(delta_aics):.3f} | {np.median(delta_aics):.3f} | {np.percentile(delta_aics,95):.3f} | - |")
    if mae_pars:
        pr_p = sum(1 for v in mae_pars if v < SM_TOL_PARAMS) / len(mae_pars) * 100
        lines.append(f"| Param MAE | {np.mean(mae_pars):.4f} | {np.median(mae_pars):.4f} | {np.percentile(mae_pars,95):.4f} | {pr_p:.1f}% |")
    lines.append("")

    lines += [
        f"### {s}-4. Top-20 모델 (AIC 기준)\n",
        "| Rank | Order | Seasonal | loglike | AIC(rs) | AIC(sm) | ΔAIC | Param MAE | Speed(ms) |",
        "|-----:|-------|---------|--------:|--------:|--------:|-----:|----------:|----------:|",
    ]
    for i, r in enumerate(sorted_rows[:20]):
        aic_sm_s = f"{r['aic_sm']:.1f}" if math.isfinite(r['aic_sm']) else "-"
        daics    = f"{r['delta_aic']:.2f}" if math.isfinite(r['delta_aic']) else "-"
        mae_s    = f"{r['mae_par']:.4f}"   if math.isfinite(r['mae_par'])   else "-"
        lines.append(
            f"| {i+1} | ARIMA{r['order']} | {r['seasonal']} | {r['ll_rs']:.2f} | {r['aic_rs']:.1f} | {aic_sm_s} | {daics} | {mae_s} | {r['spd_rs']:.2f} |"
        )
    lines.append("")

    if rmse_rows:
        lines += [
            f"### {s}-5. 예측 RMSE (Top-5 AIC, h={STEPS})\n",
            "| Order | Seasonal | RMSE |",
            "|-------|---------|-----:|",
        ]
        for order, seasonal, rmse in rmse_rows:
            rmse_s = f"{rmse:.4f}" if math.isfinite(rmse) else "-"
            lines.append(f"| ARIMA{order} | {seasonal} | {rmse_s} |")
        lines.append("")

    if sorted_rows:
        best = sorted_rows[0]
        lines += [
            f"### {s}-6. 결론\n",
            f"- 수렴률: **{n_conv/n_total*100:.1f}%** ({n_conv}/{n_total})",
            f"- 최적 모델 (AIC): **SARIMA{best['order']}×{best['seasonal']}** (AIC={best['aic_rs']:.1f})",
        ]
        if delta_lls:
            pl = sum(1 for v in delta_lls if v < SM_TOL_LOGLIKE)
            lines.append(f"- Δloglike < {SM_TOL_LOGLIKE} 합격: **{pl}/{len(delta_lls)}** ({pl/len(delta_lls)*100:.1f}%)")
        lines.append("")

    return lines


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    all_lines = [
        "# SARIMA 종합 비교 리포트\n",
        f"> 생성: {datetime.now():%Y-%m-%d %H:%M}  ",
        f"> sarimax_rs v{sarimax_rs.version()}  ",
        f"> statsmodels: {'설치됨' if HAS_SM else '미설치'}  ",
        f"> 전체 매트릭스: p,q ∈ [0..{MAX_P}], P,Q ∈ [0..{MAX_PP}], d=1, D=1  ",
        f"> statsmodels 서브셋: p,q,P,Q ∈ [0..{SM_SUBSET_MAX}]  ",
        "",
    ]

    season_summaries = []
    for s in [7, 12, 24]:
        print(f"[SARIMA s={s}] Starting...", flush=True)
        rows = bench_season(s)
        section = section_for_season(s, rows)
        all_lines.extend(section)
        n_conv = sum(1 for r in rows if r["conv"])
        season_summaries.append((s, n_conv, len(rows)))

    # 전체 요약
    all_lines += [
        "## 전체 요약\n",
        "| s | 전체 조합 | 수렴 | 수렴률 |",
        "|---|----------:|-----:|------:|",
    ]
    for s, nc, nt in season_summaries:
        all_lines.append(f"| {s} | {nt} | {nc} | {nc/nt*100:.1f}% |")
    all_lines.append("")

    OUT.write_text("\n".join(all_lines), encoding="utf-8")
    print(f"\n✓ 저장: {OUT}")
    for s, nc, nt in season_summaries:
        print(f"  s={s}: {nc}/{nt} ({nc/nt*100:.1f}%) 수렴")


if __name__ == "__main__":
    main()
