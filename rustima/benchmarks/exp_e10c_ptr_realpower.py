"""E10c ★ HEADLINE REAL: PTR vs joint MLE on Korean hourly electricity demand.

Real-data validation of the PTR algorithmic claim on 1-year and 2-year subsets
of power_demand_final.csv with temperature (ta) and humidity (hm) exog.

Compared engines:
- rustima lbfgsb (joint MLE)
- rustima profile-trust-region (PTR ★)
- statsmodels SARIMAX

Outputs:
- raw/e10c_realpower.csv
- tables/e10c_realpower.tex
- figures/e10c_loglik_gap.png
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import rustima
from statsmodels.tsa.statespace.sarimax import SARIMAX

from thesis_utils import FIGURES_DIR, Timer, csv_to_latex, save_csv, setup_mpl

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parent.parent.parent / "power_demand_final.csv"


def load_subset(years: int):
    """Load 'years' years of hourly data starting 2019-01-01."""
    df = pd.read_csv(DATA)
    df["일시"] = pd.to_datetime(df["일시"])
    start = pd.Timestamp("2019-01-01")
    end = start + pd.DateOffset(years=years)
    sub = df[(df["일시"] >= start) & (df["일시"] < end)].reset_index(drop=True)
    y = sub["power demand(MW)"].to_numpy(dtype=np.float64)
    ta = sub["ta"].to_numpy(dtype=np.float64)
    hm = sub["hm"].to_numpy(dtype=np.float64)
    # Fill NaN with column mean (simple imputation; data is mostly clean)
    for arr in [y, ta, hm]:
        m = np.isnan(arr)
        if m.any():
            arr[m] = np.nanmean(arr)
    X = np.column_stack([ta, hm]).astype(np.float64)
    return y, X


def measure_rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def fit_with_rss(fn, *args, **kwargs):
    """Run fit function while monitoring RSS."""
    base = measure_rss_mb()
    peak = base
    with Timer() as t:
        result = fn(*args, **kwargs)
    after = measure_rss_mb()
    peak = max(peak, after)
    return result, t.ms, peak - base


def fit_rustima(y, X, method, order, sorder, y_mean):
    """rustima fit. Returns dict with LL, AIC, β, time, ΔRSS."""
    y_c = y - y_mean
    base = measure_rss_mb()
    with Timer() as t:
        r = rustima.sarimax_fit(
            y_c, order, sorder,
            exog=X, method=method,
            concentrate_scale=True, maxiter=300,
        )
    drss = measure_rss_mb() - base
    # params layout: [exog(2), ar, ma, sar, sma]
    params = np.array(r["params"], dtype=float)
    n_exog = X.shape[1]
    beta_hat = params[:n_exog]
    return {
        "ll": r["loglike"],
        "aic": r["aic"],
        "beta_ta": float(beta_hat[0]),
        "beta_hm": float(beta_hat[1]),
        "n_iter": r.get("n_iter", -1),
        "converged": r["converged"],
        "method_used": r.get("method", method),
        "time_ms": t.ms,
        "drss_mb": drss,
    }


def fit_statsmodels(y, X, order, sorder, y_mean):
    y_c = y - y_mean
    base = measure_rss_mb()
    with Timer() as t:
        m = SARIMAX(y_c, order=order, seasonal_order=sorder, exog=X,
                    simple_differencing=False).fit(disp=False, maxiter=300)
    drss = measure_rss_mb() - base
    n_exog = X.shape[1]
    params = m.params.values if hasattr(m.params, "values") else np.array(m.params)
    beta_hat = params[:n_exog]
    return {
        "ll": m.llf, "aic": m.aic,
        "beta_ta": float(beta_hat[0]),
        "beta_hm": float(beta_hat[1]),
        "n_iter": m.mle_retvals.get("iterations", -1) if hasattr(m, "mle_retvals") else -1,
        "converged": True,
        "method_used": "statsmodels-lbfgsb",
        "time_ms": t.ms,
        "drss_mb": drss,
    }


ORDER = (1, 1, 1)
SORDER = (1, 0, 1, 24)


def main():
    print(f"Order: SARIMA({ORDER[0]},{ORDER[1]},{ORDER[2]})({SORDER[0]},{SORDER[1]},{SORDER[2]})_{SORDER[3]} + 2 exog (ta, hm)")
    rows = []

    for years in (1, 2):
        print(f"\n=== {years}-year subset (n = {8760 * years}) ===")
        y, X = load_subset(years)
        n = len(y)
        y_mean = y.mean()
        print(f"  Loaded n={n}, y_mean={y_mean:.1f}, ta_corr_hm={np.corrcoef(X[:,0], X[:,1])[0,1]:.3f}")

        for method, label in [("lbfgsb", "rustima lbfgsb"), ("profile-trust-region", "rustima PTR ★")]:
            try:
                r = fit_rustima(y, X, method, ORDER, SORDER, y_mean)
                rows.append([
                    f"{years}y", label,
                    f"{r['ll']:.2f}", f"{r['aic']:.2f}",
                    f"{r['beta_ta']:.4f}", f"{r['beta_hm']:.4f}",
                    r["n_iter"], r["method_used"],
                    f"{r['time_ms']:.0f}", f"{r['drss_mb']:.1f}",
                    "OK" if r["converged"] else "NOCONV",
                ])
                print(f"  {label}: LL={r['ll']:.2f}, AIC={r['aic']:.2f}, β=({r['beta_ta']:.3f}, {r['beta_hm']:.4f}), iter={r['n_iter']}, t={r['time_ms']:.0f}ms, ΔRSS={r['drss_mb']:.0f}MB")
            except Exception as e:
                rows.append([f"{years}y", label, "ERR", "—", "—", "—", "—", "—", "—", "—", str(e)[:30]])
                print(f"  {label}: ERR {e}")

        try:
            r = fit_statsmodels(y, X, ORDER, SORDER, y_mean)
            rows.append([
                f"{years}y", "statsmodels",
                f"{r['ll']:.2f}", f"{r['aic']:.2f}",
                f"{r['beta_ta']:.4f}", f"{r['beta_hm']:.4f}",
                r["n_iter"], r["method_used"],
                f"{r['time_ms']:.0f}", f"{r['drss_mb']:.1f}",
                "OK" if r["converged"] else "NOCONV",
            ])
            print(f"  statsmodels: LL={r['ll']:.2f}, AIC={r['aic']:.2f}, β=({r['beta_ta']:.3f}, {r['beta_hm']:.4f}), iter={r['n_iter']}, t={r['time_ms']:.0f}ms, ΔRSS={r['drss_mb']:.0f}MB")
        except Exception as e:
            rows.append([f"{years}y", "statsmodels", "ERR", "—", "—", "—", "—", "—", "—", "—", str(e)[:30]])
            print(f"  statsmodels: ERR {e}")

    save_csv(
        "e10c_realpower.csv",
        ["window", "engine", "loglike", "aic", "beta_ta", "beta_hm", "n_iter", "method_used", "time_ms", "drss_mb", "status"],
        rows,
    )
    csv_to_latex(
        "e10c_realpower.csv",
        "e10c_realpower.tex",
        f"PTR vs.\\ joint MLE on Korean hourly electricity demand with temperature ($ta$) and humidity ($hm$) exogenous regressors. Specification: SARIMAX({ORDER[0]},{ORDER[1]},{ORDER[2]})({SORDER[0]},{SORDER[1]},{SORDER[2]})$_{{{SORDER[3]}}}$. Both 1-year ($n=8{{,}}760$) and 2-year ($n=17{{,}}520$) training windows.",
        "tab:rustima:e10c-realpower",
    )

    # Figure: LL bars per window per engine
    plt = setup_mpl()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    windows = sorted(set(r[0] for r in rows if r[2] != "ERR"))
    engines = ["rustima lbfgsb", "rustima PTR ★", "statsmodels"]
    colors = ["#888888", "C0", "#bb6666"]
    x = np.arange(len(windows))
    width = 0.27
    for i, eng in enumerate(engines):
        lls = []
        for w in windows:
            match = [r for r in rows if r[0] == w and r[1] == eng and r[2] != "ERR"]
            lls.append(float(match[0][2]) if match else np.nan)
        ax.bar(x + (i - 1) * width, lls, width, label=eng, color=colors[i])
        for j, v in enumerate(lls):
            if np.isfinite(v):
                ax.text(x[j] + (i - 1) * width, v + abs(v) * 0.001, f"{v:.0f}", ha="center", fontsize=7, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels(windows)
    ax.set_ylabel("Log-likelihood")
    ax.set_title("rustima lbfgsb vs PTR vs statsmodels — real Korean hourly demand")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(FIGURES_DIR / "e10c_loglik_gap.png")
    plt.close(fig)
    print("\nFigure saved.")


if __name__ == "__main__":
    main()
