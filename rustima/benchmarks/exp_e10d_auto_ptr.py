"""E10d: auto_arima version — rustima auto with lbfgsb vs PTR vs pmdarima.

Tests whether the PTR optimization path affects automatic model selection
on real Korean hourly electricity demand with ta+hm exog.

Outputs:
- raw/e10d_auto_ptr.csv
- tables/e10d_auto_ptr.tex
- figures/e10d_auto_ptr.png
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

from thesis_utils import FIGURES_DIR, Timer, csv_to_latex, save_csv, setup_mpl

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parent.parent.parent / "power_demand_final.csv"


def load_subset(years: int):
    df = pd.read_csv(DATA)
    df["일시"] = pd.to_datetime(df["일시"])
    start = pd.Timestamp("2019-01-01")
    end = start + pd.DateOffset(years=years)
    sub = df[(df["일시"] >= start) & (df["일시"] < end)].reset_index(drop=True)
    y = sub["power demand(MW)"].to_numpy(dtype=np.float64)
    ta = sub["ta"].to_numpy(dtype=np.float64)
    hm = sub["hm"].to_numpy(dtype=np.float64)
    for arr in [y, ta, hm]:
        m = np.isnan(arr)
        if m.any():
            arr[m] = np.nanmean(arr)
    X = np.column_stack([ta, hm]).astype(np.float64)
    return y, X


def measure_rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def run_auto(y, X, method, max_p=2, max_q=2, max_P=1, max_Q=1):
    """Run rustima auto_arima with a given optimizer method."""
    y_c = y - y.mean()
    base = measure_rss_mb()
    with Timer() as t:
        r = rustima.auto_arima(
            y_c, exog=X,
            s=24,
            max_p=max_p, max_q=max_q, max_d=1,
            max_P=max_P, max_Q=max_Q, max_D=1,
            stepwise=True,
            criterion="aic",
            method=method,
            maxiter=200,
            trace=False,
        )
    drss = measure_rss_mb() - base
    aic = getattr(r, "best_ic", None)
    if aic is None and hasattr(r, "result"):
        aic = getattr(r.result, "aic", None)
    llf = None
    if hasattr(r, "result"):
        llf = getattr(r.result, "llf", None) or getattr(r.result, "loglike", None)
    return {
        "order": r.order if hasattr(r, "order") else getattr(r, "best_order", None),
        "seasonal": r.seasonal_order if hasattr(r, "seasonal_order") else getattr(r, "best_seasonal", None),
        "aic": aic,
        "ll": llf,
        "n_models": (r.history_dataframe().shape[0] if hasattr(r, "history_dataframe") else None),
        "time_s": t.t,
        "drss_mb": drss,
    }


def run_pmdarima(y, X):
    try:
        import pmdarima as pm
    except ImportError:
        return None
    y_c = y - y.mean()
    base = measure_rss_mb()
    err = None
    r = None
    with Timer() as t:
        try:
            r = pm.auto_arima(
                y_c, exogenous=X,
                m=24,
                max_p=2, max_q=2, max_d=1,
                max_P=1, max_Q=1, max_D=1,
                stepwise=True,
                seasonal=True,
                error_action="ignore",
                suppress_warnings=True,
                maxiter=200,
            )
        except Exception as e:
            err = str(e)[:80]
    if err is not None:
        return {"error": err, "time_s": t.t}
    drss = measure_rss_mb() - base
    aic = float(r.aic())
    # Guard: with error_action="ignore" pmdarima can return a numerically
    # degenerate model whose AIC is non-finite or implausibly small for the
    # sample size (e.g. AIC=10 for n>10000). Treat that as a divergence rather
    # than recording the garbage value as a valid result.
    if (not np.isfinite(aic)) or aic < 100.0:
        return {"error": f"diverged (AIC={aic:.1f} invalid)", "time_s": t.t}
    return {
        "order": tuple(r.order),
        "seasonal": tuple(r.seasonal_order),
        "aic": aic,
        "ll": None,
        "n_models": None,
        "time_s": t.t,
        "drss_mb": drss,
    }


def main():
    rows = []
    for years in (1, 2):
        print(f"\n=== {years}-year subset ===")
        y, X = load_subset(years)
        n = len(y)
        print(f"  n={n}")

        for method, label in [("lbfgsb", "rustima auto (lbfgsb)"), ("profile-trust-region", "rustima auto (PTR ★)")]:
            try:
                r = run_auto(y, X, method)
                aic_str = f"{r['aic']:.1f}" if r["aic"] is not None else "—"
                rows.append([
                    f"{years}y", label,
                    f"{r['time_s']:.1f}", f"{r['drss_mb']:.0f}",
                    str(r["order"]) if r["order"] else "?",
                    str(r["seasonal"]) if r["seasonal"] else "?",
                    aic_str,
                    str(r["n_models"]) if r["n_models"] else "—",
                ])
                print(f"  {label}: order={r['order']}, sorder={r['seasonal']}, AIC={aic_str}, time={r['time_s']:.1f}s, ΔRSS={r['drss_mb']:.0f}MB, n_models={r['n_models']}")
            except Exception as e:
                rows.append([f"{years}y", label, "ERR", "—", "—", "—", "—", str(e)[:30]])
                print(f"  {label}: ERR {e}")

        pmd = run_pmdarima(y, X)
        if pmd is None:
            rows.append([f"{years}y", "pmdarima", "—", "—", "—", "—", "—", "not installed"])
        elif "error" in pmd:
            rows.append([f"{years}y", "pmdarima", f"{pmd['time_s']:.1f}", "—", "—", "—", "—", pmd["error"]])
        else:
            rows.append([
                f"{years}y", "pmdarima",
                f"{pmd['time_s']:.1f}", f"{pmd['drss_mb']:.0f}",
                str(pmd["order"]), str(pmd["seasonal"]),
                f"{pmd['aic']:.1f}", "—",
            ])
            print(f"  pmdarima: order={pmd['order']}, sorder={pmd['seasonal']}, AIC={pmd['aic']:.1f}, time={pmd['time_s']:.1f}s, ΔRSS={pmd['drss_mb']:.0f}MB")

    save_csv(
        "e10d_auto_ptr.csv",
        ["window", "engine", "time_s", "drss_mb", "order", "seasonal", "aic", "n_models"],
        rows,
    )
    csv_to_latex(
        "e10d_auto_ptr.csv",
        "e10d_auto_ptr.tex",
        "Automatic SARIMAX order selection comparison on Korean hourly electricity demand with $ta$ and $hm$ exogenous regressors. Both rustima paths (lbfgsb and PTR) use the same stepwise Hyndman--Khandakar search; the PTR path applies profile-likelihood elimination of $\\beta$ at each candidate fit.",
        "tab:rustima:e10d-auto-ptr",
    )

    # Figure: AIC and time per window per engine
    plt = setup_mpl()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    windows = sorted(set(r[0] for r in rows if r[2] != "ERR" and r[2] != "—"))
    engines = ["rustima auto (lbfgsb)", "rustima auto (PTR ★)", "pmdarima"]
    colors = ["#888888", "C0", "#bb6666"]
    x = np.arange(len(windows))
    width = 0.27

    for i, eng in enumerate(engines):
        aics = []
        times = []
        for w in windows:
            match = [r for r in rows if r[0] == w and r[1] == eng and r[6] not in ("—", "ERR")]
            aics.append(float(match[0][6]) if match else np.nan)
            times.append(float(match[0][2]) if match else np.nan)
        ax1.bar(x + (i - 1) * width, aics, width, label=eng, color=colors[i])
        ax2.bar(x + (i - 1) * width, times, width, label=eng, color=colors[i])

    ax1.set_xticks(x); ax1.set_xticklabels(windows)
    ax1.set_ylabel("AIC (lower = better)")
    ax1.set_title("AIC by engine")
    ax1.legend(fontsize=8, loc="lower right")

    ax2.set_xticks(x); ax2.set_xticklabels(windows)
    ax2.set_ylabel("Time (s, log scale)")
    ax2.set_yscale("log")
    ax2.set_title("Wall-clock time")
    ax2.legend(fontsize=8, loc="upper left")
    fig.suptitle("auto_arima comparison: rustima (lbfgsb / PTR) vs pmdarima", y=1.02)
    fig.savefig(FIGURES_DIR / "e10d_auto_ptr.png")
    plt.close(fig)
    print("\nFigure saved.")


if __name__ == "__main__":
    main()
