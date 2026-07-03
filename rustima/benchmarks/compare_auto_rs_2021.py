"""rustima auto_arima on KPX 2021-2023 windows (matches slide Table 3.29+3.31).

1yr: 2021, 2yr: 2021-2022, 3yr: 2021-2023.
Settings mirror compare_auto_R_2021.R: xreg=[hm, ta], s=24, stepwise,
ic=aic, trend='n', simple_differencing=True (R-comparable pre-differencing).

Also tracks per-horizon peak RSS via resource.getrusage so we get reliable
per-horizon memory numbers (overall peak also available via /usr/bin/time -l).
"""
from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rustima" / "python"))

from rustima.auto import auto_arima  # noqa: E402

DATA = ROOT / "power_demand_final.csv"
OUT_CSV = ROOT / "rustima" / "docs" / "auto_arima_rs_2021.csv"


def _peak_rss_mb():
    """Peak RSS so far for this process, in MB. macOS reports bytes."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS: ru_maxrss is bytes; Linux: kilobytes. Detect by magnitude.
    if r > 1 << 30:  # > 1 GB -> bytes (macOS)
        return r / (1024 * 1024)
    return r / 1024  # kB -> MB


def load_window(start: str, end: str):
    df = pd.read_csv(DATA, parse_dates=["일시"])
    df = df[(df["일시"] >= start) & (df["일시"] < end)].reset_index(drop=True)
    y = df["power demand(MW)"].astype(float).to_numpy()
    X = df[["hm", "ta"]].astype(float).to_numpy()
    return y, X


def fit_horizon(label: str, start: str, end: str):
    y, X = load_window(start, end)
    print(f"\n=== [{label}] {start} .. {end}  n_obs={len(y)} ===", flush=True)
    rss_before = _peak_rss_mb()
    t0 = time.time()
    res = auto_arima(
        endog=y,
        exog=X,
        s=24,
        max_p=5, max_q=5,
        max_P=2, max_Q=2,
        max_D=1,
        trend="n",
        criterion="aic",
        stepwise=True,
        simple_differencing=True,
        trace=False,
    )
    dt = time.time() - t0
    rss_after = _peak_rss_mb()
    fit = res.result
    order = res.order
    sord = res.seasonal_order
    sigma2 = getattr(fit, "sigma2", None)
    if sigma2 is None:
        sigma2 = float(getattr(fit, "scale", float("nan")))
    print(
        f"[{label}] selected: {order}{sord}  "
        f"LL={fit.llf:.4f}  AIC={fit.aic:.4f}  BIC={fit.bic:.4f}  "
        f"sigma2={sigma2:.6f}  runtime={dt:.2f}s  evals={len(res.history)}  "
        f"peak_rss_mb={rss_after:.1f} (Δ {rss_after - rss_before:+.1f})",
        flush=True,
    )
    return {
        "horizon": label,
        "n_obs": len(y),
        "order": f"({order[0]},{order[1]},{order[2]})",
        "seasonal": f"({sord[0]},{sord[1]},{sord[2]})[{sord[3]}]",
        "loglik": float(fit.llf),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "sigma2": float(sigma2),
        "runtime_s": dt,
        "peak_rss_mb": rss_after,
        "n_models_evaluated": len(res.history),
        "status": "ok",
    }


def main():
    horizons = [
        ("1yr", "2021-01-01", "2022-01-01"),
        ("2yr", "2021-01-01", "2023-01-01"),
        ("3yr", "2021-01-01", "2024-01-01"),
    ]
    rows = []
    for label, start, end in horizons:
        try:
            rows.append(fit_horizon(label, start, end))
        except Exception as e:
            print(f"[{label}] FAILED: {type(e).__name__}: {e}", flush=True)
            rows.append({"horizon": label, "status": f"failed: {e}"})
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        print(f"wrote partial {OUT_CSV}", flush=True)

    print("\n--- summary ---")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
