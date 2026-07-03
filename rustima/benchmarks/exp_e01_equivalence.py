"""E1: Likelihood equivalence — rustima vs statsmodels across orders.

Outputs:
- raw/e01_equivalence.csv
- tables/e01_equivalence.tex (Tab-cross-1)
"""
from __future__ import annotations

import time

import numpy as np
import rustima
from statsmodels.tsa.statespace.sarimax import SARIMAX

from thesis_utils import Timer, csv_to_latex, gen_random_sarimax, save_csv

ORDERS = [
    ((1, 0, 0), (0, 0, 0, 0), "AR(1)"),
    ((0, 0, 1), (0, 0, 0, 0), "MA(1)"),
    ((1, 0, 1), (0, 0, 0, 0), "ARMA(1,1)"),
    ((2, 0, 1), (0, 0, 0, 0), "ARMA(2,1)"),
    ((1, 1, 1), (0, 0, 0, 0), "ARIMA(1,1,1)"),
    ((1, 1, 1), (1, 0, 0, 12), "SARIMA(1,1,1)(1,0,0)_12"),
    ((1, 1, 1), (1, 1, 1, 12), "SARIMA(1,1,1)(1,1,1)_12"),
    ((1, 1, 1), (1, 1, 1, 24), "SARIMA(1,1,1)(1,1,1)_24"),
]
N_TRIALS = 5  # per order
N = 500


def main():
    rows = []
    for order, sorder, name in ORDERS:
        ll_diffs, time_rs, time_sm = [], [], []
        for trial in range(N_TRIALS):
            y, _, _ = gen_random_sarimax(N, *order[:3], *sorder[:4], seed=trial)
            try:
                with Timer() as t_rs:
                    r = rustima.sarimax_fit(y, order, sorder)
                ll_rs = r["loglike"]
                with Timer() as t_sm:
                    sm = SARIMAX(y, order=order, seasonal_order=sorder, simple_differencing=False).fit(disp=False, maxiter=100)
                ll_sm = sm.llf
                ll_diffs.append(abs(ll_rs - ll_sm))
                time_rs.append(t_rs.ms)
                time_sm.append(t_sm.ms)
            except Exception as e:
                print(f"  [skip] {name} trial {trial}: {e}")
                continue
        if not ll_diffs:
            continue
        rows.append([
            name,
            f"{np.median(ll_diffs):.4e}",
            f"{np.max(ll_diffs):.4e}",
            f"{np.median(time_rs):.1f}",
            f"{np.median(time_sm):.1f}",
            f"{np.median(time_sm) / max(np.median(time_rs), 1e-6):.1f}",
        ])
        print(f"  {name}: |ΔLL| med={np.median(ll_diffs):.2e}  speedup={np.median(time_sm)/max(np.median(time_rs),1e-6):.1f}x")

    save_csv(
        "e01_equivalence.csv",
        ["order", "abs_dll_median", "abs_dll_max", "rustima_ms_median", "statsmodels_ms_median", "speedup"],
        rows,
    )
    csv_to_latex(
        "e01_equivalence.csv",
        "e01_equivalence.tex",
        "Likelihood equivalence of \\texttt{rustima} and \\texttt{statsmodels} across representative orders ($n=500$, 5 trials per order).",
        "tab:rustima:e01-equivalence",
    )
    print("\nDone. CSV+LaTeX saved.")


if __name__ == "__main__":
    main()
