"""E9: Warm-start ablation — effect of different starting strategies on iterations.

Compares optimization methods (which encapsulate different start strategies).

Outputs:
- raw/e09_warm_start.csv
"""
from __future__ import annotations

import warnings

import numpy as np
import rustima

from thesis_utils import Timer, csv_to_latex, gen_random_sarimax, save_csv

warnings.filterwarnings("ignore")

METHODS = ["lbfgsb", "lbfgsb-multi", "trust-region", "lbfgsb-adaptive", "lbfgsb-hybrid"]
ORDER = (1, 0, 1)
SORDER = (1, 0, 1, 12)
N = 500
N_TRIALS = 15


def main():
    rows = []
    for method in METHODS:
        iters = []
        times = []
        lls = []
        n_ok = 0
        for trial in range(N_TRIALS):
            y, _, _ = gen_random_sarimax(N, *ORDER, *SORDER, seed=trial + 300)
            try:
                with Timer() as t:
                    r = rustima.sarimax_fit(y, ORDER, SORDER, method=method, maxiter=200)
                iters.append(r.get("n_iter", -1))
                times.append(t.ms)
                lls.append(r["loglike"])
                if r["converged"]:
                    n_ok += 1
            except Exception as e:
                continue

        if iters:
            rows.append([
                method,
                f"{n_ok}/{N_TRIALS}",
                f"{np.median(iters):.1f}",
                f"{np.median(times):.1f}",
                f"{np.median(lls):.2f}",
            ])
            print(f"  {method}: conv={n_ok}/{N_TRIALS}, iter_med={np.median(iters):.0f}, time_med={np.median(times):.1f}ms, LL_med={np.median(lls):.2f}")

    save_csv(
        "e09_warm_start.csv",
        ["method", "convergence", "iter_median", "time_ms_median", "loglike_median"],
        rows,
    )
    csv_to_latex(
        "e09_warm_start.csv",
        "e09_warm_start.tex",
        f"Optimization methods compared on SARIMA{ORDER}{SORDER[:3]}$_{{{SORDER[3]}}}$ ($n={N}$, {N_TRIALS} trials per method).",
        "tab:rustima:e09-warm-start",
    )


if __name__ == "__main__":
    main()
