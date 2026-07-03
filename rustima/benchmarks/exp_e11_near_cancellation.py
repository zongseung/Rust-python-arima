"""E11: Near-cancellation filtering — effect on AIC and fit stability.

Tests scenarios where AR/MA polynomials have near-cancelling roots and measures
whether rustima's internal filter rejects/warns appropriately.

Outputs:
- raw/e11_near_cancellation.csv
"""
from __future__ import annotations

import warnings

import numpy as np
import rustima

from thesis_utils import csv_to_latex, gen_random_sarimax, save_csv

warnings.filterwarnings("ignore")


def near_cancellation_stats(params, p, q):
    """Compute min |AR root - MA root| distance."""
    if p == 0 or q == 0:
        return float("inf")
    try:
        ar_idx = 0
        ma_idx = p
        ar = np.asarray(params[ar_idx : ar_idx + p], dtype=float)
        ma = np.asarray(params[ma_idx : ma_idx + q], dtype=float)
        ar_poly = np.concatenate([[1.0], -ar])
        ma_poly = np.concatenate([[1.0], ma])
        ar_roots = np.roots(ar_poly)
        ma_roots = np.roots(ma_poly)
        dists = []
        for a in ar_roots:
            for m in ma_roots:
                dists.append(abs(a - m))
        return min(dists) if dists else float("inf")
    except Exception:
        return float("inf")


ORDERS = [
    ((1, 0, 1), (0, 0, 0, 0), "ARMA(1,1)"),
    ((2, 0, 1), (0, 0, 0, 0), "ARMA(2,1)"),
    ((2, 0, 2), (0, 0, 0, 0), "ARMA(2,2)"),
    ((1, 0, 1), (1, 0, 1, 12), "SARMA(1,0,1)(1,0,1)_12"),
]
N = 500
N_TRIALS = 50


def main():
    rows = []
    for order, sorder, name in ORDERS:
        dists = []
        warn_count = 0
        for trial in range(N_TRIALS):
            y, _, _ = gen_random_sarimax(N, *order, *sorder, seed=trial + 400)
            try:
                import warnings as W

                with W.catch_warnings(record=True) as cap:
                    W.simplefilter("always")
                    r = rustima.sarimax_fit(y, order, sorder)
                    if any("near-cancellation" in str(w.message) for w in cap):
                        warn_count += 1
                d = near_cancellation_stats(r["params"], order[0], order[2])
                if np.isfinite(d):
                    dists.append(d)
            except Exception:
                continue

        if not dists:
            rows.append([name, "—", "—", "—", "—"])
            continue
        rows.append([
            name,
            f"{np.median(dists):.4f}",
            f"{np.percentile(dists, 10):.4f}",
            f"{(np.array(dists) < 0.05).mean()*100:.0f}",
            f"{warn_count}/{N_TRIALS}",
        ])
        print(f"  {name}: dist_med={np.median(dists):.3f}, p10={np.percentile(dists,10):.3f}, near_cancel%={(np.array(dists)<0.05).mean()*100:.0f}, warns={warn_count}")

    save_csv(
        "e11_near_cancellation.csv",
        ["spec", "root_dist_median", "root_dist_p10", "near_cancel_pct", "warnings_emitted"],
        rows,
    )
    csv_to_latex(
        "e11_near_cancellation.csv",
        "e11_near_cancellation.tex",
        "Near-cancellation diagnostics across random fits ($n=500$, 50 trials per order). Distance is $\\min_{i,j}|r_i^{\\mathrm{AR}}-r_j^{\\mathrm{MA}}|$. \\texttt{rustima} emits user warnings when distance falls below threshold $\\delta_{\\mathrm{nc}}=0.05$.",
        "tab:rustima:e11-near-cancel",
    )


if __name__ == "__main__":
    main()
