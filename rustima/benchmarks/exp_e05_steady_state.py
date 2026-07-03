"""E5: Steady-state convergence statistics — distribution of fit times.

Since per-step KF cache activation point is internal, we proxy via fit time
distribution across many random stationary specifications.

Outputs:
- raw/e05_steady_state.csv
- figures/e05_steady_state_hist.png
"""
from __future__ import annotations

import numpy as np
import rustima

from thesis_utils import Timer, gen_random_sarimax, save_csv, setup_mpl, FIGURES_DIR

N_TRIALS = 100
N = 1000


def main():
    times_ar = []
    times_arma = []
    times_sarma = []

    for i in range(N_TRIALS):
        y, _, _ = gen_random_sarimax(N, 1, 0, 0, 0, 0, 0, 0, seed=i)
        with Timer() as t:
            rustima.sarimax_fit(y, (1, 0, 0), (0, 0, 0, 0))
        times_ar.append(t.ms)

        y, _, _ = gen_random_sarimax(N, 2, 0, 1, 0, 0, 0, 0, seed=i)
        with Timer() as t:
            rustima.sarimax_fit(y, (2, 0, 1), (0, 0, 0, 0))
        times_arma.append(t.ms)

        y, _, _ = gen_random_sarimax(N, 1, 0, 1, 1, 0, 1, 12, seed=i)
        with Timer() as t:
            rustima.sarimax_fit(y, (1, 0, 1), (1, 0, 1, 12))
        times_sarma.append(t.ms)

    rows = [
        ["AR(1)", f"{np.median(times_ar):.2f}", f"{np.percentile(times_ar, 90):.2f}", f"{np.std(times_ar):.2f}"],
        ["ARMA(2,1)", f"{np.median(times_arma):.2f}", f"{np.percentile(times_arma, 90):.2f}", f"{np.std(times_arma):.2f}"],
        ["SARIMA(1,0,1)(1,0,1)_12", f"{np.median(times_sarma):.2f}", f"{np.percentile(times_sarma, 90):.2f}", f"{np.std(times_sarma):.2f}"],
    ]
    save_csv("e05_steady_state.csv", ["spec", "fit_ms_median", "fit_ms_p90", "fit_ms_std"], rows)

    plt = setup_mpl()
    fig, ax = plt.subplots()
    ax.hist(times_ar, bins=20, alpha=0.6, label="AR(1)")
    ax.hist(times_arma, bins=20, alpha=0.6, label="ARMA(2,1)")
    ax.hist(times_sarma, bins=20, alpha=0.6, label="SARMA(1,0,1)(1,0,1)$_{12}$")
    ax.set_xlabel("Fit time (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"Fit-time distribution across {N_TRIALS} random stationary models ($n={N}$)")
    ax.legend()
    fig.savefig(FIGURES_DIR / "e05_steady_state_hist.png")
    plt.close(fig)
    print(f"\n  AR(1) median: {np.median(times_ar):.1f}ms")
    print(f"  ARMA(2,1) median: {np.median(times_arma):.1f}ms")
    print(f"  SARMA(1,0,1)(1,0,1)_12 median: {np.median(times_sarma):.1f}ms")


if __name__ == "__main__":
    main()
