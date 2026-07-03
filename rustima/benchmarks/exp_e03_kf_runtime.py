"""E3: Per-step KF runtime — fit time across state dimension k.

Outputs:
- raw/e03_kf_runtime.csv
- figures/e03_kf_runtime.png
"""
from __future__ import annotations

import numpy as np
import rustima

from thesis_utils import Timer, gen_random_sarimax, save_csv, setup_mpl, FIGURES_DIR

# To vary k, we vary seasonal period s with fixed (1,1,1)(1,1,1)
SEASONS = [0, 4, 7, 12, 24, 48]
N = 2000
N_TRIALS = 5


def _k_states(p, d, q, P, D, Q, s):
    k_order = max(p + s * P, q + s * Q + 1)
    k_diff = d + s * D
    return k_order + k_diff


def main():
    rows = []
    for s in SEASONS:
        if s == 0:
            order, sorder = (1, 1, 1), (0, 0, 0, 0)
        else:
            order, sorder = (1, 1, 1), (1, 1, 1, s)
        k = _k_states(*order, *sorder)

        ts = []
        for trial in range(N_TRIALS):
            y, _, _ = gen_random_sarimax(N, *order, *sorder, seed=trial + 100)
            with Timer() as t:
                rustima.sarimax_fit(y, order, sorder)
            ts.append(t.ms)
        # Approximate per-step μs based on full fit divided by n*opt_iters_est
        # Simplification: report fit_ms and per_step_us = fit_ms*1000 / (n * 50) where 50 ~ avg iters
        fit_med = np.median(ts)
        per_step_us = fit_med * 1000 / (N * 30)
        rows.append([s, k, N, f"{fit_med:.2f}", f"{per_step_us:.2f}"])
        print(f"  s={s}, k={k}: fit_ms={fit_med:.1f}, est_per_step_us={per_step_us:.2f}")

    save_csv(
        "e03_kf_runtime.csv",
        ["s", "k", "n", "fit_ms_median", "est_per_step_us"],
        rows,
    )

    plt = setup_mpl()
    fig, ax = plt.subplots()
    ks = [r[1] for r in rows]
    us = [float(r[4]) for r in rows]
    ax.plot(ks, us, "o-", label="rustima (sparse + cache)", color="C0")
    ax.set_xlabel("State dimension $k$")
    ax.set_ylabel("Estimated per-step μs")
    ax.set_title("Kalman-filter per-step cost vs. state dimension")
    ax.legend(loc="upper left")
    fig.savefig(FIGURES_DIR / "e03_kf_runtime.png")
    plt.close(fig)
    print("\nFigure saved: e03_kf_runtime.png")


if __name__ == "__main__":
    main()
