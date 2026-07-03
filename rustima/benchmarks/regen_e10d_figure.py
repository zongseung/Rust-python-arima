"""Regenerate figures/e10d_auto_ptr.png from the updated raw CSV only.

Standalone re-plot (does NOT re-run the experiment, unlike exp_e10d_auto_ptr.py).
Handles the 2y pmdarima "diverged" row: its wall-clock bar is kept (it really ran
~1164 s before returning an invalid model) but its AIC bar is dropped and the
panel is annotated, so the invalid AIC=10.0 is never plotted as a real value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from thesis_utils import FIGURES_DIR, RAW_DIR, setup_mpl


def main():
    df = pd.read_csv(RAW_DIR / "e10d_auto_ptr.csv")
    df["aic_num"] = pd.to_numeric(df["aic"], errors="coerce")
    df["time_num"] = pd.to_numeric(df["time_s"], errors="coerce")

    windows = ["1y", "2y"]
    engines = ["rustima auto (lbfgsb)", "rustima auto (PTR ★)", "pmdarima"]
    colors = ["#888888", "C0", "#bb6666"]

    plt = setup_mpl()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(windows))
    width = 0.27

    for i, eng in enumerate(engines):
        aics, times = [], []
        for w in windows:
            row = df[(df.window == w) & (df.engine == eng)]
            aics.append(float(row["aic_num"].iloc[0]) if len(row) else np.nan)
            times.append(float(row["time_num"].iloc[0]) if len(row) else np.nan)
        ax1.bar(x + (i - 1) * width, aics, width, label=eng, color=colors[i])
        ax2.bar(x + (i - 1) * width, times, width, label=eng, color=colors[i])

    # Annotate any engine/window whose AIC is invalid but that still consumed time
    # (e.g. 2y pmdarima diverged): no AIC bar, mark it explicitly.
    for i, eng in enumerate(engines):
        for j, w in enumerate(windows):
            row = df[(df.window == w) & (df.engine == eng)]
            if len(row) and pd.isna(row["aic_num"].iloc[0]) and not pd.isna(row["time_num"].iloc[0]):
                ymax = ax1.get_ylim()[1]
                ax1.text(x[j] + (i - 1) * width, ymax * 0.13,
                         "pmdarima\ndiverged",
                         ha="center", va="center", fontsize=8, color="#a11",
                         bbox=dict(boxstyle="round,pad=0.25", fc="#fbeaea",
                                   ec="#bb6666", lw=0.8))

    ax1.set_xticks(x); ax1.set_xticklabels(windows)
    ax1.set_ylabel("AIC (lower = better)")
    ax1.set_title("AIC by engine")
    ax1.legend(fontsize=8, loc="upper left")

    ax2.set_xticks(x); ax2.set_xticklabels(windows)
    ax2.set_ylabel("Time (s, log scale)")
    ax2.set_yscale("log")
    ax2.set_title("Wall-clock time")
    ax2.legend(fontsize=8, loc="upper left")

    out = FIGURES_DIR / "e10d_auto_ptr.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[figure] {out}")


if __name__ == "__main__":
    main()
