"""Regenerate figures/e12_scaling_exog.png from the updated raw CSV only.

Standalone so it does NOT overwrite the hand-edited table caption that the
full exp_e12rev_scaling_pack.py pipeline would regenerate. Same thesis style.

rustima draws as a scaling line; pmdarima draws its one completed window as a
point and its swap-terminated windows as X markers in a band near the top of
each panel (the watchdog-cut RSS is not a meaningful peak, so killed runs are
shown as "exceeded" rather than at a fabricated value).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from thesis_utils import FIGURES_DIR, RAW_DIR, setup_mpl


def main():
    df = pd.read_csv(RAW_DIR / "e12_scaling_exog.csv")
    print(df)

    plt = setup_mpl()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    eng_colors = {"rustima": "C0", "pmdarima": "#bb6666", "R": "C2"}

    # --- OK points/lines ---
    for eng, sub in df.groupby("engine"):
        ok = sub[sub.status == "OK"].sort_values("years")
        if len(ok) == 0:
            continue
        style = "o-" if len(ok) > 1 else "o"
        ax1.plot(ok["years"], ok["time_s"], style, label=eng,
                 color=eng_colors.get(eng, "k"))
        ax2.plot(ok["years"], ok["peak_rss_mb"], style, label=eng,
                 color=eng_colors.get(eng, "k"))

    # --- KILLED_SWAP markers: place in a band near the top of each panel ---
    killed = df[df.status == "KILLED_SWAP"].sort_values("years")
    if len(killed) > 0:
        for ax in (ax1, ax2):
            lo, hi = ax.get_ylim()
            band = hi * 0.92
            ax.set_ylim(lo, hi * 1.12)
            ax.scatter(killed["years"], [band] * len(killed),
                       marker="x", s=130, linewidths=2.2,
                       color="#bb6666", zorder=6)
            xmid = killed["years"].iloc[len(killed) // 2]
            ax.annotate("pmdarima\nKILLED_SWAP", xy=(xmid, band),
                        xytext=(xmid, band * 0.62), ha="center",
                        fontsize=8, color="#bb6666")

    ax1.set_xlabel("Years of hourly data")
    ax1.set_ylabel("Wall-clock time (s)")
    ax1.set_title("auto_arima time")
    ax1.legend(fontsize=8, loc="center left")
    ax1.set_xticks([1, 2, 3])

    ax2.set_xlabel("Years of hourly data")
    ax2.set_ylabel("Peak resident memory (MB)")
    ax2.set_title("Peak resident memory")
    ax2.legend(fontsize=8, loc="center left")
    ax2.set_xticks([1, 2, 3])

    out = FIGURES_DIR / "e12_scaling_exog.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[figure] {out}")


if __name__ == "__main__":
    main()
