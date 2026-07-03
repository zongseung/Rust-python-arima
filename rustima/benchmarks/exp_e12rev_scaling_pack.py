"""E12-rev: Repackage bench_sarima_scaling_exog 결과를 thesis_results 형식으로 변환.

Reads paper/bench_2019_2023/sarima_scaling_results_exog.csv (from
bench_sarima_scaling_exog.py) and converts to thesis LaTeX format
with a thesis-style figure.

Outputs:
- raw/e12_scaling_exog.csv
- tables/e12_scaling_exog.tex
- figures/e12_scaling_exog.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from thesis_utils import FIGURES_DIR, csv_to_latex, save_csv, setup_mpl

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT.parent / "paper" / "bench_2019_2023" / "sarima_scaling_results_exog.csv"


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"Source not found: {SRC}")

    df = pd.read_csv(SRC)
    print(df)

    rows = []
    for _, r in df.iterrows():
        status = r["status"]
        if status == "OK":
            aic_s = f"{r['aic']:.1f}" if pd.notna(r['aic']) else "—"
            order_s = str(r['order']) if pd.notna(r['order']) else "—"
            sorder_s = str(r['seasonal']) if pd.notna(r['seasonal']) else "—"
        else:
            aic_s = "—"
            order_s = "—"
            sorder_s = "—"
        rows.append([
            r["engine"],
            int(r["years"]),
            int(r["n_obs"]) if r["n_obs"] else 0,
            f"{r['time_s']:.0f}",
            f"{r['peak_rss_mb']:.0f}",
            f"{r['peak_swap_delta_mb']:.0f}",
            order_s,
            sorder_s,
            aic_s,
            status,
        ])

    save_csv(
        "e12_scaling_exog.csv",
        ["engine", "years", "n_obs", "time_s", "peak_rss_mb", "swap_delta_mb", "order", "seasonal", "aic", "status"],
        rows,
    )
    csv_to_latex(
        "e12_scaling_exog.csv",
        "e12_scaling_exog.tex",
        "Year-by-year scaling of \\texttt{rustima} \\texttt{auto\\_arima} vs.\\ \\texttt{pmdarima} on Korean hourly electricity demand with $ta$ and $hm$ exogenous regressors. Hard guard: 300\\,MB swap-delta \\texttt{SIGKILL}. \\texttt{KILLED\\_SWAP} status indicates the operating system began swapping memory pages and the per-process watchdog terminated the worker.",
        "tab:rustima:e12-scaling-exog",
    )

    # Thesis-style figure: 1 row 2 cols (time + RSS)
    plt = setup_mpl()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    eng_colors = {"rustima": "C0", "pmdarima": "#bb6666"}
    for eng, sub in df.groupby("engine"):
        ok = sub[sub.status == "OK"].sort_values("years")
        killed = sub[sub.status == "KILLED_SWAP"].sort_values("years")
        # Time
        if len(ok) > 0:
            ax1.plot(ok["years"], ok["time_s"], "o-", label=eng, color=eng_colors.get(eng, "k"))
        if len(killed) > 0:
            ax1.scatter(killed["years"], killed["time_s"].where(killed["time_s"] > 0, 1500),
                       marker="x", s=120, color=eng_colors.get(eng, "red"), zorder=5)
        # RSS
        if len(ok) > 0:
            ax2.plot(ok["years"], ok["peak_rss_mb"], "o-", label=eng, color=eng_colors.get(eng, "k"))
        if len(killed) > 0:
            killed_rss = killed["peak_rss_mb"].copy()
            killed_rss = killed_rss.where(killed_rss > 0, np.nan)
            ax2.scatter(killed["years"][killed_rss.notna()], killed_rss[killed_rss.notna()],
                       marker="x", s=120, color=eng_colors.get(eng, "red"), zorder=5)

    # Annotate the KILLED_SWAP cross
    pm_killed = df[(df.engine == "pmdarima") & (df.status == "KILLED_SWAP") & (df.time_s > 0)]
    if len(pm_killed) > 0:
        kr = pm_killed.iloc[0]
        ax1.annotate("KILLED_SWAP", xy=(kr["years"], kr["time_s"]),
                    xytext=(kr["years"] + 0.3, kr["time_s"] + 100),
                    fontsize=8, color="#bb6666",
                    arrowprops=dict(arrowstyle="->", color="#bb6666"))
        ax2.annotate("KILLED_SWAP", xy=(kr["years"], kr["peak_rss_mb"]),
                    xytext=(kr["years"] + 0.3, kr["peak_rss_mb"] - 2000),
                    fontsize=8, color="#bb6666",
                    arrowprops=dict(arrowstyle="->", color="#bb6666"))

    ax1.set_xlabel("Years of hourly data")
    ax1.set_ylabel("Wall-clock time (s)")
    ax1.set_title("auto_arima time")
    ax1.legend(fontsize=8)
    ax1.set_xticks([1, 2, 3])

    ax2.set_xlabel("Years of hourly data")
    ax2.set_ylabel("Peak resident memory (MB)")
    ax2.set_title("Peak resident memory")
    ax2.legend(fontsize=8)
    ax2.set_xticks([1, 2, 3])

    fig.suptitle("Scaling on Korean hourly demand (with $ta$, $hm$ exog)", y=1.02)
    fig.savefig(FIGURES_DIR / "e12_scaling_exog.png")
    plt.close(fig)
    print(f"\nLaTeX + figure saved.")


if __name__ == "__main__":
    main()
