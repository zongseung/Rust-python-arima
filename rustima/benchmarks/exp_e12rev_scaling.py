"""E12-rev (optional): Scaling test result repackager.

Reads paper/bench_2019_2023/scaling.json (from bench_sarima_scaling.py)
and converts to thesis LaTeX format.

This experiment is OPTIONAL — run only if you want to demonstrate the red-text
claim about swap/memory pressure (bench_sarima_scaling.py with hard guards
of 500 MB swap delta and 30-min per-engine timeout).

To run scaling test:
    uv run python benchmarks/bench_sarima_scaling.py

Then run this wrapper to convert results to thesis format.

Outputs:
- raw/e12_scaling.csv
- tables/e12_scaling.tex
- figures/e12_scaling.png
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from thesis_utils import FIGURES_DIR, csv_to_latex, save_csv, setup_mpl

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT.parent / "paper" / "bench_2019_2023" / "scaling.json"
ALT_JSON = ROOT / "paper" / "bench_2019_2023" / "scaling.json"


def main():
    src = JSON_PATH if JSON_PATH.exists() else (ALT_JSON if ALT_JSON.exists() else None)
    if src is None:
        print(f"Scaling JSON not found at {JSON_PATH} or {ALT_JSON}")
        print("Run `benchmarks/bench_sarima_scaling.py` first.")
        return

    with open(src) as f:
        data = json.load(f)

    runs = data if isinstance(data, list) else data.get("runs", [])
    if not runs:
        print(f"No runs in {src}")
        return

    rows = []
    for r in runs:
        rows.append([
            r["engine"],
            r["years"],
            r.get("n_obs", "—"),
            f"{r.get('time_s', 0):.0f}",
            f"{r.get('peak_rss_mb', 0):.0f}",
            f"{r.get('peak_swap_delta_mb', 0):.0f}",
            r.get("status", "OK"),
        ])
    save_csv(
        "e12_scaling.csv",
        ["engine", "years", "n_obs", "time_s", "peak_rss_mb", "swap_delta_mb", "status"],
        rows,
    )
    csv_to_latex(
        "e12_scaling.csv",
        "e12_scaling.tex",
        "Year-by-year scaling of \\texttt{rustima} vs.\\ \\texttt{pmdarima} \\texttt{auto\\_arima} on hourly electricity demand. Hard guards: 500\\,MB swap-delta SIGKILL and 1{,}800\\,s timeout. \\texttt{KILLED\\_SWAP} indicates the operating system began swapping, the hallmark of memory-pressure runtime degradation referenced in Section~\\ref{sec:meth:rustima}.",
        "tab:rustima:e12-scaling",
    )

    plt = setup_mpl()
    fig, ax = plt.subplots()
    df = pd.DataFrame(runs)
    for eng, sub in df.groupby("engine"):
        sub = sub.sort_values("years")
        ok = sub[sub.status == "OK"]
        ax.plot(ok["years"], ok["time_s"], "o-", label=eng)
        killed = sub[sub.status != "OK"]
        if len(killed):
            ax.scatter(killed["years"], [3e3] * len(killed), marker="x", s=80, color="red", label=f"{eng} killed/timeout" if len(killed) <= 2 else None)
    ax.set_xlabel("Years of hourly data")
    ax.set_ylabel("Total auto_arima time (s)")
    ax.set_yscale("log")
    ax.set_title("auto_arima scaling: rustima vs pmdarima")
    ax.legend()
    fig.savefig(FIGURES_DIR / "e12_scaling.png")
    plt.close(fig)
    print(f"Done. {len(rows)} runs processed.")


if __name__ == "__main__":
    main()
