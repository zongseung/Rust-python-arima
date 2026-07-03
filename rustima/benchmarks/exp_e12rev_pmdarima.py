"""E12-rev: Repackage bench_pmdarima_compare 결과를 thesis_results 형식으로 변환.

bench_pmdarima_compare.py 실행 결과 (paper/bench_2019_2023/compare_pmdarima_1y.csv)를
읽어 thesis LaTeX 표로 변환.

Outputs:
- raw/e12_pmdarima_compare.csv
- tables/e12_pmdarima_compare.tex
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from thesis_utils import csv_to_latex, save_csv

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT.parent / "paper" / "bench_2019_2023" / "compare_pmdarima_1y.csv"


def main():
    if not SRC.exists():
        # fallback: try inside rustima/paper
        alt = ROOT / "paper" / "bench_2019_2023" / "compare_pmdarima_1y.csv"
        if alt.exists():
            src = alt
        else:
            raise FileNotFoundError(
                f"compare_pmdarima_1y.csv not found.\nLooked in: {SRC}\nand: {alt}"
            )
    else:
        src = SRC

    df = pd.read_csv(src)
    print(df)

    rows = []
    for _, r in df.iterrows():
        order_str = f"{r['order']}"
        sorder_str = f"{r['seasonal']}"
        aic = f"{r['aic']:.1f}" if pd.notna(r['aic']) else "—"
        time_s = f"{r['time_s']:.1f}"
        rss = f"{r['peak_rss_mb']:.0f}"
        err = r.get('error', '')
        if pd.notna(err) and err and err != "nan":
            status = str(err)[:30]
        else:
            status = "OK"
        rows.append([r['engine'], time_s, rss, order_str, sorder_str, aic, status])

    save_csv(
        "e12_pmdarima_compare.csv",
        ["engine", "time_s", "peak_rss_mb", "order", "seasonal", "aic", "status"],
        rows,
    )
    csv_to_latex(
        "e12_pmdarima_compare.csv",
        "e12_pmdarima_compare.tex",
        "Direct comparison of \\texttt{rustima} and \\texttt{pmdarima} \\texttt{auto\\_arima} on 1 year of hourly electricity demand ($n=8{,}760$, $s=24$, exog $=$ temperature $+$ humidity). \\texttt{pmdarima} is subject to a 600\\,s hard timeout.",
        "tab:rustima:e12-pmdarima",
    )

    # Summary print
    if len(rows) >= 2:
        rs = next((r for r in rows if r[0] == "rustima"), None)
        pm = next((r for r in rows if r[0] == "pmdarima"), None)
        if rs and pm and pm[6] == "OK":
            try:
                speedup = float(pm[1]) / max(float(rs[1]), 1e-6)
                mem = float(pm[2]) / max(float(rs[2]), 1e-6)
                print(f"\nSpeedup (pmdarima / rustima): {speedup:.1f}x")
                print(f"Memory ratio (pmdarima / rustima): {mem:.2f}x")
            except (ValueError, TypeError):
                pass


if __name__ == "__main__":
    main()
