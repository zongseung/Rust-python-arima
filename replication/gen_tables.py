r"""Render outputs/raw/*.csv into booktabs LaTeX fragments in outputs/tables/.

Each fragment is a bare tabular environment meant to be \input{} or pasted
into the paper. Missing raw files are skipped with a notice so the script can
run on partial results.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import RAW_DIR, TABLES_DIR, ensure_dirs  # noqa: E402


STATUS_LABEL = {
    "ok": "ok",
    "oom": "out of memory",
    "oom_swap": "out of memory (swap)",
    "oom_swap_abs": "out of memory (swap)",
    "timeout": "timeout",
    "error": "error",
}


def _status(s):
    return STATUS_LABEL.get(str(s), str(s).replace("_", " "))


def _fmt(v, nd=2):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "--"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return str(v)


def _write(name: str, lines: list[str]) -> None:
    path = os.path.join(TABLES_DIR, name)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[tex] {path}")


def _load(name: str):
    path = os.path.join(RAW_DIR, name)
    if not os.path.exists(path):
        print(f"[skip] {name} not found")
        return None
    return pd.read_csv(path)


ENGINE_LABEL = {
    "rustima": r"\pkg{rustima}",
    "rustima_grid8": r"\pkg{rustima}, grid, 8 threads",
    "statsmodels_seq": r"\pkg{statsmodels}, sequential",
    "statsmodels_joblib": r"\pkg{statsmodels} + \pkg{joblib}",
    "statsmodels": r"\pkg{statsmodels}",
    "statsforecast": r"\pkg{StatsForecast}",
    "pmdarima": r"\pkg{pmdarima}",
    "pmdarima_grid8": r"\pkg{pmdarima}, grid, \code{n\_jobs=8}",
    "r_forecast": r"\proglang{R} \pkg{forecast}",
}


def table_parallel() -> None:
    df = _load("parallel_scaling.csv")
    if df is None:
        return
    for wname, sub in df.groupby("workload"):
        lines = [
            r"\begin{tabular}{lrrrr}",
            r"  \toprule",
            r"  Condition & Workers & Time (s) & Speedup & Peak RSS (GB) \\",
            r"  \midrule",
        ]
        base_row = sub[sub["engine"] == "statsmodels_seq"]
        base = base_row["inner_min_s"].iloc[0] if len(base_row) else None
        for _, r in sub.iterrows():
            label = ENGINE_LABEL.get(r["engine"], r["engine"])
            if r["status"] != "ok":
                lines.append(
                    f"  {label} & {r['n_jobs']} & "
                    f"\\multicolumn{{2}}{{c}}{{{_status(r['status'])}}} & "
                    f"{_fmt(r['peak_rss_gb'])} \\\\"
                )
                continue
            speedup = (
                f"{base / r['inner_min_s']:.1f}$\\times$"
                if base and r["inner_min_s"]
                else "--"
            )
            lines.append(
                f"  {label} & {r['n_jobs']} & {_fmt(r['inner_min_s'])} & "
                f"{speedup} & {_fmt(r['peak_rss_gb'])} \\\\"
            )
        lines += [r"  \bottomrule", r"\end{tabular}"]
        _write(f"parallel_scaling_{wname}.tex", lines)


def table_longseries() -> None:
    df = _load("longseries_scaling.csv")
    if df is None:
        return
    lines = [
        r"\begin{tabular}{rrrlrrr}",
        r"  \toprule",
        r"  $s$ & $n$ & $k$ & Engine & Fit time (s) & Peak RSS (GB) & Status \\",
        r"  \midrule",
    ]
    for _, r in df.iterrows():
        label = ENGINE_LABEL.get(r["engine"], r["engine"])
        lines.append(
            f"  {r['s']} & {r['n']:,} & {r['k_states']} & {label} & "
            f"{_fmt(r['fit_time_s'])} & {_fmt(r['peak_rss_gb'])} & "
            f"{_status(r['status'])} \\\\"
        )
    lines += [r"  \bottomrule", r"\end{tabular}"]
    _write("longseries_scaling.tex", lines)


def table_auto_fourway() -> None:
    df = _load("auto_fourway.csv")
    if df is not None:
        lines = [
            r"\begin{tabular}{llrrrr}",
            r"  \toprule",
            r"  Engine & Selected model & Time (s) & Peak RSS (GB)"
            r" & Native AIC & Re-fit AIC \\",
            r"  \midrule",
        ]
        for _, r in df.iterrows():
            label = ENGINE_LABEL.get(r["engine"], r["engine"])
            if r["status"] != "ok":
                lines.append(
                    f"  {label} & \\multicolumn{{2}}{{c}}{{{_status(r['status'])}}} & "
                    f"{_fmt(r['peak_rss_gb'])} & -- & -- \\\\"
                )
                continue
            model = f"{r['order']}{r['seasonal_order']}"
            lines.append(
                f"  {label} & {model} & {_fmt(r['wall_time_s'], 1)} & "
                f"{_fmt(r['peak_rss_gb'])} & {_fmt(r['aic_native'], 1)} & "
                f"{_fmt(r.get('aic_refit_rustima'), 1)} \\\\"
            )
        lines += [r"  \bottomrule", r"\end{tabular}"]
        _write("auto_fourway.tex", lines)

    df = _load("auto_grid.csv")
    if df is not None:
        lines = [
            r"\begin{tabular}{llrrr}",
            r"  \toprule",
            r"  Engine & Selected model & Time (s) & Peak RSS (GB) & AIC \\",
            r"  \midrule",
        ]
        for _, r in df.iterrows():
            label = ENGINE_LABEL.get(r["engine"], r["engine"])
            if r["status"] != "ok":
                lines.append(
                    f"  {label} & \\multicolumn{{2}}{{c}}{{{_status(r['status'])}}} & "
                    f"{_fmt(r['peak_rss_gb'])} & -- \\\\"
                )
                continue
            model = f"{r['order']}{r['seasonal_order']}"
            lines.append(
                f"  {label} & {model} & {_fmt(r['wall_time_s'], 1)} & "
                f"{_fmt(r['peak_rss_gb'])} & {_fmt(r['aic_native'], 1)} \\\\"
            )
        lines += [r"  \bottomrule", r"\end{tabular}"]
        _write("auto_grid.tex", lines)


SPEC_LABEL = {
    "SARIMA(1,0,1)(1,0,1)_24": r"SARIMA$(1,0,1)(1,0,1)_{24}$",
    "SARIMA(2,1,1)(1,0,0)_24": r"SARIMA$(2,1,1)(1,0,0)_{24}$",
    "SARIMAX(1,1,1)(1,0,1)_24 + [ta,hm]":
        r"SARIMAX$(1,1,1)(1,0,1)_{24}$ + [\code{ta},\code{hm}]",
    "SARIMAX(1,1,1)(1,1,1)_24 + [ta,hm]":
        r"SARIMAX$(1,1,1)(1,1,1)_{24}$ + [\code{ta},\code{hm}]",
}


def table_application() -> None:
    df = _load("application_extended.csv")
    if df is None:
        return
    lines = [
        r"\begin{tabular}{llrrrcr}",
        r"  \toprule",
        r"  Specification & Engine & Fit (s) & AIC & MAPE$_{48h}$ (\%)"
        r" & Conv. & RSS (GB) \\",
        r"  \midrule",
    ]
    for _, r in df.iterrows():
        label = ENGINE_LABEL.get(r["engine"], r["engine"])
        if r["status"] != "ok":
            lines.append(
                f"  {SPEC_LABEL.get(r['spec'], r['spec'])} & {label} & "
                f"\\multicolumn{{4}}{{c}}{{{_status(r['status'])}}} & "
                f"{_fmt(r['peak_rss_gb'])} \\\\"
            )
            continue
        conv = "Yes" if r["converged"] else "No"
        lines.append(
            f"  {SPEC_LABEL.get(r['spec'], r['spec'])} & {label} & "
            f"{_fmt(r['fit_time_s'], 1)} & "
            f"{_fmt(r['aic'], 0)} & {_fmt(r['mape_48h'])} & {conv} & "
            f"{_fmt(r['peak_rss_gb'])} \\\\"
        )
    lines += [r"  \bottomrule", r"\end{tabular}"]
    _write("application_extended.tex", lines)


def main() -> None:
    ensure_dirs()
    table_parallel()
    table_longseries()
    table_auto_fourway()
    table_application()


if __name__ == "__main__":
    main()
