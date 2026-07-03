"""Shared utilities for thesis experiments E1–E14.

- consistent matplotlib style
- LaTeX table emitters
- output path helpers
- common data generators
"""
from __future__ import annotations

import csv
import math
import os
import time
from pathlib import Path

import numpy as np

PAPER_ROOT = Path(__file__).resolve().parent.parent / "paper" / "thesis_results"
TABLES_DIR = PAPER_ROOT / "tables"
FIGURES_DIR = PAPER_ROOT / "figures"
RAW_DIR = PAPER_ROOT / "raw"

for d in (TABLES_DIR, FIGURES_DIR, RAW_DIR):
    d.mkdir(parents=True, exist_ok=True)


def setup_mpl():
    """Consistent matplotlib style for all thesis figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (5.5, 3.5),
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.4,
            "lines.markersize": 4,
            "grid.alpha": 0.3,
            "axes.grid": True,
        }
    )
    return plt


def save_csv(name: str, header: list[str], rows: list[list]) -> Path:
    """Save raw experiment data as CSV under thesis_results/raw/."""
    path = RAW_DIR / name
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    return path


def csv_to_latex(
    csv_name: str,
    tex_name: str,
    caption: str,
    label: str,
    fmt: dict[str, str] | None = None,
    column_align: str | None = None,
) -> Path:
    """Convert a CSV to a LaTeX table.

    fmt: {column_name: format_spec} e.g. {"time_ms": "{:.2f}"}
    column_align: e.g. "lrrrr" — if None, "l" for first column then "r" for rest
    """
    csv_path = RAW_DIR / csv_name
    tex_path = TABLES_DIR / tex_name
    with open(csv_path) as f:
        rdr = csv.reader(f)
        rows = list(rdr)
    header, body = rows[0], rows[1:]

    if column_align is None:
        column_align = "l" + "r" * (len(header) - 1)

    def fmt_cell(col_name: str, val: str) -> str:
        if fmt and col_name in fmt:
            try:
                return fmt[col_name].format(float(val))
            except (ValueError, TypeError):
                return val
        return val

    lines = [
        "\\begin{table}[!htbp]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{column_align}}}",
        "    \\toprule",
        "    " + " & ".join(h.replace("_", r"\_") for h in header) + " \\\\",
        "    \\midrule",
    ]
    for r in body:
        lines.append("    " + " & ".join(fmt_cell(header[i], r[i]) for i in range(len(r))) + " \\\\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    tex_path.write_text("\n".join(lines) + "\n")
    return tex_path


def gen_random_sarimax(
    n: int,
    p: int = 1,
    d: int = 0,
    q: int = 1,
    P: int = 0,
    D: int = 0,
    Q: int = 0,
    s: int = 0,
    n_exog: int = 0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Generate a synthetic SARIMAX series.

    Returns (y, exog or None, true_params_dict).
    """
    rng = np.random.default_rng(seed)
    ar = rng.uniform(0.3, 0.7, size=p) if p else np.array([])
    ma = rng.uniform(-0.5, 0.5, size=q) if q else np.array([])
    sar = rng.uniform(0.2, 0.5, size=P) if P else np.array([])
    sma = rng.uniform(-0.3, 0.3, size=Q) if Q else np.array([])
    beta = rng.uniform(-1.5, 1.5, size=n_exog) if n_exog else np.array([])

    # Stationarize AR
    if p:
        roots = np.roots(np.concatenate([[1.0], -ar]))
        if np.any(np.abs(roots) <= 1.0):
            ar = ar * 0.5

    eps = rng.standard_normal(n) * 0.5
    y = np.zeros(n)
    exog = rng.standard_normal((n, n_exog)) if n_exog else None

    burn = max(50, 3 * s) if s else 50
    for t in range(burn, n):
        v = eps[t]
        for i, a in enumerate(ar):
            v += a * y[t - i - 1]
        for i, m in enumerate(ma):
            if t - i - 1 >= 0:
                v += m * eps[t - i - 1]
        if exog is not None:
            v += float(exog[t] @ beta)
        y[t] = v

    if d:
        y = np.cumsum(y)
    if D and s:
        for _ in range(D):
            y_new = np.zeros_like(y)
            y_new[s:] = np.cumsum(y[s:] - y[:-s])
            y_new[:s] = y[:s]
            y = y_new

    return (
        y.astype(np.float64),
        exog.astype(np.float64) if exog is not None else None,
        {"ar": ar, "ma": ma, "sar": sar, "sma": sma, "beta": beta},
    )


class Timer:
    def __init__(self):
        self.t = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.t = time.perf_counter() - self.t0

    @property
    def ms(self) -> float:
        return self.t * 1000.0

    @property
    def us(self) -> float:
        return self.t * 1e6
