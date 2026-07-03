"""E6: Initialization-path selection — DARE / mixed Lyapunov / approximate diffuse.

Since rustima auto-selects the path internally (initialization.rs:215-228) based on
enforce_stationarity and effective_sd, we compute the rule directly for representative
specifications and report which path is exercised.

Outputs:
- raw/e06_init_path.csv
- tables/e06_init_path.tex
"""
from __future__ import annotations

from thesis_utils import csv_to_latex, save_csv

ORDERS = [
    ((1, 0, 0), (0, 0, 0, 0), True, "AR(1) stationary"),
    ((2, 0, 1), (0, 0, 0, 0), True, "ARMA(2,1) stationary"),
    ((1, 1, 1), (0, 0, 0, 0), True, "ARIMA(1,1,1)"),
    ((1, 0, 1), (1, 0, 1, 12), True, "SARMA(1,0,1)(1,0,1)_12 stationary"),
    ((1, 1, 1), (1, 0, 0, 12), True, "SARIMA(1,1,1)(1,0,0)_12"),
    ((1, 1, 1), (1, 1, 1, 12), True, "SARIMA(1,1,1)(1,1,1)_12"),
    ((1, 1, 1), (1, 1, 1, 24), True, "SARIMA(1,1,1)(1,1,1)_24"),
    ((2, 1, 2), (2, 1, 2, 24), True, "SARIMA(2,1,2)(2,1,2)_24"),
    ((1, 0, 0), (0, 0, 0, 0), False, "AR(1) unconstrained"),
    ((1, 1, 1), (0, 0, 0, 0), False, "ARIMA(1,1,1) unconstrained"),
]


def path_for(p, d, q, P, D, Q, s, enforce_stationarity):
    sd = d + s * D
    if not enforce_stationarity:
        return "diffuse"
    if sd == 0:
        return "DARE"
    return "mixed Lyapunov"


def main():
    rows = []
    counts = {"DARE": 0, "mixed Lyapunov": 0, "diffuse": 0}
    for order, sorder, enforce, name in ORDERS:
        sd = order[1] + sorder[3] * sorder[1]
        path = path_for(*order, *sorder, enforce)
        counts[path] += 1
        rows.append([name, "yes" if enforce else "no", sd, path])
        print(f"  {name}: s_d={sd}, path={path}")

    save_csv("e06_init_path.csv", ["spec", "enforce_stationarity", "s_d", "init_path"], rows)
    csv_to_latex(
        "e06_init_path.csv",
        "e06_init_path.tex",
        "Initialization-path selection in \\texttt{rustima}: DARE for pure stationary ($s_d=0$, \\texttt{enforce\\_stationarity}=true), mixed Lyapunov when $s_d>0$ under enforced stationarity, and approximate diffuse otherwise.",
        "tab:rustima:e06-init-path",
    )
    print(f"\nSummary: DARE={counts['DARE']}, mixed-Lyap={counts['mixed Lyapunov']}, diffuse={counts['diffuse']}")


if __name__ == "__main__":
    main()
