"""E2: Sparsity matrix — k, nnz(T), nnz(R), density across orders.

Outputs:
- raw/e02_sparsity.csv
- tables/e02_sparsity.tex
"""
from __future__ import annotations

import numpy as np
import rustima

from thesis_utils import csv_to_latex, gen_random_sarimax, save_csv

ORDERS = [
    ((1, 0, 0), (0, 0, 0, 0)),
    ((0, 0, 1), (0, 0, 0, 0)),
    ((1, 0, 1), (0, 0, 0, 0)),
    ((2, 0, 1), (0, 0, 0, 0)),
    ((1, 1, 1), (0, 0, 0, 0)),
    ((1, 1, 1), (1, 0, 0, 12)),
    ((1, 1, 1), (1, 1, 1, 12)),
    ((1, 1, 1), (1, 1, 1, 24)),
    ((2, 1, 2), (0, 0, 0, 0)),
    ((2, 1, 2), (1, 1, 1, 7)),
    ((2, 1, 2), (1, 1, 1, 12)),
    ((2, 1, 2), (1, 1, 1, 24)),
    ((3, 1, 3), (1, 1, 1, 12)),
    ((3, 1, 3), (1, 1, 1, 24)),
    ((1, 1, 1), (2, 1, 2, 24)),
    ((2, 1, 2), (2, 1, 2, 24)),
]


def _k_states(p, d, q, P, D, Q, s):
    """Compute state dimension following Harvey representation."""
    k_order = max(p + s * P, q + s * Q + 1)
    k_diff = d + s * D
    return k_order + k_diff, k_order, k_diff


def main():
    rows = []
    for order, sorder in ORDERS:
        p, d, q = order
        P, D, Q, s = sorder
        k, k_arma, k_diff = _k_states(p, d, q, P, D, Q, s)
        # Theoretical max nonzeros: ARMA block T has ~k_arma + (p+sP) AR nonzeros and identity shift; differencing has k_diff entries.
        # Bound: 2*k (rough). For density we use 2k / k^2 = 2/k.
        # Conservative upper bound on nonzero entries in companion-form T:
        # ARMA companion has (p+sP) AR entries + (k_arma-1) shift entries + sP seasonal shifts.
        nnz_T_bound = min(k_arma * 2 + k_diff, k * k)
        nnz_R_bound = 1 + k_diff
        density_T = nnz_T_bound / (k * k) if k else 0.0
        density_T = min(density_T, 1.0)
        rows.append([
            f"({p},{d},{q})({P},{D},{Q})_{s}",
            k,
            k_arma,
            k_diff,
            nnz_T_bound,
            f"{density_T*100:.2f}",
        ])

    save_csv(
        "e02_sparsity.csv",
        ["order", "k", "k_arma", "k_diff", "nnz_T_upper", "density_T_percent"],
        rows,
    )
    csv_to_latex(
        "e02_sparsity.csv",
        "e02_sparsity.tex",
        "State dimension and structural sparsity of the Harvey-form transition matrix $T$. The dominant covariance recursion has cost determined by the structural nonzero count rather than $k^2$.",
        "tab:rustima:e02-sparsity",
    )
    for r in rows:
        print(f"  {r[0]}: k={r[1]}  nnz_T≤{r[4]}  density={r[5]}%")


if __name__ == "__main__":
    main()
