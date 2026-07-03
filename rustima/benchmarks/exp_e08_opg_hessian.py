"""E8: OPG vs Hessian — standard error comparison vs statsmodels.

Outputs:
- raw/e08_opg.csv
"""
from __future__ import annotations

import warnings

import numpy as np
import rustima
from statsmodels.tsa.statespace.sarimax import SARIMAX

from thesis_utils import csv_to_latex, gen_random_sarimax, save_csv

warnings.filterwarnings("ignore")

ORDERS = [
    ((1, 0, 0), (0, 0, 0, 0), "AR(1)"),
    ((1, 0, 1), (0, 0, 0, 0), "ARMA(1,1)"),
    ((2, 0, 1), (0, 0, 0, 0), "ARMA(2,1)"),
    ((1, 1, 1), (0, 0, 0, 0), "ARIMA(1,1,1)"),
    ((1, 0, 1), (1, 0, 1, 12), "SARMA(1,0,1)(1,0,1)_12"),
]
N = 500
N_TRIALS = 10


def main():
    rows = []
    for order, sorder, name in ORDERS:
        ses_rs_opg = []
        ses_sm_hess = []
        for trial in range(N_TRIALS):
            y, _, _ = gen_random_sarimax(N, *order, *sorder, seed=trial + 200)
            try:
                # rustima fit + OPG inference via low-level API
                rfit = rustima.sarimax_fit(y, order, sorder, maxiter=200)
                inf_rs = rustima.sarimax_inference(y, order, sorder, np.array(rfit["params"], dtype=np.float64), method="opg")
                se_rs = np.array(inf_rs.get("std_err", []), dtype=float)

                sm = SARIMAX(y, order=order, seasonal_order=sorder, simple_differencing=False).fit(disp=False, maxiter=100)
                se_sm = sm.bse.values if hasattr(sm.bse, "values") else np.array(sm.bse)

                # Match dims
                k = min(len(se_rs), len(se_sm))
                if k:
                    ses_rs_opg.append(se_rs[:k])
                    ses_sm_hess.append(se_sm[:k])
            except Exception as e:
                continue

        if not ses_rs_opg:
            rows.append([name, "—", "—", "—"])
            continue

        rs_med = np.median([np.mean(s) for s in ses_rs_opg])
        sm_med = np.median([np.mean(s) for s in ses_sm_hess])
        ratio = rs_med / max(sm_med, 1e-9)
        rows.append([name, f"{rs_med:.4f}", f"{sm_med:.4f}", f"{ratio:.3f}"])
        print(f"  {name}: rustima OPG mean SE={rs_med:.4f}  statsmodels Hessian mean SE={sm_med:.4f}  ratio={ratio:.3f}")

    save_csv("e08_opg.csv", ["spec", "rustima_opg_se_median", "statsmodels_hess_se_median", "ratio"], rows)
    csv_to_latex(
        "e08_opg.csv",
        "e08_opg.tex",
        "Outer-product-of-gradients (\\texttt{rustima}) vs.\\ observed Hessian (\\texttt{statsmodels}) standard errors. Ratio close to 1 indicates agreement.",
        "tab:rustima:e08-opg",
    )


if __name__ == "__main__":
    main()
