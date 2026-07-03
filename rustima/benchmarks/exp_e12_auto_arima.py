"""E12: stepwise vs Rayon grid for auto_arima on hourly data.

Outputs:
- raw/e12_auto_arima.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rustima

from thesis_utils import Timer, csv_to_latex, save_csv

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parent.parent / "power_demand_2024.csv"


def main():
    df = pd.read_csv(DATA)
    # Fit the actual demand series, not whatever happens to be the last numeric
    # column (previously is_holiday_dummies, a 0/1 indicator → meaningless ARIMA).
    power_col = [c for c in df.columns if "power" in c.lower()][0]
    y = df[power_col].to_numpy(dtype=np.float64)
    y = y[:1500]  # ~62 days
    y_c = y - y.mean()
    print(f"y len = {len(y)}")

    rows = []
    for stepwise in [True, False]:
        try:
            with Timer() as t:
                res = rustima.auto_arima(
                    y_c,
                    max_p=2, max_q=2, max_d=1,
                    max_P=1, max_Q=1, max_D=0,
                    s=24,
                    stepwise=stepwise,
                    criterion="aic",
                    method="lbfgsb",
                    maxiter=100,
                    trace=False,
                )
            order = res.order if hasattr(res, "order") else getattr(res, "best_order", "?")
            sorder = res.seasonal_order if hasattr(res, "seasonal_order") else getattr(res, "best_seasonal", "?")
            aic = getattr(res, "best_ic", None)
            if aic is None:
                aic = res.aic if hasattr(res, "aic") else float("nan")
            n_cand = (res.history_dataframe().shape[0] if hasattr(res, "history_dataframe") else -1)
            rows.append([
                "stepwise" if stepwise else "grid (Rayon)",
                f"{t.ms:.0f}",
                str(order),
                str(sorder),
                f"{aic:.2f}",
                n_cand,
            ])
            print(f"  {'stepwise' if stepwise else 'grid'}: time={t.ms:.0f}ms, order={order}, sorder={sorder}, AIC={aic:.2f}, candidates={n_cand}")
        except Exception as e:
            rows.append(["stepwise" if stepwise else "grid", "ERR", "—", "—", "—", str(e)[:30]])
            print(f"  ERR: {e}")

    save_csv("e12_auto_arima.csv", ["search", "time_ms", "order", "seasonal", "aic", "n_candidates"], rows)
    csv_to_latex(
        "e12_auto_arima.csv",
        "e12_auto_arima.tex",
        "Stepwise (Hyndman--Khandakar) vs.\\ exhaustive Rayon-parallel grid search via \\texttt{auto\\_arima} on hourly demand ($n=1{,}500$, $s=24$).",
        "tab:rustima:e12-auto-arima",
    )


if __name__ == "__main__":
    main()
