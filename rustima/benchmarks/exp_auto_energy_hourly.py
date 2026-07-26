"""Reproduces the "Hourly, s=24" row of Table auto-energy in the JSS paper.

Matched auto_arima comparison (rustima vs pmdarima) on the same 744-hour
(January 2024) slice of power_demand_2024.csv, same search bounds, same
criterion. Run with `uv run python benchmarks/exp_auto_energy_hourly.py`.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "power_demand_2024.csv"
SEARCH = dict(max_p=5, max_q=5, max_P=2, max_Q=2)


def load_january_2024():
    df = pd.read_csv(DATA)
    df["datetime"] = pd.to_datetime(df["datetime"])
    jan = df[(df["datetime"] >= "2024-01-01") & (df["datetime"] < "2024-02-01")]
    return jan["power demand(MW)"].to_numpy(dtype=np.float64)


def run_rustima(y):
    from rustima.auto import auto_arima

    t0 = time.perf_counter()
    res = auto_arima(y, s=24, criterion="aic", stepwise=True, **SEARCH)
    elapsed = time.perf_counter() - t0
    return res.order, res.seasonal_order, res.best_ic, elapsed


def run_pmdarima(y):
    import pmdarima as pm

    t0 = time.perf_counter()
    model = pm.auto_arima(
        y, seasonal=True, m=24,
        information_criterion="aic", stepwise=True,
        error_action="ignore", suppress_warnings=True,
        **SEARCH,
    )
    elapsed = time.perf_counter() - t0
    return model.order, model.seasonal_order, model.aic(), elapsed


def main():
    y = load_january_2024()
    print(f"n = {len(y)}")

    rs_order, rs_seasonal, rs_aic, rs_t = run_rustima(y)
    print(f"rustima:  order={rs_order} seasonal={rs_seasonal} AIC={rs_aic:.2f} time={rs_t:.2f}s")

    pm_order, pm_seasonal, pm_aic, pm_t = run_pmdarima(y)
    print(f"pmdarima: order={pm_order} seasonal={pm_seasonal} AIC={pm_aic:.2f} time={pm_t:.2f}s")


if __name__ == "__main__":
    main()
