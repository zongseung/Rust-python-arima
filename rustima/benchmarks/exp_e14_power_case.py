"""E14 ★ Case study: hourly electricity demand with exog covariates.

Loads power_demand_2024.csv, constructs simple exog (hour-of-day, day-of-week),
runs rustima auto_arima vs statsmodels SARIMAX with a chosen order, and reports
runtime, AIC, and out-of-sample MAPE on a hold-out tail.

Outputs:
- raw/e14_power_case.csv
- figures/e14_forecast.png
- tables/e14_power_case.tex
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rustima
from statsmodels.tsa.statespace.sarimax import SARIMAX

from thesis_utils import FIGURES_DIR, Timer, csv_to_latex, save_csv, setup_mpl

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parent.parent / "power_demand_2024.csv"


def load_data():
    df = pd.read_csv(DATA)
    # Try common column names
    if "demand" in df.columns:
        y = df["demand"].to_numpy(dtype=np.float64)
    elif "load" in df.columns:
        y = df["load"].to_numpy(dtype=np.float64)
    else:
        # Take last numeric column
        num_cols = df.select_dtypes(include=[np.number]).columns
        y = df[num_cols[-1]].to_numpy(dtype=np.float64)
    # Build simple exog: sin/cos of hour-of-day, day-of-week
    n = len(y)
    hour = np.arange(n) % 24
    dow = (np.arange(n) // 24) % 7
    X = np.column_stack([
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7),
        np.cos(2 * np.pi * dow / 7),
    ]).astype(np.float64)
    return y, X


def main():
    y, X = load_data()
    n = len(y)
    print(f"Loaded n={n}")

    # Subsample to 2000 hours (~83 days) for reasonable test runtime
    if n > 2000:
        y = y[:2000]
        X = X[:2000]
        n = 2000

    # 90/10 train/test split
    n_train = int(n * 0.9)
    y_train, y_test = y[:n_train], y[n_train:]
    X_train, X_test = X[:n_train], X[n_train:]
    h = len(y_test)
    print(f"Train: n_train={n_train}, Test: h={h}")

    # Pre-detrend to avoid integer demand levels overwhelming the model
    y_mean = y_train.mean()
    y_train_c = y_train - y_mean
    y_test_c = y_test - y_mean

    order = (2, 0, 1)
    sorder = (1, 0, 1, 24)
    rows = []

    # rustima — lbfgsb
    print("\nrustima lbfgsb fit...")
    try:
        with Timer() as t:
            r = rustima.sarimax_fit(
                y_train_c, order, sorder,
                exog=X_train,
                method="lbfgsb",
                concentrate_scale=True,
                maxiter=200,
            )
        rs_ll = r["loglike"]
        rs_aic = r["aic"]
        rs_iter = r.get("n_iter", -1)
        rs_time = t.ms
        rs_params = r["params"]
        # Forecast
        fcst = rustima.sarimax_forecast(
            y_train_c, order, sorder,
            np.array(rs_params, dtype=np.float64),
            h, exog=X_train, future_exog=X_test,
            concentrate_scale=True,
        )
        rs_mean = np.array(fcst["mean"]) if isinstance(fcst, dict) and "mean" in fcst else (np.array(fcst["forecast"]) if isinstance(fcst, dict) else np.array(fcst))
        rs_rmse = float(np.sqrt(np.mean((y_test_c - rs_mean) ** 2)))
        rs_mape = float(np.mean(np.abs((y_test_c + y_mean - rs_mean - y_mean) / (y_test_c + y_mean))) * 100)
        rows.append(["rustima (lbfgsb)", f"{rs_ll:.2f}", f"{rs_aic:.2f}", rs_iter, f"{rs_time:.0f}", f"{rs_rmse:.2f}", f"{rs_mape:.2f}"])
        print(f"  LL={rs_ll:.2f}  AIC={rs_aic:.2f}  time={rs_time:.0f}ms  RMSE={rs_rmse:.2f}  MAPE={rs_mape:.2f}%")
    except Exception as e:
        rows.append(["rustima (lbfgsb)", "ERR", "—", "—", "—", str(e)[:30]])
        print(f"  ERR: {e}")

    # rustima — PTR
    print("\nrustima PTR fit...")
    try:
        with Timer() as t:
            r = rustima.sarimax_fit(
                y_train_c, order, sorder,
                exog=X_train,
                method="profile-trust-region",
                concentrate_scale=True,
                maxiter=200,
            )
        ll = r["loglike"]
        aic = r["aic"]
        n_iter = r.get("n_iter", -1)
        t_ms = t.ms
        params = r["params"]
        fcst = rustima.sarimax_forecast(
            y_train_c, order, sorder,
            np.array(params, dtype=np.float64),
            h, exog=X_train, future_exog=X_test,
            concentrate_scale=True,
        )
        mean = np.array(fcst["mean"]) if isinstance(fcst, dict) and "mean" in fcst else (np.array(fcst["forecast"]) if isinstance(fcst, dict) else np.array(fcst))
        rmse_ptr = float(np.sqrt(np.mean((y_test_c - mean) ** 2)))
        mape = float(np.mean(np.abs((y_test_c + y_mean - mean - y_mean) / (y_test_c + y_mean))) * 100)
        rows.append(["rustima (PTR ★)", f"{ll:.2f}", f"{aic:.2f}", n_iter, f"{t_ms:.0f}", f"{rmse_ptr:.2f}", f"{mape:.2f}"])
        print(f"  LL={ll:.2f}  AIC={aic:.2f}  time={t_ms:.0f}ms  RMSE={rmse_ptr:.2f}  MAPE={mape:.2f}%")
    except Exception as e:
        rows.append(["rustima (PTR ★)", "ERR", "—", "—", "—", str(e)[:30]])
        print(f"  ERR: {e}")

    # statsmodels
    print("\nstatsmodels SARIMAX fit...")
    try:
        with Timer() as t:
            m = SARIMAX(y_train_c, order=order, seasonal_order=sorder, exog=X_train, simple_differencing=False).fit(disp=False, maxiter=200)
        ll = m.llf
        aic = m.aic
        n_iter = m.mle_retvals.get("iterations", -1) if hasattr(m, "mle_retvals") else -1
        t_ms = t.ms
        fcst = m.get_forecast(steps=h, exog=X_test)
        sm_mean = fcst.predicted_mean
        if hasattr(sm_mean, "values"):
            sm_mean = sm_mean.values
        sm_rmse = float(np.sqrt(np.mean((y_test_c - sm_mean) ** 2)))
        sm_mape = float(np.mean(np.abs((y_test_c + y_mean - sm_mean - y_mean) / (y_test_c + y_mean))) * 100)
        rows.append(["statsmodels", f"{ll:.2f}", f"{aic:.2f}", n_iter, f"{t_ms:.0f}", f"{sm_rmse:.2f}", f"{sm_mape:.2f}"])
        print(f"  LL={ll:.2f}  AIC={aic:.2f}  time={t_ms:.0f}ms  RMSE={sm_rmse:.2f}  MAPE={sm_mape:.2f}%")
    except Exception as e:
        rows.append(["statsmodels", "ERR", "—", "—", "—", str(e)[:30]])
        print(f"  ERR: {e}")

    save_csv(
        "e14_power_case.csv",
        ["engine", "loglike", "aic", "iter", "time_ms", "rmse", "mape_pct"],
        rows,
    )
    csv_to_latex(
        "e14_power_case.csv",
        "e14_power_case.tex",
        "Hourly electricity-demand case study: SARIMAX$(2,0,1)(1,0,1)_{24}$ with hour/day-of-week exogenous regressors. 1{,}800 train / 200 test hours.",
        "tab:rustima:e14-power",
    )

    # Save forecast figure if at least one method succeeded
    plt = setup_mpl()
    fig, ax = plt.subplots(figsize=(8, 3))
    t_axis = np.arange(h)
    ax.plot(t_axis, y_test_c + y_mean, "k-", label="actual", linewidth=1.0)
    try:
        ax.plot(t_axis, rs_mean + y_mean, "C0--", label="rustima (lbfgsb)", linewidth=1.0)
    except Exception:
        pass
    ax.set_xlabel("hour (test set)")
    ax.set_ylabel("demand")
    ax.set_title("Hourly demand forecast (h=200)")
    ax.legend()
    fig.savefig(FIGURES_DIR / "e14_forecast.png")
    plt.close(fig)
    print("\nFigure saved.")


if __name__ == "__main__":
    main()
