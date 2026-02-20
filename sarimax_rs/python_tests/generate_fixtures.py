"""Generate statsmodels reference fixtures for Kalman filter validation.

Usage:
    uv run python python_tests/generate_fixtures.py

Produces: tests/fixtures/statsmodels_reference.json
"""

import json
import pathlib

import numpy as np
import statsmodels.api as sm


def generate_ar1_data(n=200, seed=42):
    """AR(1) with phi=0.7."""
    np.random.seed(seed)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.7 * y[t - 1] + np.random.randn()
    return y


def generate_arma11_data(n=200, seed=42):
    """ARMA(1,1) with phi=0.5, theta=0.3."""
    np.random.seed(seed)
    e = np.random.randn(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + e[t] + 0.3 * e[t - 1]
    return y


def generate_arima111_data(n=300, seed=123):
    """Random walk + AR/MA structure."""
    np.random.seed(seed)
    return np.cumsum(np.random.randn(n))


def generate_sarima_data(n=200, s=4, seed=99):
    """Seasonal data for SARIMA(1,0,0)(1,0,0,s) testing."""
    np.random.seed(seed)
    y = np.zeros(n)
    for t in range(s, n):
        y[t] = 0.5 * y[t - 1] + 0.4 * y[t - s] + np.random.randn()
    return y


def generate_sarima_full_data(n=300, s=12, seed=77):
    """Seasonal data with differencing for SARIMA(1,1,1)(1,1,1,s) testing."""
    np.random.seed(seed)
    return np.cumsum(np.cumsum(np.random.randn(n)).reshape(-1, 1).repeat(1, axis=1).flatten())


def extract_state_space_matrices(res):
    """Extract T, Z, R matrices from a fitted SARIMAX result."""
    ss = res.filter_results
    T = ss.transition[:, :, 0].tolist()
    Z = ss.design[:, :, 0].tolist()
    R = ss.selection[:, :, 0].tolist()
    return T, Z, R


def fit_and_extract(y, order, seasonal_order=(0, 0, 0, 0)):
    """Fit SARIMAX model and extract reference values."""
    model = sm.tsa.SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
        concentrate_scale=True,
    )
    res = model.fit(disp=False)

    T, Z, R = extract_state_space_matrices(res)

    # Compute loglike at fitted params
    loglike = res.llf

    # Also compute loglike using model.loglike() to double-check
    loglike_check = model.loglike(res.params)

    return {
        "params": res.params.tolist(),
        "loglike": float(loglike),
        "loglike_check": float(loglike_check),
        "scale": float(res.scale),
        "T": T,
        "Z": Z,
        "R": R,
        "nobs": int(res.nobs),
    }


def fit_and_extract_full(y, order, seasonal_order=(0, 0, 0, 0)):
    """Fit SARIMAX model with enforcement and extract full fit reference values."""
    model = sm.tsa.SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        trend="n",
        enforce_stationarity=True,
        enforce_invertibility=True,
        concentrate_scale=True,
    )
    res = model.fit(disp=False)

    return {
        "params": res.params.tolist(),
        "loglike": float(res.llf),
        "scale": float(res.scale),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "nobs": int(res.nobs),
    }


def main():
    fixtures = {}

    # --- AR(1) ---
    y_ar1 = generate_ar1_data()
    fixtures["ar1"] = {
        "data": y_ar1.tolist(),
        "order": [1, 0, 0],
        "seasonal_order": [0, 0, 0, 0],
        **fit_and_extract(y_ar1, (1, 0, 0)),
    }

    # --- ARMA(1,1) ---
    y_arma11 = generate_arma11_data()
    fixtures["arma11"] = {
        "data": y_arma11.tolist(),
        "order": [1, 0, 1],
        "seasonal_order": [0, 0, 0, 0],
        **fit_and_extract(y_arma11, (1, 0, 1)),
    }

    # --- ARIMA(1,1,1) ---
    y_arima111 = generate_arima111_data()
    fixtures["arima111"] = {
        "data": y_arima111.tolist(),
        "order": [1, 1, 1],
        "seasonal_order": [0, 0, 0, 0],
        **fit_and_extract(y_arima111, (1, 1, 1)),
    }

    # --- SARIMA(1,0,0)(1,0,0,4) ---
    y_sar = generate_sarima_data(n=200, s=4, seed=99)
    fixtures["sarima_100_100_4"] = {
        "data": y_sar.tolist(),
        "order": [1, 0, 0],
        "seasonal_order": [1, 0, 0, 4],
        **fit_and_extract(y_sar, (1, 0, 0), (1, 0, 0, 4)),
    }

    # --- SARIMA(1,1,1)(1,1,1,12) ---
    y_full = generate_sarima_full_data(n=300, s=12, seed=77)
    fixtures["sarima_111_111_12"] = {
        "data": y_full.tolist(),
        "order": [1, 1, 1],
        "seasonal_order": [1, 1, 1, 12],
        **fit_and_extract(y_full, (1, 1, 1), (1, 1, 1, 12)),
    }

    # Write to JSON
    out_dir = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "statsmodels_reference.json"

    with open(out_path, "w") as f:
        json.dump(fixtures, f, indent=2)

    print(f"Fixtures written to {out_path}")
    print(f"  AR(1) loglike:        {fixtures['ar1']['loglike']:.6f}")
    print(f"  ARMA(1,1) loglike:    {fixtures['arma11']['loglike']:.6f}")
    print(f"  ARIMA(1,1,1) loglike: {fixtures['arima111']['loglike']:.6f}")
    print(f"  SARIMA(1,0,0)(1,0,0,4) loglike:   {fixtures['sarima_100_100_4']['loglike']:.6f}")
    print(f"  SARIMA(1,1,1)(1,1,1,12) loglike:   {fixtures['sarima_111_111_12']['loglike']:.6f}")

    # --- Phase 3: Forecast reference values ---
    forecast_fixtures = {}

    for name, order, seasonal_order, data in [
        ("ar1", (1, 0, 0), (0, 0, 0, 0), y_ar1),
        ("arma11", (1, 0, 1), (0, 0, 0, 0), y_arma11),
        ("arima111", (1, 1, 1), (0, 0, 0, 0), y_arima111),
        ("sarima_100_100_4", (1, 0, 0), (1, 0, 0, 4), y_sar),
    ]:
        model = sm.tsa.SARIMAX(
            data,
            order=order,
            seasonal_order=seasonal_order,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
            concentrate_scale=True,
        )
        res = model.fit(disp=False)
        fcast = res.get_forecast(steps=10)
        ci = fcast.conf_int(alpha=0.05)

        # Standardized residuals
        resid = res.filter_results.standardized_forecasts_error[0]

        forecast_fixtures[name] = {
            "params": res.params.tolist(),
            "forecast_mean": fcast.predicted_mean.tolist(),
            "forecast_ci_lower": ci[:, 0].tolist() if isinstance(ci, np.ndarray) else ci.iloc[:, 0].tolist(),
            "forecast_ci_upper": ci[:, 1].tolist() if isinstance(ci, np.ndarray) else ci.iloc[:, 1].tolist(),
            "standardized_residuals": resid.tolist(),
        }

    forecast_path = out_dir / "statsmodels_forecast_reference.json"
    with open(forecast_path, "w") as f:
        json.dump(forecast_fixtures, f, indent=2)

    print(f"\nForecast fixtures written to {forecast_path}")
    for name, vals in forecast_fixtures.items():
        print(f"  {name}: forecast[0:3]={vals['forecast_mean'][:3]}")

    # --- Phase 2: Fit reference values ---
    fit_fixtures = {}

    fit_fixtures["ar1"] = fit_and_extract_full(y_ar1, (1, 0, 0))
    fit_fixtures["arma11"] = fit_and_extract_full(y_arma11, (1, 0, 1))
    fit_fixtures["arima111"] = fit_and_extract_full(y_arima111, (1, 1, 1))
    fit_fixtures["sarima_100_100_4"] = fit_and_extract_full(y_sar, (1, 0, 0), (1, 0, 0, 4))

    fit_path = out_dir / "statsmodels_fit_reference.json"
    with open(fit_path, "w") as f:
        json.dump(fit_fixtures, f, indent=2)

    print(f"\nFit fixtures written to {fit_path}")
    for name, vals in fit_fixtures.items():
        print(f"  {name}: params={vals['params']}, loglike={vals['loglike']:.6f}, aic={vals['aic']:.2f}, bic={vals['bic']:.2f}")


if __name__ == "__main__":
    main()
