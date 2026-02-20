"""Phase 3 integration tests: compare sarimax_rs forecast/residuals vs statsmodels.

Validates:
1. Forecast mean matches statsmodels get_forecast()
2. Confidence intervals match statsmodels conf_int()
3. Standardized residuals match statsmodels filter_results
"""

import numpy as np
import sarimax_rs


# Cross-implementation tolerances
FORECAST_TOL = 1e-4    # forecast mean tolerance
CI_TOL = 1e-3          # confidence interval tolerance
RESID_TOL = 1e-4       # standardized residual tolerance


def test_forecast_ar1_vs_statsmodels(statsmodels_fixtures, forecast_fixtures):
    """AR(1) 10-step forecast vs statsmodels."""
    case = statsmodels_fixtures["ar1"]
    ref = forecast_fixtures["ar1"]
    y = np.array(case["data"])
    params = np.array(ref["params"])

    result = sarimax_rs.sarimax_forecast(
        y, (1, 0, 0), (0, 0, 0, 0), params, steps=10, alpha=0.05
    )

    for i, (got, exp) in enumerate(zip(result["mean"], ref["forecast_mean"])):
        err = abs(got - exp)
        assert err < FORECAST_TOL, (
            f"AR(1) forecast[{i}] error: {err:.8f} (got={got:.6f}, exp={exp:.6f})"
        )


def test_forecast_arma11_vs_statsmodels(statsmodels_fixtures, forecast_fixtures):
    """ARMA(1,1) 10-step forecast vs statsmodels."""
    case = statsmodels_fixtures["arma11"]
    ref = forecast_fixtures["arma11"]
    y = np.array(case["data"])
    params = np.array(ref["params"])

    result = sarimax_rs.sarimax_forecast(
        y, (1, 0, 1), (0, 0, 0, 0), params, steps=10, alpha=0.05
    )

    for i, (got, exp) in enumerate(zip(result["mean"], ref["forecast_mean"])):
        err = abs(got - exp)
        assert err < FORECAST_TOL, (
            f"ARMA(1,1) forecast[{i}] error: {err:.8f} (got={got:.6f}, exp={exp:.6f})"
        )


def test_forecast_arima111_vs_statsmodels(statsmodels_fixtures, forecast_fixtures):
    """ARIMA(1,1,1) 10-step forecast vs statsmodels."""
    case = statsmodels_fixtures["arima111"]
    ref = forecast_fixtures["arima111"]
    y = np.array(case["data"])
    params = np.array(ref["params"])

    result = sarimax_rs.sarimax_forecast(
        y, (1, 1, 1), (0, 0, 0, 0), params, steps=10, alpha=0.05
    )

    # ARIMA with differencing may have slightly larger errors
    for i, (got, exp) in enumerate(zip(result["mean"], ref["forecast_mean"])):
        err = abs(got - exp)
        assert err < 1e-3, (
            f"ARIMA(1,1,1) forecast[{i}] error: {err:.8f} (got={got:.6f}, exp={exp:.6f})"
        )


def test_forecast_ci_vs_statsmodels(statsmodels_fixtures, forecast_fixtures):
    """AR(1) confidence intervals vs statsmodels."""
    case = statsmodels_fixtures["ar1"]
    ref = forecast_fixtures["ar1"]
    y = np.array(case["data"])
    params = np.array(ref["params"])

    result = sarimax_rs.sarimax_forecast(
        y, (1, 0, 0), (0, 0, 0, 0), params, steps=10, alpha=0.05
    )

    for i in range(10):
        lower_err = abs(result["ci_lower"][i] - ref["forecast_ci_lower"][i])
        upper_err = abs(result["ci_upper"][i] - ref["forecast_ci_upper"][i])
        assert lower_err < CI_TOL, (
            f"AR(1) CI lower[{i}] error: {lower_err:.8f}"
        )
        assert upper_err < CI_TOL, (
            f"AR(1) CI upper[{i}] error: {upper_err:.8f}"
        )


def test_residuals_vs_statsmodels(statsmodels_fixtures, forecast_fixtures):
    """Standardized residuals vs statsmodels (after burn-in)."""
    case = statsmodels_fixtures["ar1"]
    ref = forecast_fixtures["ar1"]
    y = np.array(case["data"])
    params = np.array(ref["params"])

    result = sarimax_rs.sarimax_residuals(
        y, (1, 0, 0), (0, 0, 0, 0), params
    )

    # Compare after burn-in (k_states = 1 for AR(1))
    burn = 1
    got = result["standardized_residuals"][burn:]
    exp = ref["standardized_residuals"][burn:]

    for i, (g, e) in enumerate(zip(got, exp)):
        err = abs(g - e)
        assert err < RESID_TOL, (
            f"Std residual[{i + burn}] error: {err:.8f} (got={g:.6f}, exp={e:.6f})"
        )


def test_forecast_returns_dict(statsmodels_fixtures):
    """Verify forecast returns all expected keys."""
    case = statsmodels_fixtures["ar1"]
    y = np.array(case["data"])
    params = np.array(case["params"])

    result = sarimax_rs.sarimax_forecast(
        y, (1, 0, 0), (0, 0, 0, 0), params, steps=5
    )

    expected_keys = {"mean", "variance", "ci_lower", "ci_upper"}
    assert set(result.keys()) == expected_keys
    assert len(result["mean"]) == 5
    assert len(result["variance"]) == 5
    assert len(result["ci_lower"]) == 5
    assert len(result["ci_upper"]) == 5


def test_residuals_returns_dict(statsmodels_fixtures):
    """Verify residuals returns all expected keys."""
    case = statsmodels_fixtures["ar1"]
    y = np.array(case["data"])
    params = np.array(case["params"])

    result = sarimax_rs.sarimax_residuals(
        y, (1, 0, 0), (0, 0, 0, 0), params
    )

    expected_keys = {"residuals", "standardized_residuals"}
    assert set(result.keys()) == expected_keys
    assert len(result["residuals"]) == len(y)
    assert len(result["standardized_residuals"]) == len(y)


def test_forecast_variance_increasing(statsmodels_fixtures):
    """Forecast variance should be non-decreasing for stationary models."""
    case = statsmodels_fixtures["ar1"]
    y = np.array(case["data"])
    params = np.array(case["params"])

    result = sarimax_rs.sarimax_forecast(
        y, (1, 0, 0), (0, 0, 0, 0), params, steps=20
    )

    for i in range(1, len(result["variance"])):
        assert result["variance"][i] >= result["variance"][i - 1] - 1e-10, (
            f"Variance not non-decreasing: v[{i}]={result['variance'][i]} < v[{i-1}]={result['variance'][i-1]}"
        )


def test_forecast_ci_symmetric(statsmodels_fixtures):
    """Confidence intervals should be symmetric around mean."""
    case = statsmodels_fixtures["ar1"]
    y = np.array(case["data"])
    params = np.array(case["params"])

    result = sarimax_rs.sarimax_forecast(
        y, (1, 0, 0), (0, 0, 0, 0), params, steps=10
    )

    for i in range(10):
        lower_dist = result["mean"][i] - result["ci_lower"][i]
        upper_dist = result["ci_upper"][i] - result["mean"][i]
        assert abs(lower_dist - upper_dist) < 1e-10, (
            f"CI not symmetric at step {i}: lower_dist={lower_dist}, upper_dist={upper_dist}"
        )
