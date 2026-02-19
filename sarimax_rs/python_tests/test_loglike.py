"""Phase 1 integration tests: compare sarimax_rs.sarimax_loglike() vs statsmodels."""

import numpy as np
import sarimax_rs


TOL = 1e-6


def test_ar1_loglike(statsmodels_fixtures):
    """AR(1) concentrated loglike matches statsmodels."""
    case = statsmodels_fixtures["ar1"]
    y = np.array(case["data"])
    params = np.array(case["params"])
    expected = case["loglike"]

    ll = sarimax_rs.sarimax_loglike(y, (1, 0, 0), (0, 0, 0, 0), params)
    assert abs(ll - expected) < TOL, f"AR(1) loglike error: {abs(ll - expected):.2e}"


def test_arma11_loglike(statsmodels_fixtures):
    """ARMA(1,1) concentrated loglike matches statsmodels."""
    case = statsmodels_fixtures["arma11"]
    y = np.array(case["data"])
    params = np.array(case["params"])
    expected = case["loglike"]

    ll = sarimax_rs.sarimax_loglike(y, (1, 0, 1), (0, 0, 0, 0), params)
    assert abs(ll - expected) < TOL, f"ARMA(1,1) loglike error: {abs(ll - expected):.2e}"


def test_arima111_loglike(statsmodels_fixtures):
    """ARIMA(1,1,1) concentrated loglike matches statsmodels."""
    case = statsmodels_fixtures["arima111"]
    y = np.array(case["data"])
    params = np.array(case["params"])
    expected = case["loglike"]

    ll = sarimax_rs.sarimax_loglike(y, (1, 1, 1), (0, 0, 0, 0), params)
    assert abs(ll - expected) < TOL, f"ARIMA(1,1,1) loglike error: {abs(ll - expected):.2e}"


def test_concentrate_scale_default(statsmodels_fixtures):
    """Verify concentrate_scale=True is the default."""
    case = statsmodels_fixtures["ar1"]
    y = np.array(case["data"])
    params = np.array(case["params"])

    ll_default = sarimax_rs.sarimax_loglike(y, (1, 0, 0), (0, 0, 0, 0), params)
    ll_explicit = sarimax_rs.sarimax_loglike(
        y, (1, 0, 0), (0, 0, 0, 0), params, concentrate_scale=True
    )
    assert ll_default == ll_explicit
