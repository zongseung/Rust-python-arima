"""Phase 2 integration tests: compare sarimax_rs.sarimax_fit() vs statsmodels.

Note on tolerances:
Different optimizer implementations (Rust L-BFGS vs scipy L-BFGS-B) with different
starting points may converge to slightly different optima. For ARMA/ARIMA models the
likelihood surface can have near-flat ridges, so parameter estimates commonly differ
by O(1e-2) across implementations while the loglike values remain close.
We validate that:
1. The optimizer converges
2. Parameters are within a reasonable range of statsmodels MLE
3. The loglike is not significantly worse than statsmodels
"""

import numpy as np
import sarimax_rs


# Cross-implementation tolerances (Rust vs statsmodels/scipy)
PARAM_TOL = 1e-2       # 0.01 absolute parameter tolerance
LOGLIKE_TOL = 3.0      # loglike difference tolerance
AIC_BIC_TOL = 6.0      # information criteria tolerance


def test_fit_ar1(statsmodels_fixtures, fit_fixtures):
    """AR(1) fit: params and loglike match statsmodels."""
    case = statsmodels_fixtures["ar1"]
    ref = fit_fixtures["ar1"]
    y = np.array(case["data"])

    result = sarimax_rs.sarimax_fit(y, (1, 0, 0), (0, 0, 0, 0))

    assert result["converged"], "AR(1) fit should converge"
    assert abs(result["params"][0] - ref["params"][0]) < PARAM_TOL, (
        f"AR(1) param error: {abs(result['params'][0] - ref['params'][0]):.6f}"
    )
    assert abs(result["loglike"] - ref["loglike"]) < LOGLIKE_TOL, (
        f"AR(1) loglike error: {abs(result['loglike'] - ref['loglike']):.4f}"
    )


def test_fit_arma11(statsmodels_fixtures, fit_fixtures):
    """ARMA(1,1) fit: params match statsmodels."""
    case = statsmodels_fixtures["arma11"]
    ref = fit_fixtures["arma11"]
    y = np.array(case["data"])

    result = sarimax_rs.sarimax_fit(y, (1, 0, 1), (0, 0, 0, 0))

    assert result["converged"], "ARMA(1,1) fit should converge"
    for i, (got, exp) in enumerate(zip(result["params"], ref["params"])):
        err = abs(got - exp)
        assert err < PARAM_TOL, (
            f"ARMA(1,1) param[{i}] error: {err:.6f} (got={got:.6f}, exp={exp:.6f})"
        )


def test_fit_arima111(statsmodels_fixtures, fit_fixtures):
    """ARIMA(1,1,1) fit: loglike matches statsmodels."""
    case = statsmodels_fixtures["arima111"]
    ref = fit_fixtures["arima111"]
    y = np.array(case["data"])

    result = sarimax_rs.sarimax_fit(y, (1, 1, 1), (0, 0, 0, 0))

    assert abs(result["loglike"] - ref["loglike"]) < LOGLIKE_TOL, (
        f"ARIMA(1,1,1) loglike error: {abs(result['loglike'] - ref['loglike']):.4f}"
    )


def test_fit_sarima(statsmodels_fixtures, fit_fixtures):
    """SARIMA(1,0,0)(1,0,0,4) fit: params match statsmodels."""
    case = statsmodels_fixtures["sarima_100_100_4"]
    ref = fit_fixtures["sarima_100_100_4"]
    y = np.array(case["data"])

    result = sarimax_rs.sarimax_fit(y, (1, 0, 0), (1, 0, 0, 4))

    assert result["converged"], "SARIMA fit should converge"
    for i, (got, exp) in enumerate(zip(result["params"], ref["params"])):
        err = abs(got - exp)
        assert err < PARAM_TOL, (
            f"SARIMA param[{i}] error: {err:.6f} (got={got:.6f}, exp={exp:.6f})"
        )


def test_fit_aic_bic(statsmodels_fixtures, fit_fixtures):
    """AIC/BIC values match statsmodels."""
    case = statsmodels_fixtures["ar1"]
    ref = fit_fixtures["ar1"]
    y = np.array(case["data"])

    result = sarimax_rs.sarimax_fit(y, (1, 0, 0), (0, 0, 0, 0))

    assert abs(result["aic"] - ref["aic"]) < AIC_BIC_TOL, (
        f"AIC error: {abs(result['aic'] - ref['aic']):.4f}"
    )
    assert abs(result["bic"] - ref["bic"]) < AIC_BIC_TOL, (
        f"BIC error: {abs(result['bic'] - ref['bic']):.4f}"
    )


def test_fit_convergence(statsmodels_fixtures):
    """All standard models should converge."""
    case = statsmodels_fixtures["ar1"]
    y = np.array(case["data"])

    result = sarimax_rs.sarimax_fit(y, (1, 0, 0), (0, 0, 0, 0))
    assert result["converged"]
    assert result["n_iter"] > 0
    assert result["n_obs"] == len(y)


def test_fit_custom_start_params(statsmodels_fixtures):
    """Custom start_params should work."""
    case = statsmodels_fixtures["ar1"]
    y = np.array(case["data"])
    start = np.array([0.5])

    result = sarimax_rs.sarimax_fit(
        y, (1, 0, 0), (0, 0, 0, 0), start_params=start
    )
    assert np.isfinite(result["loglike"])
    assert np.isfinite(result["params"][0])


def test_fit_nelder_mead_method(statsmodels_fixtures, fit_fixtures):
    """Nelder-Mead method should also produce reasonable results."""
    case = statsmodels_fixtures["ar1"]
    ref = fit_fixtures["ar1"]
    y = np.array(case["data"])

    result = sarimax_rs.sarimax_fit(
        y, (1, 0, 0), (0, 0, 0, 0), method="nelder-mead"
    )
    assert "nelder-mead" in result["method"]
    assert abs(result["params"][0] - ref["params"][0]) < PARAM_TOL
    assert np.isfinite(result["loglike"])


def test_fit_returns_dict(statsmodels_fixtures):
    """Verify fit returns all expected keys."""
    case = statsmodels_fixtures["ar1"]
    y = np.array(case["data"])

    result = sarimax_rs.sarimax_fit(y, (1, 0, 0), (0, 0, 0, 0))

    expected_keys = {
        "params", "loglike", "scale", "aic", "bic",
        "n_obs", "n_params", "n_iter", "converged", "method"
    }
    assert set(result.keys()) == expected_keys
