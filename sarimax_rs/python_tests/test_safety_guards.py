"""Phase 11 — Regression guard tests for runtime safety hardening.

Tests cover:
- White-noise (0,0,0) model with all optimizer methods
- Exog row-length mismatch boundary (early ValueError at PyO3 boundary)
- Unknown trend string rejection (ValueError instead of silent fallback)
"""

import numpy as np
import pytest

import sarimax_rs


# ---------------------------------------------------------------------------
# 11-1. White-noise (0,0,0) + all optimizer methods
# ---------------------------------------------------------------------------

@pytest.fixture
def white_noise_data():
    """100-point white noise series."""
    return np.random.default_rng(42).normal(size=100)


@pytest.mark.parametrize("method", ["lbfgsb", "lbfgsb-multi", "lbfgsb-strict", "nelder-mead"])
def test_white_noise_000_all_methods(white_noise_data, method):
    """ARIMA(0,0,0) — zero free params with concentrate_scale — should return
    a valid result without crashing for every optimizer method."""
    result = sarimax_rs.sarimax_fit(
        white_noise_data,
        order=(0, 0, 0),
        seasonal=(0, 0, 0, 0),
        method=method,
    )
    assert np.isfinite(result["loglike"]), f"loglike not finite for method={method}"
    assert result["n_obs"] == len(white_noise_data)


@pytest.mark.parametrize("method", ["lbfgsb", "lbfgsb-multi", "nelder-mead"])
def test_white_noise_000_with_trend_c(white_noise_data, method):
    """ARIMA(0,0,0) with constant trend — has 1 free param (intercept)."""
    result = sarimax_rs.sarimax_fit(
        white_noise_data,
        order=(0, 0, 0),
        seasonal=(0, 0, 0, 0),
        method=method,
        trend="c",
    )
    assert np.isfinite(result["loglike"])
    assert len(result["params"]) >= 1  # at least the trend coeff


def test_white_noise_000_maxiter_0(white_noise_data):
    """maxiter=0 with zero-param model — should return immediately."""
    result = sarimax_rs.sarimax_fit(
        white_noise_data,
        order=(0, 0, 0),
        seasonal=(0, 0, 0, 0),
        maxiter=0,
    )
    assert np.isfinite(result["loglike"])


# ---------------------------------------------------------------------------
# 11-2. Exog row-length mismatch boundary tests
# ---------------------------------------------------------------------------

def test_exog_row_mismatch_single_fit():
    """Exog with wrong number of rows should raise ValueError immediately."""
    y = np.random.default_rng(42).normal(size=100)
    exog_wrong = np.random.default_rng(43).normal(size=(80, 2))  # 80 != 100

    with pytest.raises(ValueError, match="exog has 80 rows but endog has 100"):
        sarimax_rs.sarimax_fit(
            y,
            order=(1, 0, 0),
            seasonal=(0, 0, 0, 0),
            exog=exog_wrong,
        )


def test_exog_row_mismatch_single_loglike():
    """Exog mismatch in loglike should also raise early."""
    y = np.random.default_rng(42).normal(size=100)
    exog_wrong = np.random.default_rng(43).normal(size=(50, 1))
    params = np.array([0.1, 0.5])  # dummy params

    with pytest.raises(ValueError, match="exog has 50 rows but endog has 100"):
        sarimax_rs.sarimax_loglike(
            y,
            order=(1, 0, 0),
            seasonal=(0, 0, 0, 0),
            params=params,
            exog=exog_wrong,
        )


def test_exog_row_mismatch_single_forecast():
    """Exog mismatch in forecast should raise early."""
    y = np.random.default_rng(42).normal(size=100)
    exog_wrong = np.random.default_rng(43).normal(size=(90, 1))
    params = np.array([0.1, 0.5])

    with pytest.raises(ValueError, match="exog has 90 rows but endog has 100"):
        sarimax_rs.sarimax_forecast(
            y,
            order=(1, 0, 0),
            seasonal=(0, 0, 0, 0),
            params=params,
            exog=exog_wrong,
        )


# ---------------------------------------------------------------------------
# 9-3. Unknown trend string rejection
# ---------------------------------------------------------------------------

def test_unknown_trend_raises_valueerror():
    """Unknown trend string should raise ValueError, not silently fallback."""
    y = np.random.default_rng(42).normal(size=100)

    with pytest.raises(ValueError, match="unknown trend"):
        sarimax_rs.sarimax_fit(
            y,
            order=(1, 0, 0),
            seasonal=(0, 0, 0, 0),
            trend="xyz",
        )


def test_unknown_trend_raises_for_loglike():
    """Unknown trend in loglike should also raise."""
    y = np.random.default_rng(42).normal(size=100)
    params = np.array([0.5])

    with pytest.raises(ValueError, match="unknown trend"):
        sarimax_rs.sarimax_loglike(
            y,
            order=(1, 0, 0),
            seasonal=(0, 0, 0, 0),
            params=params,
            trend="invalid_trend",
        )


@pytest.mark.parametrize("trend", ["n", "c", "t", "ct"])
def test_valid_trends_accepted(trend):
    """All valid trend strings should work without error."""
    y = np.random.default_rng(42).normal(size=100)
    result = sarimax_rs.sarimax_fit(
        y,
        order=(1, 0, 0),
        seasonal=(0, 0, 0, 0),
        trend=trend,
    )
    assert np.isfinite(result["loglike"])
