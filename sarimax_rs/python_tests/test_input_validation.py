import numpy as np
import pytest

import sarimax_rs


def _sample_series(n: int = 30) -> np.ndarray:
    return np.linspace(0.0, 1.0, n, dtype=np.float64)


def test_loglike_rejects_wrong_param_length():
    y = _sample_series()
    with pytest.raises(ValueError, match="parameter length mismatch"):
        sarimax_rs.sarimax_loglike(
            y,
            (1, 0, 1),
            (0, 0, 0, 0),
            np.array([0.2], dtype=np.float64),  # should be length 2
        )


def test_fit_rejects_wrong_start_params_length():
    y = _sample_series()
    with pytest.raises(ValueError, match="parameter length mismatch"):
        sarimax_rs.sarimax_fit(
            y,
            (1, 0, 1),
            (0, 0, 0, 0),
            start_params=np.array([0.2], dtype=np.float64),  # should be length 2
        )


def test_rejects_invalid_seasonal_d_s_combination():
    y = _sample_series()
    with pytest.raises(ValueError, match="requires seasonal period s >= 2"):
        sarimax_rs.sarimax_loglike(
            y,
            (0, 0, 0),
            (0, 1, 0, 0),  # D>0 with s=0
            np.array([], dtype=np.float64),
        )


def test_exog_basic_acceptance():
    """Verify that exog (2D array) is accepted and produces valid results."""
    y = _sample_series()
    n = len(y)
    # Create a 2D exog array: (n_obs, 1)
    exog = np.ones((n, 1), dtype=np.float64)

    # loglike with exog: params = [exog_coeff(1), ar(1)] = 2 params
    ll = sarimax_rs.sarimax_loglike(
        y,
        (1, 0, 0),
        (0, 0, 0, 0),
        np.array([0.0, 0.1], dtype=np.float64),  # [exog_beta, ar_phi]
        exog=exog,
    )
    assert np.isfinite(ll), f"loglike with exog should be finite, got {ll}"

    # fit with exog
    result = sarimax_rs.sarimax_fit(
        y,
        (1, 0, 0),
        (0, 0, 0, 0),
        exog=exog,
    )
    assert result["converged"], "fit with exog should converge"
    assert len(result["params"]) == 2, "should have exog_coeff + ar_coeff"


def test_forecast_rejects_missing_future_exog():
    """Bug #1: exog model forecast must require future_exog."""
    y = _sample_series(n=100)
    exog = np.ones((len(y), 1), dtype=np.float64)
    result = sarimax_rs.sarimax_fit(
        y, (1, 0, 0), (0, 0, 0, 0), exog=exog,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    params = np.array(result["params"], dtype=np.float64)
    with pytest.raises(ValueError, match="exog_forecast is required"):
        sarimax_rs.sarimax_forecast(
            y, (1, 0, 0), (0, 0, 0, 0), params,
            steps=5, exog=exog,
            # exog_forecast intentionally omitted
        )


def test_fit_rejects_non_positive_sigma2_when_not_concentrated():
    y = _sample_series()
    with pytest.raises(ValueError, match="variance sigma2 must be positive"):
        sarimax_rs.sarimax_fit(
            y,
            (1, 0, 0),
            (0, 0, 0, 0),
            start_params=np.array([0.1, -1.0], dtype=np.float64),  # ar(1), sigma2
            concentrate_scale=False,
        )
