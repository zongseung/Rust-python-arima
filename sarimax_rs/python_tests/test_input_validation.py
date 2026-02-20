import numpy as np
import pytest

import sarimax_rs


def _sample_series(n: int = 30) -> np.ndarray:
    return np.linspace(0.0, 1.0, n, dtype=np.float64)


def test_loglike_rejects_wrong_param_length():
    y = _sample_series()
    with pytest.raises(ValueError, match="params length mismatch"):
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


def test_exog_not_implemented_explicit_error():
    y = _sample_series()
    exog = np.ones_like(y)

    with pytest.raises(NotImplementedError, match="not yet supported"):
        sarimax_rs.sarimax_loglike(
            y,
            (1, 0, 0),
            (0, 0, 0, 0),
            np.array([0.1], dtype=np.float64),
            exog=exog,
        )

    with pytest.raises(NotImplementedError, match="not yet supported"):
        sarimax_rs.sarimax_fit(
            y,
            (1, 0, 0),
            (0, 0, 0, 0),
            exog=exog,
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
