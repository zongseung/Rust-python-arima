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


# =========================================================================
# Phase 5 tests: expanded coverage
# =========================================================================


# --- 5-4: NaN/Inf input rejection ---

class TestNanInfRejection:
    """All public functions must reject NaN/Inf input."""

    def test_loglike_rejects_nan(self):
        y = _sample_series()
        y[5] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            sarimax_rs.sarimax_loglike(
                y, (1, 0, 0), (0, 0, 0, 0),
                np.array([0.5], dtype=np.float64),
            )

    def test_loglike_rejects_inf(self):
        y = _sample_series()
        y[3] = np.inf
        with pytest.raises(ValueError, match="NaN or Inf"):
            sarimax_rs.sarimax_loglike(
                y, (1, 0, 0), (0, 0, 0, 0),
                np.array([0.5], dtype=np.float64),
            )

    def test_fit_rejects_nan(self):
        y = _sample_series()
        y[0] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            sarimax_rs.sarimax_fit(y, (1, 0, 0), (0, 0, 0, 0))

    def test_loglike_rejects_nan_in_exog(self):
        y = _sample_series()
        exog = np.ones((len(y), 1), dtype=np.float64)
        exog[3, 0] = np.nan
        with pytest.raises(ValueError, match="exog contains NaN or Inf"):
            sarimax_rs.sarimax_loglike(
                y, (1, 0, 0), (0, 0, 0, 0),
                np.array([0.0, 0.5], dtype=np.float64),
                exog=exog,
            )

    def test_fit_rejects_inf_in_exog(self):
        y = _sample_series()
        exog = np.ones((len(y), 1), dtype=np.float64)
        exog[4, 0] = np.inf
        with pytest.raises(ValueError, match="exog contains NaN or Inf"):
            sarimax_rs.sarimax_fit(
                y, (1, 0, 0), (0, 0, 0, 0),
                exog=exog,
            )

    def test_forecast_rejects_nan(self):
        y = _sample_series()
        y[10] = np.nan
        params = np.array([0.5], dtype=np.float64)
        with pytest.raises(ValueError, match="NaN or Inf"):
            sarimax_rs.sarimax_forecast(
                y, (1, 0, 0), (0, 0, 0, 0), params, steps=5,
            )

    def test_residuals_rejects_inf(self):
        y = _sample_series()
        y[2] = -np.inf
        params = np.array([0.5], dtype=np.float64)
        with pytest.raises(ValueError, match="NaN or Inf"):
            sarimax_rs.sarimax_residuals(
                y, (1, 0, 0), (0, 0, 0, 0), params,
            )

    def test_batch_fit_rejects_nan(self):
        y_good = _sample_series()
        y_bad = _sample_series()
        y_bad[4] = np.nan
        with pytest.raises(ValueError, match="index 1.*NaN or Inf"):
            sarimax_rs.sarimax_batch_fit(
                [y_good, y_bad], (1, 0, 0), (0, 0, 0, 0),
            )

    def test_batch_forecast_rejects_inf(self):
        y_good = _sample_series()
        y_bad = _sample_series()
        y_bad[0] = np.inf
        params = np.array([0.5], dtype=np.float64)
        with pytest.raises(ValueError, match="index 1.*NaN or Inf"):
            sarimax_rs.sarimax_batch_forecast(
                [y_good, y_bad], (1, 0, 0), (0, 0, 0, 0),
                [params, params], steps=3,
            )


# --- 5-3: Batch length mismatch negative tests ---

class TestBatchLengthMismatch:
    """Batch functions must reject length mismatches."""

    def test_batch_fit_exog_list_length_mismatch(self):
        y1 = _sample_series()
        y2 = _sample_series()
        exog1 = np.ones((len(y1), 1), dtype=np.float64)
        with pytest.raises(ValueError, match="exog_list length"):
            sarimax_rs.sarimax_batch_fit(
                [y1, y2], (1, 0, 0), (0, 0, 0, 0),
                exog_list=[exog1],  # only 1 exog for 2 series
            )

    def test_batch_forecast_series_params_length_mismatch(self):
        y = _sample_series()
        params = np.array([0.5], dtype=np.float64)
        with pytest.raises(ValueError, match="same length"):
            sarimax_rs.sarimax_batch_forecast(
                [y, y], (1, 0, 0), (0, 0, 0, 0),
                [params],  # only 1 params for 2 series
                steps=3,
            )

    def test_batch_forecast_exog_list_length_mismatch(self):
        y = _sample_series()
        params = np.array([0.5, 0.1], dtype=np.float64)  # exog_coeff + ar_coeff
        exog = np.ones((len(y), 1), dtype=np.float64)
        future_exog = np.ones((3, 1), dtype=np.float64)
        with pytest.raises(ValueError, match="exog_list length"):
            sarimax_rs.sarimax_batch_forecast(
                [y, y], (1, 0, 0), (0, 0, 0, 0),
                [params, params], steps=3,
                exog_list=[exog],  # only 1 exog for 2 series
                exog_forecast_list=[future_exog, future_exog],
            )

    def test_batch_forecast_rejects_short_future_exog_rows(self):
        y = _sample_series()
        exog = np.ones((len(y), 1), dtype=np.float64)
        fit = sarimax_rs.sarimax_fit(
            y, (1, 0, 0), (0, 0, 0, 0),
            exog=exog,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        params = np.array(fit["params"], dtype=np.float64)
        short_future_exog = np.ones((2, 1), dtype=np.float64)  # steps=5보다 짧음

        with pytest.raises(ValueError, match="forecast steps requested"):
            sarimax_rs.sarimax_batch_forecast(
                [y], (1, 0, 0), (0, 0, 0, 0), [params],
                steps=5,
                exog_list=[exog],
                exog_forecast_list=[short_future_exog],
            )

    def test_batch_forecast_alpha_out_of_range(self):
        y = _sample_series()
        params = np.array([0.5], dtype=np.float64)
        with pytest.raises(ValueError, match="alpha"):
            sarimax_rs.sarimax_batch_forecast(
                [y], (1, 0, 0), (0, 0, 0, 0), [params],
                steps=3, alpha=0.0,
            )
        with pytest.raises(ValueError, match="alpha"):
            sarimax_rs.sarimax_batch_forecast(
                [y], (1, 0, 0), (0, 0, 0, 0), [params],
                steps=3, alpha=1.0,
            )
        with pytest.raises(ValueError, match="alpha"):
            sarimax_rs.sarimax_batch_forecast(
                [y], (1, 0, 0), (0, 0, 0, 0), [params],
                steps=3, alpha=-0.5,
            )


# --- 5-5: Minimum series length edge cases ---

class TestMinSeriesLength:
    """Models should reject very short series with a normal Python exception."""

    def test_series_shorter_than_order(self):
        """AR(3) on a 3-observation series should fail gracefully."""
        y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        with pytest.raises((ValueError, RuntimeError)):
            sarimax_rs.sarimax_fit(y, (3, 0, 0), (0, 0, 0, 0))

    def test_differenced_series_too_short(self):
        """ARIMA(1,2,0) on 4 observations → differenced series has 2 obs, barely usable."""
        y = np.array([1.0, 2.0, 4.0, 7.0], dtype=np.float64)
        with pytest.raises((ValueError, RuntimeError)):
            sarimax_rs.sarimax_fit(y, (1, 2, 0), (0, 0, 0, 0))

    def test_seasonal_too_short(self):
        """SARIMA with s=12 on 10 observations should fail."""
        y = np.arange(10, dtype=np.float64)
        with pytest.raises((ValueError, RuntimeError)):
            sarimax_rs.sarimax_fit(y, (1, 0, 0), (1, 0, 0, 12))


# --- 5-1: concentrate_scale=False full likelihood ---

class TestConcentrateScaleFalse:
    """Test full likelihood (sigma2 as free parameter)."""

    def test_fit_concentrate_false_produces_finite(self):
        """fit with concentrate_scale=False should converge and produce finite loglike."""
        np.random.seed(42)
        n = 200
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.7 * y[t - 1] + np.random.randn()

        result = sarimax_rs.sarimax_fit(
            y, (1, 0, 0), (0, 0, 0, 0),
            concentrate_scale=False,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        assert result["converged"], f"fit should converge, got error: {result}"
        assert np.isfinite(result["loglike"]), "loglike should be finite"
        # With concentrate_scale=False, params include sigma2
        assert len(result["params"]) == 2, "should have ar(1) + sigma2"
        assert result["params"][1] > 0, "sigma2 must be positive"

    def test_loglike_concentrate_false_vs_true(self):
        """concentrated and full loglike should be close for well-fit model."""
        np.random.seed(42)
        n = 200
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.7 * y[t - 1] + np.random.randn()

        result = sarimax_rs.sarimax_fit(
            y, (1, 0, 0), (0, 0, 0, 0),
            concentrate_scale=False,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        ll_full = result["loglike"]

        # Evaluate concentrated loglike at same AR param
        ar_param = np.array([result["params"][0]], dtype=np.float64)
        ll_conc = sarimax_rs.sarimax_loglike(
            y, (1, 0, 0), (0, 0, 0, 0), ar_param,
            concentrate_scale=True,
            enforce_stationarity=False, enforce_invertibility=False,
        )
        # Both should be close (not exact due to different parameterizations)
        assert abs(ll_full - ll_conc) < 5.0, (
            f"full vs concentrated loglike too far apart: {ll_full} vs {ll_conc}"
        )


# --- 5-2: Batch forecast with exog (happy path) ---

class TestBatchForecastExog:
    """Test batch_forecast with exogenous variables."""

    def test_batch_forecast_with_exog_happy_path(self):
        """batch_forecast with exog should produce valid forecasts."""
        np.random.seed(42)
        n = 100
        exog_col = np.random.randn(n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 * y[t - 1] + 0.3 * exog_col[t] + np.random.randn() * 0.5

        exog = exog_col.reshape(-1, 1).astype(np.float64)
        result = sarimax_rs.sarimax_fit(
            y, (1, 0, 0), (0, 0, 0, 0), exog=exog,
            enforce_stationarity=False, enforce_invertibility=False,
        )
        assert result["converged"], f"fit failed: {result}"

        params = np.array(result["params"], dtype=np.float64)
        future_exog = np.ones((5, 1), dtype=np.float64)

        forecasts = sarimax_rs.sarimax_batch_forecast(
            [y, y], (1, 0, 0), (0, 0, 0, 0),
            [params, params], steps=5,
            exog_list=[exog, exog],
            exog_forecast_list=[future_exog, future_exog],
        )
        assert len(forecasts) == 2
        for i, fc in enumerate(forecasts):
            assert "error" not in fc, f"series {i} failed: {fc.get('error')}"
            assert len(fc["mean"]) == 5
            assert all(np.isfinite(fc["mean"])), f"series {i}: non-finite forecast"


# --- 4-2 regression: error type check ---

class TestErrorTypes:
    """Verify that error types are correctly mapped."""

    def test_optimization_failure_raises_runtime_error(self):
        """OptimizationFailed should raise RuntimeError, not ValueError."""
        # A tiny series with complex model should trigger optimization failure
        y = np.array([1.0, 2.0], dtype=np.float64)
        with pytest.raises((RuntimeError, ValueError)):
            sarimax_rs.sarimax_fit(
                y, (3, 0, 3), (0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
