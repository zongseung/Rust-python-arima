"""
Tests for exogenous variable (exog) support in sarimax_rs.

Validates:
1. loglike with exog returns finite values
2. fit with exog converges and recovers reasonable coefficients
3. forecast with exog and future_exog works correctly
4. residuals with exog works correctly
5. batch operations with per-series exog
6. Comparison against statsmodels for accuracy
"""
import numpy as np
import pytest

import sarimax_rs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_arx_data(n=300, phi=0.5, beta=2.0, seed=42):
    """Generate y_t = beta * x_t + phi * y_{t-1} + eps_t"""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    eps = rng.standard_normal(n) * 0.5
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = beta * x[t] + phi * y[t - 1] + eps[t]
    exog = x.reshape(-1, 1).astype(np.float64)
    return y.astype(np.float64), exog


def _generate_multi_exog_data(n=300, n_exog=3, phi=0.5, seed=42):
    """Generate y_t = sum(beta_j * x_j_t) + phi * y_{t-1} + eps_t"""
    rng = np.random.default_rng(seed)
    betas = np.array([1.5, -0.8, 0.3])[:n_exog]
    X = rng.standard_normal((n, n_exog))
    eps = rng.standard_normal(n) * 0.5
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = X[t] @ betas + phi * y[t - 1] + eps[t]
    return y.astype(np.float64), X.astype(np.float64), betas


# ---------------------------------------------------------------------------
# loglike tests
# ---------------------------------------------------------------------------

class TestExogLoglike:
    def test_loglike_with_single_exog(self):
        """loglike with 1 exog variable returns finite value."""
        y, exog = _generate_arx_data()
        # params: [exog_beta(1), ar(1)]
        params = np.array([2.0, 0.5], dtype=np.float64)
        ll = sarimax_rs.sarimax_loglike(
            y, (1, 0, 0), (0, 0, 0, 0), params, exog=exog
        )
        assert np.isfinite(ll), f"loglike not finite: {ll}"

    def test_loglike_with_multi_exog(self):
        """loglike with multiple exog variables returns finite value."""
        y, X, _ = _generate_multi_exog_data(n_exog=3)
        # params: [exog(3), ar(1)]
        params = np.array([1.5, -0.8, 0.3, 0.5], dtype=np.float64)
        ll = sarimax_rs.sarimax_loglike(
            y, (1, 0, 0), (0, 0, 0, 0), params, exog=X
        )
        assert np.isfinite(ll), f"loglike not finite: {ll}"

    def test_loglike_without_exog_unchanged(self):
        """loglike without exog is unchanged (backward compatible)."""
        y, _ = _generate_arx_data()
        params = np.array([0.5], dtype=np.float64)
        ll = sarimax_rs.sarimax_loglike(
            y, (1, 0, 0), (0, 0, 0, 0), params
        )
        assert np.isfinite(ll)

    def test_loglike_exog_improves_fit(self):
        """With true exog, loglike should be better than without."""
        y, exog = _generate_arx_data(beta=2.0)
        # With exog (true model)
        params_with = np.array([2.0, 0.5], dtype=np.float64)
        ll_with = sarimax_rs.sarimax_loglike(
            y, (1, 0, 0), (0, 0, 0, 0), params_with, exog=exog
        )
        # Without exog (misspecified)
        params_without = np.array([0.5], dtype=np.float64)
        ll_without = sarimax_rs.sarimax_loglike(
            y, (1, 0, 0), (0, 0, 0, 0), params_without
        )
        assert ll_with > ll_without, (
            f"loglike with exog ({ll_with}) should be better than without ({ll_without})"
        )


# ---------------------------------------------------------------------------
# fit tests
# ---------------------------------------------------------------------------

class TestExogFit:
    def test_fit_with_single_exog(self):
        """fit with 1 exog converges and returns correct param count."""
        y, exog = _generate_arx_data()
        result = sarimax_rs.sarimax_fit(
            y, (1, 0, 0), (0, 0, 0, 0), exog=exog
        )
        assert result["converged"], "fit with exog should converge"
        # params: [exog(1), ar(1)]
        assert len(result["params"]) == 2
        assert np.isfinite(result["loglike"])

    def test_fit_recovers_exog_coeff(self):
        """fit recovers the true exog coefficient approximately."""
        y, exog = _generate_arx_data(n=500, phi=0.5, beta=2.0, seed=123)
        result = sarimax_rs.sarimax_fit(
            y, (1, 0, 0), (0, 0, 0, 0), exog=exog,
            enforce_stationarity=False, enforce_invertibility=False,
        )
        assert result["converged"]
        beta_hat = result["params"][0]
        phi_hat = result["params"][1]
        assert abs(beta_hat - 2.0) < 0.5, f"beta_hat={beta_hat}, expected ~2.0"
        assert abs(phi_hat - 0.5) < 0.3, f"phi_hat={phi_hat}, expected ~0.5"

    def test_fit_multi_exog(self):
        """fit with 3 exog variables converges."""
        y, X, betas = _generate_multi_exog_data(n=500, n_exog=3, seed=99)
        result = sarimax_rs.sarimax_fit(
            y, (1, 0, 0), (0, 0, 0, 0), exog=X,
            enforce_stationarity=False, enforce_invertibility=False,
        )
        assert result["converged"]
        # params: [exog(3), ar(1)] = 4
        assert len(result["params"]) == 4
        assert np.isfinite(result["loglike"])

    def test_fit_arma_with_exog(self):
        """fit ARMA(1,1) with exog converges."""
        y, exog = _generate_arx_data(n=300, seed=77)
        result = sarimax_rs.sarimax_fit(
            y, (1, 0, 1), (0, 0, 0, 0), exog=exog,
            enforce_stationarity=False, enforce_invertibility=False,
        )
        assert result["converged"]
        # params: [exog(1), ar(1), ma(1)] = 3
        assert len(result["params"]) == 3


# ---------------------------------------------------------------------------
# forecast tests
# ---------------------------------------------------------------------------

class TestExogForecast:
    def test_forecast_with_future_exog(self):
        """forecast with exog and future_exog returns correct shape."""
        y, exog = _generate_arx_data(n=200, seed=55)
        # Fit first
        result = sarimax_rs.sarimax_fit(
            y, (1, 0, 0), (0, 0, 0, 0), exog=exog,
            enforce_stationarity=False, enforce_invertibility=False,
        )
        params = np.array(result["params"], dtype=np.float64)
        steps = 10
        future_exog = np.ones((steps, 1), dtype=np.float64) * 0.5

        fcast = sarimax_rs.sarimax_forecast(
            y, (1, 0, 0), (0, 0, 0, 0), params,
            steps=steps, exog=exog, exog_forecast=future_exog,
        )
        assert len(fcast["mean"]) == steps
        assert all(np.isfinite(fcast["mean"]))
        assert all(np.isfinite(fcast["variance"]))
        assert all(np.array(fcast["ci_lower"]) < np.array(fcast["ci_upper"]))

    def test_forecast_exog_affects_mean(self):
        """Different future exog values produce different forecast means."""
        y, exog = _generate_arx_data(n=200, beta=2.0, seed=55)
        result = sarimax_rs.sarimax_fit(
            y, (1, 0, 0), (0, 0, 0, 0), exog=exog,
            enforce_stationarity=False, enforce_invertibility=False,
        )
        params = np.array(result["params"], dtype=np.float64)
        steps = 5

        # High exog values
        future_high = np.ones((steps, 1), dtype=np.float64) * 5.0
        fcast_high = sarimax_rs.sarimax_forecast(
            y, (1, 0, 0), (0, 0, 0, 0), params,
            steps=steps, exog=exog, exog_forecast=future_high,
        )
        # Low exog values
        future_low = np.ones((steps, 1), dtype=np.float64) * (-5.0)
        fcast_low = sarimax_rs.sarimax_forecast(
            y, (1, 0, 0), (0, 0, 0, 0), params,
            steps=steps, exog=exog, exog_forecast=future_low,
        )
        # Means should differ significantly if beta > 0
        diff = np.array(fcast_high["mean"]) - np.array(fcast_low["mean"])
        assert np.all(np.abs(diff) > 0.1), (
            f"Forecast means should differ with different exog: diff={diff}"
        )


# ---------------------------------------------------------------------------
# residuals tests
# ---------------------------------------------------------------------------

class TestExogResiduals:
    def test_residuals_with_exog(self):
        """residuals with exog returns correct length."""
        y, exog = _generate_arx_data(n=200, seed=33)
        result = sarimax_rs.sarimax_fit(
            y, (1, 0, 0), (0, 0, 0, 0), exog=exog,
            enforce_stationarity=False, enforce_invertibility=False,
        )
        params = np.array(result["params"], dtype=np.float64)

        resid = sarimax_rs.sarimax_residuals(
            y, (1, 0, 0), (0, 0, 0, 0), params, exog=exog
        )
        assert len(resid["residuals"]) == len(y)
        assert len(resid["standardized_residuals"]) == len(y)
        assert all(np.isfinite(resid["residuals"]))


# ---------------------------------------------------------------------------
# batch tests
# ---------------------------------------------------------------------------

class TestExogBatch:
    def test_batch_fit_with_exog(self):
        """batch_fit with per-series exog converges."""
        y1, exog1 = _generate_arx_data(n=200, seed=1)
        y2, exog2 = _generate_arx_data(n=200, seed=2)

        results = sarimax_rs.sarimax_batch_fit(
            [y1, y2],
            (1, 0, 0), (0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
            exog_list=[exog1, exog2],
        )
        assert len(results) == 2
        for i, r in enumerate(results):
            assert "error" not in r, f"series {i} failed: {r.get('error')}"
            assert r["converged"], f"series {i} did not converge"
            assert len(r["params"]) == 2  # [exog(1), ar(1)]


# ---------------------------------------------------------------------------
# statsmodels comparison
# ---------------------------------------------------------------------------

class TestExogStatsmodelsComparison:
    @pytest.fixture
    def arx_data(self):
        return _generate_arx_data(n=300, phi=0.5, beta=2.0, seed=42)

    def test_loglike_vs_statsmodels(self, arx_data):
        """Compare loglike with statsmodels for same params."""
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            pytest.skip("statsmodels not installed")

        y, exog = arx_data

        # Fit with statsmodels
        sm_model = SARIMAX(y, order=(1, 0, 0), exog=exog,
                           enforce_stationarity=False,
                           enforce_invertibility=False,
                           concentrate_scale=True)
        sm_result = sm_model.fit(disp=False)
        sm_params = sm_result.params  # [exog(1), ar(1)]
        sm_ll = sm_result.llf_obs.sum()

        # Evaluate same params in sarimax_rs
        rs_ll = sarimax_rs.sarimax_loglike(
            y, (1, 0, 0), (0, 0, 0, 0),
            np.array(sm_params, dtype=np.float64),
            exog=exog,
        )

        # Should be close (not exact due to initialization differences;
        # approximate_diffuse vs exact diffuse can cause ~5-10 unit discrepancy)
        assert abs(rs_ll - sm_ll) < 15.0, (
            f"loglike mismatch: rs={rs_ll}, sm={sm_ll}, diff={abs(rs_ll - sm_ll)}"
        )

    def test_fit_params_vs_statsmodels(self, arx_data):
        """Compare fitted params with statsmodels."""
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            pytest.skip("statsmodels not installed")

        y, exog = arx_data

        # statsmodels fit
        sm_model = SARIMAX(y, order=(1, 0, 0), exog=exog,
                           enforce_stationarity=False,
                           enforce_invertibility=False,
                           concentrate_scale=True)
        sm_result = sm_model.fit(disp=False)
        sm_params = sm_result.params

        # sarimax_rs fit
        rs_result = sarimax_rs.sarimax_fit(
            y, (1, 0, 0), (0, 0, 0, 0), exog=exog,
            enforce_stationarity=False, enforce_invertibility=False,
        )
        rs_params = np.array(rs_result["params"])

        # Compare exog coefficient
        assert abs(rs_params[0] - sm_params[0]) < 0.5, (
            f"exog coeff: rs={rs_params[0]}, sm={sm_params[0]}"
        )
        # Compare AR coefficient
        assert abs(rs_params[1] - sm_params[1]) < 0.3, (
            f"ar coeff: rs={rs_params[1]}, sm={sm_params[1]}"
        )
