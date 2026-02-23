"""API contract tests for all 8 PyO3 functions.

Verifies return types, dict keys, dtypes, and error handling.
These tests ensure the public API surface is stable and well-defined.
"""

import numpy as np
import pytest

import sarimax_rs


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ar1_data():
    """Simple AR(1) data for contract testing."""
    np.random.seed(42)
    y = np.zeros(200)
    for t in range(1, 200):
        y[t] = 0.7 * y[t - 1] + np.random.randn()
    return y


@pytest.fixture(scope="module")
def ar1_fit_result(ar1_data):
    """Pre-fitted AR(1) result."""
    return sarimax_rs.sarimax_fit(ar1_data, (1, 0, 0), (0, 0, 0, 0))


@pytest.fixture(scope="module")
def ar1_params(ar1_fit_result):
    """Fitted params for AR(1)."""
    return np.array(ar1_fit_result["params"])


# ---------------------------------------------------------------------------
# 1. version()
# ---------------------------------------------------------------------------

class TestVersionContract:
    """sarimax_rs.version() contract."""

    def test_returns_string(self):
        v = sarimax_rs.version()
        assert isinstance(v, str)

    def test_version_format(self):
        v = sarimax_rs.version()
        parts = v.split(".")
        assert len(parts) >= 2, f"Version should be semver-like: {v}"


# ---------------------------------------------------------------------------
# 2. sarimax_loglike()
# ---------------------------------------------------------------------------

class TestLoglikeContract:
    """sarimax_rs.sarimax_loglike() contract."""

    def test_returns_float(self, ar1_data, ar1_params):
        result = sarimax_rs.sarimax_loglike(
            ar1_data, (1, 0, 0), (0, 0, 0, 0), ar1_params,
        )
        assert isinstance(result, float)

    def test_result_is_finite(self, ar1_data, ar1_params):
        result = sarimax_rs.sarimax_loglike(
            ar1_data, (1, 0, 0), (0, 0, 0, 0), ar1_params,
        )
        assert np.isfinite(result)

    def test_wrong_param_length_raises(self, ar1_data):
        with pytest.raises((ValueError, RuntimeError)):
            sarimax_rs.sarimax_loglike(
                ar1_data, (1, 0, 0), (0, 0, 0, 0),
                np.array([0.5, 0.3]),  # 2 params for AR(1) = wrong
            )

    def test_accepts_exog_none(self, ar1_data, ar1_params):
        result = sarimax_rs.sarimax_loglike(
            ar1_data, (1, 0, 0), (0, 0, 0, 0), ar1_params, exog=None,
        )
        assert np.isfinite(result)


# ---------------------------------------------------------------------------
# 3. sarimax_fit()
# ---------------------------------------------------------------------------

class TestFitContract:
    """sarimax_rs.sarimax_fit() contract."""

    REQUIRED_KEYS = {
        "params", "loglike", "aic", "bic", "scale",
        "converged", "method", "n_obs", "n_iter",
    }

    def test_returns_dict(self, ar1_data):
        result = sarimax_rs.sarimax_fit(ar1_data, (1, 0, 0), (0, 0, 0, 0))
        assert isinstance(result, dict)

    def test_required_keys_present(self, ar1_fit_result):
        missing = self.REQUIRED_KEYS - set(ar1_fit_result.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_params_is_list_of_floats(self, ar1_fit_result):
        params = ar1_fit_result["params"]
        assert isinstance(params, list)
        assert all(isinstance(p, float) for p in params)

    def test_loglike_is_float(self, ar1_fit_result):
        assert isinstance(ar1_fit_result["loglike"], float)

    def test_aic_bic_are_floats(self, ar1_fit_result):
        assert isinstance(ar1_fit_result["aic"], float)
        assert isinstance(ar1_fit_result["bic"], float)

    def test_scale_is_float(self, ar1_fit_result):
        assert isinstance(ar1_fit_result["scale"], float)

    def test_converged_is_bool(self, ar1_fit_result):
        assert isinstance(ar1_fit_result["converged"], bool)

    def test_method_is_string(self, ar1_fit_result):
        assert isinstance(ar1_fit_result["method"], str)

    def test_n_obs_is_int(self, ar1_fit_result):
        assert isinstance(ar1_fit_result["n_obs"], int)

    def test_n_iter_is_int(self, ar1_fit_result):
        assert isinstance(ar1_fit_result["n_iter"], int)

    def test_params_length_matches_order(self, ar1_data):
        # AR(2): should have 2 params
        result = sarimax_rs.sarimax_fit(ar1_data, (2, 0, 0), (0, 0, 0, 0))
        assert len(result["params"]) == 2

    def test_n_obs_matches_data_length(self, ar1_data, ar1_fit_result):
        assert ar1_fit_result["n_obs"] == len(ar1_data)


# ---------------------------------------------------------------------------
# 4. sarimax_forecast()
# ---------------------------------------------------------------------------

class TestForecastContract:
    """sarimax_rs.sarimax_forecast() contract."""

    REQUIRED_KEYS = {"mean", "variance", "ci_lower", "ci_upper"}

    def test_returns_dict(self, ar1_data, ar1_params):
        result = sarimax_rs.sarimax_forecast(
            ar1_data, (1, 0, 0), (0, 0, 0, 0), ar1_params, steps=5,
        )
        assert isinstance(result, dict)

    def test_required_keys_present(self, ar1_data, ar1_params):
        result = sarimax_rs.sarimax_forecast(
            ar1_data, (1, 0, 0), (0, 0, 0, 0), ar1_params, steps=5,
        )
        missing = self.REQUIRED_KEYS - set(result.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_output_length_matches_steps(self, ar1_data, ar1_params):
        for steps in [1, 5, 10, 20]:
            result = sarimax_rs.sarimax_forecast(
                ar1_data, (1, 0, 0), (0, 0, 0, 0), ar1_params, steps=steps,
            )
            for key in self.REQUIRED_KEYS:
                assert len(result[key]) == steps, f"{key} length != {steps}"

    def test_mean_is_list_of_floats(self, ar1_data, ar1_params):
        result = sarimax_rs.sarimax_forecast(
            ar1_data, (1, 0, 0), (0, 0, 0, 0), ar1_params, steps=5,
        )
        assert isinstance(result["mean"], list)
        assert all(isinstance(v, float) for v in result["mean"])

    def test_variance_non_negative(self, ar1_data, ar1_params):
        result = sarimax_rs.sarimax_forecast(
            ar1_data, (1, 0, 0), (0, 0, 0, 0), ar1_params, steps=10,
        )
        assert all(v >= 0 for v in result["variance"])


# ---------------------------------------------------------------------------
# 5. sarimax_residuals()
# ---------------------------------------------------------------------------

class TestResidualsContract:
    """sarimax_rs.sarimax_residuals() contract."""

    REQUIRED_KEYS = {"residuals", "standardized_residuals"}

    def test_returns_dict(self, ar1_data, ar1_params):
        result = sarimax_rs.sarimax_residuals(
            ar1_data, (1, 0, 0), (0, 0, 0, 0), ar1_params,
        )
        assert isinstance(result, dict)

    def test_required_keys_present(self, ar1_data, ar1_params):
        result = sarimax_rs.sarimax_residuals(
            ar1_data, (1, 0, 0), (0, 0, 0, 0), ar1_params,
        )
        missing = self.REQUIRED_KEYS - set(result.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_residuals_length_matches_data(self, ar1_data, ar1_params):
        result = sarimax_rs.sarimax_residuals(
            ar1_data, (1, 0, 0), (0, 0, 0, 0), ar1_params,
        )
        assert len(result["residuals"]) == len(ar1_data)
        assert len(result["standardized_residuals"]) == len(ar1_data)

    def test_residuals_are_list_of_floats(self, ar1_data, ar1_params):
        result = sarimax_rs.sarimax_residuals(
            ar1_data, (1, 0, 0), (0, 0, 0, 0), ar1_params,
        )
        assert isinstance(result["residuals"], list)
        assert all(isinstance(v, float) for v in result["residuals"])


# ---------------------------------------------------------------------------
# 6. sarimax_batch_loglike()
# ---------------------------------------------------------------------------

class TestBatchLoglikeContract:
    """sarimax_rs.sarimax_batch_loglike() contract."""

    def test_returns_list(self, ar1_data, ar1_params):
        result = sarimax_rs.sarimax_batch_loglike(
            [ar1_data, ar1_data], (1, 0, 0), (0, 0, 0, 0), ar1_params,
        )
        assert isinstance(result, list)

    def test_output_length_matches_input(self, ar1_data, ar1_params):
        series = [ar1_data] * 5
        result = sarimax_rs.sarimax_batch_loglike(
            series, (1, 0, 0), (0, 0, 0, 0), ar1_params,
        )
        assert len(result) == 5

    def test_each_element_is_dict_with_loglike(self, ar1_data, ar1_params):
        result = sarimax_rs.sarimax_batch_loglike(
            [ar1_data], (1, 0, 0), (0, 0, 0, 0), ar1_params,
        )
        assert isinstance(result[0], dict)
        assert "loglike" in result[0]
        assert isinstance(result[0]["loglike"], float)


# ---------------------------------------------------------------------------
# 7. sarimax_batch_fit()
# ---------------------------------------------------------------------------

class TestBatchFitContract:
    """sarimax_rs.sarimax_batch_fit() contract."""

    def test_returns_list_of_dicts(self, ar1_data):
        result = sarimax_rs.sarimax_batch_fit(
            [ar1_data, ar1_data], (1, 0, 0), (0, 0, 0, 0),
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, dict) for r in result)

    def test_each_dict_has_fit_keys(self, ar1_data):
        result = sarimax_rs.sarimax_batch_fit(
            [ar1_data], (1, 0, 0), (0, 0, 0, 0),
        )
        required = {"params", "loglike", "converged"}
        for r in result:
            missing = required - set(r.keys())
            assert not missing, f"Missing keys: {missing}"

    def test_empty_input_returns_empty(self):
        result = sarimax_rs.sarimax_batch_fit(
            [], (1, 0, 0), (0, 0, 0, 0),
        )
        assert result == []


# ---------------------------------------------------------------------------
# 8. sarimax_batch_forecast()
# ---------------------------------------------------------------------------

class TestBatchForecastContract:
    """sarimax_rs.sarimax_batch_forecast() contract."""

    def test_returns_list_of_dicts(self, ar1_data, ar1_params):
        result = sarimax_rs.sarimax_batch_forecast(
            [ar1_data], (1, 0, 0), (0, 0, 0, 0),
            [ar1_params], steps=5,
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)

    def test_each_dict_has_forecast_keys(self, ar1_data, ar1_params):
        result = sarimax_rs.sarimax_batch_forecast(
            [ar1_data], (1, 0, 0), (0, 0, 0, 0),
            [ar1_params], steps=5,
        )
        required = {"mean", "variance", "ci_lower", "ci_upper"}
        for r in result:
            missing = required - set(r.keys())
            assert not missing, f"Missing keys: {missing}"

    def test_output_length_matches_input(self, ar1_data, ar1_params):
        series = [ar1_data] * 3
        params = [ar1_params] * 3
        result = sarimax_rs.sarimax_batch_forecast(
            series, (1, 0, 0), (0, 0, 0, 0), params, steps=5,
        )
        assert len(result) == 3

    def test_empty_input_returns_empty(self):
        result = sarimax_rs.sarimax_batch_forecast(
            [], (1, 0, 0), (0, 0, 0, 0), [], steps=5,
        )
        assert result == []
