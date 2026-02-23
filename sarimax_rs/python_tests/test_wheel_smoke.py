"""Wheel smoke tests — basic import and functionality verification.

These tests verify that a built wheel works correctly after installation.
They should run fast (<5s) and not require statsmodels.
"""

import sys

import numpy as np
import pytest


class TestWheelSmoke:
    """Basic smoke tests for the installed wheel."""

    def test_import_sarimax_rs(self):
        """sarimax_rs module can be imported."""
        import sarimax_rs  # noqa: F401

    def test_version_string(self):
        """version() returns a non-empty string."""
        import sarimax_rs
        v = sarimax_rs.version()
        assert isinstance(v, str)
        assert len(v) > 0

    def test_basic_fit(self):
        """AR(1) fit completes and converges."""
        import sarimax_rs
        np.random.seed(42)
        y = np.zeros(100)
        for t in range(1, 100):
            y[t] = 0.5 * y[t - 1] + np.random.randn()

        result = sarimax_rs.sarimax_fit(y, (1, 0, 0), (0, 0, 0, 0))
        assert result["converged"]
        assert np.isfinite(result["loglike"])

    def test_basic_forecast(self):
        """Forecast produces finite values."""
        import sarimax_rs
        np.random.seed(42)
        y = np.zeros(100)
        for t in range(1, 100):
            y[t] = 0.5 * y[t - 1] + np.random.randn()

        result = sarimax_rs.sarimax_fit(y, (1, 0, 0), (0, 0, 0, 0))
        params = np.array(result["params"])
        fc = sarimax_rs.sarimax_forecast(
            y, (1, 0, 0), (0, 0, 0, 0), params, steps=10,
        )
        assert len(fc["mean"]) == 10
        assert all(np.isfinite(fc["mean"]))

    def test_batch_fit(self):
        """Batch fit returns correct number of results."""
        import sarimax_rs
        np.random.seed(42)
        y = np.random.randn(100)
        results = sarimax_rs.sarimax_batch_fit(
            [y, y], (1, 0, 0), (0, 0, 0, 0),
        )
        assert len(results) == 2

    def test_no_statsmodels_dependency(self):
        """sarimax_rs import does not require statsmodels."""
        # This test verifies the module can work independently
        import sarimax_rs
        # If we got here, import succeeded without statsmodels
        assert hasattr(sarimax_rs, "sarimax_fit")
        assert hasattr(sarimax_rs, "sarimax_forecast")
        assert hasattr(sarimax_rs, "sarimax_loglike")
        assert hasattr(sarimax_rs, "sarimax_residuals")
        assert hasattr(sarimax_rs, "sarimax_batch_fit")
        assert hasattr(sarimax_rs, "sarimax_batch_forecast")
        assert hasattr(sarimax_rs, "sarimax_batch_loglike")
        assert hasattr(sarimax_rs, "version")

    def test_model_wrapper_import(self):
        """SARIMAXModel can be imported from sarimax_py."""
        from sarimax_py import SARIMAXModel  # noqa: F401

    def test_model_wrapper_fit_forecast(self):
        """SARIMAXModel end-to-end fit + forecast."""
        from sarimax_py import SARIMAXModel
        np.random.seed(42)
        y = np.zeros(100)
        for t in range(1, 100):
            y[t] = 0.5 * y[t - 1] + np.random.randn()

        model = SARIMAXModel(y, order=(1, 0, 0))
        result = model.fit()
        assert result.converged
        fc = result.forecast(steps=5)
        assert len(fc.predicted_mean) == 5
        assert all(np.isfinite(fc.predicted_mean))
