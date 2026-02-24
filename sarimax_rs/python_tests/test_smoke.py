"""Smoke tests — import, version, and basic end-to-end sanity.

Covers both module-level smoke (fast, no fixtures) and wheel install checks.
All tests must complete in <5s and must NOT require statsmodels.
"""

import numpy as np


def test_import():
    """sarimax_rs module can be imported."""
    import sarimax_rs
    assert sarimax_rs is not None


def test_version():
    """version() returns a non-empty string."""
    import sarimax_rs
    v = sarimax_rs.version()
    assert isinstance(v, str) and len(v) > 0


def test_public_api_surface():
    """All expected public functions are present."""
    import sarimax_rs
    expected = [
        "sarimax_fit", "sarimax_forecast", "sarimax_loglike",
        "sarimax_residuals", "sarimax_batch_fit", "sarimax_batch_forecast",
        "sarimax_batch_loglike", "version",
    ]
    for name in expected:
        assert hasattr(sarimax_rs, name), f"missing: {name}"


def test_basic_fit():
    """AR(1) fit completes and converges."""
    import sarimax_rs
    rng = np.random.default_rng(42)
    y = np.zeros(100)
    for t in range(1, 100):
        y[t] = 0.5 * y[t - 1] + rng.normal()
    result = sarimax_rs.sarimax_fit(y, (1, 0, 0), (0, 0, 0, 0))
    assert result["converged"]
    assert np.isfinite(result["loglike"])


def test_basic_forecast():
    """Forecast produces finite values of correct length."""
    import sarimax_rs
    rng = np.random.default_rng(42)
    y = np.zeros(100)
    for t in range(1, 100):
        y[t] = 0.5 * y[t - 1] + rng.normal()
    result = sarimax_rs.sarimax_fit(y, (1, 0, 0), (0, 0, 0, 0))
    fc = sarimax_rs.sarimax_forecast(
        y, (1, 0, 0), (0, 0, 0, 0), np.array(result["params"]), steps=10,
    )
    assert len(fc["mean"]) == 10
    assert all(np.isfinite(fc["mean"]))


def test_batch_fit_length():
    """Batch fit returns correct number of results."""
    import sarimax_rs
    y = np.random.default_rng(42).normal(size=100)
    results = sarimax_rs.sarimax_batch_fit([y, y], (1, 0, 0), (0, 0, 0, 0))
    assert len(results) == 2


def test_model_wrapper_end_to_end():
    """SARIMAXModel fit + forecast end-to-end."""
    from sarimax_py import SARIMAXModel
    rng = np.random.default_rng(42)
    y = np.zeros(100)
    for t in range(1, 100):
        y[t] = 0.5 * y[t - 1] + rng.normal()
    result = SARIMAXModel(y, order=(1, 0, 0)).fit()
    assert result.converged
    fc = result.forecast(steps=5)
    assert len(fc.predicted_mean) == 5
    assert all(np.isfinite(fc.predicted_mean))
