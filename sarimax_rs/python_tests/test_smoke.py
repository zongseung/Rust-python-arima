"""Smoke tests — import, version, basic end-to-end sanity, and model wrapper.

Covers module-level smoke (fast, no fixtures), wheel install checks,
and SARIMAXModel Python orchestration layer validation.
All tests must complete in <5s and must NOT require statsmodels.
"""

import numpy as np
import pytest


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
    from conftest import generate_ar1_rng
    y = generate_ar1_rng(n=100, phi=0.5, seed=42)
    result = sarimax_rs.sarimax_fit(y, (1, 0, 0), (0, 0, 0, 0))
    assert result["converged"]
    assert np.isfinite(result["loglike"])


def test_basic_forecast():
    """Forecast produces finite values of correct length."""
    import sarimax_rs
    from conftest import generate_ar1_rng
    y = generate_ar1_rng(n=100, phi=0.5, seed=42)
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
    from conftest import generate_ar1_rng
    y = generate_ar1_rng(n=100, phi=0.5, seed=42)
    result = SARIMAXModel(y, order=(1, 0, 0)).fit()
    assert result.converged
    fc = result.forecast(steps=5)
    assert len(fc.predicted_mean) == 5
    assert all(np.isfinite(fc.predicted_mean))


# ---------------------------------------------------------------------------
# Model wrapper detailed tests (merged from test_model.py)
# ---------------------------------------------------------------------------

def _make_ar1():
    from conftest import generate_ar1
    return generate_ar1()


def test_model_get_forecast_alias():
    """get_forecast() should produce identical results to forecast()."""
    from sarimax_py.model import SARIMAXModel
    y = _make_ar1()
    result = SARIMAXModel(y, order=(1, 0, 0)).fit()
    f1 = result.forecast(steps=5)
    f2 = result.get_forecast(steps=5)
    np.testing.assert_array_equal(f1.predicted_mean, f2.predicted_mean)
    np.testing.assert_array_equal(f1.variance, f2.variance)


def test_model_residuals():
    """result.resid should have length > 0 and finite values."""
    from sarimax_py.model import SARIMAXModel
    y = _make_ar1()
    result = SARIMAXModel(y, order=(1, 0, 0)).fit()
    resid = result.resid
    assert len(resid) > 0
    assert all(np.isfinite(resid))


def test_model_summary_string():
    """summary() should contain expected keywords."""
    from sarimax_py.model import SARIMAXModel
    y = _make_ar1()
    result = SARIMAXModel(y, order=(1, 0, 0)).fit()
    s = result.summary()
    assert isinstance(s, str)
    for keyword in ("SARIMAX Results", "Model:", "Log Likelihood:", "AIC:", "BIC:", "Converged:", "coef", "Scale"):
        assert keyword in s, f"Missing keyword: {keyword}"


def test_model_aic_bic():
    """AIC and BIC should be finite floats."""
    from sarimax_py.model import SARIMAXModel
    y = _make_ar1()
    result = SARIMAXModel(y, order=(1, 0, 0)).fit()
    assert isinstance(result.aic, float) and np.isfinite(result.aic)
    assert isinstance(result.bic, float) and np.isfinite(result.bic)


def test_model_conf_int():
    """conf_int() should return (steps, 2) shaped array with lower <= upper."""
    from sarimax_py.model import SARIMAXModel
    y = _make_ar1()
    result = SARIMAXModel(y, order=(1, 0, 0)).fit()
    ci = result.forecast(steps=7).conf_int()
    assert ci.shape == (7, 2)
    for i in range(7):
        assert ci[i, 0] <= ci[i, 1], f"step {i}: ci_lower > ci_upper"


def test_model_matches_raw_api():
    """SARIMAXModel results should match sarimax_rs raw API."""
    import sarimax_rs
    from sarimax_py.model import SARIMAXModel
    y = _make_ar1()
    result = SARIMAXModel(y, order=(1, 0, 0)).fit()
    raw = sarimax_rs.sarimax_fit(y, (1, 0, 0), (0, 0, 0, 0))
    assert abs(result.llf - raw["loglike"]) < 1e-10
    np.testing.assert_allclose(result.params, raw["params"], atol=1e-10)


def test_forecast_result_attributes():
    """ForecastResult should have predicted_mean, variance, ci_lower, ci_upper."""
    from sarimax_py.model import SARIMAXModel
    y = _make_ar1()
    fcast = SARIMAXModel(y, order=(1, 0, 0)).fit().forecast(steps=3)
    for attr in ("predicted_mean", "variance", "ci_lower", "ci_upper"):
        assert hasattr(fcast, attr)
        assert len(getattr(fcast, attr)) == 3
