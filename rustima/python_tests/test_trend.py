"""Tests for trend parameter support (P0: A-2, A-3, A-4, A-5)."""
import numpy as np
import pytest
from conftest import generate_trend_data, generate_ar1_rng
from sarimax_py import SARIMAXModel


@pytest.fixture
def trend_data():
    return generate_trend_data()


@pytest.fixture
def simple_data():
    return generate_ar1_rng()


class TestTrendFit:
    def test_trend_none(self, simple_data):
        model = SARIMAXModel(simple_data, order=(1, 0, 0), trend="n")
        result = model.fit()
        assert result.converged
        assert len(result.params) == 1  # ar.L1 only

    def test_trend_constant(self, simple_data):
        model = SARIMAXModel(simple_data, order=(1, 0, 0), trend="c")
        result = model.fit()
        assert result.converged
        assert len(result.params) == 2  # intercept + ar.L1

    def test_trend_linear(self, trend_data):
        model = SARIMAXModel(trend_data, order=(1, 0, 0), trend="t")
        result = model.fit()
        assert result.converged
        assert len(result.params) == 2  # drift + ar.L1

    def test_trend_both(self, trend_data):
        model = SARIMAXModel(trend_data, order=(1, 0, 0), trend="ct")
        result = model.fit()
        assert result.converged
        assert len(result.params) == 3  # intercept + drift + ar.L1


class TestTrendParamNames:
    def test_names_none(self, simple_data):
        model = SARIMAXModel(simple_data, order=(1, 0, 0), trend="n")
        result = model.fit()
        assert result.param_names == ["ar.L1"]

    def test_names_constant(self, simple_data):
        model = SARIMAXModel(simple_data, order=(1, 0, 0), trend="c")
        result = model.fit()
        assert result.param_names == ["intercept", "ar.L1"]

    def test_names_linear(self, trend_data):
        model = SARIMAXModel(trend_data, order=(1, 0, 0), trend="t")
        result = model.fit()
        assert result.param_names == ["drift", "ar.L1"]

    def test_names_both(self, trend_data):
        model = SARIMAXModel(trend_data, order=(1, 0, 0), trend="ct")
        result = model.fit()
        assert result.param_names == ["intercept", "drift", "ar.L1"]

    def test_names_arma_with_trend(self, trend_data):
        model = SARIMAXModel(trend_data, order=(1, 0, 1), trend="c")
        result = model.fit()
        assert result.param_names == ["intercept", "ar.L1", "ma.L1"]


class TestTrendForecast:
    def test_forecast_with_constant(self, simple_data):
        model = SARIMAXModel(simple_data, order=(1, 0, 0), trend="c")
        result = model.fit()
        fc = result.forecast(steps=5)
        assert len(fc.predicted_mean) == 5
        assert np.all(np.isfinite(fc.predicted_mean))

    def test_forecast_with_both(self, trend_data):
        model = SARIMAXModel(trend_data, order=(1, 0, 0), trend="ct")
        result = model.fit()
        fc = result.forecast(steps=10)
        assert len(fc.predicted_mean) == 10
        assert np.all(np.isfinite(fc.predicted_mean))
        # With a positive linear trend, forecasts should generally increase
        assert fc.predicted_mean[-1] > fc.predicted_mean[0]


class TestTrendResiduals:
    def test_resid_with_constant(self, simple_data):
        model = SARIMAXModel(simple_data, order=(1, 0, 0), trend="c")
        result = model.fit()
        r = result.resid
        assert len(r) == len(simple_data)
        assert np.all(np.isfinite(r))

    def test_resid_with_both(self, trend_data):
        model = SARIMAXModel(trend_data, order=(1, 0, 0), trend="ct")
        result = model.fit()
        r = result.resid
        assert len(r) == len(trend_data)
        assert np.all(np.isfinite(r))


class TestTrendSummary:
    def test_summary_shows_trend(self, simple_data):
        model = SARIMAXModel(simple_data, order=(1, 0, 0), trend="c")
        result = model.fit()
        s = result.summary()
        assert "Trend: c" in s
        assert "intercept" in s

    def test_summary_trend_none(self, simple_data):
        model = SARIMAXModel(simple_data, order=(1, 0, 0), trend="n")
        result = model.fit()
        s = result.summary()
        assert "Trend: n" in s


class TestTrendDefault:
    def test_default_is_none(self, simple_data):
        model = SARIMAXModel(simple_data, order=(1, 0, 0))
        assert model.trend == "n"
        result = model.fit()
        assert len(result.params) == 1
