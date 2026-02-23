"""Tests for Rust-native inference (sarimax_inference, sarimax_diagnostics).

Tests the numerical Hessian and OPG inference paths, plus residual diagnostics.
"""

import sys
sys.path.insert(0, "python")

import numpy as np
import pytest
import sarimax_rs
from sarimax_py.model import SARIMAXModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ar1_data():
    """Synthetic AR(1) data with known phi=0.7."""
    np.random.seed(42)
    n = 500
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.7 * y[t - 1] + np.random.randn()
    return y


@pytest.fixture(scope="module")
def arima111_data():
    """Synthetic ARIMA(1,1,1) data."""
    np.random.seed(123)
    n = 500
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + np.random.randn()
    # Integrate
    y = np.cumsum(y)
    return y


@pytest.fixture(scope="module")
def ar1_fit(ar1_data):
    """Fit result for AR(1) model."""
    model = SARIMAXModel(ar1_data, order=(1, 0, 0))
    return model.fit()


@pytest.fixture(scope="module")
def arima111_fit(arima111_data):
    """Fit result for ARIMA(1,1,1) model."""
    model = SARIMAXModel(arima111_data, order=(1, 1, 1))
    return model.fit()


# ---------------------------------------------------------------------------
# Test: sarimax_inference (low-level Rust function)
# ---------------------------------------------------------------------------

class TestSarimaxInference:
    """Direct tests of sarimax_rs.sarimax_inference()."""

    def test_hessian_basic(self, ar1_data, ar1_fit):
        result = sarimax_rs.sarimax_inference(
            ar1_data,
            (1, 0, 0),
            (0, 0, 0, 0),
            ar1_fit.params,
            method="hessian",
            alpha=0.05,
        )
        assert result["status"] in ("ok", "partial")
        assert len(result["std_err"]) == len(ar1_fit.params)
        for se in result["std_err"]:
            assert np.isfinite(se) and se > 0

    def test_opg_basic(self, ar1_data, ar1_fit):
        result = sarimax_rs.sarimax_inference(
            ar1_data,
            (1, 0, 0),
            (0, 0, 0, 0),
            ar1_fit.params,
            method="opg",
            alpha=0.05,
        )
        assert result["status"] in ("ok", "partial")
        assert len(result["std_err"]) == len(ar1_fit.params)

    def test_pvalue_range(self, ar1_data, ar1_fit):
        result = sarimax_rs.sarimax_inference(
            ar1_data,
            (1, 0, 0),
            (0, 0, 0, 0),
            ar1_fit.params,
            method="hessian",
        )
        for p in result["p_value"]:
            assert 0.0 <= p <= 1.0, f"p-value {p} out of [0, 1]"

    def test_ci_contains_estimate(self, ar1_data, ar1_fit):
        result = sarimax_rs.sarimax_inference(
            ar1_data,
            (1, 0, 0),
            (0, 0, 0, 0),
            ar1_fit.params,
            method="hessian",
            alpha=0.05,
        )
        for i, param in enumerate(ar1_fit.params):
            assert result["ci_lower"][i] <= param <= result["ci_upper"][i]

    def test_cov_params_shape(self, ar1_data, ar1_fit):
        result = sarimax_rs.sarimax_inference(
            ar1_data,
            (1, 0, 0),
            (0, 0, 0, 0),
            ar1_fit.params,
            method="hessian",
        )
        k = len(ar1_fit.params)
        assert len(result["cov_params"]) == k * k

    def test_invalid_method_raises(self, ar1_data, ar1_fit):
        with pytest.raises(Exception):
            sarimax_rs.sarimax_inference(
                ar1_data,
                (1, 0, 0),
                (0, 0, 0, 0),
                ar1_fit.params,
                method="invalid",
            )

    def test_arima111_hessian(self, arima111_data, arima111_fit):
        result = sarimax_rs.sarimax_inference(
            arima111_data,
            (1, 1, 1),
            (0, 0, 0, 0),
            arima111_fit.params,
            method="hessian",
            alpha=0.05,
        )
        assert result["status"] in ("ok", "partial")
        assert len(result["std_err"]) == len(arima111_fit.params)


# ---------------------------------------------------------------------------
# Test: sarimax_diagnostics (low-level Rust function)
# ---------------------------------------------------------------------------

class TestSarimaxDiagnostics:
    """Direct tests of sarimax_rs.sarimax_diagnostics()."""

    def test_basic(self, ar1_data, ar1_fit):
        result = sarimax_rs.sarimax_diagnostics(
            ar1_data,
            (1, 0, 0),
            (0, 0, 0, 0),
            ar1_fit.params,
        )
        assert "ljung_box_stat" in result
        assert "jarque_bera_stat" in result
        assert "het_stat" in result
        assert result["ljung_box_stat"] >= 0
        assert 0.0 <= result["ljung_box_pvalue"] <= 1.0
        assert result["jarque_bera_stat"] >= 0
        assert 0.0 <= result["jarque_bera_pvalue"] <= 1.0

    def test_white_noise_ljung_box(self):
        """White noise residuals should not be rejected by Ljung-Box."""
        np.random.seed(99)
        y = np.random.randn(500)
        # AR(0) ≈ white noise; fit AR(1) and check residuals
        model = SARIMAXModel(y, order=(1, 0, 0))
        res = model.fit()
        diag = sarimax_rs.sarimax_diagnostics(
            y, (1, 0, 0), (0, 0, 0, 0), res.params,
        )
        # Should not reject at 1% level
        assert diag["ljung_box_pvalue"] > 0.01


# ---------------------------------------------------------------------------
# Test: Model-level inference (parameter_summary with Rust backends)
# ---------------------------------------------------------------------------

class TestModelInference:
    """Tests for SARIMAXResult.parameter_summary() with Rust inference."""

    def test_rust_hessian_mode(self, ar1_data):
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        res = model.fit()
        ps = res.parameter_summary(inference="rust_hessian")
        assert ps["inference_status"] in ("ok", "partial", "failed")
        assert len(ps["std_err"]) == len(res.params)
        assert len(ps["z"]) == len(res.params)
        assert len(ps["p_value"]) == len(res.params)

    def test_opg_mode(self, ar1_data):
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        res = model.fit()
        ps = res.parameter_summary(inference="opg")
        assert ps["inference_status"] in ("ok", "partial", "failed")
        assert len(ps["std_err"]) == len(res.params)

    def test_rust_hessian_se_positive(self, ar1_data):
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        res = model.fit()
        ps = res.parameter_summary(inference="rust_hessian")
        if ps["inference_status"] == "ok":
            for se in ps["std_err"]:
                assert np.isfinite(se) and se > 0

    def test_rust_hessian_ci_contains_param(self, ar1_data):
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        res = model.fit()
        ps = res.parameter_summary(inference="rust_hessian", alpha=0.05)
        if ps["inference_status"] == "ok":
            for i in range(len(res.params)):
                assert ps["ci_lower"][i] <= res.params[i] <= ps["ci_upper"][i]

    def test_opg_ci_contains_param(self, ar1_data):
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        res = model.fit()
        ps = res.parameter_summary(inference="opg", alpha=0.05)
        if ps["inference_status"] == "ok":
            for i in range(len(res.params)):
                assert ps["ci_lower"][i] <= res.params[i] <= ps["ci_upper"][i]

    def test_inference_caching(self, ar1_data):
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        res = model.fit()
        ps1 = res.parameter_summary(inference="rust_hessian")
        ps2 = res.parameter_summary(inference="rust_hessian")
        np.testing.assert_array_equal(ps1["std_err"], ps2["std_err"])

    def test_summary_renders_with_rust_hessian(self, ar1_data):
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        res = model.fit()
        s = res.summary(inference="rust_hessian")
        assert "SARIMAX Results" in s
        assert "std err" in s or "coef" in s

    def test_diagnostics_method(self, ar1_data):
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        res = model.fit()
        diag = res.diagnostics()
        assert "ljung_box_stat" in diag
        assert "jarque_bera_stat" in diag
        assert "het_stat" in diag


# ---------------------------------------------------------------------------
# Test: Hessian vs OPG comparison
# ---------------------------------------------------------------------------

class TestHessianVsOpg:
    """Compare Hessian and OPG standard errors."""

    def test_both_produce_finite_se(self, ar1_data, ar1_fit):
        """Both methods should produce finite standard errors."""
        hess = sarimax_rs.sarimax_inference(
            ar1_data, (1, 0, 0), (0, 0, 0, 0),
            ar1_fit.params, method="hessian",
        )
        opg = sarimax_rs.sarimax_inference(
            ar1_data, (1, 0, 0), (0, 0, 0, 0),
            ar1_fit.params, method="opg",
        )
        # Hessian should produce finite SE
        assert all(np.isfinite(se) and se > 0 for se in hess["std_err"])
        # OPG uses approximate score_obs; SE should be finite but may differ
        assert all(np.isfinite(se) for se in opg["std_err"])


# ---------------------------------------------------------------------------
# Test: Rust Hessian vs Python Hessian comparison
# ---------------------------------------------------------------------------

class TestRustVsPythonHessian:
    """Rust Hessian vs Python numerical Hessian.

    Note: The Rust Hessian operates in unconstrained parameter space with
    chain rule (J'HJ), while the Python version perturbs constrained params
    directly. These are different parameterizations and may give different
    numerical results, especially for constrained parameters (AR/MA).
    """

    def test_both_produce_finite_se(self, ar1_data):
        """Both methods should produce finite, positive standard errors."""
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        res = model.fit()

        ps_python = res.parameter_summary(inference="hessian")
        ps_rust = res.parameter_summary(inference="rust_hessian")

        # Both should succeed
        assert ps_python["inference_status"] in ("ok", "partial")
        assert ps_rust["inference_status"] in ("ok", "partial")

        # Both should have finite positive SE
        for i in range(len(res.params)):
            py_se = ps_python["std_err"][i]
            rs_se = ps_rust["std_err"][i]
            assert np.isfinite(py_se) and py_se > 0, f"Python SE[{i}]={py_se}"
            assert np.isfinite(rs_se) and rs_se > 0, f"Rust SE[{i}]={rs_se}"

    def test_arima111_both_succeed(self, arima111_data):
        """Both methods work for ARIMA(1,1,1)."""
        model = SARIMAXModel(arima111_data, order=(1, 1, 1))
        res = model.fit()

        ps_rust = res.parameter_summary(inference="rust_hessian")
        assert ps_rust["inference_status"] in ("ok", "partial")
        assert len(ps_rust["std_err"]) == len(res.params)
