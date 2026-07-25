"""Tests for inference, diagnostics, parameter summary, and statsmodels parity.

Covers:
- sarimax_inference (Hessian, OPG) low-level Rust function
- sarimax_diagnostics (Ljung-Box, Jarque-Bera)
- SARIMAXResult.parameter_summary() with all inference backends
- Parameter name generation
- Inference enum (_resolve_inference_mode)
- Numerical Hessian helper
- statsmodels parity comparison
"""

import numpy as np
import pytest
import warnings
import rustima
from conftest import generate_ar1, generate_random_walk
from rustima.model import (
    SARIMAXModel,
    _generate_param_names,
    _resolve_inference_mode,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ar1_data():
    """Synthetic AR(1) data with known phi=0.7."""
    return generate_ar1(n=500)


@pytest.fixture(scope="module")
def arima111_data():
    """Synthetic ARIMA(1,1,1) data."""
    # AR(1) with phi=0.5 then integrated
    ar = generate_ar1(n=500, phi=0.5, seed=123)
    return np.cumsum(ar)


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
    """Direct tests of rustima.sarimax_inference()."""

    def test_hessian_basic(self, ar1_data, ar1_fit):
        result = rustima.sarimax_inference(
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
        result = rustima.sarimax_inference(
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
        result = rustima.sarimax_inference(
            ar1_data,
            (1, 0, 0),
            (0, 0, 0, 0),
            ar1_fit.params,
            method="hessian",
        )
        for p in result["p_value"]:
            assert 0.0 <= p <= 1.0, f"p-value {p} out of [0, 1]"

    def test_ci_contains_estimate(self, ar1_data, ar1_fit):
        result = rustima.sarimax_inference(
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
        result = rustima.sarimax_inference(
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
            rustima.sarimax_inference(
                ar1_data,
                (1, 0, 0),
                (0, 0, 0, 0),
                ar1_fit.params,
                method="invalid",
            )

    def test_arima111_hessian(self, arima111_data, arima111_fit):
        result = rustima.sarimax_inference(
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
    """Direct tests of rustima.sarimax_diagnostics()."""

    def test_basic(self, ar1_data, ar1_fit):
        result = rustima.sarimax_diagnostics(
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
        diag = rustima.sarimax_diagnostics(
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
        hess = rustima.sarimax_inference(
            ar1_data, (1, 0, 0), (0, 0, 0, 0),
            ar1_fit.params, method="hessian",
        )
        opg = rustima.sarimax_inference(
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
    """inference="hessian" and its deprecated alias "rust_hessian" — both
    route to the single Rust producer since the S10 dedup."""

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


# =========================================================================
# Parameter name generation (merged from test_parameter_summary.py)
# =========================================================================

class TestParamNames:
    def test_ar1_name(self):
        assert _generate_param_names((1, 0, 0), (0, 0, 0, 0)) == ["ar.L1", "sigma2"]

    def test_arima_111_names(self):
        assert _generate_param_names((1, 1, 1), (0, 0, 0, 0)) == ["ar.L1", "ma.L1", "sigma2"]

    def test_arima_211_names(self):
        assert _generate_param_names((2, 1, 1), (0, 0, 0, 0)) == ["ar.L1", "ar.L2", "ma.L1", "sigma2"]

    def test_seasonal_names(self):
        assert _generate_param_names((1, 0, 1), (1, 0, 1, 12)) == ["ar.L1", "ma.L1", "ar.S.L12", "ma.S.L12", "sigma2"]

    def test_seasonal_p2(self):
        assert _generate_param_names((0, 0, 0), (2, 0, 0, 12)) == ["ar.S.L12", "ar.S.L24", "sigma2"]

    def test_with_exog(self):
        assert _generate_param_names((1, 0, 1), (1, 0, 1, 12), n_exog=2) == [
            "x1", "x2", "ar.L1", "ma.L1", "ar.S.L12", "ma.S.L12", "sigma2",
        ]

    def test_non_concentrate_includes_sigma2(self):
        assert _generate_param_names((1, 0, 0), (0, 0, 0, 0), concentrate_scale=False) == ["ar.L1", "sigma2"]

    def test_empty_model(self):
        assert _generate_param_names((0, 1, 0), (0, 0, 0, 0)) == ["sigma2"]


# =========================================================================
# Parameter summary basic (merged from test_parameter_summary.py)
# =========================================================================

@pytest.fixture
def seasonal_data():
    np.random.seed(7)
    n = 240
    t = np.arange(n)
    seasonal = 5.0 * np.sin(2 * np.pi * t / 12)
    return np.cumsum(np.random.randn(n) * 0.5) + seasonal


class TestParameterSummary:
    def test_returns_named_rows(self, ar1_data):
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        assert model.fit().param_names == ["ar.L1", "sigma2"]

    def test_parameter_summary_dict_keys(self, ar1_data):
        ps = SARIMAXModel(ar1_data, order=(1, 0, 0)).fit().parameter_summary(include_inference=False)
        assert set(ps.keys()) == {
            "name", "coef", "std_err", "z", "p_value",
            "ci_lower", "ci_upper", "inference_status", "inference_message",
        }
        assert ps["inference_status"] == "skipped"

    def test_parameter_summary_coef_matches_params(self, ar1_data):
        result = SARIMAXModel(ar1_data, order=(1, 0, 0)).fit()
        ps = result.parameter_summary(include_inference=False)
        np.testing.assert_array_equal(ps["coef"], result.params)

    def test_seasonal_param_names(self, seasonal_data):
        result = SARIMAXModel(seasonal_data, order=(1, 1, 1), seasonal_order=(1, 0, 0, 12)).fit()
        assert result.param_names == ["ar.L1", "ma.L1", "ar.S.L12", "sigma2"]

    def test_param_names_length_matches_params(self, seasonal_data):
        result = SARIMAXModel(seasonal_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()
        assert len(result.param_names) == len(result.params)


# =========================================================================
# Summary format (merged from test_parameter_summary.py)
# =========================================================================

class TestSummaryFormat:
    def test_contains_parameter_table_headers(self, ar1_data):
        s = SARIMAXModel(ar1_data, order=(1, 0, 0)).fit().summary()
        assert "coef" in s and "ar.L1" in s

    def test_summary_no_inference_path_fast(self, ar1_data):
        s = SARIMAXModel(ar1_data, order=(1, 0, 0)).fit().summary(include_inference=False)
        assert "std err" not in s and "ar.L1" in s

    def test_summary_inference_columns_when_enabled(self, ar1_data):
        s = SARIMAXModel(ar1_data, order=(1, 0, 0)).fit().summary(include_inference=True)
        assert "std err" in s and "P>|z|" in s


# =========================================================================
# Inference statistics (merged from test_parameter_summary.py)
# =========================================================================

class TestInferenceStats:
    def test_inference_ok_for_ar1(self, ar1_data):
        ps = SARIMAXModel(ar1_data, order=(1, 0, 0)).fit().parameter_summary(alpha=0.05, include_inference=True)
        assert ps["inference_status"] in ("ok", "partial")
        assert np.all(np.isfinite(ps["std_err"])) and np.all(ps["std_err"] > 0)
        assert np.all(ps["p_value"] >= 0) and np.all(ps["p_value"] <= 1)

    def test_inference_ci_contains_estimate(self, ar1_data):
        result = SARIMAXModel(ar1_data, order=(1, 0, 0)).fit()
        ps = result.parameter_summary(alpha=0.05, include_inference=True)
        if ps["inference_status"] == "ok":
            assert np.all(ps["ci_lower"] <= ps["coef"])
            assert np.all(ps["ci_upper"] >= ps["coef"])

    def test_inference_different_alpha(self, ar1_data):
        result = SARIMAXModel(ar1_data, order=(1, 0, 0)).fit()
        ps_05 = result.parameter_summary(alpha=0.05, include_inference=True)
        ps_01 = result.parameter_summary(alpha=0.01, include_inference=True)
        if ps_05["inference_status"] == "ok" and ps_01["inference_status"] == "ok":
            width_05 = ps_05["ci_upper"] - ps_05["ci_lower"]
            width_01 = ps_01["ci_upper"] - ps_01["ci_lower"]
            assert np.all(width_01 >= width_05 - 1e-10)

    def test_inference_failure_degrades_gracefully(self):
        np.random.seed(123)
        y = np.ones(100) + np.random.randn(100) * 1e-10
        result = SARIMAXModel(y, order=(1, 0, 0)).fit()
        ps = result.parameter_summary(include_inference=True)
        assert ps["inference_status"] in ("ok", "partial", "failed")
        s = result.summary(include_inference=True)
        assert isinstance(s, str) and "ar.L1" in s


# =========================================================================
# Inference enum (merged from test_parameter_summary.py)
# =========================================================================

class TestResolveInferenceMode:
    def test_default_is_none(self):
        assert _resolve_inference_mode() == "none"

    def test_inference_enum_values(self):
        for mode in ("none", "hessian", "statsmodels", "both"):
            assert _resolve_inference_mode(inference=mode) == mode

    def test_invalid_inference_raises(self):
        with pytest.raises(ValueError, match="inference must be one of"):
            _resolve_inference_mode(inference="invalid")

    def test_legacy_true_maps_to_hessian(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert _resolve_inference_mode(include_inference=True) == "hessian"
            assert len(w) == 1 and issubclass(w[0].category, DeprecationWarning)

    def test_legacy_false_maps_to_none(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert _resolve_inference_mode(include_inference=False) == "none"
            assert len(w) == 1

    def test_both_specified_inference_wins(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert _resolve_inference_mode(inference="statsmodels", include_inference=True) == "statsmodels"


def _has_statsmodels():
    try:
        import statsmodels
        return True
    except ImportError:
        return False


class TestInferenceEnum:
    def test_mode_none(self, ar1_data):
        ps = SARIMAXModel(ar1_data, order=(1, 0, 0)).fit().parameter_summary(inference="none")
        assert ps["inference_status"] == "skipped"
        assert np.all(np.isnan(ps["std_err"]))

    def test_mode_hessian(self, ar1_data):
        ps = SARIMAXModel(ar1_data, order=(1, 0, 0)).fit().parameter_summary(inference="hessian")
        assert ps["inference_status"] in ("ok", "partial")
        assert np.all(np.isfinite(ps["std_err"]))

    @pytest.mark.skipif(not _has_statsmodels(), reason="statsmodels not installed")
    def test_mode_statsmodels(self, ar1_data):
        ps = SARIMAXModel(ar1_data, order=(1, 0, 0)).fit().parameter_summary(inference="statsmodels")
        assert ps["inference_status"] in ("ok", "failed")

    @pytest.mark.skipif(not _has_statsmodels(), reason="statsmodels not installed")
    def test_mode_both_keys(self, ar1_data):
        ps = SARIMAXModel(ar1_data, order=(1, 0, 0)).fit().parameter_summary(inference="both")
        for key in ("hessian_std_err", "hessian_z", "hessian_p_value",
                     "sm_std_err", "sm_z", "sm_p_value",
                     "delta_std_err", "delta_ci_lower", "delta_ci_upper"):
            assert key in ps, f"Missing key: {key}"

    def test_alpha_validation(self, ar1_data):
        result = SARIMAXModel(ar1_data, order=(1, 0, 0)).fit()
        with pytest.raises(ValueError, match="alpha must be in"):
            result.parameter_summary(alpha=0.0, inference="hessian")
        with pytest.raises(ValueError, match="alpha must be in"):
            result.parameter_summary(alpha=1.0, inference="hessian")

    def test_invalid_inference_in_summary(self, ar1_data):
        with pytest.raises(ValueError, match="inference must be one of"):
            SARIMAXModel(ar1_data, order=(1, 0, 0)).fit().parameter_summary(inference="invalid_mode")


class TestSummaryInferenceEnum:
    def test_summary_inference_none(self, ar1_data):
        s = SARIMAXModel(ar1_data, order=(1, 0, 0)).fit().summary(inference="none")
        assert "std err" not in s and "coef" in s

    def test_summary_inference_hessian(self, ar1_data):
        s = SARIMAXModel(ar1_data, order=(1, 0, 0)).fit().summary(inference="hessian")
        assert "std err" in s and "Inference: hessian" in s

    @pytest.mark.skipif(not _has_statsmodels(), reason="statsmodels not installed")
    def test_summary_inference_both(self, ar1_data):
        s = SARIMAXModel(ar1_data, order=(1, 0, 0)).fit().summary(inference="both")
        assert "hess_se" in s and "sm_se" in s

    def test_summary_legacy_include_inference_true(self, ar1_data):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s = SARIMAXModel(ar1_data, order=(1, 0, 0)).fit().summary(include_inference=True)
            assert any(issubclass(x.category, DeprecationWarning) for x in w)
        assert "std err" in s


# =========================================================================
# Risk fix tests (merged from test_parameter_summary.py)
# =========================================================================

class TestRiskFixInferenceValidation:
    def test_both_specified_invalid_inference_raises(self, ar1_data):
        result = SARIMAXModel(ar1_data, order=(1, 0, 0)).fit()
        with pytest.raises(ValueError, match="inference must be one of"):
            result.parameter_summary(inference="invalid", include_inference=True)

    def test_both_specified_valid_inference_works(self, ar1_data):
        result = SARIMAXModel(ar1_data, order=(1, 0, 0)).fit()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ps = result.parameter_summary(inference="hessian", include_inference=True)
            assert any(issubclass(x.category, DeprecationWarning) for x in w)
        assert ps["inference_status"] in ("ok", "partial")


class TestRiskFixCacheInvalidation:
    def test_inference_cache_invalidated_on_param_change(self, ar1_data):
        result = SARIMAXModel(ar1_data, order=(1, 0, 0)).fit()
        ps1 = result.parameter_summary(inference="hessian", alpha=0.05)
        original_se = ps1["std_err"].copy()
        result.params[0] = result.params[0] + 0.5
        ps2 = result.parameter_summary(inference="hessian", alpha=0.05)
        assert not np.array_equal(ps2["std_err"], original_se)

    def test_same_params_same_alpha_hits_cache(self, ar1_data):
        result = SARIMAXModel(ar1_data, order=(1, 0, 0)).fit()
        ps1 = result.parameter_summary(inference="hessian", alpha=0.05)
        ps2 = result.parameter_summary(inference="hessian", alpha=0.05)
        np.testing.assert_array_equal(ps1["std_err"], ps2["std_err"])


# =========================================================================
# statsmodels parity (merged from test_parameter_summary.py)
# =========================================================================

@pytest.mark.skipif(not _has_statsmodels(), reason="statsmodels not installed")
class TestStatsmodelsParity:
    def _fit_both(self, y, order, seasonal_order=(0, 0, 0, 0)):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        res_rs = SARIMAXModel(y, order=order, seasonal_order=seasonal_order).fit()
        res_sm = SARIMAX(y, order=order, seasonal_order=seasonal_order,
                         enforce_stationarity=True, enforce_invertibility=True).fit(disp=False)
        return res_rs, res_sm

    def test_ar1_params_close(self, ar1_data):
        res_rs, res_sm = self._fit_both(ar1_data, order=(1, 0, 0))
        np.testing.assert_allclose(res_rs.params, res_sm.params, atol=0.05)

    def test_ar1_loglike_close(self, ar1_data):
        res_rs, res_sm = self._fit_both(ar1_data, order=(1, 0, 0))
        assert abs(res_rs.llf - res_sm.llf) < 3.0

    def test_seasonal_loglike_close(self, seasonal_data):
        res_rs, res_sm = self._fit_both(seasonal_data, order=(1, 1, 1), seasonal_order=(1, 0, 0, 12))
        # Tolerance 5.5: statsmodels sometimes fails to converge for seasonal
        # models, producing ΔLL ≈ 5.089.  See MEMORY.md known-issues.
        assert abs(res_rs.llf - res_sm.llf) < 5.5

    def test_ar1_inference_close(self, ar1_data):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        res_rs = SARIMAXModel(ar1_data, order=(1, 0, 0)).fit()
        ps = res_rs.parameter_summary(alpha=0.05, include_inference=True)
        res_sm = SARIMAX(ar1_data, order=(1, 0, 0)).fit(disp=False)
        if ps["inference_status"] == "ok":
            for i in range(len(ps["std_err"])):
                if np.isfinite(ps["std_err"][i]) and np.isfinite(res_sm.bse[i]):
                    ratio = ps["std_err"][i] / res_sm.bse[i]
                    assert 0.3 < ratio < 3.0


@pytest.mark.skipif(not _has_statsmodels(), reason="statsmodels not installed")
class TestRiskFixDualPvalue:
    def test_summary_both_has_dual_pvalue_columns(self, ar1_data):
        s = SARIMAXModel(ar1_data, order=(1, 0, 0)).fit().summary(inference="both")
        assert "hess_p" in s and "sm_p" in s

    def test_parameter_summary_both_has_dual_pvalue_keys(self, ar1_data):
        ps = SARIMAXModel(ar1_data, order=(1, 0, 0)).fit().parameter_summary(inference="both")
        assert "hessian_p_value" in ps and "sm_p_value" in ps


@pytest.mark.skipif(not _has_statsmodels(), reason="statsmodels not installed")
class TestRiskFixEnforcementFlags:
    def test_statsmodels_mode_respects_enforcement_flags(self, ar1_data):
        model = SARIMAXModel(ar1_data, order=(1, 0, 0),
                             enforce_stationarity=False, enforce_invertibility=False)
        ps = model.fit().parameter_summary(inference="statsmodels")
        assert ps["inference_status"] in ("ok", "failed")


class TestTightStatsmodelsParity:
    """Tight like-for-like parity for the single Rust inference producer.

    Guards DIAGNOSIS_V9 N4: the unconstrained->constrained chain rule used
    the Jacobian instead of its inverse, inflating AR SEs by 1/J^2 (6.6x at
    phi=0.68) — invisible to the loose 0.3-3.0 band and to positive/finite
    checks. Compare each method against its true statsmodels counterpart.
    """

    def test_hessian_matches_sm_approx(self, ar1_data):
        res = SARIMAXModel(ar1_data, order=(1, 0, 0)).fit()
        ps = res.parameter_summary(inference="hessian")
        from statsmodels.tsa.statespace.sarimax import SARIMAX as sm_SARIMAX
        sm_res = sm_SARIMAX(ar1_data, order=(1, 0, 0)).fit(disp=False)
        sm_se = np.sqrt(np.diag(sm_res.cov_params_approx))
        np.testing.assert_allclose(ps["std_err"], sm_se, rtol=0.02)

    def test_opg_matches_sm_default_bse(self, ar1_data):
        res = SARIMAXModel(ar1_data, order=(1, 0, 0)).fit()
        ps = res.parameter_summary(inference="opg")
        from statsmodels.tsa.statespace.sarimax import SARIMAX as sm_SARIMAX
        sm_res = sm_SARIMAX(ar1_data, order=(1, 0, 0)).fit(disp=False)
        np.testing.assert_allclose(ps["std_err"], np.asarray(sm_res.bse), rtol=0.02)
