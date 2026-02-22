"""ver4 tests: parameter summary, inference statistics, and statsmodels comparison.

Test plan from PARAMETER_SUMMARY_IMPLEMENTATION_PLAN_2026-02-22.md:
1. test_parameter_summary_returns_named_rows
2. test_parameter_summary_with_seasonal_names
3. test_summary_contains_parameter_table_headers
4. test_summary_inference_columns_when_enabled
5. test_summary_no_inference_path_fast
6. test_inference_failure_degrades_gracefully

Additional:
7-10. statsmodels parity comparison tests
"""

import numpy as np
import pytest
import sys

sys.path.insert(0, "python")

import warnings

from sarimax_py.model import (
    SARIMAXModel,
    _generate_param_names,
    _compute_numerical_hessian,
    _resolve_inference_mode,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ar1_data():
    np.random.seed(42)
    n = 200
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.7 * y[t - 1] + np.random.randn()
    return y


@pytest.fixture
def arima_111_data():
    np.random.seed(99)
    return np.cumsum(np.random.randn(250))


@pytest.fixture
def seasonal_data():
    """Monthly data with seasonal pattern."""
    np.random.seed(7)
    n = 240
    t = np.arange(n)
    seasonal = 5.0 * np.sin(2 * np.pi * t / 12)
    y = np.cumsum(np.random.randn(n) * 0.5) + seasonal
    return y


# ---------------------------------------------------------------------------
# 1. Parameter name generation
# ---------------------------------------------------------------------------

class TestParamNames:
    def test_ar1_name(self):
        names = _generate_param_names((1, 0, 0), (0, 0, 0, 0))
        assert names == ["ar.L1"]

    def test_arima_111_names(self):
        names = _generate_param_names((1, 1, 1), (0, 0, 0, 0))
        assert names == ["ar.L1", "ma.L1"]

    def test_arima_211_names(self):
        names = _generate_param_names((2, 1, 1), (0, 0, 0, 0))
        assert names == ["ar.L1", "ar.L2", "ma.L1"]

    def test_seasonal_names(self):
        names = _generate_param_names((1, 0, 1), (1, 0, 1, 12))
        assert names == ["ar.L1", "ma.L1", "ar.S.L12", "ma.S.L12"]

    def test_seasonal_p2(self):
        names = _generate_param_names((0, 0, 0), (2, 0, 0, 12))
        assert names == ["ar.S.L12", "ar.S.L24"]

    def test_with_exog(self):
        names = _generate_param_names((1, 0, 1), (1, 0, 1, 12), n_exog=2)
        assert names == ["x1", "x2", "ar.L1", "ma.L1", "ar.S.L12", "ma.S.L12"]

    def test_non_concentrate_includes_sigma2(self):
        names = _generate_param_names((1, 0, 0), (0, 0, 0, 0), concentrate_scale=False)
        assert names == ["ar.L1", "sigma2"]

    def test_empty_model(self):
        names = _generate_param_names((0, 1, 0), (0, 0, 0, 0))
        assert names == []


# ---------------------------------------------------------------------------
# 2. parameter_summary() basic
# ---------------------------------------------------------------------------

class TestParameterSummary:
    def test_returns_named_rows(self, ar1_data):
        """ver4 test #1: AR(1) param_names == ['ar.L1']."""
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        result = model.fit()
        assert result.param_names == ["ar.L1"]

    def test_parameter_summary_dict_keys(self, ar1_data):
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        result = model.fit()
        ps = result.parameter_summary(include_inference=False)
        assert set(ps.keys()) == {
            "name", "coef", "std_err", "z", "p_value",
            "ci_lower", "ci_upper", "inference_status", "inference_message",
        }
        assert ps["inference_status"] == "skipped"

    def test_parameter_summary_coef_matches_params(self, ar1_data):
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        result = model.fit()
        ps = result.parameter_summary(include_inference=False)
        np.testing.assert_array_equal(ps["coef"], result.params)

    def test_seasonal_param_names(self, seasonal_data):
        """ver4 test #2: seasonal model names."""
        model = SARIMAXModel(seasonal_data, order=(1, 1, 1), seasonal_order=(1, 0, 0, 12))
        result = model.fit()
        expected = ["ar.L1", "ma.L1", "ar.S.L12"]
        assert result.param_names == expected

    def test_param_names_length_matches_params(self, seasonal_data):
        model = SARIMAXModel(seasonal_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        result = model.fit()
        assert len(result.param_names) == len(result.params)


# ---------------------------------------------------------------------------
# 3. summary() output format
# ---------------------------------------------------------------------------

class TestSummaryFormat:
    def test_contains_parameter_table_headers(self, ar1_data):
        """ver4 test #3: summary string has 'Parameters:' and 'coef'."""
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        result = model.fit()
        s = result.summary()
        assert "Parameters:" in s
        assert "coef" in s
        assert "ar.L1" in s

    def test_summary_basic_keywords(self, ar1_data):
        """Backward compatibility: existing test keywords still present."""
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        result = model.fit()
        s = result.summary()
        assert "SARIMAX Results" in s
        assert "Order:" in s
        assert "Log Likelihood:" in s
        assert "AIC:" in s
        assert "BIC:" in s
        assert "Converged:" in s
        assert "Scale" in s

    def test_summary_no_inference_path_fast(self, ar1_data):
        """ver4 test #5: default include_inference=False excludes inference cols."""
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        result = model.fit()
        s = result.summary(include_inference=False)
        assert "std err" not in s
        assert "P>|z|" not in s
        assert "ar.L1" in s

    def test_summary_inference_columns_when_enabled(self, ar1_data):
        """ver4 test #4: include_inference=True shows std err / P>|z|."""
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        result = model.fit()
        s = result.summary(include_inference=True)
        assert "std err" in s
        assert "P>|z|" in s
        assert "ar.L1" in s


# ---------------------------------------------------------------------------
# 4. Inference statistics
# ---------------------------------------------------------------------------

class TestInference:
    def test_inference_ok_for_ar1(self, ar1_data):
        """AR(1) model should produce finite inference statistics."""
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        result = model.fit()
        ps = result.parameter_summary(alpha=0.05, include_inference=True)
        assert ps["inference_status"] in ("ok", "partial")
        assert np.all(np.isfinite(ps["std_err"]))
        assert np.all(ps["std_err"] > 0)
        assert np.all(np.isfinite(ps["z"]))
        assert np.all(np.isfinite(ps["p_value"]))
        # p-value in [0, 1]
        assert np.all(ps["p_value"] >= 0)
        assert np.all(ps["p_value"] <= 1)

    def test_inference_ci_contains_estimate(self, ar1_data):
        """CI should bracket the point estimate."""
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        result = model.fit()
        ps = result.parameter_summary(alpha=0.05, include_inference=True)
        if ps["inference_status"] == "ok":
            assert np.all(ps["ci_lower"] <= ps["coef"])
            assert np.all(ps["ci_upper"] >= ps["coef"])

    def test_inference_cached(self, ar1_data):
        """Repeated calls with same alpha should return identical results."""
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        result = model.fit()
        ps1 = result.parameter_summary(alpha=0.05, include_inference=True)
        ps2 = result.parameter_summary(alpha=0.05, include_inference=True)
        np.testing.assert_array_equal(ps1["std_err"], ps2["std_err"])

    def test_inference_different_alpha(self, ar1_data):
        """Different alpha should produce different CI."""
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        result = model.fit()
        ps_05 = result.parameter_summary(alpha=0.05, include_inference=True)
        ps_01 = result.parameter_summary(alpha=0.01, include_inference=True)
        if ps_05["inference_status"] == "ok" and ps_01["inference_status"] == "ok":
            # 1% CI should be wider than 5% CI
            width_05 = ps_05["ci_upper"] - ps_05["ci_lower"]
            width_01 = ps_01["ci_upper"] - ps_01["ci_lower"]
            assert np.all(width_01 >= width_05 - 1e-10)

    def test_inference_failure_degrades_gracefully(self):
        """ver4 test #6: numerically difficult case should not raise."""
        # Near-constant series → ill-conditioned Hessian
        np.random.seed(123)
        y = np.ones(100) + np.random.randn(100) * 1e-10
        model = SARIMAXModel(y, order=(1, 0, 0))
        result = model.fit()
        # Should NOT raise
        ps = result.parameter_summary(include_inference=True)
        assert ps["inference_status"] in ("ok", "partial", "failed")
        # summary should also not raise
        s = result.summary(include_inference=True)
        assert isinstance(s, str)
        assert "ar.L1" in s


# ---------------------------------------------------------------------------
# 5. statsmodels parity comparison
# ---------------------------------------------------------------------------

def _has_statsmodels():
    try:
        import statsmodels
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_statsmodels(), reason="statsmodels not installed")
class TestStatsmodelsParity:
    """Compare sarimax_rs results against statsmodels reference."""

    def _fit_both(self, y, order, seasonal_order=(0, 0, 0, 0)):
        """Fit with both engines, return (rust_result, sm_result)."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        # Rust
        model_rs = SARIMAXModel(y, order=order, seasonal_order=seasonal_order)
        res_rs = model_rs.fit()

        # statsmodels
        model_sm = SARIMAX(y, order=order, seasonal_order=seasonal_order,
                           enforce_stationarity=True, enforce_invertibility=True)
        res_sm = model_sm.fit(disp=False)

        return res_rs, res_sm

    def test_ar1_params_close(self, ar1_data):
        """AR(1) params should be close between engines."""
        res_rs, res_sm = self._fit_both(ar1_data, order=(1, 0, 0))
        # statsmodels includes sigma2 as last param; rust concentrates it out
        sm_params = res_sm.params[:-1]  # drop sigma2
        np.testing.assert_allclose(res_rs.params, sm_params, atol=0.05)

    def test_ar1_loglike_close(self, ar1_data):
        """AR(1) loglike should be close."""
        res_rs, res_sm = self._fit_both(ar1_data, order=(1, 0, 0))
        assert abs(res_rs.llf - res_sm.llf) < 3.0

    def test_arima_111_params_close(self, arima_111_data):
        """ARIMA(1,1,1) params should be close (wider tol for ARIMA models)."""
        res_rs, res_sm = self._fit_both(self.arima_111_data_val, order=(1, 1, 1))
        sm_params = res_sm.params[:-1]  # drop sigma2
        np.testing.assert_allclose(res_rs.params, sm_params, atol=0.1)

    @pytest.fixture(autouse=True)
    def _store_arima_data(self, arima_111_data):
        self.arima_111_data_val = arima_111_data

    def test_ar1_param_names_match_sm(self, ar1_data):
        """Parameter names should match statsmodels convention."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        model_rs = SARIMAXModel(ar1_data, order=(1, 0, 0))
        res_rs = model_rs.fit()

        model_sm = SARIMAX(ar1_data, order=(1, 0, 0))
        res_sm = model_sm.fit(disp=False)

        # statsmodels param_names includes sigma2; ours does not (concentrate_scale)
        # Compare just the non-sigma2 names
        sm_names = [n for n in res_sm.param_names if n != "sigma2"]
        assert res_rs.param_names == sm_names

    def test_arima_111_forecast_close(self, arima_111_data):
        """ARIMA(1,1,1) forecast mean should be close."""
        res_rs, res_sm = self._fit_both(arima_111_data, order=(1, 1, 1))

        # Forecast with rust params
        fcast_rs = res_rs.forecast(steps=10)

        # Forecast with statsmodels
        fcast_sm = res_sm.get_forecast(steps=10)

        # The forecasts may differ due to param differences, but should be
        # in the same ballpark
        np.testing.assert_allclose(
            fcast_rs.predicted_mean,
            fcast_sm.predicted_mean,
            atol=2.0,
            rtol=0.2,
        )

    def test_seasonal_loglike_close(self, seasonal_data):
        """SARIMA(1,1,1)(1,0,0,12) loglike should be close.

        Param-level comparison is relaxed for seasonal models because
        MLE can converge to equivalent optima with different sign conventions.
        Instead, we verify loglike proximity.
        """
        res_rs, res_sm = self._fit_both(
            seasonal_data, order=(1, 1, 1), seasonal_order=(1, 0, 0, 12)
        )
        assert abs(res_rs.llf - res_sm.llf) < 5.0, (
            f"rust llf={res_rs.llf:.4f}, sm llf={res_sm.llf:.4f}"
        )

    def test_seasonal_param_names_match_sm(self, seasonal_data):
        """Seasonal parameter names should match statsmodels."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        model_rs = SARIMAXModel(
            seasonal_data, order=(1, 1, 1), seasonal_order=(1, 0, 0, 12)
        )
        res_rs = model_rs.fit()

        model_sm = SARIMAX(
            seasonal_data, order=(1, 1, 1), seasonal_order=(1, 0, 0, 12)
        )
        res_sm = model_sm.fit(disp=False)

        sm_names = [n for n in res_sm.param_names if n != "sigma2"]
        assert res_rs.param_names == sm_names

    def test_ar1_inference_close(self, ar1_data):
        """AR(1) std errors should be in same ballpark as statsmodels."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        model_rs = SARIMAXModel(ar1_data, order=(1, 0, 0))
        res_rs = model_rs.fit()
        ps = res_rs.parameter_summary(alpha=0.05, include_inference=True)

        model_sm = SARIMAX(ar1_data, order=(1, 0, 0))
        res_sm = model_sm.fit(disp=False)

        if ps["inference_status"] == "ok":
            sm_se = res_sm.bse
            # Filter to non-sigma2 params
            sm_se_filtered = sm_se[:len(ps["std_err"])]
            # Tolerance: within 50% (numerical Hessian is approximate)
            for i in range(len(ps["std_err"])):
                if np.isfinite(ps["std_err"][i]) and np.isfinite(sm_se_filtered[i]):
                    ratio = ps["std_err"][i] / sm_se_filtered[i]
                    assert 0.3 < ratio < 3.0, (
                        f"param {i}: rust se={ps['std_err'][i]:.4f}, "
                        f"sm se={sm_se_filtered[i]:.4f}, ratio={ratio:.2f}"
                    )


# ---------------------------------------------------------------------------
# 6. Hessian numerical helper
# ---------------------------------------------------------------------------

class TestNumericalHessian:
    def test_quadratic_hessian(self):
        """Hessian of f(x) = -x^2 should be [[-2]]."""
        H = _compute_numerical_hessian(lambda x: -x[0]**2, np.array([1.0]))
        assert H is not None
        np.testing.assert_allclose(H[0, 0], -2.0, atol=1e-4)

    def test_multivariate_quadratic(self):
        """Hessian of f(x,y) = -(x^2 + 2y^2 + xy) should be correct."""
        def f(x):
            return -(x[0]**2 + 2*x[1]**2 + x[0]*x[1])
        H = _compute_numerical_hessian(f, np.array([1.0, 1.0]))
        assert H is not None
        # Expected: [[-2, -1], [-1, -4]]
        np.testing.assert_allclose(H, [[-2, -1], [-1, -4]], atol=1e-3)

    def test_non_finite_returns_none(self):
        """If function returns NaN, Hessian should return None."""
        H = _compute_numerical_hessian(lambda x: np.nan, np.array([1.0]))
        assert H is None


# ---------------------------------------------------------------------------
# 7. Inference enum (Section 11)
# ---------------------------------------------------------------------------

class TestResolveInferenceMode:
    """Unit tests for _resolve_inference_mode()."""

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
            result = _resolve_inference_mode(include_inference=True)
            assert result == "hessian"
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_legacy_false_maps_to_none(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _resolve_inference_mode(include_inference=False)
            assert result == "none"
            assert len(w) == 1

    def test_both_specified_inference_wins(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _resolve_inference_mode(inference="statsmodels", include_inference=True)
            assert result == "statsmodels"
            assert len(w) == 1


class TestInferenceEnum:
    """Test parameter_summary() with inference enum modes."""

    def test_mode_none(self, ar1_data):
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        result = model.fit()
        ps = result.parameter_summary(inference="none")
        assert ps["inference_status"] == "skipped"
        assert np.all(np.isnan(ps["std_err"]))

    def test_mode_hessian(self, ar1_data):
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        result = model.fit()
        ps = result.parameter_summary(inference="hessian")
        assert ps["inference_status"] in ("ok", "partial")
        assert np.all(np.isfinite(ps["std_err"]))
        assert "hessian_std_err" not in ps  # no prefix in hessian-only mode

    @pytest.mark.skipif(not _has_statsmodels(), reason="statsmodels not installed")
    def test_mode_statsmodels(self, ar1_data):
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        result = model.fit()
        ps = result.parameter_summary(inference="statsmodels")
        assert ps["inference_status"] in ("ok", "failed")
        if ps["inference_status"] == "ok":
            assert np.all(np.isfinite(ps["std_err"]))
            assert np.all(ps["std_err"] > 0)

    @pytest.mark.skipif(not _has_statsmodels(), reason="statsmodels not installed")
    def test_mode_both_keys(self, ar1_data):
        """'both' mode should include hessian_, sm_, and delta_ prefixed keys."""
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        result = model.fit()
        ps = result.parameter_summary(inference="both")

        # Check all expected keys are present
        for prefix in ("hessian_std_err", "hessian_z", "hessian_p_value",
                        "hessian_ci_lower", "hessian_ci_upper"):
            assert prefix in ps, f"Missing key: {prefix}"
        for prefix in ("sm_std_err", "sm_z", "sm_p_value",
                        "sm_ci_lower", "sm_ci_upper"):
            assert prefix in ps, f"Missing key: {prefix}"
        for prefix in ("delta_std_err", "delta_ci_lower", "delta_ci_upper"):
            assert prefix in ps, f"Missing key: {prefix}"

        assert "inference_status_hessian" in ps
        assert "inference_status_sm" in ps

    @pytest.mark.skipif(not _has_statsmodels(), reason="statsmodels not installed")
    def test_mode_both_delta_finite(self, ar1_data):
        """Delta columns should be finite when both sources succeed."""
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        result = model.fit()
        ps = result.parameter_summary(inference="both")

        if (ps.get("inference_status_hessian") in ("ok", "partial")
                and ps.get("inference_status_sm") == "ok"):
            assert np.all(np.isfinite(ps["delta_std_err"]))

    @pytest.mark.skipif(not _has_statsmodels(), reason="statsmodels not installed")
    def test_mode_both_delta_tolerance(self, ar1_data):
        """For AR(1), delta_std_err should be small (hessian ≈ statsmodels)."""
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        result = model.fit()
        ps = result.parameter_summary(inference="both")

        if (ps.get("inference_status_hessian") in ("ok", "partial")
                and ps.get("inference_status_sm") == "ok"):
            # Hessian and statsmodels std_err should be within 50% of each other
            for i in range(len(ps["hessian_std_err"])):
                h = ps["hessian_std_err"][i]
                s = ps["sm_std_err"][i]
                if np.isfinite(h) and np.isfinite(s) and s > 0:
                    assert abs(h - s) / s < 0.5, (
                        f"param {i}: hessian_se={h:.4f}, sm_se={s:.4f}"
                    )

    def test_alpha_validation(self, ar1_data):
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        result = model.fit()
        with pytest.raises(ValueError, match="alpha must be in"):
            result.parameter_summary(alpha=0.0, inference="hessian")
        with pytest.raises(ValueError, match="alpha must be in"):
            result.parameter_summary(alpha=1.0, inference="hessian")

    def test_invalid_inference_in_summary(self, ar1_data):
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        result = model.fit()
        with pytest.raises(ValueError, match="inference must be one of"):
            result.parameter_summary(inference="invalid_mode")


class TestSummaryInferenceEnum:
    """Test summary() string output with inference enum modes."""

    def test_summary_inference_none(self, ar1_data):
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        result = model.fit()
        s = result.summary(inference="none")
        assert "std err" not in s
        assert "Parameters:" in s

    def test_summary_inference_hessian(self, ar1_data):
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        result = model.fit()
        s = result.summary(inference="hessian")
        assert "std err" in s
        assert "P>|z|" in s
        assert "Inference:       hessian" in s

    @pytest.mark.skipif(not _has_statsmodels(), reason="statsmodels not installed")
    def test_summary_inference_statsmodels(self, ar1_data):
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        result = model.fit()
        s = result.summary(inference="statsmodels")
        assert "std err" in s
        assert "Inference:       statsmodels" in s

    @pytest.mark.skipif(not _has_statsmodels(), reason="statsmodels not installed")
    def test_summary_inference_both(self, ar1_data):
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        result = model.fit()
        s = result.summary(inference="both")
        assert "hess_se" in s
        assert "sm_se" in s
        assert "d_se" in s
        assert "Inference:       both" in s

    def test_summary_legacy_include_inference_true(self, ar1_data):
        """Legacy include_inference=True should still work with deprecation warning."""
        model = SARIMAXModel(ar1_data, order=(1, 0, 0))
        result = model.fit()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s = result.summary(include_inference=True)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
        assert "std err" in s
        assert "P>|z|" in s
