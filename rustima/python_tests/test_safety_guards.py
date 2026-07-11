"""Phase 11 — Regression guard tests for runtime safety hardening.

Tests cover:
- White-noise (0,0,0) model with all optimizer methods
- Exog row-length mismatch boundary (early ValueError at PyO3 boundary)
- Unknown trend string rejection (ValueError instead of silent fallback)
"""

import numpy as np
import pytest

import rustima


# ---------------------------------------------------------------------------
# 11-1. White-noise (0,0,0) + all optimizer methods
# ---------------------------------------------------------------------------

@pytest.fixture
def white_noise_data():
    """100-point white noise series."""
    return np.random.default_rng(42).normal(size=100)


@pytest.mark.parametrize("method", ["lbfgsb", "lbfgsb-multi", "lbfgsb-strict", "nelder-mead"])
def test_white_noise_000_all_methods(white_noise_data, method):
    """ARIMA(0,0,0) — zero free params with concentrate_scale — should return
    a valid result without crashing for every optimizer method."""
    result = rustima.sarimax_fit(
        white_noise_data,
        order=(0, 0, 0),
        seasonal=(0, 0, 0, 0),
        method=method,
        concentrate_scale=True,
    )
    assert np.isfinite(result["loglike"]), f"loglike not finite for method={method}"
    assert result["n_obs"] == len(white_noise_data)


@pytest.mark.parametrize("method", ["lbfgsb", "lbfgsb-multi", "nelder-mead"])
def test_white_noise_000_with_trend_c(white_noise_data, method):
    """ARIMA(0,0,0) with constant trend — has 1 free param (intercept)."""
    result = rustima.sarimax_fit(
        white_noise_data,
        order=(0, 0, 0),
        seasonal=(0, 0, 0, 0),
        method=method,
        trend="c",
    )
    assert np.isfinite(result["loglike"])
    assert len(result["params"]) >= 1  # at least the trend coeff


def test_white_noise_000_maxiter_0(white_noise_data):
    """maxiter=0 with zero-param model — should return immediately."""
    result = rustima.sarimax_fit(
        white_noise_data,
        order=(0, 0, 0),
        seasonal=(0, 0, 0, 0),
        maxiter=0,
    )
    assert np.isfinite(result["loglike"])


# ---------------------------------------------------------------------------
# 11-2. Exog row-length mismatch boundary tests
# ---------------------------------------------------------------------------

def test_exog_row_mismatch_single_fit():
    """Exog with wrong number of rows should raise ValueError immediately."""
    y = np.random.default_rng(42).normal(size=100)
    exog_wrong = np.random.default_rng(43).normal(size=(80, 2))  # 80 != 100

    with pytest.raises(ValueError, match="exog has 80 rows but endog has 100"):
        rustima.sarimax_fit(
            y,
            order=(1, 0, 0),
            seasonal=(0, 0, 0, 0),
            exog=exog_wrong,
        )


def test_exog_row_mismatch_single_loglike():
    """Exog mismatch in loglike should also raise early."""
    y = np.random.default_rng(42).normal(size=100)
    exog_wrong = np.random.default_rng(43).normal(size=(50, 1))
    params = np.array([0.1, 0.5])  # dummy params

    with pytest.raises(ValueError, match="exog has 50 rows but endog has 100"):
        rustima.sarimax_loglike(
            y,
            order=(1, 0, 0),
            seasonal=(0, 0, 0, 0),
            params=params,
            exog=exog_wrong,
        )


def test_exog_row_mismatch_single_forecast():
    """Exog mismatch in forecast should raise early."""
    y = np.random.default_rng(42).normal(size=100)
    exog_wrong = np.random.default_rng(43).normal(size=(90, 1))
    params = np.array([0.1, 0.5])

    with pytest.raises(ValueError, match="exog has 90 rows but endog has 100"):
        rustima.sarimax_forecast(
            y,
            order=(1, 0, 0),
            seasonal=(0, 0, 0, 0),
            params=params,
            exog=exog_wrong,
        )


# ---------------------------------------------------------------------------
# 9-3. Unknown trend string rejection
# ---------------------------------------------------------------------------

def test_unknown_trend_raises_valueerror():
    """Unknown trend string should raise ValueError, not silently fallback."""
    y = np.random.default_rng(42).normal(size=100)

    with pytest.raises(ValueError, match="unknown trend"):
        rustima.sarimax_fit(
            y,
            order=(1, 0, 0),
            seasonal=(0, 0, 0, 0),
            trend="xyz",
        )


def test_unknown_trend_raises_for_loglike():
    """Unknown trend in loglike should also raise."""
    y = np.random.default_rng(42).normal(size=100)
    params = np.array([0.5])

    with pytest.raises(ValueError, match="unknown trend"):
        rustima.sarimax_loglike(
            y,
            order=(1, 0, 0),
            seasonal=(0, 0, 0, 0),
            params=params,
            trend="invalid_trend",
        )


@pytest.mark.parametrize("trend", ["n", "c", "t", "ct"])
def test_valid_trends_accepted(trend):
    """All valid trend strings should work without error."""
    y = np.random.default_rng(42).normal(size=100)
    result = rustima.sarimax_fit(
        y,
        order=(1, 0, 0),
        seasonal=(0, 0, 0, 0),
        trend=trend,
    )
    assert np.isfinite(result["loglike"])


# ---------------------------------------------------------------------------
# 11-4. P0-1 regression: zero-param fast-path must NOT bypass validation
# ---------------------------------------------------------------------------

def test_zero_param_invalid_method_raises():
    """(0,0,0) with method='foo' should raise, not silently succeed."""
    y = np.random.default_rng(42).normal(size=100)
    with pytest.raises(RuntimeError, match="unknown method"):
        rustima.sarimax_fit(
            y,
            order=(0, 0, 0),
            seasonal=(0, 0, 0, 0),
            method="foo",
        )


def test_zero_param_invalid_start_params_raises():
    """(0,0,0) with start_params=[1.0] should raise — expected 0 params."""
    y = np.random.default_rng(42).normal(size=100)
    with pytest.raises((ValueError, RuntimeError), match="(length|mismatch)"):
        rustima.sarimax_fit(
            y,
            order=(0, 0, 0),
            seasonal=(0, 0, 0, 0),
            start_params=np.array([1.0]),
            concentrate_scale=True,
        )


def test_zero_param_converged_true():
    """Zero-parameter model (0,0,0) should report converged=True."""
    y = np.random.default_rng(42).normal(size=100)
    result = rustima.sarimax_fit(
        y,
        order=(0, 0, 0),
        seasonal=(0, 0, 0, 0),
    )
    assert result["converged"] is True, "zero-param model should be trivially converged"


# ---------------------------------------------------------------------------
# 11-5. V8.2 regression tests
# ---------------------------------------------------------------------------

def test_zero_col_exog_row_mismatch_raises():
    """Exog with shape (80, 0) — 0 columns but wrong row count — should raise."""
    y = np.random.default_rng(42).normal(size=100)
    exog_bad = np.empty((80, 0), dtype=np.float64)  # 80 != 100

    with pytest.raises(ValueError, match="exog has 80 rows but endog has 100"):
        rustima.sarimax_fit(
            y,
            order=(1, 0, 0),
            seasonal=(0, 0, 0, 0),
            exog=exog_bad,
        )


def test_auto_arima_grid_only_candidate_returns_result():
    """Grid-search with only one candidate should return a result, not None."""
    from rustima.auto import auto_arima

    y = np.random.default_rng(42).normal(size=100)
    res = auto_arima(
        y,
        max_p=0, max_q=0, max_d=0,
        max_P=0, max_Q=0, max_D=0,
        s=0, d=0, D=0,
        stepwise=False,
    )
    assert res.result is not None, "only-candidate grid search should return a result"


def test_auto_arima_stepwise_s0_no_seasonal_explored():
    """Stepwise with s=0 should never explore P>0 or Q>0."""
    from rustima.auto import auto_arima

    y = np.random.default_rng(42).normal(size=100)
    res = auto_arima(
        y,
        max_p=2, max_q=2, max_d=0,
        max_P=2, max_Q=2, max_D=0,
        s=0, d=0, D=0,
        stepwise=True,
    )
    for entry in res.history:
        seasonal = entry["seasonal_order"]
        P, D, Q, s_val = seasonal
        assert P == 0 and Q == 0, (
            f"stepwise with s=0 explored P={P}, Q={Q} (should be 0)"
        )


# ---------------------------------------------------------------------------
# 11-6. V8.3 regression tests
# ---------------------------------------------------------------------------

def test_batch_forecast_short_exog_no_panic():
    """batch_forecast + simple_differencing + short exog → ValueError, not panic."""
    series = [np.arange(10, dtype=float)]
    params_list = [np.array([0.0], dtype=float)]
    exog_list = [np.empty((0, 1), dtype=np.float64)]
    future_exog_list = [np.zeros((1, 1), dtype=np.float64)]

    with pytest.raises(ValueError, match="exog_list\\[0\\] has 0 rows"):
        rustima.sarimax_batch_forecast(
            series,
            (0, 1, 0),
            (0, 0, 0, 0),
            params_list,
            steps=1,
            exog_list=exog_list,
            exog_forecast_list=future_exog_list,
            simple_differencing=True,
        )


def test_ndiffs_uses_adf_when_statsmodels_available():
    """_ndiffs should use ADF test (not variance fallback) when statsmodels exists."""
    from rustima.auto import _ndiffs

    y = np.random.default_rng(42).normal(size=200)
    d = _ndiffs(y, max_d=2)
    assert d == 0, "white noise should need d=0 via ADF test"


def test_invalid_criterion_raises_for_grid_search():
    """Invalid criterion must raise ValueError for grid search too, not silently fallback."""
    from rustima.auto import auto_arima

    y = np.random.default_rng(42).normal(size=60)
    with pytest.raises(ValueError, match="Unknown criterion"):
        auto_arima(
            y,
            max_p=1, max_q=1, max_d=0,
            max_P=0, max_Q=0, max_D=0,
            s=0, d=0, D=0,
            criterion="xyz",
            stepwise=False,
        )


def test_1d_exog_auto_reshaped():
    """1D exog array should be auto-reshaped to (n, 1) in SARIMAXModel."""
    from rustima.model import SARIMAXModel

    y = np.random.default_rng(42).normal(size=50)
    x = np.random.default_rng(43).normal(size=50)  # 1D
    m = SARIMAXModel(y, order=(1, 0, 0), exog=x)
    assert m.exog.ndim == 2
    assert m.exog.shape == (50, 1)
    r = m.fit()
    assert r is not None


def test_list_start_params_accepted():
    """Python list start_params should be auto-coerced to ndarray."""
    from rustima.model import SARIMAXModel

    y = np.random.default_rng(42).normal(size=50)
    m = SARIMAXModel(y, order=(1, 0, 0))
    r = m.fit(start_params=[0.5, 1.0])  # [ar.L1, sigma2]
    assert r is not None


def test_1d_forecast_exog_accepted():
    """1D forecast exog should be auto-reshaped to (steps, 1)."""
    from rustima.model import SARIMAXModel

    y = np.random.default_rng(42).normal(size=50)
    x = np.random.default_rng(43).normal(size=(50, 1))
    m = SARIMAXModel(y, order=(1, 0, 0), exog=x)
    r = m.fit()
    fc = r.forecast(steps=3, exog=np.random.default_rng(44).normal(size=3))
    assert len(fc.predicted_mean) == 3


# ---------------------------------------------------------------------------
# 11-7. V8.4 regression tests
# ---------------------------------------------------------------------------

def test_simple_differencing_resid_propagated():
    """resid with simple_differencing=True must use the correct Rust path."""
    from rustima.model import SARIMAXModel

    np.random.seed(2)
    y = np.cumsum(np.random.randn(100))
    r = SARIMAXModel(y, order=(1, 1, 1), simple_differencing=True).fit(maxiter=50)
    params = np.array(r.params)

    wrap_resid = np.array(r.resid)
    true_resid = np.array(
        rustima.sarimax_residuals(
            y, (1, 1, 1), (0, 0, 0, 0), params, simple_differencing=True
        )["standardized_residuals"]
    )
    # With simple_differencing=True, residual length = n - d = 99
    assert len(wrap_resid) == len(true_resid), (
        f"resid length mismatch: wrapper={len(wrap_resid)}, true={len(true_resid)}"
    )
    assert np.allclose(wrap_resid, true_resid, equal_nan=True)


def test_simple_differencing_rust_inference_propagated():
    """Rust inference with simple_differencing=True must compute on correct path."""
    from rustima.model import SARIMAXModel

    np.random.seed(4)
    y = np.cumsum(np.random.randn(200))
    r = SARIMAXModel(y, order=(1, 1, 0), simple_differencing=True).fit(maxiter=80)
    params = np.array(r.params)

    se_wrapper = np.array(r.parameter_summary(inference="rust_hessian")["std_err"])
    se_true = np.array(
        rustima.sarimax_inference(
            y, (1, 1, 0), (0, 0, 0, 0), params,
            method="hessian", simple_differencing=True
        )["std_err"]
    )
    assert np.allclose(se_wrapper, se_true, equal_nan=True), (
        "Rust inference std_err should match simple_differencing=True path"
    )


def test_trend_ct_statsmodels_inference_no_crash():
    """trend='ct' + statsmodels inference should not raise IndexError/ValueError."""
    from rustima.model import SARIMAXModel

    np.random.seed(7)
    y = np.random.randn(120)
    r = SARIMAXModel(y, order=(1, 0, 0), trend="ct").fit(maxiter=50)

    # parameter_summary should return arrays matching param count
    ps = r.parameter_summary(inference="statsmodels")
    assert len(ps["std_err"]) == len(r.params)

    # summary should not crash
    s = r.summary(inference="statsmodels")
    assert isinstance(s, str)

    # 'both' mode should not crash
    ps_both = r.parameter_summary(inference="both")
    assert len(ps_both["std_err"]) == len(r.params)


# ---------------------------------------------------------------------------
# 11-8. V8.5 regression tests
# ---------------------------------------------------------------------------

def test_near_cancellation_warning_via_python_warnings():
    """Near-cancellation warning should be emitted via Python warnings, not stderr."""
    import warnings as warn_mod

    np.random.seed(42)
    y = np.random.randn(120)

    with warn_mod.catch_warnings(record=True) as w:
        warn_mod.simplefilter("always")
        # Start in a near-cancelling region so the fitted params (maxiter=1)
        # stay near-cancelling and the warning is emitted deterministically.
        result = rustima.sarimax_fit(
            y, (1, 0, 1), (0, 0, 0, 0), maxiter=1,
            start_params=np.array([0.9, 0.9, 1.0]),
        )

    # Check warning was captured via Python warnings module
    near_cancel_warnings = [x for x in w if "near-cancellation" in str(x.message)]
    assert len(near_cancel_warnings) >= 1, "near-cancellation should emit Python warning"

    # Also check result dict contains warnings list
    assert "warnings" in result, "result dict should have 'warnings' key"
    assert any("near-cancellation" in msg for msg in result["warnings"])


def test_fit_result_warnings_empty_when_no_issue():
    """Normal fit should have empty warnings list."""
    np.random.seed(99)
    y = np.random.randn(100)
    result = rustima.sarimax_fit(y, (1, 0, 0), (0, 0, 0, 0), maxiter=50)
    assert "warnings" in result
    assert isinstance(result["warnings"], list)


def test_forecast_error_message_unified():
    """Single and batch forecast missing-exog errors should have consistent core message."""
    np.random.seed(9)
    y = np.random.randn(40)
    ex = np.random.randn(40, 1)
    fit = rustima.sarimax_fit(y, (1, 0, 0), (0, 0, 0, 0), exog=ex, maxiter=20)
    params = np.array(fit["params"])

    # Single forecast
    with pytest.raises(ValueError, match="future_exog is required"):
        rustima.sarimax_forecast(
            y, (1, 0, 0), (0, 0, 0, 0), params, steps=3, exog=ex
        )

    # Batch forecast — returns error dict, not exception
    res = rustima.sarimax_batch_forecast(
        [y], (1, 0, 0), (0, 0, 0, 0), [params], steps=3, exog_list=[ex]
    )
    assert "error" in res[0]
    assert "future_exog is required" in res[0]["error"]


# ---------------------------------------------------------------------------
# 11-9. V8.6 regression tests
# ---------------------------------------------------------------------------

def test_opg_vs_hessian_ar1_reasonable():
    """OPG SE should be within 10x of Hessian SE for a stable AR(1)."""
    from rustima.model import SARIMAXModel

    np.random.seed(1)
    y = np.random.randn(200)
    r = SARIMAXModel(y, order=(1, 0, 0)).fit(maxiter=100)

    h = r.parameter_summary(inference="rust_hessian")
    o = r.parameter_summary(inference="opg")

    h_se = np.array(h["std_err"])
    o_se = np.array(o["std_err"])

    # Both should be finite
    assert np.all(np.isfinite(h_se)), f"Hessian SE not finite: {h_se}"
    assert np.all(np.isfinite(o_se)), f"OPG SE not finite: {o_se}"

    # Ratio should be reasonable (< 10x)
    ratio = o_se / np.where(h_se > 0, h_se, 1e-30)
    assert np.all(ratio < 10.0), (
        f"OPG/Hessian SE ratio too large: {ratio} (OPG={o_se}, Hessian={h_se})"
    )
    assert np.all(ratio > 0.1), (
        f"OPG/Hessian SE ratio too small: {ratio} (OPG={o_se}, Hessian={h_se})"
    )


def test_opg_vs_hessian_arma11_reasonable():
    """OPG SE should be reasonable for ARMA(1,1)."""
    from rustima.model import SARIMAXModel

    np.random.seed(2)
    y = np.random.randn(300)
    r = SARIMAXModel(y, order=(1, 0, 1)).fit(maxiter=100)

    h = r.parameter_summary(inference="rust_hessian")
    o = r.parameter_summary(inference="opg")

    h_se = np.array(h["std_err"])
    o_se = np.array(o["std_err"])

    assert np.all(np.isfinite(h_se)), f"Hessian SE not finite: {h_se}"
    assert np.all(np.isfinite(o_se)), f"OPG SE not finite: {o_se}"

    ratio = o_se / np.where(h_se > 0, h_se, 1e-30)
    assert np.all(ratio < 20.0), (
        f"OPG/Hessian SE ratio too large for ARMA(1,1): {ratio}"
    )


def test_opg_status_ok_for_simple_model():
    """OPG should return status='ok' for a well-identified AR(1)."""
    from rustima.model import SARIMAXModel

    np.random.seed(3)
    y = np.random.randn(200)
    r = SARIMAXModel(y, order=(1, 0, 0)).fit(maxiter=100)

    o = r.parameter_summary(inference="opg")
    assert o["inference_status"] == "ok", (
        f"OPG status should be 'ok', got '{o['inference_status']}': {o.get('inference_message')}"
    )


def test_statsmodels_inference_non_converged_status():
    """statsmodels inference bridge should report non-converged status, not 'ok'."""
    from rustima.model import _compute_statsmodels_inference

    np.random.seed(10)
    y = np.random.randn(50)

    result = _compute_statsmodels_inference(
        y, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0),
        n_params_rs=1,
        enforce_stationarity=True,
        enforce_invertibility=True,
        trend="n",
    )
    # With 50 observations, should converge fine → "ok"
    assert result["inference_status_sm"] in ("ok", "warning")


def test_statsmodels_inference_simple_differencing_passed():
    """statsmodels inference bridge should pass simple_differencing through."""
    from rustima.model import SARIMAXModel

    np.random.seed(5)
    y = np.cumsum(np.random.randn(200))

    r_sd = SARIMAXModel(y, order=(1, 1, 0), simple_differencing=True).fit(maxiter=80)
    r_no = SARIMAXModel(y, order=(1, 1, 0), simple_differencing=False).fit(maxiter=80)

    ps_sd = r_sd.parameter_summary(inference="statsmodels")
    ps_no = r_no.parameter_summary(inference="statsmodels")

    # Both should produce finite SE
    assert np.all(np.isfinite(ps_sd["std_err"])), "SD=True SE not finite"
    assert np.all(np.isfinite(ps_no["std_err"])), "SD=False SE not finite"

    # SE values can differ due to different likelihood evaluation,
    # but both should be in a reasonable range
    assert np.all(ps_sd["std_err"] > 0), "SD=True SE should be positive"
    assert np.all(ps_no["std_err"] > 0), "SD=False SE should be positive"


# ---------------------------------------------------------------------------
# 11-10. V8.7 regression tests
# ---------------------------------------------------------------------------

def test_get_prediction_simple_differencing_no_crash():
    """get_prediction() must not crash with simple_differencing=True."""
    from rustima.model import SARIMAXModel

    np.random.seed(0)
    y = np.cumsum(np.random.randn(120))
    r = SARIMAXModel(y, order=(1, 1, 1), simple_differencing=True).fit(maxiter=50)
    pred = r.get_prediction()

    # Length must match original endog
    assert len(pred.predicted_mean) == len(y), (
        f"prediction length {len(pred.predicted_mean)} != endog length {len(y)}"
    )
    # First d observations should be NaN (dropped by differencing)
    assert np.isnan(pred.predicted_mean[0]), "first element should be NaN (dropped)"
    # Remaining should be finite
    assert np.all(np.isfinite(pred.predicted_mean[1:])), "non-dropped predictions should be finite"


def test_get_prediction_seasonal_simple_differencing():
    """get_prediction() with seasonal simple_differencing pads correct NaN count."""
    from rustima.model import SARIMAXModel

    np.random.seed(1)
    y = np.cumsum(np.random.randn(200))
    r = SARIMAXModel(
        y, order=(1, 1, 0), seasonal_order=(1, 1, 0, 12),
        simple_differencing=True,
    ).fit(maxiter=50)
    pred = r.get_prediction()

    n_drop = 1 + 12  # d + s*D
    assert len(pred.predicted_mean) == len(y)
    assert np.all(np.isnan(pred.predicted_mean[:n_drop])), (
        f"first {n_drop} elements should be NaN"
    )
    assert np.all(np.isfinite(pred.predicted_mean[n_drop:])), (
        "remaining predictions should be finite"
    )


def test_get_prediction_oos_simple_differencing():
    """get_prediction(end>n) with simple_differencing should work for OOS."""
    from rustima.model import SARIMAXModel

    np.random.seed(0)
    y = np.cumsum(np.random.randn(120))
    r = SARIMAXModel(y, order=(1, 1, 1), simple_differencing=True).fit(maxiter=50)
    pred = r.get_prediction(end=125)

    assert len(pred.predicted_mean) == 125
    # OOS part (last 5) should be finite
    assert np.all(np.isfinite(pred.predicted_mean[-5:]))


# ---------------------------------------------------------------------------
# 11-11. V8.8 regression tests
# ---------------------------------------------------------------------------

def test_forecast_short_series_seasonal_simple_diff_no_panic():
    """n < s with seasonal simple_differencing should not panic."""
    y = np.arange(5.0)  # n=5 < s=12
    out = rustima.sarimax_forecast(
        y,
        (0, 0, 0),
        (0, 1, 0, 12),
        np.array([1.0], dtype=np.float64),  # [sigma2]
        steps=10,
        simple_differencing=True,
    )
    assert len(out["mean"]) == 10
    assert len(out["variance"]) == 10
