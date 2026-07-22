"""
Integration tests for simple_differencing=True (Phase 5-C).

Validates:
1. No panic (basic smoke test)
2. n_obs correction: n_obs = n - d - s*D
3. Loglike/AIC/BIC are finite and self-consistent
4. Forecast produces valid output
5. Loglike evaluation is consistent between sarimax_fit and sarimax_loglike
6. SD vs SS loglikes are in plausible range (not wildly different)
"""

import numpy as np
import pytest
import rustima


@pytest.fixture
def simple_data():
    rng = np.random.default_rng(42)
    return np.cumsum(rng.standard_normal(100))


@pytest.fixture
def seasonal_data():
    rng = np.random.default_rng(42)
    t = np.arange(200)
    return rng.standard_normal(200) + 0.5 * np.sin(2 * np.pi * t / 12)


# ---------------------------------------------------------------------------
# Smoke tests: no panic
# ---------------------------------------------------------------------------

def test_sd_arima111_no_panic(simple_data):
    r = rustima.sarimax_fit(simple_data, (1, 1, 1), (0, 0, 0, 0),
                                simple_differencing=True)
    assert r is not None


def test_sd_arima210_no_panic(simple_data):
    r = rustima.sarimax_fit(simple_data, (2, 1, 0), (0, 0, 0, 0),
                                simple_differencing=True)
    assert r is not None


def test_sd_arima012_no_panic(simple_data):
    r = rustima.sarimax_fit(simple_data, (0, 1, 2), (0, 0, 0, 0),
                                simple_differencing=True)
    assert r is not None


def test_sd_arima121_d2_no_panic(simple_data):
    r = rustima.sarimax_fit(simple_data, (1, 2, 1), (0, 0, 0, 0),
                                simple_differencing=True)
    assert r is not None


def test_sd_arima100_d1_no_panic(simple_data):
    r = rustima.sarimax_fit(simple_data, (1, 1, 0), (0, 0, 0, 0),
                                simple_differencing=True)
    assert r is not None


def test_sd_arima001_d1_no_panic(simple_data):
    r = rustima.sarimax_fit(simple_data, (0, 1, 1), (0, 0, 0, 0),
                                simple_differencing=True)
    assert r is not None


# ---------------------------------------------------------------------------
# n_obs correctness: n_obs = n - d - s*D
# ---------------------------------------------------------------------------

def test_sd_nobs_d1(simple_data):
    n = len(simple_data)
    r = rustima.sarimax_fit(simple_data, (1, 1, 1), (0, 0, 0, 0),
                                simple_differencing=True)
    assert r["n_obs"] == n - 1


def test_sd_nobs_d2(simple_data):
    n = len(simple_data)
    r = rustima.sarimax_fit(simple_data, (1, 2, 1), (0, 0, 0, 0),
                                simple_differencing=True)
    assert r["n_obs"] == n - 2


def test_sd_nobs_seasonal_D1(seasonal_data):
    n = len(seasonal_data)
    s = 12
    r = rustima.sarimax_fit(seasonal_data, (1, 0, 1), (1, 1, 1, s),
                                simple_differencing=True)
    assert r["n_obs"] == n - s


def test_sd_nobs_d1_D1(seasonal_data):
    n = len(seasonal_data)
    s = 12
    r = rustima.sarimax_fit(seasonal_data, (1, 1, 1), (1, 1, 1, s),
                                simple_differencing=True)
    assert r["n_obs"] == n - 1 - s


def test_ss_nobs_unchanged_d1(simple_data):
    """Without simple_differencing, n_obs = n (full series)."""
    n = len(simple_data)
    r = rustima.sarimax_fit(simple_data, (1, 1, 1), (0, 0, 0, 0),
                                simple_differencing=False)
    assert r["n_obs"] == n


# ---------------------------------------------------------------------------
# AIC/BIC finite and formula consistency
# ---------------------------------------------------------------------------

def test_sd_aic_bic_finite(simple_data):
    r = rustima.sarimax_fit(simple_data, (1, 1, 1), (0, 0, 0, 0),
                                simple_differencing=True)
    assert np.isfinite(r["aic"])
    assert np.isfinite(r["bic"])
    assert np.isfinite(r["loglike"])


def test_sd_aic_formula(simple_data):
    """AIC = -2*loglike + 2*k where k = n_params."""
    r = rustima.sarimax_fit(simple_data, (1, 1, 1), (0, 0, 0, 0),
                                simple_differencing=True)
    expected_aic = -2 * r["loglike"] + 2 * r["n_params"]
    assert abs(r["aic"] - expected_aic) < 1e-6


def test_sd_bic_formula(simple_data):
    """BIC = -2*loglike + k*log(n_obs) where n_obs is the effective obs count."""
    r = rustima.sarimax_fit(simple_data, (1, 1, 1), (0, 0, 0, 0),
                                simple_differencing=True)
    expected_bic = -2 * r["loglike"] + r["n_params"] * np.log(r["n_obs"])
    assert abs(r["bic"] - expected_bic) < 1e-6


# ---------------------------------------------------------------------------
# Loglike evaluation consistency (fit vs eval)
# ---------------------------------------------------------------------------

def test_sd_loglike_consistent_fit_vs_eval(simple_data):
    """sarimax_loglike at fitted params should match r['loglike'] from fit."""
    r = rustima.sarimax_fit(simple_data, (1, 1, 1), (0, 0, 0, 0),
                                simple_differencing=True)
    params_flat = np.array(r["params"])
    ll_eval = rustima.sarimax_loglike(simple_data, (1, 1, 1), (0, 0, 0, 0),
                                          params_flat, simple_differencing=True)
    assert abs(ll_eval - r["loglike"]) < 1e-4, (
        f"loglike mismatch: fit={r['loglike']:.6f}  eval={ll_eval:.6f}"
    )


def test_sd_loglike_path_same_as_arma_on_diff(simple_data):
    """
    sarimax_loglike with simple_differencing=True on y should equal
    sarimax_loglike with simple_differencing=False on diff(y) (d=1 case).
    """
    params = np.array([0.3, -0.4, 1.0])  # [ar.L1, ma.L1, sigma2]
    ll_sd = rustima.sarimax_loglike(
        simple_data, (1, 1, 1), (0, 0, 0, 0), params, simple_differencing=True
    )
    dy = np.diff(simple_data)
    ll_arma = rustima.sarimax_loglike(
        dy, (1, 0, 1), (0, 0, 0, 0), params, simple_differencing=False
    )
    assert abs(ll_sd - ll_arma) < 1e-9, (
        f"SD path ({ll_sd:.6f}) != ARMA on diff ({ll_arma:.6f})"
    )


# ---------------------------------------------------------------------------
# SD vs SS loglike comparison (sanity: should be close)
# ---------------------------------------------------------------------------

def test_sd_ss_loglike_close_d1(simple_data):
    """
    With d=1, SD and SS loglikes should be within ~3 log-units
    (different init, same effective obs count).
    """
    r_sd = rustima.sarimax_fit(simple_data, (1, 1, 1), (0, 0, 0, 0),
                                   simple_differencing=True)
    r_ss = rustima.sarimax_fit(simple_data, (1, 1, 1), (0, 0, 0, 0),
                                   simple_differencing=False)
    diff = abs(r_sd["loglike"] - r_ss["loglike"])
    assert diff < 5.0, f"SD/SS loglike diff={diff:.3f} seems too large"


# ---------------------------------------------------------------------------
# Forecast tests
# ---------------------------------------------------------------------------

def test_sd_forecast_no_panic(simple_data):
    r_fit = rustima.sarimax_fit(simple_data, (1, 1, 1), (0, 0, 0, 0),
                                    simple_differencing=True)
    params_flat = np.array(r_fit["params"])
    fc = rustima.sarimax_forecast(simple_data, (1, 1, 1), (0, 0, 0, 0),
                                      params_flat, 5, simple_differencing=True)
    assert fc is not None


def test_sd_forecast_length(simple_data):
    r_fit = rustima.sarimax_fit(simple_data, (1, 1, 1), (0, 0, 0, 0),
                                    simple_differencing=True)
    params_flat = np.array(r_fit["params"])
    steps = 10
    fc = rustima.sarimax_forecast(simple_data, (1, 1, 1), (0, 0, 0, 0),
                                      params_flat, steps, simple_differencing=True)
    assert len(fc["mean"]) == steps
    assert len(fc["ci_lower"]) == steps
    assert len(fc["ci_upper"]) == steps


def test_sd_forecast_finite(simple_data):
    r_fit = rustima.sarimax_fit(simple_data, (1, 1, 1), (0, 0, 0, 0),
                                    simple_differencing=True)
    params_flat = np.array(r_fit["params"])
    fc = rustima.sarimax_forecast(simple_data, (1, 1, 1), (0, 0, 0, 0),
                                      params_flat, 5, simple_differencing=True)
    assert all(np.isfinite(fc["mean"])), "Forecast mean contains non-finite"
    assert all(np.isfinite(fc["ci_lower"])), "Forecast ci_lower contains non-finite"
    assert all(np.isfinite(fc["ci_upper"])), "Forecast ci_upper contains non-finite"


def test_sd_forecast_ci_ordering(simple_data):
    """ci_lower <= mean <= ci_upper for all forecast steps."""
    r_fit = rustima.sarimax_fit(simple_data, (1, 1, 1), (0, 0, 0, 0),
                                    simple_differencing=True)
    params_flat = np.array(r_fit["params"])
    fc = rustima.sarimax_forecast(simple_data, (1, 1, 1), (0, 0, 0, 0),
                                      params_flat, 10, simple_differencing=True)
    for i in range(10):
        assert fc["ci_lower"][i] <= fc["mean"][i] <= fc["ci_upper"][i], (
            f"CI ordering violated at step {i}: "
            f"ci_lower={fc['ci_lower'][i]:.4f} mean={fc['mean'][i]:.4f} ci_upper={fc['ci_upper'][i]:.4f}"
        )


# ---------------------------------------------------------------------------
# get_prediction scale consistency (guards the "endog - residuals" path)
# ---------------------------------------------------------------------------
# A prior code review flagged that in-sample get_prediction() with
# simple_differencing=True computes `endog[n_drop:] - residuals`, mixing the
# original-scale endog with innovations of the *differenced* series, and asked
# whether the result is on the wrong scale. It is not: the one-step-ahead
# innovation is scale-invariant (v_t of the differenced series equals the
# one-step prediction error of the original series), so `endog - residuals`
# yields the correct original-scale prediction. These tests lock that in.

def test_sd_get_prediction_tracks_original_scale(simple_data):
    """In-sample predictions must live on the ORIGINAL series scale, tracking
    the level of endog — not the near-zero differenced scale. A scale-mixing
    bug would blow the error up to the magnitude of the series."""
    from rustima import SARIMAXModel

    m = SARIMAXModel(simple_data, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
                     trend="n", simple_differencing=True)
    pred = np.asarray(m.fit().get_prediction().predicted_mean, dtype=float)
    mask = ~np.isnan(pred)
    y = np.asarray(simple_data, dtype=float)

    # One-step predictions of a (near) random walk track the level closely.
    corr = np.corrcoef(pred[mask], y[mask])[0, 1]
    mae = np.mean(np.abs(pred[mask] - y[mask]))
    series_range = y.max() - y.min()
    assert corr > 0.85, f"in-sample pred does not track original scale: corr={corr:.3f}"
    assert mae < 0.25 * series_range, (
        f"in-sample pred MAE={mae:.3f} too large vs series range {series_range:.3f} "
        "(symptom of original/differenced scale mixing)"
    )


def test_sd_in_sample_matches_ss_at_tail_same_params():
    """With identical params, the sd=True reconstruction `endog - residuals`
    must converge to the sd=False reconstruction once the Kalman
    initialization transient dies out (non-seasonal d=1). Proves the sd=True
    get_prediction path is scale-equivalent to the well-tested sd=False path.

    Uses fixed params (not a fit) so the check is fully deterministic and
    independent of the optimizer."""
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.standard_normal(120)) + 0.5 * np.arange(120)
    order, seas = (1, 1, 1), (0, 0, 0, 0)
    params = np.array([0.5, -0.5, 1.0])  # ar1, ma1, sigma2

    def reconstruct(sd):
        out = rustima.sarimax_residuals(y, order, seas, params, None, False, "n", sd)
        resid = np.asarray(out["residuals"], dtype=float)
        n_drop = len(y) - len(resid)
        # map to original index: pred[k] corresponds to original index k + n_drop
        return y[n_drop:] - resid, n_drop

    pf, ndf = reconstruct(False)
    pt, ndt = reconstruct(True)
    # align by original index and compare the last 20 (transient fully decayed)
    lo, n = max(ndf, ndt), len(y)
    a = np.array([pf[k - ndf] for k in range(lo, n)])
    b = np.array([pt[k - ndt] for k in range(lo, n)])
    max_tail_diff = np.max(np.abs(a[-20:] - b[-20:]))
    assert max_tail_diff < 1e-6, (
        f"sd=True in-sample reconstruction diverges from sd=False at the tail "
        f"(max diff {max_tail_diff:.2e}) — scale/undiff inconsistency"
    )


# ---------------------------------------------------------------------------
# Default remains simple_differencing=False
# ---------------------------------------------------------------------------

def test_sd_default_is_false(simple_data):
    """Omitting simple_differencing gives same result as False."""
    r_explicit = rustima.sarimax_fit(simple_data, (1, 1, 1), (0, 0, 0, 0),
                                         simple_differencing=False)
    r_default = rustima.sarimax_fit(simple_data, (1, 1, 1), (0, 0, 0, 0))
    assert abs(r_explicit["loglike"] - r_default["loglike"]) < 1e-10
