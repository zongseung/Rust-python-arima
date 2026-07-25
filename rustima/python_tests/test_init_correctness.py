"""Regression tests for DIAGNOSIS_V9 N1/N2 (2026-07-25).

N1: P0 for stationary states must be the unconditional (Lyapunov) covariance.
    The old DARE-based P0 biased the loglike of every d=0, D=0 model
    (closed form for AR(1): dll ~ -0.5*log(1-phi^2) as phi -> 1).
N2: A deterministic trend must be residualized out of the start-param CSS
    stage; otherwise the AR start saturates outside the stationary box and
    L-BFGS-B stalls on the transform's boundary plateau while reporting
    convergence (self-gap of +10..+192 nats vs its own likelihood).
"""

import numpy as np
import pytest

statsmodels = pytest.importorskip("statsmodels")
import statsmodels.api as sm  # noqa: E402

import rustima  # noqa: E402


def _ar1(n, phi, seed):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n + 200)
    y = np.zeros(n + 200)
    for t in range(1, n + 200):
        y[t] = phi * y[t - 1] + e[t]
    return y[200:]


@pytest.mark.parametrize("phi", [0.5, 0.95, 0.99])
def test_d0_fixed_param_loglike_matches_statsmodels(phi):
    """N1: at FIXED params the d=0 loglike must match statsmodels ~exactly.

    Under the DARE P0 this failed by up to tens of nats near the unit root
    while sitting just under the 3.0-nat suite tolerance at moderate phi.
    """
    y = _ar1(200, phi, seed=33)
    params = np.array([phi, 1.0])  # [ar, sigma2]
    ll_rs = rustima.sarimax_loglike(y, (1, 0, 0), (0, 0, 0, 0), params)
    ll_sm = sm.tsa.SARIMAX(y, order=(1, 0, 0)).loglike(params)
    assert abs(ll_rs - ll_sm) < 1e-6, f"phi={phi}: {ll_rs} vs {ll_sm}"


def _arma_trend(n, seed):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1, n + 500)
    y = np.zeros(n + 500)
    for t in range(1, n + 500):
        y[t] = 0.5 * y[t - 1] + e[t] + 0.3 * e[t - 1]
    return y[500:] + 5.0 + 0.05 * np.arange(n)


@pytest.mark.parametrize("seed", [11, 55, 77])
def test_trend_ct_fit_has_no_self_gap(seed):
    """N2: with trend='ct', d=0, the optimizer must not strand nats below
    its OWN likelihood at statsmodels' parameter vector."""
    y = _arma_trend(200, seed)
    order, seas = (1, 0, 0), (0, 0, 0, 0)
    r_sm = sm.tsa.SARIMAX(y, order=order, trend="ct").fit(disp=False, maxiter=500)
    r_rs = rustima.SARIMAXModel(y, order=order, trend="ct").fit()
    p_sm = np.asarray(r_sm.params, float)
    ll_own = float(rustima.sarimax_loglike(y, order, seas, np.asarray(r_rs.params, float), trend="ct"))
    ll_at_sm = float(rustima.sarimax_loglike(y, order, seas, p_sm, trend="ct"))
    self_gap = ll_at_sm - ll_own  # >0 => rustima missed its own optimum
    assert self_gap < 3.0, f"seed={seed}: self-gap {self_gap:+.2f} nats"


def test_residuals_respect_enforcement_flags():
    """S3/S11: residuals/diagnostics must run under the model's enforcement
    flags (same Kalman init as the fit), not a hardcoded diffuse init."""
    y = _ar1(150, 0.7, seed=5)
    params = np.array([0.7, 1.0])
    r_enforced = rustima.sarimax_residuals(y, (1, 0, 0), (0, 0, 0, 0), params)
    r_diffuse = rustima.sarimax_residuals(
        y, (1, 0, 0), (0, 0, 0, 0), params,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    std_enforced = np.asarray(r_enforced["standardized_residuals"])
    std_diffuse = np.asarray(r_diffuse["standardized_residuals"])
    # The two inits differ (stationary P0 vs kappa*I), which must now be
    # visible through the flag...
    assert not np.allclose(std_enforced, std_diffuse)

    # ...and the result object must follow the model's (default true) flags.
    # res.resid returns STANDARDIZED residuals.
    model = rustima.SARIMAXModel(y, order=(1, 0, 0))
    res = model.fit()
    r_model = rustima.sarimax_residuals(
        y, (1, 0, 0), (0, 0, 0, 0), np.asarray(res.params, float))
    np.testing.assert_allclose(np.asarray(res.resid),
                               np.asarray(r_model["standardized_residuals"]))


def test_get_prediction_conf_int_matches_statsmodels():
    """S14: PredictionResult.conf_int() must exist (statsmodels-compat) and
    match statsmodels' in-sample one-step CI; alpha must apply to the whole
    range, not only the out-of-sample tail."""
    y = _ar1(200, 0.7, seed=9)
    res = rustima.SARIMAXModel(y, order=(1, 0, 0)).fit()
    pred = res.get_prediction(start=0, end=len(y))
    ci = pred.conf_int()
    assert ci.shape == (len(y), 2)

    sm_res = sm.tsa.SARIMAX(y, order=(1, 0, 0)).fit(disp=0)
    sm_pred = sm_res.get_prediction(start=0, end=len(y) - 1)
    sm_ci = np.asarray(sm_pred.conf_int())
    # skip the first few obs where init transients differ most
    np.testing.assert_allclose(ci[5:], sm_ci[5:], rtol=0.05, atol=0.15)

    # narrower alpha -> wider CI
    wide = pred.conf_int(alpha=0.01)
    assert np.all((wide[5:, 1] - wide[5:, 0]) > (ci[5:, 1] - ci[5:, 0]))
