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
