"""Regression test for the batched-Kalman-filter optimisation inside
``ProfiledSarimaxObjective::profile_beta_and_loglike``.

Before the batched-KF change, every likelihood evaluation in the
``profile-trust-region`` (PTR) method ran the Kalman filter ``1 + r``
times — once for ``y`` and once per exogenous column. After the change,
all ``1 + r`` series share a single covariance recursion via
``kalman_filter_batched`` while the per-series state-mean recursion runs
in O(k) instead of O(k^2). The two paths are mathematically identical
(the Rust unit test ``test_batched_matches_per_series_filter`` checks
this at the innovation level), so the optimisation trace and final
log-likelihood must be bit-equivalent to the pre-change baseline.

The baseline values below were captured from a clean fresh build of the
unmodified PTR implementation on this exact synthetic dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from rustima import SARIMAXModel


# --- Baseline captured BEFORE the batched-KF change ---
# n=200, SARIMAX(2,0,0)(1,1,0)[24] with 2 exog columns, seed=42.
# Build: maturin develop --release on commit 070e716 (pre-batched).
BASELINE_LL = -172.02511857340443
BASELINE_AIC = 356.05023714680885
BASELINE_PARAMS = [
    2.326649586061251,    # exog_1
    -1.2162988303317972,  # exog_2
    0.6486839648145728,   # ar.L1
    0.11754695800965,     # ar.L2
    -0.5415458513524884,  # sar.L24
    0.3926143947811343,   # sigma2
]


def _build_dataset(n: int = 200, seed: int = 42):
    """Reproducible synthetic dataset matching the baseline-capture run."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([
        np.sin(np.arange(n) * 0.13) + rng.standard_normal(n) * 0.1,
        np.cos(np.arange(n) * 0.07) + rng.standard_normal(n) * 0.1,
    ])
    season = np.sin(np.arange(n) * (2 * np.pi / 24))
    y = np.zeros(n)
    for t in range(n):
        prev = y[t - 1] if t > 0 else 0.0
        y[t] = (
            0.5 * prev
            + 0.3 * season[t]
            + 1.5 * X[t, 0]
            - 0.7 * X[t, 1]
            + rng.standard_normal() * 0.5
        )
    return y, X


def test_ptr_loglike_matches_pre_batched_baseline():
    """The batched-KF PTR path must reproduce the pre-change LL exactly."""
    y, X = _build_dataset()
    model = SARIMAXModel(
        endog=y,
        order=(2, 0, 0),
        seasonal_order=(1, 1, 0, 24),
        trend="n",
        exog=X,
    )
    res = model.fit(method="profile-trust-region", maxiter=500)

    # Log-likelihood preserved to within numerical noise.
    assert res.llf == pytest.approx(BASELINE_LL, abs=1e-8), (
        f"PTR LL drifted from baseline: got {res.llf!r}, expected {BASELINE_LL!r}"
    )

    # AIC is derived from LL and k; preserving LL preserves AIC.
    assert res.aic == pytest.approx(BASELINE_AIC, abs=1e-8)

    # Parameter vector preserved to within numerical noise.
    np.testing.assert_allclose(
        np.asarray(res.params, dtype=np.float64),
        np.asarray(BASELINE_PARAMS, dtype=np.float64),
        rtol=0.0,
        atol=1e-8,
    )


def test_ptr_loglike_finite_and_deterministic():
    """Sanity: repeated fits on the same data give the same result."""
    y, X = _build_dataset()
    model_a = SARIMAXModel(
        endog=y,
        order=(2, 0, 0),
        seasonal_order=(1, 1, 0, 24),
        trend="n",
        exog=X,
    )
    model_b = SARIMAXModel(
        endog=y,
        order=(2, 0, 0),
        seasonal_order=(1, 1, 0, 24),
        trend="n",
        exog=X,
    )
    res_a = model_a.fit(method="profile-trust-region", maxiter=500)
    res_b = model_b.fit(method="profile-trust-region", maxiter=500)

    assert np.isfinite(res_a.llf)
    assert res_a.llf == pytest.approx(res_b.llf, abs=1e-12)
    np.testing.assert_allclose(res_a.params, res_b.params, rtol=0.0, atol=1e-12)


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
