"""High-value option-matrix cells from the v9 structure audit (Phase 4).

The 64-cell matrix {trend} x {simple_differencing} x {concentrate_scale} x
{exog} x {seasonal} had 52 uncovered cells; these are the ones the audit
ranked as guarding real interaction surfaces:

- trend x seasonal / trend x exog: exercises the score's trend derivative
  against non-trivial state layouts (defends the single-source Trend::basis).
- sd=T x cs=T (and x seasonal): completely uncovered combination.
- sd=T x exog x seasonal: pipeline.rs differences exog columns alongside
  endog — the least-exercised branch that does real work.
- cs=T x seasonal / x exog: concentrate_scale only reachable via the
  low-level API.

Each cell fits rustima and statsmodels on the same simulated data and
compares loglike within the project's cross-implementation tolerance.
"""

import numpy as np
import pytest

statsmodels = pytest.importorskip("statsmodels")
import statsmodels.api as sm  # noqa: E402

import rustima  # noqa: E402

LOGLIKE_TOL = 3.0


def _simulate(n, seed, seasonal=False, trend_slope=0.0):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n + 200)
    y = np.zeros(n + 200)
    for t in range(1, n + 200):
        y[t] = 0.5 * y[t - 1] + e[t] + 0.3 * e[t - 1]
        if seasonal and t >= 12:
            y[t] += 0.4 * y[t - 12]
    y = y[200:]
    if trend_slope:
        y = y + 2.0 + trend_slope * np.arange(n)
    return y


def _exog(n, seed):
    rng = np.random.default_rng(seed + 1000)
    return (np.sin(np.arange(n) * 0.17) + 0.1 * rng.standard_normal(n)).reshape(-1, 1)


def _sm_loglike(y, order, seasonal, trend, exog, sd, cs):
    mod = sm.tsa.SARIMAX(
        y, exog=exog, order=order, seasonal_order=seasonal,
        trend=trend, simple_differencing=sd, concentrate_scale=cs,
    )
    return mod.fit(disp=0, maxiter=500)


CELLS = [
    # id, trend, sd, cs, exog?, order, seasonal_order
    ("c-seasonal", "c", False, False, False, (1, 0, 0), (1, 0, 0, 12)),
    ("c-exog", "c", False, False, True, (1, 0, 1), (0, 0, 0, 0)),
    ("ct-seasonal", "ct", False, False, False, (1, 0, 0), (1, 0, 0, 12)),
    ("sd-cs", "n", True, True, False, (1, 1, 1), (0, 0, 0, 0)),
    ("sd-cs-seasonal", "n", True, True, False, (1, 1, 0), (0, 1, 1, 12)),
    ("sd-exog-seasonal", "n", True, False, True, (1, 1, 0), (0, 1, 1, 12)),
    ("cs-seasonal", "n", False, True, False, (1, 0, 0), (1, 0, 0, 12)),
    ("cs-exog-seasonal", "n", False, True, True, (1, 0, 0), (1, 0, 0, 12)),
]


@pytest.mark.parametrize("cell", CELLS, ids=[c[0] for c in CELLS])
def test_matrix_cell_matches_statsmodels(cell):
    _, trend, sd, cs, use_exog, order, seasonal = cell
    n = 180
    y = _simulate(n, seed=7, seasonal=seasonal[3] > 0, trend_slope=0.05 if trend != "n" else 0.0)
    X = _exog(n, 7) if use_exog else None
    if use_exog:
        y = y + 1.5 * X[:, 0]

    r_sm = _sm_loglike(y, order, seasonal, trend, X, sd, cs)

    kwargs = dict(
        trend=trend, simple_differencing=sd, concentrate_scale=cs,
    )
    if X is not None:
        kwargs["exog"] = X
    r_rs = rustima.sarimax_fit(y, order, seasonal, **kwargs)

    assert r_rs["loglike"] == pytest.approx(r_sm.llf, abs=LOGLIKE_TOL), (
        f"loglike: rustima {r_rs['loglike']:.4f} vs statsmodels {r_sm.llf:.4f}"
    )
    # rustima must never be materially WORSE than statsmodels on its own fit
    assert r_rs["loglike"] > r_sm.llf - LOGLIKE_TOL
