"""Tests for SARIMAXModel.filter(params) and SARIMAXResult.extend(endog).

These two APIs enable walk-forward rolling without refitting:

    res = SARIMAXModel(train, order, seasonal_order).fit()
    for block in blocks:
        fc = res.get_forecast(steps=len(block)).predicted_mean
        res = res.extend(block)          # fixed params, no re-estimation

Semantics: rustima's ``extend`` refilters the FULL extended history with the
existing parameters (statsmodels ``append(refit=False)`` semantics). Because
the Kalman filter is Markovian, forecasts after the extension point are
numerically equivalent to statsmodels' state-carry-over ``extend``.
"""

import numpy as np
import pytest

from rustima import SARIMAXModel


# ---------------------------------------------------------------------------
# Synthetic data helpers (deterministic)
# ---------------------------------------------------------------------------

def _arma_series(n, seed=0):
    """Simple stationary ARMA-ish series."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.6 * y[t - 1] - 0.2 * y[t - 2] + e[t] + 0.3 * e[t - 1]
    return y + 10.0


def _seasonal_series(n, s=24, seed=1):
    """Hourly-like series with seasonal pattern + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    seasonal = 5.0 * np.sin(2 * np.pi * t / s) + 2.0 * np.cos(4 * np.pi * t / s)
    noise = rng.standard_normal(n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.5 * y[i - 1] + noise[i]
    return 100.0 + seasonal + y


ORDER = (1, 0, 1)
SEASONAL_NONE = (0, 0, 0, 0)

# LoadWarp FSDL-Net production order (rustima_requirements §2.3)
ORDER_LW = (2, 0, 2)
SEASONAL_LW = (1, 1, 1, 24)

# Hand-picked stationary/invertible params for the LoadWarp order.
# Layout: [ar.L1, ar.L2, ma.L1, ma.L2, ar.S.L24, ma.S.L24, sigma2]
PARAMS_LW = np.array([0.5, -0.15, 0.3, 0.1, 0.35, 0.25, 1.0])


# ---------------------------------------------------------------------------
# filter(params): fixed-parameter reconstruction (no fitting)
# ---------------------------------------------------------------------------

class TestFilter:
    def test_filter_forecast_matches_fit(self):
        """filter(fit_params) must reproduce the fitted result's forecasts."""
        y = _arma_series(300)
        model = SARIMAXModel(y, order=ORDER, seasonal_order=SEASONAL_NONE)
        res_fit = model.fit()

        res_filt = model.filter(res_fit.params)
        fc_fit = res_fit.get_forecast(steps=10).predicted_mean
        fc_filt = res_filt.get_forecast(steps=10).predicted_mean
        np.testing.assert_allclose(fc_filt, fc_fit, atol=1e-10)

    def test_filter_llf_matches_fit(self):
        y = _arma_series(300)
        model = SARIMAXModel(y, order=ORDER, seasonal_order=SEASONAL_NONE)
        res_fit = model.fit()
        res_filt = model.filter(res_fit.params)

        assert np.isclose(res_filt.llf, res_fit.llf, rtol=1e-8)
        assert np.isclose(res_filt.aic, res_fit.aic, rtol=1e-8)
        assert np.isclose(res_filt.bic, res_fit.bic, rtol=1e-8)
        assert np.isclose(res_filt.scale, res_fit.scale, rtol=1e-8)
        assert res_filt.nobs == res_fit.nobs

    def test_filter_no_fitting_marker(self):
        """filter() must not run the optimizer."""
        y = _arma_series(300)
        model = SARIMAXModel(y, order=ORDER, seasonal_order=SEASONAL_NONE)
        res = model.filter(np.array([0.5, 0.3, 1.0]))
        assert res.method == "filter"
        assert res.converged is True

    def test_filter_param_length_validation(self):
        y = _arma_series(300)
        model = SARIMAXModel(y, order=ORDER, seasonal_order=SEASONAL_NONE)
        with pytest.raises(ValueError, match="params"):
            model.filter(np.array([0.5, 0.3]))  # missing sigma2

    def test_filter_seasonal_order(self):
        """filter works on the LoadWarp production order."""
        y = _seasonal_series(400)
        model = SARIMAXModel(y, order=ORDER_LW, seasonal_order=SEASONAL_LW)
        res = model.filter(PARAMS_LW)
        fc = res.get_forecast(steps=24)
        assert fc.predicted_mean.shape == (24,)
        assert np.isfinite(fc.predicted_mean).all()
        assert np.isfinite(res.llf)


# ---------------------------------------------------------------------------
# extend(endog): walk-forward rolling with fixed params
# ---------------------------------------------------------------------------

class TestExtend:
    def test_extend_chain_equals_full_filter(self):
        """Chained extend == one-shot filter over the concatenated history.

        This is the Markov-equivalence property that makes stateless
        refiltering a correct implementation of extend.
        """
        y = _seasonal_series(400)
        base, b1, b2 = y[:352], y[352:376], y[376:400]

        model = SARIMAXModel(base, order=ORDER_LW, seasonal_order=SEASONAL_LW)
        r0 = model.filter(PARAMS_LW)
        r1 = r0.extend(b1)
        r2 = r1.extend(b2)
        fc_chain = r2.get_forecast(steps=24).predicted_mean

        full = SARIMAXModel(y, order=ORDER_LW, seasonal_order=SEASONAL_LW)
        fc_full = full.filter(PARAMS_LW).get_forecast(steps=24).predicted_mean

        np.testing.assert_allclose(fc_chain, fc_full, atol=1e-10)

    def test_extend_preserves_params_and_config(self):
        y = _arma_series(320)
        model = SARIMAXModel(y[:300], order=ORDER, seasonal_order=SEASONAL_NONE)
        r0 = model.fit()
        r1 = r0.extend(y[300:])

        np.testing.assert_array_equal(r1.params, r0.params)
        assert r1.model.order == ORDER
        assert r1.model.seasonal_order == SEASONAL_NONE
        assert r1.model.trend == model.trend
        assert r1.nobs == 320
        assert r1.method == "filter"

    def test_extend_forecast_uses_new_information(self):
        """Forecast after extend must differ from forecast before extend
        (the new observations shift the state)."""
        y = _seasonal_series(400)
        model = SARIMAXModel(y[:352], order=ORDER_LW, seasonal_order=SEASONAL_LW)
        r0 = model.filter(PARAMS_LW)
        fc_before = r0.get_forecast(steps=24).predicted_mean
        r1 = r0.extend(y[352:376])
        fc_after = r1.get_forecast(steps=24).predicted_mean
        assert not np.allclose(fc_before, fc_after)

    def test_extend_empty_raises(self):
        y = _arma_series(300)
        r = SARIMAXModel(y, order=ORDER).fit()
        with pytest.raises(ValueError, match="empty"):
            r.extend(np.array([]))

    def test_extend_nan_raises(self):
        y = _arma_series(300)
        r = SARIMAXModel(y, order=ORDER).fit()
        with pytest.raises(ValueError, match="NaN|Inf|finite"):
            r.extend(np.array([1.0, np.nan]))

    def test_append_alias(self):
        """append(refit=False) is an alias of extend."""
        y = _arma_series(320)
        r0 = SARIMAXModel(y[:300], order=ORDER).fit()
        fc_extend = r0.extend(y[300:]).get_forecast(steps=5).predicted_mean
        fc_append = r0.append(y[300:], refit=False).get_forecast(steps=5).predicted_mean
        np.testing.assert_allclose(fc_append, fc_extend, atol=1e-12)


class TestExtendExog:
    def test_extend_with_exog(self):
        rng = np.random.default_rng(3)
        n = 320
        x = rng.standard_normal((n, 2))
        y = _arma_series(n, seed=3) + x @ np.array([1.5, -0.8])

        r0 = SARIMAXModel(y[:300], order=ORDER, exog=x[:300]).fit()
        r1 = r0.extend(y[300:], exog=x[300:])
        fc = r1.get_forecast(steps=5, exog=x[315:320])
        assert fc.predicted_mean.shape == (5,)
        assert np.isfinite(fc.predicted_mean).all()

    def test_extend_missing_exog_raises(self):
        rng = np.random.default_rng(4)
        n = 310
        x = rng.standard_normal((n, 1))
        y = _arma_series(n, seed=4) + 2.0 * x[:, 0]
        r0 = SARIMAXModel(y[:300], order=ORDER, exog=x[:300]).fit()
        with pytest.raises(ValueError, match="exog"):
            r0.extend(y[300:])

    def test_extend_unexpected_exog_raises(self):
        y = _arma_series(310)
        r0 = SARIMAXModel(y[:300], order=ORDER).fit()
        with pytest.raises(ValueError, match="exog"):
            r0.extend(y[300:], exog=np.ones((10, 1)))

    def test_extend_exog_shape_mismatch_raises(self):
        rng = np.random.default_rng(5)
        n = 310
        x = rng.standard_normal((n, 2))
        y = _arma_series(n, seed=5)
        r0 = SARIMAXModel(y[:300], order=ORDER, exog=x[:300]).fit()
        with pytest.raises(ValueError, match="exog"):
            r0.extend(y[300:], exog=np.ones((10, 3)))  # wrong n_exog


# ---------------------------------------------------------------------------
# statsmodels parity (rustima_requirements §2.2 / §2.3)
# ---------------------------------------------------------------------------

class TestStatsmodelsParity:
    def test_extend_chain_forecast_parity(self):
        """Same params in both engines -> extend-chain forecasts match.

        statsmodels chain uses true state carry-over extend(); rustima chain
        uses full-history refiltering. Acceptance: atol <= 1e-4
        (rustima_requirements §2.3).
        """
        sm = pytest.importorskip("statsmodels.tsa.statespace.sarimax")

        y = _seasonal_series(424, seed=7)
        base, blocks = y[:352], [y[352:376], y[376:400], y[400:424]]
        steps = 24

        # --- rustima chain ---
        r = SARIMAXModel(base, order=ORDER_LW, seasonal_order=SEASONAL_LW).filter(
            PARAMS_LW
        )
        fc_rs = []
        for blk in blocks:
            fc_rs.append(r.get_forecast(steps=steps).predicted_mean)
            r = r.extend(blk)
        fc_rs_final = r.get_forecast(steps=steps).predicted_mean

        # --- statsmodels chain (state carry-over extend) ---
        m_sm = sm.SARIMAX(base, order=ORDER_LW, seasonal_order=SEASONAL_LW)
        s = m_sm.filter(PARAMS_LW)
        fc_sm = []
        for blk in blocks:
            fc_sm.append(s.get_forecast(steps=steps).predicted_mean)
            s = s.extend(blk)
        fc_sm_final = s.get_forecast(steps=steps).predicted_mean

        for a, b in zip(fc_rs, fc_sm):
            np.testing.assert_allclose(a, b, atol=1e-4)
        np.testing.assert_allclose(fc_rs_final, fc_sm_final, atol=1e-4)


# ---------------------------------------------------------------------------
# LoadWarp walk-forward integration smoke
# ---------------------------------------------------------------------------

class TestWalkForwardSmoke:
    def test_walk_forward_pattern(self):
        """Replicates the FSDL-Net Stage-2 walk-forward loop shape:
        initial fit on 7*24, then 24h blocks: forecast -> extend."""
        y = _seasonal_series(168 + 24 * 5, seed=9)
        initial, n_periods = 168, 24

        res = SARIMAXModel(
            y[:initial], order=ORDER_LW, seasonal_order=SEASONAL_LW
        ).filter(PARAMS_LW)

        preds = [np.nan] * initial
        for i in range(initial, len(y), n_periods):
            step = min(n_periods, len(y) - i)
            fc = res.get_forecast(steps=step).predicted_mean
            preds.extend(fc)
            res = res.extend(y[i : i + step])

        assert len(preds) == len(y)
        assert np.isfinite(preds[initial:]).all()
