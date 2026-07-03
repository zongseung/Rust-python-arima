"""Tests for single-pass rolling-origin forecasting.

``SARIMAXResult.rolling_forecast(start, step, horizon)`` computes h-step
forecasts from every origin ``start, start+step, ...`` in ONE Kalman-filter
pass over the full series (O(T + N*h)), with parameters held fixed.

Golden property: because the prefix of a Kalman filter pass is identical to
a filter pass over the prefix, each row must equal the forecast produced by
the (already tested) ``extend`` chain at the same origin — to machine
precision, not just statistical tolerance.
"""

import numpy as np
import pytest

from rustima import SARIMAXModel


def _seasonal_series(n, s=24, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    seasonal = 5.0 * np.sin(2 * np.pi * t / s) + 2.0 * np.cos(4 * np.pi * t / s)
    noise = rng.standard_normal(n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.5 * y[i - 1] + noise[i]
    return 100.0 + seasonal + y


def _arma_series(n, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.6 * y[t - 1] - 0.2 * y[t - 2] + e[t] + 0.3 * e[t - 1]
    return y + 10.0


ORDER = (1, 0, 1)
ORDER_LW = (2, 0, 2)
SEASONAL_LW = (1, 1, 1, 24)
PARAMS_LW = np.array([0.5, -0.15, 0.3, 0.1, 0.35, 0.25, 1.0])
PARAMS_ARMA = np.array([0.5, 0.3, 1.0])


def _extend_chain_forecasts(y, order, seasonal, params, start, step, horizon,
                            trend="n"):
    """Reference implementation: extend-chain loop (already parity-tested)."""
    res = SARIMAXModel(y[:start], order=order, seasonal_order=seasonal,
                       trend=trend).filter(params)
    means, variances, origins = [], [], []
    origin = start
    while origin <= len(y) - 1:
        fc = res.get_forecast(steps=horizon)
        means.append(fc.predicted_mean)
        variances.append(fc.variance)
        origins.append(origin)
        nxt = min(origin + step, len(y))
        res = res.extend(y[origin:nxt])
        origin = nxt
        if origin >= len(y):
            break
    return np.array(origins), np.array(means), np.array(variances)


class TestRollingEquivalence:
    def test_rows_equal_extend_chain(self):
        """Single-pass rolling == extend chain, LoadWarp order, machine tol."""
        y = _seasonal_series(168 + 24 * 6)
        start, step, horizon = 168, 24, 24

        res = SARIMAXModel(y, order=ORDER_LW, seasonal_order=SEASONAL_LW).filter(
            PARAMS_LW
        )
        roll = res.rolling_forecast(start=start, step=step, horizon=horizon)

        origins, means, variances = _extend_chain_forecasts(
            y, ORDER_LW, SEASONAL_LW, PARAMS_LW, start, step, horizon
        )

        np.testing.assert_array_equal(roll.origins, origins)
        np.testing.assert_allclose(roll.predicted_mean, means, atol=1e-8)
        np.testing.assert_allclose(roll.variance, variances, rtol=1e-8, atol=1e-10)

    def test_first_row_equals_prefix_forecast(self):
        y = _arma_series(300)
        res = SARIMAXModel(y, order=ORDER).filter(PARAMS_ARMA)
        roll = res.rolling_forecast(start=200, step=50, horizon=10)

        prefix = SARIMAXModel(y[:200], order=ORDER).filter(PARAMS_ARMA)
        fc = prefix.get_forecast(steps=10)
        np.testing.assert_allclose(roll.predicted_mean[0], fc.predicted_mean,
                                   atol=1e-8)
        np.testing.assert_allclose(roll.variance[0], fc.variance,
                                   rtol=1e-8, atol=1e-10)

    def test_trend_constant_equivalence(self):
        """trend='c' exercises the absolute-time state intercept per origin."""
        y = _arma_series(320, seed=11) + 5.0
        params_c = np.array([5.0 * (1 - 0.5 - 0.3 + 0.5 * 0.3), 0.5, 0.3, 1.0])
        # 위 절편값은 대략치면 충분 — 등가성만 검증(양쪽 동일 params 사용)
        res = SARIMAXModel(y, order=ORDER, trend="c").filter(params_c)
        roll = res.rolling_forecast(start=280, step=10, horizon=5)

        origins, means, _ = _extend_chain_forecasts(
            y, ORDER, (0, 0, 0, 0), params_c, 280, 10, 5, trend="c"
        )
        np.testing.assert_array_equal(roll.origins, origins)
        np.testing.assert_allclose(roll.predicted_mean, means, atol=1e-8)


class TestRollingShapes:
    def test_origins_and_shape(self):
        y = _arma_series(300)
        res = SARIMAXModel(y, order=ORDER).filter(PARAMS_ARMA)
        roll = res.rolling_forecast(start=250, step=20, horizon=7)

        assert list(roll.origins) == [250, 270, 290]
        assert roll.predicted_mean.shape == (3, 7)
        assert roll.variance.shape == (3, 7)
        assert roll.ci_lower.shape == (3, 7)
        assert roll.ci_upper.shape == (3, 7)
        assert np.isfinite(roll.predicted_mean).all()
        assert (roll.ci_lower <= roll.ci_upper).all()

    def test_single_origin(self):
        y = _arma_series(300)
        res = SARIMAXModel(y, order=ORDER).filter(PARAMS_ARMA)
        roll = res.rolling_forecast(start=299, step=100, horizon=3)
        assert list(roll.origins) == [299]
        assert roll.predicted_mean.shape == (1, 3)


class TestRollingValidation:
    def test_bad_start(self):
        y = _arma_series(300)
        res = SARIMAXModel(y, order=ORDER).filter(PARAMS_ARMA)
        with pytest.raises(ValueError, match="start"):
            res.rolling_forecast(start=0, step=24, horizon=24)
        with pytest.raises(ValueError, match="start"):
            res.rolling_forecast(start=300, step=24, horizon=24)  # > n-1

    def test_bad_step_horizon(self):
        y = _arma_series(300)
        res = SARIMAXModel(y, order=ORDER).filter(PARAMS_ARMA)
        with pytest.raises(ValueError, match="step"):
            res.rolling_forecast(start=200, step=0, horizon=24)
        with pytest.raises(ValueError, match="horizon|steps"):
            res.rolling_forecast(start=200, step=24, horizon=0)

    def test_simple_differencing_not_supported(self):
        y = _seasonal_series(400)
        res = SARIMAXModel(
            y, order=ORDER_LW, seasonal_order=SEASONAL_LW,
            simple_differencing=True,
        ).filter(PARAMS_LW)
        with pytest.raises((ValueError, RuntimeError),
                           match="simple_differencing"):
            res.rolling_forecast(start=352, step=24, horizon=24)


class TestRollingExog:
    def test_exog_equivalence_with_extend_chain(self):
        rng = np.random.default_rng(21)
        n = 330
        x = rng.standard_normal((n, 2))
        beta = np.array([1.5, -0.8])
        y = _arma_series(n, seed=21) + x @ beta
        params = np.array([1.5, -0.8, 0.5, 0.3, 1.0])  # [x1,x2,ar,ma,sigma2]
        start, step, horizon = 280, 20, 10

        res = SARIMAXModel(y, order=ORDER, exog=x).filter(params)
        roll = res.rolling_forecast(start=start, step=step, horizon=horizon)

        # extend 체인 참조 (exog 버전)
        ref = SARIMAXModel(y[:start], order=ORDER, exog=x[:start]).filter(params)
        means, origins = [], []
        origin = start
        while origin <= n - horizon:
            fc = ref.get_forecast(steps=horizon, exog=x[origin:origin + horizon])
            means.append(fc.predicted_mean)
            origins.append(origin)
            nxt = min(origin + step, n)
            ref = ref.extend(y[origin:nxt], exog=x[origin:nxt])
            origin = nxt

        np.testing.assert_array_equal(roll.origins, np.array(origins))
        np.testing.assert_allclose(roll.predicted_mean, np.array(means), atol=1e-8)

    def test_exog_origins_capped_by_horizon(self):
        """exog 모델은 in-sample exog 로만 예측 가능 → origin ≤ n - horizon."""
        rng = np.random.default_rng(22)
        n = 300
        x = rng.standard_normal((n, 1))
        y = _arma_series(n, seed=22) + 2.0 * x[:, 0]
        params = np.array([2.0, 0.5, 0.3, 1.0])
        res = SARIMAXModel(y, order=ORDER, exog=x).filter(params)
        roll = res.rolling_forecast(start=280, step=10, horizon=15)
        assert roll.origins.max() <= n - 15


class TestWalkForwardParity:
    def test_matches_walk_forward_all_pattern(self):
        """FSDL-Net walk_forward_all 재현: rolling 1콜 == 수동 extend 루프."""
        y = _seasonal_series(168 + 24 * 8, seed=9)
        n, initial, n_periods = len(y), 168, 24

        res = SARIMAXModel(y, order=ORDER_LW, seasonal_order=SEASONAL_LW).filter(
            PARAMS_LW
        )
        roll = res.rolling_forecast(start=initial, step=n_periods,
                                    horizon=n_periods)

        # 수동 루프 (test_extend 의 walk-forward 스모크와 동일 형태)
        manual = SARIMAXModel(
            y[:initial], order=ORDER_LW, seasonal_order=SEASONAL_LW
        ).filter(PARAMS_LW)
        preds = []
        for i in range(initial, n, n_periods):
            step = min(n_periods, n - i)
            fc = manual.get_forecast(steps=step).predicted_mean
            preds.extend(fc)
            manual = manual.extend(y[i:i + step])

        flat = np.concatenate([
            roll.predicted_mean[k][: min(n_periods, n - o)]
            for k, o in enumerate(roll.origins)
        ])
        np.testing.assert_allclose(flat, np.array(preds), atol=1e-8)
