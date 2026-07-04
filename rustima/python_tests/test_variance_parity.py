"""Forecast-variance / standardized-residual parity vs statsmodels.

Regression tests for the non-concentrated double-scaling bug: in the default
(non-concentrated) mode Q = [[sigma2]], so the filter covariance already
carries sigma2 — the reported forecast variance must be Z'PZ, NOT
Z'PZ * scale (that restoration is only correct in concentrated mode where
Q = [[1]]). The bug inflated variances/CIs by exactly sigma2 and shrank
standardized residuals by sqrt(sigma2).

Why it went unnoticed: earlier fixtures used sigma2 = 1 (identity) or the
concentrated convention.
"""

import numpy as np
import pytest

import rustima
from rustima import SARIMAXModel

sm_sarimax = pytest.importorskip("statsmodels.tsa.statespace.sarimax")


def _arma_series(n, seed=0, scale=1.5):
    """AR(1)-ish series with sigma clearly != 1 so double-scaling is visible."""
    rng = np.random.default_rng(seed)
    e = scale * rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.6 * y[t - 1] + e[t]
    return y + 10.0


def _seasonal_series(n, s=24, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    seasonal = 5.0 * np.sin(2 * np.pi * t / s)
    noise = 1.3 * rng.standard_normal(n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.5 * y[i - 1] + noise[i]
    return 100.0 + seasonal + y


ORDER_LW = (2, 0, 2)
SEASONAL_LW = (1, 1, 1, 24)
PARAMS_LW = np.array([0.5, -0.15, 0.3, 0.1, 0.35, 0.25, 1.7])  # sigma2=1.7 != 1


class TestForecastVarianceParity:
    def test_ar1_variance_matches_statsmodels(self):
        """Same params -> identical forecast variance (default mode)."""
        y = _arma_series(300)
        sm_res = sm_sarimax.SARIMAX(y, order=(1, 0, 0)).fit(disp=False)
        params = np.asarray(sm_res.params)  # [ar.L1, sigma2], sigma2 != 1
        assert abs(params[-1] - 1.0) > 0.2  # 버그가 보이려면 sigma2 != 1

        fc_sm = sm_res.get_forecast(steps=10)
        res_rs = SARIMAXModel(y, order=(1, 0, 0)).filter(params)
        fc_rs = res_rs.get_forecast(steps=10)

        np.testing.assert_allclose(fc_rs.predicted_mean, fc_sm.predicted_mean,
                                   rtol=1e-6)
        np.testing.assert_allclose(fc_rs.variance, fc_sm.var_pred_mean,
                                   rtol=1e-6)
        ci_sm = fc_sm.conf_int(alpha=0.05)
        np.testing.assert_allclose(fc_rs.ci_lower, ci_sm[:, 0], rtol=1e-6)
        np.testing.assert_allclose(fc_rs.ci_upper, ci_sm[:, 1], rtol=1e-6)

    def test_seasonal_variance_matches_statsmodels(self):
        """LoadWarp production order, filter with common params."""
        y = _seasonal_series(400)
        sm_res = sm_sarimax.SARIMAX(
            y, order=ORDER_LW, seasonal_order=SEASONAL_LW
        ).filter(PARAMS_LW)
        fc_sm = sm_res.get_forecast(steps=24)

        res_rs = SARIMAXModel(y, order=ORDER_LW,
                              seasonal_order=SEASONAL_LW).filter(PARAMS_LW)
        fc_rs = res_rs.get_forecast(steps=24)

        # rtol 1e-4: D=1 diffuse 초기화(κ 근사) 차이로 ~1e-5 잔차 존재.
        # 이중곱 버그는 ratio=sigma2(0.7 차이)라 4자릿수 여유로 여전히 검출됨.
        np.testing.assert_allclose(fc_rs.variance, fc_sm.var_pred_mean,
                                   rtol=1e-4)

    def test_concentrated_mode_unchanged(self):
        """Regression guard: concentrated path was correct and must stay so."""
        y = _arma_series(300, seed=3)
        sm_res = sm_sarimax.SARIMAX(y, order=(1, 0, 0)).fit(disp=False)
        params = np.asarray(sm_res.params)
        var_sm = sm_res.get_forecast(steps=5).var_pred_mean

        d = rustima.sarimax_forecast(
            y, (1, 0, 0), (0, 0, 0, 0), params[:-1],  # sigma2 제외
            steps=5, concentrate_scale=True,
        )
        np.testing.assert_allclose(np.asarray(d["variance"]), var_sm, rtol=5e-3)

    def test_rolling_forecast_variance_matches_statsmodels(self):
        """rolling_forecast rows carry correct variances too."""
        y = _seasonal_series(168 + 24 * 3, seed=7)
        res_rs = SARIMAXModel(y, order=ORDER_LW,
                              seasonal_order=SEASONAL_LW).filter(PARAMS_LW)
        roll = res_rs.rolling_forecast(start=168, step=24, horizon=24)

        s = sm_sarimax.SARIMAX(y[:168], order=ORDER_LW,
                               seasonal_order=SEASONAL_LW).filter(PARAMS_LW)
        for k, origin in enumerate(roll.origins):
            fc_sm = s.get_forecast(steps=24)
            np.testing.assert_allclose(roll.variance[k], fc_sm.var_pred_mean,
                                       rtol=1e-4)
            nxt = min(origin + 24, len(y))
            if nxt < len(y) or origin + 24 <= len(y):
                s = s.extend(y[origin:nxt])


class TestStandardizedResidualParity:
    def test_resid_variance_near_one(self):
        """Standardized residuals must be ~unit variance when sigma2 != 1."""
        y = _arma_series(600, seed=5, scale=2.0)
        sm_res = sm_sarimax.SARIMAX(y, order=(1, 0, 0)).fit(disp=False)
        params = np.asarray(sm_res.params)

        res_rs = SARIMAXModel(y, order=(1, 0, 0)).filter(params)
        r = res_rs.resid
        assert abs(np.var(r[10:]) - 1.0) < 0.15

    def test_resid_matches_statsmodels(self):
        y = _arma_series(400, seed=6, scale=1.8)
        sm_res = sm_sarimax.SARIMAX(y, order=(1, 0, 0)).fit(disp=False)
        params = np.asarray(sm_res.params)

        res_rs = SARIMAXModel(y, order=(1, 0, 0)).filter(params)
        r_rs = np.asarray(res_rs.resid)
        r_sm = np.asarray(sm_res.filter_results.standardized_forecasts_error[0])
        # 초기 diffuse 구간 제외하고 비교
        np.testing.assert_allclose(r_rs[10:], r_sm[10:], rtol=1e-4, atol=1e-6)
