"""
다양한 ARIMA/SARIMA 모델 차수에서 sarimax_rs vs statsmodels 정확도 비교.

테스트 범위:
- AR(p): p=1,2,3
- MA(q): q=1,2
- ARMA(p,q): (1,1), (2,1), (1,2), (2,2)
- ARIMA(p,d,q): (1,1,1), (2,1,1), (1,1,2), (2,1,2)
- SARIMA: (1,0,0)(1,0,0,4), (1,1,1)(1,0,1,12), (0,1,1)(0,1,1,12)
"""

import numpy as np
import pytest
import sarimax_rs
import statsmodels.api as sm

PARAM_TOL = 0.05      # 파라미터 허용 오차
LOGLIKE_TOL = 5.0     # 로그우도 허용 오차
AIC_TOL = 10.0        # AIC 허용 오차


def generate_data(n=300, seed=42):
    """Deterministic test data."""
    np.random.seed(seed)
    return np.cumsum(np.random.randn(n))


def generate_stationary_data(n=300, seed=42):
    """Stationary AR(1)-like data."""
    np.random.seed(seed)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + np.random.randn()
    return y


def generate_seasonal_data(n=300, s=12, seed=42):
    """Data with seasonal pattern."""
    np.random.seed(seed)
    y = np.zeros(n)
    for t in range(max(1, s), n):
        y[t] = 0.3 * y[t - 1] + (0.4 * y[t - s] if t >= s else 0) + np.random.randn()
    return y


def compare_fit(y, order, seasonal_order=(0, 0, 0, 0), label=""):
    """sarimax_rs와 statsmodels 결과를 비교하고 오차를 반환."""
    # statsmodels
    sm_model = sm.tsa.SARIMAX(
        y, order=order, seasonal_order=seasonal_order,
        trend="n", enforce_stationarity=True,
        enforce_invertibility=True, concentrate_scale=True,
    )
    sm_res = sm_model.fit(disp=False)

    # sarimax_rs
    rs_res = sarimax_rs.sarimax_fit(
        y, order=order, seasonal=tuple(seasonal_order),
        concentrate_scale=True,
        enforce_stationarity=True, enforce_invertibility=True,
    )

    # 오차 계산
    sm_params = np.array(sm_res.params)
    rs_params = np.array(rs_res["params"])

    param_errors = np.abs(sm_params - rs_params)
    max_param_err = np.max(param_errors) if len(param_errors) > 0 else 0
    loglike_err = abs(sm_res.llf - rs_res["loglike"])
    aic_err = abs(sm_res.aic - rs_res["aic"])

    result = {
        "label": label,
        "order": order,
        "seasonal": seasonal_order,
        "sm_params": sm_params.tolist(),
        "rs_params": rs_params.tolist(),
        "param_errors": param_errors.tolist(),
        "max_param_err": max_param_err,
        "sm_loglike": sm_res.llf,
        "rs_loglike": rs_res["loglike"],
        "loglike_err": loglike_err,
        "sm_aic": sm_res.aic,
        "rs_aic": rs_res["aic"],
        "aic_err": aic_err,
        "rs_converged": rs_res["converged"],
        "rs_n_iter": rs_res["n_iter"],
        "rs_method": rs_res["method"],
    }

    return result


# === Non-seasonal models ===

class TestARModels:
    """AR(p) models: p=1,2,3"""

    def test_ar1(self):
        y = generate_stationary_data(300, seed=42)
        r = compare_fit(y, (1, 0, 0), label="AR(1)")
        assert r["max_param_err"] < PARAM_TOL, f"AR(1) param err: {r['max_param_err']:.6f}"
        assert r["loglike_err"] < LOGLIKE_TOL, f"AR(1) loglike err: {r['loglike_err']:.4f}"

    def test_ar2(self):
        y = generate_stationary_data(300, seed=43)
        r = compare_fit(y, (2, 0, 0), label="AR(2)")
        assert r["max_param_err"] < PARAM_TOL, f"AR(2) param err: {r['max_param_err']:.6f}"
        assert r["loglike_err"] < LOGLIKE_TOL, f"AR(2) loglike err: {r['loglike_err']:.4f}"

    def test_ar3(self):
        y = generate_stationary_data(300, seed=44)
        r = compare_fit(y, (3, 0, 0), label="AR(3)")
        assert r["max_param_err"] < PARAM_TOL, f"AR(3) param err: {r['max_param_err']:.6f}"
        assert r["loglike_err"] < LOGLIKE_TOL, f"AR(3) loglike err: {r['loglike_err']:.4f}"


class TestMAModels:
    """MA(q) models: q=1,2"""

    def test_ma1(self):
        y = generate_stationary_data(300, seed=50)
        r = compare_fit(y, (0, 0, 1), label="MA(1)")
        assert r["max_param_err"] < PARAM_TOL, f"MA(1) param err: {r['max_param_err']:.6f}"
        assert r["loglike_err"] < LOGLIKE_TOL, f"MA(1) loglike err: {r['loglike_err']:.4f}"

    def test_ma2(self):
        y = generate_stationary_data(300, seed=51)
        r = compare_fit(y, (0, 0, 2), label="MA(2)")
        assert r["max_param_err"] < PARAM_TOL, f"MA(2) param err: {r['max_param_err']:.6f}"
        assert r["loglike_err"] < LOGLIKE_TOL, f"MA(2) loglike err: {r['loglike_err']:.4f}"


class TestARMAModels:
    """ARMA(p,q) models"""

    def test_arma11(self):
        y = generate_stationary_data(300, seed=60)
        r = compare_fit(y, (1, 0, 1), label="ARMA(1,1)")
        assert r["max_param_err"] < PARAM_TOL, f"ARMA(1,1) param err: {r['max_param_err']:.6f}"
        assert r["loglike_err"] < LOGLIKE_TOL, f"ARMA(1,1) loglike err: {r['loglike_err']:.4f}"

    def test_arma21(self):
        y = generate_stationary_data(300, seed=61)
        r = compare_fit(y, (2, 0, 1), label="ARMA(2,1)")
        assert r["max_param_err"] < PARAM_TOL, f"ARMA(2,1) param err: {r['max_param_err']:.6f}"
        assert r["loglike_err"] < LOGLIKE_TOL, f"ARMA(2,1) loglike err: {r['loglike_err']:.4f}"

    def test_arma12(self):
        y = generate_stationary_data(300, seed=62)
        r = compare_fit(y, (1, 0, 2), label="ARMA(1,2)")
        assert r["max_param_err"] < PARAM_TOL, f"ARMA(1,2) param err: {r['max_param_err']:.6f}"
        assert r["loglike_err"] < LOGLIKE_TOL, f"ARMA(1,2) loglike err: {r['loglike_err']:.4f}"

    def test_arma22(self):
        y = generate_stationary_data(300, seed=63)
        r = compare_fit(y, (2, 0, 2), label="ARMA(2,2)")
        # ARMA(2,2) on AR(1) data is overparameterized: multiple local optima
        # with nearly identical loglike but different params. Check loglike only.
        assert r["loglike_err"] < LOGLIKE_TOL, f"ARMA(2,2) loglike err: {r['loglike_err']:.4f}"


class TestARIMAModels:
    """ARIMA(p,d,q) models with differencing"""

    def test_arima111(self):
        y = generate_data(300, seed=70)
        r = compare_fit(y, (1, 1, 1), label="ARIMA(1,1,1)")
        assert r["max_param_err"] < PARAM_TOL, f"ARIMA(1,1,1) param err: {r['max_param_err']:.6f}"
        assert r["loglike_err"] < LOGLIKE_TOL, f"ARIMA(1,1,1) loglike err: {r['loglike_err']:.4f}"

    def test_arima211(self):
        y = generate_data(300, seed=71)
        r = compare_fit(y, (2, 1, 1), label="ARIMA(2,1,1)")
        assert r["max_param_err"] < PARAM_TOL, f"ARIMA(2,1,1) param err: {r['max_param_err']:.6f}"
        assert r["loglike_err"] < LOGLIKE_TOL, f"ARIMA(2,1,1) loglike err: {r['loglike_err']:.4f}"

    def test_arima112(self):
        y = generate_data(300, seed=72)
        r = compare_fit(y, (1, 1, 2), label="ARIMA(1,1,2)")
        assert r["max_param_err"] < PARAM_TOL, f"ARIMA(1,1,2) param err: {r['max_param_err']:.6f}"
        assert r["loglike_err"] < LOGLIKE_TOL, f"ARIMA(1,1,2) loglike err: {r['loglike_err']:.4f}"

    def test_arima212(self):
        y = generate_data(300, seed=73)
        r = compare_fit(y, (2, 1, 2), label="ARIMA(2,1,2)")
        assert r["max_param_err"] < PARAM_TOL, f"ARIMA(2,1,2) param err: {r['max_param_err']:.6f}"
        assert r["loglike_err"] < LOGLIKE_TOL, f"ARIMA(2,1,2) loglike err: {r['loglike_err']:.4f}"

    def test_arima310(self):
        y = generate_data(300, seed=74)
        r = compare_fit(y, (3, 1, 0), label="ARIMA(3,1,0)")
        assert r["max_param_err"] < PARAM_TOL, f"ARIMA(3,1,0) param err: {r['max_param_err']:.6f}"
        assert r["loglike_err"] < LOGLIKE_TOL, f"ARIMA(3,1,0) loglike err: {r['loglike_err']:.4f}"

    def test_arima013(self):
        y = generate_data(300, seed=75)
        r = compare_fit(y, (0, 1, 3), label="ARIMA(0,1,3)")
        assert r["max_param_err"] < PARAM_TOL, f"ARIMA(0,1,3) param err: {r['max_param_err']:.6f}"
        assert r["loglike_err"] < LOGLIKE_TOL, f"ARIMA(0,1,3) loglike err: {r['loglike_err']:.4f}"


class TestSARIMAModels:
    """SARIMA(p,d,q)(P,D,Q,s) models"""

    def test_sarima_100_100_4(self):
        y = generate_seasonal_data(300, s=4, seed=80)
        r = compare_fit(y, (1, 0, 0), (1, 0, 0, 4), label="SARIMA(1,0,0)(1,0,0,4)")
        assert r["max_param_err"] < PARAM_TOL, f"SARIMA err: {r['max_param_err']:.6f}"
        assert r["loglike_err"] < LOGLIKE_TOL, f"SARIMA loglike err: {r['loglike_err']:.4f}"

    def test_sarima_011_011_12(self):
        y = generate_seasonal_data(300, s=12, seed=81)
        r = compare_fit(y, (0, 1, 1), (0, 1, 1, 12), label="SARIMA(0,1,1)(0,1,1,12)")
        assert r["max_param_err"] < PARAM_TOL, f"SARIMA err: {r['max_param_err']:.6f}"
        assert r["loglike_err"] < LOGLIKE_TOL, f"SARIMA loglike err: {r['loglike_err']:.4f}"

    def test_sarima_110_101_12(self):
        y = generate_seasonal_data(300, s=12, seed=82)
        r = compare_fit(y, (1, 1, 0), (1, 0, 1, 12), label="SARIMA(1,1,0)(1,0,1,12)")
        assert r["max_param_err"] < PARAM_TOL, f"SARIMA err: {r['max_param_err']:.6f}"
        assert r["loglike_err"] < LOGLIKE_TOL, f"SARIMA loglike err: {r['loglike_err']:.4f}"

    def test_sarima_111_111_4(self):
        y = generate_seasonal_data(300, s=4, seed=83)
        r = compare_fit(y, (1, 1, 1), (1, 1, 1, 4), label="SARIMA(1,1,1)(1,1,1,4)")
        assert r["max_param_err"] < PARAM_TOL, f"SARIMA err: {r['max_param_err']:.6f}"
        assert r["loglike_err"] < LOGLIKE_TOL, f"SARIMA loglike err: {r['loglike_err']:.4f}"


# === 전체 오차 요약 리포트 ===

def test_comprehensive_accuracy_report():
    """전체 모델 차수에 대한 정확도 리포트 출력."""
    configs = [
        # (order, seasonal, data_fn, seed, label)
        ((1,0,0), (0,0,0,0), generate_stationary_data, 42, "AR(1)"),
        ((2,0,0), (0,0,0,0), generate_stationary_data, 43, "AR(2)"),
        ((3,0,0), (0,0,0,0), generate_stationary_data, 44, "AR(3)"),
        ((0,0,1), (0,0,0,0), generate_stationary_data, 50, "MA(1)"),
        ((0,0,2), (0,0,0,0), generate_stationary_data, 51, "MA(2)"),
        ((1,0,1), (0,0,0,0), generate_stationary_data, 60, "ARMA(1,1)"),
        ((2,0,1), (0,0,0,0), generate_stationary_data, 61, "ARMA(2,1)"),
        ((1,0,2), (0,0,0,0), generate_stationary_data, 62, "ARMA(1,2)"),
        ((2,0,2), (0,0,0,0), generate_stationary_data, 63, "ARMA(2,2)"),
        ((1,1,1), (0,0,0,0), generate_data, 70, "ARIMA(1,1,1)"),
        ((2,1,1), (0,0,0,0), generate_data, 71, "ARIMA(2,1,1)"),
        ((1,1,2), (0,0,0,0), generate_data, 72, "ARIMA(1,1,2)"),
        ((2,1,2), (0,0,0,0), generate_data, 73, "ARIMA(2,1,2)"),
        ((3,1,0), (0,0,0,0), generate_data, 74, "ARIMA(3,1,0)"),
        ((0,1,3), (0,0,0,0), generate_data, 75, "ARIMA(0,1,3)"),
        ((1,0,0), (1,0,0,4), generate_seasonal_data, 80, "SARIMA(1,0,0)(1,0,0,4)"),
        ((0,1,1), (0,1,1,12), generate_seasonal_data, 81, "SARIMA(0,1,1)(0,1,1,12)"),
        ((1,1,0), (1,0,1,12), generate_seasonal_data, 82, "SARIMA(1,1,0)(1,0,1,12)"),
        ((1,1,1), (1,1,1,4), generate_seasonal_data, 83, "SARIMA(1,1,1)(1,1,1,4)"),
    ]

    print("\n" + "=" * 100)
    print(f"{'Model':<30} {'MaxParamErr':>12} {'LoglikeErr':>12} {'AICErr':>12} {'Conv':>6} {'Iter':>6} {'Method':>15}")
    print("=" * 100)

    failures = []
    for order, seasonal, data_fn, seed, label in configs:
        try:
            if data_fn == generate_seasonal_data:
                s = seasonal[3] if seasonal[3] > 0 else 12
                y = data_fn(300, s=s, seed=seed)
            else:
                y = data_fn(300, seed=seed)

            r = compare_fit(y, order, seasonal, label=label)
            status = "OK" if r["max_param_err"] < PARAM_TOL and r["loglike_err"] < LOGLIKE_TOL else "FAIL"
            if status == "FAIL":
                failures.append(r)
            print(f"{label:<30} {r['max_param_err']:>12.6f} {r['loglike_err']:>12.4f} {r['aic_err']:>12.4f} {str(r['rs_converged']):>6} {r['rs_n_iter']:>6} {r['rs_method']:>15}  {status}")
        except Exception as e:
            print(f"{label:<30} {'ERROR':>12} {str(e)[:60]}")
            failures.append({"label": label, "error": str(e)})

    print("=" * 100)
    print(f"Total: {len(configs)} models, {len(failures)} failures")
    if failures:
        print("\nFailed models:")
        for f in failures:
            if "error" in f:
                print(f"  {f['label']}: {f['error']}")
            else:
                print(f"  {f['label']}: max_param_err={f['max_param_err']:.6f}, loglike_err={f['loglike_err']:.4f}")

    # 이 테스트는 리포트 출력 목적이므로 assertion 없음 (개별 테스트에서 처리)
