"""
실무 고차 ARIMA/SARIMA 모델 정확도 검증

실무에서 자주 등장하는 고차 모델을 대상으로 sarimax_rs vs statsmodels 비교:

비계절:
  - ARIMA(4,1,1)   : 일별 에너지 수요 (주간 자기상관)
  - ARIMA(4,1,4)   : 금융 고차 ARMA
  - ARIMA(3,1,3)   : 복합 AR+MA

계절 (s=4, 분기):
  - SARIMA(3,1,2)(2,1,1,4)  : 분기 고차

계절 (s=7, 주별):
  - SARIMA(2,1,2)(1,1,1,7)  : 주별 판매 데이터

계절 (s=12, 월별):
  - SARIMA(4,1,1)(2,1,1,12) : 월별 고차 AR
  - SARIMA(4,1,4)(2,1,2,12) : 월별 복잡 구조

계절 (s=24, 시간별):
  - SARIMA(2,1,1)(2,1,1,24) : 시간별 전력 수요 (고차 계절)
  - SARIMA(2,1,2)(1,1,1,24) : 시간별 기상 데이터
"""

import numpy as np
import pytest
import sarimax_rs
import statsmodels.api as sm

# 고차 모델 허용 오차: 파라미터 추정치가 여러 local optimum 근처에 분산될 수 있음
PARAM_TOL = 0.15      # 고차 모델 파라미터 허용 오차 (넓게)
LOGLIKE_TOL = 10.0    # 로그우도 허용 오차
AIC_TOL = 20.0        # AIC 허용 오차 (파라미터 수 많아짐)

# 초고차 모델 (q>=3, k_states>=40): loglike만 검증
LOGLIKE_ONLY_TOL = 15.0

# 비교 방향: rs_loglike >= sm_loglike - TOL 이면 OK
# (sarimax_rs가 statsmodels보다 같거나 더 좋은 해를 찾은 경우 항상 OK)
WORSE_THAN_SM_TOL = 15.0  # rs가 sm보다 이 값 이상 나쁠 때만 FAIL


# ── 데이터 생성 ─────────────────────────────────────────────────────────────

def make_ar4_data(n=400, seed=42):
    """일별 에너지 수요 형태: AR(4) 적분 과정."""
    np.random.seed(seed)
    eps = np.random.randn(n)
    dy = np.zeros(n)
    for t in range(4, n):
        dy[t] = (0.35 * dy[t-1] + 0.20 * dy[t-2]
                 - 0.10 * dy[t-3] + 0.05 * dy[t-4] + eps[t])
    return np.cumsum(dy)


def make_arma44_data(n=400, seed=43):
    """금융 시계열 형태: ARMA(4,4) 적분 과정."""
    np.random.seed(seed)
    eps = np.random.randn(n)
    dy = np.zeros(n)
    for t in range(4, n):
        dy[t] = (0.30 * dy[t-1] + 0.15 * dy[t-2]
                 - 0.08 * dy[t-3] + 0.04 * dy[t-4]
                 + eps[t]
                 + 0.25 * eps[t-1] + 0.15 * eps[t-2]
                 - 0.08 * eps[t-3] + 0.04 * eps[t-4])
    return np.cumsum(dy)


def make_arma33_data(n=400, seed=44):
    """ARMA(3,3) 적분 과정."""
    np.random.seed(seed)
    eps = np.random.randn(n)
    dy = np.zeros(n)
    for t in range(3, n):
        dy[t] = (0.40 * dy[t-1] + 0.20 * dy[t-2] - 0.10 * dy[t-3]
                 + eps[t] + 0.30 * eps[t-1] + 0.15 * eps[t-2] - 0.08 * eps[t-3])
    return np.cumsum(dy)


def make_quarterly_data(n=200, s=4, seed=50):
    """분기별 판매: AR(3)+SAR(2), 적분."""
    np.random.seed(seed)
    eps = np.random.randn(n)
    y = np.zeros(n)
    for t in range(2 * s, n):
        y[t] = (0.35 * y[t-1] + 0.20 * y[t-2] - 0.10 * y[t-3]
                + 0.50 * y[t-s] - 0.25 * y[t-2*s]
                + eps[t] + 0.20 * eps[t-1] + 0.10 * eps[t-2])
    return np.cumsum(y)


def make_weekly_data(n=400, s=7, seed=55):
    """주별 소매 판매: AR(2)+SAR(1), 적분."""
    np.random.seed(seed)
    eps = np.random.randn(n)
    y = np.zeros(n)
    for t in range(s, n):
        y[t] = (0.30 * y[t-1] + 0.15 * y[t-2]
                + 0.45 * y[t-s]
                + eps[t]
                + 0.25 * eps[t-1] + 0.15 * eps[t-2])
    return np.cumsum(y)


def make_monthly_data(n=300, s=12, seed=60):
    """월별 판매: AR(4)+SAR(2), 적분."""
    np.random.seed(seed)
    eps = np.random.randn(n)
    y = np.zeros(n)
    for t in range(2 * s, n):
        y[t] = (0.30 * y[t-1] + 0.18 * y[t-2]
                - 0.08 * y[t-3] + 0.04 * y[t-4]
                + 0.45 * y[t-s] - 0.20 * y[t-2*s]
                + eps[t])
    return np.cumsum(y)


def make_monthly_arma44_data(n=400, s=12, seed=65):
    """월별 복잡 구조: ARMA(4,4)+SAR(2)SMA(2), 적분."""
    np.random.seed(seed)
    eps = np.random.randn(n)
    y = np.zeros(n)
    for t in range(2 * s, n):
        y[t] = (0.25 * y[t-1] + 0.12 * y[t-2]
                - 0.06 * y[t-3] + 0.03 * y[t-4]
                + 0.40 * y[t-s] - 0.15 * y[t-2*s]
                + eps[t]
                + 0.20 * eps[t-1] + 0.12 * eps[t-2]
                - 0.06 * eps[t-3] + 0.03 * eps[t-4])
    return np.cumsum(y)


def make_hourly_data_high(n=720, s=24, seed=70):
    """시간별 전력: AR(2)+SAR(2), 일주기+적분.

    n=720 (30일)으로 충분한 관측치 확보.
    강한 계절 패턴으로 식별성 확보.
    """
    np.random.seed(seed)
    t = np.arange(n)
    # 강한 일주기 패턴
    seasonal_pattern = (10.0 * np.sin(2 * np.pi * t / s)
                        + 4.0 * np.sin(4 * np.pi * t / s)
                        + 2.0 * np.cos(2 * np.pi * t / s))
    eps = np.random.randn(n) * 0.5
    dy = np.zeros(n)
    for i in range(2 * s, n):
        dy[i] = (0.20 * dy[i-1] + 0.10 * dy[i-2]
                 + 0.35 * dy[i-s] - 0.15 * dy[i-2*s]
                 + seasonal_pattern[i] + eps[i])
    return np.cumsum(dy)


def make_hourly_data_mid(n=600, s=24, seed=75):
    """시간별 기상: AR(2)MA(2)+SAR(1)SMA(1), 일주기."""
    np.random.seed(seed)
    t = np.arange(n)
    seasonal_pattern = (5.0 * np.sin(2 * np.pi * t / s)
                        + 2.0 * np.sin(4 * np.pi * t / s))
    eps = np.random.randn(n)
    y = np.zeros(n)
    for i in range(s, n):
        y[i] = (0.35 * y[i-1] + 0.15 * y[i-2]
                + 0.40 * y[i-s]
                + seasonal_pattern[i]
                + eps[i] + 0.25 * eps[i-1] + 0.12 * eps[i-2])
    return np.cumsum(y - y.mean())


# ── 공통 비교 함수 ──────────────────────────────────────────────────────────

def compare_fit(y, order, seasonal=(0, 0, 0, 0), label=""):
    """sarimax_rs vs statsmodels 비교. 오차 dict 반환."""
    sm_model = sm.tsa.SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal,
        trend="n",
        enforce_stationarity=True,
        enforce_invertibility=True,
        concentrate_scale=True,
    )
    sm_res = sm_model.fit(disp=False)

    rs_res = sarimax_rs.sarimax_fit(
        y,
        order=order,
        seasonal=tuple(seasonal),
        concentrate_scale=True,
        enforce_stationarity=True,
        enforce_invertibility=True,
    )

    sm_params = np.array(sm_res.params)
    rs_params = np.array(rs_res["params"])
    loglike_err = abs(sm_res.llf - rs_res["loglike"])
    param_errors = np.abs(sm_params - rs_params)
    max_param_err = float(np.max(param_errors)) if len(param_errors) > 0 else 0.0

    # rs가 sm보다 얼마나 나쁜지 (음수면 rs가 더 좋음)
    rs_worse_by = sm_res.llf - rs_res["loglike"]  # 양수 = rs 더 나쁨

    return {
        "label": label,
        "sm_params": sm_params.tolist(),
        "rs_params": rs_params.tolist(),
        "max_param_err": max_param_err,
        "sm_loglike": sm_res.llf,
        "rs_loglike": rs_res["loglike"],
        "loglike_err": loglike_err,
        "rs_worse_by": rs_worse_by,       # 양수: rs 나쁨, 음수: rs 더 좋음
        "sm_aic": sm_res.aic,
        "rs_aic": rs_res["aic"],
        "converged": rs_res["converged"],
        "n_iter": rs_res["n_iter"],
        "method": rs_res["method"],
        "k_states": len(sm_res.filter_results.transition[:, :, 0]),
    }


# ── 비계절 고차 모델 ─────────────────────────────────────────────────────────

class TestHighOrderNonSeasonal:
    """비계절 고차 ARIMA 모델."""

    def _assert_loglike(self, r, tol=LOGLIKE_TOL):
        """rs loglike가 sm보다 tol 이상 나쁘지 않으면 OK."""
        assert r["rs_worse_by"] < tol, (
            f"{r['label']} sarimax_rs가 statsmodels보다 {r['rs_worse_by']:.2f} 더 나쁨 "
            f"(sm={r['sm_loglike']:.2f}, rs={r['rs_loglike']:.2f})"
        )

    def test_arima_411(self):
        """ARIMA(4,1,1): 일별 에너지 수요 형태."""
        y = make_ar4_data(n=400, seed=42)
        r = compare_fit(y, (4, 1, 1), label="ARIMA(4,1,1)")
        self._assert_loglike(r)
        assert r["max_param_err"] < PARAM_TOL, (
            f"ARIMA(4,1,1) max_param_err={r['max_param_err']:.6f}"
        )

    def test_arima_414(self):
        """ARIMA(4,1,4): 금융 고차 ARMA — 파라미터 개별 검증 스킵, loglike만."""
        y = make_arma44_data(n=400, seed=43)
        r = compare_fit(y, (4, 1, 4), label="ARIMA(4,1,4)")
        # 고차 ARMA는 likelihood surface가 평탄 → 더 관대한 TOL 사용
        self._assert_loglike(r, tol=LOGLIKE_ONLY_TOL)

    def test_arima_313(self):
        """ARIMA(3,1,3): 복합 ARMA 구조."""
        y = make_arma33_data(n=400, seed=44)
        r = compare_fit(y, (3, 1, 3), label="ARIMA(3,1,3)")
        self._assert_loglike(r)

    def test_arima_511(self):
        """ARIMA(5,1,1): 최대 비계절 AR 차수."""
        np.random.seed(100)
        y = np.cumsum(np.random.randn(400))
        r = compare_fit(y, (5, 1, 1), label="ARIMA(5,1,1)")
        self._assert_loglike(r)

    def test_arima_115(self):
        """ARIMA(1,1,5): 최대 비계절 MA 차수."""
        np.random.seed(101)
        y = np.cumsum(np.random.randn(400))
        r = compare_fit(y, (1, 1, 5), label="ARIMA(1,1,5)")
        self._assert_loglike(r)


# ── 분기별 고차 모델 (s=4) ──────────────────────────────────────────────────

class TestHighOrderQuarterly:
    """분기별 (s=4) 고차 SARIMA 모델."""

    def test_sarima_312_211_4(self):
        """SARIMA(3,1,2)(2,1,1,4): 분기별 고차 모델.
        k_states = max(3+4*2, 2+4*1+1) + (1+4) = 11+5 = 16
        """
        y = make_quarterly_data(n=200, s=4, seed=50)
        r = compare_fit(y, (3, 1, 2), (2, 1, 1, 4), label="SARIMA(3,1,2)(2,1,1,4)")
        assert r["rs_worse_by"] < LOGLIKE_TOL, (
            f"loglike rs_worse_by={r['rs_worse_by']:.4f}, k_states={r['k_states']}"
        )

    def test_sarima_211_211_4(self):
        """SARIMA(2,1,1)(2,1,1,4): 분기별 중간 차수."""
        y = make_quarterly_data(n=200, s=4, seed=51)
        r = compare_fit(y, (2, 1, 1), (2, 1, 1, 4), label="SARIMA(2,1,1)(2,1,1,4)")
        assert r["rs_worse_by"] < LOGLIKE_TOL, (
            f"loglike rs_worse_by={r['rs_worse_by']:.4f}"
        )


# ── 주별 고차 모델 (s=7) ────────────────────────────────────────────────────

class TestHighOrderWeekly:
    """주별 (s=7) 고차 SARIMA 모델."""

    def test_sarima_212_111_7(self):
        """SARIMA(2,1,2)(1,1,1,7): 주별 판매 데이터.
        k_states = max(2+7, 2+7+1) + (1+7) = 10+8 = 18
        """
        y = make_weekly_data(n=400, s=7, seed=55)
        r = compare_fit(y, (2, 1, 2), (1, 1, 1, 7), label="SARIMA(2,1,2)(1,1,1,7)")
        assert r["rs_worse_by"] < LOGLIKE_TOL, (
            f"loglike rs_worse_by={r['rs_worse_by']:.4f}, k_states={r['k_states']}"
        )
        assert r["max_param_err"] < PARAM_TOL, (
            f"max_param_err={r['max_param_err']:.6f}"
        )

    def test_sarima_311_111_7(self):
        """SARIMA(3,1,1)(1,1,1,7): 주별 고차 AR."""
        y = make_weekly_data(n=400, s=7, seed=56)
        r = compare_fit(y, (3, 1, 1), (1, 1, 1, 7), label="SARIMA(3,1,1)(1,1,1,7)")
        assert r["rs_worse_by"] < LOGLIKE_TOL, (
            f"loglike rs_worse_by={r['rs_worse_by']:.4f}"
        )


# ── 월별 고차 모델 (s=12) ────────────────────────────────────────────────────

class TestHighOrderMonthly:
    """월별 (s=12) 고차 SARIMA 모델."""

    def test_sarima_411_211_12(self):
        """SARIMA(4,1,1)(2,1,1,12): 월별 고차 AR.
        k_states = max(4+24, 1+12+1) + (1+12) = 28+13 = 41
        """
        y = make_monthly_data(n=300, s=12, seed=60)
        r = compare_fit(y, (4, 1, 1), (2, 1, 1, 12), label="SARIMA(4,1,1)(2,1,1,12)")
        assert r["rs_worse_by"] < LOGLIKE_TOL, (
            f"loglike rs_worse_by={r['rs_worse_by']:.4f}, k_states={r['k_states']}"
        )
        assert r["max_param_err"] < PARAM_TOL, (
            f"max_param_err={r['max_param_err']:.6f}"
        )

    def test_sarima_414_212_12(self):
        """SARIMA(4,1,4)(2,1,2,12): 월별 복잡 구조.
        k_states = max(4+24, 4+24+1) + (1+12) = 29+13 = 42
        고차 ARMA: loglike만 검증.
        """
        y = make_monthly_arma44_data(n=400, s=12, seed=65)
        r = compare_fit(y, (4, 1, 4), (2, 1, 2, 12), label="SARIMA(4,1,4)(2,1,2,12)")
        assert r["rs_worse_by"] < LOGLIKE_ONLY_TOL, (
            f"loglike rs_worse_by={r['rs_worse_by']:.4f}, k_states={r['k_states']}"
        )

    def test_sarima_212_211_12(self):
        """SARIMA(2,1,2)(2,1,1,12): 월별 중간 고차."""
        y = make_monthly_data(n=300, s=12, seed=61)
        r = compare_fit(y, (2, 1, 2), (2, 1, 1, 12), label="SARIMA(2,1,2)(2,1,1,12)")
        assert r["rs_worse_by"] < LOGLIKE_TOL, (
            f"loglike rs_worse_by={r['rs_worse_by']:.4f}"
        )

    def test_sarima_311_211_12(self):
        """SARIMA(3,1,1)(2,1,1,12): 월별 중간 고차."""
        y = make_monthly_data(n=300, s=12, seed=62)
        r = compare_fit(y, (3, 1, 1), (2, 1, 1, 12), label="SARIMA(3,1,1)(2,1,1,12)")
        assert r["rs_worse_by"] < LOGLIKE_TOL, (
            f"loglike rs_worse_by={r['rs_worse_by']:.4f}"
        )


# ── 시간별 고차 모델 (s=24) ──────────────────────────────────────────────────

class TestHighOrderHourly:
    """시간별 (s=24) 고차 SARIMA 모델 — 가장 고차원."""

    def test_sarima_211_211_24(self):
        """SARIMA(2,1,1)(2,1,1,24): 시간별 전력 수요.
        k_states = max(2+48, 1+24+1) + (1+24) = 50+25 = 75
        """
        y = make_hourly_data_high(n=720, s=24, seed=70)
        r = compare_fit(y, (2, 1, 1), (2, 1, 1, 24), label="SARIMA(2,1,1)(2,1,1,24)")
        assert r["rs_worse_by"] < LOGLIKE_ONLY_TOL, (
            f"loglike rs_worse_by={r['rs_worse_by']:.4f}, k_states={r['k_states']}"
        )

    def test_sarima_212_111_24(self):
        """SARIMA(2,1,2)(1,1,1,24): 시간별 기상.
        k_states = max(2+24, 2+24+1) + (1+24) = 27+25 = 52
        """
        y = make_hourly_data_mid(n=600, s=24, seed=75)
        r = compare_fit(y, (2, 1, 2), (1, 1, 1, 24), label="SARIMA(2,1,2)(1,1,1,24)")
        assert r["rs_worse_by"] < LOGLIKE_ONLY_TOL, (
            f"loglike rs_worse_by={r['rs_worse_by']:.4f}, k_states={r['k_states']}"
        )

    def test_sarima_111_211_24(self):
        """SARIMA(1,1,1)(2,1,1,24): 기본 + 고차 계절 AR.
        k_states = max(1+48, 1+24+1) + (1+24) = 49+25 = 74
        """
        y = make_hourly_data_high(n=720, s=24, seed=76)
        r = compare_fit(y, (1, 1, 1), (2, 1, 1, 24), label="SARIMA(1,1,1)(2,1,1,24)")
        assert r["rs_worse_by"] < LOGLIKE_ONLY_TOL, (
            f"loglike rs_worse_by={r['rs_worse_by']:.4f}, k_states={r['k_states']}"
        )


# ── 전체 리포트 ──────────────────────────────────────────────────────────────

def test_high_order_comprehensive_report():
    """실무 고차 모델 전체 정확도 리포트 출력 (assertion 없음)."""
    configs = [
        # (order, seasonal, data_fn, kwargs, label)
        ((4,1,1), (0,0,0,0), make_ar4_data,            {"n":400,"seed":42},        "ARIMA(4,1,1)"),
        ((4,1,4), (0,0,0,0), make_arma44_data,          {"n":400,"seed":43},        "ARIMA(4,1,4)"),
        ((3,1,3), (0,0,0,0), make_arma33_data,          {"n":400,"seed":44},        "ARIMA(3,1,3)"),
        ((5,1,1), (0,0,0,0), lambda **kw: np.cumsum(np.random.RandomState(100).randn(kw["n"])),
                                                         {"n":400},                  "ARIMA(5,1,1)"),
        ((1,1,5), (0,0,0,0), lambda **kw: np.cumsum(np.random.RandomState(101).randn(kw["n"])),
                                                         {"n":400},                  "ARIMA(1,1,5)"),
        ((3,1,2), (2,1,1,4), make_quarterly_data,       {"n":200,"s":4,"seed":50},  "SARIMA(3,1,2)(2,1,1,4)"),
        ((2,1,1), (2,1,1,4), make_quarterly_data,       {"n":200,"s":4,"seed":51},  "SARIMA(2,1,1)(2,1,1,4)"),
        ((2,1,2), (1,1,1,7), make_weekly_data,          {"n":400,"s":7,"seed":55},  "SARIMA(2,1,2)(1,1,1,7)"),
        ((3,1,1), (1,1,1,7), make_weekly_data,          {"n":400,"s":7,"seed":56},  "SARIMA(3,1,1)(1,1,1,7)"),
        ((4,1,1), (2,1,1,12), make_monthly_data,        {"n":300,"s":12,"seed":60}, "SARIMA(4,1,1)(2,1,1,12)"),
        ((4,1,4), (2,1,2,12), make_monthly_arma44_data, {"n":400,"s":12,"seed":65}, "SARIMA(4,1,4)(2,1,2,12)"),
        ((2,1,2), (2,1,1,12), make_monthly_data,        {"n":300,"s":12,"seed":61}, "SARIMA(2,1,2)(2,1,1,12)"),
        ((3,1,1), (2,1,1,12), make_monthly_data,        {"n":300,"s":12,"seed":62}, "SARIMA(3,1,1)(2,1,1,12)"),
        ((2,1,1), (2,1,1,24), make_hourly_data_high,    {"n":720,"s":24,"seed":70}, "SARIMA(2,1,1)(2,1,1,24)"),
        ((2,1,2), (1,1,1,24), make_hourly_data_mid,     {"n":600,"s":24,"seed":75}, "SARIMA(2,1,2)(1,1,1,24)"),
        ((1,1,1), (2,1,1,24), make_hourly_data_high,    {"n":720,"s":24,"seed":76}, "SARIMA(1,1,1)(2,1,1,24)"),
    ]

    print("\n" + "=" * 115)
    print(f"{'Model':<35} {'k_st':>5} {'MaxPErr':>9} {'WorsBy':>8} "
          f"{'sm_ll':>10} {'rs_ll':>10} {'Iter':>5} {'Method':>12} {'결과':>6}")
    print("=" * 115)

    failures = []
    for order, seasonal, fn, kwargs, label in configs:
        try:
            y = fn(**kwargs)
            r = compare_fit(y, order, seasonal, label=label)
            # rs_worse_by < 0: sarimax_rs가 statsmodels보다 더 좋음 (항상 OK)
            ok = r["rs_worse_by"] < LOGLIKE_ONLY_TOL
            status = "✓ OK" if ok else "✗ FAIL"
            better = " ★" if r["rs_worse_by"] < 0 else ""
            if not ok:
                failures.append(r)
            print(
                f"{label:<35} {r['k_states']:>5} {r['max_param_err']:>9.4f} "
                f"{r['rs_worse_by']:>8.3f} {r['sm_loglike']:>10.2f} {r['rs_loglike']:>10.2f} "
                f"{r['n_iter']:>5} {r['method']:>12} {status}{better}"
            )
        except Exception as e:
            print(f"{label:<35} {'ERROR':>5}  {str(e)[:70]}")
            failures.append({"label": label, "error": str(e)})

    print("=" * 115)
    print(f"Total: {len(configs)} models tested, {len(failures)} failures")
    print("(★ = sarimax_rs가 statsmodels보다 더 좋은 해 발견, WorsBy < 0)")
    if failures:
        print("Failures:")
        for f in failures:
            if "error" in f:
                print(f"  {f['label']}: {f['error']}")
            else:
                print(f"  {f['label']}: rs_worse_by={f['rs_worse_by']:.4f}")
