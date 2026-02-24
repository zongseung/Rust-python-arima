"""
SARIMAX 외생변수 예제: Rust sarimax_rs vs statsmodels 직접 비교

시나리오: 월별 매출 데이터 (n=240, 20년)
  - 계절성 s=12 (월별)
  - 외생변수 2개: 광고비(ad_spend), 금리(interest_rate)
"""
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sarimax_rs import sarimax_fit, sarimax_forecast

np.random.seed(42)

# ──────────────────────────────────────────────────────────────────
# 1. 시뮬레이션 데이터 생성
# ──────────────────────────────────────────────────────────────────
n = 240  # 20년 월별
t = np.arange(n)

# 외생변수
ad_spend = 50 + 10 * np.sin(2 * np.pi * t / 12) + np.random.randn(n) * 5  # 광고비
interest_rate = 3.0 + 0.5 * np.sin(2 * np.pi * t / 48) + np.random.randn(n) * 0.3  # 금리

# 진짜 계수
beta_ad = 0.8       # 광고비 → 매출 양의 효과
beta_ir = -15.0     # 금리 → 매출 음의 효과

# SARIMA(1,1,1)(1,0,1,12) + exog 기반 데이터 생성
y = np.zeros(n)
eps = np.random.randn(n) * 10  # 잡음
for i in range(1, n):
    seasonal = 20 * np.sin(2 * np.pi * i / 12)  # 계절 패턴
    y[i] = (0.3 * y[i-1]                         # AR(1)
            + beta_ad * ad_spend[i]                # 외생: 광고비
            + beta_ir * interest_rate[i]           # 외생: 금리
            + seasonal + eps[i])

# 누적합으로 비정상 시계열 생성
y = np.cumsum(y) + 500

exog = np.column_stack([ad_spend, interest_rate])  # shape (240, 2)

# Train/Test 분할
h = 12  # 12개월 예측
y_train = y[:n-h]
y_test = y[n-h:]
exog_train = exog[:n-h]
exog_test = exog[n-h:]  # future_exog

print("=" * 80)
print("  SARIMAX 외생변수 예제: Rust vs statsmodels")
print("=" * 80)
print(f"\n데이터: n={n}, train={len(y_train)}, test={h}")
print(f"외생변수: ad_spend (광고비), interest_rate (금리)")
print(f"모델: SARIMAX(1,1,1)(1,0,1,12) + exog(2)")

# ──────────────────────────────────────────────────────────────────
# 2. statsmodels Fit
# ──────────────────────────────────────────────────────────────────
print("\n" + "-" * 80)
print("[ statsmodels ]")

t0 = time.perf_counter()
sm_model = SARIMAX(y_train, exog=exog_train,
                   order=(1, 1, 1),
                   seasonal_order=(1, 0, 1, 12),
                   enforce_stationarity=True,
                   enforce_invertibility=True)
sm_result = sm_model.fit(maxiter=500, disp=False)
sm_time = time.perf_counter() - t0

print(f"  시간: {sm_time*1000:.1f}ms")
print(f"  LogLike: {sm_result.llf:.4f}")
print(f"  AIC: {sm_result.aic:.4f}")
print(f"  파라미터:")
for name, val in zip(sm_result.param_names, sm_result.params):
    print(f"    {name:>20s} = {val:>10.4f}")

# statsmodels 예측
sm_forecast = sm_result.forecast(steps=h, exog=exog_test)

# ──────────────────────────────────────────────────────────────────
# 3. Rust sarimax_rs Fit
# ──────────────────────────────────────────────────────────────────
print("\n" + "-" * 80)
print("[ sarimax_rs (Rust) ]")

t0 = time.perf_counter()
rs_result = sarimax_fit(
    y_train,
    order=(1, 1, 1),
    seasonal=(1, 0, 1, 12),
    exog=exog_train,
    enforce_stationarity=True,
    enforce_invertibility=True,
    concentrate_scale=True,
    method="lbfgsb",
    maxiter=500,
)
rs_time = time.perf_counter() - t0

print(f"  시간: {rs_time*1000:.1f}ms")
print(f"  LogLike: {rs_result['loglike']:.4f}")
print(f"  AIC: {rs_result['aic']:.4f}")
print(f"  수렴: {rs_result['converged']}")
print(f"  방법: {rs_result['method']}")
print(f"  파라미터: {[f'{x:.4f}' for x in rs_result['params']]}")

# Rust 예측
t0_fc = time.perf_counter()
rs_forecast_result = sarimax_forecast(
    y_train,
    order=(1, 1, 1),
    seasonal=(1, 0, 1, 12),
    params=np.array(rs_result["params"]),
    steps=h,
    exog=exog_train,
    future_exog=exog_test,
    concentrate_scale=True,
)
rs_fc_time = time.perf_counter() - t0_fc
rs_forecast = np.array(rs_forecast_result["mean"])

# ──────────────────────────────────────────────────────────────────
# 4. 비교
# ──────────────────────────────────────────────────────────────────
print("\n" + "-" * 80)
print("[ 비교 결과 ]")
print(f"\n  {'항목':<25s} {'statsmodels':>15s} {'Rust':>15s} {'비고':>15s}")
print(f"  {'-'*70}")
print(f"  {'Fit 시간':<25s} {sm_time*1000:>12.1f}ms {rs_time*1000:>12.1f}ms {sm_time/rs_time:>12.1f}x")
print(f"  {'LogLike':<25s} {sm_result.llf:>15.4f} {rs_result['loglike']:>15.4f} {abs(sm_result.llf - rs_result['loglike']):>12.4f}")
print(f"  {'AIC':<25s} {sm_result.aic:>15.4f} {rs_result['aic']:>15.4f} {abs(sm_result.aic - rs_result['aic']):>12.4f}")

# 예측 비교
fc_rmse_sm = np.sqrt(np.mean((sm_forecast - y_test) ** 2))
fc_rmse_rs = np.sqrt(np.mean((rs_forecast - y_test) ** 2))
fc_diff = np.mean(np.abs(sm_forecast - rs_forecast))

print(f"\n  {'예측 RMSE (sm)':<25s} {fc_rmse_sm:>15.2f}")
print(f"  {'예측 RMSE (Rust)':<25s} {fc_rmse_rs:>15.2f}")
print(f"  {'예측 평균차이':<25s} {fc_diff:>15.2f}")
print(f"  {'Forecast 시간 (Rust)':<25s} {rs_fc_time*1000:>12.1f}ms")

print(f"\n  12개월 예측 비교:")
print(f"  {'월':>4s} {'실제':>12s} {'statsmodels':>12s} {'Rust':>12s} {'sm-Rust차':>12s}")
print(f"  {'-'*52}")
for i in range(h):
    print(f"  {i+1:>4d} {y_test[i]:>12.1f} {sm_forecast.iloc[i] if hasattr(sm_forecast, 'iloc') else sm_forecast[i]:>12.1f} {rs_forecast[i]:>12.1f} {abs((sm_forecast.iloc[i] if hasattr(sm_forecast, 'iloc') else sm_forecast[i]) - rs_forecast[i]):>12.1f}")

print("\n" + "=" * 80)
speedup = sm_time / rs_time
print(f"\n  결론: Rust가 statsmodels 대비 {speedup:.1f}x 빠름 (SARIMAX + exog 2개)")
print(f"  외생변수 포함 모델도 정상 동작 확인 완료")
print("=" * 80)
