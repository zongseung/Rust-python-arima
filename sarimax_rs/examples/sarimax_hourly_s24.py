"""
SARIMAX 시간단위 예제 (s=24): Rust sarimax_rs vs statsmodels

시나리오: 시간별 전력 수요 데이터 (n=2160, 90일)
  - 계절성 s=24 (일별 패턴)
  - 외생변수 2개: 기온(temperature), 요일더미(is_weekend)
"""
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sarimax_rs import sarimax_fit, sarimax_forecast

np.random.seed(42)

# ──────────────────────────────────────────────────────────────────
# 1. 시간별 시뮬레이션 데이터 생성 (90일 = 2160시간)
# ──────────────────────────────────────────────────────────────────
n_days = 90
n = n_days * 24  # 2160
t = np.arange(n)
hour = t % 24
day = t // 24

# 외생변수
temperature = (20 + 8 * np.sin(2 * np.pi * (hour - 6) / 24)  # 일중 기온 변화
               + 5 * np.sin(2 * np.pi * day / 365)             # 계절 트렌드
               + np.random.randn(n) * 2)                       # 잡음
is_weekend = ((day % 7 >= 5).astype(float))                    # 주말=1

# 진짜 계수
beta_temp = 3.5     # 기온 → 전력 (냉방)
beta_wknd = -50.0   # 주말 → 수요 감소

# SARIMA(1,1,1)(1,0,1,24) + exog 기반 전력 수요
y = np.zeros(n)
eps = np.random.randn(n) * 15
for i in range(1, n):
    daily_pattern = 80 * np.sin(2 * np.pi * (hour[i] - 6) / 24)  # 낮에 수요 피크
    y[i] = (0.4 * y[i-1]
            + beta_temp * temperature[i]
            + beta_wknd * is_weekend[i]
            + daily_pattern + eps[i])

y = np.cumsum(y) + 5000  # 비정상화 + 기저 수요

exog = np.column_stack([temperature, is_weekend])

# Train/Test 분할: 마지막 48시간 (2일) 예측
h = 48
y_train, y_test = y[:n-h], y[n-h:]
exog_train, exog_test = exog[:n-h], exog[n-h:]

print("=" * 85)
print("  SARIMAX 시간단위 예제 (s=24): Rust vs statsmodels")
print("=" * 85)
print(f"\n  데이터: {n}시간 ({n_days}일), train={len(y_train)}, test={h}시간")
print(f"  외생변수: temperature (기온), is_weekend (주말더미)")

# ──────────────────────────────────────────────────────────────────
# 2. 모델 목록
# ──────────────────────────────────────────────────────────────────
MODELS = [
    ("SARIMAX(1,1,1)(1,0,1,24)+exog2", (1,1,1), (1,0,1,24)),
    ("SARIMAX(2,1,1)(1,0,1,24)+exog2", (2,1,1), (1,0,1,24)),
    ("SARIMAX(1,1,1)(1,1,1,24)+exog2", (1,1,1), (1,1,1,24)),
    ("SARIMAX(2,1,2)(1,1,1,24)+exog2", (2,1,2), (1,1,1,24)),
]

print(f"\n{'='*85}")
print(f"  {'모델':<40s} | {'statsmodels':>12s} | {'Rust':>12s} | {'배속':>7s} | {'LL차이':>8s} | {'RMSE sm':>8s} | {'RMSE rs':>8s}")
print(f"  {'-'*83}")

for label, order, seasonal in MODELS:
    # ── statsmodels ──
    t0 = time.perf_counter()
    try:
        sm_model = SARIMAX(y_train, exog=exog_train,
                           order=order, seasonal_order=seasonal,
                           enforce_stationarity=True,
                           enforce_invertibility=True)
        sm_result = sm_model.fit(maxiter=500, disp=False)
        sm_time = time.perf_counter() - t0
        sm_ll = sm_result.llf
        sm_fc = sm_result.forecast(steps=h, exog=exog_test)
        sm_fc = np.array(sm_fc)
        sm_rmse = np.sqrt(np.mean((sm_fc - y_test)**2))
    except Exception as e:
        sm_time = time.perf_counter() - t0
        sm_ll = None
        sm_rmse = None
        sm_fc = None

    # ── Rust ──
    t0 = time.perf_counter()
    try:
        rs_result = sarimax_fit(
            y_train, order=order, seasonal=seasonal,
            exog=exog_train,
            enforce_stationarity=True, enforce_invertibility=True,
            concentrate_scale=True, method="lbfgsb", maxiter=500,
        )
        rs_time = time.perf_counter() - t0
        rs_ll = rs_result["loglike"]
        rs_conv = rs_result["converged"]

        rs_fc_result = sarimax_forecast(
            y_train, order=order, seasonal=seasonal,
            params=np.array(rs_result["params"]),
            steps=h, exog=exog_train, future_exog=exog_test,
            concentrate_scale=True,
        )
        rs_fc = np.array(rs_fc_result["mean"])
        rs_rmse = np.sqrt(np.mean((rs_fc - y_test)**2))
    except Exception as e:
        rs_time = time.perf_counter() - t0
        rs_ll = None
        rs_rmse = None
        rs_fc = None
        rs_conv = str(e)[:30]

    # ── 출력 ──
    speedup = sm_time / rs_time if rs_time > 0 else 0
    ll_diff = abs(sm_ll - rs_ll) if (sm_ll and rs_ll) else -1
    sm_t = f"{sm_time*1000:.0f}ms"
    rs_t = f"{rs_time*1000:.0f}ms"
    sp = f"{speedup:.1f}x"
    ll_d = f"{ll_diff:.2f}" if ll_diff >= 0 else "ERR"
    sm_r = f"{sm_rmse:.1f}" if sm_rmse else "ERR"
    rs_r = f"{rs_rmse:.1f}" if rs_rmse else "ERR"

    print(f"  {label:<40s} | {sm_t:>12s} | {rs_t:>12s} | {sp:>7s} | {ll_d:>8s} | {sm_r:>8s} | {rs_r:>8s}")

print(f"  {'-'*83}")

# ──────────────────────────────────────────────────────────────────
# 3. 최종 모델 상세 비교 (가장 복잡한 모델)
# ──────────────────────────────────────────────────────────────────
print(f"\n{'='*85}")
print(f"  상세 비교: SARIMAX(2,1,2)(1,1,1,24) + exog(2)")
print(f"{'='*85}")

order, seasonal = (2,1,2), (1,1,1,24)

# statsmodels
t0 = time.perf_counter()
sm_model = SARIMAX(y_train, exog=exog_train, order=order, seasonal_order=seasonal,
                   enforce_stationarity=True, enforce_invertibility=True)
sm_result = sm_model.fit(maxiter=500, disp=False)
sm_time = time.perf_counter() - t0
sm_fc = np.array(sm_result.forecast(steps=h, exog=exog_test))

# Rust
t0 = time.perf_counter()
rs_result = sarimax_fit(
    y_train, order=order, seasonal=seasonal, exog=exog_train,
    enforce_stationarity=True, enforce_invertibility=True,
    concentrate_scale=True, method="lbfgsb", maxiter=500,
)
rs_time = time.perf_counter() - t0
rs_fc_result = sarimax_forecast(
    y_train, order=order, seasonal=seasonal,
    params=np.array(rs_result["params"]),
    steps=h, exog=exog_train, future_exog=exog_test,
    concentrate_scale=True,
)
rs_fc = np.array(rs_fc_result["mean"])

print(f"\n  statsmodels: {sm_time*1000:.0f}ms | LogLike={sm_result.llf:.2f}")
print(f"  Rust:        {rs_time*1000:.0f}ms | LogLike={rs_result['loglike']:.2f} | 수렴={rs_result['converged']} | {rs_result['method']}")
print(f"  배속:        {sm_time/rs_time:.1f}x\n")

print(f"  statsmodels 파라미터:")
for name, val in zip(sm_result.param_names, sm_result.params):
    print(f"    {name:>20s} = {val:>10.4f}")

print(f"\n  Rust 파라미터: {[f'{x:.4f}' for x in rs_result['params']]}")

# 48시간 예측 비교 (처음 24시간만 출력)
print(f"\n  48시간 예측 (처음 24시간):")
print(f"  {'시간':>4s} {'실제':>10s} {'sm':>10s} {'Rust':>10s} {'차이':>8s}")
print(f"  {'-'*46}")
for i in range(24):
    h_label = f"{hour[n-48+i]:02d}:00"
    diff = abs(sm_fc[i] - rs_fc[i])
    print(f"  {h_label:>5s} {y_test[i]:>10.1f} {sm_fc[i]:>10.1f} {rs_fc[i]:>10.1f} {diff:>8.1f}")

sm_rmse = np.sqrt(np.mean((sm_fc - y_test)**2))
rs_rmse = np.sqrt(np.mean((rs_fc - y_test)**2))
print(f"\n  48시간 전체 RMSE: statsmodels={sm_rmse:.1f}, Rust={rs_rmse:.1f}")
print(f"{'='*85}")
