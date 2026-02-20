# sarimax-rs

Rust로 구현한 고성능 SARIMAX 엔진. PyO3를 통해 Python에서 직접 호출 가능하며, statsmodels 대비 동일한 수치 정확도를 유지하면서 네이티브 속도로 동작한다.

## 왜 필요한가

Python의 `statsmodels.tsa.SARIMAX`는 시계열 분석의 사실상 표준이지만, 순수 Python + NumPy 기반이라 구조적 한계가 있다.

| 문제 | 원인 | 영향 |
|------|------|------|
| Kalman 필터 루프가 느림 | Python `for` 루프에서 행렬 연산 반복 | 긴 시계열 또는 고차 모델에서 수 초~수십 초 |
| MLE 최적화 오버헤드 | 매 iteration마다 Python 콜스택 경유 | 수백 회 반복 시 누적 지연 |
| 배치 처리 비효율 | GIL로 인한 병렬화 제한 | 수천 시계열 동시 fitting 불가 |
| 메모리 단편화 | Python 객체 오버헤드 | 대규모 상태공간에서 불필요한 할당 |

**sarimax-rs**는 이 병목을 Rust 네이티브 코드로 대체한다:

- Kalman 필터 루프: Rust `for` + nalgebra → Python 루프 대비 **제로 인터프리터 오버헤드**
- 최적화: argmin 크레이트의 L-BFGS/Nelder-Mead → 순수 Rust 내에서 수백 회 반복
- 메모리: 스택 할당 + 연속 메모리 레이아웃 → 캐시 친화적
- Python 연동: PyO3 + NumPy 바인딩 → `import sarimax_rs`로 즉시 사용

## 지원 모델

```
SARIMA(p, d, q)(P, D, Q, s)
```

- **p**: AR 차수 (자기회귀)
- **d**: 차분 차수
- **q**: MA 차수 (이동평균)
- **P**: 계절 AR 차수
- **D**: 계절 차분 차수 (0 또는 1)
- **Q**: 계절 MA 차수
- **s**: 계절 주기 (예: 12=월별, 4=분기별)

## 설치

```bash
# 요구사항: Rust 1.83+, Python 3.10+, maturin 1.7+
cd Rust-python-arima/sarimax_rs

# Python 휠 빌드 및 설치
pip install maturin
maturin develop --release

# 또는 uv 사용
uv run maturin develop --release
```

## Python API

### `sarimax_rs.sarimax_loglike`

주어진 파라미터에서 로그우도(log-likelihood)를 계산한다.

```python
import numpy as np
import sarimax_rs

y = np.array([...])  # 시계열 데이터

ll = sarimax_rs.sarimax_loglike(
    y,
    order=(1, 1, 1),           # (p, d, q)
    seasonal=(1, 1, 1, 12),    # (P, D, Q, s)
    params=np.array([0.5, 0.3, 0.2, -0.4]),  # [ar, ma, sar, sma]
    concentrate_scale=True,    # sigma2를 우도에서 집중 추정
)
```

### `sarimax_rs.sarimax_fit`

최대우도추정(MLE)으로 모델을 적합한다.

```python
result = sarimax_rs.sarimax_fit(
    y,
    order=(1, 0, 1),
    seasonal=(0, 0, 0, 0),
    enforce_stationarity=True,   # AR 정상성 제약
    enforce_invertibility=True,  # MA 가역성 제약
    method="lbfgs",              # "lbfgs" | "nelder-mead"
    maxiter=500,
)

print(result["params"])     # 추정된 파라미터
print(result["loglike"])    # 최종 로그우도
print(result["aic"])        # AIC
print(result["bic"])        # BIC
print(result["converged"])  # 수렴 여부
```

**반환값** (dict):

| 키 | 설명 |
|----|------|
| `params` | 추정된 파라미터 벡터 `[ar..., ma..., sar..., sma...]` |
| `loglike` | 최종 로그우도 |
| `scale` | 추정된 분산 (sigma2) |
| `aic` | Akaike 정보량 기준 |
| `bic` | Bayesian 정보량 기준 |
| `n_obs` | 관측치 수 |
| `n_params` | 추정 파라미터 수 (sigma2 포함) |
| `n_iter` | 최적화 반복 횟수 |
| `converged` | 수렴 여부 (bool) |
| `method` | 사용된 최적화 방법 |

### `sarimax_rs.sarimax_forecast`

적합된 파라미터로 h-step ahead 예측을 수행한다.

```python
fc = sarimax_rs.sarimax_forecast(
    y,
    order=(1, 0, 0),
    seasonal=(0, 0, 0, 0),
    params=np.array([0.65]),
    steps=10,          # 예측 스텝 수
    alpha=0.05,        # 95% 신뢰구간
)

print(fc["mean"])       # 예측 평균
print(fc["ci_lower"])   # 신뢰구간 하한
print(fc["ci_upper"])   # 신뢰구간 상한
print(fc["variance"])   # 예측 분산
```

### `sarimax_rs.sarimax_residuals`

잔차 및 표준화 잔차를 계산한다.

```python
res = sarimax_rs.sarimax_residuals(
    y,
    order=(1, 0, 1),
    seasonal=(0, 0, 0, 0),
    params=np.array([0.5, 0.3]),
)

print(res["residuals"])                # 혁신(innovation) v_t
print(res["standardized_residuals"])   # v_t / sqrt(F_t * sigma2)
```

## 아키텍처

```
Python (NumPy)
    │
    ▼  PyO3 바인딩
┌──────────────────────────────────────────────┐
│  lib.rs  (Python ↔ Rust 진입점)              │
│    sarimax_loglike / sarimax_fit /            │
│    sarimax_forecast / sarimax_residuals       │
└──────┬───────────┬──────────┬────────────────┘
       │           │          │
       ▼           ▼          ▼
   optimizer    kalman     forecast
   (L-BFGS,    (필터,     (h-step,
    NM)         우도)      잔차)
       │           │          │
       ▼           ▼          ▼
   ┌───────────────────────────────┐
   │  state_space (Harvey 표현)     │
   │  ┌─────────────────────────┐  │
   │  │ T (전이), Z (관측),     │  │
   │  │ R (선택), Q (공분산)    │  │
   │  └─────────────────────────┘  │
   └──────┬────────────────────────┘
          │
    ┌─────┴──────┐
    ▼            ▼
polynomial    params
(AR/MA 다항식  (변환, 제약,
 전개)         Monahan/Jones)
```

## 핵심 알고리즘 상세

### 1. 상태공간 구성 (`state_space.rs`)

SARIMA(p,d,q)(P,D,Q,s) 모델을 Harvey(1989) 표현으로 변환한다.

**상태 방정식:**
```
alpha_{t+1} = T * alpha_t + c_t + R * eta_t,    eta_t ~ N(0, Q)
y_t         = Z' * alpha_t + d_t + eps_t,        eps_t ~ N(0, H), H=0
```

상태 벡터 `alpha`의 크기는 `k_states = k_states_diff + k_order`이며:
- `k_states_diff = d + s*D` (차분 상태)
- `k_order = max(p + s*P, q + s*Q + 1)` (ARMA 동반행렬 차원)

**전이행렬 T** (`k_states x k_states`)는 5개 블록으로 구성된다:

```
T = ┌──────────┬────────────────┬──────────┐
    │ 차분블록  │  교차결합       │ 0        │
    │ (d x d)  │  (d → ARMA)    │          │
    ├──────────┼────────────────┤          │
    │ 0        │ 계절순환블록    │ → ARMA   │
    │          │ (s*D x s*D)    │          │
    ├──────────┼────────────────┼──────────┤
    │ 0        │ 0              │ ARMA     │
    │          │                │ 동반행렬  │
    └──────────┴────────────────┴──────────┘
```

- **차분 블록**: 상삼각 1 행렬 (누적 차분기)
- **계절 순환 블록**: D개의 s x s 순환이동 행렬
- **교차 결합**: 정규 차분 → 계절 마지막 상태, 차분 → ARMA 첫 상태
- **ARMA 동반행렬**: `reduced_ar = polymul(AR_poly, seasonal_AR_poly)`의 계수를 첫 열에, 초대각선에 1

**관측벡터 Z**: 차분 상태 + 계절 마지막 상태 + ARMA 첫 상태에 1

**선택행렬 R**: `reduced_ma = polymul(MA_poly, seasonal_MA_poly)`의 계수

예시 - SARIMA(1,1,1)(1,1,1,12): `k_states = 27`, `k_states_diff = 13`, `k_order = 14`

### 2. 칼만 필터 (`kalman.rs`)

표준 Harvey 형식의 칼만 필터로 로그우도를 계산한다.

```
각 시점 t = 0, ..., n-1에 대해:
  1. 혁신(innovation):  v_t = y_t - Z' * a_{t|t-1} - d_t
  2. 혁신 분산:          F_t = Z' * P_{t|t-1} * Z
  3. 칼만 이득:          K_t = P_{t|t-1} * Z / F_t
  4. 상태 갱신:          a_{t|t} = a_{t|t-1} + K_t * v_t
  5. 공분산 갱신 (Joseph): P_{t|t} = (I - K*Z') * P * (I - K*Z')'
  6. 예측:              a_{t+1|t} = T * a_{t|t} + c_t
  7. 공분산 예측:        P_{t+1|t} = T * P_{t|t} * T' + R*Q*R'
```

**집중 우도 (concentrate_scale=true):**
```
sigma2_hat = (1/n_eff) * SUM(v_t^2 / F_t)
loglike = -n_eff/2 * ln(2*pi) - n_eff/2 * ln(sigma2_hat)
          - n_eff/2 - 0.5 * SUM(ln(F_t))
```

- **초기화**: 근사 확산 초기화 `a_0 = 0, P_0 = kappa * I` (kappa = 1e6)
- **번인**: 처음 `k_states`개 관측치는 우도 누적에서 제외
- **수치 안정성**: Joseph 형식 공분산 갱신으로 양정치성 보장

### 3. 파라미터 변환 (`params.rs`)

최적화는 비제약 공간에서 수행하고, 평가 시 제약 공간으로 역변환한다.

**정상성 제약 (AR)** - Monahan(1984)/Jones(1980) 알고리즘:
```
비제약 → PACF:    r_k = x_k / sqrt(1 + x_k^2)
PACF → AR계수:    Levinson-Durbin 재귀
AR계수 → 제약:    constrained = -y[n-1][:]
```

모든 제약된 AR 계수는 정상성 영역 내에 있음을 보장한다. MA의 가역성 제약도 동일 알고리즘에 부호를 뒤집어 적용.

**분산 제약:**
```
제약 → 비제약:  sqrt(sigma2)    (sigma2 > 0 검증)
비제약 → 제약:  x^2
```

### 4. 최적화 (`optimizer.rs`)

음의 로그우도를 목적함수로 최소화한다.

```
목적함수:  f(theta) = -loglike(transform(theta))
그래디언트: 중심차분 (eps = 1e-7)
```

**전략:**
1. **초기값**: CSS(조건부 제곱합) 기반 추정 또는 사용자 제공
2. **L-BFGS**: MoreThuente 선탐색, 허용오차 `grad=1e-8, cost=1e-12`
3. **Nelder-Mead 폴백**: L-BFGS 실패 시 자동 전환, 5% 스케일 심플렉스
4. **정보량 기준**: `AIC = -2*ll + 2*k`, `BIC = -2*ll + k*ln(n)`

### 5. 예측 (`forecast.rs`)

칼만 필터의 최종 상태에서 h-step ahead 예측을 수행한다.

```
각 예측 스텝 h = 1, ..., steps에 대해:
  y_hat_h = Z' * a_h                    (예측 평균)
  F_h     = Z' * P_h * Z * sigma2       (예측 분산)
  CI_h    = y_hat_h +/- z_{alpha/2} * sqrt(F_h)
  a_{h+1} = T * a_h                     (상태 전파)
  P_{h+1} = T * P_h * T' + R*Q*R'       (공분산 전파)
```

예측 분산은 스텝이 증가할수록 단조 증가하며, 신뢰구간은 대칭이다.

## 수치 검증

statsmodels의 SARIMAX 결과를 기준(ground truth)으로 사용하여 검증한다.

| 모델 | 로그우도 오차 | 파라미터 오차 | 스케일 오차 |
|------|:----------:|:-----------:|:----------:|
| AR(1) | < 1e-6 | < 1e-4 | < 1e-6 |
| ARMA(1,1) | < 1e-6 | < 1e-3 | < 1e-6 |
| ARIMA(1,1,1) | < 1e-6 | < 1e-2 | < 1e-6 |
| SARIMA(1,0,0)(1,0,0,4) | < 1e-6 | < 1e-3 | < 1e-6 |
| SARIMA(1,1,1)(1,1,1,12) | < 1e-6 | < 1e-2 | < 1e-6 |

## 프로젝트 구조

```
sarimax_rs/
├── Cargo.toml                  # Rust 의존성 및 빌드 설정
├── pyproject.toml              # Python 패키지 설정 (maturin)
├── src/
│   ├── lib.rs                  # PyO3 모듈 진입점 (Python API 5개 함수)
│   ├── types.rs                # SarimaxOrder, SarimaxConfig, Trend, FitResult
│   ├── error.rs                # SarimaxError (thiserror 기반)
│   ├── params.rs               # 파라미터 구조체 + Monahan/Jones 변환
│   ├── polynomial.rs           # AR/MA 다항식 전개 (polymul, reduced_ar/ma)
│   ├── state_space.rs          # Harvey 상태공간 T, Z, R, Q 구성
│   ├── initialization.rs       # 근사 확산 초기화 (a_0=0, P_0=kappa*I)
│   ├── kalman.rs               # 칼만 필터 (loglike + full filter)
│   ├── start_params.rs         # CSS 기반 초기 파라미터 추정
│   ├── optimizer.rs            # L-BFGS + Nelder-Mead MLE 최적화
│   └── forecast.rs             # h-step 예측 + 잔차 진단
├── tests/fixtures/             # statsmodels 참조 데이터 (JSON)
├── python_tests/               # Python 통합 테스트
└── benches/                    # Criterion 벤치마크
```

## 의존성

| 크레이트 | 버전 | 용도 |
|---------|------|------|
| nalgebra | 0.34 | 동적 크기 행렬/벡터 연산 |
| argmin | 0.11 | L-BFGS, Nelder-Mead 최적화 |
| pyo3 | 0.28 | Python C-API 바인딩 |
| numpy | 0.28 | NumPy 배열 제로카피 전달 |
| thiserror | 2 | 에러 타입 매크로 |
| rayon | 1.10 | 데이터 병렬처리 (배치용) |

## 개발

```bash
# Rust 단위 테스트 (80+ 테스트)
cargo test

# Python 통합 테스트
uv run maturin develop && pytest python_tests/ -v

# 벤치마크
cargo bench
```

## 제한사항

- 계절 차분 `D > 1`은 미지원 (`D = 0` 또는 `1`만 가능)
- 외생변수(`exog`)는 미구현 (전달 시 `NotImplementedError`)
- Trend 파라미터는 Rust 내부에서만 지원, Python API 미노출
- 배치 처리(다중 시계열 동시 fitting)는 미구현

## 라이선스

MIT
