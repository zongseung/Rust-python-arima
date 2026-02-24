# Ver5.2 Spec: Simple Differencing — 대형 Seasonal Period 속도 최적화

> 작성일: 2026-02-24
> 상태: 설계 문서 (구현 예정)

---

## 1. 목표

`simple_differencing=True` 모드를 구현하여, 대형 seasonal period(s ≥ 12)에서 Kalman filter
상태공간 차원을 `k_order`만으로 축소한다. 현재 `SARIMA(1,1,1)(1,1,1,24)`의 `k_states=51`을
`k_states=26`으로 절반 감소시켜 칼만필터 비용(O(n × k³))을 약 **8배** 절감한다.

---

## 2. 배경

### 2.1 현재 상태

| 항목 | 상태 |
|------|------|
| `SarimaxConfig.simple_differencing` | 필드 존재, 기본값 `false` |
| `css.rs::apply_differencing()` | ✅ 구현 완료 — 재사용 가능 |
| `state_space.rs:52-56` | ❌ `simple_differencing=true` 시 즉시 Error 반환 |
| `lib.rs:342` | ❌ `build_config()`가 항상 `simple_differencing: false` 하드코딩 |
| `forecast.rs` | ❌ 역차분(undifferencing) 로직 없음 |
| `pipeline.rs` | ❌ 사전차분 분기 없음 |

### 2.2 statsmodels / R 참조 구현

| 항목 | statsmodels `sd=False` | statsmodels `sd=True` | R `stats::arima` |
|------|------------------------|------------------------|-----------------|
| 차분 위치 | T 행렬 내 (Harvey표현) | 데이터 사전차분 | T 행렬 내 (Harvey표현) |
| k_states | `k_order + d + s·D` | `k_order` only | `k_order + d + s·D` |
| n_obs (AIC/BIC) | 원본 N | N - d - s·D | N - d - s·D (`n.used`) |
| 예측 스케일 | 원본 스케일 (자동) | 차분 스케일 → 수동 역차분 | 원본 스케일 (자동) |
| BIC 분모 | log(N) | log(N - d - s·D) | log(n.used) |
| 대형 s 권장 | ❌ 비권장 | ✅ 권장 | 상태공간 (kappa 근사) |

**결정**: statsmodels `sd=True` + R n.used 방식을 따른다.

### 2.3 상태공간 차원 절감 효과

```
SARIMA(p,d,q)(P,D,Q,s):
  sd=false: k_states = k_order + d + s·D
  sd=true:  k_states = k_order = max(p + s·P, q + s·Q + 1)

         sd=false  sd=true   감소율   KF 비용비(k³)
(1,1,1)(1,1,1,24):  51  →  26    ×0.51   0.13 (7.7배)
(2,1,2)(1,1,1,24):  52  →  27    ×0.52   0.14 (7.2배)
(1,1,1)(1,1,1,12):  27  →  14    ×0.52   0.14 (7.2배)
```

---

## 3. 설계 결정

### 3.1 사용자 인터페이스

```python
# 명시적 플래그 (기본: False)
sarimax_rs.sarimax_fit(y, (1,1,1), (1,1,1,24), simple_differencing=True)

# s 크기와 무관하게 사용자가 직접 지정
# 자동 전환 (s >= 12) 은 이 스펙에서 구현하지 않음 (추후 검토)
```

**근거**: 자동 전환은 AIC/BIC 비교 기준이 달라져 혼란을 야기할 수 있음.
명시적 플래그로 사용자가 의도적으로 선택하도록 한다.

### 3.2 AIC/BIC 계산

```
n_eff = n_obs - d - s·D      (sd=false에서는 n_eff = n_obs)

AIC  = -2·llf + 2·k           (n_eff 무관, 동일)
BIC  = -2·llf + k·ln(n_eff)   ← sd=true 시 n_eff 사용
HQIC = -2·llf + 2·k·ln(ln(n_eff))
```

**주의**: sd=False와 sd=True의 BIC/HQIC는 서로 직접 비교 불가.
AIC는 비교 가능 (n_eff 무관).

### 3.3 예측 스케일

`forecast_pipeline()`은 항상 **원본 스케일 예측값** 반환.
내부에서 역차분(undifferencing)을 자동 수행한다.

```
sd=true 예측 파이프라인:
  1. endog_diff = apply_differencing(endog_orig)
  2. Kalman filter on endog_diff → predicted state a_{n+1|n}
  3. Forecast w_{n+h} (차분 스케일)
  4. undifference_forecast(w, endog_orig, d, dd, s) → y_{n+h} (원본 스케일)
  5. 반환: 원본 스케일 예측값 + CI
```

### 3.4 잔차(residuals) 스케일

잔차는 **차분 스케일**로 반환 (KF output 그대로).
원본 스케일 잔차가 필요하면 사용자가 직접 역차분해야 함.
(statsmodels sd=True와 동일 동작)

---

## 4. 역차분(Undifferencing) 알고리즘

SARIMA에서 `apply_differencing`은 "계절 차분 먼저, 비계절 차분 나중" 순서를 따른다.
역차분은 반대 순서로: **비계절 역차분 먼저, 계절 역차분 나중**.

### 4.1 케이스별 알고리즘

#### Case 1: d=1, D=0 (비계절 1차 차분만)

```
w[h] = Δy[n+h] = y[n+h] - y[n+h-1]
→ y[n+h] = w[h] + y[n+h-1]

buffer: y_last = y[n]  (원본 마지막 관측치 1개)

알고리즘:
  prev = y_last
  for h in 0..steps:
    y[h] = w[h] + prev
    prev = y[h]
```

#### Case 2: d=0, D=1, s=s (계절 차분만)

```
w[h] = Δ_s y[n+h] = y[n+h] - y[n+h-s]
→ y[n+h] = w[h] + y[n+h-s]

buffer: y_seasonal = y[n-s+1..n]  (마지막 s개 원본 관측치)

알고리즘:
  buf = y_seasonal.to_vec()  // 길이 s 원형 버퍼
  for h in 0..steps:
    idx = h % s
    y[h] = w[h] + buf[idx]
    buf[idx] = y[h]
```

#### Case 3: d=1, D=1, s=s (비계절 + 계절 차분)

```
차분 적용 순서: Δ_s 먼저, Δ 나중
  z = Δ_s y          (계절 차분만 적용한 시계열)
  w = Δz = Δ(Δ_s y)  (KF에 들어간 최종 차분)

역차분 순서: Δ 먼저 undo, Δ_s 나중 undo

Step 1. undo Δ (비계절 역차분):
  z[n+h] = w[h] + z[n+h-1]
  buffer: z_last = z[n] = y[n] - y[n-s]  (1개)
  → z 예측값 series를 얻음

Step 2. undo Δ_s (계절 역차분):
  y[n+h] = z[n+h] + y[n+h-s]
  buffer: y_seasonal = y[n-s+1..n]  (s개)
  → y 예측값 series를 얻음 (원본 스케일)
```

#### Case 4: d=2, D=0 (비계절 2차 차분)

```
w = Δ²y  →  먼저 1차 역차분, 다시 1차 역차분

buffer: Δy[n] = y[n] - y[n-1]  (1개)
        y[n]                    (1개)

Step 1. undo 1st Δ: z[h] = w[h] + z[h-1], z[-1] = Δy[n]
Step 2. undo 2nd Δ: y[h] = z[h] + y[h-1], y[-1] = y[n]
```

### 4.2 Rust 구현 구조

```rust
/// 예측값 역차분: 차분 스케일 w[] → 원본 스케일 y[n+1..n+steps]
///
/// endog_orig: 원본 시계열 전체 (버퍼 추출용)
/// w_forecast: 차분 스케일 예측값 (steps개)
/// d, dd, s: 차분 차수
pub fn undifference_forecast(
    w_forecast: &[f64],
    endog_orig: &[f64],
    d: usize,
    dd: usize,
    s: usize,
) -> Vec<f64>
```

---

## 5. 파일별 변경 계획

### 5.1 `src/state_space.rs`

**변경**: `simple_differencing=true` 시 Error 제거 + ARMA-only 상태공간 구성

```rust
// 현재 (제거)
if config.simple_differencing {
    return Err(...);
}

// 변경: sd 오버라이드
let sd = if config.simple_differencing { 0 } else { order.k_states_diff() };
let k_states = if config.simple_differencing { order.k_order() } else { order.k_states() };
```

- `build_transition()`: diff 블록 건너뜀 (`sd=0`으로 처리)
- `build_design()`: Z[0] = 1.0만 (diff 상태 없음)
- `build_selection()`: R[0,0] = 1.0부터 MA 계수 시작
- `build_state_intercept()`: `inject_idx = 0` (ARMA 첫 상태에 trend 주입)
- `update_params()`: `sd=0` 기준으로 인덱싱

**영향 범위**: 신규 파라미터 없음, `config.simple_differencing` 읽기만.

### 5.2 `src/forecast.rs`

**변경 1**: `undifference_forecast()` 함수 신규 추가

**변경 2**: `forecast_pipeline()` — 사전차분 + 역차분 분기

```rust
pub fn forecast_pipeline(...) -> Result<ForecastResult> {
    if config.simple_differencing {
        // 1. 사전차분
        let endog_diff = css::apply_differencing(endog, config);
        // 2. SS + KF (차분된 데이터로)
        let ss = StateSpace::new(config, params, &endog_diff, exog)?;
        let init = KalmanInit::from_config_default(&ss, config);
        let fo = kalman_filter(&endog_diff, &ss, &init, config.concentrate_scale)?;
        // 3. 차분 스케일 예측
        let fc_diff = forecast(&ss, &fo, steps, alpha, future_exog, ...)?;
        // 4. 역차분 → 원본 스케일
        let mean_orig = undifference_forecast(
            &fc_diff.mean, endog, config.order.d, config.order.dd, config.order.s
        );
        // 5. CI도 역차분 (대칭 CI를 mean 이동으로 복원)
        let ci_lower = undifference_ci(&fc_diff, &mean_orig, true);
        let ci_upper = undifference_ci(&fc_diff, &mean_orig, false);
        return Ok(ForecastResult { mean: mean_orig, ci_lower, ci_upper, ... });
    }
    // 기존 코드 (변경 없음)
    ...
}
```

**CI 역차분 방식**:
- CI 반폭(half-width) = `(upper - lower) / 2` (차분 스케일에서 계산)
- 원본 스케일 CI = `mean_orig ± half_width` (level만 이동, 분산 동일)

### 5.3 `src/pipeline.rs`

**변경**: 4개 helper 함수에 사전차분 분기 추가

```rust
pub(crate) fn kalman_eval(endog: &[f64], ...) -> Result<KalmanOutput> {
    let eff_endog: Cow<[f64]> = if config.simple_differencing {
        Cow::Owned(css::apply_differencing(endog, config))
    } else {
        Cow::Borrowed(endog)
    };
    let ss = StateSpace::new(config, params, &eff_endog, exog)?;
    let init = KalmanInit::from_config_default(&ss, config);
    kalman::kalman_loglike(&eff_endog, &ss, &init, config.concentrate_scale)
}
// kalman_eval_constrained, kalman_eval_unconstrained,
// kalman_filter_full, kalman_filter_constrained 동일 패턴
```

### 5.4 `src/optimizer.rs`

**변경**: `fit()` 진입점에서 사전차분

```rust
pub fn fit(endog: &[f64], config: &SarimaxConfig, ...) -> Result<FitResult> {
    let eff_endog: Vec<f64> = if config.simple_differencing {
        css::apply_differencing(endog, config)
    } else {
        endog.to_vec()
    };
    // 이후 eff_endog 사용 (start_params, CSS, MLE 모두)
    ...
    // n_obs 보정 (BIC용)
    let n_eff = eff_endog.len();  // 이미 차분으로 줄어든 길이
    ...
}
```

### 5.5 `src/types.rs`

**변경**: `FitResult.with_information_criteria()` — `n_eff` 지원

```rust
impl FitResult {
    pub fn with_information_criteria(mut self) -> Self {
        let k = self.n_params as f64;
        let n = self.n_obs as f64;  // 이미 n_eff (optimizer가 eff_endog.len() 사용)
        self.aic = -2.0 * self.loglike + 2.0 * k;
        self.bic = -2.0 * self.loglike + k * n.ln();
        self
    }
}
```

`n_obs`를 `eff_endog.len()`으로 설정하면 AIC/BIC 자동 보정됨.

### 5.6 `src/lib.rs`

**변경**: Python 인터페이스에 `simple_differencing` 파라미터 추가

```rust
// sarimax_fit, sarimax_forecast, sarimax_residuals, sarimax_loglike,
// sarimax_batch_fit, sarimax_batch_forecast, sarimax_batch_loglike,
// sarimax_inference, sarimax_diagnostics 모두 동일하게:

#[pyo3(signature = (..., simple_differencing=false))]
fn sarimax_fit(..., simple_differencing: bool, ...) {
    ...
    let config = build_config_v2(order, seasonal, n_exog, ..., simple_differencing);
}

// build_config()에 simple_differencing 인자 추가
fn build_config_v2(..., simple_differencing: bool) -> SarimaxConfig {
    SarimaxConfig {
        ...
        simple_differencing,
        ...
    }
}
```

### 5.7 `src/start_params.rs`

**변경**: `simple_differencing=true`일 때 이미 차분된 `endog`가 들어오므로 별도 처리 불필요.
`compute_start_params(endog_diff, config, ...)` — CSS/HR은 동일 동작.

**확인 필요**: `start_params.rs`가 내부적으로 차분을 또 적용하는지 점검 → 이중 차분 방지.

### 5.8 Python Layer (`python/sarimax_py/model.py`)

**변경**: `SARIMAXModel.__init__()` — `simple_differencing` 파라미터 전달

```python
class SARIMAXModel:
    def __init__(self, endog, order, seasonal_order=(0,0,0,0),
                 simple_differencing=False, ...):
        self.simple_differencing = simple_differencing

    def fit(self, ...):
        result = sarimax_rs.sarimax_fit(
            ...,
            simple_differencing=self.simple_differencing
        )
```

---

## 6. 변경 파일 목록

| 파일 | 변경 유형 | 예상 변경 규모 |
|------|-----------|---------------|
| `src/state_space.rs` | 수정 | ~50줄 |
| `src/forecast.rs` | 수정 (신규 함수 포함) | ~100줄 |
| `src/pipeline.rs` | 수정 | ~30줄 |
| `src/optimizer.rs` | 수정 | ~15줄 |
| `src/types.rs` | 변경 없음 (n_obs 이미 적절히 설정됨) | — |
| `src/lib.rs` | 수정 | ~50줄 |
| `src/start_params.rs` | 확인 후 수정 (이중차분 방지) | ~5줄 |
| `python/sarimax_py/model.py` | 수정 | ~10줄 |
| `python_tests/test_simple_diff.py` | 신규 | ~150줄 |

---

## 7. 테스트 계획

### 7.1 Rust 유닛 테스트 (`state_space.rs`)

```rust
// SARIMA(1,1,1)(1,1,1,24) sd=true: k_states = k_order = 26
#[test]
fn test_simple_diff_k_states() {
    let config_sd = SarimaxConfig { simple_differencing: true, ... };
    let ss = StateSpace::new(&config_sd, ...);
    assert_eq!(ss.k_states, 26);      // k_order only
    assert_eq!(ss.k_states_diff, 0);  // no diff states
}

// ARMA-only T matrix (no diff block)
#[test]
fn test_simple_diff_transition_shape()

// Design Z[0] = 1.0 only
#[test]
fn test_simple_diff_design()
```

### 7.2 Rust 유닛 테스트 (`forecast.rs`)

```rust
// d=1 역차분: w=[1,1,1], y_last=10 → [11,12,13]
#[test]
fn test_undifference_d1()

// D=1, s=4 역차분
#[test]
fn test_undifference_dd1_s4()

// d=1, D=1, s=4 역차분
#[test]
fn test_undifference_d1_dd1_s4()
```

### 7.3 Python 통합 테스트 (`test_simple_diff.py`)

```python
# 1. sd=True의 fit 결과가 sd=False와 동등한 loglike 달성
def test_loglike_equivalent_sd_modes():
    r_sd_false = sarimax_rs.sarimax_fit(y, order, seasonal)
    r_sd_true  = sarimax_rs.sarimax_fit(y, order, seasonal, simple_differencing=True)
    assert abs(r_sd_false["loglike"] - r_sd_true["loglike"]) < 1.0

# 2. BIC는 n_eff 기준으로 줄어들어야 함
def test_aic_bic_n_eff():
    r = sarimax_rs.sarimax_fit(y, (1,1,1), (1,1,1,24), simple_differencing=True)
    n_eff = len(y) - 1 - 24  # d=1, D=1, s=24
    k = 4  # ar, ma, sar, sma
    expected_aic = -2 * r["loglike"] + 2 * k
    assert abs(r["aic"] - expected_aic) < 1e-6
    expected_bic = -2 * r["loglike"] + k * math.log(n_eff)
    assert abs(r["bic"] - expected_bic) < 1e-6

# 3. 예측값이 원본 스케일로 돌아옴 (finite & 합리적 범위)
def test_forecast_original_scale():
    r_fit = sarimax_rs.sarimax_fit(y, order, seasonal, simple_differencing=True)
    r_fc  = sarimax_rs.sarimax_forecast(y, order, seasonal, r_fit["params"],
                                         steps=24, simple_differencing=True)
    assert all(np.isfinite(r_fc["mean"]))
    assert np.mean(y) * 0.1 < np.mean(r_fc["mean"]) < np.mean(y) * 10

# 4. sd=True가 sd=False보다 빠름
def test_speedup_simple_diff():
    import time
    t0 = time.perf_counter()
    sarimax_rs.sarimax_fit(y, (1,1,1), (1,1,1,24), simple_differencing=False)
    t_sd_false = time.perf_counter() - t0
    t0 = time.perf_counter()
    sarimax_rs.sarimax_fit(y, (1,1,1), (1,1,1,24), simple_differencing=True)
    t_sd_true = time.perf_counter() - t0
    assert t_sd_true < t_sd_false  # sd=True가 더 빠름

# 5. statsmodels sd=True와 loglike 비교
def test_vs_statsmodels_sd_true():
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    sm = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12),
                 simple_differencing=True).fit(disp=False)
    rs = sarimax_rs.sarimax_fit(y, (1,1,1), (1,1,1,12), simple_differencing=True)
    assert abs(rs["loglike"] - sm.llf) < 3.0
```

---

## 8. 검증 기준

### 8.1 정확도

| 지표 | 목표 |
|------|------|
| loglike gap vs sd=False | < 2.0 |
| loglike gap vs statsmodels sd=True | < 3.0 |
| forecast mean 오차 (n=720, 48h) | < 5% |

### 8.2 속도

| 케이스 | sd=False (현재) | sd=True (목표) | 기대 속도향상 |
|--------|----------------|----------------|--------------|
| SARIMA(1,1,1)(1,1,1,24) n=336 | ~350ms | **< 100ms** | >3x |
| SARIMA(1,1,1)(1,1,1,24) n=720 | ~900ms | **< 250ms** | >3x |
| SARIMA(1,1,1)(1,1,1,24) n=2160 | ~2900ms | **< 700ms** | >4x |

*이론상 k³ 감소비: 51³/26³ ≈ 7.5배, optimizer 반복 감소까지 포함하면 3-5배 실측 예상*

### 8.3 AIC/BIC 기준값 (n_eff 적용 검증)

```python
# SARIMA(1,1,1)(1,1,1,24), n=720, d=1, D=1
# n_eff = 720 - 1 - 24 = 695
# k = 4 (ar, ma, sar, sma, σ² 제외: concentrate_scale=True)
assert result["n_obs"] == 695
assert abs(result["bic"] - (-2 * result["loglike"] + 4 * log(695))) < 0.01
```

---

## 9. 구현 순서

```
Step 1. state_space.rs — simple_differencing=true 상태공간 구성     [~2h]
  └── 유닛 테스트: k_states, T, Z, R 행렬 검증

Step 2. pipeline.rs — 사전차분 분기 추가                            [~1h]
  └── kalman_eval, kalman_filter_full 등 4개 함수

Step 3. optimizer.rs — fit() 사전차분 + n_obs 보정                  [~1h]
  └── Rust cargo test (기존 147개 테스트 통과 확인)

Step 4. forecast.rs — undifference_forecast() + forecast_pipeline() [~3h]
  └── 유닛 테스트: d=1, D=1, d=1+D=1 역차분 검증

Step 5. lib.rs — Python 인터페이스 simple_differencing 파라미터    [~2h]
  └── 모든 9개 함수에 추가

Step 6. Python tests — test_simple_diff.py                          [~2h]
  └── 5개 통합 테스트 통과

Step 7. model.py — Python 레이어 업데이트                           [~30min]
  └── SARIMAXModel, SARIMAXResult 파라미터 전달
```

---

## 10. 미결 사항 (구현 시 검토)

1. **start_params.rs 이중차분 방지**: `compute_start_params()`가 내부에서 `apply_differencing`을
   다시 호출하는지 확인 필요. 호출 시 `simple_differencing=true`면 스킵해야 함.

2. **CI 역차분 분산 전파**: 현재 설계는 CI 반폭을 차분 스케일에서 그대로 이동.
   엄밀하게는 역차분 과정에서 누적 오차가 있으나, 장기 예측에서는 상태공간
   자체가 분산을 전파하므로 이 근사가 적절함.

3. **exog와의 조합**: `obs_intercept`는 차분 후 시계열에도 동일하게 적용.
   단, exog도 동일하게 차분해야 하는지는 statsmodels 동작 확인 필요.
   → 이 스펙에서는 exog 없는 케이스만 우선 지원.

4. **trend와의 조합**: `trend='c'`일 때 상수항이 차분 시계열에 어떻게 작용하는지.
   → `trend='n'`인 경우만 우선 지원.

---

## 11. 참고 문헌

1. statsmodels SARIMAX 소스: `tsa/statespace/sarimax.py` — `simple_differencing` 구현 참조
2. R `stats::arima()` 소스: `makeARIMA()` 함수 — Delta 다항식 구성 방식
3. Harvey, A.C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter.*
   Cambridge University Press. — Harvey 표현 원본
4. Durbin, J. & Koopman, S.J. (2012). *Time Series Analysis by State Space Methods* (2nd ed.)
   Oxford University Press. — 확산 초기화 이론
