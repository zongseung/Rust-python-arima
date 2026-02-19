# Rust Implementation Plan (ver1.1)

## SARIMAX Numerical Engine — statsmodels 정밀 재현 + AIC 네이티브 구현

---

# 0. ver1 → ver1.1 주요 변경점

| 항목 | ver1 | ver1.1 |
|------|------|--------|
| nalgebra | 0.33 | **0.34** (latest, verified) |
| argmin | 0.11 | **0.11** (API 검증 완료) |
| pyo3 | 0.22 | **0.23** (GIL Ref 제거 반영) |
| numpy (rust) | 0.22 | **0.23** (pyo3 0.23 매칭) |
| finitediff | 0.1 | **0.2** (central_diff API 변경) |
| 로그우도 | 일반 | **concentrated log-likelihood** (statsmodels 기본) |
| 파라미터 변환 | 미구현 | **Monahan(1984) 재매개화** |
| 초기 파라미터 | 언급만 | **CSS 2단계 OLS** 구현 |
| 파라미터 레이아웃 | 대략적 | **statsmodels 소스 기반 정확한 순서** |
| 상태공간 | 수식만 | **행렬별 구체 구성 알고리즘** |
| 에러 처리 | 없음 | **thiserror 기반 Result** |
| 빌드/패키징 | 미정 | **uv + maturin** |

---

# 1. 프로젝트 구조

```
sarimax_rs/
├── Cargo.toml
├── pyproject.toml              # maturin + uv 호환
├── uv.lock                     # Python 의존성 잠금
├── src/
│   ├── lib.rs                  # pyo3 모듈 등록
│   ├── error.rs                # thiserror 에러 (NEW)
│   ├── types.rs                # SarimaxOrder, Config, Trend (NEW)
│   ├── params.rs               # 팩/언팩 + Monahan 변환
│   ├── polynomial.rs           # 다항식 곱, AR/MA poly 생성 (NEW)
│   ├── state_space.rs          # Harvey Representation 구축
│   ├── kalman.rs               # 칼만필터 + concentrated loglike
│   ├── initialization.rs       # diffuse/stationary/mixed 초기화 (NEW)
│   ├── start_params.rs         # CSS 2단계 OLS 초기값 (NEW)
│   ├── optimizer.rs            # argmin L-BFGS + NelderMead 폴백
│   ├── information.rs          # AIC/BIC/AICc/HQIC
│   ├── forecast.rs             # h-step 예측
│   ├── selection.rs            # rayon grid/stepwise 모델 선택
│   ├── diagnostics.rs          # Ljung-Box, 잔차 분석 (NEW)
│   └── batch.rs                # rayon 배치 처리
├── tests/
│   ├── test_polynomial.rs
│   ├── test_params.rs
│   ├── test_state_space.rs
│   ├── test_kalman.rs
│   ├── test_optimizer.rs
│   ├── test_information.rs
│   └── fixtures/               # statsmodels 기준값 (NEW)
│       └── statsmodels_reference.json
├── benches/
│   ├── bench_kalman.rs
│   └── bench_fit.rs
└── python_tests/
    ├── conftest.py
    ├── test_loglike.py
    ├── test_fit.py
    ├── test_aic.py
    ├── test_vs_statsmodels.py
    └── generate_fixtures.py    # sm 기준값 생성 스크립트
```

---

# 2. Cargo.toml (모든 버전 검증 완료)

```toml
[package]
name = "sarimax-rs"
version = "0.1.0"
edition = "2021"
rust-version = "1.75"

[lib]
name = "sarimax_rs"
crate-type = ["cdylib", "rlib"]

[dependencies]
nalgebra = "0.34"
argmin = "0.11"
argmin-math = { version = "0.5", features = ["nalgebra_latest", "vec"] }
finitediff = "0.2"
statrs = "0.18"
rayon = "1.10"
pyo3 = { version = "0.23", features = ["extension-module"] }
numpy = "0.23"
thiserror = "2"
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
approx = "0.5"

[[bench]]
name = "bench_kalman"
harness = false
```

### 빌드 명령 (uv 통합)

```bash
# 환경 초기화
uv sync --extra dev

# Rust 개발 빌드 + Python 설치
uv run maturin develop

# Rust 단위 테스트
cargo test --all-targets

# Python 통합 테스트
uv run pytest python_tests -q

# 벤치마크
cargo bench
```

---

# 3. 핵심 모듈 상세 설계

## 3.1 error.rs

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SarimaxError {
    #[error("파라미터 길이 불일치: expected {expected}, got {got}")]
    ParamLengthMismatch { expected: usize, got: usize },
    #[error("상태공간 구성 실패: {0}")]
    StateSpaceError(String),
    #[error("Cholesky 분해 실패")]
    CholeskyFailed,
    #[error("최적화 실패: {0}")]
    OptimizationFailed(String),
    #[error("비정상 AR 다항식")]
    NonStationaryAR,
    #[error("비가역 MA 다항식")]
    NonInvertibleMA,
    #[error("데이터 오류: {0}")]
    DataError(String),
}
pub type Result<T> = std::result::Result<T, SarimaxError>;
```

---

## 3.2 types.rs

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SarimaxOrder {
    pub p: usize, pub d: usize, pub q: usize,
    pub pp: usize, pub dd: usize, pub qq: usize, pub s: usize,
}

impl SarimaxOrder {
    pub fn k_ar(&self) -> usize { self.p + self.s * self.pp }
    pub fn k_ma(&self) -> usize { self.q + self.s * self.qq }
    pub fn k_order(&self) -> usize { self.k_ar().max(self.k_ma() + 1) }
    pub fn k_states_diff(&self) -> usize { self.d + self.s * self.dd }
    pub fn k_states(&self) -> usize { self.k_order() + self.k_states_diff() }
}

#[derive(Debug, Clone)]
pub struct SarimaxConfig {
    pub order: SarimaxOrder,
    pub n_exog: usize,
    pub trend: Trend,
    pub enforce_stationarity: bool,   // 기본: false
    pub enforce_invertibility: bool,   // 기본: false
    pub concentrate_scale: bool,       // 기본: true
    pub simple_differencing: bool,     // 기본: false
    pub measurement_error: bool,       // 기본: false
}

#[derive(Debug, Clone, Copy)]
pub enum Trend {
    None,     // 'n': k=0
    Constant, // 'c': k=1
    Linear,   // 't': k=1
    Both,     // 'ct': k=2
}

impl Trend {
    pub fn k_trend(&self) -> usize {
        match self { Trend::None => 0, Trend::Constant | Trend::Linear => 1, Trend::Both => 2 }
    }
}
```

---

## 3.3 params.rs — statsmodels 정확 복제 + Monahan 변환

### 파라미터 레이아웃 (statsmodels 소스 확인)

```
flat = [trend(k_trend) | exog(k_exog) | ar(p) | ma(q) | sar(P) | sma(Q) | sigma2?]
```

- `concentrate_scale=true` → sigma2 제외 (최적화 대상에서 빠짐)
- `concentrate_scale=false` → sigma2 마지막에 포함
- AIC의 k 계산 시: sigma2는 concentrated여도 **항상 카운트**

```rust
#[derive(Debug, Clone)]
pub struct SarimaxParams {
    pub trend_coeffs: Vec<f64>,
    pub exog_coeffs: Vec<f64>,
    pub ar_coeffs: Vec<f64>,
    pub ma_coeffs: Vec<f64>,
    pub sar_coeffs: Vec<f64>,
    pub sma_coeffs: Vec<f64>,
    pub sigma2: Option<f64>,
}

impl SarimaxParams {
    pub fn from_flat(flat: &[f64], config: &SarimaxConfig) -> Result<Self> {
        let mut i = 0;
        let kt = config.trend.k_trend();
        let trend_coeffs = flat[i..i+kt].to_vec(); i += kt;
        let exog_coeffs = flat[i..i+config.n_exog].to_vec(); i += config.n_exog;
        let ar_coeffs = flat[i..i+config.order.p].to_vec(); i += config.order.p;
        let ma_coeffs = flat[i..i+config.order.q].to_vec(); i += config.order.q;
        let sar_coeffs = flat[i..i+config.order.pp].to_vec(); i += config.order.pp;
        let sma_coeffs = flat[i..i+config.order.qq].to_vec(); i += config.order.qq;
        let sigma2 = if !config.concentrate_scale {
            Some(*flat.get(i).ok_or(SarimaxError::ParamLengthMismatch {
                expected: i+1, got: flat.len()
            })?)
        } else { None };
        Ok(Self { trend_coeffs, exog_coeffs, ar_coeffs, ma_coeffs,
                   sar_coeffs, sma_coeffs, sigma2 })
    }

    pub fn to_flat(&self) -> Vec<f64> {
        let mut v = Vec::new();
        v.extend(&self.trend_coeffs); v.extend(&self.exog_coeffs);
        v.extend(&self.ar_coeffs); v.extend(&self.ma_coeffs);
        v.extend(&self.sar_coeffs); v.extend(&self.sma_coeffs);
        if let Some(s) = self.sigma2 { v.push(s); }
        v
    }

    /// AIC의 k: concentrated여도 sigma2 포함
    pub fn n_estimated_params(&self, config: &SarimaxConfig) -> usize {
        config.trend.k_trend() + config.n_exog
            + config.order.p + config.order.q
            + config.order.pp + config.order.qq + 1
    }
}
```

### Monahan(1984) / Jones(1980) 변환

statsmodels는 최적화 시 **비제약 ↔ 제약** 변환을 사용한다:

```rust
/// 비제약 → 정상성 AR 계수 (Monahan 1984)
/// Step 1: r[k] = x[k] / sqrt(1 + x[k]^2) → PACF로 매핑
/// Step 2: Levinson-Durbin 재귀 → AR 계수
/// Step 3: constrained = -y[n-1][:]
pub fn constrain_stationary(unconstrained: &[f64]) -> Vec<f64> {
    let n = unconstrained.len();
    if n == 0 { return vec![]; }
    let pacf: Vec<f64> = unconstrained.iter()
        .map(|&x| x / (1.0 + x * x).sqrt()).collect();
    let mut y = vec![vec![0.0; n]; n];
    for k in 0..n {
        for i in 0..k { y[k][i] = y[k-1][i] + pacf[k] * y[k-1][k-i-1]; }
        y[k][k] = pacf[k];
    }
    y[n-1].iter().map(|&v| -v).collect()
}

/// 역변환: 제약 → 비제약
pub fn unconstrain_stationary(constrained: &[f64]) -> Vec<f64> {
    let n = constrained.len();
    if n == 0 { return vec![]; }
    let mut y = vec![vec![0.0; n]; n];
    for i in 0..n { y[n-1][i] = -constrained[i]; }
    for k in (1..n).rev() {
        for i in 0..k {
            y[k-1][i] = (y[k][i] - y[k][k]*y[k][k-i-1]) / (1.0 - y[k][k]*y[k][k]);
        }
    }
    (0..n).map(|k| { let r = y[k][k]; r / (1.0 - r*r).max(1e-15).sqrt() }).collect()
}

/// MA 가역성: 부호 반전 래핑
pub fn constrain_invertible(u: &[f64]) -> Vec<f64> {
    constrain_stationary(u).iter().map(|&x| -x).collect()
}
pub fn unconstrain_invertible(c: &[f64]) -> Vec<f64> {
    unconstrain_stationary(&c.iter().map(|&x| -x).collect::<Vec<_>>())
}

/// sigma2 변환
pub fn constrain_variance(x: f64) -> f64 { x * x }
pub fn unconstrain_variance(s: f64) -> f64 { s.sqrt() }
```

---

## 3.4 polynomial.rs — 다항식 연산 (AR/MA 곱)

```rust
/// 다항식 곱 (convolution): c[k] = sum_i a[i]*b[k-i]
pub fn polymul(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() { return vec![]; }
    let mut r = vec![0.0; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() { r[i+j] += ai * bj; }
    }
    r
}

/// AR poly: 1 - phi_1*L - ... (statsmodels: 계수에 -부호)
pub fn make_ar_poly(coeffs: &[f64], max_lag: usize) -> Vec<f64> {
    let mut p = vec![0.0; max_lag + 1]; p[0] = 1.0;
    for (i, &c) in coeffs.iter().enumerate() { if i+1 <= max_lag { p[i+1] = -c; } }
    p
}

/// 계절 AR poly: 1 - Phi_1*L^s - ...
pub fn make_seasonal_ar_poly(coeffs: &[f64], s: usize) -> Vec<f64> {
    if coeffs.is_empty() { return vec![1.0]; }
    let mut p = vec![0.0; coeffs.len() * s + 1]; p[0] = 1.0;
    for (i, &c) in coeffs.iter().enumerate() { p[(i+1)*s] = -c; }
    p
}

/// MA poly: 1 + theta_1*L + ... (부호 반전 없음)
pub fn make_ma_poly(coeffs: &[f64], max_lag: usize) -> Vec<f64> {
    let mut p = vec![0.0; max_lag + 1]; p[0] = 1.0;
    for (i, &c) in coeffs.iter().enumerate() { if i+1 <= max_lag { p[i+1] = c; } }
    p
}

/// 계절 MA poly: 1 + Theta_1*L^s + ...
pub fn make_seasonal_ma_poly(coeffs: &[f64], s: usize) -> Vec<f64> {
    if coeffs.is_empty() { return vec![1.0]; }
    let mut p = vec![0.0; coeffs.len() * s + 1]; p[0] = 1.0;
    for (i, &c) in coeffs.iter().enumerate() { p[(i+1)*s] = c; }
    p
}

/// 축약 AR/MA = polymul(비계절, 계절)
pub fn reduced_ar(params: &SarimaxParams, order: &SarimaxOrder) -> Vec<f64> {
    polymul(&make_ar_poly(&params.ar_coeffs, order.p),
            &make_seasonal_ar_poly(&params.sar_coeffs, order.s))
}
pub fn reduced_ma(params: &SarimaxParams, order: &SarimaxOrder) -> Vec<f64> {
    polymul(&make_ma_poly(&params.ma_coeffs, order.q),
            &make_seasonal_ma_poly(&params.sma_coeffs, order.s))
}
```

---

## 3.5 state_space.rs — Harvey Representation

### 상태 차원 공식 (statsmodels 소스 확인)

```
k_order = max(p + s*P, q + s*Q + 1)
k_states_diff = d + s*D              (simple_differencing=false일 때)
k_states = k_order + k_states_diff
k_posdef = 1                          (k_order > 0일 때)
```

### 예시: SARIMA(1,1,1)(1,1,1,12)
- k_ar = 1 + 12 = 13, k_ma = 1 + 12 = 13
- k_order = max(13, 14) = 14
- k_states_diff = 1 + 12 = 13
- k_states = 27

### 전이 행렬 T 구성 규칙

| 블록 | 위치 | 내용 |
|------|------|------|
| 일반 차분 | [0..d, 0..d] | 상삼각 1행렬 |
| 계절 차분 | [d..d+s*D, d..d+s*D] | s×s 순환 동반 (shift+wrap) |
| 차분→ARMA | [0..d, k_states_diff] = 1 | 연결 |
| 계절→ARMA | [d+layer*s, k_states_diff] = 1 | 연결 |
| ARMA 동반 | [sd..sd+k, sd..sd+k] | 첫 열=-reduced_ar[1:], 초대각=1 |

### 관측 벡터 Z 구성 규칙
- Z[i] = 1 for i in 0..d (일반 차분)
- Z[d + layer*s] = 1 (계절 차분)
- Z[k_states_diff] = 1 (ARMA 첫 상태)

### 선택 행렬 R 구성 규칙
- R[k_states_diff, 0] = 1 (충격 진입)
- R[k_states_diff + i, 0] = reduced_ma[i] for i >= 1

### 상태 공분산 Q
- concentrate_scale=true → Q = [[1.0]]
- concentrate_scale=false → Q = [[sigma2]]

### 절편
- state_intercept[t]: trend → c[k_states_diff] += trend_data[t] * trend_coeffs
- obs_intercept[t]: exog → d[t] += exog[t] * exog_coeffs

(구체 코드: ver1의 state_space.rs와 동일하되, 위 규칙을 정확히 따름)

---

## 3.6 initialization.rs — 초기 상태 3종

| 방법 | 사용 조건 | a_0 | P_0 | burn |
|------|----------|-----|-----|------|
| ApproximateDiffuse | enforce_stationarity=false | 0 | kappa*I (1e6) | k_states_diff |
| Mixed | enforce_stationarity=true | 0 | diff=1e6*I, ARMA=Lyapunov해 | k_states_diff |
| Stationary | 모든 상태 정상 | 0 | Lyapunov 전체해 | 0 |

### Lyapunov 풀이
이산 Lyapunov 방정식 T*P*T' + Q = P를 **쌍선형 변환**으로 Sylvester 방정식으로 전환:
```
b = (T' - I)(T' + I)^{-1}
c = 2(T + I)^{-1} * Q * (T' + I)^{-1}
b'X + Xb = -c   →   Schur 분해 + back-substitution
```

---

## 3.7 kalman.rs — Concentrated Log-Likelihood

### statsmodels 기본 동작 (concentrate_scale=True)

sigma2=1로 칼만 필터 실행 후 사후 계산:

```
sigma2_hat = (1/n_eff) * sum_{t>=burn}(v_t^2 / F_t)

loglike = -n_eff/2 * ln(2*pi)
          -n_eff/2 * ln(sigma2_hat)
          -n_eff/2
          -0.5 * sum_{t>=burn}(ln(F_t))
```

### 수치 안정성 기법
1. **Joseph form** 공분산 업데이트: P = (I-KZ')P(I-KZ')' + KHK'
2. **F_t ≤ 0 가드**: skip update, 로그우도에서 제외
3. **ln(scale) 가드**: scale.max(1e-300).ln()
4. **상태 intercept**: state_pred[k_states_diff] += c_t (trend 기여)

---

## 3.8 start_params.rs — CSS 2단계 OLS

statsmodels `sarimax.py:820-907` 재현:

1. exog 효과 제거: beta = pinv(exog)*y, y -= exog*beta
2. NaN 제거 (차분 결과)
3. **Stage 1**: MA가 있으면 AR(2q) OLS → 잔차 프록시
4. **Stage 2**: 결합 OLS: y_t ~ ar_lags(y) + ma_lags(residuals)
5. 계절: 동일 과정을 계절 lag으로 반복
6. 분산 = mean(residuals^2), floor 1e-10
7. 정상성 위반 시 0으로 리셋

---

## 3.9 optimizer.rs — argmin 0.11 (검증된 API)

### L-BFGS 설정 (argmin 공식 예제 기반)

```rust
let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9)?;
let solver = LBFGS::new(linesearch, 7)        // memory = 7
    .with_tolerance_grad(1e-5)?;

Executor::new(problem, solver)
    .configure(|state| state.param(init_params).max_iters(maxiter))
    .run()?
```

### NelderMead 폴백

```rust
// n+1 꼭짓점 심플렉스
let mut simplex = vec![init_params.clone()];
for i in 0..n { let mut v = init_params.clone(); v[i] += 0.05*(1.0+v[i].abs()); simplex.push(v); }

NelderMead::new(simplex).with_sd_tolerance(1e-8)?
```

### 그래디언트 (finitediff 0.2)

```rust
impl Gradient for SarimaxProblem {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;
    fn gradient(&self, p: &Vec<f64>) -> Result<Vec<f64>, ArgminError> {
        Ok(p.central_diff(&|x| self.cost(x).unwrap_or(f64::MAX)))
    }
}
```

### 최적화 전략
1. **기본**: L-BFGS (maxiter=50)
2. **L-BFGS 실패 시**: NelderMead (maxiter=500) 자동 폴백
3. **다중 초기값**: CSS 초기값 + 랜덤 변형 3개 → 최소 cost 선택

---

## 3.10 information.rs — AIC/BIC (ver1 동일)

```rust
pub struct InformationCriteria { pub aic: f64, pub aicc: f64, pub bic: f64, pub hqic: f64 }

impl InformationCriteria {
    pub fn compute(loglike: f64, k: usize, n: usize) -> Self {
        let (kf, nf) = (k as f64, n as f64);
        let aic = 2.0*kf - 2.0*loglike;
        let aicc = if n > k+1 { aic + (2.0*kf*kf + 2.0*kf)/(nf-kf-1.0) } else { f64::INFINITY };
        let bic = kf*nf.ln() - 2.0*loglike;
        let hqic = 2.0*kf*nf.ln().ln() - 2.0*loglike;
        Self { aic, aicc, bic, hqic }
    }
}
```

---

## 3.11 lib.rs — pyo3 0.23 바인딩

```rust
#[pyfunction]
#[pyo3(signature = (y, order, seasonal, params, exog=None, concentrate_scale=true))]
fn sarimax_loglike<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    params: PyReadonlyArray1<'py, f64>,
    exog: Option<PyReadonlyArray2<'py, f64>>,
    concentrate_scale: bool,
) -> PyResult<f64> { /* ... */ }

#[pyfunction]
#[pyo3(signature = (y, order, seasonal, exog=None, method="lbfgs", maxiter=50,
                     concentrate_scale=true, enforce_stationarity=false,
                     enforce_invertibility=false, trend="n"))]
fn sarimax_fit<'py>(/* ... */) -> PyResult<Bound<'py, PyDict>> { /* ... */ }

#[pyfunction]
#[pyo3(signature = (y, p_max=3, d_max=1, q_max=3, pp_max=1, dd_max=1, qq_max=1,
                     s=12, exog=None, criterion="aic", maxiter=50))]
fn auto_select<'py>(/* ... */) -> PyResult<Bound<'py, PyDict>> { /* rayon grid */ }

#[pymodule]
fn sarimax_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sarimax_loglike, m)?)?;
    m.add_function(wrap_pyfunction!(sarimax_fit, m)?)?;
    m.add_function(wrap_pyfunction!(auto_select, m)?)?;
    Ok(())
}
```

### numpy 배열 처리

| 메서드 | 반환 | 요건 | 용도 |
|--------|------|------|------|
| `as_slice()` | `Result<&[f64]>` | C-contiguous | 1D 시계열 (가장 빠름) |
| `as_array()` | `ArrayView` | 없음 | 2D exog (stride 안전) |

---

# 4. 개발 Phase (ver1.1 확정)

| Phase | 핵심 작업 | Go/No-Go 기준 | 기간 |
|-------|----------|---------------|------|
| **0** | pyproject.toml + uv + maturin 파이프라인 | `uv run maturin develop` 성공 | 2일 |
| **1a** | error, types, polynomial, params | 단위 테스트 전부 통과 | 3일 |
| **1b** | state_space (비계절 ARIMA) | ARIMA(1,0,1) T,Z,R == sm 행렬 | 1주 |
| **1c** | initialization (approx diffuse) | 초기 상태 검증 | 2일 |
| **1d** | kalman (concentrated) | ARIMA(1,0,1) loglike < 1e-6 | 1주 |
| **1e** | lib.rs pyo3 (loglike만) | Python 호출 성공 | 2일 |
| **2a** | state_space 계절 확장 | SARIMA(1,1,1)(1,1,1,12) 검증 | 1-2주 |
| **2b** | initialization (Mixed/Lyapunov) | Lyapunov 해 검증 | 3일 |
| **2c** | exog 지원 | SARIMAX exog loglike 일치 | 3일 |
| **3a** | start_params (CSS 2단계) | sm start_params 비교 | 1주 |
| **3b** | Monahan 변환 | 왕복 변환 검증 | 3일 |
| **3c** | optimizer (L-BFGS + NM) | fit params < 1e-4 | 1주 |
| **3d** | information (AIC/BIC) | IC < 1e-4 | 1일 |
| **4a** | selection (rayon grid) | auto_select == Python 순차 | 1주 |
| **4b** | forecast | 예측 < 1e-4 | 1주 |
| **5** | batch (rayon) | 1000개 정상 | 3일 |
| **6** | diagnostics, 통합 테스트 | 전체 통과 | 1-2주 |

**총 예상: 10-14주**

---

# 5. 테스트 기준값 생성

```python
# python_tests/generate_fixtures.py
import json, numpy as np
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX

def generate():
    np.random.seed(42)
    y = np.cumsum(np.random.randn(200)) + 10
    cases = [
        {"order": (1,0,0), "seasonal": (0,0,0,0), "label": "ar1"},
        {"order": (1,0,1), "seasonal": (0,0,0,0), "label": "arma11"},
        {"order": (1,1,1), "seasonal": (0,0,0,0), "label": "arima111"},
        {"order": (1,1,1), "seasonal": (1,1,1,12), "label": "sarima"},
    ]
    fixtures = {"endog": y.tolist()}
    for c in cases:
        m = SARIMAX(y, order=c["order"], seasonal_order=c["seasonal"],
                    enforce_stationarity=False, enforce_invertibility=False)
        r = m.fit(disp=False)
        fixtures[c["label"]] = {
            "params": r.params.tolist(), "loglike": float(r.llf),
            "aic": float(r.aic), "bic": float(r.bic), "scale": float(r.scale),
            "transition": m.ssm["transition"][:,:,0].tolist(),
            "design": m.ssm["design"][:,:,0].tolist(),
            "selection": m.ssm["selection"][:,:,0].tolist(),
        }
    out = Path("tests/fixtures/statsmodels_reference.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(fixtures, f, indent=2)
```

---

# 6. 핵심 결론

1. **concentrated log-likelihood가 핵심 차이** — ver1은 일반 loglike만 고려했으나, statsmodels 기본 동작은 concentrated
2. **Monahan 변환이 필수** — enforce_stationarity 시 최적화 공간이 달라짐
3. **CSS 초기값이 수렴을 결정** — 나쁜 초기값 → L-BFGS 발산 → NelderMead 폴백 필요
4. **AIC는 trivial** — 진짜 어려운 것은 state_space.rs (27차원 행렬 정확 구성)
5. **uv + maturin** — Phase 0에서 빌드 파이프라인을 확정해야 후속 작업 가능
