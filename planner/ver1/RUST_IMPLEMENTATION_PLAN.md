# Rust-Side Implementation Plan (ver1)

## SARIMAX Numerical Engine + AIC Model Selection in Pure Rust

---

# 1. 핵심 변경점 (기존 계획 대비)

기존 `RUST_SARIMAX_IMPLEMENTATION_PLAN.md`에서 **AIC/BIC를 Python 레이어에 배치**했으나,
본 ver1에서는 **AIC/BIC/AICc를 Rust에서 직접 계산**하도록 변경한다.

### 변경 근거
- AIC = `2k - 2*loglike` → 로그우도가 Rust에서 계산되므로 AIC도 Rust에서 처리하는 것이 자연스럽다
- `argmin` crate으로 L-BFGS 최적화가 Rust 내부에서 가능
- `rayon`으로 모델 선택 grid search를 병렬화할 수 있음
- Python ↔ Rust 왕복 호출 오버헤드 제거

---

# 2. 프로젝트 구조 (확장)

```
sarimax_rs/
├── Cargo.toml
├── pyproject.toml
├── src/
│   ├── lib.rs              # pyo3 바인딩 + 모듈 선언
│   ├── params.rs           # 파라미터 언팩/팩
│   ├── state_space.rs      # SARIMAX → 상태공간 변환
│   ├── kalman.rs           # 칼만 필터 (로그우도 누적 포함)
│   ├── likelihood.rs       # 로그우도 계산 래퍼
│   ├── optimizer.rs        # argmin 기반 MLE (NEW)
│   ├── information.rs      # AIC/BIC/AICc 계산 (NEW)
│   ├── forecast.rs         # 예측 계산
│   ├── selection.rs        # auto-order selection (NEW)
│   └── batch.rs            # 배치 병렬 처리
├── tests/
│   ├── test_params.rs
│   ├── test_state_space.rs
│   ├── test_kalman.rs
│   ├── test_likelihood.rs
│   ├── test_optimizer.rs
│   ├── test_information.rs
│   └── test_selection.rs
├── benches/
│   ├── bench_kalman.rs
│   ├── bench_fit.rs
│   └── bench_batch.rs
└── python_tests/
    ├── test_loglike.py
    ├── test_fit.py
    ├── test_aic.py
    └── test_vs_statsmodels.py
```

---

# 3. 의존성 (Cargo.toml)

```toml
[package]
name = "sarimax_rs"
version = "0.1.0"
edition = "2021"

[lib]
name = "sarimax_rs"
crate-type = ["cdylib"]  # pyo3용

[dependencies]
# 핵심 수치 연산
nalgebra = "0.33"           # 행렬 연산, Cholesky, LU 분해
ndarray = "0.16"            # N차원 배열 (pyo3-numpy 호환)

# 최적화
argmin = "0.11"             # L-BFGS, BFGS, Nelder-Mead
argmin-math = { version = "0.5", features = ["nalgebra_latest-serde"] }
finitediff = "0.1"          # 유한차분 그래디언트

# 통계
statrs = "0.18"             # 통계 분포 (Normal 등)

# 병렬 처리
rayon = "1.10"              # 데이터 병렬화

# Python 바인딩
pyo3 = { version = "0.22", features = ["extension-module"] }
numpy = "0.22"              # pyo3-numpy

# 직렬화 (선택)
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
approx = "0.5"              # 부동소수점 비교

[[bench]]
name = "bench_kalman"
harness = false
```

---

# 4. 핵심 모듈 상세 설계

---

## 4.1 params.rs — 파라미터 관리

### 역할
statsmodels의 파라미터 벡터 레이아웃을 정확히 복제한다.

### 구조체

```rust
/// SARIMAX 모델 차수 정의
#[derive(Debug, Clone)]
pub struct SarimaxOrder {
    pub p: usize,  // AR order
    pub d: usize,  // differencing order
    pub q: usize,  // MA order
    pub sp: usize, // seasonal AR order (P)
    pub sd: usize, // seasonal differencing order (D)
    pub sq: usize, // seasonal MA order (Q)
    pub s: usize,  // seasonal period
}

/// 언팩된 파라미터 구조체
#[derive(Debug, Clone)]
pub struct SarimaxParams {
    pub ar_coeffs: DVector<f64>,      // AR 계수 [p]
    pub ma_coeffs: DVector<f64>,      // MA 계수 [q]
    pub sar_coeffs: DVector<f64>,     // 계절 AR 계수 [P]
    pub sma_coeffs: DVector<f64>,     // 계절 MA 계수 [Q]
    pub exog_coeffs: DVector<f64>,    // 외생변수 계수
    pub sigma2: f64,                  // 오차 분산
}

impl SarimaxParams {
    /// 1D 파라미터 벡터에서 언팩
    pub fn from_flat(params: &[f64], order: &SarimaxOrder, n_exog: usize) -> Self { ... }

    /// 1D 파라미터 벡터로 팩
    pub fn to_flat(&self) -> DVector<f64> { ... }

    /// 총 파라미터 수 (AIC의 k)
    pub fn n_params(&self) -> usize {
        self.ar_coeffs.len()
            + self.ma_coeffs.len()
            + self.sar_coeffs.len()
            + self.sma_coeffs.len()
            + self.exog_coeffs.len()
            + 1  // sigma2
    }
}
```

### statsmodels 파라미터 레이아웃
```
[exog_coeffs | ar_coeffs | ma_coeffs | sar_coeffs | sma_coeffs | sigma2]
```
> statsmodels는 exog를 맨 앞에 배치한다. 이 순서를 정확히 따른다.

---

## 4.2 state_space.rs — 상태공간 변환

### 역할
SARIMAX(p,d,q)(P,D,Q)_s 모델을 상태공간 표현(State Space Form)으로 변환한다.

### Harvey Representation

상태공간 모델:
```
x_{t+1} = T * x_t + R * eta_t      (상태 전이)
y_t     = Z * x_t + epsilon_t       (관측 방정식)
```

### 핵심 행렬 구성

```rust
/// 상태공간 행렬 집합
pub struct StateSpace {
    pub t_mat: DMatrix<f64>,  // 상태 전이 행렬 T [m x m]
    pub z_mat: DVector<f64>,  // 관측 벡터 Z [m] (SARIMAX는 1변량)
    pub r_mat: DMatrix<f64>,  // 선택 행렬 R [m x r]
    pub q_mat: DMatrix<f64>,  // 상태 노이즈 공분산 Q [r x r]
    pub h_scalar: f64,        // 관측 노이즈 분산 H (스칼라, 1변량)
    pub state_dim: usize,     // m: 상태 차원
}

impl StateSpace {
    /// SARIMAX 파라미터로부터 상태공간 행렬 구성
    pub fn from_sarimax(params: &SarimaxParams, order: &SarimaxOrder) -> Self {
        // 1. 계절/비계절 AR/MA 다항식 곱 계산
        // 2. 확장된 AR/MA 계수 도출
        // 3. T, Z, R, Q, H 행렬 구성
        // ...
    }
}
```

### 상태 차원 계산
```
m = max(p + s*P, q + s*Q + 1)   (차분 후 기준)
```

### AR/MA 다항식 곱
비계절 AR: `phi(B) = 1 - phi_1*B - ... - phi_p*B^p`
계절 AR: `Phi(B^s) = 1 - Phi_1*B^s - ... - Phi_P*B^{Ps}`
결합: `phi(B) * Phi(B^s)` → 확장된 AR 계수

---

## 4.3 kalman.rs — 칼만 필터

### 역할
칼만 필터를 실행하고 **로그우도를 누적 계산**한다.

```rust
/// 칼만 필터 결과
pub struct KalmanResult {
    pub loglike: f64,                    // 총 로그우도
    pub filtered_states: Vec<DVector<f64>>,  // 필터링된 상태
    pub innovations: Vec<f64>,           // 혁신 (예측 오차) v_t
    pub innovation_vars: Vec<f64>,       // 혁신 분산 F_t
}

/// 칼만 필터 실행
pub fn kalman_filter(
    y: &[f64],
    exog: Option<&DMatrix<f64>>,
    ss: &StateSpace,
    order: &SarimaxOrder,
) -> KalmanResult {
    let m = ss.state_dim;

    // 초기 상태: approximate diffuse initialization
    let mut state = DVector::zeros(m);
    let mut cov = DMatrix::identity(m, m) * 1e6;  // 큰 초기 공분산

    let mut loglike = 0.0;
    let mut innovations = Vec::with_capacity(y.len());
    let mut innovation_vars = Vec::with_capacity(y.len());

    for t in 0..y.len() {
        // === 예측 단계 ===
        let state_pred = &ss.t_mat * &state;
        let cov_pred = &ss.t_mat * &cov * ss.t_mat.transpose()
                       + &ss.r_mat * &ss.q_mat * ss.r_mat.transpose();

        // === 혁신 계산 ===
        let y_pred = ss.z_mat.dot(&state_pred) + exog_contribution(exog, t, ...);
        let v_t = y[t] - y_pred;
        let f_t = ss.z_mat.dot(&(&cov_pred * &ss.z_mat)) + ss.h_scalar;

        // === 로그우도 누적 ===
        // ln L = -0.5 * sum[ ln(2*pi) + ln(F_t) + v_t^2/F_t ]
        if f_t > 0.0 {
            loglike += -0.5 * (
                (2.0 * std::f64::consts::PI).ln()
                + f_t.ln()
                + v_t * v_t / f_t
            );
        }

        // === 업데이트 단계 ===
        let k_gain = &cov_pred * &ss.z_mat / f_t;  // 칼만 이득
        state = state_pred + &k_gain * v_t;

        // Joseph form (수치 안정성)
        let i_kz = DMatrix::identity(m, m)
                   - &k_gain * ss.z_mat.transpose();
        cov = &i_kz * &cov_pred * i_kz.transpose()
              + &k_gain * ss.h_scalar * k_gain.transpose();

        innovations.push(v_t);
        innovation_vars.push(f_t);
    }

    KalmanResult { loglike, filtered_states: vec![], innovations, innovation_vars }
}
```

### 수치 안정성 강화
1. **Joseph form** 공분산 업데이트 (양정치성 보장)
2. **Cholesky 분해** F_t 역행렬 대신 사용 (다변량 확장 시)
3. **F_t ≤ 0 보호**: 로그우도에 큰 페널티 부여
4. **언더플로 방지**: 로그 스케일 연산 유지

---

## 4.4 optimizer.rs — MLE 최적화 (NEW)

### 역할
`argmin`을 사용하여 순수 Rust 내에서 최대우도추정(MLE)을 수행한다.

```rust
use argmin::core::{CostFunction, Gradient, Executor};
use argmin::solver::quasinewton::LBFGS;
use argmin::solver::neldermead::NelderMead;
use finitediff::FiniteDiff;

/// 음의 로그우도 비용 함수
struct NegLogLikelihood {
    y: Vec<f64>,
    exog: Option<DMatrix<f64>>,
    order: SarimaxOrder,
}

impl CostFunction for NegLogLikelihood {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<f64, argmin::core::Error> {
        let sarimax_params = SarimaxParams::from_flat(params, &self.order, n_exog);
        let ss = StateSpace::from_sarimax(&sarimax_params, &self.order);
        let result = kalman_filter(&self.y, self.exog.as_ref(), &ss, &self.order);
        Ok(-result.loglike)  // 최소화 = 음의 로그우도 최소화
    }
}

impl Gradient for NegLogLikelihood {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, params: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        // 유한차분 그래디언트
        Ok(params.central_diff(&|p| self.cost(p).unwrap()))
    }
}

/// SARIMAX 모델 피팅
pub fn fit_sarimax(
    y: &[f64],
    exog: Option<&DMatrix<f64>>,
    order: &SarimaxOrder,
    start_params: Option<&[f64]>,
    method: &str,
    maxiter: u64,
) -> FitResult {
    let cost_fn = NegLogLikelihood { ... };

    let init_params = start_params
        .map(|p| p.to_vec())
        .unwrap_or_else(|| css_initial_params(y, order));  // CSS 초기값

    let result = match method {
        "lbfgs" => {
            let linesearch = MoreThuenteLineSearch::new();
            let solver = LBFGS::new(linesearch, 7);
            Executor::new(cost_fn, solver)
                .configure(|state| state.param(init_params).max_iters(maxiter))
                .run()
        }
        "nelder-mead" => {
            let solver = NelderMead::new(generate_simplex(&init_params));
            Executor::new(cost_fn, solver)
                .configure(|state| state.max_iters(maxiter))
                .run()
        }
        _ => panic!("Unsupported method: {}", method),
    };

    FitResult {
        params: result.state.best_param,
        loglike: -result.state.best_cost,
        n_iter: result.state.iter,
        converged: result.state.terminated(),
    }
}

/// CSS (Conditional Sum of Squares) 초기 파라미터 추정
fn css_initial_params(y: &[f64], order: &SarimaxOrder) -> Vec<f64> {
    // Hannan-Rissanen 또는 OLS 기반 초기 추정
    // ...
}
```

---

## 4.5 information.rs — 정보 기준 계산 (NEW)

### 역할
AIC, AICc, BIC, HQIC를 Rust에서 직접 계산한다.

```rust
/// 모델 정보 기준 계산 결과
#[derive(Debug, Clone)]
pub struct InformationCriteria {
    pub aic: f64,
    pub aicc: f64,
    pub bic: f64,
    pub hqic: f64,
}

impl InformationCriteria {
    /// 로그우도, 파라미터 수, 관측치 수로부터 계산
    pub fn compute(loglike: f64, k: usize, n: usize) -> Self {
        let k_f = k as f64;
        let n_f = n as f64;

        // AIC = 2k - 2*loglike
        let aic = 2.0 * k_f - 2.0 * loglike;

        // AICc = AIC + (2k^2 + 2k) / (n - k - 1)
        let aicc = if n > k + 1 {
            aic + (2.0 * k_f.powi(2) + 2.0 * k_f) / (n_f - k_f - 1.0)
        } else {
            f64::INFINITY  // 표본 크기 부족
        };

        // BIC = k*ln(n) - 2*loglike
        let bic = k_f * n_f.ln() - 2.0 * loglike;

        // HQIC = 2k*ln(ln(n)) - 2*loglike
        let hqic = 2.0 * k_f * n_f.ln().ln() - 2.0 * loglike;

        InformationCriteria { aic, aicc, bic, hqic }
    }
}

/// FitResult에서 정보 기준 자동 계산
impl FitResult {
    pub fn information_criteria(&self, n_obs: usize) -> InformationCriteria {
        InformationCriteria::compute(
            self.loglike,
            self.n_params(),
            n_obs,
        )
    }
}
```

### AIC가 Rust에서 가능한 이유 (수학적 근거)

1. **AIC 공식**: `2k - 2*ln(L)` — 단순 사칙연산
2. **k (파라미터 수)**: `SarimaxOrder`에서 직접 계산 가능
3. **ln(L) (로그우도)**: 칼만 필터에서 이미 계산
4. **MLE 최적화**: `argmin`의 L-BFGS로 가능
5. **결론**: Python에 의존할 이유가 전혀 없음

---

## 4.6 selection.rs — 자동 모델 선택 (NEW)

### 역할
R의 `auto.arima`와 유사한 자동 차수 선택을 Rust에서 수행한다.

```rust
use rayon::prelude::*;

/// 모델 선택 결과
#[derive(Debug, Clone)]
pub struct SelectionResult {
    pub best_order: SarimaxOrder,
    pub best_params: Vec<f64>,
    pub best_loglike: f64,
    pub best_criterion: f64,
    pub all_results: Vec<(SarimaxOrder, f64)>,  // (order, criterion)
}

/// Grid search 기반 모델 선택 (rayon 병렬화)
pub fn auto_select(
    y: &[f64],
    exog: Option<&DMatrix<f64>>,
    p_range: std::ops::Range<usize>,
    d_range: std::ops::Range<usize>,
    q_range: std::ops::Range<usize>,
    sp_range: std::ops::Range<usize>,
    sd_range: std::ops::Range<usize>,
    sq_range: std::ops::Range<usize>,
    s: usize,
    criterion: &str,  // "aic", "aicc", "bic", "hqic"
    maxiter: u64,
) -> SelectionResult {
    // 후보 생성
    let candidates: Vec<SarimaxOrder> = generate_candidates(
        p_range, d_range, q_range,
        sp_range, sd_range, sq_range, s
    );

    // 병렬 피팅 (rayon)
    let results: Vec<Option<(SarimaxOrder, FitResult, f64)>> = candidates
        .par_iter()
        .map(|order| {
            match fit_sarimax(y, exog, order, None, "lbfgs", maxiter) {
                Ok(fit) => {
                    let ic = fit.information_criteria(y.len());
                    let score = match criterion {
                        "aic" => ic.aic,
                        "aicc" => ic.aicc,
                        "bic" => ic.bic,
                        "hqic" => ic.hqic,
                        _ => ic.aic,
                    };
                    Some((order.clone(), fit, score))
                }
                Err(_) => None,  // 수렴 실패 → 건너뜀
            }
        })
        .collect();

    // 최적 모델 선택
    let best = results.into_iter()
        .flatten()
        .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
        .unwrap();

    SelectionResult {
        best_order: best.0,
        best_params: best.1.params,
        best_loglike: best.1.loglike,
        best_criterion: best.2,
        all_results: ...,
    }
}
```

### Stepwise 방식 (향후 확장)

Grid search 외에 R `forecast::auto.arima`의 stepwise 알고리즘도 구현 가능:
1. 기본 모델에서 시작 (예: ARIMA(2,d,2))
2. 인접 차수 탐색 (p±1, q±1)
3. 개선이 없을 때까지 반복
4. 연산량: O(grid) → O(stepwise) ≈ 1/10 수준

---

## 4.7 forecast.rs — 예측

```rust
pub fn forecast(
    y: &[f64],
    exog: Option<&DMatrix<f64>>,
    exog_future: Option<&DMatrix<f64>>,
    params: &SarimaxParams,
    order: &SarimaxOrder,
    steps: usize,
) -> (Vec<f64>, Vec<f64>) {
    // 1. 칼만 필터로 최종 상태 획득
    // 2. 상태 전이 반복으로 h-step 예측
    // 3. 예측 평균, 분산 반환
    (means, variances)
}
```

---

## 4.8 batch.rs — 배치 병렬 처리

```rust
/// N개 시계열 배치 피팅 (rayon)
pub fn batch_fit(
    y_list: &[Vec<f64>],
    exog_list: &[Option<DMatrix<f64>>],
    orders: &[SarimaxOrder],  // 시계열별 또는 공통
    maxiter: u64,
) -> Vec<FitResult> {
    y_list.par_iter()
        .zip(exog_list.par_iter())
        .zip(orders.par_iter())
        .map(|((y, exog), order)| {
            fit_sarimax(y, exog.as_ref(), order, None, "lbfgs", maxiter)
                .unwrap_or_default()
        })
        .collect()
}
```

---

## 4.9 lib.rs — pyo3 바인딩

```rust
use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

#[pyfunction]
fn sarimax_loglike(
    py: Python,
    y: PyReadonlyArray1<f64>,
    exog: Option<PyReadonlyArray2<f64>>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    params: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    // numpy → Rust slice (zero-copy)
    let y_slice = y.as_slice()?;
    let params_slice = params.as_slice()?;
    // ... Rust 연산 ...
    Ok(loglike)
}

#[pyfunction]
fn sarimax_fit(
    py: Python,
    y: PyReadonlyArray1<f64>,
    exog: Option<PyReadonlyArray2<f64>>,
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    method: &str,
    maxiter: u64,
    start_params: Option<PyReadonlyArray1<f64>>,
) -> PyResult<PyObject> {
    // Rust fit + AIC/BIC 포함 결과 반환
    // ...
}

#[pyfunction]
fn sarimax_auto_select(
    py: Python,
    y: PyReadonlyArray1<f64>,
    exog: Option<PyReadonlyArray2<f64>>,
    p_max: usize, d_max: usize, q_max: usize,
    sp_max: usize, sd_max: usize, sq_max: usize,
    s: usize,
    criterion: &str,
    maxiter: u64,
) -> PyResult<PyObject> {
    // Rust 내부 병렬 모델 선택
    // ...
}

#[pyfunction]
fn batch_loglike(
    py: Python,
    y_list: Vec<PyReadonlyArray1<f64>>,
    // ...
) -> PyResult<Vec<f64>> {
    // rayon 병렬 배치 처리
    // ...
}

#[pymodule]
fn sarimax_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sarimax_loglike, m)?)?;
    m.add_function(wrap_pyfunction!(sarimax_fit, m)?)?;
    m.add_function(wrap_pyfunction!(sarimax_auto_select, m)?)?;
    m.add_function(wrap_pyfunction!(sarimax_forecast, m)?)?;
    m.add_function(wrap_pyfunction!(batch_loglike, m)?)?;
    m.add_function(wrap_pyfunction!(batch_fit, m)?)?;
    Ok(())
}
```

---

# 5. 개발 Phase 재정의

| Phase | 내용 | 기간 | 산출물 |
|-------|------|------|--------|
| **1a** | params.rs + state_space.rs (비계절 ARIMA만) | 1주 | 상태공간 변환 |
| **1b** | kalman.rs + likelihood.rs | 1주 | 로그우도 계산 |
| **1c** | lib.rs pyo3 바인딩 (loglike만) | 0.5주 | Python에서 호출 가능 |
| **2** | state_space.rs 계절 확장 + exog | 1-2주 | 전체 SARIMAX 지원 |
| **3a** | optimizer.rs (argmin L-BFGS) | 1주 | Rust 내부 MLE |
| **3b** | information.rs (AIC/BIC/AICc/HQIC) | 0.5주 | 정보 기준 계산 |
| **3c** | selection.rs (auto-select) | 1주 | 자동 모델 선택 |
| **4** | forecast.rs | 1주 | 예측 기능 |
| **5** | batch.rs + rayon 병렬화 | 1주 | 배치 처리 |
| **6** | 검증 + 벤치마크 + 문서화 | 1-2주 | 릴리스 준비 |

**총 예상: 8-11주**

---

# 6. 테스트 전략

### Rust 단위 테스트
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_aic_computation() {
        let ic = InformationCriteria::compute(-100.0, 5, 200);
        assert_relative_eq!(ic.aic, 210.0, epsilon = 1e-10);
        // AIC = 2*5 - 2*(-100) = 10 + 200 = 210
    }

    #[test]
    fn test_bic_computation() {
        let ic = InformationCriteria::compute(-100.0, 5, 200);
        let expected_bic = 5.0 * (200.0_f64).ln() + 200.0;
        assert_relative_eq!(ic.bic, expected_bic, epsilon = 1e-10);
    }

    #[test]
    fn test_arima_loglike_vs_known() {
        // 알려진 ARIMA(1,0,0) 모델의 로그우도 검증
        // ...
    }
}
```

### Python 통합 테스트 (vs statsmodels)
```python
def test_aic_matches_statsmodels():
    """Rust AIC vs statsmodels AIC 비교"""
    sm_model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12))
    sm_result = sm_model.fit()

    rs_result = sarimax_rs.sarimax_fit(y, None, (1,1,1), (1,1,1,12))

    assert abs(rs_result['aic'] - sm_result.aic) < 1e-4
    assert abs(rs_result['bic'] - sm_result.bic) < 1e-4
```

---

# 7. 성능 목표

| 시나리오 | Python (statsmodels) | Rust (목표) | 배수 |
|---------|---------------------|-------------|------|
| 단일 loglike 호출 | ~1ms | ~0.3ms | 3x |
| 단일 fit (50 iter) | ~50ms | ~15ms | 3x |
| auto_select (100 후보) | ~5s (순차) | ~0.5s (rayon 8코어) | 10x |
| batch_fit (1000 시계열) | ~50s (순차) | ~5s (rayon) | 10x |
| rolling forecast (1000 윈도우) | ~60s | ~6s | 10x |

---

# 8. 핵심 결론

1. **AIC는 Rust에서 100% 구현 가능** — 수학적으로 단순하며, 필요한 모든 crateが 존재
2. **argmin이 핵심 enabler** — L-BFGS/BFGS/Nelder-Mead 모두 지원
3. **rayon으로 모델 선택 병렬화**가 가장 큰 성능 이점
4. **가장 어려운 부분**은 state_space.rs (상태공간 변환) — 여기에 가장 많은 시간 투자 필요
5. **점진적 확장 전략**이 핵심 — ARIMA → SARIMA → SARIMAX 순서로 기능 추가
