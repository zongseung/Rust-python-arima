# sarimax-rs v5 — 리팩토링 계획서

> 작성일: 2026-02-25
> 대상 브랜치: main (현재 Rust 11,014줄 / 15 모듈)
> 목표: 중복 제거, 유지보수성 향상, 버그 위험 감소 (기능 변경 없음)

---

## 1. 현황 진단 — 중복/개선 영역 요약

| ID | 파일 | 영역 | 중복 규모 | 우선순위 |
|----|------|------|----------|---------|
| R-1 | `lib.rs` | PyO3 함수별 보일러플레이트 5줄 × 10함수 | ~50줄 반복 | **HIGH** |
| R-2 | `initialization.rs` | `simple_differencing` 분기 3중 중복 | 3개 동일 표현식 | HIGH |
| R-3 | `score.rs` | MA/계절MA `dRQR` 계산 블록 중복 | 8줄 × 2 | HIGH |
| R-4 | `optimizer.rs` | `eval_kf_loglike_constrained`와 `pipeline`의 기능 중복 | ~25줄 | MEDIUM |
| R-5 | `optimizer.rs` | `run_css_optimization()` 전처리가 `pipeline::prepare_endog` 재구현 | ~12줄 | MEDIUM |
| R-6 | `inference.rs` | 잔차 통계(mean/variance) 독립적 재계산 3회 | ~6줄 × 3 | MEDIUM |
| R-7 | `lib.rs` | `build_config` 호출 시 일부 인자 하드코딩 불일관 | 코드 가독성 | LOW |
| R-8 | `optimizer.rs` | `eval_loglike`가 `pipeline::kalman_eval` 미활용 | 중복 SS 빌드 경로 | LOW |

---

## 2. 상세 분석

### R-1 — `lib.rs` PyO3 보일러플레이트 중복 (HIGH)

**현황:**
10개의 `#[pyfunction]`이 각각 동일한 5줄 초기화 블록을 반복한다.

```rust
// sarimax_fit, sarimax_loglike, sarimax_forecast, sarimax_residuals,
// sarimax_inference, sarimax_diagnostics 등에서 동일하게 반복
validate_endog_finite(endog)?;
let (exog_cols, n_exog) = parse_exog(exog.as_ref());
validate_exog(&exog_cols)?;
let (p, d, q) = order;
let (pp, dd, qq, s) = seasonal;
validate_order(p, d, q, pp, dd, qq, s, n_exog)?;
let config = build_config(
    order, seasonal, n_exog,
    enforce_stationarity, enforce_invertibility,
    concentrate_scale, parse_trend(trend), simple_differencing,
);
```

`validate_endog_finite`, `parse_exog`, `validate_exog`, `validate_order`, `build_config`, `parse_trend` 6개 함수가 총 54회 호출됨.

**제안:**
`prepare_request()` 헬퍼 함수 추출:

```rust
/// PyO3 함수 공통 초기화 (validate + parse + config 빌드)
fn prepare_request(
    endog: &[f64],
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    exog: Option<&PyReadonlyArray2<f64>>,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    concentrate_scale: bool,
    trend: Option<&str>,
    simple_differencing: bool,
) -> PyResult<(SarimaxConfig, Option<Vec<Vec<f64>>>)> {
    validate_endog_finite(endog)?;
    let (exog_cols, n_exog) = parse_exog(exog);
    validate_exog(&exog_cols)?;
    let (p, d, q) = order;
    let (pp, dd, qq, s) = seasonal;
    validate_order(p, d, q, pp, dd, qq, s, n_exog)?;
    let config = build_config(order, seasonal, n_exog,
        enforce_stationarity, enforce_invertibility,
        concentrate_scale, parse_trend(trend), simple_differencing);
    Ok((config, exog_cols))
}
```

**효과:** `lib.rs` 약 50줄 제거. 향후 파라미터 추가(예: `max_ar_order`) 시 수정 지점 1곳으로 통합.

---

### R-2 — `initialization.rs` `sd` 계산 3중 중복 (HIGH)

**현황:**
`simple_differencing` 분기로 diffuse state 오프셋을 계산하는 동일 표현식이 3곳에 존재.

```rust
// mixed() at line 61
let sd = if config.simple_differencing { 0 } else { config.order.k_states_diff() };

// dare() at line 134
let sd = if config.simple_differencing { 0 } else { config.order.k_states_diff() };

// from_config() at line 217
let sd = if config.simple_differencing { 0 } else { config.order.k_states_diff() };
```

동일 패턴이 `score.rs:57`에도 존재 (총 4곳).

**제안 A — `SarimaxConfig`에 메서드 추가:**

```rust
// types.rs의 SarimaxConfig impl에 추가
impl SarimaxConfig {
    /// Effective diffuse-state offset.
    /// 0 when simple_differencing=true (diff states already removed),
    /// k_states_diff() otherwise.
    #[inline]
    pub fn effective_sd(&self) -> usize {
        if self.simple_differencing { 0 } else { self.order.k_states_diff() }
    }
}
```

**제안 B — 함수 인자로 전달 (캡슐화 선호 시):**
`KalmanInit::from_config_default`에서 `sd`를 계산해 하위 함수에 전달.

**추천:** 제안 A. `SarimaxConfig`가 이미 `k_states_diff()`를 위임하는 패턴이므로 일관성 있음.

**효과:** 4곳 → 1곳. Phase 5-C에서 발생한 `Matrix slicing out of bounds` 유형 버그 재발 방지.

---

### R-3 — `score.rs` `dRQR` 계산 블록 중복 (HIGH)

**현황:**
MA 파라미터(lines 156–163)와 계절 MA 파라미터(lines 199–206) 처리에서 동일한 `dRQR` 계산 블록이 반복됨.

```rust
// MA 처리 (lines 156-163)
let mut d_rqr = DMatrix::<f64>::zeros(k, k);
let r_col = r_mat.column(0);
for row in 0..k {
    for col in 0..k {
        d_rqr[(row, col)] = sigma2 * (dr_col[row] * r_col[col] + r_col[row] * dr_col[col]);
    }
}

// 계절 MA 처리 (lines 199-206) — 완전히 동일
let mut d_rqr = DMatrix::<f64>::zeros(k, k);
let r_col = r_mat.column(0);
for row in 0..k {
    for col in 0..k {
        d_rqr[(row, col)] = sigma2 * (dr_col[row] * r_col[col] + r_col[row] * dr_col[col]);
    }
}
```

`sigma2` 파라미터 비집중 처리(lines 212-218)도 유사 구조.

**제안:**

```rust
/// Compute dR*Q*R' + R*Q*dR' for a single-column noise matrix R.
///
/// dRQR[i,j] = σ² * (dR[i] * R[j] + R[i] * dR[j])
#[inline]
fn compute_drqr(
    dr_col: &DVector<f64>,
    r_col: nalgebra::MatrixView<f64, nalgebra::Dyn, nalgebra::U1, _, _>,
    k: usize,
    sigma2: f64,
) -> DMatrix<f64> {
    let mut d_rqr = DMatrix::<f64>::zeros(k, k);
    for row in 0..k {
        for col in 0..k {
            d_rqr[(row, col)] = sigma2 * (dr_col[row] * r_col[col] + r_col[row] * dr_col[col]);
        }
    }
    d_rqr
}
```

**효과:** `score.rs` ~16줄 제거. 공식 변경 시 단일 수정 지점.

---

### R-4 — `optimizer.rs` `eval_kf_loglike_constrained`와 `pipeline` 중복 (MEDIUM)

**현황:**
`optimizer.rs:1315`의 `eval_kf_loglike_constrained`와 `pipeline.rs:61`의 `kalman_eval_constrained`가 동일한 역할.

```rust
// optimizer.rs:1315 (현재)
fn eval_kf_loglike_constrained(
    endog: &[f64], constrained: &[f64], config: &SarimaxConfig,
    exog: Option<&[Vec<f64>]>,
) -> Result<f64> {
    let sparams = SarimaxParams::from_flat(constrained, config)?;
    let ss = StateSpace::new(config, &sparams, endog, exog)?;
    let init = KalmanInit::from_config_default(&ss, config);
    let out = kalman_loglike(endog, &ss, &init, config.concentrate_scale)?;
    Ok(out.loglike)
}

// pipeline.rs:61 (기존)
pub(crate) fn kalman_eval_constrained(...) -> Result<KalmanOutput> {
    let sparams = SarimaxParams::from_flat(constrained, config)?;
    kalman_eval(endog, &sparams, config, exog)
}
```

`pipeline::kalman_eval_constrained`는 `simple_differencing`도 처리함.

**제안:**
`eval_kf_loglike_constrained` 삭제 → `pipeline::kalman_eval_constrained(...).map(|o| o.loglike)` 로 교체.

**효과:** `optimizer.rs` ~25줄 제거. `simple_differencing` 처리 일관성 보장.

---

### R-5 — `optimizer.rs` CSS 전처리 `pipeline` 재구현 (MEDIUM)

**현황:**
`run_css_optimization()` (lines 1529–1540)에서 `simple_differencing` 전처리 로직이 `pipeline::prepare_endog`를 중복 구현.

```rust
// optimizer.rs:1529 (현재)
let (eff_endog, eff_exog_owned) = if config.simple_differencing {
    let diff = apply_differencing(endog, config);
    let n_drop = endog.len() - diff.len();
    let eff_exog = exog.map(|cols| cols.iter().map(|c| c[n_drop..].to_vec()).collect());
    (diff, eff_exog)
} else {
    (endog.to_vec(), exog.map(|c| c.to_vec()))
};
```

`pipeline::prepare_endog`가 정확히 동일한 로직.

**제안:**
`pipeline::prepare_endog`를 `pub(crate)`로 노출 → `run_css_optimization`에서 재사용.

```rust
// pipeline.rs: fn prepare_endog → pub(crate) fn prepare_endog
pub(crate) fn prepare_endog(...) -> (Vec<f64>, Option<Vec<Vec<f64>>>) { ... }

// optimizer.rs: 교체
let (eff_endog, eff_exog_owned) = pipeline::prepare_endog(endog, config, exog);
```

**효과:** ~12줄 제거. 전처리 버그가 한 곳에서만 발생.

---

### R-6 — `inference.rs` 잔차 모멘트 중복 계산 (MEDIUM)

**현황:**
`mean`, `variance` 등 기초 통계가 `ljung_box_test`, `jarque_bera_test`, `heteroskedasticity_test`에서 독립적으로 계산됨.

```rust
// ljung_box_test (line 397)
let mean = resid.iter().sum::<f64>() / n as f64;

// jarque_bera_test (lines 442-445)
let mean = resid.iter().sum::<f64>() / n;
let m2 = resid.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
let m3 = resid.iter().map(|&x| (x - mean).powi(3)).sum::<f64>() / n;
let m4 = resid.iter().map(|&x| (x - mean).powi(4)).sum::<f64>() / n;
```

`sarimax_diagnostics`가 세 테스트를 모두 호출하므로 `mean` 계산이 3회 실행됨.

**제안:**

```rust
struct ResidualMoments {
    mean: f64,
    var: f64,   // E[(x-μ)²]
    m3: f64,    // E[(x-μ)³]
    m4: f64,    // E[(x-μ)⁴]
}

fn compute_moments(resid: &[f64]) -> ResidualMoments { ... }
```

`sarimax_diagnostics`에서 한 번 계산 후 세 테스트에 전달.

**효과:** ~12줄 제거. 테스트 간 통계 일관성 보장.

---

### R-7 — `lib.rs` `build_config` 인자 불일관 (LOW)

**현황:**
일부 함수(`sarimax_forecast`, `sarimax_residuals`)는 `enforce_stationarity=false, enforce_invertibility=false`를 하드코딩하나 코드에서 의도가 불명확.

```rust
// sarimax_forecast (line 595) — 하드코딩
let config = build_config(order, seasonal, n_exog, false, false, ...);

// sarimax_fit (line 489) — 파라미터로 받음
let config = build_config(order, seasonal, n_exog,
    enforce_stationarity, enforce_invertibility, ...);
```

**제안:**
상수 이름 또는 주석으로 의도 명시:

```rust
// 예측/잔차는 파라미터가 이미 유효 범위로 제공된다고 가정 — 제약 불필요
const NO_CONSTRAINT: (bool, bool) = (false, false);
let config = build_config(order, seasonal, n_exog, NO_CONSTRAINT.0, NO_CONSTRAINT.1, ...);
```

또는 R-1의 `prepare_request`에서 `is_fit: bool` 인자로 분기.

---

### R-8 — `optimizer.rs` `eval_loglike`에서 SS 직접 빌드 (LOW)

**현황:**
`SarimaxObjective::eval_loglike`가 `pipeline::kalman_eval`를 사용하지 않고 SS를 직접 빌드한다.
이는 SS 캐싱(`take_or_build_ss` / `return_ss`) 때문에 의도적이지만, `simple_differencing` 분기가 누락되어 있다.

```rust
// optimizer.rs:225 — simple_differencing 처리 없음
fn eval_loglike(&self, unconstrained: &[f64]) -> std::result::Result<f64, String> {
    let ss = self.take_or_build_ss(&sparams)?;
    let init = KalmanInit::from_config_default(&ss, &self.config);
    let result = kalman_loglike(&self.endog, &ss, &init, ...);
    ...
}
```

현재 `self.endog`가 이미 `run_css_optimization`에서 전처리되어 있기 때문에 실제 버그는 없지만, 경로 추적이 어렵다.

**제안:**
주석으로 이 설계 결정을 명시 ("SS 캐싱으로 인해 pipeline 미사용, endog는 이미 전처리됨"). 또는 장기적으로 SS 캐시를 `pipeline` 레이어로 이동.

---

## 3. 구현 계획 (작업 단위)

### Phase R-A — 고위험 중복 제거 (1~2일)

```
R-A-1  types.rs: SarimaxConfig::effective_sd() 메서드 추가           예상: 30분
R-A-2  initialization.rs: sd 3곳 → config.effective_sd() 교체       예상: 20분
R-A-3  score.rs: sd 1곳 → config.effective_sd() 교체                예상: 15분
R-A-4  score.rs: compute_drqr() 헬퍼 추출, MA/계절MA 교체            예상: 45분
R-A-5  cargo test --all-targets (154 테스트 통과 확인)                예상: 5분
R-A-6  pytest python_tests/ -v (351+ 테스트 통과 확인)               예상: 10분
```

### Phase R-B — optimizer/pipeline 통합 (1일)

```
R-B-1  pipeline.rs: prepare_endog → pub(crate)로 노출                예상: 10분
R-B-2  optimizer.rs: run_css_optimization 전처리 → pipeline 위임     예상: 30분
R-B-3  optimizer.rs: eval_kf_loglike_constrained 삭제 → pipeline 위임 예상: 30분
R-B-4  테스트 통과 확인 (cargo test + pytest)                        예상: 15분
```

### Phase R-C — lib.rs 보일러플레이트 통합 (1일)

```
R-C-1  lib.rs: prepare_request() 헬퍼 함수 설계                      예상: 30분
R-C-2  lib.rs: sarimax_loglike, sarimax_fit에 적용 (가장 단순한 2개)  예상: 30분
R-C-3  lib.rs: 나머지 8개 함수에 적용 (배치/그리드서치 주의)           예상: 1시간
R-C-4  테스트 통과 확인                                               예상: 15분
```

### Phase R-D — inference.rs 정리 (반나절)

```
R-D-1  inference.rs: ResidualMoments 구조체 + compute_moments() 추출  예상: 30분
R-D-2  inference.rs: 세 테스트 함수에서 모멘트 인자로 수신           예상: 30분
R-D-3  sarimax_diagnostics 호출 경로에서 한 번만 계산                 예상: 20분
R-D-4  테스트 통과 확인                                               예상: 10분
```

---

## 4. 의존 관계

```
R-A-1 → R-A-2 → R-A-3 (effective_sd 체인)
R-A-4 (독립)
R-A-5, R-A-6 (Phase R-A 완료 후 검증)

R-B-1 → R-B-2
R-B-3 (독립적으로 R-B-1 이후)
R-B-4 (검증)

R-C-1 → R-C-2 → R-C-3 (점진적 적용)
R-C-4 (검증)

R-D-1 → R-D-2 → R-D-3
R-D-4 (검증)
```

---

## 5. 예상 효과

| 지표 | 현재 | 리팩토링 후 |
|------|------|------------|
| `lib.rs` 줄수 | 1,135줄 | ~1,080줄 (-50) |
| `optimizer.rs` 줄수 | 2,223줄 | ~2,185줄 (-38) |
| `score.rs` 줄수 | 1,153줄 | ~1,135줄 (-18) |
| `initialization.rs` 줄수 | 534줄 | ~530줄 (-4) |
| 중복 표현식 (sd 계산) | 4곳 | 1곳 |
| 중복 표현식 (dRQR 블록) | 2곳 | 1곳 |
| `build_config` 직접 호출 | 11회 | ~3회 (나머지는 `prepare_request` 경유) |

---

## 6. 리스크 및 주의사항

| 리스크 | 대상 | 완화 방법 |
|--------|------|----------|
| SS 캐시 깨짐 | R-B-3 (eval_kf 교체) | `take_or_build_ss` 패턴 유지 여부 먼저 확인 |
| 배치/그리드서치 보일러플레이트 복잡성 | R-C-3 | 배치 함수는 시리즈별 루프라서 별도 처리 필요 |
| `prepare_request`에서 lifetime 문제 | R-C-1 | PyO3 타입은 `py.detach()` 전에 `.to_vec()` 필요, 함수 시그니처 주의 |
| 기능 회귀 | 전체 | 각 Phase 후 `cargo test + pytest` 반드시 실행 |

---

## 7. 작업 순서 권장

```
현재 상태 (simple_differencing Phase 5-C 완료)
    ↓
R-A (effective_sd + dRQR 헬퍼) — 가장 안전하고 효과 명확
    ↓
R-B (pipeline 통합) — optimizer 복잡도 감소
    ↓
R-D (inference 모멘트) — 독립적, 언제든 가능
    ↓
R-C (lib.rs 보일러플레이트) — 가장 큰 구조 변경, 마지막에 처리
```

> **참고:** 본 리팩토링은 모두 내부 구현 변경으로, Python API(`sarimax_fit`, `sarimax_forecast` 등) 및 반환값 형식은 변경되지 않음.
