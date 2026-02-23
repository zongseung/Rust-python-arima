# VER5.2 — 성능 최적화 및 고차 수렴 개선 명세서

> **작성일**: 2026-02-23
> **대상**: `sarimax_rs/src/` (Rust SARIMAX 엔진)
> **제약조건**: d ∈ {0,1}, D ∈ {0,1} (차분 최대 1차)
> **목표**: 차수 4까지 전체 모델 수렴률 향상 + Kalman filter 속도 2-5x 가속

---

## 현재 상태 (Baseline)

| 지표 | 수치 |
|------|------|
| 전체 모델 (차수 ≤ 4) | 114개 테스트 |
| EXACT (LL diff < 0.01) | 75 (65.8%) |
| CLOSE (LL diff < 1.0) | 16 (14.0%) |
| OK (LL diff < 5.0) | 12 (10.5%) |
| WARN (LL diff ≥ 5.0, 수렴) | 4 (3.5%) |
| FAIL (미수렴) | 7 (6.1%) |
| 평균 속도 배율 vs statsmodels | 2.28x |

### 프로파일링 결과

| 모델 | k_states | KF 1회 | fit 전체 | KF+Score 비중 |
|------|----------|--------|---------|--------------|
| AR(1) | 1 | 5.1μs | 1.0ms | ~100% |
| ARIMA(1,1,1) | 3 | 11.6μs | 49ms | >95% |
| SARIMA(1,1,1)(1,1,1,12) | 27 | ~200μs | 210ms | >90% |

---

## P1: Chandrasekhar Recursions — Kalman Filter O(k³) → O(k²)

### 학술 근거

| 참고문헌 | 내용 |
|----------|------|
| **Morf, Sidhu, Kailath (1974)** — "Some New Algorithms for Recursive Estimation in Constant, Linear, Discrete-Time Systems." *IEEE Trans. Automatic Control*, 19(4), 315-323. DOI: `10.1109/TAC.1974.1100576` | 원본 알고리즘. 시불변 시스템에서 Kalman gain을 공분산 행렬 없이 직접 계산 |
| **Herbst, E. (2015)** — "Using the 'Chandrasekhar Recursions' for Likelihood Evaluation of DSGE Models." *Computational Economics*, 45(4), 693-705. DOI: `10.1007/s10614-014-9430-2` | DSGE 모델에서 최대 **5x 속도 향상** 보고. Federal Reserve FEDS Working Paper 2012-35 |
| **statsmodels PR #6411** (Chad Fulton) — [github.com/statsmodels/statsmodels/pull/6411](https://github.com/statsmodels/statsmodels/pull/6411) | Python 참조 구현. `FILTER_CHANDRASEKHAR = 0x200` |
| **Chad Fulton blog** — [chadfulton.com/topics/state_space_chandrasekhar.html](http://www.chadfulton.com/topics/state_space_chandrasekhar.html) | 구현 가이드 및 수치 비교 |

### 핵심 수학

**표준 Kalman Filter (대체 대상)**:
```
P_{t+1|t} = T · P_{t|t-1} · T' - K_t · F_t^{-1} · K_t' + R·Q·R'    ... O(k³)
K_t = T · P_{t|t-1} · Z'                                                ... O(k²)
F_t = Z' · P_{t|t-1} · Z + H                                           ... O(k)
```

**Chandrasekhar Recursion (대체)**:
```
P_{t+1|t} = P_{t|t-1} + W_t · M_t · W_t'    ... O(k²) rank-1 update

초기화:
  W_1 = K_1 = T · P̄ · Z'                    (P̄는 Lyapunov 해)
  M_1 = -F_1^{-1}

업데이트:
  F_t = F_{t-1} + Z · W_{t-1} · M_{t-1} · W_{t-1}' · Z'     ... O(k) scalar
  K_t = K_{t-1} + T · W_{t-1} · M_{t-1} · W_{t-1}' · Z'     ... O(k) vector
  W_t = (T - K_t · F_t^{-1} · Z) · W_{t-1}                   ... O(k²) mat-vec
  M_t = M_{t-1} + M_{t-1} · (W' · Z') · F^{-1} · (Z · W) · M_{t-1}  ... scalar
```

**단변량(n_y=1)에서의 단순화**: W_t는 k×1 벡터, M_t는 스칼라. 공분산 예측이 O(k) outer product로 축소.

### 적용 조건

- 시불변 모델만 (SARIMAX는 시불변 ✓)
- 정상상태 초기화 필요 (diffuse 초기화 시 burn-in 후 전환)
- 현재 `kalman.rs`의 `KalmanStrategy` enum에 `Chandrasekhar` variant 추가

### 구현 계획

**파일**: `src/kalman.rs`

```rust
// 새로운 전략 추가
enum KalmanStrategy {
    Dense,
    Sparse,
    Chandrasekhar,  // 신규
}

struct ChandrasekharState {
    w: DVector<f64>,      // k x 1 (univariate)
    m: f64,               // scalar (univariate)
    k_gain: DVector<f64>, // k x 1 steady gain
    f_var: f64,           // scalar forecast error variance
}
```

**전환 로직**:
1. Diffuse burn-in 동안 → 기존 Dense/Sparse 사용
2. Burn-in 종료 후 → Chandrasekhar 초기화 (W_1, M_1 계산)
3. 이후 모든 스텝 → Chandrasekhar recursion
4. Steady-state 감지 시 → 기존 캐시 방식으로 전환 (K_inf, F_inf 고정)

**예상 효과**:
- k=27 (s=12): O(27³=19,683) → O(27²=729) = **~27x per step**
- k=51 (s=24): O(51³=132,651) → O(51²=2,601) = **~51x per step**
- 실제 효과: non-converged 스텝(~35-40)에만 적용, 전체 **KF 2-5x 가속** 예상

### 검증 기준

- [ ] 기존 테스트 122개 Rust + 257개 Python 전체 통과
- [ ] SARIMA(1,1,1)(1,1,1,12) n=500 KF 시간 50% 이상 단축
- [ ] SARIMA(1,1,1)(1,1,1,24) n=500 KF 시간 70% 이상 단축
- [ ] 수치 정확도: 표준 KF 대비 loglike 차이 < 1e-8

---

## P2: CSS 확대 + KF 검증 — 고차 MA 수렴 개선

### 학술 근거

| 참고문헌 | 내용 |
|----------|------|
| **Box, G. E. P. and Jenkins, G. M. (1976)** — *Time Series Analysis: Forecasting and Control*. Holden-Day. | CSS (Conditional Sum of Squares) 원본 정의. Chapter 7 |
| **R `stats::arima()` 문서** — [stat.ethz.ch/R-manual/R-devel/library/stats/html/arima.html](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/arima.html) | CSS-ML 2단계 기본 방법. `method="CSS-ML"` (default) |
| **Hannan, E. J. and Rissanen, J. (1982)** — "Recursive Estimation of Mixed Autoregressive-Moving Average Order." *Biometrika*, 69(1), 81-94. DOI: `10.1093/biomet/69.1.81` | ARMA 파라미터 추정의 다단계 방법 |
| **R `forecast::Arima()` 문서** — [pkg.robjhyndman.com/forecast/reference/Arima.html](https://pkg.robjhyndman.com/forecast/reference/Arima.html) | CSS-ML 실무 적용 |

### CSS 목적 함수

```
l_CSS = -n_eff/2 · log(σ²_CSS) - n_eff/2
σ²_CSS = (1/n_eff) · Σ_{t=r+1}^{T} e_t²

여기서:
  e_t = y_t - Σ_{j=1}^{p'} α_j · y_{t-j} - Σ_{j=1}^{q'} β_j · e_{t-j}
  r = max(p', q')  (초기 관측 조건부)
  p', q' = 전개된 (계절×비계절) 다항식 차수
```

비용: O(n · (p' + q')) per evaluation vs O(n · k³) for KF.

### 현재 구현의 문제

현재 CSS는 **계절 SMA 모델(qq > 0, s ≥ 12)에만** 적용됨. 비계절 ARMA(p,q) q ≥ 2에서 FAIL/WARN 발생:
- ARIMA(4,0,2): WARN
- ARIMA(2,1,2): WARN (seed에 따라)
- ARIMA(4,1,4): WARN

### 구현 계획

**파일**: `src/optimizer.rs` — `fit()` 함수

**변경 1: CSS 적용 범위 확대 + KF 검증**:
```rust
// 현재 (계절 모델만)
let has_sma = config.order.qq > 0;
let has_seasonal = (config.order.pp > 0 || config.order.qq > 0) && config.order.s >= 12;
if has_sma && has_seasonal && start_params.is_none() && maxiter > 0 { ... }

// 변경안: MA 2차 이상 또는 계절 SMA 모델
let has_ma = config.order.q > 0 || config.order.qq > 0;
let benefit_from_css = has_ma && (
    config.order.q + config.order.qq >= 2
    || (config.order.qq > 0 && config.order.s >= 4)
);
if benefit_from_css && start_params.is_none() && maxiter > 0 {
    let css_result = run_css_optimization(endog, config, &constrained_start, 100);
    if let Some(css_params) = css_result {
        // KF 검증: CSS 결과가 KF loglike에서도 더 좋은지 확인
        let css_kf_ll = eval_kf_loglike(endog, config, &css_params, exog);
        let orig_kf_ll = eval_kf_loglike(endog, config, &constrained_start, exog);
        if css_kf_ll > orig_kf_ll {
            constrained_start = css_params;
        }
    }
}
```

**핵심 안전장치**: CSS 결과를 바로 사용하지 않고, **KF loglikelihood로 교차 검증**하여 CSS가 MLE에도 더 좋은 시작점인지 확인. 이전 ARIMA(2,1,2) 회귀를 방지.

**변경 2: CSS 해석적 기울기** (선택):
```rust
// CSS 수치 기울기 (현재): n_params+1 회 평가
// CSS 해석적 기울기 (신규): 1회 forward pass에서 기울기 동시 계산
// de_t/d(phi_j) = -y_{t-j-1} - Σ β_k · de_{t-k}/d(phi_j)
// 비용: O(n · n_params · (p'+q')) vs O(n · n_params · (p'+q'))
// → 동일 복잡도지만 상수 배 절감 (함수 호출, 변환 오버헤드 제거)
```

### 예상 효과

- 비계절 ARMA(p,q) q ≥ 2: WARN/FAIL 감소 (4건 → 0-1건)
- 추가 비용: CSS 1회 + KF 검증 1회 (O(n·(p'+q')) + O(n·k²), 무시 가능)

### 검증 기준

- [ ] ARIMA(4,0,2), (2,1,2), (4,1,4) 정확도 개선 (LL diff < 5.0 → < 1.0)
- [x] ARIMA(2,1,2) seed=73 회귀 없음 (기존 테스트 통과)
- [x] 기존 257개 Python 테스트 전체 통과

---

## P3: DARE 초기화 — 정상상태 공분산 직접 계산

### 학술 근거

| 참고문헌 | 내용 |
|----------|------|
| **Laub, A. J. (1979)** — "A Schur Method for Solving Algebraic Riccati Equations." *IEEE Trans. Automatic Control*, 24(6), 913-921. | DARE Schur 분해 해법 (gold standard) |
| **Anderson, B. D. O. and Moore, J. B. (1979)** — *Optimal Filtering*. Prentice-Hall. (Dover reprint 2005, ISBN 0-486-43938-0) | Doubling algorithm: 지수 수렴 |
| **Durbin, J. and Koopman, S. J. (2012)** — *Time Series Analysis by State Space Methods*, 2nd ed. Oxford Univ. Press. Section 4.3 | 시불변 Kalman filter 수렴 분석 |
| **SciPy `solve_discrete_are()`** — [docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_discrete_are.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_discrete_are.html) | Python 참조 구현 |
| **Chad Fulton** — [chadfulton.com/topics/kalman_init_stationary.html](http://www.chadfulton.com/topics/kalman_init_stationary.html) | statsmodels 정상상태 초기화 가이드 |

### DARE 방정식

```
P_∞ = T · P_∞ · T' - T · P_∞ · Z' · (Z · P_∞ · Z' + H)^{-1} · Z · P_∞ · T' + R·Q·R'
```

**단변량(H=0, concentrate_scale) 단순화**:
```
P_∞ = T · P_∞ · T' - (T · P_∞ · z) · (z' · P_∞ · z)^{-1} · (z' · P_∞ · T') + R·R'
```

### 해법: 반복 Riccati 수렴

```rust
fn solve_dare_iterative(
    t: &DMatrix<f64>,
    z: &DVector<f64>,
    rqr: &DMatrix<f64>,
    max_iter: usize,   // 50-100
    tol: f64,           // 1e-12
) -> DMatrix<f64> {
    let k = t.nrows();
    let mut p = rqr.clone();  // P_0 = R·Q·R'
    let mut p_new = DMatrix::zeros(k, k);

    for _ in 0..max_iter {
        let pz = &p * z;                       // k x 1
        let f_inv = 1.0 / z.dot(&pz);          // scalar
        // P_new = T·P·T' - (T·P·z)·F^{-1}·(z'·P·T') + R·Q·R'
        let tpz = t * &pz;                     // k x 1
        p_new = t * &p * t.transpose() + rqr;
        p_new -= f_inv * &tpz * tpz.transpose(); // rank-1 downdate

        let diff = (&p_new - &p).norm();
        if diff < tol * p.norm().max(1.0) {
            return p_new;
        }
        std::mem::swap(&mut p, &mut p_new);
    }
    p
}
```

비용: O(max_iter · k³), 일회성. max_iter=50이면 50·k³ ≈ 50·27³ = 983,000 ops.
비교: n=500 스텝 Kalman에서 비수렴 구간 ~35 스텝 × k³ = 35·19,683 = 688,905 ops.

→ DARE 사전계산 비용 ≈ 비수렴 구간 비용과 비슷. **단, Chandrasekhar과 결합 시** 비수렴 구간 자체가 사라지므로 순 이득.

### 구현 계획

**파일**: `src/initialization.rs`

```rust
pub enum KalmanInit {
    Diffuse { p0: DMatrix<f64> },
    Stationary { a0: DVector<f64>, p0: DMatrix<f64> },
    DareSteadyState { a0: DVector<f64>, p0: DMatrix<f64>, k_inf: DVector<f64>, f_inf: f64 },
}
```

**적용 조건**: `enforce_stationarity=true` + `enforce_invertibility=true`인 경우에만. 비정상 모델(d>0 또는 D>0)은 diffuse 초기화 유지하되, diffuse 구간이 끝난 후 DARE 해로 전환.

### 예상 효과

- 긴 시계열 (n > 1000): KF 수렴 대기 시간 제거 → **5-15% 전체 가속**
- Chandrasekhar과 결합 시: 초기화 즉시 steady-state → **추가 가속**

### 검증 기준

- [x] DARE 해와 반복 KF 수렴값의 차이 < 1e-10
- [x] loglike 차이 < 1e-8 (기존 초기화 대비)
- [ ] DARE 계산 시간 < 1ms (k=27), < 5ms (k=51)

---

## P4: 병렬 Multi-Start — Rayon 기반 재시작 병렬화

### 학술 근거

| 참고문헌 | 내용 |
|----------|------|
| **Wheeler, J. and Ionides, E. L. (2025)** — "Revisiting Inference for ARMA Models." *PLOS ONE*, 20(10). DOI: `10.1371/journal.pone.0333993` | `arima2` 패키지. 표준 `arima()`가 최대 **60%** 경우 global MLE 미도달 발견. Random restart로 해결 |
| **arima2 R 패키지** — [cran.r-project.org/package=arima2](https://cran.r-project.org/package=arima2), [github.com/jeswheel/arima2](https://github.com/jeswheel/arima2) | 코드 구현. Algorithm 1 (Uniform-Root), Algorithm 2 (DL sampling) |
| **Marti, K. (2003)** — *Stochastic Optimization Methods*. Springer. ISBN: 3-540-22272-3 | 다중 시작점 최적화 이론 |

### arima2 Algorithm 1: Uniform-Root Sampling

```
For each restart k = 1, ..., K:
  1. AR/MA 다항식 근 샘플링:
     - 실수 vs 복소수: P(real) = √(1/2) ≈ 0.707
     - 실수근: magnitude ~ U(γ, 1-γ), 부호 랜덤
     - 복소근: angle τ ~ U(0, π), radius r ~ U(γ, 1-γ)
       z₁ = r·cos(τ) + i·r·sin(τ), z₂ = conj(z₁)
  2. 근으로부터 AR/MA 계수 복원
  3. 근접취소 검사: min_{i,j} |z_i^AR - z_j^MA| < α → 거부
  4. 이 시작점에서 L-BFGS 실행
  5. loglike 개선 시 결과 갱신

  M번 연속 비개선 시 조기 종료
```

### 현재 구현 (순차)

`fit_multistart()` (optimizer.rs:931):
1. 초기 L-BFGS-B → 2. zero-start → 3. SMA grid → 4. numgrad → 5. LCG 교란 → 6. DL restart → 7. NM 정제

**모든 재시작이 순차 실행** — 각 재시작은 독립적이므로 병렬화 가능.

### 구현 계획

**파일**: `src/optimizer.rs`

```rust
use rayon::prelude::*;
use std::sync::Mutex;

fn fit_multistart_parallel(
    endog: &[f64],
    config: &SarimaxConfig,
    starts: Vec<Vec<f64>>,  // 사전 생성된 시작점들
    maxiter: u64,
    exog: Option<&[Vec<f64>]>,
) -> Option<(Vec<f64>, f64, bool, String)> {
    let best = Mutex::new(None::<(Vec<f64>, f64, bool, String)>);

    starts.par_iter().for_each(|start| {
        let objective = SarimaxObjective { /* clone per thread */ };
        let bounds = compute_bounds(config);
        if let Ok((params, cost, _n, conv)) = run_lbfgsb(&objective, start.clone(), bounds, maxiter) {
            let mut guard = best.lock().unwrap();
            match guard.as_ref() {
                Some((_, prev_cost, _, _)) if cost >= *prev_cost => {},
                _ => *guard = Some((params, cost, conv, "lbfgsb".into())),
            }
        }
    });

    best.into_inner().unwrap()
}
```

**설계 고려사항**:
- `batch_fit()` 내에서 호출 시 **Rayon 중첩** 발생 → batch 모드에서는 순차 유지
- 단일 series `fit()` 호출 시에만 병렬 multi-start 활성화
- 시작점 사전 생성: DL sampling + LCG + SMA grid → `Vec<Vec<f64>>`로 수집 후 `par_iter`

### 예상 효과

- 단일 series fit: restart 5-8개 × 코어 수 → **1.5-3x 가속**
- batch 모드: 이미 series 간 병렬화 되어있으므로 추가 효과 제한적
- SARIMA(1,1,1)(1,1,1,24) fit 시간: 1.0s → 0.3-0.5s

### 검증 기준

- [ ] 단일 series fit 시간 2x 이상 단축 (4+ 코어 머신)
- [x] 결과 동일 (같은 시작점 → 같은 결과, 순서 무관)
- [x] batch 모드에서 성능 회귀 없음

---

## P5: StateSpace 캐싱 — 반복 행렬 재구축 제거

### 학술 근거

| 참고문헌 | 내용 |
|----------|------|
| **Fulton, C. (2015)** — "Estimating Time Series Models by State Space Methods in Python: Statsmodels." [chadfulton.com/files/fulton_statsmodels_2017_v1.pdf](http://www.chadfulton.com/files/fulton_statsmodels_2017_v1.pdf) | statsmodels 상태공간 구현. 행렬 캐싱 전략 |
| **Durbin & Koopman (2012)** — Section 4.3 | 시불변 시스템의 행렬 재사용 |
| **statsmodels `_representation.pyx`** — [github.com/statsmodels/statsmodels](https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/statespace/_representation.pyx) | Cython 구현. 시간 차원=1이면 같은 행렬 재사용 |

### 현재 문제

`optimizer.rs`의 `eval_negloglike()` (line 205-207):
```rust
// 매 반복마다 전체 재구축
let ss = StateSpace::new(config, &sparams, None, None)?;
let init = KalmanInit::from_config(config, &ss)?;
let (ll, _scale) = kalman_loglike(endog, &ss, &init, ...)?;
```

`StateSpace::new()` 비용:
- T 행렬 (k×k) 할당 및 초기화: O(k²)
- Z 벡터 (k) 할당
- R 벡터 (k) 할당
- polynomial 확장 (reduced_ar, reduced_ma): O(p·P·s)
- 총 비용: O(k²) ≈ 729 (k=27), 2601 (k=51)

MLE 반복 횟수: ~50-200회 → 재구축 비용: ~50×729 = 36,450 ops.
비교: KF 1회 비용 ~200μs → 총 KF 비용 50×200μs = 10ms.
→ 재구축은 전체의 ~5-10% 차지 (큰 k에서 더 중요).

### 구현 계획

**파일**: `src/state_space.rs`, `src/optimizer.rs`

**변경 1: 불변 행렬 분리**

```rust
pub struct StateSpaceInvariant {
    pub z: DVector<f64>,     // 관측 벡터 (파라미터 불변)
    pub k: usize,
}

pub struct StateSpaceMutable {
    pub t: DMatrix<f64>,     // 전이 행렬 (AR 계수 의존)
    pub r: DVector<f64>,     // 선택 벡터 (MA 계수 의존)
    pub q: DMatrix<f64>,     // 상태 잡음 (σ² 의존)
}

impl StateSpaceMutable {
    /// AR/MA 계수가 변경된 항목만 업데이트 (전체 재구축 대신)
    pub fn update_from_params(&mut self, params: &SarimaxParams, config: &SarimaxConfig) {
        // T 행렬: companion matrix의 첫 행만 업데이트
        let ar_coeffs = reduced_ar(params, &config.order);
        for (j, &c) in ar_coeffs[1..].iter().enumerate() {
            self.t[(0, j)] = -c;  // companion matrix 첫 행
        }
        // R 벡터: MA 계수 업데이트
        let ma_coeffs = reduced_ma(params, &config.order);
        for (j, &c) in ma_coeffs.iter().enumerate() {
            self.r[j] = c;
        }
    }
}
```

**변경 2: Objective에 캐시 추가**

```rust
struct SarimaxObjective {
    endog: Vec<f64>,
    config: SarimaxConfig,
    invariant: StateSpaceInvariant,      // 한 번 구축
    mutable: RefCell<StateSpaceMutable>, // 매 반복 업데이트
}
```

### 예상 효과

- 행렬 할당 제거: k=27에서 ~5-10% 오버헤드 절감
- k=51에서: ~8-12% 오버헤드 절감
- polynomial 재계산도 제거 시 추가 절감

### 검증 기준

- [x] loglike 수치 차이 0 (bit-exact)
- [ ] 벤치마크에서 fit 시간 5%+ 개선

---

## P6: 근접취소 감지 — AR/MA 다항식 근 검사

### 학술 근거

| 참고문헌 | 내용 |
|----------|------|
| **Hamilton, J. D. (1994)** — *Time Series Analysis*. Princeton Univ. Press. Proposition 4.1 | ARMA 식별가능성: AR/MA 다항식이 **서로소**(coprime)일 때에만 식별 가능 |
| **McLeod, A. I. (1993)** — "A Note on ARMA Model Parameter Redundancy." *J. Time Series Analysis*, 14(2), 207-208. DOI: `10.1111/j.1467-9892.1993.tb00138.x` | Fisher 정보행렬 특이 ↔ 공통근 존재 |
| **Cogley, T. and Startz, R. (2019)** — "Robust Estimation of ARMA Models with Near Root Cancellation." *Advances in Econometrics*, 40A, 133-155. | 근접취소 시 추론 편향 분석 + 해결 방안 |
| **Brockwell & Davis (1991)** — *Time Series: Theory and Methods*, 2nd ed. Springer. Proposition 12.2.3 | 공식 coprimality 조건 |
| **Ionides, E. L. (2020)** — "Parameter Estimation and Model Identification for ARMA Models." U. Michigan Lecture Notes, Chapter 5. | 우도 곡면이 비볼록 + ridge를 형성하는 메커니즘 설명 |
| **arima2 소스코드** — [rdrr.io/cran/arima2/src/R/arima.R](https://rdrr.io/cran/arima2/src/R/arima.R) | 실제 구현: inverted root 거리 검사 |

### arima2의 감지 알고리즘

```r
# R code from arima2
inv_ma_roots <- 1 / polyroot(c(1, tmp_ma_pars))
inv_ar_roots <- 1 / polyroot(c(1, -tmp_ar_pars))
inv_root_dist <- min(Mod(outer(inv_ar_roots, inv_ma_roots, FUN = '-')))
valid_test <- valid_test && (inv_root_dist > min_inv_root_dist)
# default min_inv_root_dist = 0 (비활성), 권장값: 0.01-0.05
```

### 다항식 근 계산

Companion matrix의 고유값을 이용:

```
AR 다항식: 1 - φ₁z - φ₂z² - ... - φ_p·z^p
Companion matrix:
  C_AR = [φ₁  φ₂  ... φ_p]
         [1    0   ... 0  ]
         [0    1   ... 0  ]
         [...            ]
         [0    0   ... 0  ]

AR 근 = 1 / eigenvalues(C_AR)
```

nalgebra로 구현:
```rust
use nalgebra::DMatrix;

fn polynomial_roots(coeffs: &[f64]) -> Vec<(f64, f64)> {
    // coeffs = [φ₁, φ₂, ..., φ_p]
    let p = coeffs.len();
    if p == 0 { return vec![]; }

    let mut companion = DMatrix::zeros(p, p);
    for i in 0..p {
        companion[(0, i)] = coeffs[i];
    }
    for i in 1..p {
        companion[(i, i-1)] = 1.0;
    }

    // 고유값 계산 (복소수)
    let schur = companion.schur();
    // Schur 분해에서 고유값 추출
    // 반환: Vec<(real, imag)>
    extract_eigenvalues(&schur)
}

fn min_root_distance(ar_coeffs: &[f64], ma_coeffs: &[f64]) -> f64 {
    let ar_roots = polynomial_roots(ar_coeffs);
    let ma_roots = polynomial_roots(ma_coeffs);

    let mut min_dist = f64::INFINITY;
    for &(ar_re, ar_im) in &ar_roots {
        for &(ma_re, ma_im) in &ma_roots {
            let dist = ((ar_re - ma_re).powi(2) + (ar_im - ma_im).powi(2)).sqrt();
            min_dist = min_dist.min(dist);
        }
    }
    min_dist
}
```

### 적용 지점

1. **Random restart 시 시작점 검증** (α = 0.01): 근접취소 시작점 거부
2. **최적화 후 결과 검증** (α = 0.05): 경고 로그 출력
3. **저차 warm-start**: 근접취소 감지 시 → p-1, q-1 모델 결과를 시작점으로 사용

```rust
// optimizer.rs의 fit_multistart() 내부
fn validate_no_near_cancellation(params: &SarimaxParams, config: &SarimaxConfig, threshold: f64) -> bool {
    if config.order.p == 0 || config.order.q == 0 {
        return true; // AR-only 또는 MA-only → 취소 불가
    }
    let dist = min_root_distance(&params.ar_coeffs, &params.ma_coeffs);
    dist > threshold
}
```

### 예상 효과

- 고차 혼합 ARMA(p,q) p+q ≥ 4: WARN 발생률 감소
- 불필요한 고차 모델에 대한 사용자 경고 제공
- Random restart 효율성 향상 (무효 시작점 조기 제거)

### 검증 기준

- [x] ARMA(2,2) 근접취소 시뮬레이션에서 경고 정상 출력
- [ ] Random restart에서 α=0.01 필터링 적용 후 수렴률 개선
- [x] 계산 비용: O(p³+q³) < 1μs (무시 가능)

---

## 구현 우선순위 및 로드맵

### Phase A: 수렴 개선 (FAIL/WARN 해결)

| 순서 | 작업 | 파일 | 예상 공수 |
|------|------|------|-----------|
| A-1 ✓ | CSS 확대 + KF 검증 (P2) | optimizer.rs | 1-2시간 |
| A-2 ✓ | 근접취소 감지 (P6) | optimizer.rs, 신규 | 2-3시간 |

### Phase B: Kalman Filter 가속

| 순서 | 작업 | 파일 | 예상 공수 |
|------|------|------|-----------|
| B-1 | Chandrasekhar recursion (P1) | kalman.rs | 4-6시간 |
| B-2 ✓ | DARE 초기화 (P3) | initialization.rs | 2-3시간 |

### Phase C: Optimizer 가속

| 순서 | 작업 | 파일 | 예상 공수 |
|------|------|------|-----------|
| C-1 ✓ | StateSpace 캐싱 (P5) | state_space.rs, optimizer.rs | 2-3시간 |
| C-2 ✓ | 병렬 Multi-Start (P4) | optimizer.rs | 2-3시간 |

### 목표 지표

| 지표 | 현재 | Phase A 후 | Phase B+C 후 |
|------|------|-----------|-------------|
| EXACT+CLOSE 비율 | 79.8% | 85%+ | 85%+ |
| FAIL 비율 | 6.1% | < 3% | < 3% |
| WARN 비율 | 3.5% | < 2% | < 2% |
| 속도 배율 vs statsmodels | 2.28x | 2.3x | **4-8x** |
| SARIMA s=12 fit 시간 | 210ms | 210ms | **50-100ms** |
| SARIMA s=24 fit 시간 | 1.0s | 1.0s | **0.2-0.5s** |

---

## 부록: 전체 차수 4 테스트 결과 (114개 모델)

### Part 1: Non-seasonal ARIMA (72개)

- d ∈ {0,1,2}, p ∈ {0..4}, q ∈ {0..4} (p+q > 0)
- 순수 AR/MA: 전체 EXACT
- 혼합 ARMA p+q ≤ 3: 대부분 EXACT/CLOSE
- 혼합 ARMA p+q ≥ 4: WARN 발생 (다중 극값)

### Part 2: Seasonal SARIMA (42개)

- s ∈ {12, 24}, 비계절 p,q ∈ {1..4}, 계절 P,Q ∈ {1..2}
- s=12, p+q ≤ 3: 대부분 EXACT
- s=24, p+q ≤ 3: EXACT/CLOSE
- s=24, p+q ≥ 4: FAIL (SARIMA(1,1,4)(1,1,1,24) 등)

### FAIL/WARN 상세

| 모델 | 등급 | LL diff | 원인 |
|------|------|---------|------|
| ARIMA(4,0,2) | WARN | 5.2 | AR/MA 다중 극값 |
| ARIMA(2,1,2) | WARN | 5.8 | seed 특정, AR/MA 상쇄 |
| ARIMA(4,1,4) | WARN | 7.1 | 8 파라미터, 극한 고차 |
| ARIMA(4,2,3) | WARN | 6.3 | d=2 + 고차 혼합 |
| SARIMA(1,1,1)(2,1,1,12) | FAIL | 0.001 | converged=False, LL 근접 |
| SARIMA(1,1,1)(2,1,2,12) | FAIL | 0.001 | converged=False, LL 근접 |
| SARIMA(1,1,4)(1,1,1,24) | FAIL | 1.36 | q=4 + s=24 |
| SARIMA(3,1,3)(1,1,1,24) | FAIL | 6.0 | 고차 + 대형 계절 |
