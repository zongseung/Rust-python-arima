# Ver5.2 Spec: OPG (Outer Product of Gradients) 기반 통계 추론

> 작성일: 2026-02-24
> 상태: 설계 문서

---

## 1. 목표

**관측별 score 외적(OPG)**으로 정보행렬을 추정하여 표준오차/p-value를 제공한다.
수치 Hessian 대비 **O(k²)배 빠르다** (loglike 평가 2k²+1회 → tangent KF 1회).

## 2. 배경

### 2.1 OPG vs 수치 Hessian 비교

| 항목 | 수치 Hessian | OPG |
|------|-------------|-----|
| 비용 | O(k²·n·k_states²) | O(n·k·k_states²) |
| 정확도 | 높음 (2차 근사) | 적당 (1차 근사) |
| 적합 상황 | 파라미터 수 적을 때 (k≤6) | 파라미터 수 많을 때 (k>6) |
| 수치 안정성 | 스텝 크기 민감 | score 품질에 의존 |
| 참조 | statsmodels `cov_type='approx'` | statsmodels `cov_type='opg'` |

### 2.2 statsmodels 참조

```python
# statsmodels/tsa/statespace/mlemodel.py
def opg_information_matrix(self, params, ...):
    score_obs = self.score_obs(params, ...).transpose()  # (k, n_eff)
    return np.inner(score_obs, score_obs) / (self.nobs - self.ssm.loglikelihood_burn)
```

### 2.3 현재 sarimax_rs 상태

- `score.rs::score()` — 합산 score 벡터 (k차원) 반환. tangent linear KF 기반.
- `score_obs()` — **미구현**. 관측별 score 필요.

## 3. 수학적 정의

### 3.1 관측별 score

로그 우도의 관측별 기여:
```
ℓ_t(θ) = -0.5·log(F_t) - 0.5·v_t²/F_t     (non-concentrated)
```

concentrated scale (σ² 집중) 하에서:
```
ℓ_t(θ) = -0.5·log(F_t) - 0.5·v_t²/(σ²·F_t) + const
         여기서 σ² = (1/n_eff)·Σ v_t²/F_t
```

관측별 score (concentrated):
```
s_t[i] = ∂ℓ_t/∂θ_i
       = -(v_t/(σ²·F_t))·∂v_t/∂θ_i
         + (v_t²/(2·σ²·F_t²))·∂F_t/∂θ_i
         - (1/(2·F_t))·∂F_t/∂θ_i
```

여기서 `∂v_t/∂θ_i`, `∂F_t/∂θ_i`는 tangent linear KF에서 계산된다.

### 3.2 OPG 정보행렬

```
I_OPG = (1/n_eff) · Σ_{t=burn}^{n-1} s_t · s_t^T
```

이것은 k×k 양반정치행렬이다.

### 3.3 추론 통계

```
Cov = inv(n_eff · I_OPG)    [pinv fallback]
SE_i = sqrt(diag(Cov)_i)
z_i = θ_i / SE_i
p_i = 2·(1 - Φ(|z_i|))
CI = θ_i ± z_{1-α/2} · SE_i
```

## 4. 계산 비용

| 단계 | 비용 |
|------|------|
| tangent linear KF (1회) | O(n · k · k_states²) |
| 관측별 score 수집 | O(n_eff · k) |
| 외적 합산 | O(n_eff · k²) |
| 행렬 역행렬 | O(k³) |
| **총합** | **O(n · k · k_states²)** |

### ARIMA(1,1,1) 비교 (k=2, n=500)

| 방법 | KF 평가 횟수 | 예상 시간 |
|------|-------------|-----------|
| 수치 Hessian | 9회 | ~0.4s |
| OPG | 1회 (tangent KF) | ~0.05s |

### SARIMA(2,1,2)(1,1,1,12) 비교 (k=6, n=500)

| 방법 | KF 평가 횟수 | 예상 시간 |
|------|-------------|-----------|
| 수치 Hessian | 73회 | ~3s |
| OPG | 1회 (tangent KF) | ~0.05s |

## 5. 구현 설계

### 5.1 `score.rs` 변경: `score_obs()` 추가

**전략: 2-pass 접근**

concentrated scale에서 σ²는 모든 관측에 의존하므로 single-pass로 per-obs score를 계산할 수 없다.

**Pass 1 — tangent linear KF 실행:**

기존 `score()` 루프와 동일하되, 관측별 중간값을 수집한다:

```rust
struct ScoreObsIntermediate {
    v_f: f64,       // v_t / F_t
    v2_f2: f64,     // v_t² / F_t²
    f_inv: f64,     // 1 / F_t
    dv: Vec<f64>,   // ∂v_t/∂θ_i, len=k
    df: Vec<f64>,   // ∂F_t/∂θ_i, len=k
}
```

이 루프 끝에서 `σ² = sum(v²/F) / n_eff` 를 계산한다.

**Pass 2 — per-obs score 벡터 생성:**

```rust
for obs in &intermediates {
    for i in 0..k {
        score_t[i] = -obs.v_f * obs.dv[i] / sigma2
                     + 0.5 * obs.v2_f2 * obs.df[i] / sigma2
                     - 0.5 * obs.f_inv * obs.df[i];
    }
    result.push(score_t.clone());
}
```

### 5.2 코드 구조: 공유 내부 함수

기존 `score()` 600줄 루프를 직접 복제하지 않고, 내부 helper를 추출한다:

```rust
/// 내부: tangent linear KF 실행 + 관측별 중간값 수집
fn score_inner(
    endog: &[f64],
    ss: &StateSpace,
    init: &KalmanInit,
    config: &SarimaxConfig,
    params: &SarimaxParams,
    concentrate_scale: bool,
    exog: Option<&[Vec<f64>]>,
    collect_per_obs: bool,  // true면 관측별 수집, false면 합산만
) -> Result<ScoreInnerResult>

struct ScoreInnerResult {
    sum_score: Vec<f64>,              // 합산 score (항상 계산)
    per_obs: Option<Vec<Vec<f64>>>,   // collect_per_obs=true일 때만
    sigma2: f64,
}
```

기존 `score()`는 `score_inner(..., collect_per_obs=false)` 호출.
새 `score_obs()`는 `score_inner(..., collect_per_obs=true)` 호출.

### 5.3 `inference.rs`: OPG 계산

```rust
pub fn opg_information_matrix(
    endog: &[f64],
    config: &SarimaxConfig,
    constrained_params: &[f64],
    exog: Option<&[Vec<f64>]>,
) -> Result<DMatrix<f64>> {
    // 1. Build StateSpace + KalmanInit
    let params = SarimaxParams::from_flat(constrained_params, config)?;
    let ss = StateSpace::new(config, &params, endog, exog)?;
    let init = KalmanInit::from_config(config, &ss)?;

    // 2. Get per-observation scores
    let scores = score::score_obs(endog, &ss, &init, config, &params,
                                   config.concentrate_scale, exog)?;

    // 3. Compute outer product sum: I = (1/n_eff) * Σ s_t * s_t'
    let k = constrained_params.len();
    let n_eff = scores.len();
    let mut info = DMatrix::zeros(k, k);
    for s_t in &scores {
        for i in 0..k {
            for j in i..k {
                let val = s_t[i] * s_t[j];
                info[(i, j)] += val;
                if i != j {
                    info[(j, i)] += val;
                }
            }
        }
    }
    info /= n_eff as f64;

    Ok(info)
}
```

## 6. API

### 6.1 Python

```python
# method="opg" 지정
result = sarimax_rs.sarimax_inference(
    y, order=(1,1,1), seasonal=(1,1,1,12),
    params=params_array, method="opg", alpha=0.05
)

# High-level
summary = fit_result.parameter_summary(inference="opg")
```

### 6.2 Rust

```rust
// compute_inference에서 method="opg" 분기
let info = opg_information_matrix(endog, config, params, exog)?;
let result = inference_from_information(&info, params, alpha, "opg");
```

## 7. concentrated scale 처리 주의사항

### 7.1 문제

concentrated scale에서 σ²는 모든 관측의 함수:
```
σ²(θ) = (1/n_eff) · Σ_t v_t(θ)² / F_t(θ)
```

따라서 관측별 score에서 ∂σ²/∂θ 항이 등장한다.
이 항은 **모든 관측에 걸쳐 합산**되므로, 엄밀한 per-obs score는 아니다.

### 7.2 해법

statsmodels와 동일하게, σ²를 **고정 상수**로 취급하여 per-obs score를 계산한다:

```
s_t[i] ≈ -(v_t/(σ̂²·F_t))·∂v_t/∂θ_i + (v_t²/(2·σ̂²·F_t²))·∂F_t/∂θ_i - (1/(2·F_t))·∂F_t/∂θ_i
```

여기서 σ̂²는 MLE에서 추정된 고정값이다. 이 근사는 OPG의 표준 관행이다.

## 8. 엣지 케이스

| 시나리오 | 처리 |
|----------|------|
| 파라미터 0개 | 빈 결과 즉시 반환 |
| n_eff < k | rank-deficient → pinv + status="partial" |
| 모든 score ≈ 0 (최적점) | I_OPG ≈ 0 → singular → pinv |
| NaN score | 해당 관측 건너뛰기 |

## 9. 테스트

### 9.1 Rust 단위 테스트

```
test_score_obs_sum_equals_score
  - score_obs() 합산 == score() 결과 (수치 오차 < 1e-10)

test_opg_positive_semidefinite
  - I_OPG 고유값 ≥ 0 확인

test_opg_vs_hessian_large_n
  - n=5000에서 OPG SE와 Hessian SE 상대 차이 < 30%
  (점근적으로 동치이므로 큰 n에서 수렴)

test_opg_speed_vs_hessian
  - OPG가 수치 Hessian 대비 5x+ 빠른지 확인
```

### 9.2 Python 통합 테스트

```
test_opg_vs_statsmodels
  - Rust OPG SE vs statsmodels cov_type='opg' SE
  - 허용 오차: 상대 차이 < 50%

test_opg_inference_api
  - parameter_summary(inference="opg") 정상 동작 확인
  - 모든 SE > 0, 모든 p ∈ [0,1]
```

## 10. 의존성

- `score.rs` 기존 tangent linear KF 인프라 재사용
- `nalgebra` DMatrix 외적 계산
- `statrs` Normal CDF (p-value)
- 새 의존성 없음
