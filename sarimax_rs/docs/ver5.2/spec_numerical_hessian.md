# Ver5.2 Spec: 수치 Hessian 기반 통계 추론

> 작성일: 2026-02-24
> 상태: 설계 문서

---

## 1. 목표

SARIMAX 모델 적합 후 **표준오차(SE), z-통계량, p-value, 신뢰구간(CI)**을 제공한다.
statsmodels `cov_type='approx'`에 해당하는 **수치 Hessian** 방식을 Rust에서 구현하여,
기존 Python 구현 대비 10x+ 속도 향상을 달성한다.

## 2. 배경

### 2.1 현재 상태

| 항목 | 상태 |
|------|------|
| `FitResult` | params, loglike, AIC, BIC만 반환 |
| 표준오차 | Hessian 미구현 → SE 없음 |
| Python fallback | `model.py`에 `_compute_numerical_hessian()` 존재 (느림) |
| `score.rs` | 해석적 1차 gradient (tangent linear KF) 구현 완료 |

### 2.2 statsmodels 참조 구현

```python
# statsmodels/tsa/statespace/mlemodel.py
def _cov_params_approx(self):
    evaluated_hessian = self.nobs_effective * self.model.hessian(
        params=self.params, transformed=True, method='approx')
    (neg_cov, singular_values) = pinv_extended(evaluated_hessian)
    return -neg_cov
```

## 3. 수학적 정의

### 3.1 수치 Hessian (중앙차분)

unconstrained 파라미터 공간에서 loglike의 2차 미분을 근사한다.

**대각 원소:**
```
H_ii = (f(x + h_i·e_i) - 2·f(x) + f(x - h_i·e_i)) / h_i²
```

**비대각 원소:**
```
H_ij = (f(x + h_i·e_i + h_j·e_j) - f(x + h_i·e_i - h_j·e_j)
      - f(x - h_i·e_i + h_j·e_j) + f(x - h_i·e_i - h_j·e_j)) / (4·h_i·h_j)
```

**스텝 크기:**
```
h_i = max(1e-5, 1e-4 · max(1.0, |x_i|))
```

### 3.2 Chain Rule: unconstrained → constrained

Hessian은 unconstrained 공간에서 계산 후 chain rule로 변환한다.

```
J_ji = ∂constrained_j / ∂unconstrained_i    (수치 Jacobian, eps=1e-7)
H_constrained = J^T · H_unconstrained · J
```

### 3.3 추론 통계

```
정보행렬:  I = -H_constrained
공분산:    Cov = inv(I)        [singular → pinv fallback]
표준오차:  SE_i = sqrt(diag(Cov)_i)
z-통계량:  z_i = θ_i / SE_i
p-value:   p_i = 2·(1 - Φ(|z_i|))    [Φ: 표준정규 CDF]
신뢰구간:  CI = θ_i ± z_{1-α/2} · SE_i
```

## 4. 계산 비용

k개 파라미터에 대해:
- 기본점 1회: `f(x)`
- 대각 k개: 각 2회 → `2k`회
- 비대각 k(k-1)/2개: 각 4회 → `2k(k-1)`회
- **총합: 2k² + 1 회 loglike 평가**

| 모델 | k | loglike 평가 횟수 | 예상 시간 (n=500) |
|------|---|-------------------|-------------------|
| ARIMA(1,1,1) | 2 | 9 | ~0.4s |
| SARIMA(1,1,1)(1,1,1,12) | 4 | 33 | ~1.4s |
| SARIMA(2,1,2)(1,1,1,12) | 6 | 73 | ~3s |

## 5. API 설계

### 5.1 Rust 내부 API

```rust
// inference.rs

pub struct InferenceResult {
    pub method: String,
    pub cov_params: Vec<f64>,   // k×k row-major
    pub std_err: Vec<f64>,
    pub z_stat: Vec<f64>,
    pub p_value: Vec<f64>,
    pub ci_lower: Vec<f64>,
    pub ci_upper: Vec<f64>,
    pub n_params: usize,
    pub status: String,         // "ok" | "partial" | "failed"
    pub message: Option<String>,
}

pub fn numerical_hessian(
    endog: &[f64],
    config: &SarimaxConfig,
    constrained_params: &[f64],
    exog: Option<&[Vec<f64>]>,
) -> Result<DMatrix<f64>>;

pub fn compute_inference(
    endog: &[f64],
    config: &SarimaxConfig,
    constrained_params: &[f64],
    method: &str,       // "hessian" or "opg"
    alpha: f64,
    exog: Option<&[Vec<f64>]>,
) -> Result<InferenceResult>;
```

### 5.2 Python API

```python
# Low-level
result = sarimax_rs.sarimax_inference(
    y, order=(1,1,1), seasonal=(1,1,1,12),
    params=params_array, method="hessian", alpha=0.05
)
# → dict {method, std_err, z, p_value, ci_lower, ci_upper, cov_params, status}

# High-level (model.py)
fit_result = model.fit()
summary = fit_result.parameter_summary(inference="hessian", alpha=0.05)
```

## 6. 구현 세부

### 6.1 loglike 평가 헬퍼

기존 `optimizer.rs`의 `SarimaxObjective::eval_loglike()` 패턴 재사용:

```
unconstrained params
  → transform_params() → constrained
  → SarimaxParams::from_flat()
  → StateSpace::new()
  → KalmanInit::from_config()
  → kalman_loglike()
  → loglike value
```

### 6.2 Jacobian 계산

기존 `optimizer.rs::apply_transform_jacobian()` (line 349) 패턴을 확장하여
J 행렬 전체를 반환하는 버전 구현:

```rust
fn compute_transform_jacobian(
    unconstrained: &[f64],
    config: &SarimaxConfig,
) -> Result<DMatrix<f64>> {
    let k = unconstrained.len();
    let eps = 1e-7;
    let c_base = transform_params(unconstrained, config)?;
    let mut j = DMatrix::zeros(k, k);
    let mut u_pert = unconstrained.to_vec();
    for i in 0..k {
        let orig = u_pert[i];
        u_pert[i] = orig + eps;
        let c_pert = transform_params(&u_pert, config)?;
        u_pert[i] = orig;
        for row in 0..k {
            j[(row, i)] = (c_pert[row] - c_base[row]) / eps;
        }
    }
    Ok(j)
}
```

### 6.3 행렬 역행렬 (pinv fallback)

```rust
fn safe_inverse(mat: &DMatrix<f64>) -> (DMatrix<f64>, bool) {
    // 1. 먼저 LU 역행렬 시도
    if let Some(inv) = mat.clone().try_inverse() {
        return (inv, false);  // not singular
    }
    // 2. SVD pseudo-inverse fallback
    let svd = nalgebra::SVD::new(mat.clone(), true, true);
    let threshold = 1e-10 * mat.nrows() as f64;
    let pinv = svd.pseudo_inverse(threshold).unwrap_or_else(|_|
        DMatrix::zeros(mat.nrows(), mat.ncols())
    );
    (pinv, true)  // singular flag
}
```

## 7. 엣지 케이스

| 시나리오 | 처리 |
|----------|------|
| 단위근 근처 파라미터 | Monahan/Jones 변환 곡률 높음 → 스텝 적응 |
| 특이 Hessian | pinv fallback + `status="partial"` |
| 파라미터 0개 | 빈 결과 즉시 반환 |
| loglike = NaN 반환 | 해당 섭동점 건너뛰고 H_ij = NaN 설정 |
| concentrate_scale=false | sigma² 파라미터 포함하여 k+1 차원 Hessian |

## 8. 테스트 전략

### 8.1 Rust 단위 테스트

```
test_numerical_hessian_ar1
  - AR(1) φ=0.5 n=500에서 Hessian 대각 음수 확인 (정보행렬 양정치)
  - H_11 ≈ analytical Fisher information -n/(1-φ²) 허용 오차 10%

test_hessian_symmetry
  - |H_ij - H_ji| < 1e-8 확인

test_inference_std_err_positive
  - 수렴된 ARIMA(1,1,1) 적합 후 모든 SE > 0 확인

test_pvalue_range
  - 모든 p-value ∈ [0, 1] 확인

test_ci_covers_param
  - 95% CI가 파라미터 점추정치 포함 확인
```

### 8.2 Python 통합 테스트

```
test_rust_hessian_matches_python
  - Rust numerical_hessian vs 기존 Python _compute_numerical_hessian
  - SE 차이 < 5% (동일 알고리즘이므로 수치 오차만 존재)

test_hessian_vs_statsmodels
  - Rust SE vs statsmodels bse
  - 허용 오차: SE 상대 차이 < 50% (시작점 차이로 인한 국소최적 차이 고려)

test_inference_speed
  - Rust Hessian ≥ 5x faster than Python Hessian
```

## 9. 의존성

- `nalgebra` — DMatrix, SVD (기존)
- `statrs` — Normal CDF for p-values (기존 Cargo.toml에 이미 포함)
- 새 의존성 없음
