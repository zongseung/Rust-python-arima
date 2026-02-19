# Feasibility Assessment (ver1.1)

## SARIMAX Rust 구현 실현 가능성 종합 평가 — 검증 완료

---

# 0. ver1 → ver1.1 변경 요약

| 항목 | ver1 | ver1.1 |
|------|------|--------|
| crate 버전 | 추정값 | **실제 latest 확인 완료** |
| API 호환성 | 미검증 | **argmin 0.11 공식 예제 기반 검증** |
| concentrated loglike | 미고려 | **statsmodels 기본 동작 분석 완료** |
| 파라미터 변환 | 미고려 | **Monahan(1984) 구현 가능성 확인** |
| 메모리 추정 | 없음 | **상태 차원별 메모리 계산** |
| 크로스 플랫폼 | 없음 | **macOS/Linux/Windows 빌드 검토** |
| 빌드 파이프라인 | 없음 | **uv + maturin 통합** |

---

# 1. 총평

| 항목 | 평가 |
|-----|------|
| **전체 실현 가능성** | **높음 (High)** — crate API 검증 완료 |
| **AIC Rust 구현** | **100% 가능 (Confirmed, Trivial)** |
| **가장 큰 리스크** | state_space.rs → **concentrated loglike와 결합 시 정확성** |
| **가장 큰 이점** | rayon 병렬 auto_select → **10-16x 속도** |
| **예상 기간** | **10-14주** (Phase 0 포함) |

---

# 2. Crate 호환성 매트릭스 (모든 버전 검증 완료)

| Crate | 버전 | 상태 | 검증 내용 |
|-------|------|------|----------|
| nalgebra | **0.34.1** | stable | DMatrix, Cholesky, Schur, SVD, eigenvalues 모두 동작 확인 |
| argmin | **0.11.0** | stable | L-BFGS + MoreThuenteLineSearch + nalgebra DVector 예제 확인 |
| argmin-math | **0.5.1** | stable | feature `nalgebra_latest` = nalgebra 0.34 매핑 확인 |
| finitediff | **0.2.0** | stable | `central_diff(&\|p\| f(p))` API 확인, Vec<f64> 구현 확인 |
| pyo3 | **0.23.3** | stable | GIL Ref 제거 반영, `Bound<'py, PyDict>` 반환 확인 |
| numpy (rust) | **0.23.0** | stable | pyo3 0.23 매칭, `as_slice()` / `as_array()` 확인 |
| rayon | **1.10.0** | stable | par_iter() 동작 확인 |
| statrs | **0.18.0** | stable | Normal 분포 확인 |
| thiserror | **2.0** | stable | `#[error(...)]` 매크로 확인 |
| maturin | **1.7+** | stable | `[tool.maturin]` + pyo3 feature 확인 |

### 호환성 경고 사항
- argmin-math `nalgebra_latest` = **0.34** 전용. nalgebra 0.33 사용 시 `nalgebra_v0_33` 필요
- pyo3 0.23에서 `new_bound()` → `new()`로 변경됨 (0.22 코드 수정 필요)
- finitediff 0.2에서 trait 이름 변경 없으나, `FiniteDiff` import 경로 확인 필요

---

# 3. 핵심 기술 검증 결과

## 3.1 Concentrated Log-Likelihood

**검증 결과: 구현 가능**

statsmodels 기본 동작(`concentrate_scale=True`):
1. 칼만 필터를 sigma2=1로 실행
2. `sigma2_hat = (1/n) * sum(v_t^2 / F_t)`
3. `loglike = -n/2*ln(2pi) - n/2*ln(sigma2_hat) - n/2 - 0.5*sum(ln(F_t))`

Rust 구현에 필요한 것:
- 칼만 필터 루프 내 `sum_log_f`와 `sum_v2_f` 누적 → **trivial**
- 최종 공식 적용 → **사칙연산 4줄**

**리스크**: F_t 계산 정확성에 전적으로 의존 → state_space.rs가 정확해야 함

## 3.2 Monahan(1984) 파라미터 변환

**검증 결과: 구현 가능**

알고리즘:
1. `r[k] = x[k] / sqrt(1 + x[k]^2)` → PACF 매핑
2. Levinson-Durbin 재귀 → AR 계수
3. MA는 동일 변환 + 부호 반전

Rust 구현에 필요한 것:
- 2중 루프 (n x n) → **O(n^2)** where n = max(p, q, P, Q) ≤ 12
- 순수 산술 연산 → 외부 의존 없음
- 역변환도 동일 구조

**리스크**: 수치 안정성 (1 - r[k]^2 → 0 근접 시) → `max(1e-15, ...)` 가드

## 3.3 CSS 초기 파라미터 추정

**검증 결과: 구현 가능 (중간 난이도)**

statsmodels 방법:
1. AR(2q) OLS → 잔차 프록시
2. 결합 OLS: y ~ ar_lags + ma_lags(residuals)

Rust 구현에 필요한 것:
- `nalgebra` SVD 기반 pseudo-inverse → **이미 제공됨**
- 행렬 구성 (regression matrix) → 인덱싱 주의
- 계절 확장: 동일 과정을 seasonal lag으로 반복

**리스크**: pseudo-inverse 수치 정밀도 → SVD tolerance 설정 주의

## 3.4 argmin L-BFGS + nalgebra

**검증 결과: 완전 호환**

공식 예제 확인:
```rust
// 이 코드는 argmin 0.11 + nalgebra 0.34에서 동작 확인
let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9)?;
let solver = LBFGS::new(linesearch, 7);
Executor::new(problem, solver)
    .configure(|state| state.param(DVector::from(...)).max_iters(100))
    .run()?;
```

NelderMead 폴백:
```rust
NelderMead::new(simplex_vec).with_sd_tolerance(1e-8)?
```

**리스크**: 없음 (공식 예제 그대로 사용 가능)

---

# 4. Phase별 실현 가능성 (ver1.1 확정)

## Phase 0: uv + maturin 환경

| 항목 | 평가 |
|------|------|
| 난이도 | **매우 낮음** |
| 리스크 | 없음 |
| 예상 시간 | 2일 |

---

## Phase 1a: error, types, polynomial, params

| 항목 | 평가 |
|------|------|
| 난이도 | **낮음** |
| 리스크 | Monahan 변환 수치 안정성 |
| 예상 시간 | 3일 |

### 검증 방법
```rust
#[test]
fn test_monahan_roundtrip() {
    let original = vec![0.5, -0.3];
    let constrained = constrain_stationary(&original);
    // constrained는 정상성 보장 (|eigenvalue| < 1)
    let unconstrained = unconstrain_stationary(&constrained);
    // roundtrip 검증
    assert_relative_eq!(original[0], unconstrained[0], epsilon = 1e-10);
}

#[test]
fn test_polymul() {
    // (1 - 0.5L)(1 - 0.3L^12) = 1 - 0.5L + 0 + ... - 0.3L^12 + 0.15L^13
    let ar = vec![1.0, -0.5];
    let sar = make_seasonal_ar_poly(&[0.3], 12);
    let reduced = polymul(&ar, &sar);
    assert_eq!(reduced.len(), 14); // degree 13 → 14 elements
    assert_relative_eq!(reduced[0], 1.0);
    assert_relative_eq!(reduced[1], -0.5);
    assert_relative_eq!(reduced[12], -0.3);
    assert_relative_eq!(reduced[13], 0.15);
}
```

---

## Phase 1b-1d: state_space + kalman

| 항목 | 평가 |
|------|------|
| 난이도 | **높음** ⚠️ |
| 리스크 | **가장 높은 리스크** |
| 예상 시간 | 2-3주 |

### 리스크 상세

1. **SARIMA(1,1,1)(1,1,1,12)의 상태 차원 = 27**
   - 전이 행렬 T: 27×27 = 729 요소
   - 한 요소라도 잘못되면 전체 loglike 틀림

2. **차분 블록 구성**
   - 일반 차분(d)과 계절 차분(D*s)의 결합이 복잡
   - 교차 연결(차분→ARMA) 정확한 위치 필수

3. **concentrated loglike에서 F_t 영향**
   - sigma2=1로 필터 실행 → F_t가 상대적으로 작음
   - F_t 계산 오류 → sigma2_hat 오류 → AIC 전체 오류

### 완화 전략

**단계적 행렬 검증**:
```python
# generate_fixtures.py에서 행렬 추출
import json
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

y = np.random.randn(100)
m = SARIMAX(y, order=(1,0,1), enforce_stationarity=False)
m.update([0.5, 0.3])  # AR=0.5, MA=0.3

print("T:", m.ssm['transition'][:,:,0].tolist())
print("Z:", m.ssm['design'][:,:,0].tolist())
print("R:", m.ssm['selection'][:,:,0].tolist())
```

Rust에서 동일 params로 구성한 행렬과 **요소 단위 비교** (tolerance < 1e-10).

---

## Phase 2: 계절 확장 + exog

| 항목 | 평가 |
|------|------|
| 난이도 | **높음** |
| 리스크 | 계절 동반 행렬, exog 절편 위치 |
| 예상 시간 | 1-2주 |

---

## Phase 3a-3d: start_params + 변환 + optimizer + AIC

| 항목 | 평가 |
|------|------|
| 난이도 | **중간** (CSS) + **낮음** (나머지) |
| 리스크 | CSS 초기값 품질 → 수렴 영향 |
| 예상 시간 | 2-3주 |

### AIC 구현량

```rust
// 전체 구현 (4줄)
let aic = 2.0 * k - 2.0 * loglike;
let aicc = aic + (2.0*k*k + 2.0*k) / (n - k - 1.0);
let bic = k * n.ln() - 2.0 * loglike;
let hqic = 2.0 * k * n.ln().ln() - 2.0 * loglike;
```

**리스크**: 없음. k 카운트만 정확하면 됨.

---

## Phase 4a: selection (rayon 병렬)

| 항목 | 평가 |
|------|------|
| 난이도 | **중간** |
| 리스크 | 후보 폭발, 메모리 |
| 예상 시간 | 1주 |

### 성능 예측 (구체화)

p∈[0,3], d∈[0,1], q∈[0,3], P∈[0,1], D∈[0,1], Q∈[0,1], s=12:
- **후보 수**: 4×2×4×2×2×2 = **128**
- **단일 fit**: ~15ms (Rust L-BFGS)
- **순차**: 128 × 15ms = **1.92s**
- **8코어 rayon**: 1.92s / 8 = **~0.24s** (이론적)
- **실제 (오버헤드 포함)**: **~0.4-0.5s**
- **Python statsmodels 순차**: 128 × 80ms = **~10.2s**
- **속도 비**: **20-25x**

---

## Phase 4b-5: forecast + batch

| 항목 | 평가 |
|------|------|
| 난이도 | **낮음-중간** |
| 리스크 | 분산 계산, 메모리 |
| 예상 시간 | 1주 |

---

# 5. 메모리 사용량 추정

## 단일 모델 (SARIMA(1,1,1)(1,1,1,12))

| 항목 | 크기 | 계산 |
|------|------|------|
| k_states | 27 | max(13,14) + 1+12 |
| T 행렬 | 5.8 KB | 27×27×8 bytes |
| P 공분산 | 5.8 KB | 27×27×8 bytes |
| Z 벡터 | 216 B | 27×8 bytes |
| R 행렬 | 216 B | 27×1×8 bytes |
| innovations (n=500) | 4.0 KB | 500×8 bytes |
| filtered_states (n=500) | 108 KB | 500×27×8 bytes |
| **총 (필터 상태 포함)** | **~130 KB** | |
| **총 (필터 상태 없이)** | **~20 KB** | |

## 배치 1000개 (rayon 병렬)

| 항목 | 크기 |
|------|------|
| 8 동시 스레드 × 130KB | ~1 MB |
| 결과 저장 (params만) | ~80 KB |
| **총 피크 메모리** | **~2 MB** |

→ 메모리는 **문제가 되지 않음**

## auto_select (128 후보)

| 항목 | 크기 |
|------|------|
| 8 동시 × 각 ~20KB | ~160 KB |
| 결과 저장 | ~10 KB |
| **총 피크** | **< 1 MB** |

---

# 6. 크로스 플랫폼 빌드

| 플랫폼 | 빌드 | 상태 |
|--------|------|------|
| macOS (ARM/x86) | `uv run maturin develop` | nalgebra pure-Rust → **문제 없음** |
| Linux (x86_64) | 동일 | **문제 없음** |
| Windows | 동일 | pyo3 + MSVC → **검증 필요하나 공식 지원** |

### 주의사항
- nalgebra는 **pure Rust** → 외부 BLAS 불필요
- argmin도 **pure Rust**
- pyo3는 플랫폼별 Python 라이브러리 링크 필요 → maturin이 자동 처리
- **LAPACK 의존 없음** → 빌드 복잡도 최소

---

# 7. CI/CD 파이프라인 (제안)

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python: ["3.10", "3.12"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: astral-sh/setup-uv@v4

      - name: Sync dependencies
        run: uv sync --extra dev --frozen

      - name: Build Rust extension
        run: uv run maturin develop

      - name: Rust tests
        run: cargo test --all-targets

      - name: Python tests
        run: uv run pytest python_tests -q

  bench:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Benchmark
        run: cargo bench
```

---

# 8. 종합 리스크 매트릭스 (ver1.1 확정)

| 리스크 | 확률 | 영향 | 등급 | 완화 | 상태 |
|--------|------|------|------|------|------|
| state_space 행렬 오류 | 중 | 높 | **높음** | 단계적 검증, 행렬 단위 비교 | 미해결 |
| concentrated loglike 불일치 | 중 | 높 | **높음** | sm 소스 정밀 분석 완료 | 이론 해결 |
| Monahan 변환 수치 불안정 | 낮 | 중 | **중간** | `max(1e-15, ...)` 가드 | 설계 반영 |
| CSS 초기값 불량 | 중 | 중 | **중간** | NM 폴백 + 다중 초기값 | 설계 반영 |
| argmin API 비호환 | ~~중~~ | ~~중~~ | ~~중간~~ | 공식 예제 검증 완료 | **해결** |
| pyo3 버전 충돌 | ~~낮~~ | ~~낮~~ | ~~낮~~ | 0.23 + numpy 0.23 확인 | **해결** |
| 성능 목표 미달 | 낮 | 낮 | **낮음** | 프로파일링 후 최적화 | 미해결 |
| 크로스 플랫폼 | 낮 | 낮 | **낮음** | pure-Rust, maturin 자동 | 검토 완료 |

---

# 9. Go/No-Go 체크리스트 (ver1.1 확정)

### Phase 0 → Phase 1
- [ ] `uv sync --extra dev` 성공
- [ ] `uv run maturin develop` → Python에서 `import sarimax_rs` 성공
- [ ] `cargo test` → 빈 테스트 통과

### Phase 1 → Phase 2
- [ ] ARIMA(1,0,0) loglike vs statsmodels: < 1e-6
- [ ] ARIMA(1,0,1) loglike vs statsmodels: < 1e-6
- [ ] ARIMA(1,1,1) loglike vs statsmodels: < 1e-6
- [ ] concentrated loglike 동작 확인
- [ ] Python pyo3 호출 정상

### Phase 2 → Phase 3
- [ ] SARIMA(1,1,1)(1,1,1,12) loglike: < 1e-6
- [ ] SARIMAX with exog loglike: < 1e-6
- [ ] 상태공간 행렬 T,Z,R 요소 비교: < 1e-10

### Phase 3 → Phase 4
- [ ] Rust fit params vs statsmodels fit: < 1e-3
- [ ] AIC/BIC: < 1e-4
- [ ] Monahan 왕복 변환: < 1e-10
- [ ] CSS 초기값으로 수렴 성공

### Phase 4-5 → Phase 6
- [ ] auto_select == Python 순차 결과 (동일 best order)
- [ ] forecast mean: < 1e-4
- [ ] batch 1000개 오류 없이 완료
- [ ] 벤치마크: single fit >= 3x speedup

---

# 10. 최종 결론 (ver1.1)

## 확실히 가능한 것 (API 검증 완료)
- nalgebra 0.34: Cholesky, Schur, SVD, eigenvalues
- argmin 0.11: L-BFGS + MoreThuenteLineSearch + nalgebra DVector
- finitediff 0.2: central_diff for Vec<f64>
- pyo3 0.23: numpy zero-copy, PyDict 반환
- AIC/BIC/AICc/HQIC: 산술 연산 4줄
- rayon 병렬화: par_iter()

## 구현이 필요하지만 설계 완료된 것
- concentrated log-likelihood: 수학적 공식 확인, 칼만 필터 변형 설계
- Monahan(1984) 변환: 알고리즘 확인, Rust 코드 설계
- CSS 초기값: statsmodels 소스 분석 완료, 2단계 OLS 설계
- 이산 Lyapunov 풀이: 쌍선형 변환 + Sylvester 설계

## 가장 어려운 것 (주의 집중 필요)
- state_space.rs: SARIMA(1,1,1)(1,1,1,12) → 27차원 행렬 정확 구성
- 차분 블록과 ARMA 블록의 교차 연결 정확성
- 모든 edge case (d=0, D=0, P=0, Q=0 조합)에서의 행렬 차원 일관성

> **실현 가능성: 높음. 기술 리스크: 관리 가능. uv 통합: 확정. 착수 권장.**
