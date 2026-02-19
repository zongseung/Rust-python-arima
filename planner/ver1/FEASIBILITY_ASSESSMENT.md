# Feasibility Assessment (ver1)

## SARIMAX Rust 구현 실현 가능성 종합 평가

---

# 1. 총평

| 항목 | 평가 |
|-----|------|
| **전체 실현 가능성** | **높음 (High)** |
| **AIC Rust 구현** | **확실히 가능 (Confirmed)** |
| **가장 큰 리스크** | state_space.rs (상태공간 변환 정확성) |
| **가장 큰 이점** | rayon 병렬 모델 선택 (10x+ 속도) |
| **예상 기간** | 8-11주 (풀타임 기준) |

---

# 2. AIC Rust 구현 가능성 — 상세 분석

## 결론: **100% 구현 가능**

### 수학적 관점
```
AIC  = 2k - 2 * ln(L)
AICc = AIC + (2k² + 2k) / (n - k - 1)
BIC  = k * ln(n) - 2 * ln(L)
```

- 모두 **단순 사칙연산** → 어떤 언어에서든 구현 가능
- 핵심은 `ln(L)` (로그우도) 계산 → 칼만 필터로 수행
- 칼만 필터는 **행렬 연산의 연속** → nalgebra로 완벽 지원

### Rust 생태계 관점

| 필요 기능 | Crate | 성숙도 | 평가 |
|----------|-------|--------|------|
| 행렬 연산 (T, Z, Q, R) | nalgebra 0.33 | 매우 높음 | Cholesky, LU, SVD 등 완비 |
| Cholesky 분해 | nalgebra | 매우 높음 | 수치 안정성 핵심 |
| L-BFGS 최적화 | argmin 0.11 | 높음 | statsmodels 기본 옵티마이저와 동일 |
| Nelder-Mead (폴백) | argmin | 높음 | 미분 불필요 |
| 유한차분 그래디언트 | finitediff 0.1 | 중간 | 초기 구현에 충분 |
| 병렬 처리 | rayon 1.10 | 매우 높음 | par_iter()로 간단 병렬화 |
| 통계 분포 | statrs 0.18 | 중간 | Normal 분포 등 기본 제공 |
| Python 바인딩 | pyo3 0.22 | 매우 높음 | numpy zero-copy 지원 |

### 기존 Rust ARIMA 생태계

| 프로젝트 | 상태 | 한계 |
|---------|------|------|
| arima crate (v0.3) | 기본 ARIMA만 | 계절성 없음, exog 없음, AIC 없음 |
| SARIMAX Medium 구현 | 교육용 | 프로덕션 불가 |

> **결론: 기존 crate로는 불가 → 직접 구현 필요하지만, 빌딩 블록은 모두 존재**

---

# 3. Phase별 실현 가능성 평가

## Phase 1a: params.rs (파라미터 관리)

| 항목 | 평가 |
|-----|------|
| 난이도 | **낮음** |
| 리스크 | statsmodels 파라미터 레이아웃 정확한 복제 |
| 실현 가능성 | **확실** |
| 비고 | 순수 데이터 구조, 외부 의존 없음 |

### 구체적 작업
- SarimaxOrder 구조체 정의
- 1D 벡터 ↔ 구조화된 파라미터 변환
- statsmodels 파라미터 순서 매칭: `[exog | ar | ma | sar | sma | sigma2]`

### 예상 시간: 2-3일

---

## Phase 1b: kalman.rs (칼만 필터)

| 항목 | 평가 |
|-----|------|
| 난이도 | **중간** |
| 리스크 | 수치 안정성, 초기 상태 설정 |
| 실현 가능성 | **높음** |
| 비고 | 알고리즘 자체는 교과서적, nalgebra로 구현 가능 |

### 구체적 작업
- 예측-업데이트 루프 구현
- 로그우도 누적 계산
- Joseph form 공분산 업데이트
- Approximate diffuse initialization

### 핵심 리스크
1. **F_t가 음수/영이 되는 경우** → 로그 계산 불가 → 가드 필요
2. **공분산 행렬 양정치성 상실** → Joseph form으로 완화
3. **초기 상태 설정** → statsmodels와 정확히 맞추려면 exact diffuse 필요 (Phase 2)

### 예상 시간: 1주

---

## Phase 1c-2: state_space.rs (상태공간 변환)

| 항목 | 평가 |
|-----|------|
| 난이도 | **높음** ⚠️ |
| 리스크 | 가장 큰 리스크 영역 |
| 실현 가능성 | **가능하지만 주의 필요** |
| 비고 | statsmodels ~2000줄의 로직 재현 필요 |

### 구체적 작업
- AR/MA 다항식 곱 계산 (비계절 × 계절)
- 상태 전이 행렬 T 구성
- 관측 벡터 Z 구성
- 선택 행렬 R, 노이즈 공분산 Q 구성
- 상태 차원 결정 로직

### 핵심 리스크
1. **다항식 곱 계산 오류** → 계수 하나라도 틀리면 전체 로그우도 틀림
2. **상태 차원 계산** → SARIMAX의 상태 차원은 복잡한 공식으로 결정
3. **exogenous 변수 통합 방식** → 관측 방정식 vs 상태 방정식 선택

### 완화 전략
- **단계적 구현**: ARIMA(p,0,q) → ARIMA(p,d,q) → SARIMA → SARIMAX
- **단위별 검증**: 각 행렬을 statsmodels 출력과 1:1 비교
- **참조 구현**: statsmodels `representation.py` 라인별 대조

### 예상 시간: 2-3주

---

## Phase 3a: optimizer.rs (MLE 최적화)

| 항목 | 평가 |
|-----|------|
| 난이도 | **중간** |
| 리스크 | 수렴 실패, 초기값 민감성 |
| 실현 가능성 | **높음** |
| 비고 | argmin이 핵심 enabler |

### argmin 활용 가능성 검증

```
argmin 0.11 기능 목록:
✅ L-BFGS (statsmodels 기본과 동일)
✅ BFGS
✅ Nelder-Mead (미분 불필요 폴백)
✅ Newton-CG
✅ Trust Region
✅ More-Thuente line search
✅ nalgebra 통합 (argmin-math)
✅ 체크포인팅
✅ Observer 패턴 (수렴 모니터링)
```

### 핵심 리스크
1. **초기 파라미터 선택** → CSS/HR 방법으로 합리적 초기값 추정 필요
2. **로컬 미니마** → 여러 초기값에서 시도
3. **그래디언트 정확도** → finitediff는 O(k) 추가 연산, 해석적 그래디언트가 이상적이지만 구현 복잡

### 예상 시간: 1주

---

## Phase 3b: information.rs (AIC/BIC)

| 항목 | 평가 |
|-----|------|
| 난이도 | **매우 낮음** |
| 리스크 | 거의 없음 |
| 실현 가능성 | **확실** |
| 비고 | 산술 연산 4줄 |

### 구현량
```rust
// 전체 핵심 로직
let aic = 2.0 * k - 2.0 * loglike;
let aicc = aic + (2.0 * k*k + 2.0 * k) / (n - k - 1.0);
let bic = k * n.ln() - 2.0 * loglike;
let hqic = 2.0 * k * n.ln().ln() - 2.0 * loglike;
```

### 예상 시간: 반나절

---

## Phase 3c: selection.rs (자동 모델 선택)

| 항목 | 평가 |
|-----|------|
| 난이도 | **중간** |
| 리스크 | 후보 폭발 (combinatorial), 수렴 실패 처리 |
| 실현 가능성 | **높음** |
| 비고 | rayon par_iter로 간단 병렬화 |

### 후보 수 추정
p∈[0,3], d∈[0,1], q∈[0,3], P∈[0,1], D∈[0,1], Q∈[0,1], s=12
= 4 × 2 × 4 × 2 × 2 × 2 = **128 후보**

8코어 rayon 병렬 시 16배치 → 각 ~50ms = **~0.4초** (vs Python 순차 ~6.4초)

### 예상 시간: 1주

---

## Phase 4: forecast.rs (예측)

| 항목 | 평가 |
|-----|------|
| 난이도 | **중간-낮음** |
| 리스크 | 분산 계산 정확성 |
| 실현 가능성 | **높음** |
| 비고 | 칼만 필터 최종 상태에서 전이 반복 |

### 예상 시간: 1주

---

## Phase 5: batch.rs (배치 처리)

| 항목 | 평가 |
|-----|------|
| 난이도 | **낮음** |
| 리스크 | 메모리 관리 |
| 실현 가능성 | **확실** |
| 비고 | rayon par_iter 래핑, 에러 핸들링 |

### 예상 시간: 3-5일

---

# 4. 종합 리스크 매트릭스

| 리스크 | 확률 | 영향 | 등급 | 완화 전략 |
|--------|------|------|------|----------|
| 상태공간 변환 오류 | 중 | 높음 | **높음** | 단계적 검증, 행렬 단위 비교 |
| 수치 불안정성 | 중 | 중 | **중간** | Joseph form, Cholesky, 가드 조건 |
| statsmodels 동일성 미달 | 중 | 중 | **중간** | 초기화 방식 정확 복제, 허용 오차 명확화 |
| 최적화 수렴 실패 | 낮 | 중 | **낮음** | CSS 초기값, Nelder-Mead 폴백 |
| 성능 목표 미달 | 낮 | 낮 | **낮음** | 프로파일링 후 핫스팟 최적화 |
| pyo3 호환성 이슈 | 낮 | 낮 | **낮음** | 공식 문서 및 커뮤니티 지원 |

---

# 5. Python vs Rust 역할 분배 (ver1 최종)

```
                      Phase 1          Phase 3+         Phase 5 (목표)
                    ──────────       ──────────       ──────────
전처리               Python           Python           Python
상태공간 변환         Rust             Rust             Rust
칼만 필터            Rust             Rust             Rust
로그우도             Rust             Rust             Rust
MLE 최적화           Python(scipy)    Rust(argmin)     Rust(argmin)
AIC/BIC 계산         Python           Rust             Rust
모델 선택            Python(순차)     Python(순차)      Rust(rayon 병렬)
배치 처리            Python(순차)     Python(순차)      Rust(rayon 병렬)
예측                 Python           Rust             Rust
결과 출력            Python           Python           Python
```

### Phase별 Rust 커버리지
- **Phase 1**: ~30% (loglike만)
- **Phase 3**: ~70% (fit + AIC + forecast)
- **Phase 5**: ~90% (batch + auto-select)

---

# 6. 권장 실행 순서

```
[Phase 1a] params.rs
    ↓
[Phase 1b] kalman.rs + likelihood.rs
    ↓  (여기서 첫 번째 검증 — loglike vs statsmodels)
[Phase 1c] lib.rs pyo3 바인딩
    ↓
[Phase 2] state_space.rs 계절 확장
    ↓  (두 번째 검증 — 전체 SARIMAX loglike)
[Phase 3a] optimizer.rs (argmin)
    ↓
[Phase 3b] information.rs (AIC/BIC) ← 가장 쉬운 단계
    ↓
[Phase 3c] selection.rs (auto-select)
    ↓
[Phase 4] forecast.rs
    ↓
[Phase 5] batch.rs
    ↓
[Phase 6] 통합 테스트 + 벤치마크 + 문서
```

---

# 7. Go/No-Go 판단 기준

각 Phase 완료 시 다음 체크리스트로 판단:

### Phase 1 완료 시
- [ ] ARIMA(1,0,0) loglike가 statsmodels와 1e-6 이내 일치
- [ ] ARIMA(1,1,1) loglike 일치
- [ ] Python에서 pyo3로 호출 성공
- → **Go**: Phase 2 진행 / **No-Go**: 상태공간 변환 재검토

### Phase 2 완료 시
- [ ] SARIMAX(1,1,1)(1,1,1,12) loglike 일치
- [ ] exog 포함 모델 loglike 일치
- → **Go**: Phase 3 진행

### Phase 3 완료 시
- [ ] Rust fit 결과가 statsmodels fit과 파라미터 1e-4 이내 일치
- [ ] AIC/BIC 1e-4 이내 일치
- [ ] auto_select 결과가 Python grid search와 동일
- → **Go**: Phase 4-5 진행

---

# 8. 최종 결론

## 가능한 것
- SARIMAX 로그우도 Rust 구현
- AIC/BIC/AICc/HQIC Rust 구현
- L-BFGS 기반 MLE Rust 구현
- rayon 병렬 모델 선택
- pyo3 Python 바인딩

## 도전적인 것
- statsmodels와 수치적 완전 동일성 (초기화 방식 차이)
- 해석적 그래디언트 구현 (유한차분으로 대체 가능)
- 모든 edge case 처리 (비정상 시계열, 단위근 근접 등)

## 불필요한 것
- Python에서 AIC 계산 (Rust에서 직접 가능하므로)
- scipy 의존 (argmin으로 대체 가능)
- 별도 칼만 필터 crate (SARIMAX 특화 구현 필요)

> **실현 가능성: 높음. AIC Rust 구현: 확실. 착수 권장.**
