# Phase 1 완료 보고서: SARIMAX Rust 수치 엔진 — 비계절 ARIMA Kalman Filter

## 1. 개요

Phase 1의 목표는 **비계절 ARIMA 모델의 concentrated log-likelihood를 Rust 칼만 필터로 계산**하고,
Python에서 PyO3를 통해 호출 가능하게 만드는 것이었다.

**Go/No-Go 기준:** statsmodels 대비 loglike 오차 < 1e-6

---

## 2. 구현 완료 내역

### 2.1 신규 생성 파일

| 파일 | 라인 수 | 설명 |
|------|---------|------|
| `src/state_space.rs` | ~260줄 | Harvey Representation 상태공간 구축 (T, Z, R, Q 행렬) |
| `src/initialization.rs` | ~60줄 | Approximate Diffuse 초기화 (P₀ = κI, burn = k_states) |
| `src/kalman.rs` | ~290줄 | Concentrated log-likelihood 칼만 필터 |
| `python_tests/generate_fixtures.py` | ~148줄 | statsmodels 기준값 생성 스크립트 |
| `python_tests/test_loglike.py` | ~55줄 | Python 통합 테스트 |
| `tests/fixtures/statsmodels_reference.json` | — | statsmodels 기준값 JSON |

### 2.2 수정 파일

| 파일 | 변경 내용 |
|------|-----------|
| `src/lib.rs` | 모듈 선언 추가 (`state_space`, `initialization`, `kalman`) + `sarimax_loglike` PyO3 함수 등록 |
| `python_tests/conftest.py` | `statsmodels_fixtures` pytest fixture 추가 |

---

## 3. 아키텍처 설명

### 3.1 상태공간 표현 (`state_space.rs`)

Harvey 표현법을 사용하여 ARIMA(p,d,q) 모델을 상태공간으로 변환한다.

```
상태 방정식: α_{t+1} = T · α_t + c_t + R · η_t    (η_t ~ N(0, Q))
관측 방정식: y_t     = Z' · α_t + d_t + ε_t       (ε_t ~ N(0, H), H=0)
```

**행렬 구성:**

- **T (전이 행렬):** k_states × k_states
  - 차분 블록 [0..d, 0..d]: 상삼각 1-행렬 (적분 연산자)
  - ARMA 동반 행렬 [d..d+k_order, d..d+k_order]: 첫 열 = AR 계수, 초대각 = 1
  - 차분→ARMA 연결: T[i, d] = 1 (i = 0..d)

- **Z (설계 벡터):** k_states × 1
  - Z[i] = 1 (i = 0..d), Z[d] = 1

- **R (선택 행렬):** k_states × 1
  - R[d] = 1, R[d+i] = reduced_ma[i] (i ≥ 1)

- **Q (상태 공분산):** 1 × 1
  - concentrate_scale=true: [[1.0]], 아니면 [[σ²]]

### 3.2 초기화 (`initialization.rs`)

Approximate Diffuse 초기화를 사용한다:
- `a₀ = 0` (영벡터)
- `P₀ = κ · I` (κ = 1e6, 단위 행렬 스케일링)
- `burn = k_states` (statsmodels와 동일: 확산 영향 관측치 건너뜀)

> **핵심 발견:** statsmodels는 `loglikelihood_burn = k_states`를 사용한다.
> `k_states_diff`가 아님. 이것은 approximate diffuse 초기화에서
> 처음 k_states개 관측치가 큰 F_t 값을 가지기 때문이다.

### 3.3 칼만 필터 (`kalman.rs`)

표준 Harvey-form 칼만 필터를 구현한다:

```
for t = 0, 1, ..., n-1:
    1. 혁신(Innovation):  v_t = y_t - Z'·a_{t|t-1} - d_t
    2. 혁신 분산:         F_t = Z'·P_{t|t-1}·Z
    3. 칼만 이득:         K_t = P_{t|t-1}·Z / F_t
    4. 상태 갱신:         a_{t|t} = a_{t|t-1} + K_t·v_t
    5. 공분산 갱신:       P_{t|t} = (I - K·Z')·P_{t|t-1}·(I - K·Z')'  [Joseph form]
    6. 예측:              a_{t+1|t} = T·a_{t|t} + c_t
    7. 예측 공분산:       P_{t+1|t} = T·P_{t|t}·T' + R·Q·R'
```

**Concentrated log-likelihood 공식 (σ² 집중):**
```
σ²_hat = (1/n_eff) · Σ(v_t² / F_t)     (t ≥ burn)
loglike = -n_eff/2·ln(2π) - n_eff/2·ln(σ²_hat) - n_eff/2 - 1/2·Σ(ln F_t)
```

**수치 안정성 조치:**
- F_t ≤ 0 일 때 갱신 건너뜀
- σ²_hat에 대해 `max(1e-300)` 가드
- Joseph form 공분산 갱신 (양정치성 보장)

### 3.4 PyO3 바인딩 (`lib.rs`)

```python
sarimax_rs.sarimax_loglike(
    y,                    # np.ndarray: 관측 시계열
    order=(p, d, q),      # ARIMA 차수
    seasonal=(P, D, Q, s),# 계절 차수 (Phase 1에서는 (0,0,0,0))
    params,               # np.ndarray: [ar..., ma...] 매개변수 벡터
    exog=None,            # 외생 변수 (미구현)
    concentrate_scale=True# σ² 집중 여부
) -> float               # log-likelihood 값
```

---

## 4. Phase 0에서 재사용한 모듈

| 모듈 | 재사용 내용 |
|------|-------------|
| `types.rs:SarimaxOrder` | `k_states()`, `k_order()`, `k_states_diff()` 차원 계산 |
| `types.rs:SarimaxConfig` | 모델 설정 전달 |
| `params.rs:SarimaxParams` | 매개변수 언패킹 구조체 |
| `polynomial.rs:reduced_ar/ma` | 축약 AR/MA 다항식 → state_space 행렬 구축에 사용 |
| `error.rs:SarimaxError` | Result 체인 에러 처리 |

---

## 5. 테스트 결과

### 5.1 Rust 단위 테스트: 54개 전부 통과

```
cargo test --all-targets
→ test result: ok. 54 passed; 0 failed
```

**모듈별 테스트 수:**
| 모듈 | 테스트 수 | 검증 내용 |
|------|-----------|-----------|
| `types` | 7개 | k_states 차원 계산, Trend 파싱 |
| `params` | 10개 | from_flat/to_flat 왕복, Monahan 변환 |
| `polynomial` | 10개 | polymul, AR/MA/계절 다항식 |
| `state_space` | 15개 | T, Z, R 행렬 요소별 검증 (< 1e-10) |
| `initialization` | 3개 | P₀ 차원/값, burn 값 |
| `kalman` | 9개 | loglike/scale vs statsmodels (< 1e-6) |

### 5.2 Python 통합 테스트: 6개 전부 통과

```
pytest python_tests -v
→ 6 passed
```

| 테스트 | 내용 |
|--------|------|
| `test_ar1_loglike` | AR(1) loglike vs statsmodels |
| `test_arma11_loglike` | ARMA(1,1) loglike vs statsmodels |
| `test_arima111_loglike` | ARIMA(1,1,1) loglike vs statsmodels |
| `test_concentrate_scale_default` | concentrate_scale=True 기본값 확인 |
| `test_import` | 모듈 import 확인 |
| `test_version` | 버전 문자열 확인 |

---

## 6. Go/No-Go 기준 달성 현황

### 6.1 Loglike 정밀도 검증

| 모델 | Rust 결과 | statsmodels 기준 | 오차 | 판정 |
|------|-----------|------------------|------|------|
| ARIMA(1,0,0) | -267.1922806999 | -267.1922806999 | 5.68e-14 | PASS |
| ARIMA(1,0,1) | -266.8745027325 | -266.8745027320 | 5.02e-10 | PASS |
| ARIMA(1,1,1) | -429.1345495569 | -429.1345495599 | 2.97e-09 | PASS |

> 모든 모델이 1e-6 기준을 10^3~10^8 배 여유로 통과

### 6.2 체크리스트

- [x] ARIMA(1,0,0) loglike vs statsmodels: < 1e-6 ✓ (5.68e-14)
- [x] ARIMA(1,0,1) loglike vs statsmodels: < 1e-6 ✓ (5.02e-10)
- [x] ARIMA(1,1,1) loglike vs statsmodels: < 1e-6 ✓ (2.97e-09)
- [x] Concentrated loglike 동작 확인 ✓
- [x] Python PyO3 호출 정상 ✓
- [x] `cargo test --all-targets` 54개 전부 통과 ✓
- [x] `pytest python_tests` 6개 전부 통과 ✓

---

## 7. 구현 중 발견된 핵심 사항

### 7.1 칼만 필터 순서: Observe → Update → Predict

초기 구현에서는 `Predict → Observe → Update` 순서를 사용했으나, 이는 초기 상태 `a_{0|-1}`에
불필요한 T 변환을 적용하여 오차를 발생시켰다. 올바른 순서는:

```
a = a₀ (초기 상태 = 이미 예측된 상태)
for t:
    1. Observe: v_t = y_t - Z'·a      ← a를 그대로 사용
    2. Update:  a_filtered = a + K·v_t
    3. Predict: a = T·a_filtered + c_t ← 다음 시점 예측
```

### 7.2 Burn-in = k_states (k_states_diff 아님)

statsmodels는 approximate diffuse 초기화 시 `loglikelihood_burn = k_states`를 사용한다.
처음에 `k_states_diff`로 설정했을 때 AR(1)에서 7.3 정도의 오차가 발생했다.
이는 확산 초기화(P₀ = 1e6·I)로 인해 처음 k_states개 관측치의 F_t가 매우 크기 때문이다.

### 7.3 uv + maturin 빌드 이슈

`uv run`은 실행 전 자동으로 `uv sync`를 수행하여 패키지를 재설치한다.
이때 maturin으로 설치한 최신 빌드가 덮어씌워지는 문제가 있었다.

**해결 방법:**
```bash
# 1. 별도 target dir로 wheel 빌드
CARGO_TARGET_DIR=target_wheel uv run maturin build --out /tmp/wheels

# 2. wheel 직접 설치
uv pip install --force-reinstall /tmp/wheels/sarimax_rs-*.whl

# 3. venv python으로 직접 테스트
.venv/bin/python -m pytest python_tests -v
```

---

## 8. 다음 단계 (Phase 2 이후)

| Phase | 내용 | 상태 |
|-------|------|------|
| Phase 0 | 프로젝트 스캐폴딩, types/params/polynomial | ✅ 완료 |
| **Phase 1** | **비계절 ARIMA Kalman loglike** | **✅ 완료** |
| Phase 1b | 계절 SARIMA 확장 (P, D, Q > 0) | 🔜 다음 |
| Phase 2 | 최적화 (L-BFGS, 초기값 추정) | ⬜ 대기 |
| Phase 3 | 예측(forecast), 정보 행렬, 진단 | ⬜ 대기 |
| Phase 4 | 배치 병렬 처리 (rayon) | ⬜ 대기 |
| Phase P-1 | Python orchestration layer | ⬜ 대기 |

---

## 9. 파일 구조 (Phase 1 완료 시점)

```
sarimax_rs/
├── Cargo.toml
├── pyproject.toml
├── src/
│   ├── lib.rs              ← PyO3 모듈 정의 + sarimax_loglike 함수
│   ├── error.rs            ← thiserror 기반 에러 타입 (Phase 0)
│   ├── types.rs            ← SarimaxOrder, SarimaxConfig 등 (Phase 0)
│   ├── params.rs           ← SarimaxParams, Monahan 변환 (Phase 0)
│   ├── polynomial.rs       ← polymul, reduced_ar/ma (Phase 0)
│   ├── state_space.rs      ← Harvey Representation [NEW]
│   ├── initialization.rs   ← Approximate Diffuse 초기화 [NEW]
│   └── kalman.rs           ← Concentrated Kalman loglike [NEW]
├── tests/
│   └── fixtures/
│       └── statsmodels_reference.json  [NEW]
├── python_tests/
│   ├── conftest.py         ← pytest fixtures [MODIFIED]
│   ├── test_smoke.py       ← Phase 0 smoke tests
│   ├── test_loglike.py     ← Phase 1 통합 테스트 [NEW]
│   └── generate_fixtures.py ← statsmodels 기준값 생성 [NEW]
└── python/
    └── sarimax_py/
        └── __init__.py     ← Phase P-1에서 구현 예정
```
