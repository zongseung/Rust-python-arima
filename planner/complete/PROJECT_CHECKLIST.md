# SARIMAX Rust 엔진 — 전체 진행 체크리스트

> 최종 갱신: 2026-02-19 (Phase 1b 계절 SARIMA 완료 시점)

---

## Phase 0: 프로젝트 스캐폴딩 ✅ 완료

- [x] `Cargo.toml` 구성 (nalgebra, pyo3, numpy, argmin, serde 등)
- [x] `pyproject.toml` 구성 (maturin 빌드, dev 의존성)
- [x] `src/error.rs` — thiserror 기반 에러 타입 정의
- [x] `src/types.rs` — SarimaxOrder, Trend, SarimaxConfig, FitResult
- [x] `src/params.rs` — SarimaxParams (from_flat/to_flat), Monahan 변환
- [x] `src/polynomial.rs` — polymul, make_ar/ma_poly, reduced_ar/ma
- [x] `python_tests/test_smoke.py` — import 및 version 검증
- [x] `cargo test` → 28개 테스트 전부 통과

---

## Phase 1: 비계절 ARIMA Kalman Log-Likelihood ✅ 완료

### 빌드 환경
- [x] `uv sync --extra dev` 정상 동작
- [x] `uv run maturin develop` → Python wheel 빌드 성공
- [x] `import sarimax_rs; sarimax_rs.version()` Python에서 확인

### statsmodels 기준값
- [x] `python_tests/generate_fixtures.py` 작성
- [x] AR(1), ARMA(1,1), ARIMA(1,1,1) 기준값 추출
- [x] `tests/fixtures/statsmodels_reference.json` 생성

### 상태공간 (`src/state_space.rs`)
- [x] `StateSpace` 구조체 정의 (T, Z, R, Q, obs_intercept, state_intercept)
- [x] `build_transition()` — 차분 블록 + ARMA 동반 행렬 + 연결
- [x] `build_design()` — Z 벡터 구성
- [x] `build_selection()` — R 행렬 (reduced_ma 기반)
- [x] `build_state_cov()` — Q 행렬 (concentrated vs 비concentrated)
- [x] T, Z, R 행렬 vs statsmodels fixture 요소별 비교 (< 1e-10)

### 초기화 (`src/initialization.rs`)
- [x] `KalmanInit::approximate_diffuse()` 구현
- [x] a₀ = 0, P₀ = κI (κ=1e6)
- [x] `loglikelihood_burn = k_states` (statsmodels와 동일)

### 칼만 필터 (`src/kalman.rs`)
- [x] `kalman_loglike()` 함수 구현
- [x] Observe → Update → Predict 순서 (Harvey-form)
- [x] Joseph form 공분산 갱신
- [x] Concentrated log-likelihood 공식 구현
- [x] F_t ≤ 0 가드, σ²_hat 안전 처리

### PyO3 바인딩 (`src/lib.rs`)
- [x] `sarimax_loglike()` pyfunction 등록
- [x] numpy 배열 입력 (PyReadonlyArray1)
- [x] order, seasonal 튜플 파라미터
- [x] concentrate_scale 기본값 true

### Go/No-Go 검증
- [x] ARIMA(1,0,0) loglike vs statsmodels: **5.68e-14** (기준 < 1e-6)
- [x] ARIMA(1,0,1) loglike vs statsmodels: **5.02e-10** (기준 < 1e-6)
- [x] ARIMA(1,1,1) loglike vs statsmodels: **2.97e-09** (기준 < 1e-6)

### 테스트
- [x] `cargo test --all-targets` → **54개** 전부 통과
- [x] `pytest python_tests` → **6개** 전부 통과

---

## Phase 1b: 계절 SARIMA 확장 ✅ 완료

- [x] `state_space.rs` 계절 차분 블록 (D > 0, 순환 이동 행렬)
- [x] 계절 AR/MA 다항식 → 축약 다항식 통합 (reduced_ar/reduced_ma via polymul)
- [x] SARIMA(1,0,0)(1,0,0,4) T, Z, R 행렬 검증 (vs statsmodels < 1e-10)
- [x] SARIMA(1,1,1)(1,1,1,12) 전체 검증 (k_states=27, T/Z/R 일치)
- [x] 계절 모델 loglike vs statsmodels < 1e-6
  - SARIMA(1,0,0)(1,0,0,4): **0.00** (완벽 일치)
  - SARIMA(1,1,1)(1,1,1,12): **2.51e-09**
- [x] 계절 모델 Python 통합 테스트 (sarimax_loglike에 sar/sma 파라미터 파싱 추가)
- [x] `build_transition()` 5-블록 알고리즘: 일반차분 + 계절순환이동 + 교차차분 + diff→ARMA + ARMA동반
- [x] `build_design()` 계절 레이어별 Z 벡터 구성

---

## Phase 2: 최적화 (Fit) ⬜ 미착수

- [ ] 초기 파라미터 추정 (Hannan-Rissanen 또는 CSS)
- [ ] L-BFGS 최적화 (argmin 크레이트 활용)
- [ ] Nelder-Mead 폴백
- [ ] enforce_stationarity / enforce_invertibility 변환
- [ ] AIC/BIC 계산
- [ ] `sarimax_rs.fit()` Python API
- [ ] fit 결과 vs statsmodels 비교

---

## Phase 3: 예측 및 진단 ⬜ 미착수

- [ ] Forecast (h-step ahead 예측 평균)
- [ ] Forecast 분산 / 신뢰구간
- [ ] Residual 계산
- [ ] 정보 행렬 (Hessian 기반 표준오차)
- [ ] `sarimax_rs.forecast()` Python API
- [ ] forecast vs statsmodels 비교

---

## Phase 4: 배치 병렬 처리 ⬜ 미착수

- [ ] Rayon 기반 멀티스레드 배치 loglike
- [ ] Rayon 기반 배치 fit
- [ ] `sarimax_rs.batch_loglike()` Python API
- [ ] `sarimax_rs.batch_fit()` Python API
- [ ] 1000개 시계열 배치 벤치마크 (목표: 3배 이상 speedup)

---

## Phase P-1: Python Orchestration Layer ⬜ 미착수

- [ ] `python/sarimax_py/` 패키지 구현
- [ ] statsmodels 호환 API (model.fit(), model.forecast())
- [ ] 데이터 전처리 (결측치, 정규화)
- [ ] 결과 리포팅 (summary 테이블)

---

## 현재 상태 요약

```
Phase 0   ████████████████████  100%  ✅ 완료
Phase 1   ████████████████████  100%  ✅ 완료
Phase 1b  ████████████████████  100%  ✅ 완료
Phase 2   ░░░░░░░░░░░░░░░░░░░░    0%  ⬜ 미착수
Phase 3   ░░░░░░░░░░░░░░░░░░░░    0%  ⬜ 미착수
Phase 4   ░░░░░░░░░░░░░░░░░░░░    0%  ⬜ 미착수
Phase P-1 ░░░░░░░░░░░░░░░░░░░░    0%  ⬜ 미착수
```

### 지금 할 수 있는 것
- **ARIMA(p,d,q)** 임의 차수의 concentrated log-likelihood 계산
- **SARIMA(p,d,q)(P,D,Q,s)** 계절 모델 log-likelihood 계산 (D ≤ 1)
- Python에서 `sarimax_rs.sarimax_loglike(y, order, seasonal, params)` 호출
- statsmodels와 수치적으로 동일한 결과 (오차 < 1e-9)
- 5개 모델 검증 완료: AR(1), ARMA(1,1), ARIMA(1,1,1), SARIMA(1,0,0)(1,0,0,4), SARIMA(1,1,1)(1,1,1,12)

### 아직 못 하는 것
- 자동 파라미터 추정 (fit)
- 예측 (forecast)
- 배치 병렬 처리
- D > 1 계절 차분 (현재 D ≤ 1만 지원)
- 외생변수 (exog) 지원
