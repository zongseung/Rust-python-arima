# SARIMA 성능 개선 구현 명세서 (ver4, 2026-02-22)

**상태: ✅ 구현 완료 (2026-02-22)**

## 1. 목적

`SARIMA_PERFORMANCE_ROOT_CAUSE_2026-02-22.md` 분석 결과를 바탕으로,
SARIMA fit 속도를 statsmodels 수준 이하로 줄이기 위한 **구체적 코드 변경 명세**.

---

## 2. 핵심 원인 요약 (root cause에서)

| 원인 | 영향도 | 실측 근거 |
|---|---|---|
| 시작값 품질 열악 → 반복 수 증가 | **최대** | statsmodels 시작값 주입 시 SARIMA(0,1,1)(0,1,1,12) 256ms→34ms (7.6x) |
| `score.rs` gradient 비용 (O(n×np×k³)) | 높음 | `lbfgsb`만 느림, `nelder-mead`는 상대적으로 빠름 |
| 상태 차원 폭발 (k_states=27 등) | 중간 | SARIMA(1,1,1)(1,1,1,12)에서 체감 |

---

## 3. 개선 항목 (우선순위 순)

### 3.1 ✅ [P1] Hannan-Rissanen 시작값 추정

**현재 문제**: `start_params.rs`는 AR/MA/SAR/SMA를 **개별 추정**한다.
- AR: Burg → 품질 양호
- MA: AR 잔차의 innovation algorithm → 편향 존재
- SAR: seasonal autocovariance의 Yule-Walker → AR-MA 교호 효과 무시
- SMA: 잔차의 seasonal innovation → 부정확

**결과**: joint estimation이 아니므로 특히 계절 MA 추정치가 statsmodels와 크게 괴리.

**개선 알고리즘: Hannan-Rissanen (1982)**

1단계: 차분된 시계열에 고차 AR(K) 적합 (K = max(10, 3*(p+q+P*s+Q*s)))
   - Burg method로 안정적 추정
   - 목적: residual proxy ε̂_t 확보

2단계: OLS 회귀로 ARMA+seasonal 계수 동시 추정
   - 종속변수: y_t (차분된 시계열)
   - 독립변수 행렬 X의 열:
     - y_{t-1}, ..., y_{t-p}       (AR lags)
     - ε̂_{t-1}, ..., ε̂_{t-q}       (MA lags: 1단계 잔차)
     - y_{t-s}, ..., y_{t-P·s}     (seasonal AR lags)
     - ε̂_{t-s}, ..., ε̂_{t-Q·s}     (seasonal MA lags: 1단계 잔차)
   - 정규방정식: β = (X'X)⁻¹ X'y

3단계: 안정성 검증
   - |AR roots| < 1 확인, 위반 시 0.9× 축소
   - |MA roots| < 1 확인, 위반 시 0.9× 축소
   - 실패 시 현재 개별 추정으로 fallback

**구현 위치**: `sarimax_rs/src/start_params.rs`
- 새 함수: `hannan_rissanen(diffed, p, q, pp, qq, s) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)>`
- `compute_start_params()`에서 `hannan_rissanen()` 우선 호출, 실패 시 기존 로직 fallback

**예상 효과**:
- SARIMA(0,1,1)(0,1,1,12): 반복 수 61 → ~10 (시작값 품질에 의존)
- SARIMA(1,1,1)(1,1,1,12): 반복 수 23 → ~13

**검증 기준**:
- `compute_start_params()` 결과가 statsmodels `start_params`와 max|diff| < 0.15
- SARIMA fit 시간이 이전 대비 30%+ 감소

**✅ 구현 결과**:
- `hannan_rissanen()` 함수 추가 (start_params.rs ~160줄)
- `cholesky_solve()` 헬퍼 추가
- 적용 범위: `qq > 0 && s > 0` (계절 MA 포함 모델만)
- Ridge 정규화(λ=1e-8) + fallback 안전장치
- 5개 Rust 단위 테스트 추가
- SARIMA(0,1,1)(0,1,1,12): 256ms → 154ms (**40% 감소**)
- SARIMA(1,1,1)(1,1,1,12): 305ms → 261ms (**14% 감소**)

---

### 3.2 ✅ [P2] Score 정상 상태(Steady-State) 최적화

**현재 문제**: `score.rs`의 tangent linear Kalman filter는 **매 시점 전체 dP 업데이트**를 수행.
- `compute_dp_predict()`: T·dP·T' (k²×k gemm) × n_params × n_obs
- SARIMA(1,1,1)(1,1,1,12): k=27, np=4, n=200 → ~4×200 = 800회의 27×27 행렬 곱

**핵심 관찰**: `kalman.rs`에는 이미 steady-state 최적화가 있다.
- P가 수렴 후 pz, F, K를 frozen 처리 → O(k) per step
- 같은 논리가 dP에도 적용 가능: P가 수렴하면 dP도 수렴

**개선 알고리즘**:

1. P 수렴 시점 감지 (기존 kalman.rs의 pz convergence 로직 재사용)
2. 수렴 후:
   - `dp[i]` 동결 (더 이상 업데이트 안 함)
   - `dpz_buf[i]` 동결 (dP*Z가 상수)
   - `df_buf[i]` 동결 (Z'*dP*Z가 상수)
3. 수렴 후에도 계속 업데이트하는 것:
   - `da[i]`: 상태 미분 (v_t에 의존하므로 시변)
   - `dv_buf[i]`: innovation 미분 (da에 의존)
   - score 누적: `sum_v_dv`, `sum_v2f2_df`, `sum_inv_f_df`

**구현 위치**: `sarimax_rs/src/score.rs`
- `score()` 함수 내부에 `steady_state` 플래그 추가
- pz convergence 감지 로직 (kalman.rs에서 패턴 차용)
- 수렴 후 inner loop를 경량 경로로 분기

**예상 효과**:
- 긴 시계열(n=200+)에서 수렴 시점 이후 score 비용 60-80% 절감
- P 수렴 시점: 통상 t=30~50 (burn-in 이후)
- n=200 기준: 150/200 = 75% timestep에서 dP 연산 스킵

**검증 기준**:
- `test_score_*` 테스트 전체 통과 (analytical vs numerical gradient tolerance 유지)
- 성능 벤치에서 score 호출 시간 감소 확인

**✅ 구현 결과**:
- pz convergence 감지 (tolerance=1e-9, 3 consecutive, min 5 steps past burn-in)
- 수렴 후 dp/dpz_buf/df_buf 동결, da/dv_buf만 업데이트
- f_inv_steady, k_gain 캐싱
- 9개 score 테스트 전체 통과 (SARIMA(1,1,1)(1,1,1,12) 포함)

---

## 4. 구현 순서

1. ✅ `start_params.rs`: Hannan-Rissanen 추가 + `compute_start_params` 연동
2. ✅ `score.rs`: steady-state 경량 경로 추가
3. ✅ 테스트: Rust unit tests + Python integration tests 통과
4. ✅ 벤치마크: SARIMA fit 시간 비교

---

## 5. 비변경 항목

- `kalman.rs`: 이미 최적화됨, 변경 불필요
- `optimizer.rs`: 시작값 품질 개선으로 간접 개선됨, 직접 변경 불필요
- API surface: 외부 인터페이스 변경 없음

---

## 6. 위험 요소

| 위험 | 완화 | 결과 |
|---|---|---|
| Hannan-Rissanen OLS가 singular matrix → panic | Tikhonov 정규화(ridge) + fallback | ✅ 해결 (λ=1e-8 ridge + Option 반환) |
| score steady-state 조기 수렴 → gradient 오차 | tolerance 보수적 설정(1e-9), 최소 step 요구 | ✅ 해결 (모든 score 테스트 통과) |
| 기존 테스트 regression | 전체 테스트 스위트 통과 필수 | ✅ 해결 (114 Rust + 235 Python 통과) |
| HR이 비계절 ARIMA에서 품질 저하 | 적용 범위 제한 (qq>0 && s>0) | ✅ 해결 (계절 MA 모델만 적용) |

---

## 7. 벤치마크 결과 (구현 후)

### 7.1 비계절 모델 (모두 우수)

| 모델 | sarimax_rs | statsmodels | 배율 |
|---|---:|---:|---:|
| ARIMA(1,0,0) | 0.3ms | 34.7ms | **115.7x** |
| ARIMA(2,1,0) | 0.4ms | 28.3ms | **70.8x** |
| ARIMA(0,1,1) | 0.4ms | 30.1ms | **75.3x** |
| ARIMA(1,1,1) | 0.4ms | 30.6ms | **76.5x** |
| ARIMA(2,1,2) | 1.4ms | 73.3ms | **52.4x** |

### 7.2 계절 모델 (개선됨, 여전히 격차 존재)

| 모델 | sarimax_rs (이전) | sarimax_rs (이후) | statsmodels | 개선율 |
|---|---:|---:|---:|---:|
| SARIMA(0,1,1)(0,1,1,12) | 255.8ms | 154.0ms | 109.4ms | **40%↓** |
| SARIMA(1,1,1)(1,1,1,12) | 305.4ms | 261.2ms | 212.5ms | **14%↓** |

### 7.3 잔존 격차 분석

계절 SARIMA에서 여전히 statsmodels보다 느린 이유:
1. **score.rs gradient 절대 비용**: k_states=14~27에서 tangent linear 행렬 연산 비용이 여전히 지배적
2. **statsmodels는 Fortran BLAS 기반 gradient**: scipy의 finite-diff가 BLAS-optimized로 동작
3. **반복 수 격차 잔존**: HR 시작값이 statsmodels 시작값에 근접했지만 완전 동일하지 않음

### 7.4 전체 테스트 시간

- Python 테스트 스위트: **46s → 12.34s (3.7x 단축)**
- Rust 테스트: 114개 전체 통과
- Python 테스트: 235개 전체 통과
