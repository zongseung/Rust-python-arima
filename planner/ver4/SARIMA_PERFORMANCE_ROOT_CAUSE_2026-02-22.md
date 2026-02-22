# SARIMA 성능 저하 원인 분석 (ver4, 2026-02-22)

## 1. 목적

질문: "왜 SARIMA에서 기대만큼 빠르게 처리되지 않는가?"  
범위: `sarimax_rs`의 단일 fit 경로(`sarimax_fit`) 기준 원인 분해.

---

## 2. 결론 요약

핵심 원인은 **Kalman filter 자체 속도 부족이 아니라 최적화 루프 비용**이다.

1. `kalman_loglike` 단독 속도는 빠름.
2. `method="lbfgsb"` 경로에서 `loglike + score(gradient)`를 반복 호출하며 비용이 커짐.
3. 고차 계절 SARIMA에서 `score.rs`의 파라미터별 행렬 연산 비용이 크게 증가.
4. Rust 시작값과 statsmodels 시작값 차이가 커서 반복 횟수/라인서치 비용이 증가.

---

## 3. 실측 결과 (로컬 실행)

### 3.1 모델별 fit 시간/반복 수

동일 데이터(README 벤치 생성기 기준), `maxiter=500`, `enforce_stationarity=True`, `enforce_invertibility=True`.

| 모델 | 엔진/방법 | 시간(ms) | 반복/평가 | converged |
|---|---|---:|---:|---|
| SARIMA(0,1,1)(0,1,1,12) | Rust `lbfgsb` | 255.8 | n_iter=61 | False |
| SARIMA(0,1,1)(0,1,1,12) | Rust `nelder-mead` | 24.7 | n_iter=73 | True |
| SARIMA(0,1,1)(0,1,1,12) | statsmodels(L-BFGS-B) | 106.4 | iterations=10 / fcalls=36 | warnflag=0 |
| SARIMA(1,1,1)(1,1,1,12) | Rust `lbfgsb` | 305.4 | n_iter=23 | False |
| SARIMA(1,1,1)(1,1,1,12) | Rust `nelder-mead` | 113.2 | n_iter=179 | True |
| SARIMA(1,1,1)(1,1,1,12) | statsmodels(L-BFGS-B) | 216.5 | iterations=13 / fcalls=75 | warnflag=0 |

관찰:

- `lbfgsb`만 느린 구간이 있고, 같은 모델에서 `nelder-mead`는 상대적으로 빠르다.
- 따라서 "Kalman 자체가 느리다"보다 "`lbfgsb`의 gradient 경로 비용/반복 구조" 영향이 큼.

### 3.2 `maxiter` 영향 확인 (`SARIMA(0,1,1)(0,1,1,12)`, Rust `lbfgsb`)

| maxiter | 시간(ms) | n_iter |
|---:|---:|---:|
| 1 | 4.7 | 2 |
| 5 | 20.7 | 6 |
| 10 | 41.2 | 12 |
| 20 | 82.7 | 25 |
| 50 | 206.4 | 51 |
| 100 | 252.8 | 61 |
| 200 | 251.7 | 61 |
| 500 | 252.3 | 61 |

관찰:

- `maxiter`를 늘릴수록 증가하다가 실제 종료점(약 n_iter 61) 이후 포화.
- 무한 반복 문제가 아니라, 해당 데이터/시작값에서 실제로 평가가 많이 필요한 상태.

### 3.3 시작값 영향 (statsmodels start_params 주입)

| 모델 | 시작값 | 시간(ms) | n_iter |
|---|---|---:|---:|
| SARIMA(0,1,1)(0,1,1,12) | Rust 기본 시작값 | 256.5 | 61 |
| SARIMA(0,1,1)(0,1,1,12) | statsmodels 시작값 주입 | 33.8 | 8 |
| SARIMA(1,1,1)(1,1,1,12) | Rust 기본 시작값 | 238.4 | 23 |
| SARIMA(1,1,1)(1,1,1,12) | statsmodels 시작값 주입 | 134.7 | 13 |

관찰:

- 시작값 품질이 반복 수/시간에 큰 영향을 준다.
- 특히 `SARIMA(0,1,1)(0,1,1,12)`에서 영향이 매우 큼.

### 3.4 시작값 벡터 차이 (예시)

- `SARIMA(0,1,1)(0,1,1,12)`  
  - statsmodels start: `[-0.093859, -0.774278]`  
  - rust start: `[-0.075402, -0.505953]`  
  - max|diff| = `0.2683`

- `SARIMA(1,1,1)(1,1,1,12)`  
  - statsmodels start: `[0.457696, -0.095128, -0.005211, -0.545498]`  
  - rust start: `[0.376126, -0.012110, -0.367586, -0.062735]`  
  - max|diff| = `0.4828`

---

## 4. 코드 기준 원인 분해

## 4.1 `lbfgsb`는 평가 시 gradient를 함께 계산

- `run_lbfgsb`는 `eval_negloglike_with_gradient`를 우선 사용한다.  
  - `sarimax_rs/src/optimizer.rs:627`
- 즉 1회 평가에 `kalman_loglike + score`가 포함된다.

## 4.2 `score.rs`가 고차 계절 모델에서 고비용

- tangent linear 상태를 파라미터 개수만큼 유지/전파:
  - `da: Vec<DVector<f64>>`, `dp: Vec<DMatrix<f64>>`
  - `sarimax_rs/src/score.rs:277`
  - `sarimax_rs/src/score.rs:278`
- 루프 내부에서 파라미터별 예측/업데이트 dense 연산 반복:
  - `sarimax_rs/src/score.rs:425`
  - `sarimax_rs/src/score.rs:464`
  - `sarimax_rs/src/score.rs:553`

의미:

- `k_states`와 `n_params`가 큰 SARIMA에서 gradient 비용이 급증한다.

## 4.3 반대로 `kalman.rs`는 최적화가 이미 적용됨

- sparse 경로 + steady-state 경로 존재:
  - `sarimax_rs/src/kalman.rs:231`
  - `sarimax_rs/src/kalman.rs:334`
- 단독 `sarimax_loglike` 실측도 sub-ms 수준으로 빠름.

따라서 병목은 Kalman core보다 optimizer/score 결합 경로.

## 4.4 시작값 품질 문제

- 시작값 계산: `compute_start_params`
  - `sarimax_rs/src/start_params.rs:361`
- AR/Burg + seasonal 추정 방식이 statsmodels start_params와 차이가 큼:
  - `sarimax_rs/src/start_params.rs:404`
  - `sarimax_rs/src/start_params.rs:419`
- 시작값 차이가 line-search 반복 증가로 직결.

## 4.5 상태 차원 자체 증가

- `k_states = k_order + k_states_diff`
  - `sarimax_rs/src/types.rs:47`
- 예: SARIMA(1,1,1)(1,1,1,12)는 `k_states=27` (테스트 주석 근거)
  - `sarimax_rs/src/types.rs:149`

---

## 5. 왜 "SARIMA만" 체감이 느려지는가

1. 계절 차수/주기 때문에 `k_states`가 커짐.
2. `lbfgsb` 경로는 gradient 기반이라 `score` 비용이 누적됨.
3. 시작값이 멀면 평가 횟수까지 늘어남.
4. 결과적으로 `빠른 Kalman` 이점이 `비싼 gradient 반복`에 상쇄됨.

---

## 6. 우선순위 개선안 (코드 수정 전 설계 관점)

1. 시작값 품질 개선 (최우선)  
   - 목표: 반복 수 즉시 감소.
   - 검증: statsmodels start 주입 대비 반복 수 격차 축소.

2. `lbfgsb`용 gradient 비용 절감  
   - `score.rs` 파라미터별 dense 연산 축소(희소화/재사용 강화).
   - 고차 계절 모델 전용 경량 경로 검토.

3. 운영 전략 분기  
   - 특정 SARIMA 패턴에서 `nelder-mead`/다른 방법이 더 빠를 수 있으므로 자동 선택 규칙 검토.

4. 성능 회귀 벤치 확장  
   - SARIMA(0,1,1)(0,1,1,12), SARIMA(1,1,1)(1,1,1,12)를 고정 리그레션 케이스로 추가.

---

## 7. 참고 파일

- 최적화 경로: `sarimax_rs/src/optimizer.rs`
- gradient 계산: `sarimax_rs/src/score.rs`
- Kalman core: `sarimax_rs/src/kalman.rs`
- 시작값 추정: `sarimax_rs/src/start_params.rs`
- 상태 차원 정의: `sarimax_rs/src/types.rs`
- 재현 벤치 스크립트: `sarimax_rs/python_tests/bench_readme.py`
