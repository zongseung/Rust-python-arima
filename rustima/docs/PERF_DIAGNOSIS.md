# sarimax-rs 고차 모델 성능 진단 보고서

**일자**: 2026-02-22
**대상**: 기본 10개 모델 + 고차 12개 모델 (AR(5)~SARIMA(2,1,2)(1,1,2,12))

---

## 1. 진단 방법론

성능 병목을 **추측이 아닌 측정**으로 식별하기 위해 3단계 분리 측정을 수행.

| Phase | 측정 대상 | 방법 | 분리하는 변수 |
|-------|-----------|------|--------------|
| **Phase 1** | 단일 loglike 평가 비용 | optimizer 제외, 동일 파라미터로 Kalman filter만 100회 호출 | per-eval 비용 (Kalman filter + StateSpace 구성) |
| **Phase 2** | fit 전체 분해 | fit 시간 / 반복 횟수 = per-iter 비용, Rust vs SM 반복 비율 | optimizer 반복 횟수 vs per-eval 비용 |
| **Phase 3** | k_states vs 비용 스케일링 | 상태 차원별 per-eval 비용 정렬 | Kalman filter의 O(k²) 스케일링 특성 |

---

## 2. Phase 1: 단일 loglike 평가 비용

**실험**: statsmodels가 찾은 최적 파라미터로 `sarimax_loglike()`를 100회 호출, 하위 20% 중앙값.

```
k_states    RS(us)    SM(us)    RS/SM    Model
--------  --------  --------  -------  ------
       1       5.0      94.9    0.05    AR(1) n=200
       3      13.6     202.4    0.07    ARMA(2,2) n=400
       4      58.2     349.0    0.17    ARMA(3,3) n=500
       5      20.3     268.9    0.08    AR(5) n=500
       6     178.1     687.6    0.26    ARMA(5,5) n=800
       8      33.2     341.3    0.10    AR(8) n=500
      11      64.1     555.7    0.12    SARIMA(2,0,2)(2,0,2,4)
      25     126.7    1274.2    0.10    SARIMA(1,0,1)(2,0,0,12)
      26     251.0    1979.2    0.13    SARIMA(1,0,1)(0,0,2,12)
      27     412.5    2203.2    0.19    SARIMA(1,1,1)(1,1,1,12)
      38    1050.2    5371.4    0.20    SARIMA(1,1,1)(2,1,1,12)
      40    1382.2    4042.3    0.34    SARIMA(2,1,2)(1,1,2,12)
```

### 결론

- **모든 모델에서 Rust Kalman filter가 3~20x 빠름** (RS/SM 비율 0.05~0.34)
- k_states가 커져도 Rust 우위 유지 (k=40에서도 3x 빠름)
- **per-eval 비용은 병목이 아님**

### 스케일링 관찰

| k_states 범위 | RS/SM 비율 | Rust 우위 |
|---------------|-----------|-----------|
| 1~8 | 0.05~0.10 | 10~20x |
| 11~27 | 0.10~0.19 | 5~10x |
| 38~40 | 0.20~0.34 | 3~5x |

k_states 증가 시 Rust 우위가 줄어드는 이유: nalgebra의 동적 행렬은 k > 20에서 BLAS 최적화된 numpy/scipy 대비 상대 효율이 감소. 그러나 절대 우위(RS/SM < 1.0)는 테스트한 모든 차원에서 유지됨.

---

## 3. Phase 2: Fit 분해 — 느린 모델의 원인 식별

**실험**: `sarimax_fit()` 전체 시간과 반복 횟수를 분리 측정.

```
Model                                    RS(ms)   SM(ms)  Speed  RS it  SM it  it x  eval x  Bottleneck
-------------------------------------------------------------------------------------------------------
AR(1) n=200                                 0.8      4.6   5.8x     44      3  14.7    0.05  -
ARMA(2,2) n=400                             4.8     27.8   5.7x     19     11   1.7    0.07  -
SARIMA(1,1,1)(1,1,1,12)                   358.3    671.4   1.9x     34     27   1.3    0.19  -
AR(5) n=500                                24.2     22.5   0.9x     18      4   4.5    0.08  iters(4.5x)
AR(8) n=500                                63.5     36.9   0.6x     20      4   5.0    0.10  iters(5.0x)
ARMA(3,3) n=500                            18.7    472.6  25.3x     25    127   0.2    0.17  -
ARMA(5,5) n=800                           472.2   1061.2   2.2x     72    112   0.6    0.26  -
SARIMA(1,0,1)(2,0,0,12)                   348.2    132.8   0.4x     25      9   2.8    0.10  iters(2.8x)
SARIMA(1,0,1)(0,0,2,12)                   283.0    206.2   0.7x     26      9   2.9    0.13  iters(2.9x)
SARIMA(2,0,2)(2,0,2,4)                    228.2    271.0   1.2x     45     36   1.2    0.12  -
SARIMA(1,1,1)(2,1,1,12)                  2489.7    958.4   0.4x     62     16   3.9    0.20  iters(3.9x)
SARIMA(2,1,2)(1,1,2,12)                  2607.5   2446.3   0.9x     41     52   0.8    0.34  overhead/marginal
```

### 핵심 발견

**느린 모델은 전부 `iters(N.Nx)` — per-eval cost가 원인인 모델은 없음.**

| 느린 모델 | Speedup | eval 비율 | iter 비율 | 진짜 원인 |
|-----------|---------|-----------|-----------|-----------|
| AR(5) | 0.9x | 0.08 (**12x 빠름**) | **4.5x** | 반복 18 vs 4 |
| AR(8) | 0.6x | 0.10 (**10x 빠름**) | **5.0x** | 반복 20 vs 4 |
| SARIMA(1,0,1)(2,0,0,12) | 0.4x | 0.10 (**10x 빠름**) | **2.8x** | 반복 25 vs 9 |
| SARIMA(1,0,1)(0,0,2,12) | 0.7x | 0.13 (**8x 빠름**) | **2.9x** | 반복 26 vs 9 |
| SARIMA(1,1,1)(2,1,1,12) | 0.4x | 0.20 (**5x 빠름**) | **3.9x** | 반복 62 vs 16 |

### 역증 (반증 배제)

만약 per-eval 비용이 병목이었다면, eval 비율(RS/SM)이 1.0을 넘어야 함. 그러나 **모든 모델에서 0.05~0.34** → per-eval은 원인이 아님을 확정.

---

## 4. Phase 3: 상태 차원과 비용의 관계

### k_states 계산 공식

```
k_states = k_states_diff + k_order
         = (d + s*D) + max(p + s*P, q + s*Q + 1)
```

| 모델 | p,d,q | P,D,Q,s | k_states_diff | k_order | k_states |
|------|-------|---------|---------------|---------|----------|
| AR(1) | 1,0,0 | - | 0 | 1 | **1** |
| AR(5) | 5,0,0 | - | 0 | 5 | **5** |
| AR(8) | 8,0,0 | - | 0 | 8 | **8** |
| ARMA(2,2) | 2,0,2 | - | 0 | 3 | **3** |
| ARMA(5,5) | 5,0,5 | - | 0 | 6 | **6** |
| SARIMA(1,0,1)(2,0,0,12) | 1,0,1 | 2,0,0,12 | 0 | 25 | **25** |
| SARIMA(1,1,1)(1,1,1,12) | 1,1,1 | 1,1,1,12 | 13 | 14 | **27** |
| SARIMA(1,1,1)(2,1,1,12) | 1,1,1 | 2,1,1,12 | 13 | 25 | **38** |
| SARIMA(2,1,2)(1,1,2,12) | 2,1,2 | 1,1,2,12 | 13 | 27 | **40** |

### Kalman filter 복잡도

1회 loglike 평가의 계산 복잡도:

| 연산 | Dense path | Sparse path (T 밀도 < 50%) |
|------|-----------|--------------------------|
| 혁신 v = y - Z'a | O(k) | O(nnz_Z) |
| pz = P*Z | O(k²) | O(nnz_Z * k) |
| 상태 갱신 a = a + K*v | O(k) | O(k) |
| 공분산 P = T*P*T' + RQR' | **O(k³)** | **O(nnz_T * k)** |

k_states=27인 SARIMA(1,1,1)(1,1,1,12)의 T 밀도는 ~4% (31/729), sparse path 사용.
k_states=40인 모델에서도 sparse path 사용되나, nnz_T * k 자체가 큼.

---

## 5. 근본 원인: Optimizer 반복 횟수

### 5.1 왜 Rust L-BFGS-B가 더 많은 반복을 하는가?

측정된 반복 배수:

| 패턴 | iter 배수 | 해당 모델 |
|------|-----------|-----------|
| 순수 AR (고차) | 4.5~5.0x | AR(5), AR(8) |
| 계절 SARIMA (s=12) | 2.8~3.9x | SARIMA(1,0,1)(2,0,0,12), SARIMA(1,1,1)(2,1,1,12) |
| 비계절 ARMA/ARIMA | 0.2~1.7x | 대부분 빠르거나 동등 |

### 5.2 원인 후보 분석

#### 후보 A: Start params 품질

**근거**: AR(5)는 5개 파라미터에 18 반복, statsmodels는 4 반복. 파라미터 수 대비 반복이 과도함.

- CSS 기반 start params (Yule-Walker)가 고차 AR에서 부정확할 가능성
- statsmodels는 Hannan-Rissanen 기반 + 추가 보정을 사용

**검증 방법**: statsmodels start params를 Rust optimizer에 주입하여 반복 횟수 비교.

#### 후보 B: lbfgsb 크레이트 라인서치

**근거**: 기본 10개 모델에서 bounds 제거만으로 반복이 48→25로 감소한 사례.

- lbfgsb 크레이트(v0.1.1)가 L-BFGS-B-C를 래핑하지만, 라인서치 동작이 scipy Fortran 구현과 미세하게 다를 가능성
- 특히 고차 파라미터 공간에서 Hessian 근사 정밀도가 다를 수 있음

**검증 방법**: 동일 start params + 동일 gradient로 scipy vs lbfgsb 크레이트의 반복 횟수 비교.

#### 후보 C: Gradient 품질

**근거**: 분석적 그래디언트(tangent linear KF)가 부정확하면 라인서치가 추가 탐색 필요.

- Monahan/Jones 변환의 Jacobian이 고차에서 수치적으로 ill-conditioned 될 가능성
- 유한차분 대비 분석적 gradient의 상대 오차가 파라미터 수에 따라 증가할 수 있음

**검증 방법**: 분석적 gradient vs 중심차분 gradient 비교, 고차 모델에서의 상대 오차 측정.

---

## 6. 개선 방향 (우선순위순)

### 6.1 Start params 품질 향상 (예상 효과: 반복 2~3x 감소)

현재 CSS 기반 Yule-Walker는 고차 AR에서 편향이 큼. 개선안:

1. **Burg 방법**: Yule-Walker 대신 Burg의 maximum entropy method 사용 (고차 AR에서 유한 표본 편향 감소)
2. **Hannan-Rissanen**: statsmodels 방식의 3단계 추정 (고차 AR 피팅 → 잔차 MA 추정 → GLS 재추정)
3. **Seasonal debiasing**: 계절 자기공분산 추정에 Bartlett 보정 적용

### 6.2 Gradient 정확도 검증 및 개선 (예상 효과: 반복 1.5~2x 감소)

1. **Jacobian 수치 안정성**: 고차 Monahan/Jones 변환의 조건수 모니터링
2. **중심차분 대체**: 분석적 gradient 오차가 threshold 초과 시 중심차분으로 폴백
3. **Hessian 초기화**: L-BFGS 메모리에 Fisher information 기반 초기 H₀ 제공

### 6.3 Optimizer 다변화 (예상 효과: 특정 모델에서 2~5x 감소)

1. **순수 AR 전용 경로**: p > 0, q = 0, Q = 0인 경우 Yule-Walker 해가 MLE와 점근적으로 동일 → optimizer 반복 없이 직접 해 반환
2. **Trust-region 방법**: L-BFGS-B 대신 trust-region Newton (Hessian 근사 사용)으로 고차 모델에서 초2차 수렴

---

## 7. 참고: 코드 경로

| 파일 | 역할 |
|------|------|
| `src/optimizer.rs:620` | `fit()` 진입점, L-BFGS-B/L-BFGS/NM 분기 |
| `src/optimizer.rs:534` | `run_lbfgsb()` — 단일 L-BFGS-B 실행 |
| `src/optimizer.rs:388` | `run_lbfgs()` — argmin L-BFGS (More-Thuente) |
| `src/start_params.rs:301` | `compute_start_params()` — CSS 초기값 추정 |
| `src/kalman.rs:83` | `kalman_core()` — sparse/dense/steady-state Kalman filter |
| `src/score.rs` | 분석적 gradient (tangent linear KF) |
| `src/state_space.rs:27` | `StateSpace::new()` — Harvey 행렬 구성 |
| `src/types.rs:47` | `k_states()` = k_order() + k_states_diff() |
