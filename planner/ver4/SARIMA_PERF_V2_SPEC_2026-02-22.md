# SARIMA 성능 개선 2차 구현 명세서 (ver4, 2026-02-22)

**상태: ✅ 구현 완료 — 모든 모델에서 statsmodels 추월 달성**

## 1. 배경

### 1차 최적화 결과 (P1: Hannan-Rissanen, P2: Score steady-state)

| 모델 | 이전 | 1차 후 | statsmodels | 배율 |
|---|---:|---:|---:|---:|
| SARIMA(0,1,1)(0,1,1,12) | 255.8ms | 154.0ms | 109.4ms | 0.71x |
| SARIMA(1,1,1)(1,1,1,12) | 305.4ms | 261.2ms | 212.5ms | 0.81x |

계절 SARIMA 모델에서 여전히 statsmodels보다 느린 상태. 비계절 모델은 52-115x 빠름.

### 잔존 병목 분석

3개 에이전트 조사 결과, **진짜 병목**은 `score.rs`의 `compute_dp_predict()` 내 dense gemm (`T*dP*T'`)이다:
- `T`는 companion matrix로 매우 sparse (k=27에서 31/729 = 4.3% density)
- `kalman.rs`에서는 이미 sparse T를 사용하여 O(nnz×k) 연산
- `score.rs`에서는 여전히 dense gemm O(k³) 사용 — **이것이 핵심 격차**

---

## 2. 조사 기반 전략 평가

### 기존 제안 3가지 재평가

| # | 방법 | 논문/GitHub 근거 | 판정 |
|---|---|---|---|
| 1 | dRQR rank-1 활용 | Kitagawa (2020), Nagakura (2013) | **영향 미미** — dRQR copy는 O(k²), 병목은 T*dP*T'의 O(k³) |
| 2 | HR 비계절 확장 | Wheeler & Ionides (2024): ARMA(2,2)+ 53.8% local optima | **비권장** — 문제는 HR이 아니라 likelihood multimodality |
| 3 | NM 자동 분기 | Lagarias et al. (1998): NM 수렴 보장 없음 | **비권장** — 대신 FD gradient 사용이 더 효과적 |

### 핵심 발견

**statsmodels는 analytical gradient를 사용하지 않는다** (Chad Fulton 확인):
```python
# statsmodels MLEModel.fit() 기본값
kwargs.setdefault("approx_grad", True)  # scipy FD gradient
kwargs.setdefault("epsilon", 1e-5)
```

- statsmodels: `approx_grad=True` → scipy L-BFGS-B의 forward FD → (np+1) KF 평가/gradient
- sarimax_rs: tangent-linear score → 1 augmented KF pass/gradient (이론적으로 우월)
- **그러나**: tangent-linear은 pre-SS에서 O(n×np×k³), FD는 O(np×n×k²)

### 비용 비교 (SARIMA(1,1,1)(1,1,1,12), k=27, np=4)

| 방법 | Pre-SS 비용/gradient | Post-SS 비용/gradient |
|---|---:|---:|
| Tangent-linear (현재, dense T) | 4 × n × 27³ = **78,732n** | 4 × n × 27² = 2,916n |
| **Tangent-linear + sparse T** | **4 × n × 31 × 27 = 3,348n** | 4 × n × 27² = 2,916n |
| Forward FD | 5 × n × 27² = 3,645n | 5 × n × 27² = 3,645n |

→ **sparse T 적용이 FD보다 더 낫다** (3,348 < 3,645, 그리고 accuracy 유지)

---

## 3. 수정된 구현 계획

### 3.1 ✅ [P3] compute_dp_predict에 sparse T 적용 (최고 우선순위)

**현재 코드** (`score.rs:688-689`):
```rust
// O(k³) dense gemm — 병목!
temp.gemm(1.0, t_mat, dp_upd, 0.0);       // T * dP
result.gemm(1.0, temp, t_mat_t, 1.0);     // (T*dP) * T'
```

**변경 후**: kalman.rs 패턴 차용
```rust
// O(nnz×k) sparse multiply
// Step 1: temp = sparse_T * dP
temp.fill(0.0);
for &(i, l, val) in sparse_t {
    for j in 0..k { tmp[i + j*k] += val * dp[l + j*k]; }
}
// Step 2: result += temp * sparse_T'
for &(j, l, val) in sparse_t {
    for i in 0..k { res[col_j + i] += val * tmp[col_l + i]; }
}
```

**예상 효과**: k=27에서 `T*dP*T'` 항 **23.5x 가속** (19,683 → 837 ops per param per step)

**구현 위치**: `score.rs` — `compute_dp_predict()` 시그니처에 `sparse_t` 추가, score() 호출부에서 sparse_t 생성 및 전달

### 3.2 [P4] FD gradient fallback (k_states 기준 자동 분기)

sparse T 최적화 후에도 여전히 느리다면 적용할 **예비 전략**:

**로직**: `k_states > K_FD_THRESHOLD` (예: 20)이면 tangent-linear 스킵, FD 사용
- optimizer.rs `evaluate` 클로저에서 `eval_negloglike_with_gradient()` 호출 조건에 k_states 체크 추가
- 임계값 초과 시 바로 FD fallback 경로 (lines 634-659) 진입

**근거**: sparse T 적용 후 tangent-linear이 FD보다 효율적일 가능성이 높으므로, P3 결과 확인 후 필요 시에만 적용

---

## 4. 참고 논문 및 GitHub

### 학술 논문

| 논문 | 핵심 내용 | 관련 |
|---|---|---|
| Koopman & Shephard (1992) "Exact score for time series models" | 단일 forward pass로 exact score 계산. Disturbance smoother 방식이 대안 | Score 알고리즘 |
| Kitagawa (2020) arXiv:2011.09638 "Differential Filter" | 우리의 tangent-linear 접근과 동일. dRQR rank 구조 미논의 | 현재 구현의 이론적 기반 |
| Wheeler & Ionides (2024) arXiv:2310.01198 | ARMA(p,q) p+q>=3에서 53.8% local optima. Multi-start 필수 | HR 비계절 확장 불필요 근거 |
| Lagarias et al. (1998) SIAM J. Optim. | NM 수렴 보장은 1D만. 2D+에서 비최적점 수렴 가능 | NM 자동 분기 비권장 근거 |
| Bonnabel et al. (2024) SIAM JMAA | Low-rank + diagonal Riccati 근사. 고차원 KF에서 O(k²)→O(k) | 향후 참고 |

### GitHub 참고 구현

| 프로젝트 | 핵심 | 링크 |
|---|---|---|
| statsmodels MLEModel | `approx_grad=True` 기본 → FD gradient | [mlemodel.py](https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/statespace/mlemodel.py) |
| statsmodels SARIMAX | CSS 기반 start_params (HR 아님) | [sarimax.py](https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/statespace/sarimax.py) |
| statsmodels HR estimator | 3-stage HR, 비계절만 지원 | [hannan_rissanen.py](https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/arima/estimators/hannan_rissanen.py) |
| R stats::arima | zeros → CSS-ML 2단계, BFGS, FD gradient | [arima.R](https://github.com/SurajGupta/r-source/blob/master/src/library/stats/R/arima.R) |
| KFKSDS (R/C++) | Dense BLAS gemm for dP, sparse 미활용 | [KSDS-deriv.cpp](https://github.com/cran/KFKSDS/blob/master/src/KSDS-deriv.cpp) |

---

## 5. 위험 요소

| 위험 | 완화 |
|---|---|
| sparse T 적용 후 수치 정확도 변화 | 기존 score 테스트 전체 통과 필수 (analytical vs numerical 비교) |
| sparse 경로가 dense보다 느린 경우 (작은 k) | `use_sparse` 플래그로 k_states 기준 분기 |
| FD fallback이 정확도 저하 | P3 결과 먼저 평가, 필요 시에만 P4 적용 |

---

## 6. 구현 순서

1. ✅ `score.rs`: sparse_t 생성 + `compute_dp_predict` sparse 경로 추가
2. ✅ `score.rs`: 표준 KF predict (P = T*P*T' + RQR')에도 sparse T 적용
3. ✅ 테스트: 114 Rust 전체 + 235 Python 전체 통과
4. ✅ 벤치마크: SARIMA fit 시간 비교 — **모든 모델에서 statsmodels 추월**
5. ❌ P4 (FD fallback): **불필요** — sparse T만으로 충분

---

## 7. 최종 벤치마크 결과

### P3 적용 후 (sparse T in score.rs)

| 모델 | 원래 | P1+P2 후 | **P3 후** | statsmodels | **배율** |
|---|---:|---:|---:|---:|---:|
| SARIMA(0,1,1)(0,1,1,12) | 255.8ms | 154.0ms | **62.7ms** | 128.8ms | **2.1x** |
| SARIMA(1,1,1)(1,1,1,12) | 305.4ms | 261.2ms | **161.8ms** | 279.4ms | **1.7x** |

### 전체 결과 요약

| 모델 | sarimax_rs | statsmodels | 배율 |
|---|---:|---:|---:|
| AR(1) | 0.2ms | 2.9ms | **12.6x** |
| AR(2) | 0.4ms | 4.8ms | **12.2x** |
| MA(1) | 0.3ms | 3.6ms | **11.5x** |
| ARMA(1,1) | 1.2ms | 15.7ms | **12.5x** |
| ARIMA(1,1,1) | 1.4ms | 12.4ms | **8.9x** |
| ARIMA(2,1,1) | 0.6ms | 38.4ms | **68.4x** |
| SARIMA(1,0,0)(1,0,0,4) | 1.0ms | 7.8ms | **8.2x** |
| SARIMA(0,1,1)(0,1,1,12) | 62.7ms | 128.8ms | **2.1x** |
| SARIMA(1,1,1)(1,1,1,12) | 161.8ms | 279.4ms | **1.7x** |

- Python 전체 테스트: **46s → 9.35s (4.9x 단축)**
- Rust 테스트: 114개 전체 통과
- Python 테스트: 235개 전체 통과
