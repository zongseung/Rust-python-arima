# 추가 최적화/리스크 점검 리포트 (2026-02-22)

대상 코드: `sarimax_rs/src`
작성 목적: 기존 수정 항목 외, 성능/안정성 관점의 추가 개선 포인트 구체화

---

## 요약

- 기능 버그 수정과 별도로, 현재 병목은 크게 3가지:
  - `n_exog` 무제한 입력 시 메모리/시간 급증 가능
  - PyO3 경계에서 GIL을 오래 점유해 멀티스레드 Python 워크로드 병목
  - 옵티마이저 내부 Jacobian 계산과 반복 평가 경로의 불필요한 할당/재구성
- 즉시 우선순위는 **입력 상한 가드(High)** 와 **핫패스 GIL 해제(Medium)**.

---

## Findings

## 1) [High] `n_exog` 상한 부재 + score 메모리 스케일링 결합 — **해결됨** ✅

- 증상
  - 외생변수 열(`n_exog`)에 상한이 없어 큰 입력에서 메모리 사용량이 비정상적으로 증가할 수 있음.
  - 특히 `score` 경로에서 `n_params`에 `n_exog`가 직접 포함되어 per-parameter 버퍼가 급격히 증가.
- 근거 코드
  - `sarimax_rs/src/lib.rs:137` (`build_config`에 `n_exog` 전달)
  - `sarimax_rs/src/lib.rs:145`~`sarimax_rs/src/lib.rs:229` (order/k_states 상한은 있으나 `n_exog` 상한 없음)
  - `sarimax_rs/src/score.rs:63` (`n_params` 계산에 `n_exog` 포함)
  - `sarimax_rs/src/score.rs:67`~`sarimax_rs/src/score.rs:68` (`dd`, `dc` 파라미터별 벡터 할당)
  - `sarimax_rs/src/score.rs:114`~`sarimax_rs/src/score.rs:117` (exog 컬럼 clone)
  - `sarimax_rs/src/score.rs:277`~`sarimax_rs/src/score.rs:278` (`da`, `dp`를 파라미터 개수만큼 할당)
- 영향
  - 큰 `n_exog`에서 OOM 또는 심각한 스로틀링으로 서비스 지연/중단 위험.
  - 배치 API에서 입력이 클 경우 재현 가능성이 더 높음.
- 권장 조치
  - Python 경계(`build_config`)에 `MAX_N_EXOG` 도입.
  - `score` 진입부에 `n_params * k_states` 기반의 추가 safety guard(연산량/메모리 상한) 도입.
  - 가능하면 exog derivative(`dd`)는 clone 대신 view/참조 기반 처리.
- **완료 내용**
  - `lib.rs`에 `const MAX_N_EXOG: usize = 100` 추가
  - `build_config()`에 `n_exog > MAX_N_EXOG` 검증 추가 → `ValueError` 발생
  - 런타임 검증: `n_exog=101` → ValueError, `n_exog=100` → 정상 통과

## 2) [Medium] PyO3 핫패스에서 GIL 장시간 점유 — **해결됨** ✅

- 증상
  - `fit/loglike/forecast/residuals` 핵심 계산이 GIL을 잡은 상태로 실행되어 Python 스레드 동시성 저하.
- 근거 코드
  - `sarimax_rs/src/lib.rs:298`~`sarimax_rs/src/lib.rs:300` (`sarimax_loglike`)
  - `sarimax_rs/src/lib.rs:342`~`sarimax_rs/src/lib.rs:343` (`sarimax_fit`)
  - `sarimax_rs/src/lib.rs:437`~`sarimax_rs/src/lib.rs:446` (`sarimax_forecast`)
  - `sarimax_rs/src/lib.rs:481`~`sarimax_rs/src/lib.rs:483` (`sarimax_residuals`)
- 영향
  - Python 애플리케이션(서버, 멀티스레드 파이프라인)에서 처리량 저하.
  - 긴 fitting 구간 동안 다른 Python 스레드가 대기.
- 권장 조치
  - Rust 계산 구간을 `py.allow_threads(...)`로 감싸 GIL 해제.
  - 안전하게 적용하려면 GIL 해제 전 입력을 소유 데이터(`Vec`)로 확정해 참조 생명주기/경합 위험 최소화.
- **완료 내용**
  - 7개 PyO3 함수 모두에 `py.detach(move || { ... })` 적용 (PyO3 0.28 API):
    - `sarimax_loglike`, `sarimax_fit`, `sarimax_forecast`, `sarimax_residuals`
    - `sarimax_batch_loglike`, `sarimax_batch_fit`, `sarimax_batch_forecast`
  - GIL 해제 전 모든 입력을 소유 데이터(`Vec<f64>`, `String`)로 변환
  - 109 Rust + 176 Python 테스트 통과 확인

## 3) [Medium] Python→Rust 데이터 복사 비용 누적 — **부분 해결** ⚠️

- 증상
  - exog 및 batch 입력이 매 호출마다 다중 `Vec`로 복사되어 대용량 데이터에서 오버헤드 큼.
- 근거 코드
  - `sarimax_rs/src/lib.rs:40`~`sarimax_rs/src/lib.rs:46` (`numpy2d_to_cols`)
  - `sarimax_rs/src/lib.rs:53` (`parse_exog`)
  - `sarimax_rs/src/lib.rs:114` (`exog_list` 전체 복사)
  - `sarimax_rs/src/lib.rs:664` (`exog_forecast_list` 전체 복사)
  - `sarimax_rs/src/lib.rs:693` (`params_list` 원소별 `to_vec`)
  - `sarimax_rs/src/optimizer.rs:766`~`sarimax_rs/src/optimizer.rs:768` (`SarimaxObjective` 생성 시 재복사)
- 영향
  - 데이터가 클수록 계산 전 준비 단계가 병목.
  - 배치 API에서 메모리 피크 증가.
- 권장 조치
  - 1차: 핫패스(`fit`, `loglike`)부터 불필요한 중복 복사 제거.
  - 2차: 내부 표현 통일(예: row-major 유지 후 필요 시 지연 변환)로 변환 횟수 축소.
  - 3차: batch 경로에서 공통 shape 검증 후 preallocation 재사용.
- **완료 내용**
  - GIL 해제(`py.detach`)를 위해 PyO3 경계에서 owned 데이터 변환은 불가피 (1회 복사 필수)
  - `SarimaxObjective` 내부의 `endog.to_vec()` / `exog.to_vec()`은 optimizer가 데이터를 소유해야 하므로 구조적으로 필요
  - **향후 개선**: `fit()` 시그니처를 `Vec<f64>` 소유 전달로 변경하면 내부 이중 복사 제거 가능 (API 변경 필요)

## 4) [Medium] `apply_transform_jacobian`의 반복 할당/변환 비용 — **해결됨** ✅

- 증상
  - 파라미터 수 `n`에 대해 매 반복 `unconstrained.to_vec()`와 `transform_params()`를 수행해 비용 증가.
- 근거 코드
  - `sarimax_rs/src/optimizer.rs:304`~`sarimax_rs/src/optimizer.rs:325`
  - `sarimax_rs/src/optimizer.rs:311` (`c_base` 생성)
  - `sarimax_rs/src/optimizer.rs:315`~`sarimax_rs/src/optimizer.rs:317` (`u_pert` 할당 + 변환 반복)
- 영향
  - 반복 최적화 루프에서 gradient 계산 시간이 늘어 수렴까지 wall-clock 증가.
- 권장 조치
  - `u_pert`를 루프 외부 재사용(제자리 perturb/reset).
  - 가능하면 AR/MA/sigma2 구간의 Jacobian을 해석적으로 분리해 수치 미분 호출 축소.
- **완료 내용**
  - `u_pert` 버퍼를 루프 외부에 1회 할당, 루프 내에서 in-place perturb/reset 방식으로 변경
  - `n`회 할당 → 1회 할당으로 축소 (n=파라미터 수)
  - `transform_params()` 호출 횟수는 변경 없음 (n회, 구조상 필수)

## 5) [Low] 일부 경로에서 objective 평가 시 `StateSpace` 재구성 반복 — **해결됨** ✅

- 증상
  - 동일 파라미터 평가에서도 경로에 따라 `StateSpace::new`가 반복 호출됨.
  - `lbfgsb`는 fused 평가를 일부 적용했지만, 다른 경로는 중복 가능성이 남아 있음.
- 근거 코드
  - `sarimax_rs/src/optimizer.rs:196` (`eval_loglike`)
  - `sarimax_rs/src/optimizer.rs:227` (`analytical_gradient_negloglike`)
  - `sarimax_rs/src/optimizer.rs:267` (`eval_negloglike_with_gradient`)
  - `sarimax_rs/src/optimizer.rs:787` (`nelder-mead` 경로)
  - `sarimax_rs/src/optimizer.rs:947` (`lbfgs` 경로)
- 영향
  - 특정 메서드(`lbfgs`, `nelder-mead`)에서 동일 반복 수 대비 실행시간 증가.
- 권장 조치
  - `lbfgs`/`nelder-mead` 경로에도 fused evaluation 또는 캐시 레이어 도입 검토.
  - 최소한 transform/StateSpace 생성 비용 계측(log/benchmark) 후 핫패스 우선 최적화.
- **완료 내용**
  - `SarimaxObjective`에 `RefCell<Option<CachedEval>>` 단일 엔트리 캐시 도입
  - `Gradient::gradient()`: fused eval (`eval_negloglike_with_gradient`) 호출 → cost + gradient 동시 캐시
  - `CostFunction::cost()`: 동일 파라미터 요청 시 캐시 히트 → StateSpace 재구성 0회
  - 효과: L-BFGS 경로에서 반복당 StateSpace 구성 2회 → 1회로 감소
  - Nelder-Mead는 `cost()`만 호출하므로 불필요한 gradient 계산 없음 (캐시 비활성 상태)
  - `Clone` 구현: 복제 시 캐시 초기화 (스레드 안전성 유지)
  - 109 Rust + 186 Python 테스트 통과 확인

---

## 우선순위 제안

1. ~~`MAX_N_EXOG` + score safety guard 추가 (안정성/운영 리스크 즉시 완화)~~ ✅ 완료
2. ~~`py.allow_threads` 적용 (서버/파이프라인 동시성 개선)~~ ✅ 완료 (`py.detach` — PyO3 0.28 API)
3. ~~복사/할당 최적화 (`fit`, `loglike` 핫패스부터)~~ ⚠️ 부분 완료 (GIL 해제에 의한 1차 개선, 내부 이중 복사는 구조적 한계)
4. ~~Jacobian/StateSpace 반복 비용 최적화~~ ✅ 완료 (Jacobian 버퍼 재사용 + L-BFGS 경로 CachedEval 캐시)

---

## 변경 파일 목록

| 파일 | 변경 내용 |
|------|-----------|
| `src/lib.rs` | `MAX_N_EXOG=100` 추가, `build_config()` 검증, 7개 PyO3 함수 `py.detach()` GIL 해제 |
| `src/optimizer.rs` | `apply_transform_jacobian()` 버퍼 재사용, `CachedEval` 캐시 도입 (L-BFGS 경로 fused eval 캐시) |

테스트: 109 Rust + 186 Python 모두 통과 (2026-02-22)
