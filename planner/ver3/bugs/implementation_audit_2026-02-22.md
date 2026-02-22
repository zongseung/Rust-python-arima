# comprehensive_review_and_plan 구현 점검 (2026-02-22)

대상 문서: `planner/ver3/comprehensive_review_and_plan.md`  
점검 범위: `sarimax_rs/src`, `sarimax_rs/python_tests`, 실제 테스트 실행 결과

---

## 결론 요약

- **소스 코드 기준**: Phase 1~5 항목은 대부분 구현됨.
- **실행(파이썬 모듈) 기준**: ✅ **완료**. `maturin develop --release` 재빌드 후 전체 176 테스트 PASS 확인 (2026-02-22).
- **Phase 6**: 성능 최적화 진행 중. 옵티마이저 단일 실행 전환으로 8/10 모델 속도 우위 달성.

---

## Phase별 구현 상태

## Phase 1: 버그 수정

1) `batch_fit` default method 정합
- 상태: ✅ **완료**
- 근거: `sarimax_rs/src/batch.rs:71` (`"lbfgsb"` 사용)

2) steady-state filtered_state 저장 오류
- 상태: ✅ **완료**
- 근거: `sarimax_rs/src/kalman.rs:215`~`sarimax_rs/src/kalman.rs:222`
  `a_filtered = a + (v/F)*pz_inf` 반영

3) sigma2 bound 문제
- 상태: ✅ **완료(설계 변경 포함)**
- 근거: `sarimax_rs/src/params.rs:180`~`sarimax_rs/src/params.rs:196` (variance exp/log 변환)
  `sarimax_rs/src/optimizer.rs:469`~`sarimax_rs/src/optimizer.rs:473` (sigma2 하한을 unconstrained `-50.0`으로 적용)

## Phase 2: Input Validation 강화

1) NaN/Inf 입력 검증
- 상태: ✅ **완료**
- 근거: `sarimax_rs/src/lib.rs:61`, `sarimax_rs/src/lib.rs:76`, `sarimax_rs/src/lib.rs:85`

2) batch exog 길이 검증
- 상태: ✅ **완료**
- 근거: `sarimax_rs/src/lib.rs:106`~`sarimax_rs/src/lib.rs:112`

3) batch n_exog 일관성 검증
- 상태: ✅ **완료**
- 근거: `sarimax_rs/src/lib.rs:114`~`sarimax_rs/src/lib.rs:123`

4) batch_forecast alpha 검증
- 상태: ✅ **완료**
- 근거: `sarimax_rs/src/lib.rs:633`~`sarimax_rs/src/lib.rs:639`

5) state_space exog row 수 검증
- 상태: ✅ **완료**
- 근거: `sarimax_rs/src/state_space.rs:70`~`sarimax_rs/src/state_space.rs:82`

6) forecast exog 길이 부족 시 에러
- 상태: ✅ **완료**
- 근거: `sarimax_rs/src/forecast.rs:60`~`sarimax_rs/src/forecast.rs:67`

## Phase 3: 수학 개선

1) variance transform exp/log
- 상태: ✅ **완료**
- 근거: `sarimax_rs/src/params.rs:180`~`sarimax_rs/src/params.rs:196`

2) Seasonal AR start params 개선
- 상태: ✅ **완료**
- 근거: `sarimax_rs/src/start_params.rs:352`~`sarimax_rs/src/start_params.rs:359`

3) z-score 고정밀 근사
- 상태: ✅ **완료**
- 근거: `sarimax_rs/src/forecast.rs:196`~`sarimax_rs/src/forecast.rs:218`

## Phase 4: API 정리

1) `sarimax_loglike` enforce default 통일
- 상태: ✅ **완료**
- 근거: `sarimax_rs/src/lib.rs:256`~`sarimax_rs/src/lib.rs:257` (`true`)

2) 에러 타입 세분화
- 상태: ✅ **완료(부분)**
- 근거: `sarimax_rs/src/lib.rs:233`~`sarimax_rs/src/lib.rs:241`
  (`OptimizationFailed`, `CholeskyFailed`만 `RuntimeError`, 나머지는 `ValueError`)

3) batch dict 조립 `unwrap` 제거
- 상태: ✅ **완료**
- 근거: `sarimax_rs/src/lib.rs` 전체 `unwrap()` 미사용 확인

## Phase 5: 테스트 커버리지

1) `concentrate_scale=false` 테스트
- 상태: ✅ **완료**
- 근거: `sarimax_rs/python_tests/test_input_validation.py:297`

2) Batch forecast + exog 테스트
- 상태: ✅ **완료**
- 근거: `sarimax_rs/python_tests/test_input_validation.py:351`

3) Batch 길이 불일치 negative 테스트
- 상태: ✅ **완료**
- 근거: `sarimax_rs/python_tests/test_input_validation.py:195`

4) NaN/Inf rejection 테스트
- 상태: ✅ **완료**
- 근거: `sarimax_rs/python_tests/test_input_validation.py:106`

5) 최소 시리즈 길이 edge case 테스트
- 상태: ✅ **완료**
- 근거: `sarimax_rs/python_tests/test_input_validation.py:273`

6) higher-order cross-loglike 테스트
- 상태: ✅ **완료**
- 근거: `sarimax_rs/python_tests/test_matrix_tier_b.py:56`~`sarimax_rs/python_tests/test_matrix_tier_b.py:89`
  (`p,q<=5`, `P,Q<=2` 포함 fixture 기준)

## Phase 6: 성능 최적화

- 상태: 🔄 **진행 중**
- 진행사항:
  - ✅ 옵티마이저 L-BFGS-B 파라미터 scipy 기본값 통일 (m=10, factr=1e7, pgtol=1e-5)
  - ✅ Multi-start 제거 → 단일 L-BFGS-B 실행 (반복 횟수 5-15x 감소)
  - ✅ Fused function+gradient 평가 (StateSpace 재생성 제거)
  - ✅ 8/10 모델 속도 우위 달성
  - 🔄 SARIMA 2개 모델 속도 최적화 진행 중 (lbfgsb crate 라인서치 비효율)

---

## 실제 실행 기준 미완료/불일치 이슈

1) ~~Python 런타임이 구버전 확장 모듈 로드~~
- 상태: ✅ **해결됨** (2026-02-22)
- 조치: `maturin develop --release` 재빌드로 해결
- 결과: 전체 176 테스트 PASS

2) ~~pytest 전체 실행 실패~~
- 상태: ✅ **해결됨** (2026-02-22)
- 결과: `176 passed, 0 failed` (재빌드 후)

3) 문서-코드 세부 불일치(경미)
- `planner/ver3/bugs/pre_existing_failures_2026-02-22.md`의 D-1 설명은 NM refinement 조건 `n_params_total >= 3`
- 실제 구현은 `>= 2`:
  - `sarimax_rs/src/optimizer.rs:727`
  - `sarimax_rs/src/optimizer.rs:828`
- 기능 오류라기보다는 반복 수/성능 목표와의 정합성 이슈.
- 비고: Phase 6 옵티마이저 개선에서 multi-start가 `"lbfgsb-multi"` 메서드로 분리되어, 기본 `"lbfgsb"` 메서드에서는 NM refinement가 더 이상 적용되지 않음.

