# SARIMAX 동등성 점검 (ver4, 2026-02-22)

대상: `sarimax_rs/src`, `sarimax_rs/python/sarimax_py`  
목적: "일반적인 SARIMAX(주로 statsmodels 기준)와 동일한가?"에 대한 현재 상태 정리

---

## 결론

- **핵심 모델 구조는 동일 계열**입니다.
  - `SARIMAX(p,d,q)(P,D,Q,s)` + `exog`
  - 상태공간 + 칼만필터 기반 MLE
- 다만 **완전 동등(fully identical)** 은 아닙니다.
  - 일부 기능 범위 제한
  - 일부 메타데이터 의미/동작 차이
  - 입력 가드 정책 차이

---

## 동일한 부분 (Parity)

1. 모델 클래스/파라미터 체계
- `order=(p,d,q)`, `seasonal_order=(P,D,Q,s)`, `exog` 지원
- 근거: `sarimax_rs/python/sarimax_py/model.py:29`, `sarimax_rs/python/sarimax_py/model.py:30`, `sarimax_rs/python/sarimax_py/model.py:31`

2. 추정 방식
- 상태공간 구성 후 칼만필터 로그우도 최대화(MLE)
- 근거: `sarimax_rs/src/lib.rs:297`, `sarimax_rs/src/lib.rs:342`, `sarimax_rs/src/optimizer.rs:653`

3. 예측/신뢰구간 API
- `forecast/get_forecast/conf_int(alpha)` 동작 제공
- 근거: `sarimax_rs/python/sarimax_py/model.py:111`, `sarimax_rs/python/sarimax_py/model.py:143`, `sarimax_rs/python/sarimax_py/model.py:206`

---

## 다른 부분 (Gap)

## 1) 기능 범위 제한

1. `seasonal D > 1` 미지원
- 근거: `sarimax_rs/src/state_space.rs:39`

2. `simple_differencing` 미지원
- 근거: `sarimax_rs/src/state_space.rs:52`

3. `measurement_error` 미지원
- 근거: `sarimax_rs/src/state_space.rs:58`

## 2) Python API 노출 범위 차이

1. Python 경계에서 `trend` 고정(`Trend::None`)
- 근거: `sarimax_rs/src/lib.rs:234`
- 영향: statsmodels의 `trend='c'/'t'/'ct'`와 완전 동일하지 않음

## 3) 최적화 결과 메타데이터 차이

1. `n_iter` 의미가 메서드별로 다름
- L-BFGS/NM: iteration count
- L-BFGS-B: function evaluation count
- 근거: `sarimax_rs/src/optimizer.rs:415`, `sarimax_rs/src/optimizer.rs:466`, `sarimax_rs/src/optimizer.rs:624`, `sarimax_rs/docs/api_reference.md:80`

2. 고차 순수 AR fast-path에서 `n_iter=0`, `converged=true`를 반환
- 근거: `sarimax_rs/src/optimizer.rs:715`, `sarimax_rs/src/optimizer.rs:729`, `sarimax_rs/src/optimizer.rs:730`
- 영향: 일반적인 "optimizer 수렴" 의미와는 해석이 다를 수 있음

## 4) 입력 검증 정책 차이(더 엄격)

1. order/k_states/steps 등 상한 가드 존재
- 근거: `sarimax_rs/src/lib.rs:145`, `sarimax_rs/src/lib.rs:224`, `sarimax_rs/src/lib.rs:384`
- 영향: statsmodels에서 허용될 수 있는 일부 입력이 여기서는 명시적 에러 처리될 수 있음

---

## ver4 권장 작업 (완전 동등성 목표)

1. ⏳ Python API에 `trend` 노출 및 상태공간/요약 출력 정합화
   - 상태: `Deferred (ver5)` — Rust 내부에 `Trend` enum 이미 구현됨, Python 노출만 필요
2. ⏳ `simple_differencing`, `measurement_error` 지원 여부 결정
   - 상태: `Deferred (ver5)` — 지원 시 구현, 미지원 유지 시 문서/에러메시지 더 명확화
3. ⏳ `n_iter` 필드 분리
   - 상태: `Deferred` — 예: `n_iter`(iteration), `n_eval`(function eval) 병행 노출
   - 현재 `api_reference.md`에 메서드별 의미 차이 문서화 완료
4. ✅ fast-path 메타데이터 정책 명확화
   - 상태: `Resolved` — `converged=True, method="burg-direct", n_iter=0`으로 명확히 구분
   - `api_reference.md`에 burg-direct fast path 설명 완료
5. ✅ statsmodels 비교 회귀 테스트 세트 추가
   - 상태: `Resolved` — `TestStatsmodelsParity` 클래스에 8개 테스트 존재
   - 동일 입력에 대해 `params/loglike/forecast/ci/inference` 허용 오차 비교

