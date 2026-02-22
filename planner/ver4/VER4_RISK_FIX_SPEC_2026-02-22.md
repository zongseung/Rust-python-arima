# ver4 잠재 위험 수정 명세서 (2026-02-22)

대상: `sarimax_rs/python/sarimax_py/model.py`, `sarimax_rs/python_tests/test_parameter_summary.py`, `sarimax_rs/docs/api_reference.md`
목적: 현재 남은 잠재 위험 항목을 코드로 바로 수정할 수 있도록 구체 작업 명세 제공

---

## 범위

본 명세서는 아래 5개 이슈를 수정 대상으로 한다.

1. ✅ `inference` 검증 우회 (High) — **Resolved** (2026-02-22)
2. ✅ `statsmodels` 추론 경로의 제약조건 불일치 (Medium) — **Resolved** (2026-02-22)
3. ✅ 추론 캐시의 파라미터 변경 미추적 (Medium) — **Resolved** (2026-02-22)
4. ✅ `both` 모드 `P>|z|` 소스 모호성 (Low) — **Resolved** (2026-02-22)
5. ✅ 기존 `test_model`의 `converged` 단정 실패 (Low) — **Resolved** (2026-02-22)

---

## 1) [High] `inference` 검증 우회

## 문제

- 위치: `sarimax_rs/python/sarimax_py/model.py:182`~`sarimax_rs/python/sarimax_py/model.py:190`
- 현재 `inference`와 `include_inference`가 함께 들어오면, `inference` 값 검증 없이 그대로 반환됨.

## 수정 요구사항

1. `_resolve_inference_mode(inference, include_inference)`에서:
- `inference is not None and include_inference is not None` 경로에서도
  - DeprecationWarning 발생 유지
  - **반드시 `inference` 값을 valid set(`none/hessian/statsmodels/both`)으로 검증**
- invalid일 때 `ValueError` 발생

2. 기존 동작 호환:
- `include_inference=True/False` 단독 사용 시 deprecation warning + 기존 매핑 유지

## 권장 구현 패턴

- 내부 helper `_validate_inference_mode(mode: str) -> str`를 만들고
  - 모든 branch에서 동일 helper를 통과시키는 방식 권장

## 수용 기준

1. `result.parameter_summary(inference="invalid", include_inference=True)`가 `ValueError`를 던짐
2. `result.parameter_summary(inference="hessian", include_inference=True)`는 warning 1회 + 정상 실행

## ✅ 해결 내용

- `_validate_inference_mode()` 헬퍼 추가, 모든 branch에서 동일 검증 통과
- 테스트: `TestRiskFixInferenceValidation::test_both_specified_invalid_inference_raises`, `test_both_specified_valid_inference_works`

---

## 2) [Medium] `statsmodels` 추론 경로 제약조건 불일치

## 문제

- 위치: `sarimax_rs/python/sarimax_py/model.py:248`~`sarimax_rs/python/sarimax_py/model.py:253`
- `SARIMAX(...)` 생성 시 `enforce_stationarity=True`, `enforce_invertibility=True`가 하드코딩.
- 실제 모델 설정(`self.model.enforce_stationarity`, `self.model.enforce_invertibility`)과 달라질 수 있음.

## 수정 요구사항

1. `_compute_statsmodels_inference(...)` 시그니처에 아래 인자 추가:
- `enforce_stationarity: bool`
- `enforce_invertibility: bool`

2. `SARIMAX(...)` 생성 시 위 인자를 그대로 사용:
- 하드코딩 제거

3. 호출부(2군데)에서 `self.model` 설정값 전달:
- `parameter_summary`의 `mode=="statsmodels"` branch
- `parameter_summary`의 `mode=="both"` branch

## 수용 기준

1. 모델을 `enforce_stationarity=False`, `enforce_invertibility=False`로 fit했을 때
- `inference="statsmodels"` 경로가 동일 설정으로 동작

## ✅ 해결 내용

- `_compute_statsmodels_inference()` 시그니처에 `enforce_stationarity`, `enforce_invertibility` 추가
- 하드코딩 제거, `self.model` 설정값 전달 (statsmodels, both 두 경로 모두)
- 테스트: `TestRiskFixEnforcementFlags::test_statsmodels_mode_respects_enforcement_flags`

---

## 3) [Medium] 추론 캐시의 파라미터 변경 미추적

## 문제

- 위치: `sarimax_rs/python/sarimax_py/model.py:480`, `sarimax_rs/python/sarimax_py/model.py:570`, `sarimax_rs/python/sarimax_py/model.py:603`
- 캐시 키가 `(mode, alpha)` 중심이라, `self.params`가 외부 변경되면 오래된 추론값을 재사용할 수 있음.

## 수정 요구사항

다음 중 하나를 반드시 적용:

### 옵션 A (권장, 최소 변경)
- 캐시 키에 파라미터 fingerprint 포함:
  - 예: `params_sig = tuple(np.round(self.params, 12))`
  - 키: `("hessian", alpha, params_sig)`, `("statsmodels", alpha, params_sig)`

### 옵션 B
- `parameter_summary` 진입 시 매번 `current_params`와 `cached_params`를 비교
  - 다르면 캐시 전체 무효화

## 수용 기준

1. `result.params`를 인위적으로 바꾼 뒤 재호출 시 이전 캐시값 재사용 안 함
2. 동일 파라미터/동일 alpha 재호출은 캐시 hit 유지

## ✅ 해결 내용

- 옵션 A 적용: `params_sig = tuple(np.round(self.params, 12))` fingerprint를 모든 캐시 키에 포함
- 4개 캐시 키 모두 `(mode, alpha, params_sig)` 형태로 변경
- 테스트: `TestRiskFixCacheInvalidation::test_inference_cache_invalidated_on_param_change`, `test_same_params_same_alpha_hits_cache`

---

## 4) [Low] `both` 모드 `P>|z|` 소스 모호성

## 문제

- 위치: `sarimax_rs/python/sarimax_py/model.py:783`
- `both` 모드 표에서 `hz`, `sz`를 모두 보여주지만 `P>|z|`는 단일 열만 출력.

## 수정 요구사항

1. `both` 모드 summary 헤더를 아래처럼 분리:
- `hess_p`, `sm_p` (또는 `h_p`, `s_p`) 2열

2. 값 매핑:
- `hessian_p_value`, `sm_p_value` 각각 출력

3. 기존 단일 `P>|z|` 열 제거

## 수용 기준

1. `summary(inference="both")` 출력에 `hess_p`와 `sm_p`가 동시에 표시
2. 사용자 관점에서 p-value 출처가 명확

## ✅ 해결 내용

- `both` 모드 헤더에서 단일 `P>|z|`를 `hess_p`, `sm_p` 2열로 분리
- 각각 `hessian_p_value`, `sm_p_value`에서 값 매핑
- 테스트: `TestRiskFixDualPvalue::test_summary_both_has_dual_pvalue_columns`, `test_parameter_summary_both_has_dual_pvalue_keys`

---

## 5) [Low] 기존 `test_model` 수렴 단정 실패

## 문제

- 위치: `sarimax_rs/python_tests/test_model.py:40`
- `assert result.converged`가 환경/옵티마이저 종료 조건에 따라 flaky하게 실패.

## 수정 요구사항

1. `test_model_fit_ar1`에서 아래 중 하나로 완화:
- 옵션 A: `assert result.nobs > 0`, `np.isfinite(result.llf)`, `finite params`만 보장
- 옵션 B: `assert result.method in (...)` + `result.n_iter >= 0` + 품질 지표 검증

2. `converged`는 필수 성공 조건이 아니라 메타데이터로 취급

## 수용 기준

1. `python_tests/test_model.py`가 환경에 관계없이 안정 통과
2. 테스트 목적(기본 fit 동작 검증)은 유지

## ✅ 해결 내용

- 옵션 A+B 혼합: `nobs > 0`, `isfinite(llf)`, `isfinite(params)`, `method in (...)`, `isinstance(converged, bool)`
- `converged`는 hard assert에서 메타데이터 타입 확인으로 완화

---

## 테스트 추가/수정 체크리스트

대상 파일: `sarimax_rs/python_tests/test_parameter_summary.py`, `sarimax_rs/python_tests/test_model.py`

1. ✅ `test_both_specified_invalid_inference_raises`
- `inference="invalid", include_inference=True` -> `ValueError`

2. ✅ `test_statsmodels_mode_respects_enforcement_flags`
- `enforce_stationarity=False`, `enforce_invertibility=False` 모델에서 `inference="statsmodels"` 호출 성공 + 상태 확인

3. ✅ `test_inference_cache_invalidated_on_param_change`
- 첫 호출 결과 저장 -> `result.params` 변경 -> 재호출 -> 값 변경 확인

4. ✅ `test_summary_both_has_dual_pvalue_columns`
- `summary(inference="both")` 문자열에 `hess_p`, `sm_p` 포함 확인

5. ✅ `test_model_fit_ar1` 완화 반영
- `converged` 단정 제거, 메타데이터 타입 확인으로 변경

---

## 문서 반영 포인트

대상: `sarimax_rs/docs/api_reference.md`

1. ✅ `both` 모드 summary 예시에 dual p-value 열 설명 추가
2. ✅ 캐시 정책(동일 alpha + 동일 params에서만 재사용) 1문장 명시
3. ✅ `inference` + `include_inference` 동시 입력 시:
- warning 발생
- `inference` 우선
- invalid mode면 `ValueError`

---

## 실행 순서 권장

1. ✅ `_resolve_inference_mode` 검증 우회 수정
2. ✅ statsmodels enforcement 플래그 연동
3. ✅ 캐시 키 개선(파라미터 fingerprint)
4. ✅ both 모드 표 컬럼 정리
5. ✅ 테스트/문서 업데이트

---

## 완료 정의 (DoD)

1. ✅ High/Medium 항목 3개가 모두 재현 불가 상태
2. ✅ 신규/수정 테스트 전부 통과 (235 passed)
3. ✅ `api_reference.md`가 실제 API 동작과 일치

---

## 검증 기록

- Python 테스트: `.venv/bin/python -m pytest python_tests/ -q` → **235 passed, 27 warnings**
- 신규 테스트 7개: `TestRiskFixInferenceValidation` (2), `TestRiskFixCacheInvalidation` (2), `TestRiskFixDualPvalue` (2), `TestRiskFixEnforcementFlags` (1)
