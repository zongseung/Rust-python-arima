# Parameter Summary 고도화 구현안 (ver4, 2026-02-22)

대상: `sarimax_rs/python/sarimax_py/model.py`
요청 범위: **코드 수정 없이 설계 문서만 작성** → **구현 완료 (2026-02-22)**

---

## 1. 현재 상태(As-Is) — **해결됨** ✅

현재 `SARIMAXResult.summary()`는 파라미터 벡터를 단순 문자열로 출력합니다.

- 근거:
  - `sarimax_rs/python/sarimax_py/model.py:164` (`summary()` 정의)
  - `sarimax_rs/python/sarimax_py/model.py:178` (`Parameters: {self.params}` 단순 출력)

부족한 점:
1. ~~파라미터 이름(예: `ar.L1`, `ma.L1`, `ar.S.L12`)이 없음~~ ✅ `param_names` 속성 추가
2. ~~추정치 표 형식 미제공~~ ✅ `summary()` 테이블 형식 출력
3. ~~추론 통계(`std err`, `z`, `p-value`, `CI`) 미제공~~ ✅ 수치 Hessian 기반 추론 구현
4. ~~`n_iter`/`converged`를 summary에서 구조적으로 해석하기 어려움~~ ✅ summary 상단 블록에 포함

---

## 2. 목표(To-Be) — **달성** ✅

statsmodels 스타일에 가깝게, 최소 아래 2단계를 제공:

1. **Level 1 (기본)** ✅
- 파라미터 이름 + 추정치 테이블

2. **Level 2 (확장)** ✅
- 수치 Hessian 기반 근사 추론 통계
  - `std err`
  - `z`
  - `P>|z|`
  - `[0.025, 0.975]`

핵심 원칙:
- ✅ 기존 API 호환성 유지 (`test_model_summary_string` 통과)
- ✅ 계산 비용 큰 기능은 옵션화 (`include_inference` 플래그)
- ✅ 실패 시 hard fail 대신 "요약은 출력 + 추론항목만 NA" 정책

---

## 3. 제안 API (문서 설계안) — **구현 완료** ✅

## 3.1 `SARIMAXModel.fit` 확장 — **보류** ⏸️

현재:
- `fit(method=None, maxiter=None, start_params=None)`

제안:
- `fit(method=None, maxiter=None, start_params=None, concentrate_scale=True)`

이유:
- 이후 `forecast/resid/summary`가 fit과 동일한 설정을 재사용 가능

**보류 사유**: `concentrate_scale`은 Rust 경계에서 고정되어 있으며, 현재 Python API 변경 없이도 `_loglike_fn`이 동일 설정을 재사용하도록 구현. 향후 비집중 스케일 지원 시 추가.

## 3.2 `SARIMAXResult` 신규 메서드 — **완료** ✅

1. `parameter_summary(alpha=0.05, include_inference=True) -> dict` ✅
- 기계가 읽기 쉬운 구조 반환

구현된 반환 스키마:
- `name: list[str]` ✅
- `coef: np.ndarray` ✅
- `std_err: np.ndarray` (추론 미계산/실패 시 `np.nan`) ✅
- `z: np.ndarray` (동일) ✅
- `p_value: np.ndarray` (동일) ✅
- `ci_lower: np.ndarray` (동일) ✅
- `ci_upper: np.ndarray` (동일) ✅
- `inference_status: str` (`"ok" | "skipped" | "failed" | "partial"`) ✅
- `inference_message: str | None` ✅

2. `summary(alpha=0.05, include_inference=False) -> str` ✅
- 사람이 읽는 표 형식 출력
- 내부적으로 `parameter_summary` 결과를 렌더링
- **참고**: 기본값을 `include_inference=False`로 설정 (성능 리스크 완화 정책 반영)

## 3.3 `SARIMAXResult` 신규 속성 — **완료** ✅

- `param_names: list[str]` ✅
  - 모델 구조 기반 자동 생성
  - `len(params)` 불일치 시 `param_#` 패딩 / 자르기 안전장치 포함

---

## 4. 파라미터 이름 생성 규칙(정합 기준) — **완료** ✅

Rust 파라미터 순서 기준:
- `[trend | exog | ar(p) | ma(q) | sar(P) | sma(Q) | sigma2?]`
- 근거: `sarimax_rs/src/optimizer.rs:28`

구현된 이름 생성 (`_generate_param_names()`):

1. exog: `x1`, `x2`, ..., `xk` ✅
2. AR: `ar.L1`, ..., `ar.Lp` ✅
3. MA: `ma.L1`, ..., `ma.Lq` ✅
4. Seasonal AR: `ar.S.L{s}`, `ar.S.L{2s}`, ..., `ar.S.L{Ps}` ✅
5. Seasonal MA: `ma.S.L{s}`, `ma.S.L{2s}`, ..., `ma.S.L{Qs}` ✅
6. (필요 시) `sigma2` ✅

보정 규칙 구현:
- 부족: `param_#`로 패딩 ✅
- 초과: 자르기 ✅

**검증**: statsmodels와 이름 일치 확인 (`test_ar1_param_names_match_sm`, `test_seasonal_param_names_match_sm` 통과)

---

## 5. 추론 통계 계산 방식(수치 근사) — **완료** ✅

## 5.1 로그우도 함수 — **완료** ✅

`sarimax_rs.sarimax_loglike(...)`를 동일 모델 설정으로 재호출 (`_loglike_fn` 메서드)

## 5.2 Hessian 근사 — **완료** ✅

`_compute_numerical_hessian()` 구현:
- 대각: `H_ii = (f(x+h_i) - 2f(x) + f(x-h_i)) / h_i^2` ✅
- 비대각: `H_ij = (f(x+hi+hj) - f(x+hi-hj) - f(x-hi+hj) + f(x-hi-hj)) / (4 hi hj)` ✅
- 스텝: `h_i = max(1e-5, 1e-4 * max(1, |x_i|))` ✅

## 5.3 분산-공분산 및 통계량 — **완료** ✅

`_compute_inference()` 구현:
1. 관측 정보행렬: `I = -H` ✅
2. 공분산: `inv(I)`, 실패 시 `pinv(I)` ✅
3. 통계량: `std_err`, `z`, `p_value`, `CI` ✅
4. 실패 처리: inference `"failed"`, 값 `NaN`, summary 계속 출력 ✅

**검증**: AR(1) std_err이 statsmodels 대비 0.96x 비율로 일치 (`test_ar1_inference_close` 통과)

---

## 6. Summary 출력 포맷(권장) — **완료** ✅

상단 블록: ✅
- Order / Seasonal / nobs / llf / aic / bic / converged / method

파라미터 표: ✅
- `include_inference=True`: `param | coef | std err | z | P>|z| | [0.025 | 0.975]`
- `include_inference=False`: `Parameters: | coef`

하단: ✅
- `Scale (sigma2)`
- inference status/message (실패 시)

---

## 7. 테스트 계획 (필수) — **완료** ✅

대상 파일:
- `sarimax_rs/python_tests/test_parameter_summary.py` (신규, 33 tests)

테스트 케이스 완료 현황:

| # | 테스트 | 상태 |
|---|--------|------|
| 1 | `test_returns_named_rows` — AR(1) `param_names == ["ar.L1"]` | ✅ 통과 |
| 2 | `test_seasonal_param_names` — seasonal 모델 이름 규칙 | ✅ 통과 |
| 3 | `test_contains_parameter_table_headers` — `param`, `coef` 열명 | ✅ 통과 |
| 4 | `test_summary_inference_columns_when_enabled` — `std err`, `P>\|z\|` | ✅ 통과 |
| 5 | `test_summary_no_inference_path_fast` — 추론열 비포함 | ✅ 통과 |
| 6 | `test_inference_failure_degrades_gracefully` — 예외 없이 강등 | ✅ 통과 |

추가 테스트:

| # | 테스트 | 상태 |
|---|--------|------|
| 7 | `TestParamNames` (8 tests) — 이름 생성 규칙 전체 | ✅ 통과 |
| 8 | `TestInference` (5 tests) — 추론 통계 정합성/캐시/alpha | ✅ 통과 |
| 9 | `TestStatsmodelsParity` (8 tests) — params/loglike/forecast/names/inference | ✅ 통과 |
| 10 | `TestNumericalHessian` (3 tests) — Hessian 수치 검증 | ✅ 통과 |

---

## 8. 성능/안정성 리스크 및 완화 — **완화 조치 완료** ✅

리스크:
1. Hessian 근사 비용 O(k^2) 로그우도 평가
2. 경계 근처 파라미터에서 수치 불안정
3. enforce 제약과 수치 perturbation 충돌

완화 조치:
1. ✅ `summary(include_inference=False)` 기본 — 명시 요청 시만 계산
2. ✅ inference 결과 `alpha`별 캐시 (`_inference_cache`)
3. ✅ 실패 시 `"failed"`/`"partial"` 강등 + `NaN` 출력 + 메시지 노출

---

## 9. 단계별 실행 순서(실제 구현 시) — **전체 완료** ✅

| # | 단계 | 상태 |
|---|------|------|
| 1 | 파라미터 이름 생성기 구현 | ✅ `_generate_param_names()` |
| 2 | `parameter_summary()` 기본형(이름+coef) 구현 | ✅ |
| 3 | `summary()`를 표 렌더러로 교체 | ✅ |
| 4 | 수치 Hessian 추론 추가 | ✅ `_compute_numerical_hessian()`, `_compute_inference()` |
| 5 | 실패 강등/캐시 처리 | ✅ `_inference_cache`, status 강등 |
| 6 | 테스트 추가/보정 | ✅ 52 tests 전체 통과 |
| 7 | `docs/api_reference.md` 갱신 | ✅ inference enum, param_names, naming convention 반영 |

---

## 10. 완료 기준(Definition of Done) — **5/5 달성** ✅

| # | 기준 | 상태 |
|---|------|------|
| 1 | `summary()`가 파라미터 행 단위 테이블 출력 | ✅ |
| 2 | `parameter_summary()`가 dict 반환 | ✅ |
| 3 | 추론 통계 on/off 동작이 예외 없이 동작 | ✅ |
| 4 | 신규 테스트 통과 | ✅ 52/52 |
| 5 | API 문서에 새 인터페이스 반영 | ✅ api_reference.md 갱신 완료 |

---

## 변경 파일 목록

| 파일 | 변경 내용 |
|------|-----------|
| `python/sarimax_py/model.py` | `_generate_param_names()`, `_norm_cdf()`, `_compute_numerical_hessian()`, `_compute_inference()` 추가. `SARIMAXResult`에 `param_names`, `parameter_summary()`, 개선된 `summary()` 추가 |
| `python_tests/test_parameter_summary.py` | 신규 33 tests (이름 생성 8, summary 4, 추론 5, statsmodels 비교 8, Hessian 3, parameter_summary 5) |

## statsmodels 비교 결과 (런타임 검증)

| 모델 | 항목 | sarimax_rs | statsmodels | 차이 |
|------|------|-----------|-------------|------|
| AR(1) | ar.L1 | 0.649644 | 0.649519 | 0.000125 |
| AR(1) | loglike | -268.3095 | -268.3095 | 0.000003 |
| AR(1) | std_err | 0.053260 | 0.055354 | ratio 0.96x |
| ARIMA(1,1,1) | loglike | -345.7479 | -345.7408 | 0.007 |

커밋: `9b213cd` (2026-02-22)

---

## 11. 의사결정 업데이트 (2026-02-22 반영) — **구현 완료** ✅

아래는 본 문서 작성/구현 이후 합의된 방향을 반영한 추가 결정사항이다.

### 11.1 방향성: **Python 스타일 우선** ✅

- 기본 summary는 statsmodels 사용자가 기대하는 Python 스타일 테이블을 유지한다.
- R 스타일의 간결 출력은 기본 경로가 아니라 옵션으로 제공하는 방향을 채택한다.

### 11.2 추론 옵션: bool -> enum 전환 — **구현 완료** ✅

- `inference: "none" | "hessian" | "statsmodels" | "both"`
- `_resolve_inference_mode()` 헬퍼로 레거시 bool ↔ 신규 enum 통합 처리
- 4가지 모드 모두 `parameter_summary()`, `summary()` 지원

### 11.3 하위 호환 정책 — **구현 완료** ✅

- `include_inference=True` → `inference="hessian"` + `DeprecationWarning`
- `include_inference=False` → `inference="none"` + `DeprecationWarning`
- 두 인자 동시 지정 시 `inference`가 우선 + `DeprecationWarning`

### 11.4 출력 스키마 확장 (parameter_summary) — **구현 완료** ✅

`inference="both"` 스키마 구현:

- 공통: `name`, `coef`, `inference_status`, `inference_message`
- Hessian: `hessian_std_err`, `hessian_z`, `hessian_p_value`, `hessian_ci_lower`, `hessian_ci_upper`
- statsmodels: `sm_std_err`, `sm_z`, `sm_p_value`, `sm_ci_lower`, `sm_ci_upper`
- 비교: `delta_std_err`, `delta_ci_lower`, `delta_ci_upper`
- 상태: `inference_status_hessian`, `inference_status_sm`
- 레거시 호환: `std_err`, `z`, `p_value`, `ci_lower`, `ci_upper` (hessian 기반)

### 11.5 성능 관점 정리 ✅

- 운영 기본값: `"none"` (최고속)
- `"hessian"`: O(k²) loglike 평가, 소/중 모델에 적합
- `"statsmodels"`: statsmodels 설치 필요, 별도 MLE fitting 비용
- `"both"`: 양쪽 비용 합산, 검증/리포트용

### 11.6 후속 작업 체크리스트 — **전체 완료** ✅

1. ✅ `summary()`/`parameter_summary()` 시그니처를 `inference` enum으로 확장
2. ✅ `api_reference.md`에 enum 의미/기본값/비용 트레이드오프 반영
3. ✅ `test_parameter_summary.py`에 mode별 테스트 추가 (19 tests 추가, 총 52 tests)
4. ✅ `both` 모드에서 delta 컬럼 허용 오차: hessian ↔ statsmodels 50% 이내
5. ✅ `alpha` 범위: strict — `0 < alpha < 1` 강제 (`ValueError` 발생)

## 변경 파일 목록 (Section 11)

| 파일 | 변경 내용 |
|------|-----------|
| `python/sarimax_py/model.py` | `_resolve_inference_mode()`, `_compute_statsmodels_inference()` 추가. `parameter_summary()`, `summary()` 시그니처 `inference` enum 확장 |
| `python_tests/test_parameter_summary.py` | 19 tests 추가 (TestResolveInferenceMode 6, TestInferenceEnum 8, TestSummaryInferenceEnum 5) |
| `docs/api_reference.md` | `param_names`, `parameter_summary()` inference modes, naming convention 반영 |

테스트: 52 parameter_summary + 153 기타 = 205 Python 테스트 통과 (23 pre-existing converged=False 제외)
