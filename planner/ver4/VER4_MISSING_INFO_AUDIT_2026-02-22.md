# ver4 누락 정보 구성 점검 (2026-02-22)

대상: `sarimax_rs/python/sarimax_py/model.py`, `sarimax_rs/docs/api_reference.md`, `sarimax_rs/python_tests`  
목적: ver4 구현 이후 남아있는 누락/불일치 항목을 문서화

---

## 요약

- ver4 핵심 기능(파라미터 이름/요약 테이블/수치추론)은 구현됨.
- 다만 배포 전 반드시 정리해야 할 누락 항목이 3개 있음:
1. `parameter_summary/summary`의 `alpha` 경계값 검증 누락 (High)
2. API 문서 미동기화 (Medium)
3. 기존 `test_model` 수렴 테스트 1건 실패 상태 지속 (Low, 기존 이슈 연동)

---

## Findings

## 1) [High] `parameter_summary`/`summary`의 `alpha` 검증 누락

- 상태: ✅ `Resolved` (2026-02-22, commit f76a7f7)
- 근거 코드:
  - `sarimax_rs/python/sarimax_py/model.py:403` (`parameter_summary(alpha=...)`)
  - `sarimax_rs/python/sarimax_py/model.py:503` (`summary(alpha=...)`)
  - 현재 위 경로에서 `0 < alpha < 1` 검증 없음
- 영향:
  - 잘못된 `alpha` 입력이 예외 없이 통과
  - CI가 역전되거나(`lower > upper`) `inf/-inf`, `nan` 발생 가능
  - 그럼에도 `inference_status="ok"`로 표시될 수 있어 품질 메타데이터 오염
- 재현 예:
  - `alpha=2` -> `ci_lower=inf`, `ci_upper=-inf`
  - `alpha=1.5` -> 하한/상한 역전 가능
  - `alpha=-0.1` -> `nan` CI 가능
- 권장 조치:
  - `parameter_summary`와 `summary` 진입부에 `if not (0.0 < alpha < 1.0): raise ValueError(...)` 추가
  - `python_tests/test_parameter_summary.py`에 invalid-alpha negative test 추가

## 2) [Medium] API 문서 미동기화

- 상태: ✅ `Resolved` (2026-02-22, commit f76a7f7)
- 근거 코드/문서:
  - 구현 존재: `sarimax_rs/python/sarimax_py/model.py:369` (`param_names`)
  - 구현 존재: `sarimax_rs/python/sarimax_py/model.py:403` (`parameter_summary`)
  - 구현 존재: `sarimax_rs/python/sarimax_py/model.py:503` (`summary(alpha, include_inference)`)
  - 문서 미반영: `sarimax_rs/docs/api_reference.md:214`~`sarimax_rs/docs/api_reference.md:217`
- 영향:
  - 사용자 관점에서 새 API discoverability 저하
  - 문서 기반 사용 시 런타임 동작과 차이 발생
- 권장 조치:
  - `SARIMAXResult` 섹션에 다음 추가
    - `param_names` 속성
    - `parameter_summary(alpha=0.05, include_inference=True)`
    - `summary(alpha=0.05, include_inference=False)`
  - `alpha` 유효범위/추론 계산 비용/실패 시 degraded status 명시

## 3) [Low] 기존 `test_model`의 수렴 단정 실패 지속

- 상태: ✅ `Resolved` (2026-02-22)
- 근거:
  - 실패 위치: `sarimax_rs/python_tests/test_model.py:40` (`assert result.converged`)
  - 관측값: AR(1)에서도 `method='lbfgsb'`, `n_iter=21`, `converged=False` 케이스 존재
- 원인: `uv run`이 `maturin develop --release`로 빌드된 .so를 캐시된 stale 버전으로 덮어씌움
  - `maturin develop`은 `.venv/` 에 정상 설치하나, `uv run`이 pyproject.toml 기반으로 재sync하면서 stale wheel 재설치
  - 직접 `.venv/bin/python`으로 실행하면 `converged=True` 정상 동작 확인
- 해결: 빌드 환경 문제이며 코드 버그 아님. 228 tests all pass with direct venv python.

---

## 검증 기록

1. 신규 ver4 테스트
- 명령: `.venv/bin/python -m pytest python_tests/test_parameter_summary.py -q`
- 결과: `52 passed` (inference enum 확장 후)

2. 전체 통합 테스트
- 명령: `.venv/bin/python -m pytest python_tests/ -v`
- 결과: `228 passed, 26 warnings` (전수 통과)
- 주의: `uv run python` 대신 `.venv/bin/python` 사용 필수 (uv run이 stale wheel 재설치하는 문제)

3. Rust 테스트
- 명령: `cargo test --all-targets`
- 결과: `109 passed` (test_fit_ar1_lbfgsb_convergence 포함)

---

## 우선순위 권고

1. `alpha` 검증 누락 수정 (High)
2. `api_reference.md` 문서 동기화 (Medium)
3. `test_model` 수렴 단정 정책 재정의 (Low)

