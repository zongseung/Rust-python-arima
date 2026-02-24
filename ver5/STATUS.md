# sarimax-rs v5 — 현황 진단 및 구현 로드맵

> 진단일: 2026-02-24
> Rust 10,523줄 / Python 1,012줄 / 테스트 147(Rust) + 277(Python)

---

## 1. 현재 완성도 요약

### Rust 엔진 (src/) — 15 모듈, 완성

| 모듈 | 줄수 | 기능 | 상태 |
|------|-----|------|------|
| optimizer.rs | 2,247 | L-BFGS-B, NM, 다중시작, CSS 전최적화 | 완성 |
| score.rs | 1,193 | 접선선형 score (해석적 기울기) | 완성 |
| lib.rs | 1,006 | PyO3 진입점 (10개 함수 export) | **trend 미노출** |
| kalman.rs | 1,004 | 칼만필터, Chandrasekhar, 정상상태 감지 | 완성 |
| start_params.rs | 974 | Hannan-Rissanen, Burg, CSS 초기값 | 완성 |
| state_space.rs | 835 | Harvey 표현 T,Z,R,Q + 추세 state_intercept | 완성 |
| inference.rs | 797 | Hessian, OPG, 표준오차, 신뢰구간 | 완성 |
| initialization.rs | 524 | DARE, Lyapunov, 혼합, 근사확산 | 완성 |
| forecast.rs | 452 | h-step 예측 + 신뢰구간 | **trend 미래반영 없음** |
| batch.rs | 366 | Rayon 병렬 배치 fit/forecast/loglike | 완성 |
| params.rs | 337 | 모나한-존스 변환, 플랫 벡터 파싱 | 완성 |
| css.rs | 316 | 조건부제곱합 우도 | 완성 |
| polynomial.rs | 218 | 계절 다항식 곱셈 | 완성 |
| types.rs | 221 | 타입 정의, Trend enum (4종) | 완성 |
| error.rs | 33 | 에러 enum (8종) | 완성 |

### Python 래퍼 (python/sarimax_py/) — 미배포

| 파일 | 줄수 | 기능 | 상태 |
|------|-----|------|------|
| model.py | 1,008 | SARIMAXModel, SARIMAXResult, ForecastResult | **wheel에 미포함** |
| __init__.py | 4 | export | 동일 |

### 테스트 — 충분

| 범주 | 개수 | 비고 |
|------|------|------|
| Rust 단위 테스트 | 147 | 전 모듈 커버 |
| Python 통합 테스트 | 277 | 25개 파일, exog/s=24/배치 포함 |
| Fixture JSON | 5개 (1.5MB) | statsmodels 참조값 |

---

## 2. 문제점 분류

### P0 — 배포 차단 (이것 없으면 pip install 불가)

| # | 문제 | 위치 | 설명 |
|---|------|------|------|
| P0-1 | **sarimax_py가 wheel에 미포함** | pyproject.toml | `python-source = "python"` 한 줄 누락. pip install 해도 SARIMAXModel 사용 불가 |
| P0-2 | **trend 파라미터 PyO3 미노출** | lib.rs:build_config() | Rust 내부에 Trend enum 완성인데 `Trend::None`으로 하드코딩 |
| P0-3 | **Python API에 trend 없음** | model.py:SARIMAXModel | `__init__`에 trend 파라미터 자체가 없음 |
| P0-4 | **forecast에서 trend 미래 미반영** | forecast.rs | 필터링 시 c_t 적용하지만, h-step 예측에서 t=n+1,n+2... trend 미적용 |

### P1 — 기능 갭 (statsmodels 호환성)

| # | 문제 | 위치 | 설명 |
|---|------|------|------|
| P1-1 | **trend param_names 미생성** | model.py:_generate_param_names | trend='c'이면 `const`, 'ct'이면 `const`,`trend` 이름 필요 |
| P1-2 | **CI/CD 워크플로 없음** | .github/workflows/ | 디렉토리 자체 없음. PR 검증, 릴리즈 자동화 없음 |
| P1-3 | **diagnostics plot 없음** | model.py | `result.plot_diagnostics()` 미구현 (matplotlib 의존성은 pyproject.toml에 있음) |
| P1-4 | **Jupyter _repr_html_ 없음** | model.py | summary()가 텍스트 전용. 노트북에서 HTML 표 렌더링 안됨 |

### P2 — 편의 기능 (사용성)

| # | 문제 | 위치 | 설명 |
|---|------|------|------|
| P2-1 | **auto_arima 없음** | 미구현 | AIC/BIC 기반 차수 자동 탐색 |
| P2-2 | **pandas 연동 없음** | model.py | forecast 결과가 numpy만. DataFrame/Series 반환 옵션 없음 |
| P2-3 | **predict (in-sample) 없음** | model.py | statsmodels의 `get_prediction(start, end)` 미구현 |
| P2-4 | **model summary 상단 정보 부족** | model.py:summary() | Date, Dep Variable, Model 표기 등 statsmodels 형식 미달 |
| P2-5 | **CLAUDE.md 구버전 정보** | CLAUDE.md | "89 tests" → 실제 147개, "44 pytest" → 실제 277개, exog "NotImplementedError" → 이미 구현됨 |

### P3 — 고급/미래 (v5 이후)

| # | 문제 | 설명 |
|---|------|------|
| P3-1 | exact diffuse 초기화 | Durbin-Koopman 확산 칼만필터 (현재는 근사확산) |
| P3-2 | WASM 빌드 | 브라우저 예측 |
| P3-3 | Arrow/Polars 연동 | 데이터 엔지니어링 파이프라인 |
| P3-4 | 잔차 진단 통계 확장 | ACF/PACF plot, QQ plot |

---

## 3. 구현 계획

### Phase A — 배포 차단 해소 (P0, 필수)

```
A-1  pyproject.toml에 python-source 추가          예상: 10분
A-2  lib.rs에 trend 파라미터 노출 (모든 PyO3 함수)  예상: 1시간
A-3  model.py에 trend 파라미터 추가                 예상: 30분
A-4  forecast.rs에 trend 미래반영 구현              예상: 1시간
A-5  param_names에 trend 이름 추가                  예상: 20분
A-6  trend 통합 테스트 작성                         예상: 1시간
```

### Phase B — statsmodels 호환 (P1)

```
B-1  CI/CD 워크플로 작성 (ci.yml, release.yml)     예상: 2시간
B-2  plot_diagnostics() 구현 (4-패널)              예상: 2시간
B-3  _repr_html_() 구현 (summary/forecast)         예상: 1시간
B-4  maturin build + wheel 테스트                  예상: 30분
```

### Phase C — 사용성 (P2)

```
C-1  pandas DataFrame 반환 옵션                    예상: 1시간
C-2  get_prediction(start, end) in-sample 예측     예상: 2시간
C-3  summary() 포맷 개선 (statsmodels 스타일)      예상: 1시간
C-4  CLAUDE.md 업데이트                             예상: 20분
C-5  auto_arima (AIC grid search)                  예상: 4시간
```

---

## 4. 의존 관계

```
A-1 ─────────────────────────────────────→ B-4 (wheel 빌드 검증)
A-2 → A-3 → A-4 → A-5 → A-6 (trend 체인)
A-6 ─────────────────────────────────────→ B-1 (CI에 trend 테스트 포함)
B-2 (plot) ←── matplotlib 이미 devdeps에 있음
B-3 (_repr_html_) ←── 독립
C-1 (pandas) ←── 독립
C-2 (in-sample) ←── forecast.rs 이해 필요
C-5 (auto_arima) ←── A-2,A-3 완료 후 (trend 포함 탐색)
```

---

## 5. 파일별 수정 요약

| 파일 | Phase | 변경 내용 |
|------|-------|----------|
| `pyproject.toml` | A-1 | `python-source = "python"` 추가 |
| `src/lib.rs` | A-2 | build_config()에 trend 파라미터, 모든 #[pyfunction]에 trend: Option<&str> |
| `python/sarimax_py/model.py` | A-3,B-2,B-3,C-1,C-2,C-3 | trend, plot, html, pandas, predict |
| `src/forecast.rs` | A-4 | forecast_pipeline()에 state_intercept 미래반영 |
| `python_tests/test_trend.py` | A-6 | trend='c','t','ct' 적합/예측/잔차 테스트 |
| `.github/workflows/ci.yml` | B-1 | push/PR 트리거, rust-test + python-test |
| `.github/workflows/release.yml` | B-1 | tag 트리거, 다중 플랫폼 wheel 빌드 |
| `CLAUDE.md` | C-4 | 테스트 수, exog 상태, trend 지원 반영 |
