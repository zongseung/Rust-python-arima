# sarimax-rs v5 변경 내역

## 개요

P0(배포 차단 해소) + P2 UX(사용성 개선) 구현 완료.
핵심 변경: **trend 파라미터 전면 지원**, **Polars 기반 DataFrame**, **auto_arima**, **Rayon 병렬 grid search**.

---

## P0: 배포 차단 해소

### A-1. pyproject.toml 수정
**파일**: `pyproject.toml`

- `[tool.maturin]`에 `python-source = "python"` 추가 → wheel에 `sarimax_py` 패키지 포함
- `polars>=1.0` 의존성 추가 (pandas 대체)

### A-2. lib.rs — trend 파라미터 노출
**파일**: `src/lib.rs`

Rust 엔진 내부에서는 `Trend` enum을 완벽 지원했으나, PyO3 경계에서 `Trend::None`으로 하드코딩되어 있었음.

- `build_config()`에 `trend: Trend` 파라미터 추가 (하드코딩 제거)
- `parse_trend()` 헬퍼 추가: `Option<&str>` → `Trend::from_str()`
- **9개 PyO3 함수**에 `trend: Option<&str>` 추가:
  - `sarimax_loglike`, `sarimax_fit`, `sarimax_forecast`
  - `sarimax_residuals`, `sarimax_batch_loglike`, `sarimax_batch_fit`
  - `sarimax_batch_forecast`, `sarimax_inference`, `sarimax_diagnostics`

### A-3. model.py — Python API trend 지원
**파일**: `python/sarimax_py/model.py`

- `SARIMAXModel.__init__`에 `trend="n"` 파라미터 추가
- `fit()`, `forecast()`, `resid`, `diagnostics()`, `_loglike_fn()`, `_compute_rust_inference()` — 모두 `trend=self.trend` 전달

### A-4. forecast.rs — trend 미래 반영 수정
**파일**: `src/forecast.rs`

Kalman filter 단계에서는 state_intercept(`c_t`)이 정상 적용되었으나, **forecast 루프에서 누락**되어 있었음.

```
수정 전: a_{h+1} = T * a_h              (trend 무시)
수정 후: a_{h+1} = T * a_h + c_{n+h}    (trend 반영)
```

- `forecast()` 시그니처에 `config`, `params`, `n_obs` 추가
- 예측 루프에서 trend 종류별 state_intercept 계산:
  - Constant: `c = trend_coeffs[0]`
  - Linear: `c = trend_coeffs[0] * (n+h)`
  - Both: `c = trend_coeffs[0] + trend_coeffs[1] * (n+h)`
- inject 위치: `a[k_states_diff]` (ARMA state 시작점)

### A-5. param_names — trend 이름 생성
**파일**: `python/sarimax_py/model.py`

`_generate_param_names()`에 trend 파라미터 추가:
- `'c'` → `['intercept']`
- `'t'` → `['drift']`
- `'ct'` → `['intercept', 'drift']`
- 이름 순서: `[trend | exog | ar | ma | sar | sma]`

### A-6. trend 통합 테스트
**파일**: `python_tests/test_trend.py` (신규, 16개 테스트)

- `TestTrendFit` — trend='n','c','t','ct' 적합 + params 개수 검증
- `TestTrendParamNames` — 이름 생성 정확성
- `TestTrendForecast` — trend 적용된 forecast 유한성 + 방향성
- `TestTrendResiduals` — residuals 유한성
- `TestTrendSummary` — summary에 trend 표시
- `TestTrendDefault` — 기본값 'n' 검증

---

## P2 UX: 사용성 개선

### C-1. Polars DataFrame
**파일**: `python/sarimax_py/model.py`

pandas 대신 **Polars** 사용.

#### `ForecastResult.to_dataframe()`
```python
fc = result.forecast(steps=5)
df = fc.to_dataframe()
# shape: (5, 5) — columns: step, mean, variance, ci_lower, ci_upper
```

#### `SARIMAXResult.params_table()`
```python
pt = result.params_table(inference="hessian")
# shape: (k, 7) — columns: name, coef, std_err, z, p_value, ci_lower, ci_upper
```

#### `PredictionResult.to_dataframe()`
```python
pred = result.get_prediction(start=0, end=210)
df = pred.to_dataframe()
# shape: (210, 2) — columns: index, predicted_mean
```

**테스트**: `python_tests/test_polars.py` (신규, 14개 테스트)

### C-2. get_prediction — in-sample 예측
**파일**: `python/sarimax_py/model.py`

```python
pred = result.get_prediction(start=0, end=None, alpha=0.05, exog=None)
```

- in-sample: `endog - residuals` (1-step-ahead predictions)
- out-of-sample (`end > nobs`): forecast 결과 이어붙임
- 반환: `PredictionResult` (신규 클래스)

### C-3. summary() 개선
**파일**: `python/sarimax_py/model.py`

statsmodels 스타일 2열 레이아웃으로 전면 재작성:

```
==============================================================================
                               SARIMAX Results
==============================================================================
Model: SARIMAX(1,1,1)(0,0,0)[0]                   Log Likelihood:     -267.731
No. Observations: 200                              AIC:                541.462
Trend: n                                           BIC:                551.357
Method: lbfgsb                                     HQIC:               545.466
Converged: True                                     Scale:            0.863165
Date: 2026-02-24
------------------------------------------------------------------------------
                       coef    std err        z    P>|z|  [0.025  0.975]
------------------------------------------------------------------------------
           ar.L1     0.4932     0.4919   1.003   0.316  -0.471   1.457
           ma.L1    -0.5533     0.4684  -1.181   0.238  -1.471   0.365
==============================================================================
```

추가 사항:
- **HQIC** (Hannan-Quinn): `-2*ll + 2*k*ln(ln(n))`
- **날짜** 표시
- `inference="both"` 모드: hessian/statsmodels 비교 테이블

### C-4. CLAUDE.md 업데이트
**파일**: `CLAUDE.md`

- 테스트 수 업데이트 (Rust 149, Python 322)
- trend 지원 문서화
- Polars 의존성 추가
- Python layer 설명 갱신 (auto_arima, PredictionResult 등)
- param vector layout 수정: `[trend(kt) | exog(k) | ar(p) | ma(q) | sar(P) | sma(Q)]`

### C-5. auto_arima
**파일**: `python/sarimax_py/auto.py` (신규)

#### Stepwise (기본)
Hyndman-Khandakar 알고리즘:
1. 시작 모델: `(0,d,0)`, `(2,d,2)`, `(1,d,0)`, `(0,d,1)` + seasonal 후보
2. 1차 이웃 탐색: p±1, q±1, P±1, Q±1, (p±1,q±1) 동시
3. IC 개선 없으면 중단

#### Grid Search (Rayon 병렬)
- 모든 `(p,q,P,Q)` 조합 생성
- **`sarimax_rs.sarimax_grid_search()`** 단일 호출 → Rayon `par_iter()`로 병렬 fitting
- GIL 해제 상태에서 모든 조합 동시 실행

```python
from sarimax_py import auto_arima

res = auto_arima(y, max_p=5, max_q=5, s=12, stepwise=False, trace=True)
print(res.summary())
print(res.result.forecast(steps=12).to_dataframe())
print(res.history_dataframe())  # Polars DataFrame
```

#### 자동 차분 탐지
- `d`: ADF test (scipy 있을 때) 또는 분산 감소 휴리스틱
- `D`: seasonal autocorrelation 기반 (0 or 1)

#### AutoARIMAResult
- `.result` — 최적 `SARIMAXResult`
- `.history` — 탐색 이력 (list of dict)
- `.history_dataframe()` — Polars DataFrame
- `.summary()` — 요약 문자열

**테스트**: `python_tests/test_auto.py` (신규, 16개 테스트)

---

## Rayon 병렬 Grid Search

### 문제
`auto_arima(stepwise=False)` grid search가 Python `for` 루프로 조합마다 순차 호출.
각 조합은 독립적이므로 병렬화 가능.

### 해결: `sarimax_grid_search` Rust 함수

#### batch.rs — `grid_search_fit()`
```rust
pub fn grid_search_fit(
    endog: &[f64],
    configs: &[SarimaxConfig],   // 조합별 서로 다른 order
    method: Option<&str>,
    maxiter: Option<u64>,
    exog: Option<&[Vec<f64>]>,
) -> Vec<Result<FitResult>>
```
- `configs.par_iter().map(|config| optimizer::fit(endog, config, ...)).collect()`
- 에러 격리: 실패 조합은 `Err`, 다른 조합에 영향 없음

#### lib.rs — PyO3 래퍼
```rust
fn sarimax_grid_search(
    y: array,
    order_list: [(p,d,q), ...],
    seasonal_list: [(P,D,Q,s), ...],
    enforce_stationarity=True,
    enforce_invertibility=True,
    trend=None, method=None, maxiter=None, exog=None,
) -> list[dict]
```
- 각 결과 dict에 `order`, `seasonal_order` 키 포함
- 실패 조합: `{"error": "...", "converged": false}`

#### Python 사용 예
```python
import sarimax_rs

results = sarimax_rs.sarimax_grid_search(
    y,
    order_list=[(0,1,0), (1,1,0), (1,1,1), (2,1,1)],
    seasonal_list=[(0,0,0,0)] * 4,
    trend="c",
)
for r in results:
    if "error" not in r:
        print(f"{r['order']}: AIC={r['aic']:.3f}")
```

---

## 수정 파일 요약

| 파일 | Phase | 변경 내용 |
|------|-------|----------|
| `pyproject.toml` | A-1 | python-source, polars dep |
| `src/lib.rs` | A-2, Grid | trend param (9개 함수) + `sarimax_grid_search` |
| `src/forecast.rs` | A-4 | trend state_intercept in forecast loop |
| `src/batch.rs` | Grid | `grid_search_fit()` Rayon 병렬 |
| `python/sarimax_py/model.py` | A-3,5, C-1~3 | trend, Polars, prediction, summary |
| `python/sarimax_py/auto.py` | C-5, Grid | auto_arima (신규) |
| `python/sarimax_py/__init__.py` | C-5 | auto_arima, PredictionResult export |
| `python/sarimax_rs/__init__.py` | A-1 | maturin module 디렉토리 (신규) |
| `python_tests/test_trend.py` | A-6 | 16개 테스트 (신규) |
| `python_tests/test_polars.py` | C-1 | 14개 테스트 (신규) |
| `python_tests/test_auto.py` | C-5, Grid | 16개 테스트 (신규) |
| `CLAUDE.md` | C-4 | 문서 업데이트 |

## 테스트 결과

| 구분 | 통과 | 총계 | 비고 |
|------|------|------|------|
| Rust | 149 | 149 | +2 (grid_search) |
| Python | 322 | 323 | 1개 기존 batch flaky |
| **신규 Python** | **46** | **46** | trend 16 + polars 14 + auto 16 |
