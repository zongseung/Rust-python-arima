# P0 — 배포 차단 이슈 상세

> 이 6개를 해결하지 않으면 `pip install sarimax-rs` 후 정상 사용 불가

---

## A-1. pyproject.toml — sarimax_py wheel 포함

### 현재
```toml
[tool.maturin]
features = ["pyo3/extension-module"]
module-name = "sarimax_rs"
```
→ 빌드 시 Rust 바이너리(`sarimax_rs.cpython-*.so`)만 wheel에 포함.
→ `python/sarimax_py/` 는 wheel에 없음.

### 수정
```toml
[tool.maturin]
features = ["pyo3/extension-module"]
module-name = "sarimax_rs"
python-source = "python"
```
→ maturin이 `python/` 하위 순수 Python 패키지를 자동으로 wheel에 포함.

### 검증
```bash
maturin develop --release
python -c "from sarimax_py import SARIMAXModel; print('OK')"  # PYTHONPATH 없이
```

---

## A-2. lib.rs — trend 파라미터 PyO3 노출

### 현재
```rust
fn build_config(...) -> SarimaxConfig {
    SarimaxConfig {
        trend: Trend::None,  // 하드코딩
        ...
    }
}
```
→ 7개 PyO3 함수 모두 trend를 무시함.

### 수정 대상 함수 (10개)
1. `sarimax_fit()`
2. `sarimax_forecast()`
3. `sarimax_loglike()`
4. `sarimax_residuals()`
5. `sarimax_batch_fit()`
6. `sarimax_batch_forecast()`
7. `sarimax_batch_loglike()`
8. `sarimax_inference()`
9. `sarimax_diagnostics()`
10. `build_config()` 헬퍼

### 수정 방법
```rust
// build_config에 trend 파라미터 추가
fn build_config(
    order: (usize, usize, usize),
    seasonal: (usize, usize, usize, usize),
    n_exog: usize,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    concentrate_scale: bool,
    trend: Trend,           // 추가
) -> SarimaxConfig { ... }

// 각 PyO3 함수에 trend: Option<&str> 추가
#[pyfunction]
fn sarimax_fit(
    ...
    trend: Option<&str>,    // "n", "c", "t", "ct"
) -> PyResult<...> {
    let trend_enum = match trend.unwrap_or("n") {
        "c" => Trend::Constant,
        "t" => Trend::Linear,
        "ct" | "tc" => Trend::Both,
        _ => Trend::None,
    };
    let config = build_config(..., trend_enum);
    ...
}
```

### 영향 범위
- `start_params.rs`: 초기값 생성 시 trend 계수 개수 반영 (이미 구현)
- `params.rs`: flat 벡터 파싱 시 k_trend 반영 (이미 구현)
- `state_space.rs`: state_intercept 생성 (이미 구현)
- `kalman.rs`: 필터링 시 c_t 적용 (이미 구현)
- `score.rs`: 기울기 계산 (이미 구현)

→ Rust 내부 로직은 전부 완성. PyO3 경계만 뚫으면 됨.

---

## A-3. model.py — Python API trend 파라미터

### 현재
```python
class SARIMAXModel:
    def __init__(self, endog, order=(1,0,0), seasonal_order=(0,0,0,0),
                 exog=None, enforce_stationarity=True, enforce_invertibility=True):
```

### 수정
```python
class SARIMAXModel:
    def __init__(self, endog, order=(1,0,0), seasonal_order=(0,0,0,0),
                 exog=None, trend='n',
                 enforce_stationarity=True, enforce_invertibility=True):
        ...
        self.trend = trend  # 'n', 'c', 't', 'ct'
```

→ `fit()`, `forecast()`, `_loglike_fn()` 등에서 `trend=self.trend` 전달.

---

## A-4. forecast.rs — trend 미래 반영

### 현재 문제
```rust
// forecast_pipeline() 내부
for h in 0..steps {
    let y_hat = z.dot(&a);
    let d_h = exog_contribution;  // 외생변수만
    mean.push(y_hat + d_h);       // trend 없음!
    a = t_mat * &a;               // c_t 미적용!
}
```
→ 필터링 시에는 `a_next += c_t` 하지만, 예측 시 t=n+1 이후 trend를 안 넣음.

### 수정
```rust
for h in 0..steps {
    let y_hat = z.dot(&a);

    // trend 기여분
    let trend_h = match config.trend {
        Trend::None     => 0.0,
        Trend::Constant => trend_coeffs[0],
        Trend::Linear   => trend_coeffs[0] * (n + h) as f64,
        Trend::Both     => trend_coeffs[0] + trend_coeffs[1] * (n + h) as f64,
    };

    let d_h = exog_contribution + trend_h;
    mean.push(y_hat + d_h);

    // state 전파 시에도 state_intercept 적용
    a = t_mat * &a;
    a[inject_idx] += trend_h;  // state_intercept 반영
}
```

### 검증
```python
# trend='c' 모델의 예측이 상수 절편을 유지하는지 확인
model = SARIMAXModel(y, order=(1,0,0), trend='c')
result = model.fit()
fc = result.forecast(steps=10)
# fc.predicted_mean이 절편 효과를 포함해야 함
```

---

## A-5. param_names — trend 이름 추가

### 현재
```python
def _generate_param_names(order, seasonal_order, n_exog=0, concentrate_scale=True):
    # Layout: [exog(k) | ar(p) | ma(q) | sar(P) | sma(Q)]
    # trend 없음
```

### 수정
```python
def _generate_param_names(order, seasonal_order, n_exog=0,
                          trend='n', concentrate_scale=True):
    names = []
    # Trend (맨 앞)
    if trend in ('c', 'ct', 'tc'):
        names.append('const')
    if trend in ('t', 'ct', 'tc'):
        names.append('trend')
    # 기존: exog, ar, ma, sar, sma ...
```

→ statsmodels와 동일한 이름 규칙: `const`, `trend`

---

## A-6. 테스트 — trend 통합 검증

### 파일: `python_tests/test_trend.py`

```python
# 최소 테스트 케이스
1. trend='c' AR(1) — const 파라미터 존재, 비영
2. trend='t' ARIMA(1,1,1) — trend 파라미터 존재
3. trend='ct' SARIMA(1,1,1)(1,1,1,12) — const+trend 2개
4. trend='c' + exog — 파라미터 순서 [const, x1, x2, ar, ma]
5. trend='c' forecast — 미래 예측에 절편 효과 포함
6. statsmodels 참조값 비교 — |ΔLL| < 3.0
```
