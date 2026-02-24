# P2 — 사용성 개선 상세

---

## C-1. pandas DataFrame 반환

### 현재
```python
fc = result.forecast(steps=12)
fc.predicted_mean   # numpy array
fc.ci_lower         # numpy array
```

### 개선
```python
fc = result.forecast(steps=12)
fc.to_dataframe()
#   step  predicted_mean  ci_lower  ci_upper
# 0    1          12.34      10.2     14.5
# 1    2          12.56      10.0     15.1
# ...
```

구현:
```python
class ForecastResult:
    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame({
            'predicted_mean': self.predicted_mean,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
        }, index=range(1, len(self.predicted_mean) + 1))
```

SARIMAXResult에도:
```python
class SARIMAXResult:
    def params_table(self, inference='hessian'):
        import pandas as pd
        ps = self.parameter_summary(inference=inference)
        return pd.DataFrame({
            'coef': ps['coef'],
            'std_err': ps['std_err'],
            'z': ps['z'],
            'p_value': ps['p_value'],
        }, index=ps['name'])
```

---

## C-2. get_prediction (in-sample 예측)

### statsmodels API
```python
pred = result.get_prediction(start=100, end=250)
pred.predicted_mean   # 길이 151
pred.conf_int()       # (151, 2)
```

### 구현 방안
- `sarimax_rs.sarimax_filter()` — Rust에서 전체 필터 상태 반환 (이미 kalman_filter() 존재)
- 또는 `sarimax_residuals()`의 내부 상태를 확장하여 filtered_mean 반환

```python
def get_prediction(self, start=None, end=None, alpha=0.05):
    """In-sample + out-of-sample 예측."""
    # 1) 전체 in-sample filtered states 구함
    full_filter = sarimax_rs.sarimax_filter(...)  # 새 API 필요

    # 2) start:end 구간 슬라이싱
    # 3) end > n이면 h-step forecast 추가
    return PredictionResult(mean, var, ci_lower, ci_upper)
```

**필요한 Rust 추가**: `sarimax_filter()` PyO3 함수 (kalman_filter의 전체 상태 반환)

---

## C-3. summary() 포맷 개선

### 현재
```
SARIMAX Results
==============================================================================
  Order:           (1, 1, 1)
  Seasonal:        (0, 0, 0, 0)
  Observations:    200
  ...
```

### statsmodels 스타일 목표
```
                               SARIMAX Results
==============================================================================
Dep. Variable:                      y   No. Observations:                  200
Model:               ARIMA(1, 1, 1)   Log Likelihood:                -267.731
Date:                Mon, 24 Feb 2026   AIC:                           541.462
Time:                        14:30:00   BIC:                           551.357
Sample:                             0   HQIC:                          545.423
                                  200   Scale:                           0.863
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1          0.4932      0.4919      1.003      0.316     -0.471       1.457
ma.L1         -0.5533      0.4684     -1.181      0.238     -1.471       0.365
==============================================================================
Ljung-Box (L1) (Q):    0.00   Jarque-Bera (JB):    0.80
Prob(Q):               0.96   Prob(JB):             0.67
Heteroskedasticity(H): 1.00   Skew:                -0.12
Prob(H) (two-sided):   1.00   Kurtosis:             2.86
==============================================================================
```

### 구현 포인트
- 상단 2열 레이아웃 (변수명/관측수, 모델/LL, 날짜/AIC 등)
- 하단 진단 통계 (이미 `diagnostics()` 구현됨)
- HQIC 계산 추가: `hqic = -2*ll + 2*k*ln(ln(n))`

---

## C-4. CLAUDE.md 업데이트

### 수정 필요 항목
| 항목 | 현재 (오래됨) | 실제 |
|------|-------------|------|
| Rust 테스트 수 | "89 tests" | 147 tests |
| Python 테스트 수 | "44 pytest" | 277 tests |
| exog 상태 | "NotImplementedError" | 완전 지원 |
| std errors | "not available" | Hessian/OPG 기반 가용 |
| PyO3 함수 수 | "7 functions" | 10 functions |
| 모듈 목록 | inference.rs 누락 | 15 modules |

---

## C-5. auto_arima (AIC grid search)

### 설계
```python
from sarimax_py import auto_arima

result = auto_arima(
    y,
    max_p=5, max_q=5, max_P=2, max_Q=2, s=12,
    d=None,        # None이면 자동 (KPSS test)
    D=None,        # None이면 자동 (OCSB test)
    criterion='aic',
    stepwise=True, # True: stepwise (빠름), False: 전탐색
    n_jobs=-1,     # Rayon 병렬
)
```

### 구현 전략
1. **차분 차수 자동 결정**: 단위근 검정 (ADF/KPSS → d), 계절 단위근 (OCSB → D)
2. **Stepwise 탐색**: Hyndman-Khandakar 알고리즘
   - 기본 모델 (0,d,0)(0,D,0,s), (2,d,2)(1,D,1,s), (1,d,0)(1,D,0,s) 등
   - 1차 이웃 탐색: p±1, q±1, P±1, Q±1
   - AIC 개선 없으면 중단
3. **전탐색**: 모든 (p,q,P,Q) 조합 → `sarimax_batch_fit`로 병렬 실행
4. **결과**: 최적 모델의 FitResult + 탐색 이력 DataFrame

### 의존
- 단위근 검정: scipy 또는 Rust 구현
- Phase A (trend) 완료 필요: trend도 탐색 대상
