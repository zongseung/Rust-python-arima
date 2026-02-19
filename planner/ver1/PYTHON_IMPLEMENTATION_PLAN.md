# Python-Side Implementation Plan (ver1)

## SARIMAX Python Orchestration Layer 설계

---

# 1. 역할 정의

Python은 **오케스트레이션 레이어**로서 다음을 담당한다:

- 데이터 입출력 및 전처리
- Rust 엔진 호출 인터페이스
- 결과 후처리 및 시각화
- statsmodels 호환 API 제공

> **핵심 원칙**: 수치 연산은 Rust에 위임, Python은 파이프라인 관리에 집중

---

# 2. 프로젝트 구조

```
sarimax_rs/
├── python/
│   ├── sarimax_py/
│   │   ├── __init__.py
│   │   ├── model.py          # SarimaxModel 클래스
│   │   ├── preprocessing.py  # 데이터 전처리
│   │   ├── results.py        # 결과 래핑 클래스
│   │   ├── plotting.py       # 시각화 유틸
│   │   └── compat.py         # statsmodels 호환 레이어
│   ├── tests/
│   │   ├── test_model.py
│   │   ├── test_preprocessing.py
│   │   ├── test_vs_statsmodels.py
│   │   └── test_aic_selection.py
│   └── benchmarks/
│       ├── bench_single.py
│       ├── bench_batch.py
│       └── bench_vs_statsmodels.py
```

---

# 3. 핵심 모듈 설계

## 3.1 preprocessing.py

### 책임
- 결측치 처리 (선형보간, forward-fill)
- 차분(differencing) 적용 (d, D)
- exogenous 변수 정합성 검증
- numpy 배열 변환 및 타입 검증

### API
```python
def validate_endog(y: ArrayLike) -> np.ndarray:
    """endogenous 시계열 검증 및 float64 변환"""

def validate_exog(exog: Optional[ArrayLike], n_obs: int) -> Optional[np.ndarray]:
    """exogenous 변수 검증: shape, dtype, NaN 체크"""

def apply_differencing(y: np.ndarray, d: int, D: int, s: int) -> np.ndarray:
    """비계절/계절 차분 적용"""

def prepare_data(y, exog, order, seasonal_order):
    """전체 전처리 파이프라인 실행, Rust 엔진에 전달할 데이터 준비"""
```

---

## 3.2 model.py

### SarimaxModel 클래스

```python
import sarimax_rs  # Rust pyo3 바인딩

class SarimaxModel:
    def __init__(
        self,
        endog,
        exog=None,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        trend='n',
        enforce_stationarity=False,
        enforce_invertibility=False,
    ):
        self.endog = validate_endog(endog)
        self.exog = validate_exog(exog, len(endog))
        self.order = order
        self.seasonal_order = seasonal_order
        # ...

    def fit(self, method='lbfgs', maxiter=50, start_params=None):
        """
        Phase 1: scipy.optimize -> Rust loglike
        Phase 3+: 순수 Rust fit (argmin L-BFGS)
        """
        if self._use_rust_fit:
            # Rust 내부 최적화 (Phase 3+)
            result = sarimax_rs.sarimax_fit(
                self.endog, self.exog,
                self.order, self.seasonal_order,
                method=method, maxiter=maxiter,
                start_params=start_params
            )
        else:
            # Phase 1: Python 최적화 + Rust loglike
            from scipy.optimize import minimize
            result = minimize(
                lambda params: -sarimax_rs.sarimax_loglike(
                    self.endog, self.exog,
                    self.order, self.seasonal_order,
                    params
                ),
                x0=start_params or self._default_start_params(),
                method=method,
                options={'maxiter': maxiter}
            )

        return SarimaxResults(self, result)

    def loglike(self, params):
        """Rust 엔진으로 log-likelihood 계산"""
        return sarimax_rs.sarimax_loglike(
            self.endog, self.exog,
            self.order, self.seasonal_order,
            params
        )
```

---

## 3.3 results.py

### SarimaxResults 클래스

```python
class SarimaxResults:
    def __init__(self, model, fit_result):
        self.model = model
        self.params = fit_result.params
        self.loglike = fit_result.loglike
        self.nobs = len(model.endog)

    @property
    def aic(self):
        """Phase 1: Python 계산 / Phase 3+: Rust에서 직접 반환"""
        k = self._n_params
        return 2 * k - 2 * self.loglike

    @property
    def bic(self):
        k = self._n_params
        n = self.nobs
        return k * np.log(n) - 2 * self.loglike

    @property
    def aicc(self):
        k = self._n_params
        n = self.nobs
        return self.aic + (2 * k**2 + 2 * k) / (n - k - 1)

    def forecast(self, steps, exog_future=None):
        """Rust 엔진으로 예측 수행"""
        mean, var = sarimax_rs.sarimax_forecast(
            self.model.endog, self.model.exog,
            self.model.order, self.model.seasonal_order,
            self.params, steps, exog_future
        )
        return ForecastResult(mean, var, steps)

    def summary(self):
        """statsmodels 스타일 결과 요약 출력"""
        # 파라미터 테이블, AIC/BIC, 잔차 통계 등
```

---

## 3.4 compat.py (statsmodels 호환 레이어)

```python
def from_statsmodels_params(sm_result):
    """statsmodels 결과 객체에서 파라미터 추출"""

def compare_loglike(our_result, sm_result, tol=1e-6):
    """두 모델의 log-likelihood 비교 검증"""

def migration_guide(sm_code: str) -> str:
    """statsmodels 코드를 sarimax_rs 코드로 변환 안내"""
```

---

# 4. 모델 선택 (Auto-SARIMAX)

## Python-Side Auto Selection (Phase 1)

```python
def auto_sarimax(
    endog,
    exog=None,
    p_range=range(0, 4),
    d_range=range(0, 2),
    q_range=range(0, 4),
    P_range=range(0, 2),
    D_range=range(0, 2),
    Q_range=range(0, 2),
    s=12,
    criterion='aic',
    n_jobs=-1,  # joblib 병렬화
):
    """
    Grid search로 최적 SARIMAX order 선택.
    Phase 1: Python 루프 + Rust loglike
    Phase 4: Rust batch_fit으로 전환
    """
    candidates = generate_candidates(p_range, d_range, q_range,
                                      P_range, D_range, Q_range, s)
    results = []
    for order, seasonal in candidates:
        try:
            model = SarimaxModel(endog, exog, order, seasonal)
            result = model.fit()
            score = getattr(result, criterion)
            results.append((order, seasonal, score, result))
        except Exception:
            continue  # 수렴 실패 시 건너뜀

    best = min(results, key=lambda x: x[2])
    return best
```

## Rust-Side Auto Selection (Phase 4)

```python
def auto_sarimax_rust(endog, exog=None, ...):
    """완전 Rust 기반 병렬 모델 선택 - rayon 활용"""
    return sarimax_rs.auto_select(
        endog, exog,
        p_range, d_range, q_range,
        P_range, D_range, Q_range, s,
        criterion='aic'
    )
```

---

# 5. 테스트 전략

## 단위 테스트
| 테스트 대상 | 검증 항목 |
|------------|----------|
| preprocessing | 차분, 결측치, dtype 변환 |
| model.fit() | 수렴 여부, params shape |
| results.aic | 수식 정확성 |
| forecast | 예측값 범위, shape |

## 통합 테스트 (vs statsmodels)
| 비교 항목 | 허용 오차 |
|----------|----------|
| log-likelihood | < 1e-6 |
| AIC/BIC | < 1e-4 |
| forecast mean | < 1e-4 |
| forecast variance | < 1e-3 |
| residuals | < 1e-5 |

## 성능 벤치마크
| 시나리오 | 측정 대상 |
|---------|----------|
| 단일 fit | 실행 시간 (ms) |
| 100개 배치 | 총 시간, 개당 시간 |
| 1000개 배치 | 총 시간, 메모리 |
| auto_sarimax | 후보 수 대비 총 시간 |

---

# 6. 의존성

```
# requirements.txt
numpy>=1.24
scipy>=1.10
maturin>=1.0        # Rust 빌드
sarimax_rs           # Rust pyo3 패키지 (로컬)

# dev
pytest>=7.0
statsmodels>=0.14    # 비교 검증용
joblib>=1.3          # 병렬 auto_sarimax
matplotlib>=3.7      # 시각화
```

---

# 7. 단계별 Python 개발 일정

| 단계 | 내용 | Rust 의존 |
|-----|------|----------|
| P-1 | preprocessing, validate 모듈 | 없음 |
| P-2 | SarimaxModel.loglike (Rust 호출) | Phase 1 |
| P-3 | SarimaxModel.fit (scipy + Rust loglike) | Phase 1 |
| P-4 | SarimaxResults, AIC/BIC | Phase 1 |
| P-5 | forecast (Rust 호출) | Phase 2 |
| P-6 | auto_sarimax (Python grid search) | Phase 2 |
| P-7 | auto_sarimax_rust (Rust batch) | Phase 4 |
| P-8 | compat.py, migration 도구 | Phase 2+ |

---

# 8. 핵심 설계 결정

1. **Phase 1에서는 AIC를 Python에서 계산** (loglike만 Rust)
2. **Phase 3+에서 AIC를 Rust로 이전 가능** (argmin + nalgebra 조합)
3. **auto_sarimax는 Python 루프로 시작**, Rust batch로 점진적 전환
4. **statsmodels 호환 API** 유지로 기존 코드 마이그레이션 최소화
5. **numpy zero-copy** (pyo3-numpy) 활용으로 데이터 복사 오버헤드 제거
