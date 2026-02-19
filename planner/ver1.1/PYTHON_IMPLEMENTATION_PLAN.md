# Python Implementation Plan (ver1.1)

## SARIMAX Python Orchestration Layer — uv 기반 환경 + 구체화

---

# 0. ver1 → ver1.1 주요 변경점

| 항목 | ver1 | ver1.1 |
|------|------|--------|
| 패키지 관리 | pip/requirements.txt | **uv** |
| 빌드 도구 | maturin (명령만) | **uv + maturin 통합 파이프라인** |
| pyproject.toml | 없음 | **구체 작성** |
| 전처리 | 함수 시그니처만 | **차분 알고리즘 + edge case** 구체화 |
| 결과 클래스 | AIC만 | **진단 테스트 + summary 테이블** |
| edge case | 없음 | **상수 시계열, 짧은 시계열, NaN** |
| pandas 지원 | 없음 | **Series/DataFrame 입력 지원** |

---

# 1. pyproject.toml (완전)

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "sarimax-rs"
version = "0.1.0"
requires-python = ">=3.10"
description = "High-performance SARIMAX via Rust + PyO3"
dependencies = [
    "numpy>=1.24",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "statsmodels>=0.14",
    "scipy>=1.10",
    "pandas>=2.0",
    "matplotlib>=3.7",
    "maturin>=1.7",
]

[tool.maturin]
features = ["pyo3/extension-module"]
python-source = "python"
module-name = "sarimax_rs"

[tool.pytest.ini_options]
testpaths = ["python_tests"]
```

---

# 2. uv 환경 설정

```bash
# pyproject.toml 이 없을 때만 초기화
# uv init
uv add numpy
uv add --group dev pytest statsmodels scipy pandas matplotlib
uv add --group dev maturin

# 가상환경 동기화
uv sync --extra dev

# Rust 확장 모듈 개발 설치
uv run maturin develop --release

# 테스트 실행
uv run pytest python_tests -q

# statsmodels 기준값 생성
uv run python python_tests/generate_fixtures.py

# 벤치마크 실행
uv run python python/benchmarks/bench_vs_statsmodels.py
```

### uv 의존성 정리

| 카테고리 | 패키지 | 용도 |
|---------|--------|------|
| **런타임** | numpy>=1.24 | 배열 전달 (pyo3-numpy) |
| **dev** | pytest>=7.0 | 테스트 프레임워크 |
| **dev** | statsmodels>=0.14 | 검증 기준값 |
| **dev** | scipy>=1.10 | Phase 1 최적화 (폴백용) |
| **dev** | pandas>=2.0 | DataFrame 입력 지원 테스트 |
| **dev** | matplotlib>=3.7 | 시각화/진단 플롯 |
| **dev** | maturin>=1.7 | Rust 확장 모듈 빌드/설치 |

> **핵심**: 런타임 의존은 **numpy만**. statsmodels/scipy는 dev 전용.

---

# 3. 프로젝트 구조

```
sarimax_rs/
├── python/
│   └── sarimax_py/
│       ├── __init__.py
│       ├── model.py              # SarimaxModel 클래스
│       ├── preprocessing.py      # 데이터 전처리 + 차분
│       ├── results.py            # SarimaxResults + 진단
│       ├── auto.py               # auto_sarimax 선택기
│       ├── plotting.py           # 진단 플롯
│       └── compat.py             # statsmodels 호환
├── python_tests/
│   ├── conftest.py               # 공통 fixtures
│   ├── test_preprocessing.py
│   ├── test_model.py
│   ├── test_results.py
│   ├── test_auto.py
│   ├── test_vs_statsmodels.py    # 핵심 검증
│   └── generate_fixtures.py
├── pyproject.toml
└── uv.lock
```

---

# 4. 핵심 모듈 상세 설계

## 4.1 preprocessing.py

### 차분 알고리즘

```python
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional

def validate_endog(y: ArrayLike) -> np.ndarray:
    """endogenous 시계열 검증"""
    y = np.asarray(y, dtype=np.float64).squeeze()
    if y.ndim != 1:
        raise ValueError(f"endog must be 1-d, got {y.ndim}-d")
    if len(y) < 2:
        raise ValueError(f"endog must have >= 2 observations, got {len(y)}")
    if np.all(y == y[0]):
        import warnings
        warnings.warn("endog is constant — model may not converge")
    return y

def validate_exog(exog: Optional[ArrayLike], n_obs: int) -> Optional[np.ndarray]:
    """exogenous 변수 검증"""
    if exog is None:
        return None
    exog = np.asarray(exog, dtype=np.float64)
    if exog.ndim == 1:
        exog = exog.reshape(-1, 1)
    if exog.shape[0] != n_obs:
        raise ValueError(f"exog rows ({exog.shape[0]}) != endog length ({n_obs})")
    if np.any(np.isnan(exog)):
        raise ValueError("exog contains NaN")
    return exog

def apply_differencing(y: np.ndarray, d: int, D: int, s: int) -> np.ndarray:
    """
    차분 적용 순서 (statsmodels 동일):
    1. 계절 차분 먼저: y_t = y_t - y_{t-s}  (D회 반복)
    2. 일반 차분:       y_t = y_t - y_{t-1}  (d회 반복)
    """
    result = y.copy()
    # 계절 차분
    for _ in range(D):
        result = result[s:] - result[:-s]
    # 일반 차분
    for _ in range(d):
        result = np.diff(result)
    return result

def prepare_data(y, exog, order, seasonal_order):
    """
    전체 전처리 파이프라인.
    pandas Series/DataFrame도 처리.
    """
    # pandas → numpy
    if hasattr(y, 'values'):
        y = y.values
    if exog is not None and hasattr(exog, 'values'):
        exog = exog.values

    y = validate_endog(y)
    exog = validate_exog(exog, len(y))

    return y, exog

def build_trend_data(n_obs: int, trend: str, offset: int = 1) -> Optional[np.ndarray]:
    """
    Trend 데이터 행렬 구성 (statsmodels prepare_trend_data 동일).
    - 'n': None
    - 'c': ones(n_obs, 1)
    - 't': arange(offset, n_obs+offset).reshape(-1,1)
    - 'ct': [ones, arange]
    """
    if trend == 'n':
        return None
    time_trend = np.arange(offset, n_obs + offset, dtype=np.float64)
    if trend == 'c':
        return np.ones((n_obs, 1), dtype=np.float64)
    elif trend == 't':
        return time_trend.reshape(-1, 1)
    elif trend == 'ct':
        return np.column_stack([np.ones(n_obs), time_trend])
    else:
        raise ValueError(f"Unknown trend: {trend!r}")
```

---

## 4.2 model.py

```python
import numpy as np
import sarimax_rs  # Rust pyo3 바인딩
from .preprocessing import prepare_data, build_trend_data
from .results import SarimaxResults

class SarimaxModel:
    """
    SARIMAX 모델.

    Usage:
        model = SarimaxModel(y, order=(1,1,1), seasonal_order=(1,1,1,12))
        result = model.fit()
        print(result.aic)
        forecast = result.forecast(steps=12)
    """

    def __init__(
        self,
        endog,
        exog=None,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        trend='n',
        enforce_stationarity=False,
        enforce_invertibility=False,
        concentrate_scale=True,
    ):
        self.endog, self.exog = prepare_data(endog, exog, order, seasonal_order)
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.concentrate_scale = concentrate_scale
        self.trend_data = build_trend_data(len(self.endog), trend)

    @property
    def n_params(self) -> int:
        """총 추정 파라미터 수 (AIC의 k)"""
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order
        k_trend = {'n': 0, 'c': 1, 't': 1, 'ct': 2}[self.trend]
        k_exog = 0 if self.exog is None else self.exog.shape[1]
        return k_trend + k_exog + p + q + P + Q + 1  # +1 for sigma2

    def loglike(self, params: np.ndarray) -> float:
        """주어진 파라미터에서 log-likelihood 계산 (Rust 엔진)"""
        return sarimax_rs.sarimax_loglike(
            self.endog,
            self.order[:3],
            self.seasonal_order,
            np.asarray(params, dtype=np.float64),
            exog=self.exog,
            concentrate_scale=self.concentrate_scale,
        )

    def fit(
        self,
        method='lbfgs',
        maxiter=50,
        start_params=None,
    ) -> 'SarimaxResults':
        """
        모델 피팅 (Rust 엔진).

        Parameters
        ----------
        method : str
            'lbfgs' (기본) 또는 'nelder-mead'
        maxiter : int
            최대 반복 횟수
        start_params : array-like, optional
            초기 파라미터 (None이면 CSS 자동 추정)
        """
        result_dict = sarimax_rs.sarimax_fit(
            self.endog,
            self.order[:3],
            self.seasonal_order,
            exog=self.exog,
            method=method,
            maxiter=maxiter,
            start_params=(
                np.asarray(start_params, dtype=np.float64)
                if start_params is not None else None
            ),
            concentrate_scale=self.concentrate_scale,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
            trend=self.trend,
        )

        return SarimaxResults(self, result_dict)
```

---

## 4.3 results.py

```python
import numpy as np
from dataclasses import dataclass
from typing import Optional
import sarimax_rs

class SarimaxResults:
    """피팅 결과 래핑"""

    def __init__(self, model, result_dict: dict):
        self.model = model
        self._dict = result_dict
        self.params = np.array(result_dict['params'])
        self.loglike = result_dict['loglike']
        self.scale = result_dict.get('scale', np.nan)
        self.n_obs = result_dict.get('n_obs', len(model.endog))
        self.converged = result_dict.get('converged', True)
        self.n_iter = result_dict.get('n_iter', 0)

    @property
    def aic(self) -> float:
        """Akaike Information Criterion"""
        # Rust에서 이미 계산되었으면 사용, 아니면 Python 계산
        if 'aic' in self._dict:
            return self._dict['aic']
        k = self.model.n_params
        return 2 * k - 2 * self.loglike

    @property
    def bic(self) -> float:
        """Bayesian Information Criterion"""
        if 'bic' in self._dict:
            return self._dict['bic']
        k = self.model.n_params
        return k * np.log(self.n_obs) - 2 * self.loglike

    @property
    def aicc(self) -> float:
        """Corrected AIC (small sample)"""
        if 'aicc' in self._dict:
            return self._dict['aicc']
        k = self.model.n_params
        n = self.n_obs
        if n <= k + 1:
            return np.inf
        return self.aic + (2 * k**2 + 2 * k) / (n - k - 1)

    @property
    def hqic(self) -> float:
        """Hannan-Quinn Information Criterion"""
        if 'hqic' in self._dict:
            return self._dict['hqic']
        k = self.model.n_params
        return 2 * k * np.log(np.log(self.n_obs)) - 2 * self.loglike

    @property
    def residuals(self) -> np.ndarray:
        """필터링된 잔차 (innovations)"""
        if 'innovations' in self._dict:
            return np.array(self._dict['innovations'])
        # 재계산 필요 시 Rust 호출
        return np.array([])

    def forecast(self, steps: int, exog_future=None):
        """h-step ahead 예측"""
        result = sarimax_rs.sarimax_forecast(
            self.model.endog,
            self.model.order[:3],
            self.model.seasonal_order,
            self.params,
            steps,
            exog=self.model.exog,
            exog_future=(
                np.asarray(exog_future, dtype=np.float64)
                if exog_future is not None else None
            ),
        )
        return ForecastResult(
            mean=np.array(result['mean']),
            variance=np.array(result['variance']),
            steps=steps,
        )

    def summary(self) -> str:
        """statsmodels 스타일 결과 요약"""
        lines = []
        lines.append("=" * 60)
        lines.append(f"SARIMAX Results (Rust Engine)")
        lines.append("=" * 60)
        p, d, q = self.model.order
        P, D, Q, s = self.model.seasonal_order
        lines.append(f"  Order:          ({p},{d},{q})")
        lines.append(f"  Seasonal:       ({P},{D},{Q},{s})")
        lines.append(f"  Observations:   {self.n_obs}")
        lines.append(f"  Log-Likelihood: {self.loglike:.4f}")
        lines.append(f"  AIC:            {self.aic:.4f}")
        lines.append(f"  BIC:            {self.bic:.4f}")
        lines.append(f"  AICc:           {self.aicc:.4f}")
        lines.append(f"  HQIC:           {self.hqic:.4f}")
        lines.append(f"  Scale (sigma2): {self.scale:.6f}")
        lines.append(f"  Converged:      {self.converged}")
        lines.append(f"  Iterations:     {self.n_iter}")
        lines.append("-" * 60)
        lines.append("  Parameters:")

        # 파라미터 이름 생성
        names = self._param_names()
        for name, val in zip(names, self.params):
            lines.append(f"    {name:20s} {val:12.6f}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def _param_names(self):
        """파라미터 이름 목록 생성"""
        names = []
        # trend
        trend_names = {'n': [], 'c': ['intercept'], 't': ['drift'], 'ct': ['intercept', 'drift']}
        names.extend(trend_names.get(self.model.trend, []))
        # exog
        n_exog = 0 if self.model.exog is None else self.model.exog.shape[1]
        names.extend([f'x{i+1}' for i in range(n_exog)])
        # AR
        p = self.model.order[0]
        names.extend([f'ar.L{i+1}' for i in range(p)])
        # MA
        q = self.model.order[2]
        names.extend([f'ma.L{i+1}' for i in range(q)])
        # Seasonal AR/MA
        P, D, Q, s = self.model.seasonal_order
        names.extend([f'ar.S.L{(i+1)*s}' for i in range(P)])
        names.extend([f'ma.S.L{(i+1)*s}' for i in range(Q)])
        # sigma2
        if not self.model.concentrate_scale:
            names.append('sigma2')
        return names


@dataclass
class ForecastResult:
    mean: np.ndarray
    variance: np.ndarray
    steps: int

    @property
    def confidence_interval(self, alpha: float = 0.05):
        """예측 신뢰구간"""
        from statistics import NormalDist
        z = NormalDist().inv_cdf(1 - alpha / 2)
        std = np.sqrt(self.variance)
        return self.mean - z * std, self.mean + z * std
```

---

## 4.4 auto.py — 자동 모델 선택

```python
import numpy as np
from typing import Optional
from .model import SarimaxModel

def auto_sarimax(
    endog,
    exog=None,
    p_range=range(0, 4),
    d_range=range(0, 2),
    q_range=range(0, 4),
    P_range=range(0, 2),
    D_range=range(0, 2),
    Q_range=range(0, 2),
    s: int = 12,
    criterion: str = 'aic',
    maxiter: int = 50,
    use_rust: bool = True,
) -> dict:
    """
    Grid search로 최적 SARIMAX order 선택.

    use_rust=True: Rust rayon 병렬 (Phase 4+)
    use_rust=False: Python 순차 루프 (Phase 1-3)

    Returns
    -------
    dict with keys: order, seasonal_order, {criterion}, result
    """
    if use_rust:
        import sarimax_rs
        result = sarimax_rs.auto_select(
            np.asarray(endog, dtype=np.float64),
            p_max=max(p_range), d_max=max(d_range), q_max=max(q_range),
            pp_max=max(P_range), dd_max=max(D_range), qq_max=max(Q_range),
            s=s, exog=exog, criterion=criterion, maxiter=maxiter,
        )
        return result

    # Python 폴백 (순차)
    best = None
    best_score = np.inf

    candidates = [
        ((p, d, q), (P, D, Q, s))
        for p in p_range for d in d_range for q in q_range
        for P in P_range for D in D_range for Q in Q_range
    ]

    for order, seasonal in candidates:
        try:
            model = SarimaxModel(endog, exog=exog, order=order,
                                  seasonal_order=seasonal)
            result = model.fit(maxiter=maxiter)
            score = getattr(result, criterion)
            if score < best_score:
                best_score = score
                best = {
                    'order': order,
                    'seasonal_order': seasonal,
                    criterion: score,
                    'result': result,
                }
        except Exception:
            continue  # 수렴 실패 → 건너뜀

    if best is None:
        raise RuntimeError("모든 후보 모델이 수렴에 실패했습니다")
    return best
```

---

## 4.5 compat.py — statsmodels 호환

```python
import numpy as np

def from_statsmodels(sm_result):
    """statsmodels 결과에서 파라미터/설정 추출"""
    return {
        'params': sm_result.params.tolist(),
        'loglike': float(sm_result.llf),
        'aic': float(sm_result.aic),
        'bic': float(sm_result.bic),
        'order': sm_result.model.order,
        'seasonal_order': sm_result.model.seasonal_order,
    }

def compare(our_result, sm_result, tol=1e-4):
    """수치 비교 검증"""
    checks = {}
    checks['loglike'] = abs(our_result.loglike - sm_result.llf) < tol
    checks['aic'] = abs(our_result.aic - sm_result.aic) < tol
    checks['bic'] = abs(our_result.bic - sm_result.bic) < tol
    checks['params'] = np.allclose(our_result.params, sm_result.params, atol=tol)
    return checks
```

---

# 5. 테스트 전략

## conftest.py (공통 fixtures)

```python
import pytest
import numpy as np

@pytest.fixture
def airline_data():
    """statsmodels CO2 데이터 (결측 제거 후 사용)"""
    from statsmodels.datasets import co2
    data = co2.load().data
    return data['co2'].dropna().values

@pytest.fixture
def simple_ar1():
    """AR(1) 시뮬레이션 데이터"""
    np.random.seed(42)
    n = 200
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.7 * y[t-1] + np.random.randn()
    return y

@pytest.fixture
def random_walk():
    """랜덤워크 데이터"""
    np.random.seed(123)
    return np.cumsum(np.random.randn(300))
```

## 검증 허용 오차 (ver1.1 확정)

| 비교 항목 | 허용 오차 | 비고 |
|----------|----------|------|
| log-likelihood | < 1e-6 | 동일 params 입력 시 |
| AIC/BIC | < 1e-4 | fit 결과 비교 |
| 파라미터 | < 1e-3 | 최적화 경로 차이 허용 |
| forecast mean | < 1e-4 | |
| forecast variance | < 1e-3 | |
| 상태공간 행렬 | < 1e-10 | 동일 params에서 구성 |

---

# 6. 벤치마크

```python
# python/benchmarks/bench_vs_statsmodels.py
import time
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import sarimax_rs

def bench_single_fit():
    np.random.seed(42)
    y = np.cumsum(np.random.randn(500))

    # statsmodels
    t0 = time.perf_counter()
    sm = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12),
                 enforce_stationarity=False).fit(disp=False)
    t_sm = time.perf_counter() - t0

    # sarimax_rs
    t0 = time.perf_counter()
    rs = sarimax_rs.sarimax_fit(y, (1,1,1), (1,1,1,12))
    t_rs = time.perf_counter() - t0

    print(f"statsmodels: {t_sm*1000:.1f}ms")
    print(f"sarimax_rs:  {t_rs*1000:.1f}ms")
    print(f"speedup:     {t_sm/t_rs:.1f}x")

def bench_auto_select():
    np.random.seed(42)
    y = np.cumsum(np.random.randn(300))

    # Python 순차
    t0 = time.perf_counter()
    # ... 128 candidates sequentially
    t_py = time.perf_counter() - t0

    # Rust rayon 병렬
    t0 = time.perf_counter()
    rs = sarimax_rs.auto_select(y, p_max=3, d_max=1, q_max=3,
                                 pp_max=1, dd_max=1, qq_max=1, s=12)
    t_rs = time.perf_counter() - t0

    print(f"Python sequential: {t_py:.1f}s")
    print(f"Rust parallel:     {t_rs:.1f}s")

if __name__ == "__main__":
    bench_single_fit()
    bench_auto_select()
```

---

# 7. 단계별 Python 개발 일정 (ver1.1)

| 단계 | 내용 | Rust 의존 | 기간 |
|-----|------|----------|------|
| **P-0** | pyproject.toml + uv 환경 | Rust Phase 0 | 1일 |
| **P-1** | preprocessing.py + tests | 없음 | 2일 |
| **P-2** | model.py (loglike 호출) | Phase 1e | 2일 |
| **P-3** | model.py (fit 호출) | Phase 3c | 2일 |
| **P-4** | results.py (AIC/BIC/summary) | Phase 3d | 2일 |
| **P-5** | forecast 호출 | Phase 4b | 1일 |
| **P-6** | auto.py (Python 폴백) | Phase 3d | 2일 |
| **P-7** | auto.py (Rust rayon) | Phase 4a | 1일 |
| **P-8** | compat.py + 통합 테스트 | Phase 3c+ | 2일 |
| **P-9** | 벤치마크 + 시각화 | Phase 5 | 2일 |

---

# 8. Edge Case 처리

| 상황 | 처리 |
|------|------|
| 상수 시계열 (분산 0) | warning + sigma2 floor 1e-10 |
| 관측치 < k_params | ValueError 발생 |
| NaN in endog | `simple_differencing=True` 시 NaN 제거, False 시 결측 처리 |
| exog shape 불일치 | 즉시 ValueError |
| 수렴 실패 | converged=False, warning, 마지막 파라미터 반환 |
| p=d=q=P=D=Q=0 | 백색 잡음 모델 (loglike = 가우시안 LL) |
| s=0 with P,D,Q>0 | ValueError |
