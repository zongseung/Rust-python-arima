# P1 — statsmodels 호환성 이슈 상세

---

## B-1. CI/CD 워크플로

### 현재
`.github/workflows/` 디렉토리 없음. 모든 검증이 수동.

### 필요 파일

**ci.yml** (push/PR 트리거):
```yaml
jobs:
  rust-test:
    - cargo test --all-targets
  python-test:
    - maturin develop --release
    - pytest python_tests/ -v --ignore=python_tests/test_matrix_tier_b.py
```

**nightly.yml** (야간 전체):
```yaml
jobs:
  tier-b:
    - pytest python_tests/test_matrix_tier_b.py -v
    - pytest python_tests/test_high_order_accuracy.py -v
  bench:
    - cargo bench
```

**release.yml** (tag v* 트리거):
```yaml
jobs:
  build-wheels:
    matrix:
      os: [ubuntu-latest, macos-latest, windows-latest]
      python: ['3.10', '3.11', '3.12', '3.13']
    steps:
      - maturin build --release
      - twine upload
```

---

## B-2. plot_diagnostics()

### 현재
model.py에 미구현. matplotlib은 pyproject.toml devdeps에 이미 있음.

### 구현
```python
def plot_diagnostics(self, fig=None, figsize=(10, 8)):
    """statsmodels 호환 4-패널 잔차 진단 플롯."""
    import matplotlib.pyplot as plt

    resid = self.resid
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # (1) Standardized residuals over time
    axes[0,0].plot(resid)
    axes[0,0].set_title('Standardized Residuals')

    # (2) Histogram + KDE
    axes[0,1].hist(resid, bins=30, density=True, alpha=0.7)
    axes[0,1].set_title('Histogram plus estimated density')

    # (3) Q-Q plot
    from scipy import stats
    stats.probplot(resid, dist="norm", plot=axes[1,0])
    axes[1,0].set_title('Normal Q-Q')

    # (4) Correlogram (ACF)
    _plot_acf(resid, ax=axes[1,1], lags=40)
    axes[1,1].set_title('Correlogram')

    fig.tight_layout()
    return fig
```

### 의존성
- matplotlib: 이미 devdeps
- scipy.stats.probplot: QQ플롯
- ACF 계산: numpy 자체 구현 또는 statsmodels.graphics 차용

---

## B-3. Jupyter _repr_html_()

### SARIMAXResult에 추가
```python
def _repr_html_(self):
    """Jupyter 노트북에서 HTML 표로 렌더링."""
    ps = self.parameter_summary(inference='hessian')
    rows = []
    for i, name in enumerate(ps['name']):
        rows.append(f"<tr><td>{name}</td>"
                    f"<td>{ps['coef'][i]:.4f}</td>"
                    f"<td>{ps['std_err'][i]:.4f}</td>"
                    f"<td>{ps['z'][i]:.3f}</td>"
                    f"<td>{ps['p_value'][i]:.3f}</td></tr>")
    return f"""
    <div><b>SARIMAX Results</b></div>
    <table>
    <tr><td>Order:</td><td>{self.model.order}</td>
        <td>Log Likelihood:</td><td>{self.llf:.4f}</td></tr>
    <tr><td>Seasonal:</td><td>{self.model.seasonal_order}</td>
        <td>AIC:</td><td>{self.aic:.4f}</td></tr>
    </table>
    <table border="1">
    <tr><th></th><th>coef</th><th>std err</th><th>z</th><th>P>|z|</th></tr>
    {''.join(rows)}
    </table>
    """
```

### ForecastResult에 추가
```python
def _repr_html_(self):
    rows = [f"<tr><td>{i+1}</td><td>{m:.4f}</td><td>{lo:.4f}</td><td>{hi:.4f}</td></tr>"
            for i,(m,lo,hi) in enumerate(zip(self.predicted_mean,
                                              self.ci_lower, self.ci_upper))]
    return f"""
    <table border="1">
    <tr><th>Step</th><th>Mean</th><th>CI Lower</th><th>CI Upper</th></tr>
    {''.join(rows)}
    </table>
    """
```

---

## B-4. Wheel 빌드 검증

### A-1 수정 후 검증 절차
```bash
# 1) 빌드
cd sarimax_rs
CARGO_TARGET_DIR=target_wheel .venv/bin/maturin build --release --out /tmp/wheels

# 2) 새 venv에서 설치
python -m venv /tmp/test_env
/tmp/test_env/bin/pip install /tmp/wheels/sarimax_rs-*.whl

# 3) sarimax_py가 포함되는지 확인
/tmp/test_env/bin/python -c "from sarimax_py import SARIMAXModel; print('OK')"

# 4) 기본 기능 테스트
/tmp/test_env/bin/python -c "
from sarimax_py import SARIMAXModel
import numpy as np
y = np.cumsum(np.random.randn(200))
r = SARIMAXModel(y, order=(1,1,1)).fit()
print(r.summary())
fc = r.forecast(steps=5)
print(fc.predicted_mean)
"
```
