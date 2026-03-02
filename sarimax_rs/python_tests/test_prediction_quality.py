"""
예측 품질 검증 (Section 2.7) — MAE/RMSE/MAPE, 95% CI coverage 백테스트.

Rolling-origin cross-validation:
  - 최소 3 split
  - Horizon: 1, 6
  - 모델: AR(1), ARIMA(1,1,1), SARIMA(1,1,1)(1,1,1,12)
"""

import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pytest

import sarimax_rs


# ─────────────────────────────────────────
# 데이터 생성
# ─────────────────────────────────────────

def _gen_ar1(n=500, seed=42):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.6 * y[i - 1] + rng.normal()
    return y


def _gen_random_walk(n=500, seed=42):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=n))


def _gen_seasonal(n=300, s=12, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    seasonal = 2.0 * np.sin(2 * math.pi * t / s)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.5 * y[i - 1] + seasonal[i] + rng.normal(scale=0.5)
    return y


# ─────────────────────────────────────────
# 지표 계산 헬퍼
# ─────────────────────────────────────────

def compute_mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))


def compute_rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))


def compute_mape(actual, predicted):
    mask = np.abs(actual) > 1e-8  # 0 근처 값 제외
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def compute_ci_coverage(actual, ci_lower, ci_upper):
    inside = (actual >= ci_lower) & (actual <= ci_upper)
    return np.mean(inside)


def compute_avg_width(ci_lower, ci_upper):
    return np.mean(ci_upper - ci_lower)


# ─────────────────────────────────────────
# Rolling-origin 백테스트
# ─────────────────────────────────────────

def rolling_backtest(y, order, seasonal, origins, horizon, alpha=0.05):
    """
    여러 origin 지점에서 모델을 fit하고 h-step forecast를 수행.

    Parameters
    ----------
    y : array
    order : tuple (p,d,q)
    seasonal : tuple (P,D,Q,s)
    origins : list[int]  — 학습 데이터 끝 인덱스
    horizon : int
    alpha : float

    Returns
    -------
    dict with metrics aggregated across origins
    """
    all_actual   = []
    all_pred     = []
    all_ci_lower = []
    all_ci_upper = []

    for origin in origins:
        train = y[:origin]
        actual = y[origin:origin + horizon]
        if len(actual) < horizon:
            continue

        result = sarimax_rs.sarimax_fit(train, order, seasonal)
        if not result["converged"]:
            continue

        params = np.array(result["params"])
        fc = sarimax_rs.sarimax_forecast(
            train, order, seasonal, params, steps=horizon, alpha=alpha,
        )

        all_actual.append(actual)
        all_pred.append(np.array(fc["mean"]))
        all_ci_lower.append(np.array(fc["ci_lower"]))
        all_ci_upper.append(np.array(fc["ci_upper"]))

    if not all_actual:
        return None

    actual   = np.concatenate(all_actual)
    pred     = np.concatenate(all_pred)
    ci_lo    = np.concatenate(all_ci_lower)
    ci_hi    = np.concatenate(all_ci_upper)

    return {
        "mae": compute_mae(actual, pred),
        "rmse": compute_rmse(actual, pred),
        "mape": compute_mape(actual, pred),
        "ci_coverage": compute_ci_coverage(actual, ci_lo, ci_hi),
        "avg_width": compute_avg_width(ci_lo, ci_hi),
        "n_forecasts": len(actual),
    }


# ─────────────────────────────────────────
# 테스트 설정
# ─────────────────────────────────────────

# AR(1) data
_ar1_data = _gen_ar1(500, seed=42)
_ar1_origins = [300, 350, 400]

# ARIMA(1,1,1) data
_arima_data = _gen_random_walk(500, seed=42)
_arima_origins = [300, 350, 400]

# SARIMA data
_sarima_data = _gen_seasonal(300, s=12, seed=42)
_sarima_origins = [200, 225, 250]


# ─────────────────────────────────────────
# 테스트 함수
# ─────────────────────────────────────────

class TestCICoverage:
    """95% CI coverage가 합리적 범위 내인지 검증."""

    def test_ar1_ci_coverage_h1(self):
        """AR(1) horizon=1: CI coverage >= 80%."""
        metrics = rolling_backtest(_ar1_data, (1, 0, 0), (0, 0, 0, 0), _ar1_origins, horizon=1)
        assert metrics is not None, "All fits failed"
        assert 0.80 <= metrics["ci_coverage"] <= 1.0, (
            f"AR(1) h=1 coverage={metrics['ci_coverage']:.3f}, expected [0.80, 1.0]"
        )

    def test_ar1_ci_coverage_h6(self):
        """AR(1) horizon=6: CI coverage >= 75%."""
        metrics = rolling_backtest(_ar1_data, (1, 0, 0), (0, 0, 0, 0), _ar1_origins, horizon=6)
        assert metrics is not None, "All fits failed"
        assert 0.75 <= metrics["ci_coverage"] <= 1.0, (
            f"AR(1) h=6 coverage={metrics['ci_coverage']:.3f}, expected [0.75, 1.0]"
        )

    def test_sarima_ci_coverage(self):
        """SARIMA(1,1,1)(1,1,1,12) horizon=1: CI coverage >= 75%."""
        metrics = rolling_backtest(
            _sarima_data, (1, 1, 1), (1, 1, 1, 12), _sarima_origins, horizon=1,
        )
        assert metrics is not None, "All fits failed"
        assert 0.75 <= metrics["ci_coverage"] <= 1.0, (
            f"SARIMA h=1 coverage={metrics['ci_coverage']:.3f}, expected [0.75, 1.0]"
        )


class TestForecastMetrics:
    """MAE/RMSE/MAPE가 유한하고 합리적인지 검증."""

    def test_arima111_rmse_finite(self):
        """ARIMA(1,1,1) 모든 horizon에서 RMSE finite & positive."""
        for h in [1, 6]:
            metrics = rolling_backtest(
                _arima_data, (1, 1, 1), (0, 0, 0, 0), _arima_origins, horizon=h,
            )
            assert metrics is not None, f"All fits failed at h={h}"
            assert np.isfinite(metrics["rmse"]) and metrics["rmse"] > 0, (
                f"ARIMA(1,1,1) h={h} RMSE={metrics['rmse']}"
            )

    def test_arima111_mape_bounded(self):
        """ARIMA(1,1,1) horizon=1: MAPE < 200%."""
        metrics = rolling_backtest(
            _arima_data, (1, 1, 1), (0, 0, 0, 0), _arima_origins, horizon=1,
        )
        assert metrics is not None, "All fits failed"
        # MAPE는 random walk에서 매우 클 수 있으므로 NaN 허용
        if np.isfinite(metrics["mape"]):
            assert metrics["mape"] < 200, (
                f"ARIMA(1,1,1) h=1 MAPE={metrics['mape']:.1f}%"
            )


class TestCIWidth:
    """CI width가 horizon 증가에 따라 넓어지는지 검증."""

    def test_ci_width_increases_with_horizon(self):
        """모든 모델에서 CI width h=6 >= h=1."""
        configs = [
            (_ar1_data, (1, 0, 0), (0, 0, 0, 0), _ar1_origins, "AR(1)"),
            (_arima_data, (1, 1, 1), (0, 0, 0, 0), _arima_origins, "ARIMA(1,1,1)"),
        ]
        for y, order, seasonal, origins, label in configs:
            m1 = rolling_backtest(y, order, seasonal, origins, horizon=1)
            m6 = rolling_backtest(y, order, seasonal, origins, horizon=6)
            if m1 is None or m6 is None:
                continue
            # h=6 CI는 h=1보다 넓어야 함 (약간의 수치 노이즈 허용)
            assert m6["avg_width"] >= m1["avg_width"] - 0.01, (
                f"{label}: h=6 width={m6['avg_width']:.4f} < h=1 width={m1['avg_width']:.4f}"
            )


class TestPredictionQualityReport:
    """ver5/result/prediction_quality_report.md 생성."""

    def test_generate_report(self):
        """전체 모델 예측 품질 리포트 생성."""
        configs = [
            ("AR(1)", _ar1_data, (1, 0, 0), (0, 0, 0, 0), _ar1_origins),
            ("ARIMA(1,1,1)", _arima_data, (1, 1, 1), (0, 0, 0, 0), _arima_origins),
            ("SARIMA(1,1,1)(1,1,1,12)", _sarima_data, (1, 1, 1), (1, 1, 1, 12), _sarima_origins),
        ]
        horizons = [1, 6]

        lines = []
        lines.append("# 예측 품질 리포트\n")
        lines.append(f"> 생성: {datetime.now():%Y-%m-%d %H:%M}  ")
        lines.append(f"> sarimax_rs v{sarimax_rs.version()}\n")

        lines.append("## Rolling-origin 백테스트 결과\n")
        lines.append("| Model | Horizon | MAE | RMSE | MAPE (%) | CI Coverage | Avg CI Width | #Forecasts |")
        lines.append("|-------|--------:|----:|-----:|---------:|------------:|-------------:|-----------:|")

        for label, y, order, seasonal, origins in configs:
            for h in horizons:
                m = rolling_backtest(y, order, seasonal, origins, horizon=h)
                if m is None:
                    lines.append(f"| {label} | {h} | N/A | N/A | N/A | N/A | N/A | 0 |")
                    continue
                mape_str = f"{m['mape']:.1f}" if np.isfinite(m['mape']) else "N/A"
                lines.append(
                    f"| {label} | {h} "
                    f"| {m['mae']:.4f} | {m['rmse']:.4f} | {mape_str} "
                    f"| {m['ci_coverage']:.3f} | {m['avg_width']:.4f} "
                    f"| {m['n_forecasts']} |"
                )

        lines.append("")

        out_path = Path(__file__).resolve().parent.parent.parent / "ver5" / "result" / "prediction_quality_report.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines), encoding="utf-8")
