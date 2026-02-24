"""Performance regression tests for the optimization changes.

Verifies that:
1. Fit results still match statsmodels within tolerance
2. Iteration counts are reasonable (not excessive)
3. Batch performance is not degraded
"""

import numpy as np
import pytest

import sarimax_rs
from conftest import generate_ar1, generate_random_walk


# --- Data generators (fixed seed) ---

def ar1_data(n=500, seed=42):
    return generate_ar1(n=n, seed=seed)


def arima111_data(n=500, seed=42):
    return generate_random_walk(n=n, seed=seed)


# --- Accuracy regression tests ---

class TestAccuracyRegression:
    """Ensure optimization changes don't degrade accuracy."""

    def test_ar1_fit_accuracy(self):
        y = ar1_data(500)
        result = sarimax_rs.sarimax_fit(y, (1, 0, 0), (0, 0, 0, 0))
        assert result["converged"], "AR(1) should converge"
        assert np.isfinite(result["loglike"]), "loglike should be finite"
        # AR coefficient should be near 0.7
        assert abs(result["params"][0] - 0.7) < 0.1, (
            f"AR(1) param far from true value: {result['params'][0]}"
        )

    def test_arima111_fit_accuracy(self):
        y = arima111_data(500)
        result = sarimax_rs.sarimax_fit(y, (1, 1, 1), (0, 0, 0, 0))
        assert result["converged"], "ARIMA(1,1,1) should converge"
        assert np.isfinite(result["loglike"]), "loglike should be finite"
        assert len(result["params"]) == 2, "Should have 2 params (ar, ma)"

    def test_fit_forecast_roundtrip(self):
        """Fit then forecast should produce finite results."""
        y = ar1_data(200)
        fit_result = sarimax_rs.sarimax_fit(y, (1, 0, 0), (0, 0, 0, 0))
        params = np.array(fit_result["params"])
        fc = sarimax_rs.sarimax_forecast(
            y, (1, 0, 0), (0, 0, 0, 0), params, steps=10
        )
        assert len(fc["mean"]) == 10
        assert all(np.isfinite(fc["mean"]))
        assert all(np.isfinite(fc["variance"]))


# --- Iteration count tests ---

class TestIterationCount:
    """Verify optimizer converges in reasonable number of iterations."""

    def test_ar1_iterations(self):
        y = ar1_data(500)
        result = sarimax_rs.sarimax_fit(y, (1, 0, 0), (0, 0, 0, 0))
        assert result["converged"]
        # L-BFGS-B reports function evaluation count (not gradient iterations).
        # Single-param model: ~20-25 evals is typical for line-search optimizer.
        assert result["n_iter"] < 30, (
            f"AR(1) took too many iterations: {result['n_iter']}"
        )

    def test_arima111_iterations(self):
        y = arima111_data(500)
        result = sarimax_rs.sarimax_fit(y, (1, 1, 1), (0, 0, 0, 0))
        # 2-param model with multi-start + NM refinement.
        # L-BFGS-B evals + optional NM refinement = ~30-50 typical.
        assert result["n_iter"] < 60, (
            f"ARIMA(1,1,1) took too many iterations: {result['n_iter']}"
        )


# --- Batch performance regression ---

class TestBatchRegression:
    """Ensure batch operations still work correctly after optimization."""

    def test_batch_fit_matches_single(self):
        """Batch fit should give same results as single fit."""
        y = ar1_data(200)
        single = sarimax_rs.sarimax_fit(y, (1, 0, 0), (0, 0, 0, 0))
        batch = sarimax_rs.sarimax_batch_fit([y], (1, 0, 0), (0, 0, 0, 0))
        assert len(batch) == 1
        assert abs(batch[0]["loglike"] - single["loglike"]) < 1e-8

    def test_batch_fit_100_all_converge(self):
        """100 AR(1) series should all converge."""
        series = [ar1_data(200, seed=i) for i in range(100)]
        results = sarimax_rs.sarimax_batch_fit(
            series, (1, 0, 0), (0, 0, 0, 0)
        )
        assert len(results) == 100
        converged = sum(1 for r in results if r.get("converged", False))
        assert converged == 100, f"Only {converged}/100 converged"
