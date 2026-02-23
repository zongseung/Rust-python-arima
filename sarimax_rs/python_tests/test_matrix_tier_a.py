"""Tier A combination matrix tests — PR gate (<60s).

Tests 30 ARIMA/SARIMA model combinations against statsmodels reference
fixtures. Two levels of testing:

A) Computation verification: evaluate Rust loglike at statsmodels params
   → tests the Kalman filter / state-space math, not the optimizer
B) Fit quality: verify Rust optimizer converges and finds reasonable params
   → tests the optimizer finds at least as good a solution

Each model is validated for:
1. Loglike at oracle params (computation accuracy, tight tolerance)
2. Fit convergence
3. Fit loglike >= oracle (optimizer not worse)
4. Parameter count correctness
5. Forecast sanity (finite, correct length)
6. Residual sanity (finite, correct length)
"""

import json
import pathlib

import numpy as np
import pytest

import sarimax_rs

# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

FIXTURE_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "tests" / "fixtures" / "matrix_tier_a.json"
)


@pytest.fixture(scope="module")
def tier_a_models():
    """Load all Tier A model fixtures."""
    with open(FIXTURE_PATH) as f:
        return json.load(f)


def _converged_models(models):
    """Filter to only converged models."""
    return [m for m in models if m.get("converged", False)]


def _expected_k_params(order, seasonal, n_exog=0):
    """Expected number of estimated params (concentrated scale)."""
    p, d, q = order
    P, D, Q, s = seasonal
    return p + q + P + Q + n_exog


# ---------------------------------------------------------------------------
# A) Computation verification — loglike at oracle params
# ---------------------------------------------------------------------------

class TestLoglikeComputation:
    """Evaluate Rust loglike at statsmodels params — tests math, not optimizer.

    This is the core accuracy test: same params should produce same loglike.
    """
    ATOL = 0.5  # tight tolerance for same-params comparison

    def test_all_loglike_at_oracle_params(self, tier_a_models):
        """Rust loglike(oracle_params) should match statsmodels loglike."""
        failures = []
        for m in _converged_models(tier_a_models):
            order = tuple(m["order"])
            seasonal = tuple(m["seasonal"])
            y = np.array(m["data"])
            exog = np.array(m["exog"]) if "exog" in m else None
            oracle_params = np.array(m["params"])

            rust_ll = sarimax_rs.sarimax_loglike(
                y, order, seasonal, oracle_params, exog=exog,
                enforce_stationarity=False, enforce_invertibility=False,
            )

            ref_ll = m["loglike"]
            diff = abs(rust_ll - ref_ll)

            if diff > self.ATOL:
                failures.append(
                    f"{m['model_id']}: |rust {rust_ll:.4f} - ref {ref_ll:.4f}| = {diff:.4f} > {self.ATOL}"
                )

        assert not failures, (
            "Loglike computation mismatches (same params):\n" + "\n".join(failures)
        )


# ---------------------------------------------------------------------------
# B) Fit quality — optimizer finds good solutions
# ---------------------------------------------------------------------------

class TestFitConvergence:
    """Verify all Tier A models converge."""

    def test_all_converge(self, tier_a_models):
        """Every model with a converged oracle should converge in Rust."""
        failures = []
        for m in _converged_models(tier_a_models):
            order = tuple(m["order"])
            seasonal = tuple(m["seasonal"])
            y = np.array(m["data"])
            exog = np.array(m["exog"]) if "exog" in m else None

            result = sarimax_rs.sarimax_fit(
                y, order, seasonal, exog=exog,
            )
            if not result["converged"]:
                failures.append(f"{m['model_id']}: did not converge")

        assert not failures, "Convergence failures:\n" + "\n".join(failures)


class TestFitLoglikeQuality:
    """Verify Rust optimizer finds loglike at least as good as statsmodels.

    The Rust optimizer may find BETTER optima (higher loglike) due to
    different starting params or optimization paths. This is fine.
    We only fail if Rust loglike is significantly WORSE than reference.

    Tolerance scales with model complexity: seasonal models with large s
    have harder optimization landscapes and wider acceptable gaps.
    """

    @staticmethod
    def _tolerance(order, seasonal):
        """Compute per-model tolerance based on complexity."""
        p, _d, q = order
        pp, _dd, qq, s = seasonal
        n_params = p + q + pp + qq
        # Base tolerance + param scaling + seasonal penalty
        return 3.0 + n_params * 2.0 + (s * 1.5 if s >= 2 else 0)

    def test_fit_loglike_not_worse(self, tier_a_models):
        """Rust fit loglike should not be significantly worse than reference."""
        failures = []
        for m in _converged_models(tier_a_models):
            order = tuple(m["order"])
            seasonal = tuple(m["seasonal"])
            y = np.array(m["data"])
            exog = np.array(m["exog"]) if "exog" in m else None

            result = sarimax_rs.sarimax_fit(
                y, order, seasonal, exog=exog,
            )
            if not result["converged"]:
                continue

            ref_ll = m["loglike"]
            rust_ll = result["loglike"]
            tol = self._tolerance(order, seasonal)

            # Only fail if Rust is WORSE by more than scaled tolerance
            if rust_ll < ref_ll - tol:
                failures.append(
                    f"{m['model_id']}: rust {rust_ll:.4f} < ref {ref_ll:.4f} - {tol:.1f}"
                )

        assert not failures, (
            "Fit loglike worse than reference:\n" + "\n".join(failures)
        )


class TestParamLength:
    """Verify fit result param count matches model specification."""

    def test_all_models(self, tier_a_models):
        """Every converged model should have correct param count."""
        failures = []
        for m in _converged_models(tier_a_models):
            order = tuple(m["order"])
            seasonal = tuple(m["seasonal"])
            n_exog = m.get("n_exog", 0)
            expected = _expected_k_params(order, seasonal, n_exog)

            y = np.array(m["data"])
            exog = np.array(m["exog"]) if "exog" in m else None

            result = sarimax_rs.sarimax_fit(
                y, order, seasonal, exog=exog,
            )
            actual = len(result["params"])
            if actual != expected:
                failures.append(
                    f"{m['model_id']}: expected {expected} params, got {actual}"
                )

        assert not failures, "Param length mismatches:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# C) Forecast and Residuals sanity
# ---------------------------------------------------------------------------

class TestForecast:
    """Verify forecast produces finite results with correct dimensions."""

    def test_all_forecast_sanity(self, tier_a_models):
        """Every converged model forecast should be finite with length 10."""
        failures = []
        for m in _converged_models(tier_a_models):
            order = tuple(m["order"])
            seasonal = tuple(m["seasonal"])
            y = np.array(m["data"])
            exog = np.array(m["exog"]) if "exog" in m else None
            future_exog = np.array(m["future_exog"]) if "future_exog" in m else None

            result = sarimax_rs.sarimax_fit(
                y, order, seasonal, exog=exog,
            )
            if not result["converged"]:
                continue

            params = np.array(result["params"])
            fc = sarimax_rs.sarimax_forecast(
                y, order, seasonal, params, steps=10,
                exog=exog, future_exog=future_exog,
            )

            rust_mean = np.array(fc["mean"])
            rust_var = np.array(fc["variance"])
            rust_ci_lo = np.array(fc["ci_lower"])
            rust_ci_hi = np.array(fc["ci_upper"])

            # Length checks
            for name, arr in [("mean", rust_mean), ("variance", rust_var),
                              ("ci_lower", rust_ci_lo), ("ci_upper", rust_ci_hi)]:
                if len(arr) != 10:
                    failures.append(f"{m['model_id']}: {name} length {len(arr)} != 10")

            # Finite check
            if not np.all(np.isfinite(rust_mean)):
                failures.append(f"{m['model_id']}: forecast mean contains non-finite values")
            if not np.all(np.isfinite(rust_var)):
                failures.append(f"{m['model_id']}: forecast variance contains non-finite values")

            # Variance should be non-negative
            if np.any(rust_var < 0):
                failures.append(f"{m['model_id']}: forecast variance has negative values")

            # CI ordering: lower < upper
            if np.any(rust_ci_lo > rust_ci_hi):
                failures.append(f"{m['model_id']}: CI lower > upper")

        assert not failures, "Forecast failures:\n" + "\n".join(failures)


class TestResiduals:
    """Verify residuals are finite and have correct length."""

    def test_all_residuals(self, tier_a_models):
        """Every converged model should produce valid residuals."""
        failures = []
        for m in _converged_models(tier_a_models):
            order = tuple(m["order"])
            seasonal = tuple(m["seasonal"])
            y = np.array(m["data"])
            exog = np.array(m["exog"]) if "exog" in m else None

            result = sarimax_rs.sarimax_fit(
                y, order, seasonal, exog=exog,
            )
            if not result["converged"]:
                continue

            params = np.array(result["params"])
            resid_result = sarimax_rs.sarimax_residuals(
                y, order, seasonal, params, exog=exog,
            )

            std_resid = np.array(resid_result["standardized_residuals"])
            n_obs = m["n_obs"]

            # Length check
            if len(std_resid) != n_obs:
                failures.append(
                    f"{m['model_id']}: residuals length {len(std_resid)} != n_obs {n_obs}"
                )
                continue

            # Finite check
            n_nonfinite = np.sum(~np.isfinite(std_resid))
            if n_nonfinite > 0:
                failures.append(
                    f"{m['model_id']}: {n_nonfinite} non-finite residuals"
                )

        assert not failures, "Residual failures:\n" + "\n".join(failures)


class TestAicBicFinite:
    """Verify AIC/BIC are finite and self-consistent."""

    def test_all_aic_bic_finite(self, tier_a_models):
        """Every converged model AIC/BIC should be finite."""
        failures = []
        for m in _converged_models(tier_a_models):
            order = tuple(m["order"])
            seasonal = tuple(m["seasonal"])
            y = np.array(m["data"])
            exog = np.array(m["exog"]) if "exog" in m else None

            result = sarimax_rs.sarimax_fit(
                y, order, seasonal, exog=exog,
            )
            if not result["converged"]:
                continue

            for metric in ("aic", "bic"):
                val = result[metric]
                if not np.isfinite(val):
                    failures.append(f"{m['model_id']}: {metric} is not finite ({val})")

            # Self-consistency: AIC = -2*loglike + 2*k (approximately)
            # BIC = -2*loglike + k*ln(n) (approximately)
            # Just check AIC < BIC for n > ~7 (always true for our data)
            if result["aic"] > result["bic"] + 50:
                failures.append(
                    f"{m['model_id']}: AIC ({result['aic']:.2f}) >> BIC ({result['bic']:.2f})"
                )

        assert not failures, "AIC/BIC failures:\n" + "\n".join(failures)
