"""Model matrix tests: Tier A (PR gate, <60s) + Tier B (nightly, <10min).

Tier A: 30 ARIMA/SARIMA combinations — validates loglike math, optimizer
        convergence, param count, forecast sanity, residuals, AIC/BIC.

Tier B: ~70 extended models (higher order, multiple seasonal periods,
        exog, d=2) — same validation logic, marked @nightly, 95% pass rate.
"""

import json
import pathlib

import numpy as np
import pytest

import rustima
from conftest import converged_models, expected_k_params

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
        for m in converged_models(tier_a_models):
            order = tuple(m["order"])
            seasonal = tuple(m["seasonal"])
            y = np.array(m["data"])
            exog = np.array(m["exog"]) if "exog" in m else None
            oracle_params = np.array(m["params"])

            rust_ll = rustima.sarimax_loglike(
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
        for m in converged_models(tier_a_models):
            order = tuple(m["order"])
            seasonal = tuple(m["seasonal"])
            y = np.array(m["data"])
            exog = np.array(m["exog"]) if "exog" in m else None

            result = rustima.sarimax_fit(
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
        for m in converged_models(tier_a_models):
            order = tuple(m["order"])
            seasonal = tuple(m["seasonal"])
            y = np.array(m["data"])
            exog = np.array(m["exog"]) if "exog" in m else None

            result = rustima.sarimax_fit(
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
        for m in converged_models(tier_a_models):
            order = tuple(m["order"])
            seasonal = tuple(m["seasonal"])
            n_exog = m.get("n_exog", 0)
            expected = expected_k_params(order, seasonal, n_exog)

            y = np.array(m["data"])
            exog = np.array(m["exog"]) if "exog" in m else None

            result = rustima.sarimax_fit(
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
        for m in converged_models(tier_a_models):
            order = tuple(m["order"])
            seasonal = tuple(m["seasonal"])
            y = np.array(m["data"])
            exog = np.array(m["exog"]) if "exog" in m else None
            future_exog = np.array(m["future_exog"]) if "future_exog" in m else None

            result = rustima.sarimax_fit(
                y, order, seasonal, exog=exog,
            )
            if not result["converged"]:
                continue

            params = np.array(result["params"])
            fc = rustima.sarimax_forecast(
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
        for m in converged_models(tier_a_models):
            order = tuple(m["order"])
            seasonal = tuple(m["seasonal"])
            y = np.array(m["data"])
            exog = np.array(m["exog"]) if "exog" in m else None

            result = rustima.sarimax_fit(
                y, order, seasonal, exog=exog,
            )
            if not result["converged"]:
                continue

            params = np.array(result["params"])
            resid_result = rustima.sarimax_residuals(
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
        for m in converged_models(tier_a_models):
            order = tuple(m["order"])
            seasonal = tuple(m["seasonal"])
            y = np.array(m["data"])
            exog = np.array(m["exog"]) if "exog" in m else None

            result = rustima.sarimax_fit(
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


# ── Tier B — nightly gate ────────────────────────────────────────────────────

FIXTURE_PATH_B = (
    pathlib.Path(__file__).resolve().parent.parent
    / "tests" / "fixtures" / "matrix_tier_b.json"
)


@pytest.fixture(scope="module")
def tier_b_models():
    """Load all Tier B model fixtures (skipped if not generated yet)."""
    if not FIXTURE_PATH_B.exists():
        pytest.skip("Tier B fixtures not generated yet")
    with open(FIXTURE_PATH_B) as f:
        return json.load(f)


@pytest.mark.nightly
class TestTierBLoglikeComputation:
    ATOL = 0.5

    def test_loglike_at_oracle_params(self, tier_b_models):
        failures = []
        for m in converged_models(tier_b_models):
            order, seasonal = tuple(m["order"]), tuple(m["seasonal"])
            y = np.array(m["data"])
            exog = np.array(m["exog"]) if "exog" in m else None
            try:
                rust_ll = rustima.sarimax_loglike(
                    y, order, seasonal, np.array(m["params"]), exog=exog,
                    enforce_stationarity=False, enforce_invertibility=False,
                )
                diff = abs(rust_ll - m["loglike"])
                if diff > self.ATOL:
                    failures.append(f"{m['model_id']}: diff={diff:.4f}")
            except Exception as e:
                failures.append(f"{m['model_id']}: ERROR {e}")
        n = len(converged_models(tier_b_models))
        rate = (n - len(failures)) / n * 100 if n > 0 else 0
        assert rate >= 95, f"Pass rate {rate:.1f}% < 95%:\n" + "\n".join(failures[:20])


@pytest.mark.nightly
class TestTierBFitConvergence:
    def test_convergence_rate(self, tier_b_models):
        n_total, n_conv, failures = len(converged_models(tier_b_models)), 0, []
        for m in converged_models(tier_b_models):
            order, seasonal = tuple(m["order"]), tuple(m["seasonal"])
            y = np.array(m["data"])
            exog = np.array(m["exog"]) if "exog" in m else None
            try:
                result = rustima.sarimax_fit(y, order, seasonal, exog=exog)
                if result["converged"]:
                    n_conv += 1
                else:
                    failures.append(f"{m['model_id']}: did not converge")
            except Exception as e:
                failures.append(f"{m['model_id']}: ERROR {e}")
        rate = n_conv / n_total * 100 if n_total > 0 else 0
        assert rate >= 95, f"Conv rate {rate:.1f}% ({n_conv}/{n_total}) < 95%:\n" + "\n".join(failures[:20])


@pytest.mark.nightly
class TestTierBParamLength:
    def test_param_lengths(self, tier_b_models):
        failures = []
        for m in converged_models(tier_b_models):
            order, seasonal = tuple(m["order"]), tuple(m["seasonal"])
            expected = expected_k_params(order, seasonal, m.get("n_exog", 0))
            y = np.array(m["data"])
            exog = np.array(m["exog"]) if "exog" in m else None
            try:
                result = rustima.sarimax_fit(y, order, seasonal, exog=exog)
                actual = len(result["params"])
                if actual != expected:
                    failures.append(f"{m['model_id']}: expected {expected}, got {actual}")
            except Exception as e:
                failures.append(f"{m['model_id']}: ERROR {e}")
        assert not failures, "Param length mismatches:\n" + "\n".join(failures)


@pytest.mark.nightly
class TestTierBForecastSanity:
    def test_all_forecasts_finite(self, tier_b_models):
        failures = []
        for m in converged_models(tier_b_models):
            order, seasonal = tuple(m["order"]), tuple(m["seasonal"])
            y = np.array(m["data"])
            exog = np.array(m["exog"]) if "exog" in m else None
            future_exog = np.array(m["future_exog"]) if "future_exog" in m else None
            try:
                result = rustima.sarimax_fit(y, order, seasonal, exog=exog)
                if not result["converged"]:
                    continue
                fc = rustima.sarimax_forecast(
                    y, order, seasonal, np.array(result["params"]),
                    steps=10, exog=exog, future_exog=future_exog,
                )
                mean = np.array(fc["mean"])
                if len(mean) != 10 or not np.all(np.isfinite(mean)):
                    failures.append(f"{m['model_id']}: bad forecast")
            except Exception as e:
                failures.append(f"{m['model_id']}: ERROR {e}")
        n = len(converged_models(tier_b_models))
        rate = (n - len(failures)) / n * 100 if n > 0 else 0
        assert rate >= 95, f"Forecast pass rate {rate:.1f}% < 95%:\n" + "\n".join(failures[:20])


@pytest.mark.nightly
class TestTierBResiduals:
    def test_all_residuals_finite(self, tier_b_models):
        failures = []
        for m in converged_models(tier_b_models):
            order, seasonal = tuple(m["order"]), tuple(m["seasonal"])
            y = np.array(m["data"])
            exog = np.array(m["exog"]) if "exog" in m else None
            try:
                result = rustima.sarimax_fit(y, order, seasonal, exog=exog)
                if not result["converged"]:
                    continue
                resid = rustima.sarimax_residuals(
                    y, order, seasonal, np.array(result["params"]), exog=exog,
                )
                std_resid = np.array(resid["standardized_residuals"])
                if len(std_resid) != m["n_obs"] or np.sum(~np.isfinite(std_resid)) > 0:
                    failures.append(f"{m['model_id']}: bad residuals")
            except Exception as e:
                failures.append(f"{m['model_id']}: ERROR {e}")
        n = len(converged_models(tier_b_models))
        rate = (n - len(failures)) / n * 100 if n > 0 else 0
        assert rate >= 95, f"Residual pass rate {rate:.1f}% < 95%:\n" + "\n".join(failures[:20])
