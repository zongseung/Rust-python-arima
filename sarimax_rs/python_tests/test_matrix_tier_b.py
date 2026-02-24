"""Tier B combination matrix tests — Nightly (<10 min).

Extended test matrix with ~70 models including:
- Higher order (p,q up to 5, P,Q up to 2)
- Multiple seasonal periods (s=2,4,7,12,24)
- Short (n=50) and long (n=1000) series
- Exogenous variable models
- d=2 differencing

Uses same validation logic as Tier A but with nightly marker.
"""

import json
import pathlib

import numpy as np
import pytest

import sarimax_rs
from conftest import converged_models, expected_k_params

# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

FIXTURE_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "tests" / "fixtures" / "matrix_tier_b.json"
)


@pytest.fixture(scope="module")
def tier_b_models():
    """Load all Tier B model fixtures."""
    if not FIXTURE_PATH.exists():
        pytest.skip("Tier B fixtures not generated yet")
    with open(FIXTURE_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tests (marked nightly)
# ---------------------------------------------------------------------------

@pytest.mark.nightly
class TestTierBLoglikeComputation:
    """Evaluate Rust loglike at statsmodels params for all Tier B models."""
    ATOL = 0.5

    def test_loglike_at_oracle_params(self, tier_b_models):
        failures = []
        for m in converged_models(tier_b_models):
            order = tuple(m["order"])
            seasonal = tuple(m["seasonal"])
            y = np.array(m["data"])
            exog = np.array(m["exog"]) if "exog" in m else None
            oracle_params = np.array(m["params"])

            try:
                rust_ll = sarimax_rs.sarimax_loglike(
                    y, order, seasonal, oracle_params, exog=exog,
                    enforce_stationarity=False, enforce_invertibility=False,
                )
                diff = abs(rust_ll - m["loglike"])
                if diff > self.ATOL:
                    failures.append(
                        f"{m['model_id']}: |{rust_ll:.4f} - {m['loglike']:.4f}| = {diff:.4f}"
                    )
            except Exception as e:
                failures.append(f"{m['model_id']}: ERROR {e}")

        n_total = len(converged_models(tier_b_models))
        n_fail = len(failures)
        pass_rate = (n_total - n_fail) / n_total * 100 if n_total > 0 else 0

        assert pass_rate >= 95, (
            f"Loglike pass rate {pass_rate:.1f}% < 95%:\n" + "\n".join(failures[:20])
        )


@pytest.mark.nightly
class TestTierBFitConvergence:
    """Verify convergence rate for Tier B models."""

    def test_convergence_rate(self, tier_b_models):
        n_total = len(converged_models(tier_b_models))
        n_converged = 0
        failures = []

        for m in converged_models(tier_b_models):
            order = tuple(m["order"])
            seasonal = tuple(m["seasonal"])
            y = np.array(m["data"])
            exog = np.array(m["exog"]) if "exog" in m else None

            try:
                result = sarimax_rs.sarimax_fit(
                    y, order, seasonal, exog=exog,
                )
                if result["converged"]:
                    n_converged += 1
                else:
                    failures.append(f"{m['model_id']}: did not converge")
            except Exception as e:
                failures.append(f"{m['model_id']}: ERROR {e}")

        rate = n_converged / n_total * 100 if n_total > 0 else 0
        assert rate >= 95, (
            f"Convergence rate {rate:.1f}% ({n_converged}/{n_total}) < 95%:\n"
            + "\n".join(failures[:20])
        )


@pytest.mark.nightly
class TestTierBParamLength:
    """Verify param count for Tier B models."""

    def test_param_lengths(self, tier_b_models):
        failures = []
        for m in converged_models(tier_b_models):
            order = tuple(m["order"])
            seasonal = tuple(m["seasonal"])
            n_exog = m.get("n_exog", 0)
            expected = expected_k_params(order, seasonal, n_exog)

            y = np.array(m["data"])
            exog = np.array(m["exog"]) if "exog" in m else None

            try:
                result = sarimax_rs.sarimax_fit(
                    y, order, seasonal, exog=exog,
                )
                actual = len(result["params"])
                if actual != expected:
                    failures.append(
                        f"{m['model_id']}: expected {expected}, got {actual}"
                    )
            except Exception as e:
                failures.append(f"{m['model_id']}: ERROR {e}")

        assert not failures, "Param length mismatches:\n" + "\n".join(failures)


@pytest.mark.nightly
class TestTierBForecastSanity:
    """Verify forecast sanity for Tier B models."""

    def test_all_forecasts_finite(self, tier_b_models):
        failures = []
        for m in converged_models(tier_b_models):
            order = tuple(m["order"])
            seasonal = tuple(m["seasonal"])
            y = np.array(m["data"])
            exog = np.array(m["exog"]) if "exog" in m else None
            future_exog = np.array(m["future_exog"]) if "future_exog" in m else None

            try:
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

                mean = np.array(fc["mean"])
                if len(mean) != 10:
                    failures.append(f"{m['model_id']}: mean length {len(mean)} != 10")
                if not np.all(np.isfinite(mean)):
                    failures.append(f"{m['model_id']}: non-finite forecast mean")
            except Exception as e:
                failures.append(f"{m['model_id']}: ERROR {e}")

        n_total = len(converged_models(tier_b_models))
        n_fail = len(failures)
        pass_rate = (n_total - n_fail) / n_total * 100 if n_total > 0 else 0

        assert pass_rate >= 95, (
            f"Forecast pass rate {pass_rate:.1f}% < 95%:\n" + "\n".join(failures[:20])
        )


@pytest.mark.nightly
class TestTierBResiduals:
    """Verify residual sanity for Tier B models."""

    def test_all_residuals_finite(self, tier_b_models):
        failures = []
        for m in converged_models(tier_b_models):
            order = tuple(m["order"])
            seasonal = tuple(m["seasonal"])
            y = np.array(m["data"])
            exog = np.array(m["exog"]) if "exog" in m else None

            try:
                result = sarimax_rs.sarimax_fit(
                    y, order, seasonal, exog=exog,
                )
                if not result["converged"]:
                    continue

                params = np.array(result["params"])
                resid = sarimax_rs.sarimax_residuals(
                    y, order, seasonal, params, exog=exog,
                )

                std_resid = np.array(resid["standardized_residuals"])
                if len(std_resid) != m["n_obs"]:
                    failures.append(
                        f"{m['model_id']}: resid length {len(std_resid)} != {m['n_obs']}"
                    )
                n_bad = np.sum(~np.isfinite(std_resid))
                if n_bad > 0:
                    failures.append(f"{m['model_id']}: {n_bad} non-finite residuals")
            except Exception as e:
                failures.append(f"{m['model_id']}: ERROR {e}")

        n_total = len(converged_models(tier_b_models))
        n_fail = len(failures)
        pass_rate = (n_total - n_fail) / n_total * 100 if n_total > 0 else 0

        assert pass_rate >= 95, (
            f"Residual pass rate {pass_rate:.1f}% < 95%:\n" + "\n".join(failures[:20])
        )
