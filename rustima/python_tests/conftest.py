import json
import pathlib

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Shared data generators (importable by test files and benchmarks)
# ---------------------------------------------------------------------------

def generate_ar1(n=200, phi=0.7, seed=42):
    """Generate AR(1) time series with np.random.seed."""
    np.random.seed(seed)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + np.random.randn()
    return y


def generate_ar1_rng(n=200, phi=0.6, seed=42):
    """Generate AR(1) time series with np.random.default_rng."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.normal()
    return y


def generate_random_walk(n=200, seed=42):
    """Generate random walk (cumulative sum of white noise)."""
    np.random.seed(seed)
    return np.cumsum(np.random.randn(n))


def generate_trend_data(n=200, intercept=5.0, slope=0.1, phi=0.5, seed=42):
    """Generate data with linear trend + AR(1) noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + rng.normal()
    return intercept + slope * t + noise


def generate_stationary_data(n=300, phi=0.5, seed=42):
    """Generate stationary AR(1) with moderate autocorrelation."""
    return generate_ar1(n=n, phi=phi, seed=seed)


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

def converged_models(models):
    """Filter to only converged models."""
    return [m for m in models if m.get("converged", False)]


def expected_k_params(order, seasonal, n_exog=0):
    """Expected number of estimated params.

    Non-concentrated layout (current default): [exog|ar|ma|sar|sma|sigma2],
    so sigma2 adds +1.
    """
    p, _d, q = order
    P, _D, Q, _s = seasonal
    return p + q + P + Q + n_exog + 1


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_ar1():
    """AR(1) simulated data with phi=0.7."""
    return generate_ar1(n=200, phi=0.7, seed=42)


@pytest.fixture
def random_walk():
    """Random walk data."""
    return generate_random_walk(n=300, seed=123)


@pytest.fixture
def statsmodels_fixtures():
    """Load statsmodels reference fixtures."""
    path = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "statsmodels_reference.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def fit_fixtures():
    """Load statsmodels fit reference fixtures (Phase 2)."""
    path = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "statsmodels_fit_reference.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def forecast_fixtures():
    """Load statsmodels forecast reference fixtures (Phase 3)."""
    path = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "statsmodels_forecast_reference.json"
    with open(path) as f:
        return json.load(f)
