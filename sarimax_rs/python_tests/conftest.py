import json
import pathlib

import pytest
import numpy as np


@pytest.fixture
def simple_ar1():
    """AR(1) simulated data with phi=0.7."""
    np.random.seed(42)
    n = 200
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.7 * y[t - 1] + np.random.randn()
    return y


@pytest.fixture
def random_walk():
    """Random walk data."""
    np.random.seed(123)
    return np.cumsum(np.random.randn(300))


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
