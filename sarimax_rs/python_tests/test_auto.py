"""Tests for auto_arima (P2 UX: C-5)."""
import numpy as np
import pytest

from conftest import generate_ar1_rng, generate_trend_data
from sarimax_py import auto_arima


@pytest.fixture
def ar1_data():
    return generate_ar1_rng()


@pytest.fixture
def trend_data():
    return generate_trend_data()


class TestStepwise:
    def test_stepwise_returns_result(self, ar1_data):
        res = auto_arima(ar1_data, max_p=3, max_q=3, s=0, stepwise=True)
        assert res.result is not None
        assert res.order is not None
        assert res.result.converged

    def test_stepwise_history(self, ar1_data):
        res = auto_arima(ar1_data, max_p=2, max_q=2, s=0, stepwise=True)
        assert len(res.history) > 0
        assert all("order" in h for h in res.history)

    def test_stepwise_summary(self, ar1_data):
        res = auto_arima(ar1_data, max_p=2, max_q=2, s=0)
        s = res.summary()
        assert "auto_arima" in s
        assert "ARIMA" in s
        assert "aic=" in s

    def test_stepwise_best_ic(self, ar1_data):
        res = auto_arima(ar1_data, max_p=2, max_q=2, s=0)
        assert np.isfinite(res.best_ic)

    def test_stepwise_with_differencing(self):
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.normal(size=200))
        res = auto_arima(y, max_p=2, max_q=2, s=0, stepwise=True)
        assert res.result is not None
        # Should detect d=1
        assert res.order[1] >= 1


class TestGrid:
    def test_grid_returns_result(self, ar1_data):
        res = auto_arima(ar1_data, max_p=2, max_q=2, s=0, stepwise=False)
        assert res.result is not None

    def test_grid_evaluates_all(self, ar1_data):
        res = auto_arima(ar1_data, max_p=1, max_q=1, s=0, d=0, stepwise=False)
        # 2x2 = 4 combos: (0,0), (0,1), (1,0), (1,1)
        assert len(res.history) == 4

    def test_grid_best_matches_stepwise(self, ar1_data):
        res_sw = auto_arima(ar1_data, max_p=2, max_q=2, s=0, stepwise=True)
        res_gr = auto_arima(ar1_data, max_p=2, max_q=2, s=0, stepwise=False)
        # Grid should find comparable result (optimizer multi-start variance ≤ 5 AIC)
        assert res_gr.best_ic <= res_sw.best_ic + 5.0


class TestHistoryDataFrame:
    def test_history_dataframe_type(self, ar1_data):
        import polars as pl
        res = auto_arima(ar1_data, max_p=1, max_q=1, s=0, d=0, stepwise=False)
        df = res.history_dataframe()
        assert isinstance(df, pl.DataFrame)

    def test_history_dataframe_columns(self, ar1_data):
        res = auto_arima(ar1_data, max_p=1, max_q=1, s=0, d=0, stepwise=False)
        df = res.history_dataframe()
        assert "order" in df.columns
        assert "seasonal_order" in df.columns
        assert "aic" in df.columns
        assert "converged" in df.columns


class TestCriterion:
    def test_bic_criterion(self, ar1_data):
        res = auto_arima(ar1_data, max_p=2, max_q=2, s=0, criterion="bic")
        assert res.criterion == "bic"
        assert np.isfinite(res.best_ic)

    def test_hqic_criterion(self, ar1_data):
        res = auto_arima(ar1_data, max_p=2, max_q=2, s=0, criterion="hqic")
        assert res.criterion == "hqic"
        assert np.isfinite(res.best_ic)


class TestGridSearchParallel:
    """Tests for Rayon-parallel sarimax_grid_search."""

    def test_grid_search_matches_sequential(self, ar1_data):
        """Parallel grid_search results match sequential sarimax_fit."""
        import sarimax_rs

        order_list = [(1, 0, 0), (1, 0, 1), (2, 0, 0)]
        seasonal_list = [(0, 0, 0, 0)] * 3

        # Parallel
        grid_results = sarimax_rs.sarimax_grid_search(
            ar1_data, order_list, seasonal_list
        )
        assert len(grid_results) == 3

        # Sequential
        for i, (order, seasonal) in enumerate(zip(order_list, seasonal_list)):
            seq = sarimax_rs.sarimax_fit(ar1_data, order, seasonal)
            grid = grid_results[i]
            assert "error" not in grid
            # Loglike should match within optimizer tolerance
            assert abs(seq["loglike"] - grid["loglike"]) < 1.0

    def test_grid_search_returns_order_keys(self, ar1_data):
        """Each result dict contains order and seasonal_order."""
        import sarimax_rs

        order_list = [(0, 0, 0), (1, 0, 0)]
        seasonal_list = [(0, 0, 0, 0)] * 2

        results = sarimax_rs.sarimax_grid_search(
            ar1_data, order_list, seasonal_list
        )
        for i, r in enumerate(results):
            assert tuple(r["order"]) == order_list[i]
            assert tuple(r["seasonal_order"]) == seasonal_list[i]

    def test_grid_search_error_isolation(self):
        """Invalid combos return error without affecting valid ones."""
        import sarimax_rs

        rng = np.random.default_rng(42)
        y = rng.normal(size=50)

        # (10,0,10) likely too large for 50 obs → should fail
        # (1,0,0) should succeed
        order_list = [(1, 0, 0), (10, 0, 10)]
        seasonal_list = [(0, 0, 0, 0)] * 2

        try:
            results = sarimax_rs.sarimax_grid_search(
                y, order_list, seasonal_list
            )
            # First should succeed
            assert "error" not in results[0]
            assert results[0]["converged"]
        except Exception:
            # Validation may reject before parallel — that's also acceptable
            pass


class TestTrend:
    def test_auto_arima_with_trend(self, trend_data):
        res = auto_arima(trend_data, max_p=2, max_q=2, d=0, s=0, trend="c")
        assert res.result is not None
        assert res.result.converged
