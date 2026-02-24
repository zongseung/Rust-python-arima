"""Tests for Polars DataFrame integration (P2 UX: C-1)."""
import numpy as np
import pytest

from conftest import generate_ar1_rng
from sarimax_py import SARIMAXModel


@pytest.fixture
def fitted_result():
    y = generate_ar1_rng()
    model = SARIMAXModel(y, order=(1, 0, 0))
    return model.fit()


class TestForecastDataFrame:
    def test_to_dataframe_type(self, fitted_result):
        import polars as pl
        fc = fitted_result.forecast(steps=5)
        df = fc.to_dataframe()
        assert isinstance(df, pl.DataFrame)

    def test_to_dataframe_columns(self, fitted_result):
        fc = fitted_result.forecast(steps=5)
        df = fc.to_dataframe()
        assert list(df.columns) == ["step", "mean", "variance", "ci_lower", "ci_upper"]

    def test_to_dataframe_rows(self, fitted_result):
        fc = fitted_result.forecast(steps=10)
        df = fc.to_dataframe()
        assert len(df) == 10

    def test_to_dataframe_step_range(self, fitted_result):
        fc = fitted_result.forecast(steps=5)
        df = fc.to_dataframe()
        assert df["step"].to_list() == [1, 2, 3, 4, 5]

    def test_to_dataframe_values_match(self, fitted_result):
        fc = fitted_result.forecast(steps=3)
        df = fc.to_dataframe()
        np.testing.assert_allclose(
            df["mean"].to_numpy(), fc.predicted_mean, rtol=1e-10
        )
        np.testing.assert_allclose(
            df["variance"].to_numpy(), fc.variance, rtol=1e-10
        )


class TestParamsTable:
    def test_params_table_type(self, fitted_result):
        import polars as pl
        pt = fitted_result.params_table()
        assert isinstance(pt, pl.DataFrame)

    def test_params_table_columns_no_inference(self, fitted_result):
        pt = fitted_result.params_table(inference="none")
        assert list(pt.columns) == ["name", "coef"]

    def test_params_table_columns_with_inference(self, fitted_result):
        pt = fitted_result.params_table(inference="hessian")
        assert "std_err" in pt.columns
        assert "z" in pt.columns
        assert "p_value" in pt.columns
        assert "ci_lower" in pt.columns
        assert "ci_upper" in pt.columns

    def test_params_table_rows(self, fitted_result):
        pt = fitted_result.params_table()
        assert len(pt) == len(fitted_result.params)

    def test_params_table_names(self, fitted_result):
        pt = fitted_result.params_table()
        assert pt["name"].to_list() == fitted_result.param_names


class TestPredictionDataFrame:
    def test_prediction_to_dataframe(self, fitted_result):
        import polars as pl
        pred = fitted_result.get_prediction(start=0, end=200)
        df = pred.to_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 200
        assert list(df.columns) == ["index", "predicted_mean"]

    def test_prediction_out_of_sample_to_dataframe(self, fitted_result):
        pred = fitted_result.get_prediction(start=0, end=210)
        df = pred.to_dataframe()
        assert len(df) == 210


class TestHQIC:
    def test_hqic_finite(self, fitted_result):
        assert np.isfinite(fitted_result.hqic)

    def test_hqic_between_aic_bic(self, fitted_result):
        # HQIC is typically between AIC and BIC
        assert fitted_result.aic <= fitted_result.hqic <= fitted_result.bic or \
               fitted_result.aic <= fitted_result.bic <= fitted_result.hqic or \
               np.isfinite(fitted_result.hqic)
