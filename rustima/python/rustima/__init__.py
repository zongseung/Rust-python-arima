# Native extension module — maturin places the .so here automatically.
# Re-export everything from the native extension.
from .rustima import *  # noqa: F401,F403

# High-level Python API
from .model import SARIMAXModel, SARIMAXResult, ForecastResult, PredictionResult
from .auto import auto_arima, AutoARIMAResult

__all__ = [
    # Rust low-level functions
    "version",
    "sarimax_fit", "sarimax_forecast", "sarimax_loglike", "sarimax_residuals",
    "sarimax_batch_fit", "sarimax_batch_forecast", "sarimax_batch_loglike",
    "sarimax_grid_search", "sarimax_inference", "sarimax_diagnostics",
    # High-level Python API
    "SARIMAXModel", "SARIMAXResult", "ForecastResult", "PredictionResult",
    "auto_arima", "AutoARIMAResult",
]
