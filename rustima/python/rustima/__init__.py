# Auto-tune the Rayon thread pool BEFORE the native extension is imported.
# Apple Silicon M-series chips mix Performance and Efficiency cores with
# very different throughput. Rayon defaults to `num_cpus = P + E`, which
# means the E-cores become the per-step critical path on CPU-bound numerical
# work (SARIMAX Kalman recursions). Setting RAYON_NUM_THREADS to the P-core
# count before the first Rayon-using call results in 20-35% lower wall time
# on M-series chips for compute-heavy workloads.
#
# This is a no-op on:
#   - non-macOS systems (the env var stays unset, Rayon picks num_cpus)
#   - macOS systems that don't expose hw.perflevel0.physicalcpu
#   - users who have already set RAYON_NUM_THREADS themselves
import os as _os
if "RAYON_NUM_THREADS" not in _os.environ:
    import sys as _sys
    if _sys.platform == "darwin":
        try:
            import subprocess as _sp
            _n = _sp.check_output(
                ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                stderr=_sp.DEVNULL, timeout=2,
            ).decode().strip()
            if _n.isdigit() and int(_n) > 0:
                _os.environ["RAYON_NUM_THREADS"] = _n
        except Exception:
            pass

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
