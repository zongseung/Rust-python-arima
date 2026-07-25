"""Regression tests for DIAGNOSIS_V9 robustness fixes (Phase 2, 2026-07-25).

C1/C2: rayon's global pool is not fork-safe — parallel APIs called in a
       fork()ed child must raise instead of deadlocking forever.
C3:    sarimax_rolling_forecast retains one k_states^2 covariance per origin;
       the unbounded product n_origins * k_states^2 must be capped.
S9:    batch error isolation must hold per series (documented guarantee).
W1:    masked arrays must be rejected, not silently read as raw data.
W2:    sarimax_inference must reject wrong-length params like every other
       entry point instead of returning empty arrays.
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import rustima


def _series(seed=0, n=80):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)


@pytest.mark.skipif(os.name != "posix", reason="fork is POSIX-only")
def test_batch_in_forked_child_raises_instead_of_hanging():
    """C1: parent creates the rayon pool via a batch call; a fork()ed child's
    batch call must fail fast with a clear error, not deadlock."""
    script = textwrap.dedent("""
        import os, sys
        import numpy as np
        import rustima

        y = np.random.default_rng(0).standard_normal(80)
        rustima.sarimax_batch_fit([y] * 4, (1, 0, 0), (0, 0, 0, 0))  # create pool

        pid = os.fork()
        if pid == 0:  # child
            try:
                out = rustima.sarimax_batch_fit([y] * 4, (1, 0, 0), (0, 0, 0, 0))
                errs = [o for o in out if "error" in o]
                os._exit(0 if len(errs) == 4 and "fork" in errs[0]["error"] else 3)
            except ValueError:
                os._exit(0)
            except Exception:
                os._exit(4)
        _, status = os.waitpid(pid, 0)
        sys.exit(os.waitstatus_to_exitcode(status))
    """)
    proc = subprocess.run(
        [sys.executable, "-c", script], timeout=60, capture_output=True, text=True
    )
    assert proc.returncode == 0, (
        f"forked-child batch call did not fail cleanly: rc={proc.returncode} "
        f"stderr={proc.stderr[-500:]}"
    )


def test_rolling_forecast_memory_capped():
    """C3: origin snapshots beyond the 2 GiB cap must be refused up front."""
    y = _series(1, 900)
    params = np.array([0.3, 0.2, 1.0])  # [sar1, sar2, sigma2]
    with pytest.raises(ValueError, match="rolling forecast would retain"):
        rustima.sarimax_rolling_forecast(
            y, (0, 0, 0), (2, 0, 0, 365), params, start=1, step=1, horizon=1
        )


def test_batch_error_isolation_per_series():
    """S9: one bad series must produce a per-series error dict, not kill the
    batch (this guarantee previously had zero test coverage)."""
    good = _series(2)
    # NaN/Inf are rejected for the whole batch at the boundary, so use a
    # finite series that fails INSIDE the worker (non-finite Kalman stats).
    bad = np.full(80, 1e300)
    out = rustima.sarimax_batch_fit([good, bad, good], (1, 0, 0), (0, 0, 0, 0))
    assert len(out) == 3
    assert "error" in out[1]
    assert "loglike" in out[0] and "loglike" in out[2]


def test_masked_array_rejected():
    """W1: np.asarray silently drops the mask — reject masked input."""
    y = np.ma.masked_array(_series(4, 60), mask=[True] * 5 + [False] * 55)
    with pytest.raises(ValueError, match="masked"):
        rustima.SARIMAXModel(y, order=(1, 0, 0))
    with pytest.raises(ValueError, match="masked"):
        rustima.auto_arima(y, max_p=1, max_q=1)


def test_inference_rejects_wrong_length_params():
    """W2: empty/wrong-length params must raise, not return empty arrays."""
    y = _series(5)
    with pytest.raises(ValueError, match="[Pp]aram"):
        rustima.sarimax_inference(y, (1, 0, 0), (0, 0, 0, 0), np.array([]))
