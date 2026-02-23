# Error Codes and Messages

All errors from `sarimax_rs` are raised as Python `ValueError`.

## Error Types

| Error | Message Pattern | Cause | Solution |
|-------|----------------|-------|----------|
| **ParamLengthMismatch** | `parameter length mismatch: expected N, got M` | Wrong number of parameters for model order | Ensure `len(params) == p + q + P + Q + n_exog` |
| **StateSpaceError** | `state space construction failed: ...` | Invalid model specification | Check order/seasonal values; ensure `s >= 2` when `P, D, Q > 0` |
| **CholeskyFailed** | `Cholesky decomposition failed: covariance matrix is not positive-definite` | Numerical issue in Kalman filter | Try different starting params or adjust model order |
| **OptimizationFailed** | `optimization failed: ...` | Optimizer did not converge | Increase `maxiter`, try `method="nelder_mead"`, or adjust starting params |
| **NonStationaryAR** | `non-stationary AR polynomial` | AR roots inside unit circle | Set `enforce_stationarity=False` or reduce AR order |
| **NonInvertibleMA** | `non-invertible MA polynomial` | MA roots inside unit circle | Set `enforce_invertibility=False` or reduce MA order |
| **DataError** | `data error: ...` | Invalid input data | Check for NaN/Inf in data, ensure sufficient length |

## Input Validation Errors

These are raised before model computation begins:

| Error | Message | Cause |
|-------|---------|-------|
| `ValueError` | `exog rows (X) must match y length (Y)` | Exogenous variable row count != data length |
| `ValueError` | `future_exog rows (X) must match steps (Y)` | Future exog row count != forecast steps |
| `ValueError` | `alpha must be in (0, 1), got X` | Invalid confidence level |
| `ValueError` | `D > 0 requires s >= 2` | Seasonal differencing without valid period |
| `ValueError` | `start_params length mismatch` | Wrong number of starting parameters |

## Handling Errors

```python
import sarimax_rs

try:
    result = sarimax_rs.sarimax_fit(y, (1, 0, 0), (0, 0, 0, 0))
except ValueError as e:
    if "parameter length" in str(e):
        # Fix parameter count
        pass
    elif "optimization failed" in str(e):
        # Try different method or starting params
        result = sarimax_rs.sarimax_fit(
            y, (1, 0, 0), (0, 0, 0, 0), method="nelder_mead"
        )
    else:
        raise
```

## Convergence Failures

When `sarimax_fit` returns `{"converged": False}`, the result is still valid but may not represent the global optimum. Common causes:

1. **Insufficient iterations** — Increase `maxiter`
2. **Poor starting params** — Provide better `start_params`
3. **Model overparameterization** — Reduce model order
4. **Near-unit-root data** — Try `enforce_stationarity=False`
