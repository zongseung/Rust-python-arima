# Supported ARIMA/SARIMA Combinations

## Parameter Ranges

| Parameter | Valid Range | Notes |
|-----------|------------|-------|
| `p` (AR order) | 0, 1, 2, ... | Tested up to 5 |
| `d` (differencing) | 0, 1, 2 | Tested up to 2 |
| `q` (MA order) | 0, 1, 2, ... | Tested up to 5 |
| `P` (seasonal AR) | 0, 1, 2 | Tested up to 2 |
| `D` (seasonal diff) | 0, 1 | D > 0 requires s >= 2 |
| `Q` (seasonal MA) | 0, 1, 2 | Tested up to 2 |
| `s` (seasonal period) | 0, 2, 3, ... | 0 = no seasonality; tested 2, 4, 7, 12, 24 |

## Tier A: PR Gate Models (30 models, <60s)

These are tested on every commit/PR.

### Non-seasonal (15 models)

| Model | Order | Seasonal |
|-------|-------|----------|
| Pure AR(1) | (1,0,0) | (0,0,0,0) |
| Pure AR(2) | (2,0,0) | (0,0,0,0) |
| Pure AR(3) | (3,0,0) | (0,0,0,0) |
| Pure MA(1) | (0,0,1) | (0,0,0,0) |
| Pure MA(2) | (0,0,2) | (0,0,0,0) |
| Pure MA(3) | (0,0,3) | (0,0,0,0) |
| ARMA(1,1) | (1,0,1) | (0,0,0,0) |
| ARMA(2,1) | (2,0,1) | (0,0,0,0) |
| ARMA(1,2) | (1,0,2) | (0,0,0,0) |
| ARMA(2,2) | (2,0,2) | (0,0,0,0) |
| ARIMA(1,1,0) | (1,1,0) | (0,0,0,0) |
| ARIMA(0,1,1) | (0,1,1) | (0,0,0,0) |
| ARIMA(1,1,1) | (1,1,1) | (0,0,0,0) |
| ARIMA(2,1,1) | (2,1,1) | (0,0,0,0) |
| ARIMA(1,1,2) | (1,1,2) | (0,0,0,0) |

### Seasonal (13 models)

| Model | Order | Seasonal |
|-------|-------|----------|
| SARIMA s=4 D=0 | (1,0,0) | (1,0,0,4) |
| SARIMA s=4 D=0 | (1,0,1) | (1,0,1,4) |
| SARIMA s=4 D=0 | (0,0,1) | (0,0,1,4) |
| SARIMA s=4 D=1 | (1,1,0) | (1,1,0,4) |
| SARIMA s=4 D=1 | (0,1,1) | (0,1,1,4) |
| SARIMA s=4 D=1 | (1,1,1) | (1,1,1,4) |
| SARIMA s=12 | (1,0,0) | (1,0,0,12) |
| SARIMA s=12 | (1,1,0) | (1,1,0,12) |
| SARIMA s=12 | (0,1,1) | (0,1,1,12) |
| SARIMA s=12 | (1,1,1) | (1,1,1,12) |
| SARIMA s=7 | (1,0,0) | (1,0,0,7) |
| SARIMA s=7 | (1,1,1) | (1,1,1,7) |

### Other (2 models)

| Model | Order | Seasonal |
|-------|-------|----------|
| High-order ARMA | (3,0,3) | (0,0,0,0) |
| High-order ARIMA | (3,1,3) | (0,0,0,0) |
| ARIMA d=2 | (1,2,1) | (0,0,0,0) |

## Tier B: Nightly Models (~70 models, <10min)

Tier B includes all Tier A models plus:
- Higher order: AR(4), AR(5), MA(4), MA(5)
- Mixed high: ARMA(3,1), ARMA(1,3), ARMA(3,2), ARMA(2,3)
- d=2: ARIMA(1,2,1), ARIMA(2,2,2)
- Seasonal s=2: two models
- Seasonal s=4 higher P/Q: P=2, Q=2
- Seasonal s=12 extended: four additional models
- Seasonal s=24 (hourly): two models
- Short series (n=50, n=80, n=100): four models
- Long series (n=1000): three models
- d+D combinations: three models
- Seasonal s=7 extended: three additional models
- With exogenous variables: five models (n_exog=1,2,3)

## Known Limitations

### Numerical Sensitivity

The following combinations may show increased numerical sensitivity:
- Very high order (p+q > 6): larger state space, slower convergence
- Near-unit-root data with high differencing (d=2, D=1)
- s=24 with seasonal differencing: very large state space (k_states > 25)
- Short series (n < 3 * k_states): insufficient data for reliable estimation

### Minimum Data Length

Recommended minimum observations:
- Non-seasonal: `n >= 2 * (p + q) + 20`
- Seasonal: `n >= 2 * s + 2 * (p + q + P + Q) + 20`

### Unsupported Combinations

| Combination | Reason |
|-------------|--------|
| (0,0,0)(0,0,0,0) | Zero-parameter model — no optimization needed |
| D > 1 | Not supported |
| s = 1 | Invalid seasonal period |
