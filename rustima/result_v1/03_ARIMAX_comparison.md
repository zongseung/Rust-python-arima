# ARIMAX Comparison: sarimax_rs vs statsmodels

Generated: 2026-02-25

## Summary

| Metric | Value |
|--------|-------|
| Total models | 15 |
| Valid comparisons | 15 |
| Very close (Δ<0.1) | 0 (0.0%) |
| Close (Δ<1.0) | 0 (0.0%) |
| Medium (1.0≤Δ<5.0) | 13 |
| Far (Δ≥5.0) | 2 |
| Mean Δ(loglike) | 3.8120 |
| Median Δ(loglike) | 3.8191 |
| P95 Δ(loglike) | 6.4346 |
| Max Δ(loglike) | 6.4492 |
| Mean speedup (sm/rs) | 14.0x |
| Mean time (rs) | 4 ms |
| Mean time (sm) | 60 ms |

## Detailed Results

| Model | LL_rs | LL_sm | Δ(LL) | AIC_rs | AIC_sm | rs(ms) | sm(ms) | Speedup |
|-------|-------|-------|-------|--------|--------|--------|--------|--------|
| ARIMAX(0,1,1)+k2 | -763.63 | -761.16 | 2.465 | 1535.25 | 1530.32 | 1 | 20 | 16.7x |
| ARIMAX(0,1,2)+k2 | -717.48 | -713.70 | 3.772 | 1444.95 | 1437.41 | 2 | 24 | 14.7x |
| ARIMAX(0,1,3)+k2 | -709.96 | -705.04 | 4.919 | 1431.91 | 1422.07 | 3 | 49 | 14.2x |
| ARIMAX(1,1,0)+k2 | -809.55 | -808.11 | 1.444 | 1627.11 | 1624.22 | 1 | 11 | 10.2x |
| ARIMAX(1,1,1)+k2 | -719.30 | -716.66 | 2.643 | 1448.61 | 1443.32 | 2 | 26 | 15.7x |
| ARIMAX(1,1,2)+k2 | -712.93 | -709.11 | 3.819 | 1437.86 | 1430.22 | 4 | 36 | 9.8x |
| ARIMAX(1,1,3)+k2 | -709.62 | -704.93 | 4.694 | 1433.24 | 1423.86 | 6 | 77 | 12.3x |
| ARIMAX(2,1,0)+k2 | -718.81 | -716.10 | 2.709 | 1447.62 | 1442.20 | 2 | 19 | 10.3x |
| ARIMAX(2,1,1)+k2 | -706.69 | -704.19 | 2.501 | 1425.38 | 1420.38 | 2 | 31 | 16.2x |
| ARIMAX(2,1,2)+k2 | -705.22 | -701.49 | 3.720 | 1424.43 | 1416.99 | 2 | 49 | 21.0x |
| ARIMAX(2,1,3)+k2 | -705.14 | -698.71 | 6.428 | 1426.28 | 1413.42 | 10 | 70 | 7.0x |
| ARIMAX(3,1,0)+k2 | -710.96 | -707.13 | 3.833 | 1433.92 | 1426.25 | 3 | 26 | 9.3x |
| ARIMAX(3,1,1)+k2 | -705.04 | -701.17 | 3.873 | 1424.08 | 1416.34 | 8 | 45 | 5.9x |
| ARIMAX(3,1,2)+k2 | -705.03 | -701.12 | 3.910 | 1426.07 | 1418.25 | 9 | 90 | 10.2x |
| ARIMAX(3,1,3)+k2 | -705.03 | -698.58 | 6.449 | 1428.06 | 1415.16 | 9 | 331 | 36.4x |

## Models with Large Differences (Δ≥5.0)

- **ARIMAX(3,1,3)+k2**: Δ=6.449 (rs=-705.03, sm=-698.58)
- **ARIMAX(2,1,3)+k2**: Δ=6.428 (rs=-705.14, sm=-698.71)

---
*Comparison: sarimax_rs vs statsmodels.tsa.statespace.SARIMAX*
*Note: statsmodels uses exact diffuse initialization; sarimax_rs uses approximate diffuse (kappa=1e6).*
