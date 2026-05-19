# Profiled Kalman-GLS Trust-Region Plan

Academic name: **Profiled Kalman-GLS Trust-Region**
(equivalently: *innovations-form profile likelihood with trust-region BFGS*).
API method string (unchanged for code compatibility): `profile-trust-region`.

## Goal

Replace exog-specific post-fit polishing with a defensible profiled likelihood
method for SARIMAX with external regressors.

Current trust-region optimization treats exogenous coefficients as ordinary
unbounded nonlinear optimizer coordinates:

```text
[trend | exog beta | AR | MA | seasonal AR | seasonal MA | sigma2]
```

This can leave weakly identified exog coefficients in a poor basin. The new
method removes only the exog block from the optimizer and solves it exactly at
each likelihood evaluation.

## Proposed Method

Add a new method — *Profiled Kalman-GLS Trust-Region*
(API string `profile-trust-region`):

The nonlinear optimizer sees:

```text
[trend | AR | MA | seasonal AR | seasonal MA | sigma2]
```

For each nonlinear parameter vector theta:

1. Build the SARIMA state-space model with exog beta set to zero.
2. Run the Kalman filter on `y` to obtain innovations `v_y,t` and innovation
   variances `F_t`.
3. Run the same innovation transform on each exog column `x_j` with zero trend
   and zero exog intercept to obtain `v_xj,t`.
4. Solve the innovation-space GLS problem:

   ```text
   beta_hat(theta) =
       argmin_beta sum_t (v_y,t - v_x,t beta)^2 / F_t
   ```

5. Evaluate the Gaussian likelihood at the profiled residual innovations:

   ```text
   r_t = v_y,t - v_x,t beta_hat(theta)
   ```

6. Return the full parameter vector with beta inserted back into the standard
   rustima layout.

## TLKF Gradient

The existing tangent-linear Kalman filter remains useful. After beta is solved
for the current theta, the profile gradient follows the envelope theorem:

```text
d L(theta, beta_hat(theta)) / d theta
  = partial L(theta, beta) / partial theta | beta = beta_hat(theta)
```

Therefore the implementation can:

1. Insert `beta_hat(theta)` into the full parameter vector.
2. Call the existing TLKF `score()` implementation.
3. Apply the existing transform Jacobian.
4. Drop the exog beta entries from the gradient before giving it to the
   trust-region optimizer.

No derivative of `beta_hat(theta)` is required.

## Implementation Scope / Status

This is introduced as an opt-in method so existing optimizers remain unchanged:

- `trust-region` (API string): existing behavior.
- `profile-trust-region` (API string) = *Profiled Kalman-GLS Trust-Region*:
  profiled exog beta via innovation-space GLS + TLKF gradient + trust-region BFGS.

If no exog is present, the Profiled Kalman-GLS Trust-Region method falls back
to the regular trust-region path.

Current implementation details:

- `simple_differencing=true` now applies the same differencing operator to exog
  columns that it applies to endog.
- Exact likelihood initialization now stabilizes the Lyapunov covariance with a
  scale-aware ridge rather than falling back to diffuse initialization because
  of near-singular MA covariance matrices.
- When the profiled method is called without user-provided `start_params`, it
  first gets a nonlinear warm start from the regular trust-region fit (budget
  share: `(maxiter*3)/4`), then runs the profiled objective on the remaining
  budget. User-provided starts bypass the warm-start stage. A failed warm-start
  is silently absorbed and the profiled stage proceeds from the original
  CSS-derived `start`.

## Computational Optimization: Batched Kalman Filtering

### Motivation

Each profiled-likelihood evaluation runs a Kalman filter once on `y` and once
per exog column `x_j` (steps 2–3 of the Proposed Method). With `r` exog
regressors, the per-evaluation cost is therefore

```text
N_KF^PTR_naive = (1 + r) + 1 = r + 2    full Kalman filter passes
```

i.e. `1 + r` filters for the GLS residualisation plus one TLKF pass for the
score. At `r = 2` (the 2019 power-demand SARIMAX case has `hm`, `ta`) the
naive form costs four full Kalman passes per likelihood evaluation, and the
trust-region BFGS routinely requests hundreds of evaluations.

### Key invariance observation

At fixed ψ the state-space matrices `T(ψ)`, `Z`, `R(ψ)`, `Q` are identical
across all `1 + r` filter passes — only the observation sequence changes
(`y` vs each `x_j`). As noted in the linearity argument that justifies the
profiled-β decomposition, the Kalman gain is a function of the state-space
matrices and the innovation variance and does *not* depend on the observed
values. Consequently the predicted state covariance `P_{t|t-1}`, the
innovation variance `F_t`, and the Kalman gain `K_t` are bit-identical
across the `1 + r` passes. Only the predicted state mean
`α^{(s)}_{t|t-1}` and its resulting innovation `v_{s,t}` are series-specific.

### Algorithmic restructure

A batched Kalman filter exploits this invariance by computing the shared
covariance recursion once and the cheap mean recursion once per series. For
each step `t` and `1 + r` parallel series indexed by `s`:

```text
P_{t|t-1} = T P_{t-1|t-1} T^T + R Q R^T              shared, O(k^2)
F_t       = Z P_{t|t-1} Z^T + H_t                    shared, O(k^2)
K_t       = T P_{t|t-1} Z^T / F_t                    shared, O(k^2)
for s in 0..(1+r):
    v_{s,t}            = z_{s,t} - Z α^{(s)}_{t|t-1}     O(k)
    α^{(s)}_{t+1|t}    = T α^{(s)}_{t|t-1} + K_t v_{s,t} O(k)
P_{t|t}   = P_{t|t-1} - K_t F_t K_t^T                shared, O(k^2)
```

The sparse companion structure of `T` is exploited as in the single-series
filter; only the dense covariance algebra changes.

### Cost analysis

Let `c_cov = O(k^2)` denote the per-step cost of the covariance recursion
(prediction + innovation variance + gain + update) and `c_mean = O(k)` the
per-step cost of one mean recursion. For `n` observations:

```text
Naive    : (1 + r) * n * (c_cov + c_mean)
Batched  : n * c_cov + (1 + r) * n * c_mean
```

The covariance-only speedup is `(1 + r) * c_cov / (c_cov + (1 + r) * c_mean)`,
which approaches `1 + r` whenever `c_mean ≪ c_cov`. At `r = 2` and the typical
state dimension `k = 30` for `(3,0,3)(1,1,1)[24]`, the covariance algebra
dominates and the empirical speedup is close to 3× on the GLS pass and ~2×
on total per-evaluation wall time. The current implementation batches the
objective (GLS residualisation) pass but leaves the TLKF score pass as a
single-series filter; the gradient pass would gain a further factor if
batched, but is left as future work.

### Updated per-iteration cost

The naive equation given earlier remains correct as a worst-case bound. With
the batched objective the count refines to:

```text
Naive PTR   : N_KF^PTR_naive   = (1 + r) + 1 = r + 2    full filter passes
Batched PTR : N_KF^PTR_batched = 1 + 1       = 2        full filter passes
              (the (1 + r) mean recursions add O(n k) overhead per series,
               negligible against the dominant O(n k^2) covariance work)
```

### Implementation location

The batched filter lives at `rustima/src/kalman.rs::kalman_filter_batched`
and is invoked from
`rustima/src/optimizer.rs::ProfiledSarimaxObjective::profile_beta_and_loglike`.
The single-series `kalman_filter` is preserved verbatim for callers that still
need it (forecast, residuals, TLKF score). No mutable state is shared between
evaluations — a fresh `StateSpace` is rebuilt at each ψ, as in the naive path.

### Why this is safe

By construction the batched filter reproduces the single-series filter for
each input observation sequence: the `P_{t|t-1}`, `F_t`, `K_t` recursions are
the same operations on the same matrices, and the linearity of the mean
recursion is precisely the property that justifies the PTR β identifiability
(the linearity argument cited above). Unit tests check that the batched
innovations and innovation variances reproduce the per-series
`kalman_filter` outputs elementwise to `1e-10`, so the batched path is a
numerical no-op relative to the naive path within working-precision rounding.

## Validation

Initial checks:

- Synthetic ARX data: profile method recovers a meaningful exog beta and finite
  likelihood.
- Existing exog score test remains valid.
- `cargo check` and focused optimizer tests pass.

Comparison checks:

- Compare the Profiled Kalman-GLS Trust-Region method
  (API string `profile-trust-region`) against R `stats::arima(xreg, method="CSS-ML")`
  and statsmodels `SARIMAX(simple_differencing=True)` separately, because exact
  state-space likelihood and conditional/simple-differenced likelihood are not
  the same objective.

Observed on the 2019 power-demand SARIMAX `(3,0,3)(1,1,1)[24]` case:

- Profiled Kalman-GLS Trust-Region, `simple_differencing=true`:
  `LL=-69618.46`, `AIC=139258.91`, `ta=-117.76`, `hm=+1.47`.
- R `stats::arima(..., method="CSS-ML")` reference (exact MLE final stage):
  `LL=-69427.71`, `AIC=138877.43`, `ta=-122.80`, `hm=+1.72`.
- statsmodels `simple_differencing=True`: `LL=-69443.51`, `AIC=138909.02`,
  `ta=-268.45`, `hm=-5.10`.
- Profiled Kalman-GLS Trust-Region, exact: `LL=-70983.87`, `AIC=141989.73`,
  `ta=-159.16`, `hm=+18.93`.
- statsmodels exact: `LL=-70992.04`, `AIC=142006.07`,
  `ta=-271.24`, `hm=+13.60`.

Note: R `CSS-ML` reports the final-stage *exact* MLE log-likelihood on the
same simple-differenced n = 8736 observations, so it IS directly comparable to
the simple-differenced Python rows; the exact-state-space rows (n = 8760) are
a different objective and must not be compared head-to-head against the
simple-differenced rows. Like-for-like: exact-vs-exact (Rust vs statsmodels)
is within ~8 LL units; simple-diff-vs-R-CSS-ML is the comparison where Rust
agrees with R on the sign of `hm` while statsmodels disagrees.

### Parameter-level comparison

Full parameter table (β_hm, β_ta, ar1..ar3, ma1..ma3, sar1, sma1, σ²) for all
five fits is in [`param_compare_2019.md`](param_compare_2019.md). Key findings:

- **ARMA non-seasonal (ar1, ma1, ma2, ma3)**: all five fits agree within ~0.02.
- **Seasonal (sar1, sma1)**: four of five cluster at (0.246, −0.93); the
  current Rust profile-TR sd run drifts to (0.281, −0.904) and reports
  `converged=false`. Bumping `maxiter` from 200 to 500 produced **bit-identical
  results** (same LL, same params, same 1.3 s runtime), confirming the plateau
  is not iteration-budget-bound — the trust-region terminates by its own
  gradient-tolerance / radius-collapse criterion well before maxiter.
- **σ²**: exact-vs-exact within 0.2% (554,682 vs 555,656); sd group within 5%.
- **Exog β (hm, ta)**: under-identified direction, but a clear two-cluster
  pattern emerges:

  | fit | hm | ta |
  |---|---:|---:|
  | R CSS-ML | +1.72 | −122.80 |
  | Rust profile-TR sd | +1.47 | −117.76 |
  | statsmodels sd | −5.12 | −268.36 |
  | Rust profile-TR exact | +18.93 | −159.16 |
  | statsmodels exact | +13.66 | −271.24 |

  Rust profile-TR sd and R CSS-ML cluster at (hm ≈ +1.5, ta ≈ −120). Both
  statsmodels fits cluster at (ta ≈ −270). For hm/ta specifically, Rust ↔ R
  agreement is the tightest pair (Δhm = 0.25, Δta = 5.0); the next-closest
  pair (Rust exact ↔ sm exact) is an order of magnitude further apart.

- **Exact-vs-exact LL**: Rust profile-TR exact reaches −70,983.87 vs
  statsmodels exact −70,992.03 — Rust higher by 8.17.

### Why does the simple-diff LL gap (~191) persist?

A direct evaluation of Rust's `sarimax_loglike` at statsmodels' converged θ
isolates the cause:

| call | Rust LL | statsmodels LL | Δ |
|---|---:|---:|---:|
| `enforce_stationarity=False, enforce_invertibility=False` | −69159.80 | −69159.80 | **0.0** |
| `enforce_stationarity=True,  enforce_invertibility=True`  | −69692.79 | −69443.49 | 249 |

Under matched `enforce=False` the two engines compute **bit-identical**
Kalman log-likelihoods at the same θ — confirming the profiled GLS objective
and Rust's underlying KF are correct. The 249-unit gap under `enforce=True`
traces to `initialization.rs::KalmanInit::from_config_default`: the two
engines pick different stationary initial state covariances for the
`enforce=True` branch, which changes the early F_t and accumulates over the
n = 8736 observations.

Implication: the simple-diff LL gap is **not** a Profiled Kalman-GLS Trust-
Region bug — it is a documented initialization-scheme divergence between the
two engines in the `enforce=True` path. Para­meter estimates remain close
(ARMA within 0.02; hm/ta closest to R), but the LL scales are not directly
comparable on this branch.

### Paper-direction recommendation

- **Headline comparison**: exact-vs-exact (Rust ↔ statsmodels), Δ LL = 8.17,
  ARMA agreement to 3 decimals. This is a like-for-like, init-matched
  comparison and the strongest validation evidence.
- **Secondary**: ta/hm parity with R CSS-ML on the simple-diff path. Phrase as
  "Rust profile-TR is the only Python-side fit whose exog β reproduces R's
  sign and magnitude," and explicitly disclaim direct LL/AIC comparison
  against statsmodels-sd because of the `enforce=True` init divergence.
- **Caveat to disclose**: Rust profile-TR sd reports `converged=false` and is
  insensitive to `maxiter`; the limiting factor is the trust-region gradient
  tolerance + the init divergence above, not optimizer budget.

Generated by `python_tests/compare_profile_methods_2019.py` (Python: Rust +
statsmodels), `python_tests/compare_profile_methods_R.R` (R), and
`python_tests/diagnose_profile_tr_plateau.py` (start-point sensitivity +
enforce-flag isolation).
