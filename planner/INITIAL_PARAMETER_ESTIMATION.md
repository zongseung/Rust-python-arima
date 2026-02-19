# Initial Parameter Estimation Methods for ARIMA/SARIMAX

## Research Summary for Rust Implementation

This document covers the mathematical formulations, algorithms, and implementation
details needed to compute good starting parameter values for SARIMAX model optimization.
All information is derived from direct analysis of the statsmodels source code and
classical time series references (primarily Brockwell & Davis 2016).

------------------------------------------------------------------------

# 1. Conditional Sum of Squares (CSS)

## 1.1 Mathematical Formulation

For an ARIMA(p,d,q) model, after differencing to remove integration:

```
y_t = phi_1 * y_{t-1} + ... + phi_p * y_{t-p} + e_t + theta_1 * e_{t-1} + ... + theta_q * e_{t-q}
```

The **Conditional Sum of Squares** objective is:

```
S(phi, theta) = sum_{t=r+1}^{n} e_t^2
```

where `r = max(p, 2q)` (statsmodels uses `r = max(k + k_ma, k_ar)` where `k = 2 * k_ma`),
and the residuals `e_t` are computed recursively by conditioning on `e_t = 0` for `t <= 0`:

```
e_t = y_t - phi_1 * y_{t-1} - ... - phi_p * y_{t-p} - theta_1 * e_{t-1} - ... - theta_q * e_{t-q}
```

## 1.2 statsmodels Implementation: `_conditional_sum_squares`

The statsmodels approach does NOT iteratively minimize the CSS objective. Instead, it uses
a **two-stage OLS regression** to get initial parameter estimates:

### Stage 1: Estimate residuals from a long AR model

If there are MA terms (`k_ma > 0`):

```
k = 2 * k_ma
r = max(k + k_ma, k_ar)

# Fit AR(k) model via OLS to get residual proxies
Y = endog[k:]                        # dependent variable
X = lagmat(endog, k, trim='both')    # lagged endog matrix, k columns
params_ar = pinv(X) . Y
residuals = Y - X . params_ar
```

### Stage 2: Estimate ARMA parameters via OLS

```
Y = endog[r:]
X = []

# Add trend data columns if present
if k_trend > 0:
    X = [X, trend_data[:(-r), :]]

# Add AR lag columns (only columns corresponding to non-zero polynomial terms)
if k_ar > 0:
    cols = nonzero_indices(polynomial_ar)[1:] - 1   # skip the leading 1
    X = [X, lagmat(endog, k_ar)[r:, cols]]

# Add MA lag columns (using residuals from Stage 1)
if k_ma > 0:
    cols = nonzero_indices(polynomial_ma)[1:] - 1
    X = [X, lagmat(residuals, k_ma)[r-k:, cols]]

# Solve via OLS
params = pinv(X) . Y
residuals = Y - X . params
```

### Stage 3: Extract parameters and variance

```
# params = [trend_params, ar_params, ma_params]  (in order)
sigma2 = mean(residuals[k_params_ma:]^2)
```

### Pseudocode

```
FUNCTION css_start_params(endog, k_ar, polynomial_ar, k_ma, polynomial_ma, k_trend, trend_data):
    k = 2 * k_ma
    r = max(k + k_ma, k_ar)
    k_params_ar = count_nonzero(polynomial_ar) - 1  // exclude leading 1
    k_params_ma = count_nonzero(polynomial_ma) - 1

    IF k_ar + k_ma + k_trend == 0:
        RETURN ([], [], [], [])

    // Stage 1: get residual estimates for MA terms
    IF k_ma > 0:
        Y = endog[k:]
        X = lag_matrix(endog, k)  // n-k rows, k columns
        ar_temp = least_squares(X, Y)
        residuals = Y - X * ar_temp

    // Stage 2: joint ARMA estimation
    Y = endog[r:]
    X = empty_matrix(len(Y), 0)

    IF k_trend > 0:
        X = concat_columns(X, trend_data[0..len(Y)])
    IF k_ar > 0:
        ar_cols = get_nonzero_lag_indices(polynomial_ar)
        X = concat_columns(X, lag_matrix(endog, k_ar)[r:, ar_cols])
    IF k_ma > 0:
        ma_cols = get_nonzero_lag_indices(polynomial_ma)
        X = concat_columns(X, lag_matrix(residuals, k_ma)[r-k:, ma_cols])

    params = least_squares(X, Y)    // via pseudo-inverse
    residuals = Y - X * params

    // Stage 3: unpack
    offset = 0
    trend_params = params[offset..offset+k_trend]; offset += k_trend
    ar_params    = params[offset..offset+k_params_ar]; offset += k_params_ar
    ma_params    = params[offset..offset+k_params_ma]; offset += k_params_ma
    variance     = mean(residuals[k_params_ma:]^2)

    RETURN (trend_params, ar_params, ma_params, variance)
```

## 1.3 Extension to Seasonal ARIMA

statsmodels runs CSS **twice**:

1. First call: estimates non-seasonal ARMA(p,q) parameters + trend
2. Second call: estimates seasonal ARMA(P,Q) parameters using the same `endog`
   (with seasonal polynomial structure)

The seasonal polynomial has non-consecutive lags. For example, with `s=12` and `P=1`,
the polynomial is `1 - Phi_1 * L^12`. The `polynomial_seasonal_ar` array has nonzero
entries only at positions 0 and 12.

The key insight: the `cols = polynomial.nonzero()[0][1:] - 1` indexing automatically
handles non-consecutive lags by selecting only the columns in the lag matrix that
correspond to active lag positions.

## 1.4 Edge Cases

- **Too few observations**: If `lagmat` fails due to insufficient data, all parameters
  default to zero and variance is set to `var(endog - mean(endog))` or 1.0 if endog is empty.
- **Variance floor**: The final variance is bounded below by `1e-10`:
  `params_variance = max(params_variance, 1e-10)`
- **No state error**: If there are no ARMA terms at all, the seasonal variance may be
  substituted, or `inner(endog, endog) / nobs` is used.
- **Missing values**: CSS cannot handle NaN; they are removed before estimation.

------------------------------------------------------------------------

# 2. Hannan-Rissanen Method

## 2.1 Algorithm Overview

The Hannan-Rissanen (HR) procedure estimates ARMA(p,q) parameters in three steps.
Reference: Brockwell & Davis (2016), Section 5.1.4.

**Assumption**: The input series is stationary (differencing should be applied first).

## 2.2 Step-by-Step Algorithm

### Step 1: Fit a long AR model via Yule-Walker to get residual estimates

```
// Choose initial AR order
IF initial_ar_order not specified:
    initial_ar_order = max(floor(ln(n)^2), 2 * max(p, q))

// Estimate AR(initial_ar_order) via Yule-Walker
ar_params_long, _ = yule_walker(endog, order=initial_ar_order, method='mle')

// Compute residuals
X = lag_matrix(endog, initial_ar_order)
Y = endog[initial_ar_order:]
residuals = Y - X . ar_params_long
```

### Step 2: Estimate ARMA parameters via OLS regression

```
// Build lag matrices
lagged_endog = lag_matrix(endog, max_ar_order)
lagged_resid = lag_matrix(residuals, max_ma_order)

// Align indices
ix = initial_ar_order + max_ma_order - max_ar_order

X = concat_columns(
    lagged_endog[ix:, ar_lag_indices],     // AR regressors
    lagged_resid[:, ma_lag_indices]          // MA regressors (using residuals as proxy)
)
Y = endog[initial_ar_order + max_ma_order:]

// OLS estimation
result = OLS(Y, X).fit()
ar_params = result.params[0:k_ar]
ma_params = result.params[k_ar:]
sigma2    = result.scale    // = sum(resid^2) / (n - k)
```

### Step 3: Bias correction (optional, applied if stationary and invertible)

This step improves the estimates by correcting for finite-sample bias:

```
IF is_stationary(ar_params) AND is_invertible(ma_params):
    Z = zeros(n)

    // Compute filtered series Z
    FOR t = max(p, q) TO n-1:
        tmp_ar = dot(-ar_poly[1:], endog[t-p:t][::-1])
        tmp_ma = dot(ma_poly[1:], Z[t-q:t][::-1])
        Z[t] = endog[t] - tmp_ar - tmp_ma

    // Compute auxiliary series V and W using linear filtering
    V = lfilter([1], ar_poly, Z)              // Filter Z through 1/AR(z)
    W = lfilter([1, -ma_poly[1:]], [1], Z)    // Filter Z through MA(z)^{-1}... approx

    // Build regression for bias correction
    lagged_V = lag_matrix(V, max_ar_order)
    lagged_W = lag_matrix(W, max_ma_order)
    exog_bias = concat_columns(lagged_V[align:, ar_ix], lagged_W[align:, ma_ix])

    // OLS for bias correction
    result_bias = OLS(Z[max(p,q):], exog_bias).fit()

    // Update parameters (additive correction)
    ar_params = ar_params + result_bias.params[0:k_ar]
    ma_params = ma_params + result_bias.params[k_ar:]

    // Recompute sigma2 with corrected parameters
    resid = original_Y - original_X . concat(ar_params, ma_params)
    sigma2 = inner(resid, resid) / len(resid)
```

## 2.3 Extension to Seasonal Models

The `innovations_mle` function in statsmodels extends HR to seasonal models by a
two-pass approach:

1. **First pass**: Apply HR to estimate non-seasonal ARMA(p,q) parameters, obtain residuals
2. **Second pass**: Apply HR to those residuals with seasonal lag structure

```
// Step 1: Non-seasonal HR
hr_params, hr_results = hannan_rissanen(endog, ar_order=p, ma_order=q)

// Step 2: Seasonal HR on residuals
seasonal_ar_order = [P*s]    // e.g., for P=1, s=12: ar at lag 12
seasonal_ma_order = [Q*s]    // e.g., for Q=1, s=12: ma at lag 12
seasonal_hr, _ = hannan_rissanen(hr_results.resid,
                                  ar_order=seasonal_ar_order,
                                  ma_order=seasonal_ma_order)

// Combine
sp.ar_params = hr_params.ar_params
sp.ma_params = hr_params.ma_params
sp.seasonal_ar_params = seasonal_hr.ar_params
sp.seasonal_ma_params = seasonal_hr.ma_params
sp.sigma2 = seasonal_hr.sigma2
```

**Important fallback**: If the estimated parameters are non-stationary or non-invertible,
they are reset to zeros:
```
IF NOT is_stationary(sp):
    sp.ar_params = [0] * k_ar
    sp.seasonal_ar_params = [0] * k_seasonal_ar

IF NOT is_invertible(sp) AND enforce_invertibility:
    sp.ma_params = [0] * k_ma
    sp.seasonal_ma_params = [0] * k_seasonal_ma
```

## 2.4 Edge Cases for Hannan-Rissanen

- **Pure AR model** (q=0): Reduces to direct OLS regression of y_t on lagged y values
- **Pure MA model** (p=0): Still needs the initial long AR fit for residual estimation
- **No AR or MA**: Just compute variance of the series
- **Non-consecutive lags**: Supported via lag index arrays (e.g., AR lags at [1, 3] only)
- **Non-stationary/non-invertible result**: Skip bias correction step, fall back to zeros

------------------------------------------------------------------------

# 3. Yule-Walker Equations

## 3.1 Mathematical Formulation

For a stationary AR(p) process:
```
y_t = phi_1 * y_{t-1} + phi_2 * y_{t-2} + ... + phi_p * y_{t-p} + e_t
```

The Yule-Walker equations relate autocovariances to AR parameters:

```
gamma(k) = phi_1 * gamma(k-1) + phi_2 * gamma(k-2) + ... + phi_p * gamma(k-p)    for k >= 1
```

In matrix form:
```
[gamma(0)   gamma(1)   ... gamma(p-1)] [phi_1]   [gamma(1)]
[gamma(1)   gamma(0)   ... gamma(p-2)] [phi_2] = [gamma(2)]
[...        ...        ... ...       ] [...]     [...]
[gamma(p-1) gamma(p-2) ... gamma(0)  ] [phi_p]   [gamma(p)]

R * phi = r
```

where R is a symmetric Toeplitz matrix of autocovariances.

## 3.2 Autocovariance Estimation

Two methods for estimating gamma(k):

**MLE method** (biased, default in statsmodels for Yule-Walker used by HR):
```
gamma_hat(k) = (1/n) * sum_{t=k+1}^{n} (y_t - y_bar)(y_{t-k} - y_bar)
```

**Adjusted method** (unbiased):
```
gamma_hat(k) = (1/(n-k)) * sum_{t=k+1}^{n} (y_t - y_bar)(y_{t-k} - y_bar)
```

## 3.3 Solution Algorithm

```
FUNCTION yule_walker(x, order_p, method='mle'):
    x = x - mean(x)        // demean
    n = len(x)

    // Compute autocovariances
    r = zeros(order_p + 1)
    r[0] = sum(x^2) / n
    FOR k = 1 TO order_p:
        IF method == 'mle':
            r[k] = sum(x[0:n-k] * x[k:n]) / n
        ELSE:  // adjusted
            r[k] = sum(x[0:n-k] * x[k:n]) / (n - k)

    // Form Toeplitz matrix and solve
    R = toeplitz(r[0:order_p])    // p x p symmetric Toeplitz matrix
    phi = solve(R, r[1:order_p+1])

    // If R is singular, use pseudo-inverse
    IF singular_error:
        phi = pinv(R) . r[1:order_p+1]

    // Estimate innovation variance
    sigma_sq = r[0] - dot(r[1:], phi)
    sigma = sqrt(max(sigma_sq, 0))    // handle numerical issues

    RETURN phi, sigma
```

## 3.4 Computational Notes for Rust

- The Toeplitz structure can be exploited via Levinson-Durbin recursion (O(p^2)
  instead of O(p^3) for general matrix solve)
- For the initial AR order in HR, `p_init = max(floor(ln(n)^2), 2*max(p,q))`,
  this can be large, making Levinson-Durbin worthwhile
- The Toeplitz matrix is always positive semi-definite when using MLE autocovariances
- With adjusted autocovariances, the matrix may not be positive definite -- handle
  this with a fallback to pseudo-inverse

------------------------------------------------------------------------

# 4. Burg's Method

## 4.1 Algorithm

Burg's method estimates AR parameters by minimizing the sum of forward and backward
prediction errors. It produces partial autocorrelation coefficients (PACF) which are
then converted to AR parameters via Levinson-Durbin recursion.

```
FUNCTION burg(endog, order_p):
    x = endog - mean(endog)    // demean
    n = len(x)

    // Compute PACF via Burg's algorithm (minimizes forward + backward errors)
    // Then convert PACF to AR params via levinson_durbin_pacf

    pacf, sigma_sequence = pacf_burg(x, order_p)
    ar_params, _ = levinson_durbin_pacf(pacf)

    sigma2 = sigma_sequence[order_p]    // final prediction error variance
    RETURN ar_params, sigma2
```

### Levinson-Durbin from PACF

Given partial autocorrelations `alpha_1, alpha_2, ..., alpha_p`:

```
FUNCTION levinson_durbin_pacf(pacf):
    p = len(pacf) - 1    // pacf[0] is unused or 1
    phi = zeros(p, p)

    phi[0, 0] = pacf[1]
    FOR k = 1 TO p-1:
        phi[k, k] = pacf[k+1]
        FOR j = 0 TO k-1:
            phi[k, j] = phi[k-1, j] - pacf[k+1] * phi[k-1, k-1-j]

    ar_params = phi[p-1, 0:p]
    RETURN ar_params
```

## 4.2 When to Use

- Burg's method is preferred over Yule-Walker when the series is short
- Only works for consecutive AR orders (no seasonal or gap lags)
- Produces estimates that are always stationary (all partial autocorrelations < 1)

------------------------------------------------------------------------

# 5. OLS-based Initialization

## 5.1 For AR Parameters

Simple OLS regression of current value on lagged values:

```
Y = [y_{p+1}, y_{p+2}, ..., y_n]^T
X = [y_p    y_{p-1}   ... y_1  ]
    [y_{p+1} y_p      ... y_2  ]
    [...     ...       ... ...  ]
    [y_{n-1} y_{n-2}   ... y_{n-p}]

phi_hat = (X^T X)^{-1} X^T Y
```

Or equivalently via pseudo-inverse: `phi_hat = pinv(X) . Y`

## 5.2 For Exogenous Coefficients

```
beta_hat = pinv(exog) . endog
endog_adjusted = endog - exog . beta_hat
```

This is exactly what statsmodels does in `start_params`:

```python
if self._k_exog > 0:
    params_exog = np.linalg.pinv(exog).dot(endog)
    endog = endog - np.dot(exog, params_exog)
```

The exogenous effects are removed FIRST, before estimating ARMA parameters.

## 5.3 For sigma2 (Innovation Variance)

```
sigma2_hat = (1/n_resid) * sum(residuals^2)
```

where residuals come from the CSS regression. Note that statsmodels skips
the first `k_params_ma` residuals:

```
sigma2 = mean(residuals[k_params_ma:]^2)
```

------------------------------------------------------------------------

# 6. statsmodels Default `start_params` -- Complete Flow

## 6.1 The Full Algorithm

The `start_params` property in `statsmodels.tsa.statespace.sarimax.SARIMAX` follows
this exact sequence:

```
FUNCTION start_params(model):
    endog = model.endog
    exog  = model.exog

    // Step 0: Apply differencing if not using simple_differencing
    IF NOT simple_differencing AND (d > 0 OR D > 0):
        endog = diff(endog, d, D, s)
        exog  = diff(exog, d, D, s) if exog is not None
        trend_data = trend_data[0:len(endog)]

    // Step 0.5: Remove missing values (CSS cannot handle NaN)
    IF any(isnan(endog)):
        mask = NOT isnan(endog)
        endog = endog[mask]
        exog = exog[mask]
        trend_data = trend_data[mask]

    // Step 1: Remove exogenous effects via OLS
    params_exog = []
    IF k_exog > 0:
        params_exog = pinv(exog) . endog
        endog = endog - exog . params_exog
    IF state_regression:  // exog in state vector, not estimated via MLE
        params_exog = []

    // Step 2: Estimate non-seasonal ARMA + trend via CSS
    (params_trend, params_ar, params_ma, params_variance) =
        CSS(endog, k_ar, polynomial_ar, k_ma, polynomial_ma, k_trend, trend_data)

    // Step 3: Validate stationarity/invertibility
    IF k_ar > 0 AND enforce_stationarity AND NOT is_invertible([1, -params_ar]):
        WARN "Non-stationary AR starting parameters, using zeros"
        params_ar = zeros

    IF k_ma > 0 AND enforce_invertibility AND NOT is_invertible([1, params_ma]):
        WARN "Non-invertible MA starting parameters, using zeros"
        params_ma = zeros

    // Step 4: Estimate seasonal ARMA via CSS (separate call)
    (_, params_seasonal_ar, params_seasonal_ma, params_seasonal_variance) =
        CSS(endog, k_seasonal_ar, polynomial_seasonal_ar,
            k_seasonal_ma, polynomial_seasonal_ma)

    // Step 5: Validate seasonal stationarity/invertibility
    IF k_seasonal_ar > 0 AND enforce_stationarity AND NOT is_invertible([1, -params_seasonal_ar]):
        WARN; params_seasonal_ar = zeros

    IF k_seasonal_ma > 0 AND enforce_invertibility AND NOT is_invertible([1, params_seasonal_ma]):
        WARN; params_seasonal_ma = zeros

    // Step 6: Handle variance
    IF state_error AND params_variance is empty:
        IF params_seasonal_variance is not empty:
            params_variance = params_seasonal_variance
        ELIF k_exog > 0:
            params_variance = inner(endog, endog)
        ELSE:
            params_variance = inner(endog, endog) / nobs

    params_variance = max(params_variance, 1e-10)    // floor at 1e-10

    IF concentrate_scale:
        params_variance = []    // remove from parameter vector

    // Step 7: Assemble final parameter vector
    RETURN concat(
        params_trend,
        params_exog,
        params_ar,
        params_ma,
        params_seasonal_ar,
        params_seasonal_ma,
        params_exog_variance,       // [1]*k_exog if time_varying, else []
        params_measurement_variance, // 1 if measurement_error, else []
        params_variance             // sigma2
    )
```

## 6.2 Summary

**statsmodels uses CSS (two-stage OLS) for start_params**, NOT Hannan-Rissanen.
Hannan-Rissanen is used in the `innovations_mle` estimator (a separate estimation
method), not in the main SARIMAX state-space model's default initialization.

------------------------------------------------------------------------

# 7. Parameter Bounds and Constraints

## 7.1 Overview of the Constraint Strategy

statsmodels uses **parameter transformations** rather than box bounds during optimization.
The optimizer works in an **unconstrained** space, and a bijective transformation maps
unconstrained parameters to the constrained space for likelihood evaluation.

```
Optimizer (L-BFGS, etc.)
    |
    v
unconstrained_params  (R^k, no bounds)
    |
    v  transform_params()
constrained_params    (stationary AR, invertible MA, sigma2 > 0)
    |
    v
likelihood evaluation (Kalman filter)
```

## 7.2 Stationarity Constraint on AR Polynomials (Monahan 1984)

**Reference**: Monahan, John F. 1984. "A Note on Enforcing Stationarity in
Autoregressive-moving Average Models." Biometrika 71(2): 403-404.

### Forward transform: unconstrained -> stationary AR coefficients

```
FUNCTION constrain_stationary_univariate(unconstrained):
    // unconstrained: array of length p (real-valued, no bounds)
    n = len(unconstrained)
    y = zeros(n, n)

    // Map unconstrained to partial autocorrelations in (-1, 1)
    r = unconstrained / sqrt(1 + unconstrained^2)    // element-wise

    // Levinson-Durbin recursion to get AR coefficients from partial autocorrelations
    FOR k = 0 TO n-1:
        FOR i = 0 TO k-1:
            y[k, i] = y[k-1, i] + r[k] * y[k-1, k-i-1]
        y[k, k] = r[k]

    RETURN -y[n-1, :]    // final row, negated
```

**Key insight**: The mapping `x -> x / sqrt(1 + x^2)` maps R -> (-1, 1), ensuring
all partial autocorrelations are strictly within the unit interval. The Levinson-Durbin
recursion then guarantees the resulting AR polynomial is stationary.

### Inverse transform: stationary AR coefficients -> unconstrained

```
FUNCTION unconstrain_stationary_univariate(constrained):
    n = len(constrained)
    y = zeros(n, n)
    y[n-1, :] = -constrained    // load the coefficients (negated)

    // Reverse Levinson-Durbin to recover partial autocorrelations
    FOR k = n-1 DOWNTO 1:
        FOR i = 0 TO k-1:
            y[k-1, i] = (y[k, i] - y[k, k] * y[k, k-i-1]) / (1 - y[k, k]^2)

    r = diagonal(y)    // partial autocorrelations

    // Inverse of the sigmoid: (-1,1) -> R
    x = r / sqrt(1 - r^2)

    RETURN x
```

### Numerical edge case
When `|y[k, k]| >= 1` (partial autocorrelation at or beyond boundary), the division
`1 - y[k,k]^2` becomes zero or negative. This can happen if the input AR coefficients
are exactly on the stationarity boundary. In practice, this should be handled by
clamping or returning a large value.

## 7.3 Invertibility Constraint on MA Polynomials

The MA invertibility constraint is enforced **identically** to the AR stationarity
constraint, but with a sign convention difference:

```
// Transform: unconstrained -> invertible MA
constrained_ma = -constrain_stationary_univariate(unconstrained_ma)
//              ^^^ note the negation

// Inverse: invertible MA -> unconstrained
unconstrained_ma = unconstrain_stationary_univariate(-constrained_ma)
//                                                   ^^^ negate first
```

The negation comes from the sign convention difference between AR and MA polynomials:
- AR: `(1 - phi_1 L - phi_2 L^2 - ...) y_t = e_t`  (roots outside unit circle)
- MA: `y_t = (1 + theta_1 L + theta_2 L^2 + ...) e_t`  (roots outside unit circle)

## 7.4 sigma2 Constraint (Positive Variance)

```
// Transform: unconstrained -> positive
sigma2_constrained = unconstrained_sigma2^2

// Inverse: positive -> unconstrained
unconstrained_sigma2 = sqrt(sigma2_constrained)
```

## 7.5 Stationarity/Invertibility Check

To check if a polynomial `c(L) = 1 + c_1 L + ... + c_p L^p` is invertible:

```
FUNCTION is_invertible(polynomial, threshold=1-1e-10):
    // Form companion matrix from polynomial coefficients
    C = companion_matrix(polynomial)
    eigenvalues = eigenvalues(C)
    RETURN all(|eigenvalues| < threshold)
```

The companion matrix for polynomial `[c_0, c_1, ..., c_p]` is:

```
    [-c_1/c_0   1   0  ...  0]
    [-c_2/c_0   0   1  ...  0]
    [  ...      .   .   .   .]
    [-c_{p-1}/c_0 0  0  ... 1]
    [-c_p/c_0   0   0  ...  0]
```

The eigenvalues of this matrix are the roots of the polynomial, and all must have
modulus strictly less than 1 for stationarity/invertibility.

## 7.6 Summary of Transform Ordering

The `transform_params` function applies transformations in parameter-vector order:

```
[trend_params]                -> pass through (no constraint)
[exog_params]                 -> pass through (if mle_regression)
[ar_params]                   -> constrain_stationary (if enforce_stationarity)
[ma_params]                   -> -constrain_stationary (if enforce_invertibility)
[seasonal_ar_params]          -> constrain_stationary (if enforce_stationarity)
[seasonal_ma_params]          -> -constrain_stationary (if enforce_invertibility)
[exog_variance]               -> x^2 (if time_varying_regression)
[measurement_variance]        -> x^2 (if measurement_error)
[state_variance (sigma2)]     -> x^2 (if state_error and not concentrate_scale)
```

## 7.7 The `enforce_stationarity=False` / `enforce_invertibility=False` Case

When these flags are False:
- AR/MA parameters are passed through without transformation
- The optimizer has no stationarity/invertibility guarantee
- Parameters may converge to non-stationary/non-invertible values
- This is faster but may produce invalid models
- For Rust implementation with `enforce=False`, the optimizer operates directly
  on the parameter space with no bounds

------------------------------------------------------------------------

# 8. Innovations Algorithm (for completeness)

## 8.1 Algorithm

The innovations algorithm computes MA(q) coefficients from autocovariances.
Reference: Brockwell & Davis (2016), Section 5.1.3.

Given autocovariances `gamma(0), gamma(1), ..., gamma(q)`:

```
FUNCTION innovations_algo(acovf, nobs):
    // acovf[k] = gamma(k)
    theta = zeros(nobs, nobs)
    v = zeros(nobs)

    v[0] = acovf[0]

    FOR n = 1 TO nobs-1:
        FOR k = 0 TO n-1:
            sum_term = 0
            FOR j = 0 TO k-1:
                sum_term += theta[k, k-1-j] * theta[n, n-1-j] * v[j]
            theta[n, n-1-k] = (acovf[n-k] - sum_term) / v[k]

        v[n] = acovf[0]
        FOR j = 0 TO n-1:
            v[n] -= theta[n, n-1-j]^2 * v[j]

    RETURN theta, v
```

The MA(q) estimates are: `ma_params = theta[q, 0:q]`
The innovation variance is: `sigma2 = v[q]`

------------------------------------------------------------------------

# 9. Implementation Recommendations for Rust

## 9.1 Recommended Default Start Parameter Method

Follow the statsmodels SARIMAX approach:

1. Apply differencing to remove integration
2. Remove exogenous effects via OLS (pseudo-inverse)
3. Estimate non-seasonal ARMA via CSS (two-stage OLS)
4. Estimate seasonal ARMA via CSS (two-stage OLS on same endog)
5. Validate stationarity/invertibility; fall back to zeros
6. Floor variance at 1e-10
7. Assemble parameter vector in correct order

## 9.2 Key Rust Implementation Components Needed

### Linear algebra operations:
- `lag_matrix(x, max_lag)`: Creates an (n-max_lag) x max_lag matrix of lagged values
- `pseudo_inverse(X)`: Moore-Penrose pseudo-inverse (via SVD or QR)
- `toeplitz(r)`: Symmetric Toeplitz matrix from first row/column
- `solve(A, b)`: Solve linear system (use Cholesky for Toeplitz)
- `eigenvalues(A)`: For stationarity check (companion matrix)

### Numerical operations:
- `constrain_stationary_univariate(x)`: Monahan (1984) transform
- `unconstrain_stationary_univariate(x)`: Inverse Monahan transform
- `is_invertible(polynomial)`: Companion matrix eigenvalue check

### Suggested Rust crates:
- `nalgebra` or `ndarray` + `ndarray-linalg` for linear algebra
- SVD decomposition for pseudo-inverse
- Eigenvalue decomposition for companion matrix

## 9.3 Edge Cases Checklist

1. **Empty or very short series**: Return all-zero parameters, variance = 1.0
2. **Singular lag matrix**: Use pseudo-inverse (SVD-based) instead of direct solve
3. **Non-stationary CSS estimates**: Fall back to zero AR/seasonal AR parameters
4. **Non-invertible CSS estimates**: Fall back to zero MA/seasonal MA parameters
5. **sigma2 <= 0**: Floor at 1e-10
6. **NaN in endog**: Remove before CSS computation
7. **No ARMA terms at all**: sigma2 = var(endog) or inner(endog,endog)/n
8. **Partial autocorrelation at boundary** (|r| = 1): Clamp to `1 - epsilon`
   in unconstrain operation to avoid division by zero
9. **Non-consecutive lag orders** (seasonal): Index into lag matrix using
   nonzero positions of the polynomial array
10. **All parameters fixed**: No optimization needed, return fixed values directly

## 9.4 Precision Targets

- For numerical equivalence with statsmodels, target `|loglike_rust - loglike_python| < 1e-6`
- The Monahan transform is smooth and well-conditioned away from boundary
- Pseudo-inverse should use double precision (f64) throughout
- Autocovariance computation: use the MLE (biased) estimator for consistency

------------------------------------------------------------------------

# 10. Complete Worked Example

## ARIMA(1,1,1) with exogenous regressor

Given: y = [100, 102, 101, 104, 103, 107, ...], exog = [1.0, 1.2, 0.9, ...], order=(1,1,1)

```
Step 1: Difference y
    z = diff(y, 1) = [2, -1, 3, -1, 4, ...]   // length n-1

Step 2: Difference exog
    exog_d = diff(exog, 1) = [0.2, -0.3, ...]

Step 3: Remove exog effect
    beta = pinv(exog_d) . z
    z_adj = z - exog_d * beta

Step 4: CSS for ARMA(1,1)
    k = 2*1 = 2, r = max(2+1, 1) = 3

    // Stage 1: Fit AR(2) to get residuals
    Y = z_adj[2:]
    X = lagmat(z_adj, 2)
    ar_temp = pinv(X) . Y
    resid = Y - X * ar_temp

    // Stage 2: Joint estimation
    Y = z_adj[3:]
    X = [lagmat(z_adj, 1)[3:, 0],    // AR(1) column
         lagmat(resid, 1)[1:, 0]]      // MA(1) column (residual proxy)
    params = pinv(X) . Y
    phi_1 = params[0]
    theta_1 = params[1]
    resid_final = Y - X * params
    sigma2 = mean(resid_final[1:]^2)   // skip first k_params_ma residuals

Step 5: Validate
    IF |root of (1 - phi_1 * z)| < 1:  phi_1 = 0
    IF |root of (1 + theta_1 * z)| < 1: theta_1 = 0
    sigma2 = max(sigma2, 1e-10)

Step 6: Assemble
    start_params = [beta, phi_1, theta_1, sigma2]
```

------------------------------------------------------------------------

# References

1. Brockwell, P.J. and Davis, R.A. (2016). "Introduction to Time Series and Forecasting." Springer.
   - Section 5.1.1: Yule-Walker estimation
   - Section 5.1.2: Burg's method
   - Section 5.1.3: Innovations algorithm
   - Section 5.1.4: Hannan-Rissanen method
   - Section 5.2: Maximum likelihood estimation

2. Monahan, J.F. (1984). "A Note on Enforcing Stationarity in Autoregressive-Moving
   Average Models." Biometrika 71(2): 403-404.
   (Used for the constrain/unconstrain stationarity transforms)

3. Gomez, V. and Maravall, A. (2001). "Automatic Modeling Methods for Univariate Series."
   A Course in Time Series Analysis, 171-201.
   (Used for choosing initial AR order in Hannan-Rissanen)

4. Harvey, A.C. (1989). "Forecasting, Structural Time Series Models and the Kalman Filter."
   Cambridge University Press.

5. statsmodels source code:
   - `statsmodels/tsa/statespace/sarimax.py` (SARIMAX model, start_params, CSS, transforms)
   - `statsmodels/tsa/statespace/tools.py` (constrain/unconstrain, is_invertible, companion_matrix)
   - `statsmodels/tsa/arima/estimators/hannan_rissanen.py` (HR algorithm)
   - `statsmodels/tsa/arima/estimators/yule_walker.py` (YW wrapper)
   - `statsmodels/tsa/arima/estimators/burg.py` (Burg wrapper)
   - `statsmodels/tsa/arima/estimators/innovations.py` (innovations MLE)
   - `statsmodels/regression/linear_model.py` (yule_walker, burg implementations)
   - `statsmodels/tsa/arima/params.py` (is_stationary, is_invertible properties)
