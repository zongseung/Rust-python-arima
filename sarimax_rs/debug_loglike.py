"""
Comprehensive diagnostic script for comparing sarimax_rs vs statsmodels
SARIMA(1,1,1)(1,1,1,12) loglikelihood computation.

This script:
1. Generates deterministic data (np.cumsum of standard_normal, seed=42, n=500)
2. Fits SARIMA(1,1,1)(1,1,1,12) with both statsmodels and sarimax_rs
3. Evaluates loglike on BOTH sides using statsmodels fitted params
4. Compares state space matrices (T, Z, R, Q), innovation variances (F_t),
   loglike formula components, and initialization (P_0, loglikelihood_burn)
5. Uses enforce_stationarity=False, enforce_invertibility=False,
   concentrate_scale=True on both sides for clean comparison.
"""

import numpy as np
import sarimax_rs
import statsmodels.api as sm

# ===========================================================================
# 0. Helpers
# ===========================================================================

SEPARATOR = "=" * 80
SUBSEP = "-" * 60


def section(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def subsection(title):
    print(f"\n{SUBSEP}")
    print(f"  {title}")
    print(SUBSEP)


def compare_scalar(name, val_sm, val_rs, tol=1e-6):
    diff = abs(val_sm - val_rs)
    rel_diff = diff / max(abs(val_sm), 1e-15)
    ok = "OK" if diff < tol else "** MISMATCH **"
    print(f"  {name:30s}  sm={val_sm:+20.12f}  rs={val_rs:+20.12f}  "
          f"diff={diff:.2e}  rel={rel_diff:.2e}  [{ok}]")
    return diff < tol


def compare_array(name, arr_sm, arr_rs, tol=1e-6, max_print=10):
    if arr_sm.shape != arr_rs.shape:
        print(f"  {name}: SHAPE MISMATCH sm={arr_sm.shape} rs={arr_rs.shape}")
        return False
    diff = np.abs(arr_sm - arr_rs)
    max_diff = np.max(diff) if diff.size > 0 else 0.0
    ok = "OK" if max_diff < tol else "** MISMATCH **"
    print(f"  {name:30s}  max_diff={max_diff:.2e}  [{ok}]  shape={arr_sm.shape}")
    if max_diff >= tol:
        # Show first few mismatches
        flat_sm = arr_sm.ravel()
        flat_rs = arr_rs.ravel()
        flat_diff = diff.ravel()
        worst_idx = np.argsort(flat_diff)[::-1][:max_print]
        for idx in worst_idx:
            if flat_diff[idx] >= tol:
                coords = np.unravel_index(idx, arr_sm.shape) if arr_sm.ndim > 1 else idx
                print(f"    [{coords}] sm={flat_sm[idx]:+.12f}  rs={flat_rs[idx]:+.12f}  "
                      f"diff={flat_diff[idx]:.2e}")
    return max_diff < tol


# ===========================================================================
# 1. Generate deterministic data
# ===========================================================================

section("1. Data Generation")

np.random.seed(42)
n = 500
y = np.cumsum(np.random.standard_normal(n))
print(f"  n = {n}")
print(f"  y[:5] = {y[:5]}")
print(f"  y.mean() = {y.mean():.6f}, y.std() = {y.std():.6f}")

ORDER = (1, 1, 1)
SEASONAL = (1, 1, 1, 12)

# ===========================================================================
# 2. Fit both models
# ===========================================================================

section("2. Fit models")

# --- statsmodels ---
subsection("2a. statsmodels fit")
sm_model = sm.tsa.SARIMAX(
    y,
    order=ORDER,
    seasonal_order=SEASONAL,
    enforce_stationarity=False,
    enforce_invertibility=False,
    concentrate_scale=True,
)
sm_res = sm_model.fit(disp=False)
sm_params = sm_res.params
print(f"  statsmodels params = {sm_params}")
print(f"  statsmodels loglike = {sm_res.llf:.12f}")
print(f"  statsmodels scale (sigma2) = {sm_res.scale:.12f}")
print(f"  statsmodels nobs_effective = {sm_res.nobs_effective}")
print(f"  param names = {list(sm_res.param_names)}")

# --- sarimax_rs ---
subsection("2b. sarimax_rs fit")
rs_result = sarimax_rs.sarimax_fit(
    y,
    order=ORDER,
    seasonal=SEASONAL,
    enforce_stationarity=False,
    enforce_invertibility=False,
    concentrate_scale=True,
)
rs_params = np.array(rs_result["params"])
print(f"  sarimax_rs params  = {rs_params}")
print(f"  sarimax_rs loglike = {rs_result['loglike']:.12f}")
print(f"  sarimax_rs scale   = {rs_result['scale']:.12f}")
print(f"  sarimax_rs n_obs   = {rs_result['n_obs']}")
print(f"  converged = {rs_result['converged']}, method = {rs_result['method']}")

subsection("2c. Param comparison (own fits)")
for i, (sv, rv) in enumerate(zip(sm_params, rs_params)):
    compare_scalar(f"param[{i}]", sv, rv, tol=0.05)

# ===========================================================================
# 3. Evaluate loglike on BOTH sides using statsmodels params
# ===========================================================================

section("3. Loglike evaluation at statsmodels params")

# Re-evaluate statsmodels loglike at its own params (should match sm_res.llf)
sm_loglike_eval = sm_model.loglike(sm_params)
print(f"  statsmodels loglike(sm_params) = {sm_loglike_eval:.12f}")

# Evaluate sarimax_rs loglike at statsmodels params
rs_loglike_eval = sarimax_rs.sarimax_loglike(
    y,
    order=ORDER,
    seasonal=SEASONAL,
    params=sm_params,
    enforce_stationarity=False,
    enforce_invertibility=False,
    concentrate_scale=True,
)
print(f"  sarimax_rs loglike(sm_params)  = {rs_loglike_eval:.12f}")
compare_scalar("loglike at sm_params", sm_loglike_eval, rs_loglike_eval, tol=1e-4)

# ===========================================================================
# 4. State space matrix comparison
# ===========================================================================

section("4. State Space Matrices")

# Access statsmodels internals
ssm = sm_model.ssm
# Manually set the params so the ssm representation is updated
sm_model.update(sm_params)

# statsmodels matrices
T_sm = np.array(ssm["transition", :, :, 0])
Z_sm = np.array(ssm["design", :, :, 0])  # (1, k_states)
R_sm = np.array(ssm["selection", :, :, 0])
Q_sm = np.array(ssm["state_cov", :, :, 0])

k_states_sm = T_sm.shape[0]
print(f"  statsmodels k_states = {k_states_sm}")
print(f"  T shape = {T_sm.shape}")
print(f"  Z shape = {Z_sm.shape}")
print(f"  R shape = {R_sm.shape}")
print(f"  Q shape = {Q_sm.shape}")

# -- Build sarimax_rs matrices manually using the polynomial logic --
# We reconstruct what sarimax_rs would build internally

subsection("4a. Expected dimensions")

# SARIMA(1,1,1)(1,1,1,12):
# k_ar = p + s*P = 1 + 12*1 = 13
# k_ma = q + s*Q = 1 + 12*1 = 13
# k_order = max(13, 14) = 14
# k_states_diff = d + s*D = 1 + 12*1 = 13
# k_states = 14 + 13 = 27
p, d, q = ORDER
P, D, Q, s = SEASONAL
k_ar = p + s * P
k_ma = q + s * Q
k_order = max(k_ar, k_ma + 1)
k_states_diff = d + s * D
k_states = k_order + k_states_diff

print(f"  Expected: k_ar={k_ar}, k_ma={k_ma}, k_order={k_order}")
print(f"            k_states_diff={k_states_diff}, k_states={k_states}")
print(f"  SM:       k_states={k_states_sm}")

# -- Manually build the T, Z, R matrices as sarimax_rs does --
# (This mirrors state_space.rs logic)

ar_poly = np.array([1.0, -sm_params[0]])  # 1 - phi*L
sar_poly = np.zeros(s + 1)
sar_poly[0] = 1.0
sar_poly[s] = -sm_params[2]  # 1 - Phi*L^s

ma_poly = np.array([1.0, sm_params[1]])  # 1 + theta*L
sma_poly = np.zeros(s + 1)
sma_poly[0] = 1.0
sma_poly[s] = sm_params[3]  # 1 + Theta*L^s

reduced_ar_poly = np.polymul(ar_poly, sar_poly)
reduced_ma_poly = np.polymul(ma_poly, sma_poly)

print(f"\n  Reduced AR poly (len={len(reduced_ar_poly)}): {reduced_ar_poly[:5]}...")
print(f"  Reduced MA poly (len={len(reduced_ma_poly)}): {reduced_ma_poly[:5]}...")

# -- Build T (transition) as sarimax_rs does --
T_rs = np.zeros((k_states, k_states))

# 1. Regular differencing block [0..d, 0..d]: upper triangular ones
for i in range(d):
    for j in range(i, d):
        T_rs[i, j] = 1.0

# 2. Seasonal differencing: cyclic shift blocks
for layer in range(D):
    base = d + layer * s
    T_rs[base, base + s - 1] = 1.0  # wrap
    for i in range(s - 1):
        T_rs[base + i + 1, base + i] = 1.0  # shift down

# 3. Cross-diff: regular diff states -> last seasonal state
if D > 0:
    last_seasonal = d + s * D - 1
    for i in range(d):
        T_rs[i, last_seasonal] = 1.0

# 4. Diff -> ARMA connections
for i in range(d):
    T_rs[i, k_states_diff] = 1.0
for layer in range(D):
    T_rs[d + layer * s, k_states_diff] = 1.0

# 5. ARMA companion matrix
sd = k_states_diff
ko = k_order
for i in range(ko):
    idx = i + 1
    if idx < len(reduced_ar_poly):
        T_rs[sd + i, sd] = -reduced_ar_poly[idx]
for i in range(ko - 1):
    T_rs[sd + i, sd + i + 1] = 1.0

# -- Build Z (design) as sarimax_rs does --
Z_rs = np.zeros(k_states)
for i in range(d):
    Z_rs[i] = 1.0
for layer in range(D):
    Z_rs[d + (layer + 1) * s - 1] = 1.0
Z_rs[sd] = 1.0

# -- Build R (selection) as sarimax_rs does --
R_rs = np.zeros((k_states, 1))
R_rs[sd, 0] = 1.0
for i in range(1, ko):
    if i < len(reduced_ma_poly):
        R_rs[sd + i, 0] = reduced_ma_poly[i]

# -- Q is [[1.0]] for concentrate_scale --
Q_rs = np.array([[1.0]])

subsection("4b. T (transition) comparison")
compare_array("T (sm vs rs_manual)", T_sm, T_rs, tol=1e-10)

subsection("4c. Z (design) comparison")
Z_sm_flat = Z_sm.flatten()
compare_array("Z (sm vs rs_manual)", Z_sm_flat, Z_rs, tol=1e-10)

subsection("4d. R (selection) comparison")
compare_array("R (sm vs rs_manual)", R_sm, R_rs, tol=1e-10)

subsection("4e. Q (state_cov) comparison")
compare_array("Q (sm vs rs_manual)", Q_sm, Q_rs, tol=1e-10)

# Non-zero structure
subsection("4f. T matrix non-zero structure")
nz_sm = np.count_nonzero(T_sm)
nz_rs = np.count_nonzero(T_rs)
print(f"  T non-zeros: sm={nz_sm}, rs={nz_rs}, total={k_states*k_states}")

# Print the T matrix non-zero entries
if not np.allclose(T_sm, T_rs, atol=1e-10):
    print("\n  T matrix differences (non-zero positions):")
    for i in range(k_states):
        for j in range(k_states):
            if abs(T_sm[i, j] - T_rs[i, j]) > 1e-10:
                print(f"    T[{i},{j}]: sm={T_sm[i,j]:+.10f}  rs={T_rs[i,j]:+.10f}")

# ===========================================================================
# 5. Initialization comparison
# ===========================================================================

section("5. Initialization (P_0, loglikelihood_burn)")

# statsmodels initialization
sm_model.update(sm_params)
sm_model.ssm.initialize_approximate_diffuse(1e6)
init_sm = sm_model.ssm.initialization

print(f"  statsmodels initialization_type: approximate_diffuse")
print(f"  statsmodels loglikelihood_burn: {sm_model.loglikelihood_burn}")

# sarimax_rs initialization: approximate_diffuse
# P_0 = kappa * I, burn = k_states
kappa = 1e6
P0_rs = kappa * np.eye(k_states)
burn_rs = k_states

print(f"  sarimax_rs loglikelihood_burn: {burn_rs}")

# Compare burn values
compare_scalar("loglikelihood_burn", float(sm_model.loglikelihood_burn), float(burn_rs), tol=0.5)

# Get P_0 from statsmodels
# statsmodels stores initialization in several possible locations depending on version
P0_sm = None
P0_sm_diag = np.zeros(k_states)
for attr_path in [
    "_initialization.initial_state_cov",
    "_initial_state_cov",
    "initialization.initial_state_cov",
]:
    try:
        obj = sm_model.ssm
        for part in attr_path.split("."):
            obj = getattr(obj, part)
        P0_sm = np.array(obj)
        print(f"  (Accessed P_0 via ssm.{attr_path})")
        break
    except (AttributeError, TypeError):
        continue

# Fallback: infer P_0 from F_t[0]
# F_0 = Z' P_0 Z. For approximate diffuse, P_0 = kappa * I, so
# F_0 = kappa * sum(Z_i^2). We can back-compute kappa from F_0.
if P0_sm is None:
    print("  Could not access P_0 directly; inferring from F_t[0]")
    # After filtering, we know F_0 from statsmodels
    sm_model.update(sm_params)
    sm_model.ssm.initialize_approximate_diffuse(1e6)
    tmp_res = sm_model.ssm.filter()
    F0_sm = tmp_res.forecasts_error_cov[0, 0, 0]
    z_sq_sum = np.sum(Z_rs ** 2)
    kappa_inferred = F0_sm / z_sq_sum
    P0_sm = kappa_inferred * np.eye(k_states)
    print(f"  F_0 (sm) = {F0_sm:.6f}, Z'Z = {z_sq_sum:.6f}")
    print(f"  Inferred kappa = {kappa_inferred:.6f}")

if P0_sm is not None:
    P0_sm_diag = np.diag(P0_sm)
    print(f"\n  P_0 diagonal (sm first 5): {P0_sm_diag[:5]}")
    print(f"  P_0 diagonal (rs first 5): {np.diag(P0_rs)[:5]}")
    compare_array("P_0 diagonal", P0_sm_diag, np.diag(P0_rs), tol=1.0)

    # Check if statsmodels uses a different kappa per state
    unique_diag = np.unique(np.round(P0_sm_diag, 2))
    if len(unique_diag) > 1:
        print(f"\n  ** NOTE: statsmodels P_0 has non-uniform diagonal: {unique_diag}")
        print(f"     This indicates mixed initialization (stationary + diffuse).")
        print(f"     Diff states (0..{k_states_diff}): kappa={P0_sm_diag[0]:.2f}")
        if k_states_diff < k_states:
            print(f"     ARMA states ({k_states_diff}..{k_states}): "
                  f"P_0[{k_states_diff},{k_states_diff}]={P0_sm_diag[k_states_diff]:.6f}")

# ===========================================================================
# 6. Run Kalman filter and compare innovation variances F_t
# ===========================================================================

section("6. Kalman Filter Innovation Variances (F_t)")

# Run statsmodels filter
sm_model.update(sm_params)
sm_model.ssm.initialize_approximate_diffuse(1e6)
sm_res_filter = sm_model.ssm.filter()

# F_t from statsmodels: forecasts_error_cov[0,0,:]
F_sm = sm_res_filter.forecasts_error_cov[0, 0, :]
v_sm = sm_res_filter.forecasts_error[0, :]

print(f"  F_t shape (sm): {F_sm.shape}")
print(f"  v_t shape (sm): {v_sm.shape}")
print(f"  F_t[:5] (sm): {F_sm[:5]}")
print(f"  F_t[-5:] (sm): {F_sm[-5:]}")
print(f"  v_t[:5] (sm): {v_sm[:5]}")

# ===========================================================================
# 7. Loglike formula components comparison
# ===========================================================================

section("7. Loglike Formula Components")

burn_sm = int(sm_model.loglikelihood_burn)

subsection("7a. statsmodels components")
n_eff_sm = n - burn_sm
F_post_burn_sm = F_sm[burn_sm:]
v_post_burn_sm = v_sm[burn_sm:]

sum_log_f_sm = np.sum(np.log(F_post_burn_sm))
sum_v2_f_sm = np.sum(v_post_burn_sm**2 / F_post_burn_sm)
sigma2_hat_sm = sum_v2_f_sm / n_eff_sm

print(f"  burn = {burn_sm}")
print(f"  n_eff = {n_eff_sm}")
print(f"  sum_log_F = {sum_log_f_sm:.12f}")
print(f"  sum_v2/F  = {sum_v2_f_sm:.12f}")
print(f"  sigma2_hat = {sigma2_hat_sm:.12f}")

# Concentrated loglike formula:
# ll = -0.5 * n_eff * ln(2*pi) - 0.5 * n_eff * ln(sigma2_hat) - 0.5 * n_eff - 0.5 * sum_log_F
ll_manual_sm = (
    -0.5 * n_eff_sm * np.log(2 * np.pi)
    - 0.5 * n_eff_sm * np.log(max(sigma2_hat_sm, 1e-300))
    - 0.5 * n_eff_sm
    - 0.5 * sum_log_f_sm
)
print(f"  ll (manual formula) = {ll_manual_sm:.12f}")
print(f"  ll (sm_res.llf)     = {sm_res.llf:.12f}")
print(f"  ll (model.loglike)  = {sm_loglike_eval:.12f}")

subsection("7b. sarimax_rs components (reconstructed)")
# sarimax_rs uses burn = k_states (27 for SARIMA(1,1,1)(1,1,1,12))
n_eff_rs = n - burn_rs
print(f"  burn = {burn_rs}")
print(f"  n_eff = {n_eff_rs}")

# Simulate the Kalman filter manually using sarimax_rs matrices
# to extract the exact F_t values
a = np.zeros(k_states)
P = kappa * np.eye(k_states)
T = T_rs
Z = Z_rs
R = R_rs
Q_mat = Q_rs
RQR = R @ Q_mat @ R.T

F_rs_manual = np.zeros(n)
v_rs_manual = np.zeros(n)

for t in range(n):
    # Innovation
    v_t = y[t] - Z @ a
    v_rs_manual[t] = v_t

    # Innovation variance
    Pz = P @ Z
    f_t = Z @ Pz
    F_rs_manual[t] = f_t

    if f_t > 0:
        # Kalman gain
        K = Pz / f_t

        # Update
        a_upd = a + K * v_t
        P_upd = P - np.outer(K, Pz)

        # Predict
        a = T @ a_upd
        P = T @ P_upd @ T.T + RQR
    else:
        # Skip update, just predict
        a = T @ a
        P = T @ P @ T.T + RQR

F_post_burn_rs = F_rs_manual[burn_rs:]
v_post_burn_rs = v_rs_manual[burn_rs:]

sum_log_f_rs = np.sum(np.log(F_post_burn_rs))
sum_v2_f_rs = np.sum(v_post_burn_rs**2 / F_post_burn_rs)
sigma2_hat_rs = sum_v2_f_rs / n_eff_rs

print(f"  sum_log_F = {sum_log_f_rs:.12f}")
print(f"  sum_v2/F  = {sum_v2_f_rs:.12f}")
print(f"  sigma2_hat = {sigma2_hat_rs:.12f}")

ll_manual_rs = (
    -0.5 * n_eff_rs * np.log(2 * np.pi)
    - 0.5 * n_eff_rs * np.log(max(sigma2_hat_rs, 1e-300))
    - 0.5 * n_eff_rs
    - 0.5 * sum_log_f_rs
)
print(f"  ll (manual formula) = {ll_manual_rs:.12f}")
print(f"  ll (sarimax_rs API) = {rs_loglike_eval:.12f}")

subsection("7c. Component-by-component comparison")
compare_scalar("burn", float(burn_sm), float(burn_rs), tol=0.5)
compare_scalar("n_eff", float(n_eff_sm), float(n_eff_rs), tol=0.5)
compare_scalar("sum_log_F", sum_log_f_sm, sum_log_f_rs, tol=1e-4)
compare_scalar("sum_v2/F", sum_v2_f_sm, sum_v2_f_rs, tol=1e-4)
compare_scalar("sigma2_hat", sigma2_hat_sm, sigma2_hat_rs, tol=1e-4)
compare_scalar("loglike (manual)", ll_manual_sm, ll_manual_rs, tol=1e-4)
compare_scalar("loglike (API)", sm_loglike_eval, rs_loglike_eval, tol=1e-4)

# ===========================================================================
# 8. F_t time series comparison
# ===========================================================================

section("8. F_t Time Series Comparison")

# Compare F_t from statsmodels filter vs our manual reconstruction
subsection("8a. F_t: statsmodels filter vs rs manual filter")
print(f"  F_sm[:10] = {F_sm[:10]}")
print(f"  F_rs[:10] = {F_rs_manual[:10]}")
print(f"  F_sm[-5:] = {F_sm[-5:]}")
print(f"  F_rs[-5:] = {F_rs_manual[-5:]}")

f_diff = np.abs(F_sm - F_rs_manual)
max_f_diff = np.max(f_diff)
max_f_idx = np.argmax(f_diff)
print(f"\n  Max F_t diff = {max_f_diff:.2e} at t={max_f_idx}")
print(f"  F_sm[{max_f_idx}] = {F_sm[max_f_idx]:.12f}")
print(f"  F_rs[{max_f_idx}] = {F_rs_manual[max_f_idx]:.12f}")

# Compare innovations v_t
subsection("8b. v_t: statsmodels vs rs manual")
v_diff = np.abs(v_sm - v_rs_manual)
max_v_diff = np.max(v_diff)
max_v_idx = np.argmax(v_diff)
print(f"  Max v_t diff = {max_v_diff:.2e} at t={max_v_idx}")
if max_v_diff > 1e-6:
    print(f"  v_sm[{max_v_idx}] = {v_sm[max_v_idx]:.12f}")
    print(f"  v_rs[{max_v_idx}] = {v_rs_manual[max_v_idx]:.12f}")

# Find first point of divergence in F_t
subsection("8c. Divergence analysis")
first_diverge = None
for t in range(n):
    if abs(F_sm[t] - F_rs_manual[t]) > 1e-6:
        first_diverge = t
        break

if first_diverge is not None:
    print(f"  First F_t divergence (>1e-6) at t={first_diverge}")
    lo = max(0, first_diverge - 2)
    hi = min(n, first_diverge + 5)
    for t in range(lo, hi):
        print(f"    t={t:4d}  F_sm={F_sm[t]:+.12e}  F_rs={F_rs_manual[t]:+.12e}  "
              f"diff={abs(F_sm[t]-F_rs_manual[t]):.2e}")
else:
    print(f"  No F_t divergence > 1e-6 found. Matrices match!")

# ===========================================================================
# 9. Burn-in analysis (the most likely source of discrepancy)
# ===========================================================================

section("9. Burn-in Analysis")

print(f"\n  statsmodels burn = {burn_sm}")
print(f"  sarimax_rs  burn = {burn_rs}")

if burn_sm != burn_rs:
    print(f"\n  ** BURN MISMATCH ** (sm={burn_sm}, rs={burn_rs})")
    print(f"  This means the two sides sum over DIFFERENT sets of observations.")
    print()

    # Show what happens if we use the SAME burn for manual computation
    for test_burn in [burn_sm, burn_rs]:
        n_eff_test = n - test_burn
        F_post = F_sm[test_burn:]
        v_post = v_sm[test_burn:]
        s_log_f = np.sum(np.log(F_post))
        s_v2_f = np.sum(v_post**2 / F_post)
        s2 = s_v2_f / n_eff_test
        ll_test = (
            -0.5 * n_eff_test * np.log(2 * np.pi)
            - 0.5 * n_eff_test * np.log(max(s2, 1e-300))
            - 0.5 * n_eff_test
            - 0.5 * s_log_f
        )
        print(f"  Using burn={test_burn}: n_eff={n_eff_test}, "
              f"sigma2_hat={s2:.12f}, loglike={ll_test:.12f}")
else:
    print(f"  Burn values match.")

# ===========================================================================
# 10. Summary
# ===========================================================================

section("10. Summary")

all_ok = True

# Matrix checks
print("\n  Matrix checks:")
t_ok = np.allclose(T_sm, T_rs, atol=1e-10)
z_ok = np.allclose(Z_sm.flatten(), Z_rs, atol=1e-10)
r_ok = np.allclose(R_sm, R_rs, atol=1e-10)
q_ok = np.allclose(Q_sm, Q_rs, atol=1e-10)
print(f"    T matrix:  {'OK' if t_ok else 'MISMATCH'}")
print(f"    Z vector:  {'OK' if z_ok else 'MISMATCH'}")
print(f"    R matrix:  {'OK' if r_ok else 'MISMATCH'}")
print(f"    Q matrix:  {'OK' if q_ok else 'MISMATCH'}")
all_ok = all_ok and t_ok and z_ok and r_ok and q_ok

# Initialization
burn_ok = (burn_sm == burn_rs)
print(f"\n  Initialization:")
print(f"    burn match: {'OK' if burn_ok else 'MISMATCH (sm=' + str(burn_sm) + ', rs=' + str(burn_rs) + ')'}")
all_ok = all_ok and burn_ok

# F_t match
f_ok = np.allclose(F_sm, F_rs_manual, atol=1e-6)
print(f"    F_t match:  {'OK' if f_ok else 'MISMATCH'}")
all_ok = all_ok and f_ok

# v_t match
v_ok = np.allclose(v_sm, v_rs_manual, atol=1e-6)
print(f"    v_t match:  {'OK' if v_ok else 'MISMATCH'}")
all_ok = all_ok and v_ok

# Loglike
ll_ok = abs(sm_loglike_eval - rs_loglike_eval) < 1e-4
print(f"\n  Loglike:")
print(f"    sm  = {sm_loglike_eval:.12f}")
print(f"    rs  = {rs_loglike_eval:.12f}")
print(f"    diff = {abs(sm_loglike_eval - rs_loglike_eval):.2e}")
print(f"    match: {'OK' if ll_ok else 'MISMATCH'}")
all_ok = all_ok and ll_ok

print(f"\n{'='*80}")
if all_ok:
    print("  ALL CHECKS PASSED")
else:
    print("  SOME CHECKS FAILED -- see details above")
    print()
    # Identify the root cause
    if not burn_ok:
        print("  ROOT CAUSE CANDIDATE: loglikelihood_burn mismatch")
        print(f"    statsmodels uses burn={burn_sm}, sarimax_rs uses burn={burn_rs}")
        print(f"    For approximate_diffuse init, statsmodels typically uses burn=k_states_diff")
        print(f"    while sarimax_rs uses burn=k_states (entire state dimension).")
        print(f"    k_states_diff = {k_states_diff}, k_states = {k_states}")
    if not t_ok:
        print("  ROOT CAUSE CANDIDATE: Transition matrix T mismatch")
    if not z_ok:
        print("  ROOT CAUSE CANDIDATE: Design vector Z mismatch")
    if not r_ok:
        print("  ROOT CAUSE CANDIDATE: Selection matrix R mismatch")
    if not q_ok:
        print("  ROOT CAUSE CANDIDATE: State covariance Q mismatch")
    if not f_ok and t_ok and z_ok and r_ok and q_ok and burn_ok:
        print("  ROOT CAUSE CANDIDATE: P_0 initialization mismatch")
        print("    All system matrices (T, Z, R, Q) match and burn matches,")
        print("    but F_t innovation variances diverge from t=0.")
        print()
        print("    sarimax_rs uses P_0 = kappa * I (pure approximate diffuse).")
        print("    statsmodels may use mixed initialization even with")
        print("    enforce_stationarity=False if initialize_approximate_diffuse")
        print("    sets different kappa values per state block.")
        print()
        # Quantify: does the F_t difference matter for loglike?
        print("    However, the final LOGLIKE values match closely because the")
        print("    concentrated scale formula (sigma2_hat = sum(v^2/F) / n_eff)")
        print("    absorbs the F_t scaling differences:")
        print(f"      sigma2_hat (sm) = {sigma2_hat_sm:.12f}")
        print(f"      sigma2_hat (rs) = {sigma2_hat_rs:.12f}")
        print(f"      loglike diff    = {abs(sm_loglike_eval - rs_loglike_eval):.2e}")
print(f"{'='*80}")
