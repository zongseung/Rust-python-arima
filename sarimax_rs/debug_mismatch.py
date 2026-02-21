"""
Debug cross-loglike mismatches for:
1. SARIMA(2,1,1)(1,1,1,12)
2. SARIMAX(1,1,1)(1,1,1,12)+exog

Compare step-by-step: manual Python Kalman filter (no steady-state) vs sarimax_rs.
"""
import numpy as np
import sarimax_rs
import statsmodels.api as sm

np.random.seed(42)
n = 500
y = np.cumsum(np.random.standard_normal(n))
exog = np.column_stack([
    np.sin(2 * np.pi * np.arange(n) / 12),
    np.cos(2 * np.pi * np.arange(n) / 12),
])


def manual_kalman_loglike(y, T, Z, R, Q, obs_intercept, kappa, burn):
    """Pure Python Kalman filter (no steady-state) for validation."""
    k = T.shape[0]
    n = len(y)
    RQR = R @ Q @ R.T
    a = np.zeros(k)
    P = kappa * np.eye(k)

    sum_log_f = 0.0
    sum_v2_f = 0.0
    innovations = []
    F_vals = []

    for t in range(n):
        d_t = obs_intercept[t] if t < len(obs_intercept) else 0.0
        v_t = y[t] - Z @ a - d_t
        innovations.append(v_t)

        Pz = P @ Z
        f_t = Z @ Pz
        F_vals.append(f_t)

        if f_t > 0:
            K = Pz / f_t
            a_upd = a + K * v_t
            P_upd = P - np.outer(K, Pz)
            a = T @ a_upd
            P = T @ P_upd @ T.T + RQR

            if t >= burn:
                sum_log_f += np.log(f_t)
                sum_v2_f += v_t * v_t / f_t
        else:
            a = T @ a
            P = T @ P @ T.T + RQR

    n_eff = n - burn
    sigma2_hat = sum_v2_f / n_eff
    ll = (-0.5 * n_eff * np.log(2 * np.pi)
          - 0.5 * n_eff * np.log(max(sigma2_hat, 1e-300))
          - 0.5 * n_eff
          - 0.5 * sum_log_f)

    return ll, sum_log_f, sum_v2_f, sigma2_hat, np.array(F_vals), np.array(innovations)


def build_ss_matrices(order, seasonal, params, n, n_exog=0, exog=None):
    """Build state space matrices matching sarimax_rs logic."""
    p, d, q = order
    PP, DD, QQ, s = seasonal

    k_ar = p + s * PP
    k_ma = q + s * QQ
    k_order = max(k_ar, k_ma + 1)
    k_states_diff = d + s * DD
    k_states = k_order + k_states_diff
    sd = k_states_diff
    ko = k_order

    # Reduced polynomials
    i = 0
    if n_exog > 0:
        i += n_exog  # exog coeffs first
    ar_poly = np.zeros(p + 1); ar_poly[0] = 1.0
    for j in range(p):
        ar_poly[j + 1] = -params[i + j]
    i += p
    ma_poly = np.zeros(q + 1); ma_poly[0] = 1.0
    for j in range(q):
        ma_poly[j + 1] = params[i + j]
    i += q
    sar_poly = np.zeros(s + 1) if s > 0 else np.array([1.0])
    sar_poly[0] = 1.0
    if PP > 0:
        for j in range(PP):
            sar_poly[s * (j + 1)] = -params[i + j]
        i += PP
    sma_poly = np.zeros(s + 1) if s > 0 else np.array([1.0])
    sma_poly[0] = 1.0
    if QQ > 0:
        for j in range(QQ):
            sma_poly[s * (j + 1)] = params[i + j]
        i += QQ

    red_ar = np.polymul(ar_poly, sar_poly)
    red_ma = np.polymul(ma_poly, sma_poly)

    # T matrix
    T = np.zeros((k_states, k_states))
    for ii in range(d):
        for jj in range(ii, d):
            T[ii, jj] = 1.0
    for layer in range(DD):
        base = d + layer * s
        T[base, base + s - 1] = 1.0
        for ii in range(s - 1):
            T[base + ii + 1, base + ii] = 1.0
    if DD > 0:
        last_seasonal = d + s * DD - 1
        for ii in range(d):
            T[ii, last_seasonal] = 1.0
    for ii in range(d):
        T[ii, sd] = 1.0
    for layer in range(DD):
        T[d + layer * s, sd] = 1.0
    for ii in range(ko):
        idx = ii + 1
        if idx < len(red_ar):
            T[sd + ii, sd] = -red_ar[idx]
    for ii in range(ko - 1):
        T[sd + ii, sd + ii + 1] = 1.0

    # Z vector
    Z = np.zeros(k_states)
    for ii in range(d):
        Z[ii] = 1.0
    for layer in range(DD):
        Z[d + (layer + 1) * s - 1] = 1.0
    if sd < k_states:
        Z[sd] = 1.0

    # R (selection)
    R = np.zeros((k_states, 1))
    R[sd, 0] = 1.0
    for ii in range(1, ko):
        if ii < len(red_ma):
            R[sd + ii, 0] = red_ma[ii]

    Q = np.array([[1.0]])

    # obs_intercept (exog)
    obs_intercept = np.zeros(n)
    if exog is not None and n_exog > 0:
        beta = params[:n_exog]
        obs_intercept = exog @ beta

    return T, Z, R, Q, obs_intercept, k_states


def diagnose_model(label, order, seasonal, use_exog=False):
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")

    exog_arg = exog if use_exog else None
    n_exog = 2 if use_exog else 0
    s = seasonal[3]

    # Fit with statsmodels
    sm_model = sm.tsa.SARIMAX(
        y, order=order,
        seasonal_order=seasonal,
        exog=exog_arg,
        enforce_stationarity=False,
        enforce_invertibility=False,
        concentrate_scale=True,
    )
    sm_res = sm_model.fit(disp=False, maxiter=500)
    sm_params = np.array(sm_res.params)
    print(f"  sm params: {sm_params}")
    print(f"  sm loglike: {sm_res.llf:.10f}")

    # Evaluate loglike on both sides at sm_params
    sm_ll = sm_model.loglike(sm_params)
    rs_ll = sarimax_rs.sarimax_loglike(
        y, order=order, seasonal=seasonal,
        params=sm_params, exog=exog_arg,
        enforce_stationarity=False,
        enforce_invertibility=False,
        concentrate_scale=True,
    )
    print(f"\n  Cross-loglike at sm_params:")
    print(f"    sm: {sm_ll:.10f}")
    print(f"    rs: {rs_ll:.10f}")
    print(f"    diff: {abs(sm_ll - rs_ll):.2e}")

    # Build SS matrices and run manual Kalman
    T, Z, R, Q_mat, obs_int, k_states = build_ss_matrices(
        order, seasonal, sm_params, n, n_exog=n_exog, exog=exog_arg)

    # Compare T, Z, R with statsmodels
    sm_model.update(sm_params)
    T_sm = np.array(sm_model.ssm["transition", :, :, 0])
    Z_sm = np.array(sm_model.ssm["design", :, :, 0]).flatten()
    R_sm = np.array(sm_model.ssm["selection", :, :, 0])
    Q_sm = np.array(sm_model.ssm["state_cov", :, :, 0])

    t_ok = np.allclose(T_sm, T, atol=1e-10)
    z_ok = np.allclose(Z_sm, Z, atol=1e-10)
    r_ok = np.allclose(R_sm, R, atol=1e-10)
    q_ok = np.allclose(Q_sm, Q_mat, atol=1e-10)
    print(f"\n  Matrix checks: T={'OK' if t_ok else 'FAIL'} Z={'OK' if z_ok else 'FAIL'} "
          f"R={'OK' if r_ok else 'FAIL'} Q={'OK' if q_ok else 'FAIL'}")

    if not t_ok:
        diff_t = np.abs(T_sm - T)
        worst = np.unravel_index(np.argmax(diff_t), diff_t.shape)
        print(f"    T max diff at {worst}: sm={T_sm[worst]:.10f} rs={T[worst]:.10f}")
    if not z_ok:
        diff_z = np.abs(Z_sm - Z)
        worst = np.argmax(diff_z)
        print(f"    Z max diff at {worst}: sm={Z_sm[worst]:.10f} rs={Z[worst]:.10f}")
    if not r_ok:
        diff_r = np.abs(R_sm - R)
        worst = np.unravel_index(np.argmax(diff_r), diff_r.shape)
        print(f"    R max diff at {worst}: sm={R_sm[worst]:.10f} rs={R[worst]:.10f}")

    # Manual Kalman (no steady-state)
    kappa = 1e6
    burn = k_states
    ll_manual, slf, sv2f, s2, F_manual, v_manual = manual_kalman_loglike(
        y, T, Z, R, Q_mat, obs_int, kappa, burn)

    # Run statsmodels Kalman
    sm_model.ssm.initialize_approximate_diffuse(kappa)
    sm_filt = sm_model.ssm.filter()
    F_sm_arr = sm_filt.forecasts_error_cov[0, 0, :]
    v_sm_arr = sm_filt.forecasts_error[0, :]

    # Compare
    f_diff = np.max(np.abs(F_sm_arr - F_manual))
    v_diff = np.max(np.abs(v_sm_arr - v_manual))
    print(f"\n  Manual Kalman vs sm:")
    print(f"    max F_t diff: {f_diff:.2e}")
    print(f"    max v_t diff: {v_diff:.2e}")
    print(f"    manual loglike: {ll_manual:.10f}")
    print(f"    sm loglike:     {sm_ll:.10f}")
    print(f"    diff:           {abs(ll_manual - sm_ll):.2e}")

    print(f"\n  Manual Kalman vs sarimax_rs:")
    print(f"    manual loglike: {ll_manual:.10f}")
    print(f"    rs loglike:     {rs_ll:.10f}")
    print(f"    diff:           {abs(ll_manual - rs_ll):.2e}")

    if abs(ll_manual - sm_ll) < 1e-4 and abs(ll_manual - rs_ll) > 1e-4:
        print(f"\n  >>> DIAGNOSIS: sarimax_rs Kalman filter has a bug for this model")
        print(f"      (manual matches sm, but rs deviates)")
        print(f"      Likely cause: steady-state optimization or sparse T path")
    elif abs(ll_manual - sm_ll) > 1e-4:
        print(f"\n  >>> DIAGNOSIS: Matrix construction differs")
        print(f"      (manual Python doesn't match sm either)")
    else:
        print(f"\n  >>> DIAGNOSIS: All implementations agree")


diagnose_model("SARIMA(2,1,1)(1,1,1,12)", (2,1,1), (1,1,1,12))
diagnose_model("SARIMAX(1,1,1)(1,1,1,12)+exog", (1,1,1), (1,1,1,12), use_exog=True)
diagnose_model("SARIMA(1,1,1)(1,1,1,12) [control]", (1,1,1), (1,1,1,12))
