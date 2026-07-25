//! Analytical gradient (score vector) via tangent linear Kalman filter.
//!
//! Computes dloglike/dtheta for each parameter theta in a single forward pass,
//! avoiding the O(n_params+1) cost of numerical differentiation.
//!
//! Mathematical basis: Harvey (1989) score formula with concentrated scale.
//!
//! Concentrated log-likelihood:
//!   ll_c = -n/2*ln(2pi) - n/2*ln(sigma2) - n/2 - 1/2*Sum ln(F_t)
//!   where sigma2 = (1/n_eff)*Sum(v_t^2/F_t)
//!
//! Score:
//!   dll_c/dtheta_i = -(1/sigma2)*Sum(v/F)*dv/dtheta + 1/2*Sum[v^2/(sigma2*F^2) - 1/F]*dF/dtheta

use nalgebra::{DMatrix, DVector};

use crate::error::{Result, SarimaxError};
use crate::initialization::{solve_lyapunov_general, KalmanInit};
use crate::kalman::{
    check_convergence, SparseT, SparseZ, STEADY_STATE_MIN_STEPS,
};
use crate::params::SarimaxParams;
use crate::polynomial::{
    make_ar_poly, make_ma_poly, make_seasonal_ar_poly, make_seasonal_ma_poly, polymul,
};
use crate::state_space::StateSpace;
use crate::types::{SarimaxConfig, Trend};

// ---------------------------------------------------------------------------
// System matrix derivatives
// ---------------------------------------------------------------------------

/// Entries with absolute value below this are pruned from the derivative
/// sparsity patterns (structural zeros from polynomial cancellation).
const DERIV_SPARSITY_EPS: f64 = 1e-15;

/// Compute dR*Q*R' + R*Q*dR' for a single-column noise matrix R.
///
/// dRQR[i,j] = σ² * (dR[i] * R[j] + R[i] * dR[j])
#[inline]
fn compute_drqr(
    dr_col: &DVector<f64>,
    r_mat: &DMatrix<f64>,
    k: usize,
    sigma2: f64,
) -> DMatrix<f64> {
    let mut d_rqr = DMatrix::<f64>::zeros(k, k);
    let r_col = r_mat.column(0);
    for row in 0..k {
        for col in 0..k {
            d_rqr[(row, col)] = sigma2 * (dr_col[row] * r_col[col] + r_col[row] * dr_col[col]);
        }
    }
    d_rqr
}

/// Precomputed derivatives of system matrices w.r.t. each constrained parameter.
struct SystemDerivatives {
    n_params: usize,
    /// dT[i]: sparse (row, col, val) entries for dT/dtheta_i.
    dt: Vec<Vec<(usize, usize, f64)>>,
    /// dRQR[i]: d(R*Q*R')/dtheta_i as full k x k matrix.
    drqr: Vec<Option<DMatrix<f64>>>,
    /// dd[i]: obs intercept derivative per time step. Exog param j -> x_j[t].
    dd: Vec<Vec<f64>>,
    /// dc[i]: state intercept derivative per time step. Trend params only.
    dc: Vec<Vec<f64>>,
}

fn precompute_derivatives(
    config: &SarimaxConfig,
    params: &SarimaxParams,
    ss: &StateSpace,
    endog_len: usize,
    exog: Option<&[Vec<f64>]>,
) -> SystemDerivatives {
    let order = &config.order;
    let k = ss.k_states;
    let sd = config.effective_sd();
    let ko = order.k_order();
    let p = order.p;
    let q = order.q;
    let pp = order.pp;
    let qq = order.qq;
    let s = order.s;
    let kt = config.trend.k_trend();
    let n_exog = config.n_exog;
    let n = endog_len;

    let n_params = kt + n_exog + p + q + pp + qq + if config.concentrate_scale { 0 } else { 1 };

    let mut dt: Vec<Vec<(usize, usize, f64)>> = vec![vec![]; n_params];
    let mut drqr: Vec<Option<DMatrix<f64>>> = vec![None; n_params];
    let mut dd: Vec<Vec<f64>> = vec![vec![]; n_params];
    let mut dc: Vec<Vec<f64>> = vec![vec![]; n_params];

    let ar_poly = make_ar_poly(&params.ar_coeffs, p);
    let sar_poly = make_seasonal_ar_poly(&params.sar_coeffs, s);
    let ma_poly = make_ma_poly(&params.ma_coeffs, q);
    let sma_poly = make_seasonal_ma_poly(&params.sma_coeffs, s);

    let r_mat = &ss.selection;
    let q_mat = &ss.state_cov;
    let sigma2 = q_mat[(0, 0)];

    let mut param_idx = 0;

    // ---- Trend parameters ----
    for ti in 0..kt {
        let mut c_deriv = vec![0.0; n * k];
        let inject_idx = sd;
        for t in 0..n {
            c_deriv[t * k + inject_idx] = config.trend.basis(ti, t);
        }
        dc[param_idx] = c_deriv;
        param_idx += 1;
    }

    // ---- Exog parameters ----
    for j in 0..n_exog {
        if let Some(x) = exog {
            dd[param_idx] = x[j].clone();
        }
        param_idx += 1;
    }

    // ---- AR parameters phi_j -> dT ----
    for j in 0..p {
        let mut d_ar = vec![0.0; p + 1];
        d_ar[j + 1] = -1.0;
        let d_reduced = polymul(&d_ar, &sar_poly);
        let mut entries = Vec::new();
        for i in 0..ko {
            let idx = i + 1;
            if idx < d_reduced.len() {
                let val = -d_reduced[idx];
                if val.abs() > DERIV_SPARSITY_EPS {
                    entries.push((sd + i, sd, val));
                }
            }
        }
        dt[param_idx] = entries;
        param_idx += 1;
    }

    // ---- MA parameters theta_j -> dR -> dRQR ----
    for j in 0..q {
        let mut d_ma = vec![0.0; q + 1];
        d_ma[j + 1] = 1.0;
        let d_reduced = polymul(&d_ma, &sma_poly);
        let mut dr_col = DVector::<f64>::zeros(k);
        for i in 0..ko {
            if i < d_reduced.len() && d_reduced[i].abs() > DERIV_SPARSITY_EPS {
                dr_col[sd + i] = d_reduced[i];
            }
        }
        drqr[param_idx] = Some(compute_drqr(&dr_col, r_mat, k, sigma2));
        param_idx += 1;
    }

    // ---- Seasonal AR parameters Phi_j -> dT ----
    for j in 0..pp {
        let len = pp * s + 1;
        let mut d_sar = vec![0.0; len];
        d_sar[(j + 1) * s] = -1.0;
        let d_reduced = polymul(&ar_poly, &d_sar);
        let mut entries = Vec::new();
        for i in 0..ko {
            let idx = i + 1;
            if idx < d_reduced.len() {
                let val = -d_reduced[idx];
                if val.abs() > DERIV_SPARSITY_EPS {
                    entries.push((sd + i, sd, val));
                }
            }
        }
        dt[param_idx] = entries;
        param_idx += 1;
    }

    // ---- Seasonal MA parameters Theta_j -> dR -> dRQR ----
    for j in 0..qq {
        let len = qq * s + 1;
        let mut d_sma = vec![0.0; len];
        d_sma[(j + 1) * s] = 1.0;
        let d_reduced = polymul(&ma_poly, &d_sma);
        let mut dr_col = DVector::<f64>::zeros(k);
        for i in 0..ko {
            if i < d_reduced.len() && d_reduced[i].abs() > DERIV_SPARSITY_EPS {
                dr_col[sd + i] = d_reduced[i];
            }
        }
        drqr[param_idx] = Some(compute_drqr(&dr_col, r_mat, k, sigma2));
        param_idx += 1;
    }

    // ---- Sigma2 (non-concentrated) -> dQ -> dRQR ----
    if !config.concentrate_scale && param_idx < n_params {
        let mut d_rqr = DMatrix::<f64>::zeros(k, k);
        let r_col = r_mat.column(0);
        for row in 0..k {
            for col in 0..k {
                d_rqr[(row, col)] = r_col[row] * r_col[col];
            }
        }
        drqr[param_idx] = Some(d_rqr);
    }

    SystemDerivatives {
        n_params,
        dt,
        drqr,
        dd,
        dc,
    }
}

// ---------------------------------------------------------------------------
// Score computation
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared tangent-linear Kalman pass
// ---------------------------------------------------------------------------

/// Receives one record per effective observation (t >= burn) from
/// `tlkf_pass`. `dv`/`df` hold the per-parameter innovation and
/// innovation-variance derivatives at that observation; `df` stays frozen at
/// its steady-state values once the filter converges.
trait TlkfSink {
    fn record(&mut self, v_t: f64, f_inv: f64, dv: &[f64], df: &[f64]);
}

/// Running sums for the aggregate score (`score()`).
struct ScoreSums {
    sum_v_dv: Vec<f64>,     // Sum (v/F)*dv
    sum_v2f2_df: Vec<f64>,  // Sum (v^2/F^2)*dF
    sum_inv_f_df: Vec<f64>, // Sum (1/F)*dF
}

impl ScoreSums {
    fn new(np: usize) -> Self {
        Self {
            sum_v_dv: vec![0.0; np],
            sum_v2f2_df: vec![0.0; np],
            sum_inv_f_df: vec![0.0; np],
        }
    }
}

impl TlkfSink for ScoreSums {
    fn record(&mut self, v_t: f64, f_inv: f64, dv: &[f64], df: &[f64]) {
        for i in 0..dv.len() {
            self.sum_v_dv[i] += v_t * f_inv * dv[i];
            self.sum_v2f2_df[i] += v_t * v_t * f_inv * f_inv * df[i];
            self.sum_inv_f_df[i] += f_inv * df[i];
        }
    }
}

/// Per-observation records for `score_obs()`.
struct ObsRecords {
    v: Vec<f64>,
    f_inv: Vec<f64>,
    dv: Vec<Vec<f64>>,
    df: Vec<Vec<f64>>,
}

impl ObsRecords {
    fn with_capacity(n_eff: usize) -> Self {
        Self {
            v: Vec::with_capacity(n_eff),
            f_inv: Vec::with_capacity(n_eff),
            dv: Vec::with_capacity(n_eff),
            df: Vec::with_capacity(n_eff),
        }
    }
}

impl TlkfSink for ObsRecords {
    fn record(&mut self, v_t: f64, f_inv: f64, dv: &[f64], df: &[f64]) {
        self.v.push(v_t);
        self.f_inv.push(f_inv);
        self.dv.push(dv.to_vec());
        self.df.push(df.to_vec());
    }
}

/// Tangent-linear Kalman filter pass shared by `score()` and `score_obs()`.
///
/// Runs the standard Kalman recursion together with its tangent-linear
/// counterpart (da, dP per parameter), emitting one `sink.record()` per
/// effective observation, and returns `Sum v_t^2/F_t` (for the concentrated
/// sigma^2).
///
/// Steady-state detection: once pz = P*Z converges, dpz = dP*Z and
/// dF = Z'*dP*Z also converge. After convergence dp, dpz_buf and df_buf are
/// frozen and the expensive O(k^2*np) dP update and O(k^3*np) dP predict
/// steps are skipped. Uses the same constants and check_convergence() from
/// kalman.rs.
fn tlkf_pass<S: TlkfSink>(
    endog: &[f64],
    ss: &StateSpace,
    init: &KalmanInit,
    derivs: &SystemDerivatives,
    sink: &mut S,
) -> Result<f64> {
    let n = endog.len();
    let k = ss.k_states;
    let burn = init.loglikelihood_burn;
    let np = derivs.n_params;

    let z = &ss.design;
    let t_mat = &ss.transition;
    let r_mat = &ss.selection;
    let q_mat = &ss.state_cov;
    let rqr = r_mat * q_mat * r_mat.transpose();
    let t_mat_t = t_mat.transpose();
    let has_state_intercept = ss.state_intercept.len() == n * k;

    // Build shared sparse representations from kalman.rs
    let sparse_t = SparseT::from_dense(t_mat, k);
    let use_sparse_t = sparse_t.is_sparse();
    let sparse_z = SparseZ::from_dense(z, k);

    // Standard KF state
    let mut a = init.initial_state.clone();
    let mut p = init.initial_state_cov.clone();

    // Tangent linear state
    let mut da: Vec<DVector<f64>> = vec![DVector::zeros(k); np];
    let mut dp: Vec<DMatrix<f64>> = vec![DMatrix::zeros(k, k); np];

    // dP_0/dtheta: zero for diffuse initializations (P_0 = kappa*I), but the
    // Lyapunov P_0 of the stationary ARMA block depends on the AR/MA/sigma2
    // parameters. Differentiating P = T P T' + RQR' gives another discrete
    // Lyapunov equation, dP = T dP T' + (dT P T' + T P dT' + dRQR), solved
    // with the same doubling core (rhs is symmetric but may be indefinite).
    if let Some((sb, ko)) = init.stationary_block {
        let t_arma = t_mat.view((sb, sb), (ko, ko)).into_owned();
        let p_arma = init.initial_state_cov.view((sb, sb), (ko, ko)).into_owned();
        let t_arma_t = t_arma.transpose();
        for i in 0..np {
            let mut rhs = DMatrix::<f64>::zeros(ko, ko);
            let mut nonzero = false;
            if !derivs.dt[i].is_empty() {
                let mut dt_block = DMatrix::<f64>::zeros(ko, ko);
                for &(row, col, val) in &derivs.dt[i] {
                    if row >= sb && row < sb + ko && col >= sb && col < sb + ko {
                        dt_block[(row - sb, col - sb)] = val;
                        nonzero = true;
                    }
                }
                if nonzero {
                    let m = &dt_block * &p_arma * &t_arma_t;
                    rhs += &m + m.transpose();
                }
            }
            if let Some(ref drqr_i) = derivs.drqr[i] {
                rhs += drqr_i.view((sb, sb), (ko, ko)).into_owned();
                nonzero = true;
            }
            if !nonzero {
                continue;
            }
            // On solver failure leave dp[i] = 0 (previous behavior).
            if let Some(dp0) = solve_lyapunov_general(&t_arma, &rhs) {
                for r in 0..ko {
                    for c in 0..ko {
                        dp[i][(sb + r, sb + c)] = dp0[(r, c)];
                    }
                }
            }
        }
    }

    // Work buffers
    let mut pz = DVector::<f64>::zeros(k);
    let mut a_next = DVector::<f64>::zeros(k);
    let mut temp_kk = DMatrix::<f64>::zeros(k, k);

    // Per-parameter temporary buffers for tangent linear innovation
    let mut dv_buf = vec![0.0_f64; np];
    let mut df_buf = vec![0.0_f64; np];
    let mut dpz_buf: Vec<DVector<f64>> = (0..np).map(|_| DVector::zeros(k)).collect();

    // Tangent linear predict work buffers
    let mut da_next_i = DVector::<f64>::zeros(k);
    let mut dp_next_i = DMatrix::<f64>::zeros(k, k);
    let mut dt_a = DVector::<f64>::zeros(k);
    let mut temp2 = DMatrix::<f64>::zeros(k, k);

    let mut sum_v2_f = 0.0; // Sum v^2/F  (for sigma^2)

    // Steady-state detection state
    let mut pz_prev = DVector::<f64>::zeros(k);
    let mut ss_converged = false;
    let mut ss_consec = 0_usize;
    let mut f_inv_steady = 0.0;

    for t in 0..n {
        // ---- Step 1: Innovation (from predicted state) ----
        let d_t = if t < ss.obs_intercept.len() {
            ss.obs_intercept[t]
        } else {
            0.0
        };
        let v_t = endog[t] - sparse_z.dot(&a) - d_t;

        if ss_converged {
            // ---- STEADY-STATE PATH: P, dP, dpz, dF are frozen ----
            let f_inv = f_inv_steady;

            // Tangent linear innovation: only dv changes (depends on da)
            for i in 0..np {
                let dd_i_t = if !derivs.dd[i].is_empty() && t < derivs.dd[i].len() {
                    derivs.dd[i][t]
                } else {
                    0.0
                };
                dv_buf[i] = -dd_i_t - sparse_z.dot(&da[i]);
            }

            if t >= burn {
                sum_v2_f += v_t * v_t * f_inv;
                // dpz_buf and df_buf are frozen from the convergence point
                sink.record(v_t, f_inv, &dv_buf, &df_buf);
            }

            // Standard KF update (steady-state): a_{t|t} = a + K_inf * v_t / F
            // Actually: a_{t|t} = a + (v/F)*pz
            // Then predict: a_{t+1|t} = T * a_{t|t} + c
            // Combined: a_{t+1|t} = T*a + T*(v/F)*pz + c = T*a + v*K_inf + c
            // where K_inf = T*pz/F (precomputed)

            // Tangent linear update + predict for da (combined, no dP update):
            // da_{t|t}_i = da_i + (dv_i/F - v*dF_i/F^2)*pz + (v/F)*dpz_i
            // da_{t+1|t}_i = dT_i * a_{t|t} + T * da_{t|t}_i + dc_i
            for i in 0..np {
                let coeff1 = dv_buf[i] * f_inv - v_t * df_buf[i] * f_inv * f_inv;
                let coeff2 = v_t * f_inv;
                {
                    let da_s = da[i].as_mut_slice();
                    let pz_s = pz.as_slice();
                    let dpz_s = dpz_buf[i].as_slice();
                    for r in 0..k {
                        da_s[r] += coeff1 * pz_s[r] + coeff2 * dpz_s[r];
                    }
                }

                // Predict step for da
                sparse_dt_vec(&derivs.dt[i], a.as_slice(), k, &mut dt_a);
                // Add K_inf * v contribution to state before computing dT*a
                // Actually a_{t|t} = a + (v/F)*pz, so dT*a_{t|t} uses the updated a
                // But we need a_{t|t} not a_{t+1|t} for dT:
                // For now, use pre-update a + correction
                {
                    // dt_a already has dT*a (pre-update)
                    // Need to add dT * (v/F)*pz
                    for &(row, col, val) in &derivs.dt[i] {
                        dt_a.as_mut_slice()[row] += val * v_t * f_inv * pz[col];
                    }
                }
                da_next_i.gemv(1.0, t_mat, &da[i], 0.0);
                for r in 0..k {
                    da_next_i[r] += dt_a[r];
                }
                if !derivs.dc[i].is_empty() {
                    let base = t * k;
                    for r in 0..k {
                        da_next_i[r] += derivs.dc[i][base + r];
                    }
                }
                da[i].copy_from(&da_next_i);
            }

            // Standard KF state predict: a_{t+1|t} = T*(a + (v/F)*pz) + c
            a.axpy(v_t * f_inv, &pz, 1.0);
            a_next.gemv(1.0, t_mat, &a, 0.0);
            if has_state_intercept {
                for r in 0..k {
                    a_next[r] += ss.state_intercept[t * k + r];
                }
            }
            std::mem::swap(&mut a, &mut a_next);
            continue;
        }

        // ---- NON-STEADY-STATE PATH (full computation) ----
        sparse_z.p_mul_z_into(&p, &mut pz, k);
        let f_t = sparse_z.z_dot_pz(&pz);

        if f_t <= 0.0 {
            if t >= burn {
                return Err(SarimaxError::DataError(format!(
                    "F_t <= 0 at t={} in score computation",
                    t
                )));
            }
            // Burn-in with F<=0: skip update, just predict
            // Standard KF predict
            a_next.gemv(1.0, t_mat, &a, 0.0);
            if has_state_intercept {
                for r in 0..k {
                    a_next[r] += ss.state_intercept[t * k + r];
                }
            }
            if use_sparse_t {
                sparse_t.t_mul_p_into(&p, &mut temp_kk);
                sparse_t.temp_tt_plus_rqr_into(&temp_kk, &rqr, &mut p);
            } else {
                temp_kk.gemm(1.0, t_mat, &p, 0.0);
                p.gemm(1.0, &temp_kk, &t_mat_t, 0.0);
                p += &rqr;
            }

            // Tangent linear predict (no update since F<=0)
            for i in 0..np {
                sparse_dt_vec(&derivs.dt[i], a.as_slice(), k, &mut dt_a);
                da_next_i.gemv(1.0, t_mat, &da[i], 0.0);
                for r in 0..k {
                    da_next_i[r] += dt_a[r];
                }
                if !derivs.dc[i].is_empty() {
                    let base = t * k;
                    for r in 0..k {
                        da_next_i[r] += derivs.dc[i][base + r];
                    }
                }
                da[i].copy_from(&da_next_i);

                compute_dp_predict(
                    &derivs.dt[i],
                    &derivs.drqr[i],
                    t_mat,
                    &t_mat_t,
                    &sparse_t,
                    use_sparse_t,
                    &p,
                    &dp[i],
                    k,
                    &mut dp_next_i,
                    &mut temp_kk,
                    &mut temp2,
                );
                dp[i].copy_from(&dp_next_i);
            }

            std::mem::swap(&mut a, &mut a_next);
            continue;
        }

        let f_inv = 1.0 / f_t;

        // ---- Step 2: Tangent linear innovation (compute and save dv, dpz, dF) ----
        for i in 0..np {
            let dd_i_t = if !derivs.dd[i].is_empty() && t < derivs.dd[i].len() {
                derivs.dd[i][t]
            } else {
                0.0
            };
            dv_buf[i] = -dd_i_t - sparse_z.dot(&da[i]);
            sparse_z.p_mul_z_into(&dp[i], &mut dpz_buf[i], k);
            df_buf[i] = sparse_z.z_dot_pz(&dpz_buf[i]);
        }

        if t >= burn {
            sum_v2_f += v_t * v_t * f_inv;
            sink.record(v_t, f_inv, &dv_buf, &df_buf);
        }

        // ---- Step 3: Standard KF update ----
        // a_{t|t} = a + (v/F)*pz
        a.axpy(v_t * f_inv, &pz, 1.0);
        // P_{t|t} = P - (1/F)*pz*pz'
        p.ger(-f_inv, &pz, &pz, 1.0);

        // ---- Step 4: Tangent linear update ----
        for i in 0..np {
            let coeff1 = dv_buf[i] * f_inv - v_t * df_buf[i] * f_inv * f_inv;
            let coeff2 = v_t * f_inv;
            {
                let da_s = da[i].as_mut_slice();
                let pz_s = pz.as_slice();
                let dpz_s = dpz_buf[i].as_slice();
                for r in 0..k {
                    da_s[r] += coeff1 * pz_s[r] + coeff2 * dpz_s[r];
                }
            }
            {
                let dp_data = dp[i].as_mut_slice();
                let pz_s = pz.as_slice();
                let dpz_s = dpz_buf[i].as_slice();
                let coeff_dpz = -f_inv;
                let coeff_df = df_buf[i] * f_inv * f_inv;
                for col in 0..k {
                    let col_off = col * k;
                    for row in 0..k {
                        dp_data[col_off + row] += coeff_dpz
                            * (dpz_s[row] * pz_s[col] + pz_s[row] * dpz_s[col])
                            + coeff_df * pz_s[row] * pz_s[col];
                    }
                }
            }
        }

        // ---- Step 5: Tangent linear predict (using a_{t|t}, P_{t|t}, da_{t|t}, dP_{t|t}) ----
        for i in 0..np {
            // da_{t+1|t} = dT_i * a_{t|t} + T * da_{t|t} + dc_i
            sparse_dt_vec(&derivs.dt[i], a.as_slice(), k, &mut dt_a);
            da_next_i.gemv(1.0, t_mat, &da[i], 0.0);
            for r in 0..k {
                da_next_i[r] += dt_a[r];
            }
            if !derivs.dc[i].is_empty() {
                let base = t * k;
                for r in 0..k {
                    da_next_i[r] += derivs.dc[i][base + r];
                }
            }
            da[i].copy_from(&da_next_i);

            // dP_{t+1|t} = dT_i*P_{t|t}*T' + T*dP_{t|t}*T' + T*P_{t|t}*dT_i' + dRQR_i
            compute_dp_predict(
                &derivs.dt[i],
                &derivs.drqr[i],
                t_mat,
                &t_mat_t,
                &sparse_t,
                use_sparse_t,
                &p,
                &dp[i],
                k,
                &mut dp_next_i,
                &mut temp_kk,
                &mut temp2,
            );
            dp[i].copy_from(&dp_next_i);
        }

        // ---- Step 6: Standard KF predict ----
        a_next.gemv(1.0, t_mat, &a, 0.0);
        if has_state_intercept {
            for r in 0..k {
                a_next[r] += ss.state_intercept[t * k + r];
            }
        }
        if use_sparse_t {
            sparse_t.t_mul_p_into(&p, &mut temp_kk);
            sparse_t.temp_tt_plus_rqr_into(&temp_kk, &rqr, &mut p);
        } else {
            temp_kk.gemm(1.0, t_mat, &p, 0.0);
            p.gemm(1.0, &temp_kk, &t_mat_t, 0.0);
            p += &rqr;
        }

        // ---- Step 7: Steady-state convergence check (pz-vector based) ----
        // Same criterion as kalman.rs: check if pz = P*Z has converged.
        // Once pz converges, P has converged, so dP also converges.
        if t >= burn + STEADY_STATE_MIN_STEPS {
            // Recompute pz from the PREDICTED P (just updated above)
            sparse_z.p_mul_z_into(&p, &mut pz, k);

            if check_convergence(&pz, &pz_prev, &mut ss_consec) {
                ss_converged = true;
                let f_steady = sparse_z.z_dot_pz(&pz);
                f_inv_steady = 1.0 / f_steady;
                // dpz_buf and df_buf are already at their steady-state values
                // from the last non-steady iteration
            }
            pz_prev.copy_from(&pz);
        } else if t >= burn {
            // Before min_steps, just track pz for comparison
            sparse_z.p_mul_z_into(&p, &mut pz, k);
            pz_prev.copy_from(&pz);
        }

        std::mem::swap(&mut a, &mut a_next);
    }

    Ok(sum_v2_f)
}

/// Compute the score vector dloglike/dtheta using the tangent linear Kalman filter.
///
/// Returns gradient w.r.t. **constrained** parameters.
pub fn score(
    endog: &[f64],
    ss: &StateSpace,
    init: &KalmanInit,
    config: &SarimaxConfig,
    params: &SarimaxParams,
    concentrate_scale: bool,
    exog: Option<&[Vec<f64>]>,
) -> Result<Vec<f64>> {
    let n = endog.len();
    let burn = init.loglikelihood_burn;

    if n <= burn {
        return Err(SarimaxError::DataError(format!(
            "Not enough observations: n={} <= burn={}",
            n, burn
        )));
    }
    let n_eff = n - burn;

    let derivs = precompute_derivatives(config, params, ss, n, exog);
    let np = derivs.n_params;
    if np == 0 {
        return Ok(vec![]);
    }

    let mut sums = ScoreSums::new(np);
    let sum_v2_f = tlkf_pass(endog, ss, init, &derivs, &mut sums)?;

    // ---- Assemble score ----
    // Concentrated (scale-free filter, Q=1): ll = -n/2(ln2π+1+ln σ̂²) - ½Σln F,
    // whose score divides the innovation terms by σ̂² = (1/n)Σv²/F.
    // Non-concentrated (Q=σ² inside the filter): ll = c - ½Σln F - ½Σv²/F and
    // F_t already carries the scale, so dividing by σ² again would
    // double-scale the innovation terms — the divisor must be 1.
    let sigma2_hat = if concentrate_scale {
        let s = sum_v2_f / n_eff as f64;
        if s <= 0.0 {
            return Err(SarimaxError::DataError(
                "concentrated sigma^2 <= 0 in score computation".into(),
            ));
        }
        s
    } else {
        1.0
    };

    // score_i = -(1/sigma^2)*Sum(v/F)*dv + (1/(2*sigma^2))*Sum(v^2/F^2)*dF - (1/2)*Sum(1/F)*dF
    let result: Vec<f64> = (0..np)
        .map(|i| {
            -sums.sum_v_dv[i] / sigma2_hat + 0.5 * sums.sum_v2f2_df[i] / sigma2_hat
                - 0.5 * sums.sum_inv_f_df[i]
        })
        .collect();

    Ok(result)
}

/// Compute per-observation score vectors dl_t/dtheta using Harvey (1989) analytical formula.
///
/// Runs the same tangent-linear Kalman filter as `score()`, but stores per-observation
/// derivatives (dv_t, dF_t, v_t, 1/F_t) and assembles true per-obs scores:
///
///   dl_t/dθ_i = -(v_t / (σ²·F_t)) · dv_t/dθ_i
///              + ½ · [v_t²/(σ²·F_t²) - 1/F_t] · dF_t/dθ_i
///
/// where σ² = σ²_hat = (1/n_eff) · Σ(v_t²/F_t) (concentrated MLE).
///
/// Returns n_eff score vectors of length n_params (constrained space).
/// Used by OPG inference: I_opg = (1/n_eff) * sum(s_t * s_t')
pub fn score_obs(
    endog: &[f64],
    ss: &StateSpace,
    init: &KalmanInit,
    config: &SarimaxConfig,
    params: &SarimaxParams,
    concentrate_scale: bool,
    exog: Option<&[Vec<f64>]>,
) -> Result<Vec<Vec<f64>>> {
    let n = endog.len();
    let burn = init.loglikelihood_burn;

    if n <= burn {
        return Err(SarimaxError::DataError(format!(
            "Not enough observations: n={} <= burn={}",
            n, burn
        )));
    }
    let n_eff = n - burn;

    let derivs = precompute_derivatives(config, params, ss, n, exog);
    let np = derivs.n_params;
    if np == 0 {
        return Ok(vec![vec![]; n_eff]);
    }

    let mut recs = ObsRecords::with_capacity(n_eff);
    let sum_v2_f = tlkf_pass(endog, ss, init, &derivs, &mut recs)?;

    // ---- Assemble per-observation Harvey scores ----
    // Same convention as score(): with a non-concentrated filter Q=σ², F_t
    // already carries the scale, so the per-obs divisor must be 1 (dividing
    // by σ² again would double-scale the innovation terms).
    let sigma2 = if concentrate_scale {
        let s = sum_v2_f / n_eff as f64;
        if s <= 0.0 {
            return Err(SarimaxError::DataError(
                "concentrated sigma^2 <= 0 in score_obs computation".into(),
            ));
        }
        s
    } else {
        1.0
    };

    // Harvey (1989) per-obs score:
    //   dl_t/dθ_i = -(v_t / (σ²·F_t)) · dv_t/dθ_i
    //              + ½ · [v_t²/(σ²·F_t²) - 1/F_t] · dF_t/dθ_i
    let scores: Vec<Vec<f64>> = (0..recs.v.len())
        .map(|t| {
            let vt = recs.v[t];
            let fi = recs.f_inv[t];
            (0..np)
                .map(|i| {
                    let dv_i = recs.dv[t][i];
                    let df_i = recs.df[t][i];
                    -vt * fi * dv_i / sigma2
                        + 0.5 * (vt * vt * fi * fi * df_i / sigma2 - fi * df_i)
                })
                .collect()
        })
        .collect();

    Ok(scores)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Sparse dT * vector multiply (score-specific: dT is per-parameter derivative).
#[inline]
fn sparse_dt_vec(dt: &[(usize, usize, f64)], x: &[f64], k: usize, result: &mut DVector<f64>) {
    let res = result.as_mut_slice();
    for v in res[..k].iter_mut() {
        *v = 0.0;
    }
    for &(row, col, val) in dt {
        res[row] += val * x[col];
    }
}

/// Accumulate `result += src * sparse_T'` using the sparse T representation.
///
/// O(nnz * k) per call, matching the Kalman predict sparse pattern.
#[inline(always)]
fn sparse_accum_right_t_transpose(
    result: &mut DMatrix<f64>,
    src: &DMatrix<f64>,
    sparse_t: &SparseT,
    k: usize,
) {
    let src_data = src.as_slice();
    let res = result.as_mut_slice();
    for &(j, l, val) in &sparse_t.entries {
        let col_l = l * k;
        let col_j = j * k;
        for i in 0..k {
            res[col_j + i] += val * src_data[col_l + i];
        }
    }
}

/// Accumulate `result += src * T'` using either sparse or dense path.
#[inline(always)]
fn accum_mul_t_transpose(
    result: &mut DMatrix<f64>,
    src: &DMatrix<f64>,
    sparse_t: &SparseT,
    t_mat_t: &DMatrix<f64>,
    use_sparse_t: bool,
    k: usize,
) {
    if use_sparse_t {
        sparse_accum_right_t_transpose(result, src, sparse_t, k);
    } else {
        result.gemm(1.0, src, t_mat_t, 1.0);
    }
}

/// Compute `out = T * mat` using either sparse or dense path.
#[inline(always)]
fn t_mul_into(
    out: &mut DMatrix<f64>,
    mat: &DMatrix<f64>,
    sparse_t: &SparseT,
    t_mat: &DMatrix<f64>,
    use_sparse_t: bool,
) {
    if use_sparse_t {
        sparse_t.t_mul_p_into(mat, out);
    } else {
        out.gemm(1.0, t_mat, mat, 0.0);
    }
}

/// Compute dP prediction step:
///   dP_next = dT*P*T' + T*dP*T' + T*P*dT' + dRQR
///
/// When `use_sparse_t` is true, uses sparse T representation (O(nnz*k) per term)
/// instead of dense gemm (O(k^3)), matching the kalman.rs sparse predict pattern.
/// `p_upd` and `dp_upd` must be the UPDATED (post-observation) values.
fn compute_dp_predict(
    dt_sparse: &[(usize, usize, f64)],
    drqr: &Option<DMatrix<f64>>,
    t_mat: &DMatrix<f64>,
    t_mat_t: &DMatrix<f64>,
    sparse_t: &SparseT,
    use_sparse_t: bool,
    p_upd: &DMatrix<f64>,
    dp_upd: &DMatrix<f64>,
    k: usize,
    result: &mut DMatrix<f64>,
    temp: &mut DMatrix<f64>,
    temp2: &mut DMatrix<f64>,
) {
    // Start with dRQR
    if let Some(ref d) = drqr {
        result.copy_from(d);
    } else {
        result.fill(0.0);
    }

    // Term 1: T * dP * T'
    t_mul_into(temp, dp_upd, sparse_t, t_mat, use_sparse_t);
    accum_mul_t_transpose(result, temp, sparse_t, t_mat_t, use_sparse_t, k);

    if dt_sparse.is_empty() {
        return;
    }

    // Term 2: dT * P * T'  (dT is always sparse)
    temp2.fill(0.0);
    let p_data = p_upd.as_slice();
    let tmp2 = temp2.as_mut_slice();
    for &(i, l, val) in dt_sparse {
        for j in 0..k {
            tmp2[i + j * k] += val * p_data[l + j * k];
        }
    }
    accum_mul_t_transpose(result, temp2, sparse_t, t_mat_t, use_sparse_t, k);

    // Term 3: T * P * dT'  (dT' is always sparse)
    t_mul_into(temp, p_upd, sparse_t, t_mat, use_sparse_t);
    let tmp_data = temp.as_slice();
    let res = result.as_mut_slice();
    for &(i, l, val) in dt_sparse {
        let col_l = l * k;
        let col_i = i * k;
        for r in 0..k {
            res[col_i + r] += val * tmp_data[col_l + r];
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::initialization::KalmanInit;
    use crate::kalman::kalman_loglike;
    use crate::params::SarimaxParams;
    use crate::state_space::StateSpace;
    use crate::test_helpers::{load_fixtures, make_config, make_seasonal_config};
    use crate::types::{SarimaxConfig, SarimaxOrder, Trend};

    /// Richardson-extrapolated central differences for robust numerical gradient.
    /// Uses two step sizes and extrapolates to O(h^4) accuracy.
    fn numerical_gradient(
        endog: &[f64],
        config: &SarimaxConfig,
        params_flat: &[f64],
        exog: Option<&[Vec<f64>]>,
    ) -> Vec<f64> {
        let n = params_flat.len();
        let mut grad = vec![0.0; n];

        for i in 0..n {
            // Adaptive step size: h = max(eps, |x|*eps)
            let h = 1e-4_f64.max(params_flat[i].abs() * 1e-4);
            let h2 = h / 2.0;

            let mut p_plus = params_flat.to_vec();
            let mut p_minus = params_flat.to_vec();
            p_plus[i] = params_flat[i] + h;
            p_minus[i] = params_flat[i] - h;
            let g_h = (eval_loglike(endog, config, &p_plus, exog)
                - eval_loglike(endog, config, &p_minus, exog))
                / (2.0 * h);

            p_plus[i] = params_flat[i] + h2;
            p_minus[i] = params_flat[i] - h2;
            let g_h2 = (eval_loglike(endog, config, &p_plus, exog)
                - eval_loglike(endog, config, &p_minus, exog))
                / (2.0 * h2);

            // Richardson extrapolation: (4*g_{h/2} - g_h) / 3 -> O(h^4)
            grad[i] = (4.0 * g_h2 - g_h) / 3.0;
        }
        grad
    }

    fn eval_loglike(
        endog: &[f64],
        config: &SarimaxConfig,
        params_flat: &[f64],
        exog: Option<&[Vec<f64>]>,
    ) -> f64 {
        let sparams = SarimaxParams::from_flat(params_flat, config).unwrap();
        let ss = StateSpace::new(config, &sparams, endog, exog).unwrap();
        let init = KalmanInit::from_config_default(&ss, config);
        let output = kalman_loglike(endog, &ss, &init, config.concentrate_scale).unwrap();
        output.loglike
    }

    fn assert_gradient_close(analytical: &[f64], numerical: &[f64], tol: f64, label: &str) {
        assert_eq!(
            analytical.len(),
            numerical.len(),
            "{}: gradient length mismatch",
            label
        );
        for (i, (&a, &n)) in analytical.iter().zip(numerical.iter()).enumerate() {
            let abs_err = (a - n).abs();
            let denom = n.abs().max(a.abs()).max(1.0);
            let rel_err = abs_err / denom;
            assert!(
                rel_err < tol,
                "{}: param[{}] gradient mismatch: analytical={:.8e}, numerical={:.8e}, abs_err={:.8e}, rel_err={:.8e}",
                label, i, a, n, abs_err, rel_err
            );
        }
    }

    #[test]
    fn test_score_ar1() {
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let phi = 0.6527425084139002;
        let scale = case["scale"].as_f64().unwrap();

        let config = make_config(1, 0, 0);
        let params_flat = vec![phi, scale]; // ar(1) + sigma2

        let sparams = SarimaxParams::from_flat(&params_flat, &config).unwrap();
        let ss = StateSpace::new(&config, &sparams, &data, None).unwrap();
        let init = KalmanInit::from_config_default(&ss, &config);

        let analytical = score(
            &data,
            &ss,
            &init,
            &config,
            &sparams,
            config.concentrate_scale,
            None,
        )
        .unwrap();
        let numerical = numerical_gradient(&data, &config, &params_flat, None);

        assert_gradient_close(&analytical, &numerical, 1e-3, "AR(1)");
    }

    #[test]
    fn test_score_arma11() {
        let fixtures = load_fixtures();
        let case = &fixtures["arma11"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let params_vec: Vec<f64> = case["params"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        let config = make_config(1, 0, 1);
        // fixture params: [ar, ma, sigma2]
        let params_flat = params_vec.clone();

        let sparams = SarimaxParams::from_flat(&params_flat, &config).unwrap();
        let ss = StateSpace::new(&config, &sparams, &data, None).unwrap();
        let init = KalmanInit::from_config_default(&ss, &config);

        let analytical = score(
            &data,
            &ss,
            &init,
            &config,
            &sparams,
            config.concentrate_scale,
            None,
        )
        .unwrap();
        let numerical = numerical_gradient(&data, &config, &params_flat, None);

        assert_gradient_close(&analytical, &numerical, 1e-3, "ARMA(1,1)");
    }

    #[test]
    fn test_score_arima111() {
        let fixtures = load_fixtures();
        let case = &fixtures["arima111"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let params_vec: Vec<f64> = case["params"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        let config = make_config(1, 1, 1);
        // fixture params: [ar, ma, sigma2]
        let params_flat = params_vec.clone();

        let sparams = SarimaxParams::from_flat(&params_flat, &config).unwrap();
        let ss = StateSpace::new(&config, &sparams, &data, None).unwrap();
        let init = KalmanInit::from_config_default(&ss, &config);

        let analytical = score(
            &data,
            &ss,
            &init,
            &config,
            &sparams,
            config.concentrate_scale,
            None,
        )
        .unwrap();
        let numerical = numerical_gradient(&data, &config, &params_flat, None);

        assert_gradient_close(&analytical, &numerical, 1e-3, "ARIMA(1,1,1)");
    }

    #[test]
    fn test_score_sarima_100_100_4() {
        let fixtures = load_fixtures();
        let case = &fixtures["sarima_100_100_4"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let params_vec: Vec<f64> = case["params"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        let config = make_seasonal_config(1, 0, 0, 1, 0, 0, 4);
        // fixture params: [ar, sar, sigma2]
        let params_flat = params_vec.clone();

        let sparams = SarimaxParams::from_flat(&params_flat, &config).unwrap();
        let ss = StateSpace::new(&config, &sparams, &data, None).unwrap();
        let init = KalmanInit::from_config_default(&ss, &config);

        let analytical = score(
            &data,
            &ss,
            &init,
            &config,
            &sparams,
            config.concentrate_scale,
            None,
        )
        .unwrap();
        let numerical = numerical_gradient(&data, &config, &params_flat, None);

        assert_gradient_close(&analytical, &numerical, 1e-3, "SARIMA(1,0,0)(1,0,0,4)");
    }

    #[test]
    fn test_score_sarima_111_111_12() {
        // Use well-conditioned parameters (away from unit root)
        // to avoid numerical differentiation instability.
        let fixtures = load_fixtures();
        let case = &fixtures["sarima_111_111_12"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        let config = make_seasonal_config(1, 1, 1, 1, 1, 1, 12);
        // params: [ar, ma, sar, sma, sigma2] -- well within stationary/invertible region
        let params_flat = vec![0.5, 0.3, 0.2, -0.4, 1.0];

        let sparams = SarimaxParams::from_flat(&params_flat, &config).unwrap();
        let ss = StateSpace::new(&config, &sparams, &data, None).unwrap();
        let init = KalmanInit::from_config_default(&ss, &config);

        let analytical = score(
            &data,
            &ss,
            &init,
            &config,
            &sparams,
            config.concentrate_scale,
            None,
        )
        .unwrap();
        let numerical = numerical_gradient(&data, &config, &params_flat, None);

        assert_gradient_close(&analytical, &numerical, 1e-3, "SARIMA(1,1,1)(1,1,1,12)");
    }

    #[test]
    fn test_score_sarima_111_111_12_fixture_params() {
        // Test with fixture params (phi=0.99, near unit root).
        // Numerical gradient is unreliable here, so use wider tolerance.
        let fixtures = load_fixtures();
        let case = &fixtures["sarima_111_111_12"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let params_vec: Vec<f64> = case["params"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        let config = make_seasonal_config(1, 1, 1, 1, 1, 1, 12);
        // fixture params: [ar, ma, sar, sma, sigma2]
        let params_flat = params_vec.clone();

        let sparams = SarimaxParams::from_flat(&params_flat, &config).unwrap();
        let ss = StateSpace::new(&config, &sparams, &data, None).unwrap();
        let init = KalmanInit::from_config_default(&ss, &config);

        let analytical = score(
            &data,
            &ss,
            &init,
            &config,
            &sparams,
            config.concentrate_scale,
            None,
        )
        .unwrap();
        let numerical = numerical_gradient(&data, &config, &params_flat, None);

        // Wider tolerance: near-unit-root AR makes numerical gradient unreliable
        assert_gradient_close(
            &analytical,
            &numerical,
            0.06,
            "SARIMA(1,1,1)(1,1,1,12) fixture",
        );
    }

    #[test]
    fn test_score_with_exog() {
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let n = data.len();

        let exog_col: Vec<f64> = (0..n).map(|t| (t as f64) * 0.01).collect();
        let exog = [exog_col];

        let config = SarimaxConfig {
            order: SarimaxOrder::new(1, 0, 0, 0, 0, 0, 0),
            n_exog: 1,
            trend: Trend::None,
            enforce_stationarity: false,
            enforce_invertibility: false,
            concentrate_scale: true,
            simple_differencing: false,
        };
        let params_flat = vec![0.5, 0.65];

        let sparams = SarimaxParams::from_flat(&params_flat, &config).unwrap();
        let ss = StateSpace::new(&config, &sparams, &data, Some(&exog[..])).unwrap();
        let init = KalmanInit::from_config_default(&ss, &config);

        let analytical =
            score(&data, &ss, &init, &config, &sparams, true, Some(&exog[..])).unwrap();
        let numerical = numerical_gradient(&data, &config, &params_flat, Some(&exog[..]));

        assert_gradient_close(&analytical, &numerical, 1e-3, "SARIMAX with exog");
    }

    #[test]
    fn test_score_with_trend() {
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        let config = SarimaxConfig {
            order: SarimaxOrder::new(1, 0, 0, 0, 0, 0, 0),
            n_exog: 0,
            trend: Trend::Constant,
            enforce_stationarity: false,
            enforce_invertibility: false,
            concentrate_scale: true,
            simple_differencing: false,
        };
        let params_flat = vec![0.1, 0.65];

        let sparams = SarimaxParams::from_flat(&params_flat, &config).unwrap();
        let ss = StateSpace::new(&config, &sparams, &data, None).unwrap();
        let init = KalmanInit::from_config_default(&ss, &config);

        let analytical = score(&data, &ss, &init, &config, &sparams, true, None).unwrap();
        let numerical = numerical_gradient(&data, &config, &params_flat, None);

        assert_gradient_close(&analytical, &numerical, 1e-3, "AR(1) with trend='c'");
    }
}
