//! Analytical gradient (score vector) via tangent linear Kalman filter.
//!
//! Computes ∂loglike/∂θ for each parameter θ in a single forward pass,
//! avoiding the O(n_params+1) cost of numerical differentiation.
//!
//! Mathematical basis: Harvey (1989) score formula with concentrated scale.
//!
//! Concentrated log-likelihood:
//!   ll_c = -n/2·ln(2π) - n/2·ln(σ²) - n/2 - 1/2·Σ ln(F_t)
//!   where σ² = (1/n_eff)·Σ(v_t²/F_t)
//!
//! Score:
//!   ∂ll_c/∂θ_i = -(1/σ²)·Σ(v/F)·∂v/∂θ + 1/2·Σ[v²/(σ²F²) - 1/F]·∂F/∂θ

use nalgebra::{DMatrix, DVector};

use crate::error::{Result, SarimaxError};
use crate::initialization::KalmanInit;
use crate::polynomial::{polymul, make_ar_poly, make_ma_poly, make_seasonal_ar_poly, make_seasonal_ma_poly};
use crate::state_space::StateSpace;
use crate::types::{SarimaxConfig, Trend};
use crate::params::SarimaxParams;

// ---------------------------------------------------------------------------
// System matrix derivatives
// ---------------------------------------------------------------------------

/// Precomputed derivatives of system matrices w.r.t. each constrained parameter.
struct SystemDerivatives {
    n_params: usize,
    /// dT[i]: sparse (row, col, val) entries for ∂T/∂θ_i.
    dt: Vec<Vec<(usize, usize, f64)>>,
    /// dRQR[i]: ∂(R·Q·R')/∂θ_i as full k×k matrix.
    drqr: Vec<Option<DMatrix<f64>>>,
    /// dd[i]: obs intercept derivative per time step. Exog param j → x_j[t].
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
    let sd = order.k_states_diff();
    let ko = order.k_order();
    let p = order.p;
    let q = order.q;
    let pp = order.pp;
    let qq = order.qq;
    let s = order.s;
    let kt = config.trend.k_trend();
    let n_exog = config.n_exog;
    let n = endog_len;

    let n_params = kt + n_exog + p + q + pp + qq
        + if config.concentrate_scale { 0 } else { 1 };

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
        match config.trend {
            Trend::Constant => {
                for t in 0..n { c_deriv[t * k + inject_idx] = 1.0; }
            }
            Trend::Linear => {
                for t in 0..n { c_deriv[t * k + inject_idx] = t as f64; }
            }
            Trend::Both => {
                if ti == 0 {
                    for t in 0..n { c_deriv[t * k + inject_idx] = 1.0; }
                } else {
                    for t in 0..n { c_deriv[t * k + inject_idx] = t as f64; }
                }
            }
            Trend::None => {}
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

    // ---- AR parameters phi_j → dT ----
    for j in 0..p {
        let mut d_ar = vec![0.0; p + 1];
        d_ar[j + 1] = -1.0;
        let d_reduced = polymul(&d_ar, &sar_poly);
        let mut entries = Vec::new();
        for i in 0..ko {
            let idx = i + 1;
            if idx < d_reduced.len() {
                let val = -d_reduced[idx];
                if val.abs() > 1e-15 {
                    entries.push((sd + i, sd, val));
                }
            }
        }
        dt[param_idx] = entries;
        param_idx += 1;
    }

    // ---- MA parameters theta_j → dR → dRQR ----
    for j in 0..q {
        let mut d_ma = vec![0.0; q + 1];
        d_ma[j + 1] = 1.0;
        let d_reduced = polymul(&d_ma, &sma_poly);
        let mut dr_col = DVector::<f64>::zeros(k);
        for i in 0..ko {
            if i < d_reduced.len() && d_reduced[i].abs() > 1e-15 {
                dr_col[sd + i] = d_reduced[i];
            }
        }
        let mut d_rqr = DMatrix::<f64>::zeros(k, k);
        let r_col = r_mat.column(0);
        for row in 0..k {
            for col in 0..k {
                d_rqr[(row, col)] = sigma2 * (dr_col[row] * r_col[col] + r_col[row] * dr_col[col]);
            }
        }
        drqr[param_idx] = Some(d_rqr);
        param_idx += 1;
    }

    // ---- Seasonal AR parameters Phi_j → dT ----
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
                if val.abs() > 1e-15 {
                    entries.push((sd + i, sd, val));
                }
            }
        }
        dt[param_idx] = entries;
        param_idx += 1;
    }

    // ---- Seasonal MA parameters Theta_j → dR → dRQR ----
    for j in 0..qq {
        let len = qq * s + 1;
        let mut d_sma = vec![0.0; len];
        d_sma[(j + 1) * s] = 1.0;
        let d_reduced = polymul(&ma_poly, &d_sma);
        let mut dr_col = DVector::<f64>::zeros(k);
        for i in 0..ko {
            if i < d_reduced.len() && d_reduced[i].abs() > 1e-15 {
                dr_col[sd + i] = d_reduced[i];
            }
        }
        let mut d_rqr = DMatrix::<f64>::zeros(k, k);
        let r_col = r_mat.column(0);
        for row in 0..k {
            for col in 0..k {
                d_rqr[(row, col)] = sigma2 * (dr_col[row] * r_col[col] + r_col[row] * dr_col[col]);
            }
        }
        drqr[param_idx] = Some(d_rqr);
        param_idx += 1;
    }

    // ---- Sigma2 (non-concentrated) → dQ → dRQR ----
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

    SystemDerivatives { n_params, dt, drqr, dd, dc }
}

// ---------------------------------------------------------------------------
// Score computation
// ---------------------------------------------------------------------------

/// Compute the score vector ∂loglike/∂θ using the tangent linear Kalman filter.
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
    let k = ss.k_states;
    let burn = init.loglikelihood_burn;

    if n <= burn {
        return Err(SarimaxError::DataError(format!(
            "Not enough observations: n={} <= burn={}", n, burn
        )));
    }
    let n_eff = n - burn;

    let derivs = precompute_derivatives(config, params, ss, n, exog);
    let np = derivs.n_params;
    if np == 0 {
        return Ok(vec![]);
    }

    let z = &ss.design;
    let t_mat = &ss.transition;
    let r_mat = &ss.selection;
    let q_mat = &ss.state_cov;
    let rqr = r_mat * q_mat * r_mat.transpose();
    let t_mat_t = t_mat.transpose();
    let has_state_intercept = ss.state_intercept.len() == n * k;

    let sparse_z: Vec<(usize, f64)> = (0..k)
        .filter_map(|i| if z[i] != 0.0 { Some((i, z[i])) } else { None })
        .collect();

    // Standard KF state
    let mut a = init.initial_state.clone();
    let mut p = init.initial_state_cov.clone();

    // Tangent linear state
    let mut da: Vec<DVector<f64>> = vec![DVector::zeros(k); np];
    let mut dp: Vec<DMatrix<f64>> = vec![DMatrix::zeros(k, k); np];

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

    // Score accumulators
    let mut sum_v_dv = vec![0.0; np];       // Σ (v/F)·dv
    let mut sum_v2f2_df = vec![0.0; np];    // Σ (v²/F²)·dF
    let mut sum_inv_f_df = vec![0.0; np];   // Σ (1/F)·dF
    let mut sum_v2_f = 0.0;                 // Σ v²/F  (for σ²)

    for t in 0..n {
        // ---- Step 1: Innovation (from predicted state) ----
        let d_t = if t < ss.obs_intercept.len() { ss.obs_intercept[t] } else { 0.0 };
        let v_t = endog[t] - sparse_z_dot(&sparse_z, a.as_slice()) - d_t;
        sparse_z_mvp(&sparse_z, &p, k, &mut pz);
        let f_t: f64 = sparse_z.iter().map(|&(i, v)| v * pz[i]).sum();

        if f_t <= 0.0 {
            if t >= burn {
                return Err(SarimaxError::DataError(format!(
                    "F_t <= 0 at t={} in score computation", t
                )));
            }
            // Burn-in with F<=0: skip update, just predict
            // Standard KF predict
            a_next.gemv(1.0, t_mat, &a, 0.0);
            if has_state_intercept {
                for r in 0..k { a_next[r] += ss.state_intercept[t * k + r]; }
            }
            temp_kk.gemm(1.0, t_mat, &p, 0.0);
            p.gemm(1.0, &temp_kk, &t_mat_t, 0.0);
            p += &rqr;

            // Tangent linear predict (no update since F<=0)
            for i in 0..np {
                sparse_dt_vec(&derivs.dt[i], a.as_slice(), k, &mut dt_a);
                da_next_i.gemv(1.0, t_mat, &da[i], 0.0);
                for r in 0..k { da_next_i[r] += dt_a[r]; }
                if !derivs.dc[i].is_empty() {
                    let base = t * k;
                    for r in 0..k { da_next_i[r] += derivs.dc[i][base + r]; }
                }
                da[i].copy_from(&da_next_i);

                compute_dp_predict(
                    &derivs.dt[i], &derivs.drqr[i],
                    t_mat, &t_mat_t, &p, &dp[i],
                    k, &mut dp_next_i, &mut temp_kk, &mut temp2,
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
            dv_buf[i] = -dd_i_t - sparse_z_dot(&sparse_z, da[i].as_slice());
            sparse_z_mvp(&sparse_z, &dp[i], k, &mut dpz_buf[i]);
            df_buf[i] = sparse_z.iter().map(|&(idx, v)| v * dpz_buf[i][idx]).sum();

            if t >= burn {
                sum_v_dv[i] += v_t * f_inv * dv_buf[i];
                sum_v2f2_df[i] += v_t * v_t * f_inv * f_inv * df_buf[i];
                sum_inv_f_df[i] += f_inv * df_buf[i];
            }
        }

        if t >= burn {
            sum_v2_f += v_t * v_t * f_inv;
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
                        dp_data[col_off + row] +=
                            coeff_dpz * (dpz_s[row] * pz_s[col] + pz_s[row] * dpz_s[col])
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
            for r in 0..k { da_next_i[r] += dt_a[r]; }
            if !derivs.dc[i].is_empty() {
                let base = t * k;
                for r in 0..k { da_next_i[r] += derivs.dc[i][base + r]; }
            }
            da[i].copy_from(&da_next_i);

            // dP_{t+1|t} = dT_i*P_{t|t}*T' + T*dP_{t|t}*T' + T*P_{t|t}*dT_i' + dRQR_i
            compute_dp_predict(
                &derivs.dt[i], &derivs.drqr[i],
                t_mat, &t_mat_t, &p, &dp[i],
                k, &mut dp_next_i, &mut temp_kk, &mut temp2,
            );
            dp[i].copy_from(&dp_next_i);
        }

        // ---- Step 6: Standard KF predict ----
        a_next.gemv(1.0, t_mat, &a, 0.0);
        if has_state_intercept {
            for r in 0..k { a_next[r] += ss.state_intercept[t * k + r]; }
        }
        temp_kk.gemm(1.0, t_mat, &p, 0.0);
        p.gemm(1.0, &temp_kk, &t_mat_t, 0.0);
        p += &rqr;

        std::mem::swap(&mut a, &mut a_next);
    }

    // ---- Assemble score ----
    let sigma2_hat = if concentrate_scale {
        let s = sum_v2_f / n_eff as f64;
        if s <= 0.0 {
            return Err(SarimaxError::DataError(
                "concentrated σ² <= 0 in score computation".into()
            ));
        }
        s
    } else {
        ss.state_cov[(0, 0)]
    };

    // score_i = -(1/σ²)·Σ(v/F)·dv + (1/(2σ²))·Σ(v²/F²)·dF - (1/2)·Σ(1/F)·dF
    let result: Vec<f64> = (0..np)
        .map(|i| {
            -sum_v_dv[i] / sigma2_hat
            + 0.5 * sum_v2f2_df[i] / sigma2_hat
            - 0.5 * sum_inv_f_df[i]
        })
        .collect();

    Ok(result)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn sparse_z_dot(sparse_z: &[(usize, f64)], x: &[f64]) -> f64 {
    sparse_z.iter().map(|&(i, v)| v * x[i]).sum()
}

#[inline]
fn sparse_z_mvp(sparse_z: &[(usize, f64)], p: &DMatrix<f64>, k: usize, result: &mut DVector<f64>) {
    let p_data = p.as_slice();
    let res = result.as_mut_slice();
    for v in res[..k].iter_mut() { *v = 0.0; }
    for &(j, zv) in sparse_z {
        let col_start = j * k;
        for r in 0..k {
            res[r] += zv * p_data[col_start + r];
        }
    }
}

#[inline]
fn sparse_dt_vec(dt: &[(usize, usize, f64)], x: &[f64], k: usize, result: &mut DVector<f64>) {
    let res = result.as_mut_slice();
    for v in res[..k].iter_mut() { *v = 0.0; }
    for &(row, col, val) in dt {
        res[row] += val * x[col];
    }
}

/// Compute dP prediction step:
///   dP_next = dT*P*T' + T*dP*T' + T*P*dT' + dRQR
///
/// `p_upd` and `dp_upd` must be the UPDATED (post-observation) values.
fn compute_dp_predict(
    dt_sparse: &[(usize, usize, f64)],
    drqr: &Option<DMatrix<f64>>,
    t_mat: &DMatrix<f64>,
    t_mat_t: &DMatrix<f64>,
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

    // Term: T * dP * T'
    temp.gemm(1.0, t_mat, dp_upd, 0.0);
    result.gemm(1.0, temp, t_mat_t, 1.0);

    if dt_sparse.is_empty() {
        return;
    }

    // Term: dT * P * T'
    // Compute temp2 = dT * P (sparse dT × dense P)
    temp2.fill(0.0);
    let p_data = p_upd.as_slice();
    let tmp2 = temp2.as_mut_slice();
    for &(i, l, val) in dt_sparse {
        // temp2[i, j] += val * P[l, j]
        // column-major: P[l,j] = p_data[l + j*k]
        for j in 0..k {
            tmp2[i + j * k] += val * p_data[l + j * k];
        }
    }
    // result += temp2 * T'
    result.gemm(1.0, temp2, t_mat_t, 1.0);

    // Term: T * P * dT'
    // Compute temp = T * P
    temp.gemm(1.0, t_mat, p_upd, 0.0);
    // result += temp * dT'
    // dT'[l, i] = dT[i, l], so result[:, i] += val * temp[:, l]
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
    use crate::types::{SarimaxConfig, SarimaxOrder, Trend};

    fn load_fixtures() -> serde_json::Value {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/statsmodels_reference.json"
        );
        let data = std::fs::read_to_string(path).expect("fixtures file not found");
        serde_json::from_str(&data).expect("invalid JSON")
    }

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
                     - eval_loglike(endog, config, &p_minus, exog)) / (2.0 * h);

            p_plus[i] = params_flat[i] + h2;
            p_minus[i] = params_flat[i] - h2;
            let g_h2 = (eval_loglike(endog, config, &p_plus, exog)
                      - eval_loglike(endog, config, &p_minus, exog)) / (2.0 * h2);

            // Richardson extrapolation: (4*g_{h/2} - g_h) / 3 → O(h^4)
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
        let init = KalmanInit::from_config(&ss, config, KalmanInit::default_kappa());
        let output = kalman_loglike(endog, &ss, &init, config.concentrate_scale).unwrap();
        output.loglike
    }

    fn make_config(p: usize, d: usize, q: usize) -> SarimaxConfig {
        SarimaxConfig {
            order: SarimaxOrder::new(p, d, q, 0, 0, 0, 0),
            n_exog: 0,
            trend: Trend::None,
            enforce_stationarity: false,
            enforce_invertibility: false,
            concentrate_scale: true,
            simple_differencing: false,
            measurement_error: false,
        }
    }

    fn make_seasonal_config(
        p: usize, d: usize, q: usize,
        pp: usize, dd: usize, qq: usize, s: usize,
    ) -> SarimaxConfig {
        SarimaxConfig {
            order: SarimaxOrder::new(p, d, q, pp, dd, qq, s),
            n_exog: 0,
            trend: Trend::None,
            enforce_stationarity: false,
            enforce_invertibility: false,
            concentrate_scale: true,
            simple_differencing: false,
            measurement_error: false,
        }
    }

    fn assert_gradient_close(analytical: &[f64], numerical: &[f64], tol: f64, label: &str) {
        assert_eq!(analytical.len(), numerical.len(),
            "{}: gradient length mismatch", label);
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
            .as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let phi = 0.6527425084139002;

        let config = make_config(1, 0, 0);
        let params_flat = vec![phi];

        let sparams = SarimaxParams::from_flat(&params_flat, &config).unwrap();
        let ss = StateSpace::new(&config, &sparams, &data, None).unwrap();
        let init = KalmanInit::from_config(&ss, &config, KalmanInit::default_kappa());

        let analytical = score(&data, &ss, &init, &config, &sparams, true, None).unwrap();
        let numerical = numerical_gradient(&data, &config, &params_flat, None);

        assert_gradient_close(&analytical, &numerical, 1e-3, "AR(1)");
    }

    #[test]
    fn test_score_arma11() {
        let fixtures = load_fixtures();
        let case = &fixtures["arma11"];
        let data: Vec<f64> = case["data"]
            .as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let params_vec: Vec<f64> = case["params"]
            .as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();

        let config = make_config(1, 0, 1);
        let params_flat = params_vec[..2].to_vec();

        let sparams = SarimaxParams::from_flat(&params_flat, &config).unwrap();
        let ss = StateSpace::new(&config, &sparams, &data, None).unwrap();
        let init = KalmanInit::from_config(&ss, &config, KalmanInit::default_kappa());

        let analytical = score(&data, &ss, &init, &config, &sparams, true, None).unwrap();
        let numerical = numerical_gradient(&data, &config, &params_flat, None);

        assert_gradient_close(&analytical, &numerical, 1e-3, "ARMA(1,1)");
    }

    #[test]
    fn test_score_arima111() {
        let fixtures = load_fixtures();
        let case = &fixtures["arima111"];
        let data: Vec<f64> = case["data"]
            .as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let params_vec: Vec<f64> = case["params"]
            .as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();

        let config = make_config(1, 1, 1);
        let params_flat = params_vec[..2].to_vec();

        let sparams = SarimaxParams::from_flat(&params_flat, &config).unwrap();
        let ss = StateSpace::new(&config, &sparams, &data, None).unwrap();
        let init = KalmanInit::from_config(&ss, &config, KalmanInit::default_kappa());

        let analytical = score(&data, &ss, &init, &config, &sparams, true, None).unwrap();
        let numerical = numerical_gradient(&data, &config, &params_flat, None);

        assert_gradient_close(&analytical, &numerical, 1e-3, "ARIMA(1,1,1)");
    }

    #[test]
    fn test_score_sarima_100_100_4() {
        let fixtures = load_fixtures();
        let case = &fixtures["sarima_100_100_4"];
        let data: Vec<f64> = case["data"]
            .as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let params_vec: Vec<f64> = case["params"]
            .as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();

        let config = make_seasonal_config(1, 0, 0, 1, 0, 0, 4);
        let params_flat = params_vec.clone();

        let sparams = SarimaxParams::from_flat(&params_flat, &config).unwrap();
        let ss = StateSpace::new(&config, &sparams, &data, None).unwrap();
        let init = KalmanInit::from_config(&ss, &config, KalmanInit::default_kappa());

        let analytical = score(&data, &ss, &init, &config, &sparams, true, None).unwrap();
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
            .as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();

        let config = make_seasonal_config(1, 1, 1, 1, 1, 1, 12);
        // params: [ar, ma, sar, sma] — well within stationary/invertible region
        let params_flat = vec![0.5, 0.3, 0.2, -0.4];

        let sparams = SarimaxParams::from_flat(&params_flat, &config).unwrap();
        let ss = StateSpace::new(&config, &sparams, &data, None).unwrap();
        let init = KalmanInit::from_config(&ss, &config, KalmanInit::default_kappa());

        let analytical = score(&data, &ss, &init, &config, &sparams, true, None).unwrap();
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
            .as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let params_vec: Vec<f64> = case["params"]
            .as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();

        let config = make_seasonal_config(1, 1, 1, 1, 1, 1, 12);
        let params_flat = params_vec.clone();

        let sparams = SarimaxParams::from_flat(&params_flat, &config).unwrap();
        let ss = StateSpace::new(&config, &sparams, &data, None).unwrap();
        let init = KalmanInit::from_config(&ss, &config, KalmanInit::default_kappa());

        let analytical = score(&data, &ss, &init, &config, &sparams, true, None).unwrap();
        let numerical = numerical_gradient(&data, &config, &params_flat, None);

        // Wider tolerance: near-unit-root AR makes numerical gradient unreliable
        assert_gradient_close(&analytical, &numerical, 0.06, "SARIMA(1,1,1)(1,1,1,12) fixture");
    }

    #[test]
    fn test_score_with_exog() {
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let n = data.len();

        let exog_col: Vec<f64> = (0..n).map(|t| (t as f64) * 0.01).collect();
        let exog = vec![exog_col];

        let config = SarimaxConfig {
            order: SarimaxOrder::new(1, 0, 0, 0, 0, 0, 0),
            n_exog: 1,
            trend: Trend::None,
            enforce_stationarity: false,
            enforce_invertibility: false,
            concentrate_scale: true,
            simple_differencing: false,
            measurement_error: false,
        };
        let params_flat = vec![0.5, 0.65];

        let sparams = SarimaxParams::from_flat(&params_flat, &config).unwrap();
        let ss = StateSpace::new(&config, &sparams, &data, Some(&exog[..])).unwrap();
        let init = KalmanInit::from_config(&ss, &config, KalmanInit::default_kappa());

        let analytical = score(&data, &ss, &init, &config, &sparams, true, Some(&exog[..])).unwrap();
        let numerical = numerical_gradient(&data, &config, &params_flat, Some(&exog[..]));

        assert_gradient_close(&analytical, &numerical, 1e-3, "SARIMAX with exog");
    }

    #[test]
    fn test_score_with_trend() {
        let fixtures = load_fixtures();
        let case = &fixtures["ar1"];
        let data: Vec<f64> = case["data"]
            .as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();

        let config = SarimaxConfig {
            order: SarimaxOrder::new(1, 0, 0, 0, 0, 0, 0),
            n_exog: 0,
            trend: Trend::Constant,
            enforce_stationarity: false,
            enforce_invertibility: false,
            concentrate_scale: true,
            simple_differencing: false,
            measurement_error: false,
        };
        let params_flat = vec![0.1, 0.65];

        let sparams = SarimaxParams::from_flat(&params_flat, &config).unwrap();
        let ss = StateSpace::new(&config, &sparams, &data, None).unwrap();
        let init = KalmanInit::from_config(&ss, &config, KalmanInit::default_kappa());

        let analytical = score(&data, &ss, &init, &config, &sparams, true, None).unwrap();
        let numerical = numerical_gradient(&data, &config, &params_flat, None);

        assert_gradient_close(&analytical, &numerical, 1e-3, "AR(1) with trend='c'");
    }
}
