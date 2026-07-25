//! Shared Kalman pipeline helpers.
//!
//! Centralises the repeated pattern of: params -> StateSpace -> KalmanInit -> Kalman filter.

use crate::css::apply_differencing;
use crate::error::Result;
use crate::initialization::KalmanInit;
use crate::kalman;
use crate::params::SarimaxParams;
use crate::state_space::StateSpace;
use crate::types::SarimaxConfig;

/// Pre-difference endog and exog when `simple_differencing=true`.
///
/// Returns `(eff_endog, eff_exog)`:
/// - `eff_endog`: differenced series of length n - d - s*D
/// - `eff_exog`: exog columns after the same differencing operations
///
/// Returns `Err` if any differenced exog column cannot be aligned to `eff_endog`.
pub(crate) fn prepare_endog(
    endog: &[f64],
    config: &SarimaxConfig,
    exog: Option<&[Vec<f64>]>,
) -> Result<(Vec<f64>, Option<Vec<Vec<f64>>>)> {
    if !config.simple_differencing {
        return Ok((endog.to_vec(), None));
    }

    let eff = apply_differencing(endog, config);

    let eff_exog = match exog {
        Some(cols) => {
            let mut diffed = Vec::with_capacity(cols.len());
            for (j, col) in cols.iter().enumerate() {
                let xj = apply_differencing(col, config);
                if xj.len() != eff.len() {
                    return Err(crate::error::SarimaxError::InvalidInput(format!(
                        "exog column {} has {} rows after simple_differencing, but y has {} observations",
                        j, xj.len(), eff.len()
                    )));
                }
                diffed.push(xj);
            }
            Some(diffed)
        }
        None => None,
    };

    Ok((eff, eff_exog))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SarimaxOrder, Trend};

    fn config(order: SarimaxOrder) -> SarimaxConfig {
        SarimaxConfig {
            order,
            n_exog: 1,
            enforce_stationarity: false,
            enforce_invertibility: false,
            concentrate_scale: false,
            trend: Trend::None,
            simple_differencing: true,
        }
    }

    #[test]
    fn prepare_endog_differences_exog_with_y() {
        let cfg = config(SarimaxOrder {
            p: 0,
            d: 1,
            q: 0,
            pp: 0,
            dd: 0,
            qq: 0,
            s: 0,
        });
        let y = vec![1.0, 3.0, 6.0, 10.0];
        let x = vec![vec![2.0, 5.0, 9.0, 14.0]];

        let (eff_y, eff_x) = prepare_endog(&y, &cfg, Some(&x)).unwrap();

        assert_eq!(eff_y, vec![2.0, 3.0, 4.0]);
        assert_eq!(eff_x.unwrap()[0], vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn prepare_endog_seasonal_differences_exog_with_y() {
        let cfg = config(SarimaxOrder {
            p: 0,
            d: 0,
            q: 0,
            pp: 0,
            dd: 1,
            qq: 0,
            s: 2,
        });
        let y = vec![1.0, 2.0, 4.0, 7.0, 11.0];
        let x = vec![vec![3.0, 5.0, 10.0, 18.0, 31.0]];

        let (eff_y, eff_x) = prepare_endog(&y, &cfg, Some(&x)).unwrap();

        assert_eq!(eff_y, vec![3.0, 5.0, 7.0]);
        assert_eq!(eff_x.unwrap()[0], vec![7.0, 13.0, 21.0]);
    }
}

/// Run Kalman filter loglikelihood from a `SarimaxParams` struct.
#[inline]
pub(crate) fn kalman_eval(
    endog: &[f64],
    params: &SarimaxParams,
    config: &SarimaxConfig,
    exog: Option<&[Vec<f64>]>,
) -> Result<kalman::KalmanOutput> {
    if config.simple_differencing {
        let (eff_endog, eff_exog_owned) = prepare_endog(endog, config, exog)?;
        let ss = StateSpace::new(config, params, &eff_endog, eff_exog_owned.as_deref())?;
        let init = KalmanInit::from_config_default(&ss, config);
        kalman::kalman_loglike(&eff_endog, &ss, &init, config.concentrate_scale)
    } else {
        let ss = StateSpace::new(config, params, endog, exog)?;
        let init = KalmanInit::from_config_default(&ss, config);
        kalman::kalman_loglike(endog, &ss, &init, config.concentrate_scale)
    }
}

/// Run Kalman filter loglikelihood from constrained (flat) parameters.
#[inline]
pub(crate) fn kalman_eval_constrained(
    endog: &[f64],
    constrained: &[f64],
    config: &SarimaxConfig,
    exog: Option<&[Vec<f64>]>,
) -> Result<kalman::KalmanOutput> {
    let sparams = SarimaxParams::from_flat(constrained, config)?;
    kalman_eval(endog, &sparams, config, exog)
}

/// Run Kalman filter loglikelihood from unconstrained parameters (transforms first).
#[inline]
pub(crate) fn kalman_eval_unconstrained(
    endog: &[f64],
    unconstrained: &[f64],
    config: &SarimaxConfig,
    exog: Option<&[Vec<f64>]>,
) -> Result<kalman::KalmanOutput> {
    let constrained = crate::optimizer::transform_params(unconstrained, config)?;
    kalman_eval_constrained(endog, &constrained, config, exog)
}

/// Run full Kalman filter (returning state history) from a `SarimaxParams` struct.
#[inline]
pub(crate) fn kalman_filter_full(
    endog: &[f64],
    params: &SarimaxParams,
    config: &SarimaxConfig,
    exog: Option<&[Vec<f64>]>,
) -> Result<kalman::KalmanFilterOutput> {
    if config.simple_differencing {
        let (eff_endog, eff_exog_owned) = prepare_endog(endog, config, exog)?;
        let ss = StateSpace::new(config, params, &eff_endog, eff_exog_owned.as_deref())?;
        let init = KalmanInit::from_config_default(&ss, config);
        kalman::kalman_filter(&eff_endog, &ss, &init, config.concentrate_scale)
    } else {
        let ss = StateSpace::new(config, params, endog, exog)?;
        let init = KalmanInit::from_config_default(&ss, config);
        kalman::kalman_filter(endog, &ss, &init, config.concentrate_scale)
    }
}
