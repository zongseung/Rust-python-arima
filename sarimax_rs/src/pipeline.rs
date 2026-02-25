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

/// Pre-difference endog (and trim exog) when `simple_differencing=true`.
///
/// Returns `(eff_endog, eff_exog)`:
/// - `eff_endog`: differenced series of length n - d - s*D
/// - `eff_exog`: exog columns trimmed to the same length (drop first `n_drop` rows)
///
/// Returns `Err` if any exog column is shorter than `n_drop`.
pub(crate) fn prepare_endog<'a>(
    endog: &[f64],
    config: &SarimaxConfig,
    exog: Option<&'a [Vec<f64>]>,
) -> Result<(Vec<f64>, Option<Vec<Vec<f64>>>)> {
    if !config.simple_differencing {
        return Ok((endog.to_vec(), None));
    }

    let eff = apply_differencing(endog, config);
    let n_drop = endog.len() - eff.len();

    let eff_exog = match exog {
        Some(cols) => {
            let mut trimmed = Vec::with_capacity(cols.len());
            for (j, col) in cols.iter().enumerate() {
                if col.len() < n_drop {
                    return Err(crate::error::SarimaxError::InvalidInput(format!(
                        "exog column {} has {} rows, but simple_differencing requires dropping {} rows",
                        j, col.len(), n_drop
                    )));
                }
                trimmed.push(col[n_drop..].to_vec());
            }
            Some(trimmed)
        }
        None => None,
    };

    Ok((eff, eff_exog))
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

/// Run full Kalman filter (returning state history) from constrained (flat) parameters.
#[inline]
pub(crate) fn kalman_filter_constrained(
    endog: &[f64],
    constrained: &[f64],
    config: &SarimaxConfig,
    exog: Option<&[Vec<f64>]>,
) -> Result<kalman::KalmanFilterOutput> {
    let sparams = SarimaxParams::from_flat(constrained, config)?;
    kalman_filter_full(endog, &sparams, config, exog)
}
