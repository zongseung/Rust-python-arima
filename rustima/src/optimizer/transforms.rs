//! Parameter space transformations (constrained ↔ unconstrained).

use crate::error::{Result, SarimaxError};
use crate::params;
use crate::types::SarimaxConfig;

// ---------------------------------------------------------------------------
// Parameter transformations (constrained ↔ unconstrained)
// ---------------------------------------------------------------------------

/// Transform constrained parameters to unconstrained space for optimization.
///
/// Layout: `[trend | exog | ar(p) | ma(q) | sar(P) | sma(Q) | sigma2?]`
pub(super) fn expected_param_len(config: &SarimaxConfig) -> usize {
    config.trend.k_trend()
        + config.n_exog
        + config.order.p
        + config.order.q
        + config.order.pp
        + config.order.qq
        + if config.concentrate_scale { 0 } else { 1 }
}

/// Direction of parameter transformation.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TransformDirection {
    /// Constrained → unconstrained (for optimization)
    ToUnconstrained,
    /// Unconstrained → constrained (recover model params)
    ToConstrained,
}

/// Shared implementation for both transform and untransform.
///
/// Layout: `[trend | exog | ar(p) | ma(q) | sar(P) | sma(Q) | sigma2?]`
///
/// The only difference between the two directions is which `params::` helper
/// is called for each AR/MA block and for variance.
fn transform_params_inner(
    input: &[f64],
    config: &SarimaxConfig,
    direction: TransformDirection,
) -> Result<Vec<f64>> {
    let expected = expected_param_len(config);
    if input.len() != expected {
        return Err(SarimaxError::ParamLengthMismatch {
            expected,
            got: input.len(),
        });
    }

    let kt = config.trend.k_trend();
    let n_exog = config.n_exog;
    let p = config.order.p;
    let q = config.order.q;
    let pp = config.order.pp;
    let qq = config.order.qq;

    let mut out = Vec::with_capacity(input.len());
    let mut i = 0;

    // Trend + exog: pass through
    out.extend_from_slice(&input[i..i + kt + n_exog]);
    i += kt + n_exog;

    // AR coefficients
    if config.enforce_stationarity && p > 0 {
        out.extend(match direction {
            TransformDirection::ToUnconstrained => params::unconstrain_stationary(&input[i..i + p]),
            TransformDirection::ToConstrained => params::constrain_stationary(&input[i..i + p]),
        });
    } else {
        out.extend_from_slice(&input[i..i + p]);
    }
    i += p;

    // MA coefficients
    if config.enforce_invertibility && q > 0 {
        out.extend(match direction {
            TransformDirection::ToUnconstrained => params::unconstrain_invertible(&input[i..i + q]),
            TransformDirection::ToConstrained => params::constrain_invertible(&input[i..i + q]),
        });
    } else {
        out.extend_from_slice(&input[i..i + q]);
    }
    i += q;

    // Seasonal AR
    if config.enforce_stationarity && pp > 0 {
        out.extend(match direction {
            TransformDirection::ToUnconstrained => {
                params::unconstrain_stationary(&input[i..i + pp])
            }
            TransformDirection::ToConstrained => params::constrain_stationary(&input[i..i + pp]),
        });
    } else {
        out.extend_from_slice(&input[i..i + pp]);
    }
    i += pp;

    // Seasonal MA
    if config.enforce_invertibility && qq > 0 {
        out.extend(match direction {
            TransformDirection::ToUnconstrained => {
                params::unconstrain_invertible(&input[i..i + qq])
            }
            TransformDirection::ToConstrained => params::constrain_invertible(&input[i..i + qq]),
        });
    } else {
        out.extend_from_slice(&input[i..i + qq]);
    }
    i += qq;

    // sigma2
    if !config.concentrate_scale && i < input.len() {
        out.push(match direction {
            TransformDirection::ToUnconstrained => params::unconstrain_variance(input[i])?,
            TransformDirection::ToConstrained => params::constrain_variance(input[i]),
        });
    }

    Ok(out)
}

/// Transform constrained parameters to unconstrained space for optimization.
///
/// Layout: `[trend | exog | ar(p) | ma(q) | sar(P) | sma(Q) | sigma2?]`
pub fn untransform_params(constrained: &[f64], config: &SarimaxConfig) -> Result<Vec<f64>> {
    transform_params_inner(constrained, config, TransformDirection::ToUnconstrained)
}

/// Transform unconstrained parameters back to constrained space.
pub fn transform_params(unconstrained: &[f64], config: &SarimaxConfig) -> Result<Vec<f64>> {
    transform_params_inner(unconstrained, config, TransformDirection::ToConstrained)
}

/// Apply the chain rule: grad_unconstrained = J' · grad_constrained.
///
/// Delegates to [`crate::params::transform_jacobian_t_vec`].
pub(super) fn apply_transform_jacobian(
    score_constrained: &[f64],
    unconstrained: &[f64],
    config: &SarimaxConfig,
) -> std::result::Result<Vec<f64>, String> {
    crate::params::transform_jacobian_t_vec(unconstrained, config, score_constrained)
        .map_err(|e| e.to_string())
}
