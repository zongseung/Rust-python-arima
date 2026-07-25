/// SARIMAX model order specification.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SarimaxOrder {
    pub p: usize,  // AR order
    pub d: usize,  // differencing order
    pub q: usize,  // MA order
    pub pp: usize, // seasonal AR order (P)
    pub dd: usize, // seasonal differencing order (D)
    pub qq: usize, // seasonal MA order (Q)
    pub s: usize,  // seasonal period
}

impl SarimaxOrder {
    pub fn new(p: usize, d: usize, q: usize, pp: usize, dd: usize, qq: usize, s: usize) -> Self {
        Self {
            p,
            d,
            q,
            pp,
            dd,
            qq,
            s,
        }
    }

    /// Extended AR order: p + s*P
    pub fn k_ar(&self) -> usize {
        self.p + self.s * self.pp
    }

    /// Extended MA order: q + s*Q
    pub fn k_ma(&self) -> usize {
        self.q + self.s * self.qq
    }

    /// State space ARMA dimension: max(k_ar, k_ma + 1)
    pub fn k_order(&self) -> usize {
        std::cmp::max(self.k_ar(), self.k_ma() + 1)
    }

    /// Differencing state dimension: d + s*D
    pub fn k_states_diff(&self) -> usize {
        self.d + self.s * self.dd
    }

    /// Total state dimension
    pub fn k_states(&self) -> usize {
        self.k_order() + self.k_states_diff()
    }
}

/// Trend specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trend {
    None,     // 'n': k_trend = 0
    Constant, // 'c': k_trend = 1
    Linear,   // 't': k_trend = 1
    Both,     // 'ct': k_trend = 2
}

impl Trend {
    pub fn k_trend(&self) -> usize {
        match self {
            Trend::None => 0,
            Trend::Constant | Trend::Linear => 1,
            Trend::Both => 2,
        }
    }

    /// Value of the ti-th trend basis column at 0-based time t:
    /// 'c' → [1], 't' → [t], 'ct' → [1, t].
    ///
    /// Single source for the basis convention — state_space (intercept),
    /// score (d(intercept)/d(coeff)) and forecast (future intercept) must
    /// all agree on it or the gradient silently diverges from the loglike.
    pub fn basis(&self, ti: usize, t: usize) -> f64 {
        match (self, ti) {
            (Trend::Constant, 0) | (Trend::Both, 0) => 1.0,
            (Trend::Linear, 0) | (Trend::Both, 1) => t as f64,
            _ => 0.0,
        }
    }

    /// Deterministic trend contribution at time t: coeffs · basis(t).
    pub fn intercept(&self, coeffs: &[f64], t: usize) -> f64 {
        (0..self.k_trend())
            .map(|ti| coeffs.get(ti).copied().unwrap_or(0.0) * self.basis(ti, t))
            .sum()
    }

    pub fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s {
            "n" => Ok(Trend::None),
            "c" => Ok(Trend::Constant),
            "t" => Ok(Trend::Linear),
            "ct" | "tc" => Ok(Trend::Both),
            _ => Err(format!(
                "unknown trend '{}', expected one of: 'n', 'c', 't', 'ct'",
                s
            )),
        }
    }
}

/// Model configuration.
#[derive(Debug, Clone)]
pub struct SarimaxConfig {
    pub order: SarimaxOrder,
    pub n_exog: usize,
    pub trend: Trend,
    pub enforce_stationarity: bool,
    pub enforce_invertibility: bool,
    pub concentrate_scale: bool,
    pub simple_differencing: bool,
}

impl SarimaxConfig {
    /// Effective diffuse-state offset.
    ///
    /// Returns 0 when `simple_differencing=true` (diff states already removed
    /// from the state space), otherwise `k_states_diff()`.
    #[inline]
    pub fn effective_sd(&self) -> usize {
        if self.simple_differencing { 0 } else { self.order.k_states_diff() }
    }
}

impl Default for SarimaxConfig {
    fn default() -> Self {
        Self {
            order: SarimaxOrder::new(1, 0, 0, 0, 0, 0, 0),
            n_exog: 0,
            trend: Trend::None,
            enforce_stationarity: false,
            enforce_invertibility: false,
            concentrate_scale: false,
            simple_differencing: false,
        }
    }
}

/// Fit result returned by the optimizer.
#[derive(Debug, Clone)]
pub struct FitResult {
    pub params: Vec<f64>,
    pub loglike: f64,
    pub scale: f64,
    pub n_obs: usize,
    pub n_params: usize,
    /// Total optimizer work units consumed.
    ///
    /// - **L-BFGS / Nelder-Mead** (argmin): solver iterations (`state.get_iter()`).
    /// - **L-BFGS-B** (Fortran wrapper): objective function evaluations, because the
    ///   crate does not expose a native iteration counter. For L-BFGS-B each iteration
    ///   typically requires 1–3 function+gradient evaluations (line search), so
    ///   `n_iter` ≈ 1–3× actual Fortran iterations.
    /// - **Multi-start methods** (`lbfgsb-multi`, `lbfgs`): cumulative sum across all
    ///   sub-runs (initial + restarts + NM refinement), capped by `maxiter`.
    /// - **`maxiter=0`**: always returns `n_iter=0` via the early-return guard.
    pub n_iter: u64,
    pub converged: bool,
    pub method: String,
    pub aic: f64,
    pub bic: f64,
    /// Runtime warnings (e.g., near-cancellation detection).
    pub warnings: Vec<String>,
}

impl FitResult {
    /// Compute AIC and BIC from loglike, n_params, n_obs.
    pub fn with_information_criteria(mut self) -> Self {
        let k = self.n_params as f64;
        let n = self.n_obs as f64;
        self.aic = -2.0 * self.loglike + 2.0 * k;
        self.bic = -2.0 * self.loglike + k * n.ln();
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sarima_111_111_12_k_states() {
        // SARIMA(1,1,1)(1,1,1,12)
        // k_ar = 1 + 12*1 = 13
        // k_ma = 1 + 12*1 = 13
        // k_order = max(13, 14) = 14
        // k_states_diff = 1 + 12*1 = 13
        // k_states = 14 + 13 = 27
        let order = SarimaxOrder::new(1, 1, 1, 1, 1, 1, 12);
        assert_eq!(order.k_ar(), 13);
        assert_eq!(order.k_ma(), 13);
        assert_eq!(order.k_order(), 14);
        assert_eq!(order.k_states_diff(), 13);
        assert_eq!(order.k_states(), 27);
    }

    #[test]
    fn test_arima_110_k_states() {
        // ARIMA(1,1,0): k_ar=1, k_ma=0, k_order=max(1,1)=1, k_diff=1, k_states=2
        let order = SarimaxOrder::new(1, 1, 0, 0, 0, 0, 0);
        assert_eq!(order.k_ar(), 1);
        assert_eq!(order.k_ma(), 0);
        assert_eq!(order.k_order(), 1);
        assert_eq!(order.k_states_diff(), 1);
        assert_eq!(order.k_states(), 2);
    }

    #[test]
    fn test_arma_11_k_states() {
        // ARMA(1,1): k_ar=1, k_ma=1, k_order=max(1,2)=2, k_diff=0, k_states=2
        let order = SarimaxOrder::new(1, 0, 1, 0, 0, 0, 0);
        assert_eq!(order.k_ar(), 1);
        assert_eq!(order.k_ma(), 1);
        assert_eq!(order.k_order(), 2);
        assert_eq!(order.k_states_diff(), 0);
        assert_eq!(order.k_states(), 2);
    }

    #[test]
    fn test_ar2_k_states() {
        // AR(2): k_ar=2, k_ma=0, k_order=max(2,1)=2, k_diff=0, k_states=2
        let order = SarimaxOrder::new(2, 0, 0, 0, 0, 0, 0);
        assert_eq!(order.k_order(), 2);
        assert_eq!(order.k_states(), 2);
    }

    #[test]
    fn test_trend_from_str() {
        assert_eq!(Trend::from_str("n").unwrap(), Trend::None);
        assert_eq!(Trend::from_str("c").unwrap(), Trend::Constant);
        assert_eq!(Trend::from_str("t").unwrap(), Trend::Linear);
        assert_eq!(Trend::from_str("ct").unwrap(), Trend::Both);
        assert_eq!(Trend::from_str("tc").unwrap(), Trend::Both);
        assert!(Trend::from_str("unknown").is_err());
        assert!(Trend::from_str("xyz").is_err());
    }

    #[test]
    fn test_trend_k_trend() {
        assert_eq!(Trend::None.k_trend(), 0);
        assert_eq!(Trend::Constant.k_trend(), 1);
        assert_eq!(Trend::Linear.k_trend(), 1);
        assert_eq!(Trend::Both.k_trend(), 2);
    }

    #[test]
    fn test_default_config() {
        let config = SarimaxConfig::default();
        assert_eq!(config.order.p, 1);
        assert_eq!(config.order.d, 0);
        assert_eq!(config.order.q, 0);
        assert!(!config.concentrate_scale);
        assert!(!config.enforce_stationarity);
        assert!(!config.enforce_invertibility);
        assert!(!config.simple_differencing);
        assert_eq!(config.n_exog, 0);
        assert_eq!(config.trend, Trend::None);
    }
}
