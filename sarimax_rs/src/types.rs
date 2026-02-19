/// SARIMAX model order specification.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SarimaxOrder {
    pub p: usize,   // AR order
    pub d: usize,   // differencing order
    pub q: usize,   // MA order
    pub pp: usize,  // seasonal AR order (P)
    pub dd: usize,  // seasonal differencing order (D)
    pub qq: usize,  // seasonal MA order (Q)
    pub s: usize,   // seasonal period
}

impl SarimaxOrder {
    pub fn new(p: usize, d: usize, q: usize, pp: usize, dd: usize, qq: usize, s: usize) -> Self {
        Self { p, d, q, pp, dd, qq, s }
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

    pub fn from_str(s: &str) -> Self {
        match s {
            "c" => Trend::Constant,
            "t" => Trend::Linear,
            "ct" | "tc" => Trend::Both,
            _ => Trend::None,
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
    pub measurement_error: bool,
}

impl Default for SarimaxConfig {
    fn default() -> Self {
        Self {
            order: SarimaxOrder::new(1, 0, 0, 0, 0, 0, 0),
            n_exog: 0,
            trend: Trend::None,
            enforce_stationarity: false,
            enforce_invertibility: false,
            concentrate_scale: true,
            simple_differencing: false,
            measurement_error: false,
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
    pub n_iter: u64,
    pub converged: bool,
    pub method: String,
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
        assert_eq!(Trend::from_str("n"), Trend::None);
        assert_eq!(Trend::from_str("c"), Trend::Constant);
        assert_eq!(Trend::from_str("t"), Trend::Linear);
        assert_eq!(Trend::from_str("ct"), Trend::Both);
        assert_eq!(Trend::from_str("tc"), Trend::Both);
        assert_eq!(Trend::from_str("unknown"), Trend::None);
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
        assert!(config.concentrate_scale);
        assert!(!config.enforce_stationarity);
    }
}
