use thiserror::Error;

#[derive(Error, Debug)]
pub enum SarimaxError {
    #[error("parameter length mismatch: expected {expected}, got {got}")]
    ParamLengthMismatch { expected: usize, got: usize },

    #[error("state space construction failed: {0}")]
    StateSpaceError(String),

    #[error("Cholesky decomposition failed: covariance matrix is not positive-definite")]
    CholeskyFailed,

    #[error("optimization failed: {0}")]
    OptimizationFailed(String),

    #[error("non-stationary AR polynomial")]
    NonStationaryAR,

    #[error("non-invertible MA polynomial")]
    NonInvertibleMA,

    #[error("data error: {0}")]
    DataError(String),
}

pub type Result<T> = std::result::Result<T, SarimaxError>;
