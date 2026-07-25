"""statsmodels-compatible SARIMAX model backed by Rust engine (rustima)."""

import math

import numpy as np
from . import rustima


# ---------------------------------------------------------------------------
# Parameter naming helpers
# ---------------------------------------------------------------------------

def _hqic(llf, n_params, n_obs):
    """Hannan-Quinn information criterion; NaN when n_obs <= 1."""
    if n_obs <= 1:
        return float("nan")
    return -2.0 * llf + 2.0 * n_params * math.log(math.log(n_obs))


def _generate_param_names(order, seasonal_order, n_exog=0, concentrate_scale=False, trend="n"):
    """Generate statsmodels-style parameter names from model specification.

    Layout: [trend(kt) | exog(k) | ar(p) | ma(q) | sar(P) | sma(Q) | sigma2?]

    Parameters
    ----------
    order : tuple (p, d, q)
    seasonal_order : tuple (P, D, Q, s)
    n_exog : int
    concentrate_scale : bool
    trend : str

    Returns
    -------
    list[str]
    """
    p, _d, q = order
    P, _D, Q, s = seasonal_order
    names = []

    # Trend
    if trend in ("c", "ct"):
        names.append("intercept")
    if trend in ("t", "ct"):
        names.append("drift")

    # Exogenous
    for i in range(1, n_exog + 1):
        names.append(f"x{i}")

    # AR
    for i in range(1, p + 1):
        names.append(f"ar.L{i}")

    # MA
    for i in range(1, q + 1):
        names.append(f"ma.L{i}")

    # Seasonal AR
    for i in range(1, P + 1):
        names.append(f"ar.S.L{i * s}")

    # Seasonal MA
    for i in range(1, Q + 1):
        names.append(f"ma.S.L{i * s}")

    # sigma2 (only when not concentrated)
    if not concentrate_scale:
        names.append("sigma2")

    return names


# ---------------------------------------------------------------------------
# Numerical inference helpers
# ---------------------------------------------------------------------------

def _norm_ppf(p):
    """Inverse normal CDF (percent point function).

    Uses scipy if available, otherwise falls back to a rational
    approximation (Abramowitz & Stegun 26.2.23, |error| < 4.5e-4).
    """
    try:
        from scipy.stats import norm
        return norm.ppf(p)
    except ImportError:
        pass

    if p < 0.5:
        return -_norm_ppf(1.0 - p)
    if p == 0.5:
        return 0.0

    t = np.sqrt(-2.0 * np.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t**2) / (1.0 + d1 * t + d2 * t**2 + d3 * t**3)


def _norm_cdf(x):
    """Standard normal CDF.

    Uses scipy if available, otherwise math.erf.
    """
    try:
        from scipy.stats import norm
        return norm.cdf(x)
    except ImportError:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _inference_nan_dict(k, status, message, prefix=""):
    """Construct a failure dict with NaN arrays for inference."""
    nan = np.full(k, np.nan)
    if prefix:
        return {
            f"{prefix}_std_err": nan.copy(), f"{prefix}_z": nan.copy(),
            f"{prefix}_p_value": nan.copy(), f"{prefix}_ci_lower": nan.copy(),
            f"{prefix}_ci_upper": nan.copy(),
            f"inference_status_{prefix}": status,
            f"inference_message_{prefix}": message,
        }
    return {
        "std_err": nan.copy(), "z": nan.copy(), "p_value": nan.copy(),
        "ci_lower": nan.copy(), "ci_upper": nan.copy(),
        "inference_status": status, "inference_message": message,
    }


_VALID_INFERENCE_MODES = ("none", "hessian", "statsmodels", "both", "rust_hessian", "opg")


def _validate_inference_mode(mode):
    """Validate inference mode string against the allowed set.

    Raises ValueError if mode is not in _VALID_INFERENCE_MODES.
    """
    if mode not in _VALID_INFERENCE_MODES:
        raise ValueError(
            f"inference must be one of {_VALID_INFERENCE_MODES}, got {mode!r}"
        )
    if mode == "rust_hessian":
        import warnings
        warnings.warn(
            "'rust_hessian' is deprecated; use inference='hessian' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    return mode


def _resolve_inference_mode(inference=None, include_inference=None):
    """Resolve inference mode from new enum or legacy bool parameter.

    Parameters
    ----------
    inference : str or None
        New enum: "none", "hessian", "statsmodels", "both".
    include_inference : bool or None
        Legacy parameter (deprecated).

    Returns
    -------
    str : resolved mode ("none", "hessian", "statsmodels", "both")
    """
    if inference is not None and include_inference is not None:
        import warnings
        warnings.warn(
            "Both 'inference' and 'include_inference' specified; "
            "'inference' takes precedence.",
            DeprecationWarning,
            stacklevel=3,
        )
        return _validate_inference_mode(inference)

    if inference is not None:
        return _validate_inference_mode(inference)

    if include_inference is not None:
        import warnings
        warnings.warn(
            "include_inference is deprecated; use inference='hessian' or "
            "inference='none' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return "hessian" if include_inference else "none"

    # Default
    return "none"


def _compute_statsmodels_inference(endog, order, seasonal_order, alpha=0.05,
                                   exog=None, n_params_rs=None,
                                   enforce_stationarity=True,
                                   enforce_invertibility=True,
                                   trend="n",
                                   simple_differencing=False):
    """Compute inference statistics using statsmodels as reference.

    Parameters
    ----------
    endog : np.ndarray
    order : tuple (p, d, q)
    seasonal_order : tuple (P, D, Q, s)
    alpha : float
    exog : np.ndarray or None
    n_params_rs : int or None
        Number of non-sigma2 params in rustima (for alignment).
    enforce_stationarity : bool
        Pass through to statsmodels SARIMAX.
    enforce_invertibility : bool
        Pass through to statsmodels SARIMAX.
    trend : str
        Trend specification ('n', 'c', 't', 'ct').
    simple_differencing : bool
        Pass through to statsmodels SARIMAX.

    Returns
    -------
    dict with sm_ prefixed keys, or failed dict on error.
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        return _inference_nan_dict(n_params_rs or 0, "failed", "statsmodels not installed", prefix="sm")

    try:
        model_sm = SARIMAX(
            endog, order=order, seasonal_order=seasonal_order,
            exog=exog,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
            trend=trend,
            simple_differencing=simple_differencing,
        )
        res_sm = model_sm.fit(disp=False)

        # statsmodels includes sigma2 as last param; align to rustima count
        k = n_params_rs if n_params_rs is not None else len(res_sm.params) - 1
        ci = res_sm.conf_int(alpha=alpha)

        # Shape guard: statsmodels may return fewer params than expected
        if len(res_sm.bse) < k:
            return _inference_nan_dict(
                n_params_rs or 0, "failed",
                f"statsmodels returned {len(res_sm.bse)} params but expected >= {k}",
                prefix="sm",
            )

        # Check convergence status
        sm_converged = getattr(res_sm, "mle_retvals", {}).get("converged", True)
        if sm_converged:
            status = "ok"
            message = None
        else:
            status = "warning"
            message = "statsmodels optimizer did not converge"

        return dict(
            sm_std_err=np.array(res_sm.bse[:k]),
            sm_z=np.array(res_sm.zvalues[:k]),
            sm_p_value=np.array(res_sm.pvalues[:k]),
            sm_ci_lower=np.array(ci[:k, 0]),
            sm_ci_upper=np.array(ci[:k, 1]),
            inference_status_sm=status,
            inference_message_sm=message,
        )
    except (ValueError, RuntimeError, ArithmeticError, OverflowError,
            np.linalg.LinAlgError) as e:
        return _inference_nan_dict(n_params_rs or 0, "failed", str(e), prefix="sm")
    except Exception as e:
        # Unexpected errors (FFI panic, OOM, etc.) propagate for debugging
        raise


def _compute_rust_inference(endog, order, seasonal_order, params, method, alpha=0.05,
                            exog=None, enforce_stationarity=True,
                            enforce_invertibility=True, trend="n",
                            simple_differencing=False):
    """Compute inference using the Rust sarimax_inference function.

    Parameters
    ----------
    endog : np.ndarray
    order : tuple (p, d, q)
    seasonal_order : tuple (P, D, Q, s)
    params : np.ndarray
    method : str
        "hessian" or "opg".
    alpha : float
    exog : np.ndarray or None
    enforce_stationarity : bool
    enforce_invertibility : bool
    trend : str
    simple_differencing : bool

    Returns
    -------
    dict with keys: std_err, z, p_value, ci_lower, ci_upper,
                    cov_params, inference_status, inference_message
    """
    k = len(params)
    nan_arr = np.full(k, np.nan)

    try:
        kwargs = dict(
            method=method, alpha=alpha,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
            trend=trend,
            simple_differencing=simple_differencing,
        )
        if exog is not None:
            kwargs["exog"] = exog

        result = rustima.sarimax_inference(
            endog, order, seasonal_order,
            np.array(params, dtype=np.float64),
            **kwargs,
        )

        return dict(
            std_err=np.array(result["std_err"]),
            z=np.array(result["z_stat"]),
            p_value=np.array(result["p_value"]),
            ci_lower=np.array(result["ci_lower"]),
            ci_upper=np.array(result["ci_upper"]),
            cov_params=np.array(result["cov_params"]).reshape(k, k) if k > 0 else np.array([]),
            inference_status=result["status"],
            inference_message=result["message"],
        )
    except (ValueError, RuntimeError, ArithmeticError, OverflowError) as e:
        return dict(
            std_err=nan_arr.copy(),
            z=nan_arr.copy(),
            p_value=nan_arr.copy(),
            ci_lower=nan_arr.copy(),
            ci_upper=nan_arr.copy(),
            cov_params=np.full((k, k), np.nan) if k > 0 else np.array([]),
            inference_status="failed",
            inference_message=str(e),
        )
    except Exception as e:
        # Unexpected errors (FFI panic, OOM, etc.) propagate for debugging
        raise


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------

class SARIMAXModel:
    """SARIMAX model with statsmodels-compatible API.

    Parameters
    ----------
    endog : array_like
        Endogenous (observed) time series.
    order : tuple (p, d, q)
        ARIMA order.
    seasonal_order : tuple (P, D, Q, s)
        Seasonal ARIMA order.
    exog : array_like, optional
        Exogenous variables, shape (n_obs, n_exog).
    trend : str
        Trend specification: 'n' (none), 'c' (constant), 't' (linear),
        'ct' (constant + linear).
    enforce_stationarity : bool
        Enforce AR stationarity constraints during fitting.
    enforce_invertibility : bool
        Enforce MA invertibility constraints during fitting.
    simple_differencing : bool
        If True, pre-difference the series externally before Kalman filtering.
        Reduces state dimension (faster for high-order seasonal models) at the
        cost of losing the first d + s*D observations from the likelihood.
        AIC/BIC are computed with n_obs = n - d - s*D (R-style convention).
    """

    def __init__(
        self,
        endog,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        exog=None,
        trend="n",
        enforce_stationarity=True,
        enforce_invertibility=True,
        simple_differencing=False,
    ):
        # --- endog validation ---
        # np.asarray silently DROPS the mask of a masked array and treats
        # masked entries as raw data — reject rather than compute on garbage.
        if np.ma.isMaskedArray(endog) or np.ma.isMaskedArray(exog):
            raise ValueError(
                "masked arrays are not supported; fill or drop masked values first"
            )
        self.endog = np.asarray(endog, dtype=np.float64)
        if self.endog.ndim != 1:
            raise ValueError(f"endog must be 1-dimensional, got ndim={self.endog.ndim}")
        if len(self.endog) < 10:
            raise ValueError(f"endog too short: {len(self.endog)} < 10")
        if not np.isfinite(self.endog).all():
            raise ValueError("endog contains NaN or Inf values")

        # --- order validation ---
        if len(order) != 3:
            raise ValueError(f"order must have 3 elements (p,d,q), got {len(order)}")
        p, d, q = order
        if not all(isinstance(v, (int, np.integer)) and v >= 0 for v in (p, d, q)):
            raise ValueError(f"order (p,d,q) must be non-negative integers, got {order}")
        if d > 2:
            raise ValueError(f"Non-seasonal differencing d must be <= 2, got d={d}")

        # --- seasonal_order validation ---
        if len(seasonal_order) != 4:
            raise ValueError(
                f"seasonal_order must have 4 elements (P,D,Q,s), got {len(seasonal_order)}"
            )
        P, D, Q, s = seasonal_order
        if not all(isinstance(v, (int, np.integer)) and v >= 0 for v in (P, D, Q, s)):
            raise ValueError(
                f"seasonal_order (P,D,Q,s) must be non-negative integers, got {seasonal_order}"
            )
        if D > 1:
            raise ValueError(f"Seasonal differencing D must be 0 or 1, got D={D}")
        if s < 0:
            raise ValueError(f"Seasonal period s must be non-negative, got s={s}")
        if s == 1:
            raise ValueError("Seasonal period s=1 is equivalent to s=0 (non-seasonal)")

        # --- trend validation ---
        if trend not in ("n", "c", "t", "ct"):
            raise ValueError(f"trend must be one of 'n','c','t','ct', got {trend!r}")

        # --- simple_differencing validation ---
        if not isinstance(simple_differencing, (bool, np.bool_)):
            raise TypeError(
                f"simple_differencing must be bool, got {type(simple_differencing).__name__}"
            )

        self.order = order
        self.seasonal_order = seasonal_order

        # --- exog validation ---
        if exog is not None:
            self.exog = np.asarray(exog, dtype=np.float64)
            if self.exog.ndim == 1:
                self.exog = self.exog.reshape(-1, 1)
            if self.exog.shape[0] != len(self.endog):
                raise ValueError(
                    f"exog row count {self.exog.shape[0]} != endog length {len(self.endog)}"
                )
            if not np.isfinite(self.exog).all():
                raise ValueError("exog contains NaN or Inf values")
        else:
            self.exog = None

        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.simple_differencing = simple_differencing
        self._fit_result = None

    @property
    def nobs(self):
        return len(self.endog)

    @property
    def n_exog(self):
        if self.exog is None:
            return 0
        return self.exog.shape[1] if self.exog.ndim == 2 else 1

    def _model_kwargs(self, **extra):
        """Build common kwargs dict for Rust calls that need model config.

        Includes: enforce_stationarity, enforce_invertibility, trend,
        simple_differencing, and exog (if present).
        """
        kw = dict(
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
            trend=self.trend,
            simple_differencing=self.simple_differencing,
        )
        if self.exog is not None:
            kw["exog"] = self.exog
        kw.update(extra)
        return kw

    def fit(self, method=None, maxiter=None, start_params=None):
        """Fit the SARIMAX model via MLE.

        Returns
        -------
        SARIMAXResult
        """
        if start_params is not None:
            start_params = np.asarray(start_params, dtype=np.float64)
        kwargs = self._model_kwargs(
            start_params=start_params,
            method=method,
            maxiter=maxiter,
        )

        result_dict = rustima.sarimax_fit(
            self.endog,
            self.order,
            self.seasonal_order,
            **kwargs,
        )
        self._fit_result = SARIMAXResult(self, result_dict)
        return self._fit_result

    def filter(self, params):
        """Construct a result at the given parameters WITHOUT fitting.

        Runs a single Kalman-filter pass to evaluate the log-likelihood at
        ``params`` and returns a :class:`SARIMAXResult` bound to this model.
        No optimization is performed (statsmodels ``.filter(params)``
        semantics). This is the building block for walk-forward rolling
        (:meth:`SARIMAXResult.extend`) and for reconstructing a result from
        serialized parameters without re-estimation.

        Parameters
        ----------
        params : array_like
            Full parameter vector in statsmodels layout
            ``[trend | exog | ar | ma | sar | sma | sigma2]``.

        Returns
        -------
        SARIMAXResult
            ``result.method == "filter"``, ``converged == True``.
        """
        params = np.ascontiguousarray(np.asarray(params, dtype=np.float64).ravel())
        expected_names = _generate_param_names(
            self.order, self.seasonal_order, self.n_exog, trend=self.trend
        )
        if len(params) != len(expected_names):
            raise ValueError(
                f"params length {len(params)} != expected {len(expected_names)} "
                f"for this specification: {expected_names}"
            )

        llf = float(
            rustima.sarimax_loglike(
                self.endog,
                self.order,
                self.seasonal_order,
                params,
                **self._model_kwargs(),
            )
        )

        k = len(params)
        _p, d, _q = self.order
        _P, D, _Q, s = self.seasonal_order
        n_eff = self.nobs - ((d + s * D) if self.simple_differencing else 0)
        sigma2_idx = expected_names.index("sigma2")

        result_dict = {
            "params": params,
            "loglike": llf,
            "scale": float(params[sigma2_idx]),
            "aic": -2.0 * llf + 2.0 * k,
            "bic": -2.0 * llf + k * np.log(n_eff),
            "n_obs": n_eff,
            "converged": True,
            "method": "filter",
            "n_iter": 0,
            "n_params": k,
        }
        return SARIMAXResult(self, result_dict)


class SARIMAXResult:
    """Fit result wrapper (statsmodels ResultsWrapper compatible).

    Attributes
    ----------
    params : np.ndarray
        Estimated parameters.
    llf : float
        Log-likelihood at optimum.
    aic : float
        Akaike information criterion.
    bic : float
        Bayesian information criterion.
    scale : float
        Estimated variance (sigma^2).
    nobs : int
        Number of observations used.
    converged : bool
        Whether the optimizer converged.
    method : str
        Optimization method used.
    param_names : list[str]
        Parameter names matching params vector.
    """

    def __init__(self, model, result_dict):
        self.model = model
        self.params = np.array(result_dict["params"])
        self.llf = result_dict["loglike"]
        self.scale = result_dict["scale"]
        self.aic = result_dict["aic"]
        self.bic = result_dict["bic"]
        self.nobs = result_dict["n_obs"]
        self.converged = result_dict["converged"]
        self.method = result_dict["method"]
        self._n_iter = result_dict["n_iter"]
        self._n_params = result_dict["n_params"]
        self._resid = None
        self._inference_cache = {}

    @property
    def param_names(self):
        """Parameter names matching the params vector."""
        names = _generate_param_names(
            self.model.order,
            self.model.seasonal_order,
            n_exog=self.model.n_exog,
            trend=self.model.trend,
        )
        # A mismatch means the Rust param layout and the Python naming have
        # drifted — padding with param_N here would print values under the
        # WRONG names in summary()/params_table() (DIAGNOSIS_V9 S5).
        if len(names) != len(self.params):
            raise RuntimeError(
                f"param_names length {len(names)} != params length "
                f"{len(self.params)}; Rust param layout and Python naming "
                f"have drifted"
            )
        return names

    def _rs_kwargs(self, **extra):
        """Build common kwargs dict for rustima function calls.

        Delegates to SARIMAXModel._model_kwargs so residuals/forecast/
        diagnostics run under the SAME enforcement flags (and therefore the
        same Kalman initialization) as the fit. Previously the enforcement
        flags were dropped here, silently switching these paths to
        approximate-diffuse init.
        """
        return self.model._model_kwargs(**extra)

    @property
    def _rs_args(self):
        """Common positional args: (endog, order, seasonal_order, params)."""
        return (self.model.endog, self.model.order, self.model.seasonal_order, self.params)

    # ------------------------------------------------------------------
    # Cached inference helpers (used by parameter_summary)
    # ------------------------------------------------------------------

    def _get_rust_inference(self, mode, alpha, params_sig):
        """Return cached Rust-based (hessian/OPG) inference dict."""
        cache_key = (mode, alpha, params_sig)
        if cache_key not in self._inference_cache:
            rust_method = "hessian" if mode == "rust_hessian" else "opg"
            self._inference_cache[cache_key] = _compute_rust_inference(
                self.model.endog,
                self.model.order,
                self.model.seasonal_order,
                self.params,
                method=rust_method,
                alpha=alpha,
                exog=self.model.exog,
                enforce_stationarity=self.model.enforce_stationarity,
                enforce_invertibility=self.model.enforce_invertibility,
                trend=self.model.trend,
                simple_differencing=self.model.simple_differencing,
            )
        return self._inference_cache[cache_key]

    def _get_sm_inference(self, alpha, params_sig):
        """Return cached statsmodels inference dict."""
        cache_key = ("statsmodels", alpha, params_sig)
        if cache_key not in self._inference_cache:
            self._inference_cache[cache_key] = _compute_statsmodels_inference(
                self.model.endog,
                self.model.order,
                self.model.seasonal_order,
                alpha=alpha,
                exog=self.model.exog,
                n_params_rs=len(self.params),
                enforce_stationarity=self.model.enforce_stationarity,
                enforce_invertibility=self.model.enforce_invertibility,
                trend=self.model.trend,
                simple_differencing=self.model.simple_differencing,
            )
        return self._inference_cache[cache_key]

    def _build_both_result(self, result, alpha, params_sig):
        """Build combined hessian + statsmodels result for 'both' mode."""
        hess = self._get_rust_inference("rust_hessian", alpha, params_sig)
        sm = self._get_sm_inference(alpha, params_sig)

        # Legacy keys from hessian (default view)
        result.update(
            std_err=hess["std_err"],
            z=hess["z"],
            p_value=hess["p_value"],
            ci_lower=hess["ci_lower"],
            ci_upper=hess["ci_upper"],
        )

        # Prefixed hessian keys
        result.update(
            hessian_std_err=hess["std_err"],
            hessian_z=hess["z"],
            hessian_p_value=hess["p_value"],
            hessian_ci_lower=hess["ci_lower"],
            hessian_ci_upper=hess["ci_upper"],
            inference_status_hessian=hess["inference_status"],
        )

        # statsmodels keys
        result.update(
            sm_std_err=sm["sm_std_err"],
            sm_z=sm["sm_z"],
            sm_p_value=sm["sm_p_value"],
            sm_ci_lower=sm["sm_ci_lower"],
            sm_ci_upper=sm["sm_ci_upper"],
            inference_status_sm=sm["inference_status_sm"],
        )

        # Delta columns (hessian - statsmodels)
        result.update(
            delta_std_err=hess["std_err"] - sm["sm_std_err"],
            delta_ci_lower=hess["ci_lower"] - sm["sm_ci_lower"],
            delta_ci_upper=hess["ci_upper"] - sm["sm_ci_upper"],
        )

        # Combined status
        h_ok = hess["inference_status"] in ("ok", "partial")
        s_ok = sm["inference_status_sm"] == "ok"
        if h_ok and s_ok:
            result["inference_status"] = "ok"
        elif h_ok or s_ok:
            result["inference_status"] = "partial"
        else:
            result["inference_status"] = "failed"

        msgs = []
        if hess.get("inference_message"):
            msgs.append(f"hessian: {hess['inference_message']}")
        if sm.get("inference_message_sm"):
            msgs.append(f"statsmodels: {sm['inference_message_sm']}")
        result["inference_message"] = "; ".join(msgs) if msgs else None

        return result

    # ------------------------------------------------------------------

    def parameter_summary(self, alpha=0.05, include_inference=None, inference=None):
        """Return parameter summary as a machine-readable dict.

        Parameters
        ----------
        alpha : float
            Significance level for confidence intervals (0 < alpha < 1).
        include_inference : bool, optional
            **Deprecated.** Use ``inference`` instead.
            ``True`` maps to ``inference="hessian"``,
            ``False`` maps to ``inference="none"``.
        inference : str, optional
            Inference mode. One of:

            - ``"none"``  — coefficients only (fastest).
            - ``"hessian"``  — observed-information Hessian std err / z / CI
              (computed in Rust; single producer for inference).
            - ``"statsmodels"``  — fit statsmodels SARIMAX internally and
              borrow its inference statistics.
            - ``"both"``  — compute both hessian and statsmodels, include
              delta columns for comparison.

            Default is ``"none"`` when neither parameter is given, or
            ``"hessian"`` when legacy ``include_inference=True`` is used.

        Returns
        -------
        dict
            Always contains ``name`` and ``coef``.
            Additional keys depend on the inference mode.
        """
        mode = _resolve_inference_mode(inference, include_inference)

        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")

        names = self.param_names
        k = len(self.params)
        nan_arr = lambda: np.full(k, np.nan)  # noqa: E731

        params_sig = tuple(np.round(self.params, 12))

        result = dict(
            name=names,
            coef=self.params.copy(),
        )

        if mode == "none":
            result.update(
                std_err=nan_arr(), z=nan_arr(), p_value=nan_arr(),
                ci_lower=nan_arr(), ci_upper=nan_arr(),
                inference_status="skipped", inference_message=None,
            )
            return result

        if mode == "hessian":
            result.update(self._get_rust_inference("rust_hessian", alpha, params_sig))
            return result

        if mode in ("rust_hessian", "opg"):
            result.update(self._get_rust_inference(mode, alpha, params_sig))
            return result

        if mode == "statsmodels":
            sm = self._get_sm_inference(alpha, params_sig)
            result.update(
                std_err=sm["sm_std_err"],
                z=sm["sm_z"],
                p_value=sm["sm_p_value"],
                ci_lower=sm["sm_ci_lower"],
                ci_upper=sm["sm_ci_upper"],
                inference_status=sm["inference_status_sm"],
                inference_message=sm["inference_message_sm"],
            )
            return result

        # mode == "both"
        return self._build_both_result(result, alpha, params_sig)

    def forecast(self, steps=1, alpha=0.05, exog=None):
        """H-step ahead forecast.

        Parameters
        ----------
        steps : int
            Number of forecast steps.
        alpha : float
            Significance level for confidence intervals.
        exog : array_like, optional
            Future exogenous variables, shape (steps, n_exog).
            Required if the model was fit with exog.

        Returns
        -------
        ForecastResult
        """
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        kwargs = self._rs_kwargs(steps=steps, alpha=alpha)
        if exog is not None:
            exog = np.asarray(exog, dtype=np.float64)
            if exog.ndim == 1:
                exog = exog.reshape(-1, 1)
            kwargs["future_exog"] = exog

        result = rustima.sarimax_forecast(*self._rs_args, **kwargs)
        return ForecastResult(result, alpha=alpha)

    def get_forecast(self, steps=1, alpha=0.05, exog=None):
        """Alias for forecast() (statsmodels compatibility)."""
        return self.forecast(steps=steps, alpha=alpha, exog=exog)

    def rolling_forecast(self, start, step=1, horizon=1, alpha=0.05):
        """Single-pass rolling-origin h-step forecasts (fixed parameters).

        One Kalman-filter pass over the full sample captures the predicted
        state at every origin ``start, start+step, ...``; each origin's
        h-step forecast is propagated from its snapshot. Total cost
        O(T + N·horizon) — versus O(N·T) for an :meth:`extend` chain — while
        producing numerically identical forecasts (Markov property).

        Origins run while ``origin <= nobs - 1``; models with exog are
        capped at ``nobs - horizon`` (in-sample exog must cover each
        forecast window). ``simple_differencing=True`` is not supported yet.

        Parameters
        ----------
        start : int
            First forecast origin (number of observations consumed).
        step : int
            Origin spacing (e.g. 24 for daily rolling on hourly data).
        horizon : int
            Forecast steps per origin.
        alpha : float
            CI significance level.

        Returns
        -------
        RollingForecastResult
            ``origins`` (N,), ``predicted_mean``/``variance``/``ci_lower``/
            ``ci_upper`` (N, horizon).
        """
        result = rustima.sarimax_rolling_forecast(
            self.model.endog,
            self.model.order,
            self.model.seasonal_order,
            self.params,
            start=start,
            step=step,
            horizon=horizon,
            alpha=alpha,
            **self._rs_kwargs(),
        )
        return RollingForecastResult(result, alpha=alpha)

    def extend(self, endog, exog=None):
        """Extend the sample with new observations, keeping parameters fixed.

        Returns a new :class:`SARIMAXResult` whose sample is the original
        history plus ``endog``, filtered at the SAME parameters — no
        re-estimation. Subsequent :meth:`forecast` calls start after the new
        observations, enabling walk-forward rolling::

            res = SARIMAXModel(train, order, seasonal_order).fit()
            for block in blocks:
                fc = res.get_forecast(steps=len(block)).predicted_mean
                res = res.extend(block)

        Implementation note
        -------------------
        rustima refilters the FULL extended history with the fixed parameters
        (statsmodels ``append(refit=False)`` semantics). Because the Kalman
        filter is Markovian, post-extension forecasts are numerically
        equivalent to statsmodels' state-carry-over ``extend``. Unlike
        statsmodels ``extend``, ``llf``/``aic``/``bic`` here cover the full
        extended sample rather than only the new observations.

        Parameters
        ----------
        endog : array_like
            New observations that come AFTER the current sample.
        exog : array_like, optional
            Exogenous values for the new observations, shape
            ``(len(endog), n_exog)``. Required iff the model has exog.

        Returns
        -------
        SARIMAXResult
        """
        new = np.asarray(endog, dtype=np.float64).ravel()
        if new.size == 0:
            raise ValueError("endog is empty: extend() requires at least one new observation")
        if not np.isfinite(new).all():
            raise ValueError("endog contains NaN or Inf values")

        m = self.model
        if m.exog is not None:
            if exog is None:
                raise ValueError(
                    "model was built with exog; extend() requires exog for the new observations"
                )
            ex = np.asarray(exog, dtype=np.float64)
            if ex.ndim == 1:
                ex = ex.reshape(-1, 1)
            if ex.shape != (new.size, m.n_exog):
                raise ValueError(
                    f"exog shape {ex.shape} != expected ({new.size}, {m.n_exog})"
                )
            full_exog = np.vstack([m.exog, ex])
        else:
            if exog is not None:
                raise ValueError(
                    "model was built without exog; unexpected exog passed to extend()"
                )
            full_exog = None

        new_model = SARIMAXModel(
            np.concatenate([m.endog, new]),
            order=m.order,
            seasonal_order=m.seasonal_order,
            exog=full_exog,
            trend=m.trend,
            enforce_stationarity=m.enforce_stationarity,
            enforce_invertibility=m.enforce_invertibility,
            simple_differencing=m.simple_differencing,
        )
        return new_model.filter(self.params)

    def append(self, endog, exog=None, refit=False, **fit_kwargs):
        """Append new observations (statsmodels-compatible convenience).

        ``refit=False`` (default) is an alias for :meth:`extend` — parameters
        stay fixed. ``refit=True`` re-estimates parameters on the extended
        sample via :meth:`SARIMAXModel.fit`.
        """
        extended = self.extend(endog, exog=exog)
        if refit:
            return extended.model.fit(**fit_kwargs)
        return extended

    @property
    def resid(self):
        """Standardized residuals."""
        if self._resid is None:
            result = rustima.sarimax_residuals(*self._rs_args, **self._rs_kwargs())
            self._resid = np.array(result["standardized_residuals"])
        return self._resid

    def diagnostics(self):
        """Compute residual diagnostic tests.

        Returns
        -------
        dict
            Keys: ljung_box_stat, ljung_box_pvalue, ljung_box_df,
                  jarque_bera_stat, jarque_bera_pvalue, het_stat, het_pvalue.
        """
        return rustima.sarimax_diagnostics(*self._rs_args, **self._rs_kwargs())

    def params_table(self, alpha=0.05, inference="none"):
        """Return parameter table as a Polars DataFrame.

        Parameters
        ----------
        alpha : float
            Significance level for CI.
        inference : str
            Inference mode ("none", "hessian", "rust_hessian", "opg",
            "statsmodels", "both").

        Returns
        -------
        polars.DataFrame
        """
        import polars as pl

        ps = self.parameter_summary(alpha=alpha, inference=inference)
        data = {
            "name": ps["name"],
            "coef": ps["coef"].tolist(),
        }
        if inference != "none" and ps.get("inference_status") != "skipped":
            for key in ("std_err", "z", "p_value", "ci_lower", "ci_upper"):
                if key in ps and ps[key] is not None:
                    data[key] = ps[key].tolist()
        return pl.DataFrame(data)

    def get_prediction(self, start=None, end=None, alpha=0.05, exog=None):
        """In-sample one-step-ahead predictions with optional out-of-sample.

        Parameters
        ----------
        start : int or None
            Start index (default 0).
        end : int or None
            End index exclusive (default nobs). If end > nobs,
            out-of-sample forecast is appended.
        alpha : float
            Significance level for out-of-sample CI.
        exog : array_like, optional
            Future exogenous for out-of-sample portion.

        Returns
        -------
        PredictionResult
        """
        n_endog = len(self.model.endog)

        # --- start/end validation ---
        if start is not None and not isinstance(start, (int, np.integer)):
            raise TypeError(f"start must be int or None, got {type(start).__name__}")
        if end is not None and not isinstance(end, (int, np.integer)):
            raise TypeError(f"end must be int or None, got {type(end).__name__}")
        if start is not None and start < 0:
            raise ValueError(f"start must be >= 0, got {start}")
        if end is not None and end < 0:
            raise ValueError(f"end must be >= 0, got {end}")
        if start is not None and end is not None and start > end:
            raise ValueError(f"start ({start}) must be <= end ({end})")

        if start is None:
            start = 0
        if end is None:
            end = n_endog

        # In-sample: one-step-ahead predictions = endog - residuals (innovations)
        resid_out = rustima.sarimax_residuals(*self._rs_args, **self._rs_kwargs())
        residuals = np.array(resid_out["residuals"])
        in_sample_se = np.sqrt(np.array(resid_out["prediction_variances"]))

        # When simple_differencing=True, Rust returns residuals of length
        # n_eff = n - d - s*D (dropped observations).  Pad the front with
        # NaN so the prediction array aligns with the original endog index.
        n_drop = n_endog - len(residuals)
        if n_drop > 0:
            endog_eff = self.model.endog[n_drop:]
            in_sample_pred = np.concatenate([
                np.full(n_drop, np.nan),
                endog_eff - residuals,
            ])
            in_sample_se = np.concatenate([np.full(n_drop, np.nan), in_sample_se])
        else:
            in_sample_pred = self.model.endog - residuals

        # Out-of-sample
        if end > n_endog:
            fc = self.forecast(steps=end - n_endog, alpha=alpha, exog=exog)
            all_pred = np.concatenate([in_sample_pred, fc.predicted_mean])
            all_se = np.concatenate([in_sample_se, np.sqrt(fc.variance)])
        else:
            all_pred = in_sample_pred
            all_se = in_sample_se

        return PredictionResult(all_pred[start:end], all_se[start:end], alpha=alpha)

    @property
    def hqic(self):
        """Hannan-Quinn information criterion."""
        return _hqic(self.llf, self._n_params, self.nobs)

    def summary(self, alpha=0.05, include_inference=None, inference=None):
        """Return a statsmodels-style summary string.

        Parameters
        ----------
        alpha : float
            Significance level for inference CI.
        include_inference : bool, optional
            **Deprecated.** Use ``inference`` instead.
        inference : str, optional
            Inference mode: ``"none"`` | ``"hessian"`` | ``"statsmodels"``
            | ``"both"``.  Default ``"none"``.
        """
        import datetime

        ps = self.parameter_summary(
            alpha=alpha, include_inference=include_inference, inference=inference,
        )
        mode = _resolve_inference_mode(inference, include_inference)

        p, d, q = self.model.order
        pp, dd, qq, s = self.model.seasonal_order
        w = 78
        hw = w // 2

        lines = []
        lines.append("=" * w)
        lines.append(f"{'SARIMAX Results':^{w}}")
        lines.append("=" * w)

        # 2-column header
        model_str = f"SARIMAX({p},{d},{q})({pp},{dd},{qq})[{s}]"
        pairs = [
            (f"Model: {model_str}", f"Log Likelihood: {self.llf:>12.3f}"),
            (f"No. Observations: {self.nobs}", f"AIC: {self.aic:>22.3f}"),
            (f"Trend: {self.model.trend}", f"BIC: {self.bic:>22.3f}"),
            (f"Method: {self.method}", f"HQIC: {self.hqic:>21.3f}"),
            (f"Converged: {self.converged}", f"Scale: {self.scale:>19.6f}"),
            (f"Date: {datetime.date.today()}", ""),
        ]
        for left, right in pairs:
            lines.append(f"{left:<{hw}}{right:>{hw}}")
        lines.append("-" * w)

        names = ps["name"]
        coefs = ps["coef"]
        has_inf = mode != "none" and ps.get("inference_status") != "skipped"

        if has_inf and mode == "both":
            header = (
                f"{'':>16s} {'coef':>10s} "
                f"{'hess_se':>9s} {'sm_se':>9s} {'d_se':>9s} "
                f"{'hess_z':>8s} {'sm_z':>8s} "
                f"{'hess_p':>8s} {'sm_p':>8s}"
            )
            lines.append(header)
            lines.append("-" * 98)
            hse = ps.get("hessian_std_err", np.full(len(names), np.nan))
            sse = ps.get("sm_std_err", np.full(len(names), np.nan))
            dse = ps.get("delta_std_err", np.full(len(names), np.nan))
            hz = ps.get("hessian_z", np.full(len(names), np.nan))
            sz = ps.get("sm_z", np.full(len(names), np.nan))
            hpval = ps.get("hessian_p_value", np.full(len(names), np.nan))
            spval = ps.get("sm_p_value", np.full(len(names), np.nan))
            for i, name in enumerate(names):
                def _f(v, fmt=".4f"):
                    return f"{v:{fmt}}" if np.isfinite(v) else "NaN"
                lines.append(
                    f"{name:>16s} {coefs[i]:>10.4f} "
                    f"{_f(hse[i]):>9s} {_f(sse[i]):>9s} {_f(dse[i]):>9s} "
                    f"{_f(hz[i], '.3f'):>8s} {_f(sz[i], '.3f'):>8s} "
                    f"{_f(hpval[i], '.3f'):>8s} {_f(spval[i], '.3f'):>8s}"
                )
        elif has_inf:
            std_err = ps["std_err"]
            z = ps["z"]
            pval = ps["p_value"]
            ci_lo = ps["ci_lower"]
            ci_hi = ps["ci_upper"]

            header = (f"{'':>16s} {'coef':>10s} {'std err':>10s} "
                      f"{'z':>8s} {'P>|z|':>8s} "
                      f"{'[' + f'{alpha/2:.3f}':>7s} {f'{1-alpha/2:.3f}' + ']':>7s}")
            lines.append(header)
            lines.append("-" * w)
            for i, name in enumerate(names):
                se_s = f"{std_err[i]:.4f}" if np.isfinite(std_err[i]) else "NaN"
                z_s = f"{z[i]:.3f}" if np.isfinite(z[i]) else "NaN"
                p_s = f"{pval[i]:.3f}" if np.isfinite(pval[i]) else "NaN"
                lo_s = f"{ci_lo[i]:.3f}" if np.isfinite(ci_lo[i]) else "NaN"
                hi_s = f"{ci_hi[i]:.3f}" if np.isfinite(ci_hi[i]) else "NaN"
                lines.append(
                    f"{name:>16s} {coefs[i]:>10.4f} {se_s:>10s} "
                    f"{z_s:>8s} {p_s:>8s} {lo_s:>7s} {hi_s:>7s}"
                )
        else:
            header = f"{'':>16s} {'coef':>10s}"
            lines.append(header)
            lines.append("-" * w)
            for i, name in enumerate(names):
                lines.append(f"{name:>16s} {coefs[i]:>10.4f}")

        lines.append("=" * w)

        if has_inf and ps.get("inference_message"):
            lines.append(f"Notes: {ps['inference_message']}")
        if mode != "none":
            lines.append(f"Inference: {mode}")

        return "\n".join(lines)


class ForecastResult:
    """Forecast result wrapper.

    Attributes
    ----------
    predicted_mean : np.ndarray
        Point forecasts.
    variance : np.ndarray
        Forecast variance at each step.
    ci_lower : np.ndarray
        Lower confidence interval bounds (at original alpha).
    ci_upper : np.ndarray
        Upper confidence interval bounds (at original alpha).
    """

    def __init__(self, result_dict, alpha=0.05):
        self.predicted_mean = np.array(result_dict["mean"])
        self.variance = np.array(result_dict["variance"])
        self.ci_lower = np.array(result_dict["ci_lower"])
        self.ci_upper = np.array(result_dict["ci_upper"])
        self._alpha = alpha

    def conf_int(self, alpha=None):
        """Return confidence intervals as (n, 2) array.

        Parameters
        ----------
        alpha : float, optional
            Significance level. If None or equal to the original alpha,
            returns the precomputed CI. Otherwise recomputes CI from
            the stored variance.

        Returns
        -------
        np.ndarray of shape (n, 2)
            Columns are [lower, upper].
        """
        if alpha is not None and not (0.0 < alpha < 1.0):
            raise ValueError(
                f"alpha must be in (0, 1), got {alpha!r}"
            )

        if alpha is None or alpha == self._alpha:
            return np.column_stack([self.ci_lower, self.ci_upper])

        z = _norm_ppf(1.0 - alpha / 2.0)
        std = np.sqrt(self.variance)
        lower = self.predicted_mean - z * std
        upper = self.predicted_mean + z * std
        return np.column_stack([lower, upper])

    def to_dataframe(self):
        """Return forecast as a Polars DataFrame.

        Returns
        -------
        polars.DataFrame
            Columns: step, mean, variance, ci_lower, ci_upper.
        """
        import polars as pl

        n = len(self.predicted_mean)
        return pl.DataFrame({
            "step": list(range(1, n + 1)),
            "mean": self.predicted_mean.tolist(),
            "variance": self.variance.tolist(),
            "ci_lower": self.ci_lower.tolist(),
            "ci_upper": self.ci_upper.tolist(),
        })


class RollingForecastResult:
    """Rolling-origin forecast result (one row per origin).

    Attributes
    ----------
    origins : np.ndarray, shape (N,)
        Forecast origins (observations consumed before each forecast).
    predicted_mean : np.ndarray, shape (N, horizon)
    variance : np.ndarray, shape (N, horizon)
    ci_lower : np.ndarray, shape (N, horizon)
    ci_upper : np.ndarray, shape (N, horizon)
    """

    def __init__(self, result_dict, alpha=0.05):
        self.origins = np.asarray(result_dict["origins"], dtype=np.int64)
        self.predicted_mean = np.asarray(result_dict["mean"], dtype=np.float64)
        self.variance = np.asarray(result_dict["variance"], dtype=np.float64)
        self.ci_lower = np.asarray(result_dict["ci_lower"], dtype=np.float64)
        self.ci_upper = np.asarray(result_dict["ci_upper"], dtype=np.float64)
        self._alpha = alpha

    def to_dataframe(self):
        """Long-format Polars DataFrame: origin, step, mean, variance, ci."""
        import polars as pl

        n_origins, horizon = self.predicted_mean.shape
        return pl.DataFrame({
            "origin": np.repeat(self.origins, horizon),
            "step": np.tile(np.arange(1, horizon + 1), n_origins),
            "mean": self.predicted_mean.ravel(),
            "variance": self.variance.ravel(),
            "ci_lower": self.ci_lower.ravel(),
            "ci_upper": self.ci_upper.ravel(),
        })


class PredictionResult:
    """In-sample (and optionally out-of-sample) prediction result.

    Attributes
    ----------
    predicted_mean : np.ndarray
        One-step-ahead predicted values for the requested range.
    se_mean : np.ndarray
        Standard errors of the predictions (sqrt of the one-step innovation
        variance in-sample, forecast variance out-of-sample).
    """

    def __init__(self, predicted_mean, se_mean=None, alpha=0.05):
        self.predicted_mean = np.asarray(predicted_mean, dtype=np.float64)
        if se_mean is None:
            se_mean = np.full(len(self.predicted_mean), np.nan)
        self.se_mean = np.asarray(se_mean, dtype=np.float64)
        self._alpha = alpha

    def conf_int(self, alpha=None):
        """Confidence interval as an (n, 2) array (statsmodels-compatible).

        Parameters
        ----------
        alpha : float, optional
            Significance level; defaults to the alpha given to
            ``get_prediction``.
        """
        if alpha is None:
            alpha = self._alpha
        z = _norm_ppf(1.0 - alpha / 2.0)
        return np.column_stack([
            self.predicted_mean - z * self.se_mean,
            self.predicted_mean + z * self.se_mean,
        ])

    def to_dataframe(self):
        """Return predictions as a Polars DataFrame.

        Returns
        -------
        polars.DataFrame
            Columns: index, predicted_mean.
        """
        import polars as pl

        n = len(self.predicted_mean)
        return pl.DataFrame({
            "index": list(range(n)),
            "predicted_mean": self.predicted_mean.tolist(),
        })
