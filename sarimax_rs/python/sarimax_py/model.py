"""statsmodels-compatible SARIMAX model backed by Rust engine (sarimax_rs)."""

import math

import numpy as np
import sarimax_rs


# ---------------------------------------------------------------------------
# Parameter naming helpers
# ---------------------------------------------------------------------------

def _generate_param_names(order, seasonal_order, n_exog=0, concentrate_scale=True):
    """Generate statsmodels-style parameter names from model specification.

    Layout: [exog(k) | ar(p) | ma(q) | sar(P) | sma(Q) | sigma2?]

    Parameters
    ----------
    order : tuple (p, d, q)
    seasonal_order : tuple (P, D, Q, s)
    n_exog : int
    concentrate_scale : bool

    Returns
    -------
    list[str]
    """
    p, _d, q = order
    P, _D, Q, s = seasonal_order
    names = []

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


def _compute_numerical_hessian(loglike_fn, params):
    """Compute the Hessian of loglike_fn at params via central differences.

    Parameters
    ----------
    loglike_fn : callable
        f(params) -> float (log-likelihood).
    params : np.ndarray
        Parameter vector at MLE.

    Returns
    -------
    np.ndarray of shape (k, k)
        Hessian matrix, or None if evaluation failed.
    """
    k = len(params)
    x = np.array(params, dtype=np.float64)

    # Step sizes: h_i = max(1e-5, 1e-4 * max(1, |x_i|))
    h = np.maximum(1e-5, 1e-4 * np.maximum(1.0, np.abs(x)))

    f0 = loglike_fn(x)
    if not np.isfinite(f0):
        return None

    H = np.zeros((k, k))

    # Diagonal: H_ii = (f(x+h_i) - 2f(x) + f(x-h_i)) / h_i^2
    fp = np.empty(k)
    fm = np.empty(k)
    for i in range(k):
        xp = x.copy()
        xp[i] += h[i]
        xm = x.copy()
        xm[i] -= h[i]
        fp[i] = loglike_fn(xp)
        fm[i] = loglike_fn(xm)
        H[i, i] = (fp[i] - 2.0 * f0 + fm[i]) / (h[i] ** 2)

    # Off-diagonal: H_ij = (f(x+hi+hj) - f(x+hi-hj) - f(x-hi+hj) + f(x-hi-hj)) / (4 hi hj)
    for i in range(k):
        for j in range(i + 1, k):
            xpp = x.copy()
            xpp[i] += h[i]
            xpp[j] += h[j]
            xpm = x.copy()
            xpm[i] += h[i]
            xpm[j] -= h[j]
            xmp = x.copy()
            xmp[i] -= h[i]
            xmp[j] += h[j]
            xmm = x.copy()
            xmm[i] -= h[i]
            xmm[j] -= h[j]

            fpp = loglike_fn(xpp)
            fpm = loglike_fn(xpm)
            fmp = loglike_fn(xmp)
            fmm = loglike_fn(xmm)

            H[i, j] = (fpp - fpm - fmp + fmm) / (4.0 * h[i] * h[j])
            H[j, i] = H[i, j]

    if not np.all(np.isfinite(H)):
        return None

    return H


def _compute_inference(loglike_fn, params, alpha=0.05):
    """Compute inference statistics from numerical Hessian.

    Returns
    -------
    dict with keys: std_err, z, p_value, ci_lower, ci_upper,
                    inference_status, inference_message
    """
    k = len(params)
    nan_array = np.full(k, np.nan)

    H = _compute_numerical_hessian(loglike_fn, params)

    if H is None:
        return dict(
            std_err=nan_array.copy(),
            z=nan_array.copy(),
            p_value=nan_array.copy(),
            ci_lower=nan_array.copy(),
            ci_upper=nan_array.copy(),
            inference_status="failed",
            inference_message="Hessian computation produced non-finite values",
        )

    # Observed information matrix: I = -H
    info = -H

    # Covariance: inv(I), fallback to pinv
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        try:
            cov = np.linalg.pinv(info)
        except np.linalg.LinAlgError:
            return dict(
                std_err=nan_array.copy(),
                z=nan_array.copy(),
                p_value=nan_array.copy(),
                ci_lower=nan_array.copy(),
                ci_upper=nan_array.copy(),
                inference_status="failed",
                inference_message="Information matrix inversion failed",
            )

    diag = np.diag(cov)

    # Check for negative variances (non-PD covariance)
    if np.any(diag < 0):
        # Try pinv as fallback
        cov = np.linalg.pinv(info)
        diag = np.diag(cov)

    std_err = np.where(diag >= 0, np.sqrt(diag), np.nan)
    z_stat = np.where(std_err > 0, params / std_err, np.nan)
    p_value = np.array([
        2.0 * (1.0 - _norm_cdf(abs(zi))) if np.isfinite(zi) else np.nan
        for zi in z_stat
    ])

    z_crit = _norm_ppf(1.0 - alpha / 2.0)
    ci_lower = params - z_crit * std_err
    ci_upper = params + z_crit * std_err

    # Determine status
    if np.all(np.isfinite(std_err)):
        status = "ok"
        message = None
    else:
        status = "partial"
        n_bad = np.sum(~np.isfinite(std_err))
        message = f"{n_bad} of {k} parameters have non-finite standard errors"

    return dict(
        std_err=std_err,
        z=z_stat,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        inference_status=status,
        inference_message=message,
    )


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
    enforce_stationarity : bool
        Enforce AR stationarity constraints during fitting.
    enforce_invertibility : bool
        Enforce MA invertibility constraints during fitting.
    """

    def __init__(
        self,
        endog,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        exog=None,
        enforce_stationarity=True,
        enforce_invertibility=True,
    ):
        self.endog = np.asarray(endog, dtype=np.float64)
        self.order = order
        self.seasonal_order = seasonal_order
        self.exog = np.asarray(exog, dtype=np.float64) if exog is not None else None
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self._fit_result = None

    @property
    def nobs(self):
        return len(self.endog)

    @property
    def n_exog(self):
        if self.exog is None:
            return 0
        return self.exog.shape[1] if self.exog.ndim == 2 else 1

    def fit(self, method=None, maxiter=None, start_params=None):
        """Fit the SARIMAX model via MLE.

        Returns
        -------
        SARIMAXResult
        """
        kwargs = dict(
            start_params=start_params,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
            method=method,
            maxiter=maxiter,
        )
        if self.exog is not None:
            kwargs["exog"] = self.exog

        result_dict = sarimax_rs.sarimax_fit(
            self.endog,
            self.order,
            self.seasonal_order,
            **kwargs,
        )
        self._fit_result = SARIMAXResult(self, result_dict)
        return self._fit_result


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
        )
        # Safety: pad or trim to match actual params length
        k = len(self.params)
        if len(names) < k:
            names.extend(f"param_{i}" for i in range(len(names), k))
        elif len(names) > k:
            names = names[:k]
        return names

    def _loglike_fn(self, params):
        """Evaluate log-likelihood at given params (for Hessian computation)."""
        try:
            kwargs = dict(
                enforce_stationarity=self.model.enforce_stationarity,
                enforce_invertibility=self.model.enforce_invertibility,
            )
            if self.model.exog is not None:
                kwargs["exog"] = self.model.exog
            return sarimax_rs.sarimax_loglike(
                self.model.endog,
                self.model.order,
                self.model.seasonal_order,
                np.array(params, dtype=np.float64),
                **kwargs,
            )
        except Exception:
            return np.nan

    def parameter_summary(self, alpha=0.05, include_inference=True):
        """Return parameter summary as a machine-readable dict.

        Parameters
        ----------
        alpha : float
            Significance level for confidence intervals.
        include_inference : bool
            If True, compute numerical Hessian-based inference statistics.

        Returns
        -------
        dict with keys:
            name, coef, std_err, z, p_value, ci_lower, ci_upper,
            inference_status, inference_message
        """
        names = self.param_names
        k = len(self.params)

        result = dict(
            name=names,
            coef=self.params.copy(),
        )

        if not include_inference:
            result.update(
                std_err=np.full(k, np.nan),
                z=np.full(k, np.nan),
                p_value=np.full(k, np.nan),
                ci_lower=np.full(k, np.nan),
                ci_upper=np.full(k, np.nan),
                inference_status="skipped",
                inference_message=None,
            )
            return result

        # Check cache
        cache_key = alpha
        if cache_key not in self._inference_cache:
            self._inference_cache[cache_key] = _compute_inference(
                self._loglike_fn, self.params, alpha=alpha,
            )
        inf = self._inference_cache[cache_key]

        result.update(inf)
        return result

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
        kwargs = dict(steps=steps, alpha=alpha)
        if self.model.exog is not None:
            kwargs["exog"] = self.model.exog
        if exog is not None:
            kwargs["future_exog"] = np.asarray(exog, dtype=np.float64)

        result = sarimax_rs.sarimax_forecast(
            self.model.endog,
            self.model.order,
            self.model.seasonal_order,
            self.params,
            **kwargs,
        )
        return ForecastResult(result, alpha=alpha)

    def get_forecast(self, steps=1, alpha=0.05, exog=None):
        """Alias for forecast() (statsmodels compatibility)."""
        return self.forecast(steps=steps, alpha=alpha, exog=exog)

    @property
    def resid(self):
        """Standardized residuals."""
        if self._resid is None:
            kwargs = {}
            if self.model.exog is not None:
                kwargs["exog"] = self.model.exog
            result = sarimax_rs.sarimax_residuals(
                self.model.endog,
                self.model.order,
                self.model.seasonal_order,
                self.params,
                **kwargs,
            )
            self._resid = np.array(result["standardized_residuals"])
        return self._resid

    def summary(self, alpha=0.05, include_inference=False):
        """Return a summary string of the model fit.

        Parameters
        ----------
        alpha : float
            Significance level for inference CI (if include_inference=True).
        include_inference : bool
            If True, compute and display std err / z / p-value / CI columns.
        """
        ps = self.parameter_summary(alpha=alpha, include_inference=include_inference)

        lines = [
            "SARIMAX Results",
            "=" * 78,
            f"  Order:           {self.model.order}",
            f"  Seasonal:        {self.model.seasonal_order}",
            f"  Observations:    {self.nobs}",
            f"  Log Likelihood:  {self.llf:.4f}",
            f"  AIC:             {self.aic:.4f}",
            f"  BIC:             {self.bic:.4f}",
            f"  Converged:       {self.converged}",
            f"  Method:          {self.method}",
            "=" * 78,
        ]

        # Parameter table
        names = ps["name"]
        coefs = ps["coef"]
        has_inf = include_inference and ps["inference_status"] != "skipped"

        if has_inf:
            std_err = ps["std_err"]
            z = ps["z"]
            pval = ps["p_value"]
            ci_lo = ps["ci_lower"]
            ci_hi = ps["ci_upper"]

            header = f"{'':>16s} {'coef':>10s} {'std err':>10s} {'z':>10s} {'P>|z|':>10s} {'[' + f'{alpha/2:.3f}':>7s} {f'{1-alpha/2:.3f}' + ']':>7s}"
            lines.append(header)
            lines.append("-" * 78)
            for i, name in enumerate(names):
                se_s = f"{std_err[i]:.4f}" if np.isfinite(std_err[i]) else "NaN"
                z_s = f"{z[i]:.3f}" if np.isfinite(z[i]) else "NaN"
                p_s = f"{pval[i]:.3f}" if np.isfinite(pval[i]) else "NaN"
                lo_s = f"{ci_lo[i]:.3f}" if np.isfinite(ci_lo[i]) else "NaN"
                hi_s = f"{ci_hi[i]:.3f}" if np.isfinite(ci_hi[i]) else "NaN"
                lines.append(
                    f"{name:>16s} {coefs[i]:>10.4f} {se_s:>10s} {z_s:>10s} {p_s:>10s} {lo_s:>7s} {hi_s:>7s}"
                )
        else:
            header = f"{'Parameters:':>16s} {'coef':>10s}"
            lines.append(header)
            lines.append("-" * 78)
            for i, name in enumerate(names):
                lines.append(f"{name:>16s} {coefs[i]:>10.4f}")

        lines.append("-" * 78)
        lines.append(f"  Scale (sigma2): {self.scale:.6f}")

        if has_inf and ps["inference_message"]:
            lines.append(f"  Note: {ps['inference_message']}")

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
