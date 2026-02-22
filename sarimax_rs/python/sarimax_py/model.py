"""statsmodels-compatible SARIMAX model backed by Rust engine (sarimax_rs)."""

import numpy as np
import sarimax_rs


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

    def summary(self):
        """Return a summary string of the model fit."""
        lines = [
            "SARIMAX Results",
            "=" * 50,
            f"Order:          {self.model.order}",
            f"Seasonal:       {self.model.seasonal_order}",
            f"Observations:   {self.nobs}",
            f"Log Likelihood: {self.llf:.4f}",
            f"AIC:            {self.aic:.4f}",
            f"BIC:            {self.bic:.4f}",
            f"Converged:      {self.converged}",
            f"Method:         {self.method}",
            "-" * 50,
            f"Parameters:     {self.params}",
            f"Scale (sigma2): {self.scale:.6f}",
        ]
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

        # Recompute CI with the new alpha using scipy if available,
        # otherwise use a rational approximation (Abramowitz & Stegun).
        z = _norm_ppf(1.0 - alpha / 2.0)
        std = np.sqrt(self.variance)
        lower = self.predicted_mean - z * std
        upper = self.predicted_mean + z * std
        return np.column_stack([lower, upper])


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

    # Rational approximation for 0.5 < p < 1
    if p < 0.5:
        return -_norm_ppf(1.0 - p)
    if p == 0.5:
        return 0.0

    t = np.sqrt(-2.0 * np.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t**2) / (1.0 + d1 * t + d2 * t**2 + d3 * t**3)
