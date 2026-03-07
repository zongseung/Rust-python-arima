"""auto_arima — automatic ARIMA order selection.

Implements Hyndman-Khandakar stepwise algorithm and exhaustive grid search
backed by Rust parallel batch fitting.
"""

import itertools
import math

import numpy as np

from .model import SARIMAXModel


def _ndiffs(y, max_d=2, test="adf"):
    """Estimate the number of non-seasonal differences needed.

    Uses the ADF test from statsmodels when available, otherwise falls
    back to a simple variance reduction heuristic.
    """
    if max_d == 0:
        return 0
    try:
        from statsmodels.tsa.stattools import adfuller
        for d in range(max_d + 1):
            yd = np.diff(y, n=d) if d > 0 else y
            if len(yd) < 10:
                return d
            _, p, *_ = adfuller(yd, maxlag=min(10, len(yd) // 3))
            if p < 0.05:
                return d
        return max_d
    except ImportError:
        # Fallback: difference until variance stops decreasing
        prev_var = np.var(y)
        for d in range(1, max_d + 1):
            yd = np.diff(y, n=d)
            v = np.var(yd)
            if v >= prev_var:
                return d - 1
            prev_var = v
        return max_d


def _nsdiffs(y, s, max_D=1):
    """Estimate seasonal differences (0 or 1).

    Uses a variance-ratio test: seasonal differencing should substantially
    reduce variance for a seasonally integrated series (ratio << 1), but
    leaves stationary series roughly unchanged (ratio ≈ 1) or increases
    variance (ratio > 1).

    Threshold 0.64 ≈ 2*(1 − 0.68): requires at least a 36 % variance
    reduction before suggesting D=1.  Strong seasonal ARMA processes
    produce ratios 0.8–1.0 and are correctly classified as D=0.
    """
    if s <= 1 or max_D == 0:
        return 0
    n = len(y)
    if n < 2 * s:
        return 0
    yd = y - np.mean(y)
    var_orig = np.var(yd)
    if var_orig == 0:
        return 0
    # Seasonal difference: y[t] − y[t−s]
    y_sdiff = yd[s:] - yd[:-s]
    ratio = np.var(y_sdiff) / var_orig
    # For a seasonal unit root ratio ≪ 1; for stationary SARMA ratio ≈ 1
    return 1 if ratio < 0.64 else 0


def _ic(result, criterion="aic"):
    """Extract information criterion from fit result."""
    if criterion == "aic":
        return result.aic
    elif criterion == "bic":
        return result.bic
    elif criterion == "hqic":
        return result.hqic
    raise ValueError(f"Unknown criterion: {criterion}")


def _safe_ic(value):
    """Convert non-finite IC to None for history storage."""
    return value if math.isfinite(value) else None


def _make_history_entry(order, seasonal, criterion, ic_val, converged,
                        error_type=None, error_message=None):
    """Build a history dict entry for auto_arima search tracking."""
    entry = {
        "order": order,
        "seasonal_order": seasonal,
        criterion: _safe_ic(ic_val),
        "converged": converged,
    }
    if error_type is not None:
        entry["error_type"] = error_type
        entry["error_message"] = error_message
    return entry


def _trace_print(p, d, q, P, D, Q, s, criterion, ic_val):
    """Print a trace line for auto_arima model evaluation."""
    status = f"{ic_val:.3f}" if math.isfinite(ic_val) else "FAILED"
    print(f"  ARIMA({p},{d},{q})({P},{D},{Q})[{s}] : {criterion}={status}")


def _try_fit(endog, order, seasonal_order, trend="n",
             enforce_stationarity=True, enforce_invertibility=True,
             method=None, maxiter=None,
             exog=None, simple_differencing=False, trace=False):
    """Attempt to fit a model, returning (result, error_info) tuple.

    Returns
    -------
    tuple : (SARIMAXResult or None, dict or None)
        On success: (result, None).
        On expected failure: (None, {"error_type": ..., "error_message": ...}).
        On unexpected failure: raises the exception.
    """
    try:
        model = SARIMAXModel(
            endog, order=order, seasonal_order=seasonal_order,
            exog=exog,
            trend=trend,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
            simple_differencing=simple_differencing,
        )
        result = model.fit(method=method, maxiter=maxiter)
        return result, None
    except (ValueError, RuntimeError, ArithmeticError,
            OverflowError, ZeroDivisionError) as exc:
        if trace:
            print(f"    [FAILED] ARIMA{order}{seasonal_order}: {type(exc).__name__}: {exc}")
        return None, {"error_type": type(exc).__name__, "error_message": str(exc)}
    except Exception as exc:
        # Unexpected errors (FFI panic, OOM, etc.) propagate for debugging
        if trace:
            print(f"    [UNEXPECTED] ARIMA{order}{seasonal_order}: {type(exc).__name__}: {exc}")
        raise


def auto_arima(
    endog,
    exog=None,
    max_p=5,
    max_q=5,
    max_d=2,
    max_P=2,
    max_Q=2,
    max_D=1,
    s=0,
    d=None,
    D=None,
    trend="n",
    criterion="aic",
    stepwise=True,
    enforce_stationarity=True,
    enforce_invertibility=True,
    method=None,
    maxiter=None,
    simple_differencing=False,
    trace=False,
):
    """Automatic ARIMA order selection.

    Parameters
    ----------
    endog : array_like
        Time series.
    max_p, max_q : int
        Maximum AR and MA orders.
    max_d : int
        Maximum differencing order (auto-detected if d is None).
    max_P, max_Q : int
        Maximum seasonal AR and MA orders.
    max_D : int
        Maximum seasonal differencing.
    s : int
        Seasonal period (0 = non-seasonal).
    d : int or None
        Differencing order. Auto-detected if None.
    D : int or None
        Seasonal differencing. Auto-detected if None.
    trend : str
        Trend specification: 'n', 'c', 't', 'ct'.
    criterion : str
        'aic', 'bic', or 'hqic'.
    stepwise : bool
        If True, use Hyndman-Khandakar stepwise (fast).
        If False, exhaustive grid search (slow but thorough).
    enforce_stationarity : bool
    enforce_invertibility : bool
    method : str or None
    maxiter : int or None
    trace : bool
        If True, print each model evaluated.

    Returns
    -------
    AutoARIMAResult
        Best fit result with search history.
    """
    endog = np.asarray(endog, dtype=np.float64)
    if exog is not None:
        exog = np.asarray(exog, dtype=np.float64)
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)

    # Validate criterion early (before branching to stepwise/grid)
    _VALID_CRITERIA = {"aic", "bic", "hqic"}
    if criterion not in _VALID_CRITERIA:
        raise ValueError(
            f"Unknown criterion: {criterion!r}. Must be one of {sorted(_VALID_CRITERIA)}"
        )

    # Auto-detect differencing
    if d is None:
        d = _ndiffs(endog, max_d=max_d)
    if D is None:
        D = _nsdiffs(endog, s, max_D=max_D)

    if stepwise:
        return _stepwise(
            endog, d=d, D=D, s=s,
            max_p=max_p, max_q=max_q, max_P=max_P, max_Q=max_Q,
            trend=trend, criterion=criterion,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
            method=method, maxiter=maxiter, trace=trace,
            exog=exog, simple_differencing=simple_differencing,
        )
    else:
        return _grid_search(
            endog, d=d, D=D, s=s,
            max_p=max_p, max_q=max_q, max_P=max_P, max_Q=max_Q,
            trend=trend, criterion=criterion,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
            method=method, maxiter=maxiter, trace=trace,
            exog=exog, simple_differencing=simple_differencing,
        )


def _stepwise(endog, d, D, s, max_p, max_q, max_P, max_Q,
              trend, criterion, enforce_stationarity, enforce_invertibility,
              method, maxiter, trace,
              exog=None, simple_differencing=False):
    """Hyndman-Khandakar stepwise algorithm."""
    # Non-seasonal model: suppress seasonal order exploration entirely
    if s == 0:
        max_P = 0
        max_Q = 0

    history = []
    visited = set()

    def _eval(p, q, P, Q):
        key = (p, q, P, Q)
        if key in visited:
            return None, math.inf
        visited.add(key)

        order = (p, d, q)
        seasonal = (P, D, Q, s)
        result, err_info = _try_fit(
            endog, order, seasonal,
            trend=trend,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
            method=method, maxiter=maxiter,
            exog=exog, simple_differencing=simple_differencing,
            trace=trace,
        )
        if result is None:
            ic_val = math.inf
        else:
            ic_val = _ic(result, criterion)
            if not math.isfinite(ic_val):
                ic_val = math.inf

        err_kw = {}
        if err_info is not None:
            err_kw["error_type"] = err_info["error_type"]
            err_kw["error_message"] = err_info["error_message"]

        history.append(_make_history_entry(
            order, seasonal, criterion, ic_val,
            result.converged if result else False,
            **err_kw,
        ))
        if trace:
            _trace_print(p, d, q, P, D, Q, s, criterion, ic_val)
        return result, ic_val

    # Initial candidates (Hyndman-Khandakar starting models)
    starts = [
        (0, 0, 0, 0),
        (2, 2, 0, 0) if max_p >= 2 and max_q >= 2 else (1, 1, 0, 0),
        (1, 0, 0, 0),
        (0, 1, 0, 0),
    ]
    if s > 0:
        starts.extend([
            (0, 0, 1, 0),
            (0, 0, 0, 1),
            (1, 0, 1, 0),
            (0, 1, 0, 1),
        ])

    best_result = None
    best_ic = math.inf

    for p0, q0, P0, Q0 in starts:
        p0 = min(p0, max_p)
        q0 = min(q0, max_q)
        P0 = min(P0, max_P)
        Q0 = min(Q0, max_Q)
        res, ic = _eval(p0, q0, P0, Q0)
        if ic < best_ic:
            best_ic = ic
            best_result = res

    # Stepwise neighborhood search
    improved = True
    while improved:
        improved = False
        bp, bq, bP, bQ = None, None, None, None
        # Extract current best order
        if best_result:
            bp = best_result.model.order[0]
            bq = best_result.model.order[2]
            bP = best_result.model.seasonal_order[0]
            bQ = best_result.model.seasonal_order[2]
        else:
            break

        # Neighbors: p±1, q±1, P±1, Q±1
        neighbors = []
        for dp, dq, dP, dQ in [
            (1, 0, 0, 0), (-1, 0, 0, 0),
            (0, 1, 0, 0), (0, -1, 0, 0),
            (0, 0, 1, 0), (0, 0, -1, 0),
            (0, 0, 0, 1), (0, 0, 0, -1),
            (1, 1, 0, 0), (-1, -1, 0, 0),
        ]:
            np_ = bp + dp
            nq_ = bq + dq
            nP_ = bP + dP
            nQ_ = bQ + dQ
            if 0 <= np_ <= max_p and 0 <= nq_ <= max_q and \
               0 <= nP_ <= max_P and 0 <= nQ_ <= max_Q:
                neighbors.append((np_, nq_, nP_, nQ_))

        for np_, nq_, nP_, nQ_ in neighbors:
            res, ic = _eval(np_, nq_, nP_, nQ_)
            if ic < best_ic:
                best_ic = ic
                best_result = res
                improved = True

    return AutoARIMAResult(best_result, history, criterion)


def _grid_search(endog, d, D, s, max_p, max_q, max_P, max_Q,
                 trend, criterion, enforce_stationarity, enforce_invertibility,
                 method, maxiter, trace,
                 exog=None, simple_differencing=False):
    """Exhaustive grid search using Rust Rayon-parallel grid_search."""
    import rustima

    p_range = range(0, max_p + 1)
    q_range = range(0, max_q + 1)
    P_range = range(0, max_P + 1) if s > 0 else [0]
    Q_range = range(0, max_Q + 1) if s > 0 else [0]

    combos = list(itertools.product(p_range, q_range, P_range, Q_range))
    if len(combos) == 0:
        raise ValueError("No valid model combinations to evaluate.")

    order_list = [(p, d, q) for p, q, P, Q in combos]
    seasonal_list = [(P, D, Q, s) for p, q, P, Q in combos]

    kwargs = dict(
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
        trend=trend,
        simple_differencing=simple_differencing,
    )
    if exog is not None:
        kwargs["exog"] = exog
    if method is not None:
        kwargs["method"] = method
    if maxiter is not None:
        kwargs["maxiter"] = maxiter

    # Single Rust call — all combos fitted in parallel via Rayon
    raw_results = rustima.sarimax_grid_search(
        endog, order_list, seasonal_list, **kwargs,
    )

    history = []
    best_result = None
    best_ic = math.inf

    for raw in raw_results:
        order = tuple(raw["order"])
        seasonal = tuple(raw["seasonal_order"])
        p, _d, q = order
        P, _D, Q, _s = seasonal
        converged = raw.get("converged", False)
        has_error = "error" in raw

        if not has_error:
            ic_val = raw.get("aic", math.inf)
            if criterion == "bic":
                ic_val = raw.get("bic", math.inf)
            elif criterion == "hqic":
                # HQIC not in raw dict, compute from loglike
                ll = raw.get("loglike", -math.inf)
                k = raw.get("n_params", 0)
                n = raw.get("n_obs", 2)
                ic_val = -2.0 * ll + 2.0 * k * math.log(math.log(n)) if n > 1 else math.inf
            if not math.isfinite(ic_val):
                ic_val = math.inf
        else:
            ic_val = math.inf

        err_kw = {}
        if has_error:
            err_kw["error_type"] = raw.get("error_type", "RuntimeError")
            err_kw["error_message"] = raw.get("error", "unknown error")

        history.append(_make_history_entry(
            order, seasonal, criterion, ic_val, converged, **err_kw,
        ))
        if trace:
            _trace_print(p, _d, q, P, _D, Q, _s, criterion, ic_val)

        if ic_val < best_ic:
            best_ic = ic_val
            # Reconstruct SARIMAXResult from raw dict for the best model
            best_raw = raw
            best_order = order
            best_seasonal = seasonal

    # Build a proper SARIMAXResult for the best model
    if best_ic < math.inf:
        model = SARIMAXModel(
            endog, order=best_order, seasonal_order=best_seasonal,
            exog=exog,
            trend=trend,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
            simple_differencing=simple_differencing,
        )
        model._fit_result = True  # Mark as fitted
        from .model import SARIMAXResult
        best_result = SARIMAXResult(model, best_raw)

    return AutoARIMAResult(best_result, history, criterion)


class AutoARIMAResult:
    """Result of auto_arima search.

    Attributes
    ----------
    result : SARIMAXResult or None
        Best model fit result.
    history : list[dict]
        Search history with order, seasonal_order, and IC for each model.
    criterion : str
        IC used for selection.
    """

    def __init__(self, result, history, criterion):
        self.result = result
        self.history = history
        self.criterion = criterion

    @property
    def order(self):
        """Best model order (p, d, q)."""
        return self.result.model.order if self.result else None

    @property
    def seasonal_order(self):
        """Best model seasonal order (P, D, Q, s)."""
        return self.result.model.seasonal_order if self.result else None

    @property
    def best_ic(self):
        """Best information criterion value."""
        return _ic(self.result, self.criterion) if self.result else math.inf

    def summary(self, alpha=0.05, inference=None):
        """Return statsmodels-style summary of the best model.

        Parameters
        ----------
        alpha : float
            Significance level for confidence intervals.
        inference : str, optional
            ``"none"`` | ``"hessian"`` | ``"opg"`` | ``"both"``.
            Default ``"hessian"``.
        """
        if self.result is None:
            return "auto_arima: no valid model found."
        if inference is None:
            inference = "hessian"
        n_models = len(self.history)
        n_converged = sum(1 for h in self.history if h["converged"])
        body = self.result.summary(alpha=alpha, inference=inference)
        footer = (
            f"Search: {n_models} models evaluated "
            f"({n_converged} converged), criterion={self.criterion}"
        )
        return body + "\n" + footer

    def search_summary(self):
        """Short summary of the search (order, IC, model count)."""
        if self.result is None:
            return "auto_arima: no valid model found."
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order
        ic = self.best_ic
        n_models = len(self.history)
        n_converged = sum(1 for h in self.history if h["converged"])
        return (
            f"auto_arima: Best ARIMA({p},{d},{q})({P},{D},{Q})[{s}]\n"
            f"  {self.criterion}={ic:.3f}\n"
            f"  Models evaluated: {n_models} ({n_converged} converged)"
        )

    def history_dataframe(self):
        """Return search history as a Polars DataFrame."""
        import polars as pl

        orders = [str(h["order"]) for h in self.history]
        seasonals = [str(h["seasonal_order"]) for h in self.history]
        ics = [h.get(self.criterion) for h in self.history]
        converged = [h["converged"] for h in self.history]
        return pl.DataFrame({
            "order": orders,
            "seasonal_order": seasonals,
            self.criterion: ics,
            "converged": converged,
        })
