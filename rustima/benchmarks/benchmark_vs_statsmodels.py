"""Comprehensive rustima vs statsmodels benchmark.

Measures:
1. Log-likelihood difference
2. Parameter estimate comparison (summary)
3. Fitting speed (wall time ratio)

Models tested:
- ARIMA: (1,0,0), (2,0,0), (1,1,1), (2,1,2), (0,1,1)
- SARIMA: s=12 (1,0,0)(1,0,0,12), s=24 (1,0,0)(1,0,0,24), s=7 (1,0,0)(1,0,0,7)
- ARIMAX: (1,0,0) + 1 exog, (2,0,1) + 2 exog
- SARIMAX: (1,0,0)(1,0,0,12) + 1 exog

auto_arima tested:
- non-seasonal (max_p=4, max_q=4) vs pmdarima
- seasonal s=12 (max_p=3, max_q=3, max_P=2, max_Q=2) vs pmdarima
- seasonal s=7  vs pmdarima
"""

import time
import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import rustima
from statsmodels.tsa.statespace.sarimax import SARIMAX as SM_SARIMAX

warnings.filterwarnings("ignore")

RNG = np.random.default_rng(42)
N = 300  # sample size for most models
N_SEASONAL_24 = 480  # need more data for s=24
REPEAT = 5  # timing repeats


# ─────────────────────────────────────────────────────────────────────────────
# Data generators
# ─────────────────────────────────────────────────────────────────────────────

def gen_arma(n, ar=(0.6,), ma=(-0.3,), sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    eps = rng.normal(scale=sigma, size=n + 50)
    y = np.zeros(n + 50)
    p = len(ar); q = len(ma)
    for t in range(max(p, q), n + 50):
        y[t] = sum(ar[i] * y[t-1-i] for i in range(p)) + eps[t] + sum(ma[i] * eps[t-1-i] for i in range(q))
    return y[-n:]

def gen_arima(n, d=1, seed=1):
    y = gen_arma(n + d, seed=seed)
    return np.cumsum(y)[:n]

def gen_sarima(n, s=12, seed=2):
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=n + s * 2)
    y = np.zeros(n + s * 2)
    for t in range(s, n + s * 2):
        y[t] = 0.5 * y[t-1] + 0.4 * y[t-s] - 0.2 * y[t-s-1] + eps[t] + 0.3 * eps[t-s]
    return y[-n:]

def gen_sarima24(n, seed=3):
    return gen_sarima(n, s=24, seed=seed)

def gen_sarima7(n, seed=4):
    return gen_sarima(n, s=7, seed=seed)


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    label: str
    order: tuple
    seasonal_order: tuple
    n_obs: int
    n_params: int
    # Log-likelihood
    rs_llf: float = np.nan
    sm_llf: float = np.nan
    llf_diff: float = np.nan
    llf_rel_pct: float = np.nan
    # AIC
    rs_aic: float = np.nan
    sm_aic: float = np.nan
    aic_diff: float = np.nan
    # Params
    rs_params: list = field(default_factory=list)
    sm_params: list = field(default_factory=list)
    param_max_abs_diff: float = np.nan
    param_mean_abs_diff: float = np.nan
    # Speed
    rs_time: float = np.nan
    sm_time: float = np.nan
    speedup: float = np.nan
    # Status
    rs_converged: bool = False
    sm_converged: bool = False
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def run_bench(label, order, seasonal_order, y, exog=None, n_reps=REPEAT):
    p, d, q = order
    P, D, Q, s = seasonal_order
    n = len(y)
    k_ar = p; k_ma = q; k_sar = P; k_sma = Q
    k_exog = exog.shape[1] if exog is not None else 0
    n_params = k_ar + k_ma + k_sar + k_sma + k_exog
    r = BenchResult(label=label, order=order, seasonal_order=seasonal_order,
                    n_obs=n, n_params=n_params)
    try:
        # ── rustima ────────────────────────────────────────────────────
        times_rs = []
        rs_result = None
        for _ in range(n_reps):
            t0 = time.perf_counter()
            kw = {}
            if exog is not None:
                kw["exog"] = exog.astype(np.float64)
            rs_result = rustima.sarimax_fit(
                y.astype(np.float64), order, seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
                **kw
            )
            times_rs.append(time.perf_counter() - t0)

        r.rs_llf = rs_result["loglike"]
        r.rs_aic = rs_result["aic"]
        r.rs_params = list(rs_result["params"])
        r.rs_converged = rs_result["converged"]
        r.rs_time = np.median(times_rs)

        # ── statsmodels ───────────────────────────────────────────────────
        times_sm = []
        sm_result = None
        for _ in range(n_reps):
            t0 = time.perf_counter()
            mod = SM_SARIMAX(
                y, order=order, seasonal_order=seasonal_order,
                exog=exog,
                enforce_stationarity=False, enforce_invertibility=False,
                concentrate_scale=True,
            )
            sm_result = mod.fit(disp=False)
            times_sm.append(time.perf_counter() - t0)

        r.sm_llf = sm_result.llf
        r.sm_aic = sm_result.aic
        # statsmodels params exclude sigma2 when concentrate_scale=True;
        # slice to match rustima param count (exclude last if mismatch)
        sm_p = np.array(sm_result.params)
        if len(sm_p) > n_params:
            sm_p = sm_p[:n_params]
        r.sm_params = list(sm_p)
        r.sm_converged = sm_result.mle_retvals.get("converged", False)
        r.sm_time = np.median(times_sm)

        # ── Differences ───────────────────────────────────────────────────
        r.llf_diff = r.rs_llf - r.sm_llf
        r.llf_rel_pct = 100.0 * abs(r.llf_diff) / max(abs(r.sm_llf), 1e-6)
        r.aic_diff = r.rs_aic - r.sm_aic
        min_k = min(len(r.rs_params), len(r.sm_params))
        if min_k > 0:
            diffs = np.abs(np.array(r.rs_params[:min_k]) - np.array(r.sm_params[:min_k]))
            r.param_max_abs_diff = float(np.max(diffs))
            r.param_mean_abs_diff = float(np.mean(diffs))
        r.speedup = r.sm_time / r.rs_time if r.rs_time > 0 else np.nan

    except Exception as e:
        r.error = str(e)[:120]
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────────────

def build_cases():
    cases = []

    # ── ARIMA ──────────────────────────────────────────────────────────────
    y_ar1  = gen_arma(N, ar=(0.6,), seed=10)
    y_ar2  = gen_arma(N, ar=(0.6, -0.2), seed=11)
    y_ma1  = gen_arma(N, ma=(-0.5,), seed=12)
    y_arma = gen_arma(N, ar=(0.5,), ma=(-0.3,), seed=13)
    y_arma21 = gen_arma(N, ar=(0.5, -0.2), ma=(-0.4,), seed=14)
    y_arima = gen_arima(N, d=1, seed=15)
    y_arima212 = gen_arima(N, d=1, seed=16)

    cases += [
        ("ARIMA(1,0,0)", (1,0,0), (0,0,0,0), y_ar1,  None),
        ("ARIMA(2,0,0)", (2,0,0), (0,0,0,0), y_ar2,  None),
        ("ARIMA(0,0,1)", (0,0,1), (0,0,0,0), y_ma1,  None),
        ("ARIMA(1,0,1)", (1,0,1), (0,0,0,0), y_arma, None),
        ("ARIMA(2,0,1)", (2,0,1), (0,0,0,0), y_arma21, None),
        ("ARIMA(1,1,1)", (1,1,1), (0,0,0,0), y_arima,  None),
        ("ARIMA(2,1,2)", (2,1,2), (0,0,0,0), y_arima212, None),
    ]

    # ── SARIMA ─────────────────────────────────────────────────────────────
    y_s12  = gen_sarima(N,             s=12, seed=20)
    y_s7   = gen_sarima(N,             s=7,  seed=21)
    y_s24  = gen_sarima24(N_SEASONAL_24,     seed=22)

    cases += [
        ("SARIMA(1,0,0)(1,0,0)12", (1,0,0), (1,0,0,12), y_s12, None),
        ("SARIMA(1,0,1)(1,0,1)12", (1,0,1), (1,0,1,12), y_s12, None),
        ("SARIMA(1,1,1)(1,1,0)12", (1,1,1), (1,1,0,12), y_s12, None),
        ("SARIMA(1,0,0)(1,0,0)7",  (1,0,0), (1,0,0,7),  y_s7,  None),
        ("SARIMA(1,0,1)(0,1,1)7",  (1,0,1), (0,1,1,7),  y_s7,  None),
        ("SARIMA(1,0,0)(1,0,0)24", (1,0,0), (1,0,0,24), y_s24, None),
        ("SARIMA(2,0,1)(1,0,0)24", (2,0,1), (1,0,0,24), y_s24, None),
    ]

    # ── ARIMAX ─────────────────────────────────────────────────────────────
    rng = np.random.default_rng(30)
    x1 = rng.normal(size=(N, 1))
    x2 = rng.normal(size=(N, 2))
    y_ax1 = gen_arma(N, ar=(0.5,), seed=31) + 1.5 * x1[:, 0]
    y_ax2 = gen_arma(N, ar=(0.4,), ma=(-0.2,), seed=32) + 0.8 * x2[:, 0] - 0.6 * x2[:, 1]

    cases += [
        ("ARIMAX(1,0,0)+1x",  (1,0,0), (0,0,0,0), y_ax1, x1),
        ("ARIMAX(1,0,1)+2x",  (1,0,1), (0,0,0,0), y_ax2, x2),
        ("ARIMAX(2,1,1)+1x",  (2,1,1), (0,0,0,0), np.cumsum(y_ax1), x1),
    ]

    # ── SARIMAX ────────────────────────────────────────────────────────────
    x3 = rng.normal(size=(N, 1))
    y_sx = gen_sarima(N, s=12, seed=40) + 1.2 * x3[:, 0]
    x4 = rng.normal(size=(N_SEASONAL_24, 1))
    y_sx24 = gen_sarima24(N_SEASONAL_24, seed=41) + 0.9 * x4[:, 0]

    cases += [
        ("SARIMAX(1,0,0)(1,0,0)12+1x", (1,0,0), (1,0,0,12), y_sx,  x3),
        ("SARIMAX(1,0,1)(1,0,1)12+1x", (1,0,1), (1,0,1,12), y_sx,  x3),
        ("SARIMAX(1,0,0)(1,0,0)24+1x", (1,0,0), (1,0,0,24), y_sx24, x4),
    ]

    return cases


# ─────────────────────────────────────────────────────────────────────────────
# auto_arima benchmark
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AutoBenchResult:
    label: str
    n_obs: int
    s: int
    # Selected orders
    rs_order: Optional[tuple] = None
    rs_seasonal: Optional[tuple] = None
    pm_order: Optional[tuple] = None
    pm_seasonal: Optional[tuple] = None
    order_match: bool = False
    # AIC from each library's own engine (NOT comparable cross-engine)
    rs_aic_own: float = np.nan   # rustima AIC for rs-selected model
    pm_aic_own: float = np.nan   # pmdarima AIC for pm-selected model
    # AIC re-fit with rustima engine for FAIR cross-model comparison
    rs_aic_refit: float = np.nan  # rustima re-fit of rs-selected model
    pm_aic_refit: float = np.nan  # rustima re-fit of pm-selected model
    aic_diff_refit: float = np.nan  # rs_aic_refit - pm_aic_refit (fair)
    # Models evaluated
    rs_n_models: int = 0
    # Speed
    rs_time: float = np.nan
    pm_time: float = np.nan
    speedup: float = np.nan
    error: Optional[str] = None


def run_auto_bench(label, y, s, max_p=3, max_q=3, max_P=2, max_Q=2,
                   max_d=2, max_D=1, n_reps=3):
    from rustima.auto import auto_arima as rs_auto
    try:
        import pmdarima as pm
    except ImportError:
        r = AutoBenchResult(label=label, n_obs=len(y), s=s)
        r.error = "pmdarima not installed"
        return r

    r = AutoBenchResult(label=label, n_obs=len(y), s=s)
    seasonal = s > 1

    try:
        # ── rustima auto_arima ─────────────────────────────────────────
        times_rs = []
        rs_res = None
        for _ in range(n_reps):
            t0 = time.perf_counter()
            rs_res = rs_auto(
                y, s=s, max_p=max_p, max_q=max_q,
                max_P=max_P if seasonal else 0,
                max_Q=max_Q if seasonal else 0,
                max_d=max_d, max_D=max_D if seasonal else 0,
                stepwise=True, criterion="aic",
            )
            times_rs.append(time.perf_counter() - t0)

        r.rs_order = rs_res.order
        r.rs_seasonal = rs_res.seasonal_order
        r.rs_aic_own = rs_res.best_ic
        r.rs_n_models = len(rs_res.history)
        r.rs_time = np.median(times_rs)

        # ── pmdarima auto_arima ───────────────────────────────────────────
        times_pm = []
        pm_res = None
        for _ in range(n_reps):
            t0 = time.perf_counter()
            pm_res = pm.auto_arima(
                y, m=s if seasonal else 1,
                start_p=0, start_q=0,
                start_P=0, start_Q=0,
                max_p=max_p, max_q=max_q,
                max_P=max_P if seasonal else 0,
                max_Q=max_Q if seasonal else 0,
                max_d=max_d, max_D=max_D if seasonal else 0,
                stepwise=True, information_criterion="aic",
                error_action="ignore", suppress_warnings=True,
            )
            times_pm.append(time.perf_counter() - t0)

        r.pm_order = pm_res.order
        r.pm_seasonal = pm_res.seasonal_order
        r.pm_aic_own = pm_res.aic()
        r.pm_time = np.median(times_pm)

        # ── Fair cross-model comparison: re-fit BOTH with rustima ────
        # rustima AIC for rs-selected model
        rs_refit = rustima.sarimax_fit(
            y.astype(np.float64), r.rs_order, r.rs_seasonal,
            enforce_stationarity=False, enforce_invertibility=False,
        )
        r.rs_aic_refit = rs_refit["aic"]

        # rustima AIC for pm-selected model
        pm_seasonal_full = r.pm_seasonal if len(r.pm_seasonal) == 4 else (*r.pm_seasonal, s)
        pm_refit = rustima.sarimax_fit(
            y.astype(np.float64), r.pm_order, pm_seasonal_full,
            enforce_stationarity=False, enforce_invertibility=False,
        )
        r.pm_aic_refit = pm_refit["aic"]

        # ── Summary metrics ──────────────────────────────────────────────
        r.order_match = (r.rs_order == r.pm_order and
                         r.rs_seasonal[:3] == r.pm_seasonal[:3])
        r.aic_diff_refit = r.rs_aic_refit - r.pm_aic_refit  # same engine → fair
        r.speedup = r.pm_time / r.rs_time if r.rs_time > 0 else np.nan

    except Exception as e:
        r.error = str(e)[:120]
    return r


def build_auto_cases():
    cases = []

    # non-seasonal
    y_ns = gen_arma(N, ar=(0.6,), ma=(-0.3,), seed=50)
    cases.append(("AutoARIMA non-seasonal", y_ns, 0))

    # seasonal s=12
    y_s12 = gen_sarima(N, s=12, seed=51)
    cases.append(("AutoARIMA s=12", y_s12, 12))

    # seasonal s=7
    y_s7 = gen_sarima(N, s=7, seed=52)
    cases.append(("AutoARIMA s=7", y_s7, 7))

    # seasonal s=24 (need more data)
    y_s24 = gen_sarima24(N_SEASONAL_24, seed=54)
    cases.append(("AutoARIMA s=24", y_s24, 24))

    # I(1) process
    y_i1 = gen_arima(N, d=1, seed=53)
    cases.append(("AutoARIMA I(1)", y_i1, 0))

    return cases


def run_all_auto():
    cases = build_auto_cases()
    results = []
    for i, (label, y, s) in enumerate(cases):
        print(f"  [{i+1}/{len(cases)}] {label} ... ", end="", flush=True)
        r = run_auto_bench(label, y, s)
        if r.error:
            print(f"ERROR: {r.error[:60]}")
        else:
            match = "✓" if r.order_match else "✗"
            print(
                f"rs={r.rs_order}  pm={r.pm_order}  "
                f"ΔAIC(refit)={r.aic_diff_refit:+.3f}  match={match}  speedup={r.speedup:.1f}x"
            )
        results.append(r)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cases = build_cases()
    results = []
    for i, (label, order, seasonal, y, exog) in enumerate(cases):
        print(f"[{i+1:2d}/{len(cases)}] {label} ... ", end="", flush=True)
        r = run_bench(label, order, seasonal, y, exog=exog)
        if r.error:
            print(f"ERROR: {r.error[:60]}")
        else:
            print(f"llf_diff={r.llf_diff:+.4f}  param_max={r.param_max_abs_diff:.4f}  speedup={r.speedup:.1f}x")
        results.append(r)
    return results


def main_auto():
    print("\n=== auto_arima benchmark (vs pmdarima) ===")
    return run_all_auto()


if __name__ == "__main__":
    results = main()
    auto_results = main_auto()
