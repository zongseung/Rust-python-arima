"""Generate statsmodels reference fixtures for combination matrix testing.

Usage:
    .venv/bin/python python_tests/generate_matrix_fixtures.py [--tier a|b|all]

Produces:
    tests/fixtures/matrix_tier_a.json  (~30 models)
    tests/fixtures/matrix_tier_b.json  (~70 models, includes Tier A)
"""

import argparse
import json
import pathlib
import sys
import traceback
import warnings

import numpy as np
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Model combination definitions
# ---------------------------------------------------------------------------

# Tier A: PR gate (~30 models, <60s total)
TIER_A_COMBOS = [
    # --- Non-seasonal ---
    # Pure AR
    {"order": (1, 0, 0), "seasonal": (0, 0, 0, 0), "n": 200, "seed": 42},
    {"order": (2, 0, 0), "seasonal": (0, 0, 0, 0), "n": 200, "seed": 42},
    {"order": (3, 0, 0), "seasonal": (0, 0, 0, 0), "n": 200, "seed": 42},
    # Pure MA
    {"order": (0, 0, 1), "seasonal": (0, 0, 0, 0), "n": 200, "seed": 43},
    {"order": (0, 0, 2), "seasonal": (0, 0, 0, 0), "n": 200, "seed": 43},
    {"order": (0, 0, 3), "seasonal": (0, 0, 0, 0), "n": 200, "seed": 43},
    # ARMA
    {"order": (1, 0, 1), "seasonal": (0, 0, 0, 0), "n": 200, "seed": 44},
    {"order": (2, 0, 1), "seasonal": (0, 0, 0, 0), "n": 200, "seed": 44},
    {"order": (1, 0, 2), "seasonal": (0, 0, 0, 0), "n": 200, "seed": 44},
    {"order": (2, 0, 2), "seasonal": (0, 0, 0, 0), "n": 200, "seed": 44},
    # ARIMA
    {"order": (1, 1, 0), "seasonal": (0, 0, 0, 0), "n": 200, "seed": 45},
    {"order": (0, 1, 1), "seasonal": (0, 0, 0, 0), "n": 200, "seed": 45},
    {"order": (1, 1, 1), "seasonal": (0, 0, 0, 0), "n": 200, "seed": 45},
    {"order": (2, 1, 1), "seasonal": (0, 0, 0, 0), "n": 200, "seed": 45},
    {"order": (1, 1, 2), "seasonal": (0, 0, 0, 0), "n": 200, "seed": 45},
    # --- Seasonal s=4, D=0 ---
    {"order": (1, 0, 0), "seasonal": (1, 0, 0, 4), "n": 200, "seed": 50},
    {"order": (1, 0, 1), "seasonal": (1, 0, 1, 4), "n": 200, "seed": 50},
    {"order": (0, 0, 1), "seasonal": (0, 0, 1, 4), "n": 200, "seed": 50},
    # --- Seasonal s=4, D=1 ---
    {"order": (1, 1, 0), "seasonal": (1, 1, 0, 4), "n": 200, "seed": 51},
    {"order": (0, 1, 1), "seasonal": (0, 1, 1, 4), "n": 200, "seed": 51},
    {"order": (1, 1, 1), "seasonal": (1, 1, 1, 4), "n": 200, "seed": 51},
    # --- Seasonal s=12 ---
    {"order": (1, 0, 0), "seasonal": (1, 0, 0, 12), "n": 300, "seed": 60},
    {"order": (1, 1, 0), "seasonal": (1, 1, 0, 12), "n": 300, "seed": 60},
    {"order": (0, 1, 1), "seasonal": (0, 1, 1, 12), "n": 300, "seed": 60},
    {"order": (1, 1, 1), "seasonal": (1, 1, 1, 12), "n": 300, "seed": 60},
    # --- Seasonal s=7 ---
    {"order": (1, 0, 0), "seasonal": (1, 0, 0, 7), "n": 200, "seed": 70},
    {"order": (1, 1, 1), "seasonal": (1, 1, 1, 7), "n": 200, "seed": 70},
    # --- Higher order ---
    {"order": (3, 0, 3), "seasonal": (0, 0, 0, 0), "n": 300, "seed": 80},
    {"order": (3, 1, 3), "seasonal": (0, 0, 0, 0), "n": 300, "seed": 80},
    # --- ARIMA d=2 ---
    {"order": (1, 2, 1), "seasonal": (0, 0, 0, 0), "n": 200, "seed": 90},
]

# Tier B additional combos (on top of Tier A)
TIER_B_EXTRA_COMBOS = [
    # Extended AR
    {"order": (4, 0, 0), "seasonal": (0, 0, 0, 0), "n": 300, "seed": 100},
    {"order": (5, 0, 0), "seasonal": (0, 0, 0, 0), "n": 300, "seed": 100},
    # Extended MA
    {"order": (0, 0, 4), "seasonal": (0, 0, 0, 0), "n": 300, "seed": 101},
    {"order": (0, 0, 5), "seasonal": (0, 0, 0, 0), "n": 300, "seed": 101},
    # Mixed high order ARMA
    {"order": (3, 0, 1), "seasonal": (0, 0, 0, 0), "n": 300, "seed": 102},
    {"order": (1, 0, 3), "seasonal": (0, 0, 0, 0), "n": 300, "seed": 102},
    {"order": (3, 0, 2), "seasonal": (0, 0, 0, 0), "n": 300, "seed": 102},
    {"order": (2, 0, 3), "seasonal": (0, 0, 0, 0), "n": 300, "seed": 102},
    # ARIMA d=2
    {"order": (1, 2, 1), "seasonal": (0, 0, 0, 0), "n": 300, "seed": 103},
    {"order": (2, 2, 2), "seasonal": (0, 0, 0, 0), "n": 300, "seed": 103},
    # Seasonal s=2
    {"order": (1, 0, 0), "seasonal": (1, 0, 0, 2), "n": 200, "seed": 110},
    {"order": (1, 0, 1), "seasonal": (1, 0, 1, 2), "n": 200, "seed": 110},
    # Seasonal s=4 higher P/Q
    {"order": (1, 0, 0), "seasonal": (2, 0, 0, 4), "n": 200, "seed": 111},
    {"order": (0, 0, 1), "seasonal": (0, 0, 2, 4), "n": 200, "seed": 111},
    {"order": (1, 0, 1), "seasonal": (2, 0, 1, 4), "n": 300, "seed": 111},
    # Seasonal s=12 extended
    {"order": (2, 0, 0), "seasonal": (1, 0, 0, 12), "n": 300, "seed": 112},
    {"order": (0, 0, 2), "seasonal": (0, 0, 1, 12), "n": 300, "seed": 112},
    {"order": (2, 1, 1), "seasonal": (1, 1, 0, 12), "n": 300, "seed": 112},
    {"order": (1, 1, 2), "seasonal": (0, 1, 1, 12), "n": 300, "seed": 112},
    # Seasonal s=24 (hourly)
    {"order": (1, 0, 0), "seasonal": (1, 0, 0, 24), "n": 500, "seed": 113},
    {"order": (1, 1, 1), "seasonal": (1, 1, 1, 24), "n": 500, "seed": 113},
    # Short series (n=50)
    {"order": (1, 0, 0), "seasonal": (0, 0, 0, 0), "n": 50, "seed": 120},
    {"order": (1, 0, 1), "seasonal": (0, 0, 0, 0), "n": 50, "seed": 120},
    {"order": (1, 1, 1), "seasonal": (0, 0, 0, 0), "n": 80, "seed": 120},
    # Short seasonal (n=100)
    {"order": (1, 0, 0), "seasonal": (1, 0, 0, 4), "n": 100, "seed": 121},
    # Longer series (n=1000)
    {"order": (1, 0, 0), "seasonal": (0, 0, 0, 0), "n": 1000, "seed": 130},
    {"order": (1, 1, 1), "seasonal": (0, 0, 0, 0), "n": 1000, "seed": 130},
    {"order": (1, 0, 0), "seasonal": (1, 0, 0, 12), "n": 1000, "seed": 130},
    # d=1, D=1 combinations
    {"order": (2, 1, 0), "seasonal": (1, 1, 0, 4), "n": 200, "seed": 140},
    {"order": (0, 1, 2), "seasonal": (0, 1, 1, 4), "n": 200, "seed": 140},
    {"order": (2, 1, 2), "seasonal": (1, 1, 1, 4), "n": 300, "seed": 140},
    # Seasonal s=7 extended
    {"order": (2, 0, 0), "seasonal": (1, 0, 0, 7), "n": 200, "seed": 150},
    {"order": (0, 0, 2), "seasonal": (0, 0, 1, 7), "n": 200, "seed": 150},
    {"order": (1, 1, 0), "seasonal": (1, 1, 0, 7), "n": 200, "seed": 150},
    # Exog combos
    {"order": (1, 0, 0), "seasonal": (0, 0, 0, 0), "n": 200, "seed": 160, "n_exog": 2},
    {"order": (1, 1, 1), "seasonal": (0, 0, 0, 0), "n": 200, "seed": 160, "n_exog": 1},
    {"order": (1, 0, 0), "seasonal": (1, 0, 0, 4), "n": 200, "seed": 160, "n_exog": 2},
    {"order": (0, 1, 1), "seasonal": (0, 1, 1, 4), "n": 200, "seed": 160, "n_exog": 3},
    {"order": (1, 1, 1), "seasonal": (1, 1, 1, 12), "n": 300, "seed": 160, "n_exog": 2},
]


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def generate_data(order, seasonal, n, seed, n_exog=0):
    """Generate synthetic time series data appropriate for the model.

    Uses np.random.randn innovations. For seasonal models, adds seasonal
    component. For differenced models, uses cumulative sums.
    """
    np.random.seed(seed)
    p, d, q = order
    P, D, Q, s = seasonal

    # Base innovations
    e = np.random.randn(n + 100)  # extra burn-in
    y = np.zeros(n + 100)

    # Simple AR structure (first few lags)
    ar_coef = 0.5 if p >= 1 else 0.0
    ma_coef = 0.3 if q >= 1 else 0.0
    sar_coef = 0.4 if P >= 1 else 0.0

    for t in range(max(s if s > 0 else 1, p + 1), n + 100):
        val = e[t]
        if p >= 1:
            val += ar_coef * y[t - 1]
        if q >= 1 and t >= 1:
            val += ma_coef * e[t - 1]
        if s > 0 and P >= 1 and t >= s:
            val += sar_coef * y[t - s]
        y[t] = val

    # Trim burn-in
    y = y[100:]

    # Apply differencing (cumsum = inverse of diff)
    for _ in range(d):
        y = np.cumsum(y)
    if s > 0:
        for _ in range(D):
            # Seasonal cumsum
            result = np.zeros_like(y)
            for i in range(len(y)):
                result[i] = y[i] + (result[i - s] if i >= s else 0.0)
            y = result

    # Exogenous variables
    exog = None
    future_exog = None
    if n_exog > 0:
        np.random.seed(seed + 1000)
        exog = np.random.randn(n, n_exog) * 0.5
        beta = np.linspace(0.5, 1.5, n_exog)
        y += exog @ beta
        # Future exog for forecast
        np.random.seed(seed + 2000)
        future_exog = np.random.randn(10, n_exog) * 0.5

    return y, exog, future_exog


def make_model_id(order, seasonal, n, n_exog=0):
    """Create a unique model identifier string."""
    p, d, q = order
    P, D, Q, s = seasonal
    base = f"arima_{p}{d}{q}_{P}{D}{Q}{s}_n{n}"
    if n_exog > 0:
        base += f"_exog{n_exog}"
    return base


# ---------------------------------------------------------------------------
# Statsmodels oracle
# ---------------------------------------------------------------------------

def fit_oracle(y, order, seasonal, exog=None, future_exog=None):
    """Fit a model with statsmodels and extract reference values.

    Returns dict with all reference data, or None if fitting fails.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = sm.tsa.SARIMAX(
                y,
                exog=exog,
                order=order,
                seasonal_order=seasonal,
                trend="n",
                enforce_stationarity=False,
                enforce_invertibility=False,
                concentrate_scale=True,
            )
            res = model.fit(disp=False, maxiter=500)

            # Forecast
            fcast = res.get_forecast(steps=10, exog=future_exog)
            ci = fcast.conf_int(alpha=0.05)
            if hasattr(ci, 'iloc') and ci.shape[1] >= 2:
                ci_lower = ci.iloc[:, 0].tolist()
                ci_upper = ci.iloc[:, 1].tolist()
            elif hasattr(ci, '__len__') and len(ci) > 0:
                ci_lower = ci[:, 0].tolist()
                ci_upper = ci[:, 1].tolist()
            else:
                ci_lower = [float('nan')] * 10
                ci_upper = [float('nan')] * 10

            # Standardized residuals
            std_resid = res.filter_results.standardized_forecasts_error[0]

            # Forecast variance (MSE)
            forecast_var = fcast.var_pred_mean
            if hasattr(forecast_var, 'tolist'):
                forecast_var = forecast_var.tolist()
            else:
                forecast_var = list(forecast_var)

            # Fit concentrates the scale out (concentrate_scale=True), so
            # res.params omits sigma2.  The current rustima engine uses the full
            # non-concentrated layout [exog|ar|ma|sar|sma|sigma2], so append
            # sigma2 (= res.scale) as the trailing parameter.
            return {
                "params": res.params.tolist() + [float(res.scale)],
                "loglike": float(res.llf),
                "aic": float(res.aic),
                "bic": float(res.bic),
                "scale": float(res.scale),
                "converged": True,
                "n_obs": int(res.nobs),
                "forecast_mean": fcast.predicted_mean.tolist(),
                "forecast_variance": forecast_var,
                "forecast_ci_lower": ci_lower,
                "forecast_ci_upper": ci_upper,
                "standardized_residuals": std_resid.tolist(),
            }
        except Exception as exc:
            return {
                "error": str(exc),
                "converged": False,
            }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_tier(combos, tier_name):
    """Generate fixtures for a list of model combinations."""
    fixtures = []
    n_success = 0
    n_fail = 0

    for i, combo in enumerate(combos):
        order = combo["order"]
        seasonal = combo["seasonal"]
        n = combo["n"]
        seed = combo["seed"]
        n_exog = combo.get("n_exog", 0)

        model_id = make_model_id(order, seasonal, n, n_exog)
        print(f"  [{i+1}/{len(combos)}] {model_id} ...", end=" ", flush=True)

        try:
            y, exog, future_exog = generate_data(order, seasonal, n, seed, n_exog)

            result = fit_oracle(y, order, seasonal, exog, future_exog)

            entry = {
                "model_id": model_id,
                "order": list(order),
                "seasonal": list(seasonal),
                "n_obs": n,
                "seed": seed,
                "data": y.tolist(),
            }
            if exog is not None:
                entry["exog"] = exog.tolist()
                entry["n_exog"] = n_exog
            if future_exog is not None:
                entry["future_exog"] = future_exog.tolist()

            entry.update(result)
            fixtures.append(entry)

            if result.get("converged", False):
                n_success += 1
                print(f"OK (loglike={result['loglike']:.4f})")
            else:
                n_fail += 1
                print(f"FAIL ({result.get('error', 'unknown')})")
        except Exception as exc:
            n_fail += 1
            print(f"ERROR ({exc})")
            traceback.print_exc()
            fixtures.append({
                "model_id": model_id,
                "order": list(order),
                "seasonal": list(seasonal),
                "n_obs": n,
                "seed": seed,
                "converged": False,
                "error": str(exc),
            })

    print(f"\n{tier_name} summary: {n_success} OK, {n_fail} failed, {len(combos)} total")
    return fixtures


def main():
    parser = argparse.ArgumentParser(description="Generate matrix test fixtures")
    parser.add_argument("--tier", choices=["a", "b", "all"], default="all",
                        help="Which tier to generate (default: all)")
    args = parser.parse_args()

    out_dir = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.tier in ("a", "all"):
        print("=== Generating Tier A fixtures ===")
        tier_a = generate_tier(TIER_A_COMBOS, "Tier A")
        tier_a_path = out_dir / "matrix_tier_a.json"
        with open(tier_a_path, "w") as f:
            json.dump(tier_a, f, indent=2)
        print(f"Written to {tier_a_path}\n")

    if args.tier in ("b", "all"):
        print("=== Generating Tier B fixtures ===")
        all_combos = TIER_A_COMBOS + TIER_B_EXTRA_COMBOS
        tier_b = generate_tier(all_combos, "Tier B")
        tier_b_path = out_dir / "matrix_tier_b.json"
        with open(tier_b_path, "w") as f:
            json.dump(tier_b, f, indent=2)
        print(f"Written to {tier_b_path}")


if __name__ == "__main__":
    main()
