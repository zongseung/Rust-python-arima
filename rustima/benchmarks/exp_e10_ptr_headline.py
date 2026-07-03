"""E10 ★ HEADLINE: PTR vs joint MLE — weakly-identified β SARIMAX.

Designs synthetic SARIMAX-X data where the exog block is weakly identified
relative to the ARMA structure, then compares:
- rustima with method="lbfgsb" (joint MLE)
- rustima with method="profile-trust-region" (PTR)
- statsmodels SARIMAX
- (R forecast::Arima via Rscript subprocess, optional — skipped if R unavailable)

Outputs:
- raw/e10_ptr.csv
- raw/e10_ptr_summary.csv
- figures/e10_ptr_convergence.png
- figures/e10_ptr_beta_recovery.png
"""
from __future__ import annotations

import warnings

import numpy as np
import rustima
from statsmodels.tsa.statespace.sarimax import SARIMAX

from thesis_utils import FIGURES_DIR, Timer, csv_to_latex, save_csv, setup_mpl

warnings.filterwarnings("ignore")


def gen_weakly_identified(n=300, beta_true=None, phi=0.7, sigma=0.5, seed=0, collinear=True):
    """Generate SARIMAX-X data with weakly-identified β.

    Weak identification mechanism:
    - exog X[:, 0] is highly correlated with the slow AR drift (collinear=True)
    - exog X[:, 1] has scale 100x larger than X[:, 0] (scale mismatch)

    This makes the likelihood surface flat in the β direction.
    """
    if beta_true is None:
        beta_true = np.array([1.5, 0.01])

    rng = np.random.default_rng(seed)
    n_exog = len(beta_true)
    X = np.zeros((n, n_exog))

    # First column: nearly collinear with AR trend
    z = rng.standard_normal(n)
    X[:, 0] = np.cumsum(z) * 0.3 if collinear else z
    # Second column: large scale
    X[:, 1] = rng.standard_normal(n) * 100.0

    eps = rng.standard_normal(n) * sigma
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + X[t] @ beta_true + eps[t]

    return y.astype(np.float64), X.astype(np.float64), beta_true


def fit_rustima(y, X, method):
    try:
        with Timer() as t:
            r = rustima.sarimax_fit(
                y,
                (1, 0, 0),
                (0, 0, 0, 0),
                exog=X,
                method=method,
                concentrate_scale=True,
                maxiter=200,
            )
        return {
            "ll": r["loglike"],
            "params": r["params"],
            "n_iter": r.get("n_iter", -1),
            "converged": r["converged"],
            "time_ms": t.ms,
        }
    except Exception as e:
        return {"error": str(e), "time_ms": -1.0}


def fit_statsmodels(y, X):
    try:
        with Timer() as t:
            m = SARIMAX(y, order=(1, 0, 0), exog=X, simple_differencing=False).fit(disp=False, maxiter=200)
        return {
            "ll": m.llf,
            "params": m.params.values if hasattr(m.params, "values") else np.array(m.params),
            "n_iter": m.mle_retvals.get("iterations", -1) if hasattr(m, "mle_retvals") else -1,
            "converged": True,
            "time_ms": t.ms,
        }
    except Exception as e:
        return {"error": str(e), "time_ms": -1.0}


def beta_mse(estimated_params, beta_true, layout_offset=0):
    """Extract β from full param vector (assuming [trend?, exog, ar, ma, sar, sma])."""
    n_beta = len(beta_true)
    try:
        beta_hat = np.array(estimated_params[layout_offset : layout_offset + n_beta], dtype=float)
        return float(np.mean((beta_hat - beta_true) ** 2))
    except Exception:
        return float("nan")


def main():
    N_TRIALS = 50
    beta_true = np.array([1.5, 0.01])
    n = 400

    results = {
        "rustima_lbfgsb": [],
        "rustima_ptr": [],
        "statsmodels": [],
    }

    rows = []
    for trial in range(N_TRIALS):
        y, X, _ = gen_weakly_identified(n=n, beta_true=beta_true, seed=trial)

        rs_lbfgsb = fit_rustima(y, X, "lbfgsb")
        rs_ptr = fit_rustima(y, X, "profile-trust-region")
        sm = fit_statsmodels(y, X)

        results["rustima_lbfgsb"].append(rs_lbfgsb)
        results["rustima_ptr"].append(rs_ptr)
        results["statsmodels"].append(sm)

        for engine, r in [("rustima_lbfgsb", rs_lbfgsb), ("rustima_ptr", rs_ptr), ("statsmodels", sm)]:
            if "error" in r:
                rows.append([trial, engine, "ERR", "—", "—", "—", r.get("time_ms", -1)])
            else:
                bm = beta_mse(r["params"], beta_true, layout_offset=0)
                rows.append([
                    trial,
                    engine,
                    "OK" if r["converged"] else "NOCONV",
                    f"{r['ll']:.4f}",
                    f"{bm:.4f}",
                    r["n_iter"],
                    f"{r['time_ms']:.1f}",
                ])

    save_csv(
        "e10_ptr.csv",
        ["trial", "engine", "status", "loglike", "beta_mse", "n_iter", "time_ms"],
        rows,
    )

    # Summary
    summary_rows = []
    for engine in ["rustima_lbfgsb", "rustima_ptr", "statsmodels"]:
        ok = [r for r in results[engine] if "error" not in r and r.get("converged", False)]
        n_ok = len(ok)
        if n_ok == 0:
            summary_rows.append([engine, 0, "—", "—", "—", "—"])
            continue
        lls = [r["ll"] for r in ok]
        beta_mses = [beta_mse(r["params"], beta_true, 0) for r in ok]
        beta_mses = [m for m in beta_mses if not np.isnan(m)]
        times = [r["time_ms"] for r in ok]
        iters = [r["n_iter"] for r in ok if r["n_iter"] > 0]
        summary_rows.append([
            engine,
            f"{n_ok}/{N_TRIALS}",
            f"{np.median(lls):.4f}",
            f"{np.median(beta_mses):.4f}" if beta_mses else "—",
            f"{np.median(iters):.1f}" if iters else "—",
            f"{np.median(times):.1f}",
        ])
        print(f"  {engine}: success={n_ok}/{N_TRIALS}  LL_med={np.median(lls):.2f}  β-MSE_med={np.median(beta_mses) if beta_mses else float('nan'):.4f}  iters_med={np.median(iters) if iters else -1:.0f}")

    save_csv(
        "e10_ptr_summary.csv",
        ["engine", "convergence", "loglike_median", "beta_mse_median", "iter_median", "time_ms_median"],
        summary_rows,
    )
    csv_to_latex(
        "e10_ptr_summary.csv",
        "e10_ptr_summary.tex",
        "PTR vs.\\ joint MLE on weakly-identified $\\beta$ ($n=400$, 50 trials, true $\\beta=(1.5, 0.01)$ with collinear $x_1$ and scale-mismatched $x_2$).",
        "tab:rustima:e10-ptr",
    )

    # Convergence figure
    plt = setup_mpl()
    fig, ax = plt.subplots()
    engines = ["rustima_lbfgsb", "rustima_ptr", "statsmodels"]
    succ = [sum(1 for r in results[e] if "error" not in r and r.get("converged", False)) / N_TRIALS * 100 for e in engines]
    colors = ["#888888", "C0", "#bb6666"]
    ax.bar(["rustima\nlbfgsb", "rustima\nPTR ★", "statsmodels"], succ, color=colors)
    ax.set_ylabel("Convergence rate (%)")
    ax.set_ylim([0, 105])
    ax.set_title(f"Convergence under weakly-identified $\\beta$ (n={n}, {N_TRIALS} trials)")
    for i, v in enumerate(succ):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=9)
    fig.savefig(FIGURES_DIR / "e10_ptr_convergence.png")
    plt.close(fig)

    # β recovery
    fig, ax = plt.subplots()
    labels = []
    data = []
    for e, label in zip(engines, ["lbfgsb", "PTR ★", "statsmodels"]):
        bm = [beta_mse(r["params"], beta_true, 0) for r in results[e] if "error" not in r]
        bm = [m for m in bm if not np.isnan(m)]
        if bm:
            data.append(bm)
            labels.append(label)
    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, c in zip(bp["boxes"], ["#888888", "C0", "#bb6666"]):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_ylabel(r"$\beta$ MSE (lower is better)")
        ax.set_yscale("log")
        ax.set_title("Recovery of true $\\beta$ under weak identification")
        fig.savefig(FIGURES_DIR / "e10_ptr_beta_recovery.png")
    plt.close(fig)
    print("\nFigures saved.")


if __name__ == "__main__":
    main()
