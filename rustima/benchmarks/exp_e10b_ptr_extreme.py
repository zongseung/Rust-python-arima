"""E10b ★★ HEADLINE STRONG: PTR vs joint MLE under EXTREME weak identification.

Designs three weak-id regimes:
  (i)  MILD     — moderate collinearity 0.8, scale mismatch 100x
  (ii) MODERATE — strong collinearity 0.95, scale mismatch 1000x
  (iii) EXTREME — near-perfect collinearity 0.995, scale mismatch 10000x,
                  small sample n=200

For each regime, compare:
- rustima lbfgsb (joint MLE)
- rustima profile-trust-region (PTR ★)
- statsmodels SARIMAX

Outputs:
- raw/e10b_ptr_extreme.csv
- raw/e10b_ptr_summary.csv
- tables/e10b_ptr_summary.tex
- figures/e10b_convergence.png
- figures/e10b_beta_mse.png
- figures/e10b_ll_gap.png
"""
from __future__ import annotations

import warnings

import numpy as np
import rustima
from statsmodels.tsa.statespace.sarimax import SARIMAX

from thesis_utils import FIGURES_DIR, Timer, csv_to_latex, save_csv, setup_mpl

warnings.filterwarnings("ignore")


REGIMES = {
    "mild": {
        "collinearity": 0.80,
        "scale_mismatch": 100.0,
        "n": 400,
        "beta_true": np.array([1.5, 0.01]),
    },
    "moderate": {
        "collinearity": 0.95,
        "scale_mismatch": 1000.0,
        "n": 400,
        "beta_true": np.array([1.5, 0.001]),
    },
    "extreme": {
        "collinearity": 0.995,
        "scale_mismatch": 10000.0,
        "n": 200,
        "beta_true": np.array([1.5, 0.0001]),
    },
}


def gen_weakly_id(n: int, beta_true, collinearity: float, scale_mismatch: float, phi=0.7, sigma=0.5, seed=0):
    """Generate SARIMAX-X with controlled weak identification.

    - x[:, 0] is correlation-controlled with the latent AR drift (collinearity ∈ [0, 1])
    - x[:, 1] has scale = scale_mismatch * std(x[:, 0])
    """
    rng = np.random.default_rng(seed)

    # First simulate the AR(1) latent drift
    eps = rng.standard_normal(n) * sigma
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = phi * z[t - 1] + eps[t]

    # x[:, 0]: correlated with z
    noise = rng.standard_normal(n) * z.std()
    coll = float(np.clip(collinearity, 0.0, 0.999))
    x0 = coll * z + np.sqrt(max(1 - coll * coll, 1e-12)) * noise

    # x[:, 1]: orthogonal but with large scale
    x1 = rng.standard_normal(n) * (x0.std() * scale_mismatch)
    X = np.column_stack([x0, x1]).astype(np.float64)

    # Build y
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + X[t] @ beta_true + eps[t] * 0.3  # mostly z contribution
    return y.astype(np.float64), X, beta_true


def fit_rustima(y, X, method, maxiter=300):
    try:
        with Timer() as t:
            r = rustima.sarimax_fit(
                y, (1, 0, 0), (0, 0, 0, 0),
                exog=X, method=method,
                concentrate_scale=True, maxiter=maxiter,
            )
        return {
            "ll": r["loglike"], "params": r["params"],
            "n_iter": r.get("n_iter", -1),
            "converged": r["converged"], "time_ms": t.ms,
            "error": None,
        }
    except Exception as e:
        return {"error": str(e)[:80], "time_ms": -1.0}


def fit_statsmodels(y, X, maxiter=300):
    try:
        with Timer() as t:
            m = SARIMAX(y, order=(1, 0, 0), exog=X, simple_differencing=False).fit(disp=False, maxiter=maxiter)
        params = m.params.values if hasattr(m.params, "values") else np.array(m.params)
        return {
            "ll": m.llf, "params": params,
            "n_iter": m.mle_retvals.get("iterations", -1) if hasattr(m, "mle_retvals") else -1,
            "converged": True, "time_ms": t.ms,
            "error": None,
        }
    except Exception as e:
        return {"error": str(e)[:80], "time_ms": -1.0}


def beta_mse(params, beta_true, layout_offset=0):
    n_beta = len(beta_true)
    try:
        beta_hat = np.array(params[layout_offset : layout_offset + n_beta], dtype=float)
        return float(np.mean((beta_hat - beta_true) ** 2))
    except Exception:
        return float("nan")


def main():
    N_TRIALS = 50
    all_rows = []
    summary_rows = []
    detailed = {}

    for regime_name, cfg in REGIMES.items():
        print(f"\n=== Regime: {regime_name} (coll={cfg['collinearity']}, scale={cfg['scale_mismatch']}, n={cfg['n']}) ===")
        beta_true = cfg["beta_true"]
        results = {"rustima_lbfgsb": [], "rustima_ptr": [], "statsmodels": []}

        for trial in range(N_TRIALS):
            y, X, _ = gen_weakly_id(
                n=cfg["n"], beta_true=beta_true,
                collinearity=cfg["collinearity"],
                scale_mismatch=cfg["scale_mismatch"],
                seed=trial + 1000,
            )
            r_lb = fit_rustima(y, X, "lbfgsb")
            r_pt = fit_rustima(y, X, "profile-trust-region")
            r_sm = fit_statsmodels(y, X)

            for eng, r in [("rustima_lbfgsb", r_lb), ("rustima_ptr", r_pt), ("statsmodels", r_sm)]:
                results[eng].append(r)
                if r.get("error"):
                    all_rows.append([regime_name, trial, eng, "ERR", "—", "—", "—", r.get("time_ms", -1)])
                else:
                    bm = beta_mse(r["params"], beta_true)
                    all_rows.append([
                        regime_name, trial, eng,
                        "OK" if r.get("converged", False) else "NOCONV",
                        f"{r['ll']:.4f}", f"{bm:.6f}",
                        r.get("n_iter", -1), f"{r['time_ms']:.1f}",
                    ])

        detailed[regime_name] = results

        # Per-regime summary
        for eng in ["rustima_lbfgsb", "rustima_ptr", "statsmodels"]:
            successes = [r for r in results[eng] if r.get("error") is None and r.get("converged", False)]
            n_ok = len(successes)
            if n_ok == 0:
                summary_rows.append([regime_name, eng, "0/50", "—", "—", "—", "—"])
                continue
            lls = [r["ll"] for r in successes]
            bms = [beta_mse(r["params"], beta_true) for r in successes]
            bms = [m for m in bms if not np.isnan(m)]
            iters = [r["n_iter"] for r in successes if r["n_iter"] > 0]
            times = [r["time_ms"] for r in successes]
            summary_rows.append([
                regime_name, eng,
                f"{n_ok}/{N_TRIALS}",
                f"{np.median(lls):.4f}",
                f"{np.median(bms):.6f}" if bms else "—",
                f"{np.median(iters):.0f}" if iters else "—",
                f"{np.median(times):.1f}",
            ])
            print(f"  {eng}: OK={n_ok}/{N_TRIALS}, LL_med={np.median(lls):.2f}, β-MSE={np.median(bms) if bms else float('nan'):.4e}, iter={np.median(iters) if iters else -1:.0f}, t={np.median(times):.1f}ms")

    save_csv(
        "e10b_ptr_extreme.csv",
        ["regime", "trial", "engine", "status", "loglike", "beta_mse", "n_iter", "time_ms"],
        all_rows,
    )
    save_csv(
        "e10b_ptr_summary.csv",
        ["regime", "engine", "convergence", "loglike_median", "beta_mse_median", "iter_median", "time_ms_median"],
        summary_rows,
    )
    csv_to_latex(
        "e10b_ptr_summary.csv",
        "e10b_ptr_summary.tex",
        "PTR vs.\\ joint MLE under three weak-identification regimes (mild, moderate, extreme). The \\emph{extreme} regime uses near-perfect collinearity (0.995) and 10{,}000-fold scale mismatch with $n=200$.",
        "tab:rustima:e10b-ptr",
    )

    # Figures
    plt = setup_mpl()
    engines = ["rustima_lbfgsb", "rustima_ptr", "statsmodels"]
    eng_labels = ["lbfgsb (joint MLE)", "PTR ★", "statsmodels"]
    eng_colors = ["#888888", "C0", "#bb6666"]

    # (1) Convergence rate per regime
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(REGIMES))
    width = 0.27
    for i, eng in enumerate(engines):
        rates = []
        for regime in REGIMES:
            res = detailed[regime][eng]
            rate = sum(1 for r in res if r.get("error") is None and r.get("converged", False)) / N_TRIALS * 100
            rates.append(rate)
        ax.bar(x + (i - 1) * width, rates, width, label=eng_labels[i], color=eng_colors[i])
        for j, r in enumerate(rates):
            ax.text(x[j] + (i - 1) * width, r + 2, f"{r:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(list(REGIMES.keys()))
    ax.set_ylabel("Convergence rate (%)")
    ax.set_ylim([0, 115])
    ax.set_title("Convergence under three weak-identification regimes")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(FIGURES_DIR / "e10b_convergence.png")
    plt.close(fig)

    # (2) β-MSE per regime (log scale, boxplot)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    positions = np.arange(len(REGIMES)) * (len(engines) + 1)
    for i, eng in enumerate(engines):
        data = []
        for regime in REGIMES:
            beta_true = REGIMES[regime]["beta_true"]
            res = detailed[regime][eng]
            bms = [beta_mse(r["params"], beta_true) for r in res if r.get("error") is None]
            bms = [m for m in bms if not np.isnan(m) and m > 0]
            data.append(bms if bms else [1e-10])
        pos = positions + i
        bp = ax.boxplot(data, positions=pos, widths=0.7, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(eng_colors[i])
            patch.set_alpha(0.6)
    ax.set_xticks(positions + 1)
    ax.set_xticklabels(list(REGIMES.keys()))
    ax.set_yscale("log")
    ax.set_ylabel(r"$\beta$ MSE (log scale)")
    ax.set_title("Recovery of $\\beta$ across regimes")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.6) for c in eng_colors]
    ax.legend(handles, eng_labels, fontsize=8, loc="upper left")
    fig.savefig(FIGURES_DIR / "e10b_beta_mse.png")
    plt.close(fig)

    # (3) LL gap relative to PTR per regime
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, eng in enumerate(engines):
        gaps = []
        for regime in REGIMES:
            res_e = detailed[regime][eng]
            res_p = detailed[regime]["rustima_ptr"]
            lls_e = [r["ll"] for r in res_e if r.get("error") is None and r.get("converged", False)]
            lls_p = [r["ll"] for r in res_p if r.get("error") is None and r.get("converged", False)]
            if lls_e and lls_p:
                gaps.append(np.median(lls_p) - np.median(lls_e))
            else:
                gaps.append(np.nan)
        ax.plot(list(REGIMES.keys()), gaps, "o-", label=eng_labels[i], color=eng_colors[i])
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("LL gap (PTR $\\ell$ − engine $\\ell$)")
    ax.set_title("Log-likelihood deficit of competing engines vs PTR")
    ax.legend(fontsize=8)
    fig.savefig(FIGURES_DIR / "e10b_ll_gap.png")
    plt.close(fig)

    print("\nFigures saved.")
    print("\nSummary table:")
    for r in summary_rows:
        print("  " + " | ".join(str(c) for c in r))


if __name__ == "__main__":
    main()
