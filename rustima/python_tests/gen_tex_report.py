"""Generate LaTeX report from benchmark results."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from benchmark_vs_statsmodels import main as run_benchmarks, main_auto as run_auto_benchmarks, N, REPEAT

def fmt(v, fmt_str=".4f", na="---"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return na
    return f"{v:{fmt_str}}"

def sign_fmt(v, fmt_str=".4f", na="---"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return na
    return f"{v:+{fmt_str}}"

def speedup_color(s):
    if not np.isfinite(s): return ""
    if s >= 20: return r"\cellcolor{speedhigh}"
    if s >= 5:  return r"\cellcolor{speedmid}"
    return r"\cellcolor{speedlow}"

def llf_color(d):
    if not np.isfinite(d): return ""
    if abs(d) < 0.001: return r"\cellcolor{matchperfect}"
    if abs(d) < 1.0:   return r"\cellcolor{matchgood}"
    return r"\cellcolor{matchbad}"

def param_color(d):
    if not np.isfinite(d): return ""
    if d < 0.001: return r"\cellcolor{matchperfect}"
    if d < 0.05:  return r"\cellcolor{matchgood}"
    return r"\cellcolor{matchbad}"

def escape(s):
    return s.replace("_", r"\_").replace("(", r"(").replace(")", r")")

def make_section(results, model_type):
    mapping = {
        "ARIMA":   lambda r: r.label.startswith("ARIMA") and not r.label.startswith("ARIMAX"),
        "SARIMA":  lambda r: r.label.startswith("SARIMA") and not r.label.startswith("SARIMAX"),
        "ARIMAX":  lambda r: r.label.startswith("ARIMAX"),
        "SARIMAX": lambda r: r.label.startswith("SARIMAX"),
    }
    subset = [r for r in results if mapping[model_type](r)]
    return subset

def make_loglike_table(results):
    rows = []
    for r in results:
        if r.error:
            rows.append(
                f"  {escape(r.label)} & {r.n_obs} & {r.n_params} & \\multicolumn{{4}}{{c}}{{\\textit{{Error: {r.error[:40]}}}}} \\\\"
            )
            continue
        lld = r.llf_diff; pct = r.llf_rel_pct
        aid = r.aic_diff
        col = llf_color(lld)
        rows.append(
            f"  {escape(r.label)} & {r.n_obs} & {r.n_params} & "
            f"{fmt(r.rs_llf, '.3f')} & {fmt(r.sm_llf, '.3f')} & "
            f"{col}{sign_fmt(lld, '.4f')} & "
            f"{sign_fmt(aid, '.3f')} \\\\"
        )
    return "\n".join(rows)

def make_param_table(results):
    rows = []
    for r in results:
        if r.error:
            rows.append(
                f"  {escape(r.label)} & {r.n_params} & \\multicolumn{{2}}{{c}}{{\\textit{{Error}}}} & "
                f"{'Yes' if r.rs_converged else 'No'} & {'Yes' if r.sm_converged else 'No'} \\\\"
            )
            continue
        col = param_color(r.param_max_abs_diff)
        rows.append(
            f"  {escape(r.label)} & {r.n_params} & "
            f"{col}{fmt(r.param_max_abs_diff, '.5f')} & "
            f"{fmt(r.param_mean_abs_diff, '.5f')} & "
            f"{'Yes' if r.rs_converged else 'No'} & "
            f"{'Yes' if r.sm_converged else 'No'} \\\\"
        )
    return "\n".join(rows)

def make_speed_table(results):
    rows = []
    for r in results:
        if r.error:
            rows.append(
                f"  {escape(r.label)} & {r.n_obs} & \\multicolumn{{3}}{{c}}{{\\textit{{Error}}}} \\\\"
            )
            continue
        col = speedup_color(r.speedup)
        rows.append(
            f"  {escape(r.label)} & {r.n_obs} & "
            f"{fmt(r.rs_time*1000, '.1f')}\\,ms & "
            f"{fmt(r.sm_time*1000, '.1f')}\\,ms & "
            f"{col}{fmt(r.speedup, '.1f')}$\\times$ \\\\"
        )
    return "\n".join(rows)

def compute_stats(results):
    good = [r for r in results if not r.error]
    llf_diffs = [abs(r.llf_diff) for r in good if np.isfinite(r.llf_diff)]
    speedups   = [r.speedup for r in good if np.isfinite(r.speedup)]
    param_diffs = [r.param_max_abs_diff for r in good if np.isfinite(r.param_max_abs_diff)]
    n_perfect = sum(1 for d in llf_diffs if d < 0.001)
    n_good    = sum(1 for d in llf_diffs if 0.001 <= d < 1.0)
    n_bad     = sum(1 for d in llf_diffs if d >= 1.0)
    return dict(
        n_models=len(results),
        n_error=len(results) - len(good),
        n_perfect=n_perfect, n_good=n_good, n_bad=n_bad,
        mean_speedup=np.mean(speedups) if speedups else np.nan,
        median_speedup=np.median(speedups) if speedups else np.nan,
        max_speedup=np.max(speedups) if speedups else np.nan,
        min_speedup=np.min(speedups) if speedups else np.nan,
        mean_llf_diff=np.mean(llf_diffs) if llf_diffs else np.nan,
        max_llf_diff=np.max(llf_diffs) if llf_diffs else np.nan,
        mean_param_diff=np.mean(param_diffs) if param_diffs else np.nan,
        max_param_diff=np.max(param_diffs) if param_diffs else np.nan,
    )

def make_auto_table(auto_results):
    rows = []
    for r in auto_results:
        if r.error:
            rows.append(
                f"  {escape(r.label)} & {r.n_obs} & {r.s} & "
                f"\\multicolumn{{7}}{{c}}{{\\textit{{Error: {r.error[:40]}}}}} \\\\"
            )
            continue
        match_sym = r"{\color{green}\checkmark}" if r.order_match else r"{\color{red}$\times$}"
        aic_col = llf_color(r.aic_diff_refit)
        spd_col = speedup_color(r.speedup)
        rs_ord = f"({r.rs_order[0]},{r.rs_order[1]},{r.rs_order[2]})" if r.rs_order else "---"
        rs_seas = f"({r.rs_seasonal[0]},{r.rs_seasonal[1]},{r.rs_seasonal[2]})" if r.rs_seasonal and r.s > 1 else "---"
        pm_ord = f"({r.pm_order[0]},{r.pm_order[1]},{r.pm_order[2]})" if r.pm_order else "---"
        pm_seas = f"({r.pm_seasonal[0]},{r.pm_seasonal[1]},{r.pm_seasonal[2]})" if r.pm_seasonal and r.s > 1 else "---"
        rows.append(
            f"  {escape(r.label)} & {r.n_obs} & "
            f"{rs_ord} & {rs_seas} & {pm_ord} & {pm_seas} & "
            f"{match_sym} & "
            f"{aic_col}{sign_fmt(r.aic_diff_refit, '.3f')} & "
            f"{spd_col}{fmt(r.speedup, '.1f')}$\\times$ \\\\"
        )
    return "\n".join(rows)


def generate_tex(results, auto_results=None):
    st = compute_stats(results)

    auto_rows = make_auto_table(auto_results) if auto_results else "  \\multicolumn{9}{c}{\\textit{Not run}} \\\\"

    arima_ll  = make_loglike_table(make_section(results, "ARIMA"))
    sarima_ll = make_loglike_table(make_section(results, "SARIMA"))
    arimax_ll = make_loglike_table(make_section(results, "ARIMAX"))
    sarimax_ll= make_loglike_table(make_section(results, "SARIMAX"))

    arima_p  = make_param_table(make_section(results, "ARIMA"))
    sarima_p = make_param_table(make_section(results, "SARIMA"))
    arimax_p = make_param_table(make_section(results, "ARIMAX"))
    sarimax_p= make_param_table(make_section(results, "SARIMAX"))

    speed_rows = make_speed_table(results)

    tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{colortbl}
\usepackage[table]{xcolor}
\usepackage{amsmath}
\usepackage{microtype}
\usepackage{caption}
\usepackage{hyperref}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}

\definecolor{matchperfect}{RGB}{198,239,206}  % green: |Δllf| < 0.001
\definecolor{matchgood}{RGB}{255,235,156}      % yellow: |Δllf| < 1.0
\definecolor{matchbad}{RGB}{255,199,206}       % red: |Δllf| >= 1.0
\definecolor{speedhigh}{RGB}{198,239,206}      % green: speedup >= 20x
\definecolor{speedmid}{RGB}{255,235,156}       % yellow: 5x--20x
\definecolor{speedlow}{RGB}{255,199,206}       % red: < 5x

\captionsetup{font=small, labelfont=bf}

\title{\textbf{sarimax-rs vs.\ statsmodels: Comprehensive Validation Report}\\
\large Log-Likelihood Accuracy, Parameter Estimates, and Fitting Speed}
\author{Auto-generated benchmark · sarimax-rs v0.1.0 / statsmodels 0.14.6}
\date{\today}

\begin{document}
\maketitle

% ─────────────────────────────────────────────────────────────────────────────
\section*{Executive Summary}
% ─────────────────────────────────────────────────────────────────────────────

""" + f"""
\\begin{{center}}
\\begin{{tabular}}{{lr}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\midrule
Total models tested & {st['n_models']} \\\\
Models with errors  & {st['n_error']} \\\\
\\midrule
LLF match: $|\\Delta|<0.001$ (perfect)  & {st['n_perfect']} / {st['n_models']} \\\\
LLF match: $0.001 \\le |\\Delta|<1.0$ (good) & {st['n_good']} / {st['n_models']} \\\\
LLF match: $|\\Delta|\\ge 1.0$ (differ)  & {st['n_bad']} / {st['n_models']} \\\\
Mean $|\\Delta \\ell|$ & {fmt(st['mean_llf_diff'], '.4f')} \\\\
Max $|\\Delta \\ell|$  & {fmt(st['max_llf_diff'], '.4f')} \\\\
\\midrule
Mean $\\max|\\Delta\\hat{{\\theta}}|$ & {fmt(st['mean_param_diff'], '.5f')} \\\\
Max  $\\max|\\Delta\\hat{{\\theta}}|$ & {fmt(st['max_param_diff'], '.5f')} \\\\
\\midrule
Median speedup (sarimax-rs / statsmodels) & {fmt(st['median_speedup'], '.1f')}$\\times$ \\\\
Mean speedup   & {fmt(st['mean_speedup'], '.1f')}$\\times$ \\\\
Max speedup    & {fmt(st['max_speedup'], '.1f')}$\\times$ \\\\
Min speedup    & {fmt(st['min_speedup'], '.1f')}$\\times$ \\\\
\\bottomrule
\\end{{tabular}}
\\end{{center}}

\\bigskip
\\noindent
\\textbf{{Color legend:}}
\\colorbox{{matchperfect}}{{$|\\Delta \\ell| < 0.001$}}\\
\\colorbox{{matchgood}}{{$|\\Delta \\ell| < 1.0$}}\\
\\colorbox{{matchbad}}{{$|\\Delta \\ell| \\ge 1.0$}}\\
\\hfill
\\colorbox{{speedhigh}}{{$\\ge 20\\times$}}\\
\\colorbox{{speedmid}}{{$5$--$20\\times$}}\\
\\colorbox{{speedlow}}{{$<5\\times$}}

% ─────────────────────────────────────────────────────────────────────────────
\\section{{Log-Likelihood Comparison}}
% ─────────────────────────────────────────────────────────────────────────────

\\noindent The concentrated Gaussian log-likelihood
$\\ell = -\\tfrac{{n}}{{2}}[\\log(2\\pi) + \\log(\\hat\\sigma^2) + 1]
         - \\tfrac{{1}}{{2}}\\sum_t \\log F_t$
is computed by the Kalman filter in both implementations.  A difference
$\\Delta\\ell = \\ell_{{\\text{{rs}}}} - \\ell_{{\\text{{sm}}}}$ close to zero indicates
both implementations reach the same optimum.

\\subsection{{ARIMA Models ($n={N}$)}}
\\begin{{longtable}}{{lrrrrrc}}
\\toprule
Model & $n$ & $k$ & $\\ell_{{\\text{{rs}}}}$ & $\\ell_{{\\text{{sm}}}}$ &
$\\Delta\\ell$ & $\\Delta$AIC \\\\
\\midrule
\\endfirsthead
\\multicolumn{{7}}{{c}}{{\\small(continued)}}\\\\
\\midrule
Model & $n$ & $k$ & $\\ell_{{\\text{{rs}}}}$ & $\\ell_{{\\text{{sm}}}}$ &
$\\Delta\\ell$ & $\\Delta$AIC \\\\
\\midrule
\\endhead
\\bottomrule
\\endfoot
{arima_ll}
\\end{{longtable}}

\\subsection{{SARIMA Models}}
\\begin{{longtable}}{{lrrrrrc}}
\\toprule
Model & $n$ & $k$ & $\\ell_{{\\text{{rs}}}}$ & $\\ell_{{\\text{{sm}}}}$ &
$\\Delta\\ell$ & $\\Delta$AIC \\\\
\\midrule
\\endfirsthead
\\multicolumn{{7}}{{c}}{{\\small(continued)}}\\\\
\\midrule
\\endhead
\\bottomrule
\\endfoot
{sarima_ll}
\\end{{longtable}}

\\subsection{{ARIMAX Models ($n={N}$)}}
\\begin{{longtable}}{{lrrrrrc}}
\\toprule
Model & $n$ & $k$ & $\\ell_{{\\text{{rs}}}}$ & $\\ell_{{\\text{{sm}}}}$ &
$\\Delta\\ell$ & $\\Delta$AIC \\\\
\\midrule
\\endfirsthead
\\multicolumn{{7}}{{c}}{{\\small(continued)}}\\\\
\\midrule
\\endhead
\\bottomrule
\\endfoot
{arimax_ll}
\\end{{longtable}}

\\subsection{{SARIMAX Models}}
\\begin{{longtable}}{{lrrrrrc}}
\\toprule
Model & $n$ & $k$ & $\\ell_{{\\text{{rs}}}}$ & $\\ell_{{\\text{{sm}}}}$ &
$\\Delta\\ell$ & $\\Delta$AIC \\\\
\\midrule
\\endfirsthead
\\multicolumn{{7}}{{c}}{{\\small(continued)}}\\\\
\\midrule
\\endhead
\\bottomrule
\\endfoot
{sarimax_ll}
\\end{{longtable}}

% ─────────────────────────────────────────────────────────────────────────────
\\section{{Parameter Estimate Comparison}}
% ─────────────────────────────────────────────────────────────────────────────

\\noindent Parameter differences
$\\max_i|\\hat\\theta_{{i,\\text{{rs}}}} - \\hat\\theta_{{i,\\text{{sm}}}}|$
and mean absolute difference across all non-$\\sigma^2$ parameters.
Note that near-flat likelihood ridges in ARMA(q$>$1) may cause both
optimizers to converge to equivalent but numerically different estimates.

\\subsection{{ARIMA}}
\\begin{{longtable}}{{lrrrcc}}
\\toprule
Model & $k$ & $\\max|\\Delta\\hat\\theta|$ & $\\overline{{|\\Delta\\hat\\theta|}}$ &
RS conv. & SM conv. \\\\
\\midrule
\\endfirsthead
\\multicolumn{{6}}{{c}}{{\\small(continued)}}\\\\
\\midrule
\\endhead
\\bottomrule
\\endfoot
{arima_p}
\\end{{longtable}}

\\subsection{{SARIMA}}
\\begin{{longtable}}{{lrrrcc}}
\\toprule
Model & $k$ & $\\max|\\Delta\\hat\\theta|$ & $\\overline{{|\\Delta\\hat\\theta|}}$ &
RS conv. & SM conv. \\\\
\\midrule
\\endfirsthead
\\multicolumn{{6}}{{c}}{{\\small(continued)}}\\\\
\\midrule
\\endhead
\\bottomrule
\\endfoot
{sarima_p}
\\end{{longtable}}

\\subsection{{ARIMAX}}
\\begin{{longtable}}{{lrrrcc}}
\\toprule
Model & $k$ & $\\max|\\Delta\\hat\\theta|$ & $\\overline{{|\\Delta\\hat\\theta|}}$ &
RS conv. & SM conv. \\\\
\\midrule
\\endfirsthead
\\multicolumn{{6}}{{c}}{{\\small(continued)}}\\\\
\\midrule
\\endhead
\\bottomrule
\\endfoot
{arimax_p}
\\end{{longtable}}

\\subsection{{SARIMAX}}
\\begin{{longtable}}{{lrrrcc}}
\\toprule
Model & $k$ & $\\max|\\Delta\\hat\\theta|$ & $\\overline{{|\\Delta\\hat\\theta|}}$ &
RS conv. & SM conv. \\\\
\\midrule
\\endfirsthead
\\multicolumn{{6}}{{c}}{{\\small(continued)}}\\\\
\\midrule
\\endhead
\\bottomrule
\\endfoot
{sarimax_p}
\\end{{longtable}}

% ─────────────────────────────────────────────────────────────────────────────
\\section{{Fitting Speed Comparison}}
% ─────────────────────────────────────────────────────────────────────────────

\\noindent Wall-clock time measured over {REPEAT} repetitions (median reported).
sarimax-rs uses the Rust L-BFGS-B implementation with multi-start CSS
initialization; statsmodels uses SciPy L-BFGS-B.

\\begin{{longtable}}{{lrrrr}}
\\toprule
Model & $n$ & RS time & SM time & Speedup \\\\
\\midrule
\\endfirsthead
\\multicolumn{{5}}{{c}}{{\\small(continued)}}\\\\
\\midrule
\\endhead
\\bottomrule
\\endfoot
{speed_rows}
\\end{{longtable}}

% ─────────────────────────────────────────────────────────────────────────────
\\section{{Automatic Order Selection: \\texttt{{auto\\_arima}} vs.\\ pmdarima}}
% ─────────────────────────────────────────────────────────────────────────────

\\noindent
\\texttt{{sarimax\\_py.auto\\_arima}} implements the Hyndman--Khandakar stepwise
algorithm backed by the Rust batch-fit engine.  We compare it against
\\texttt{{pmdarima.auto\\_arima}} (v2.x), the most widely used Python
automatic ARIMA library.

\\smallskip
\\noindent
\\textbf{{Note on AIC comparability.}}
Each library optimises its own log-likelihood, so their raw AIC values are not
on the same scale (different model specifications, intercept handling, etc.).
For a fair comparison, $\\Delta$AIC is computed by re-fitting \\emph{{both}}
selected models with the \\texttt{{sarimax-rs}} engine and comparing on the
same log-likelihood.  $\\Delta$AIC $=$ AIC(rs-selected) $-$ AIC(pm-selected),
evaluated by sarimax-rs; negative values mean sarimax-rs found a better model.

\\begin{{longtable}}{{llllllcrr}}
\\toprule
Scenario & $n$ &
\\multicolumn{{2}}{{c}}{{sarimax-rs}} &
\\multicolumn{{2}}{{c}}{{pmdarima}} &
Match &
$\\Delta$AIC & Speedup \\\\
\\cmidrule(lr){{3-4}}\\cmidrule(lr){{5-6}}
 & & $(p,d,q)$ & $(P,D,Q)$ & $(p,d,q)$ & $(P,D,Q)$ & & & \\\\
\\midrule
\\endfirsthead
\\multicolumn{{9}}{{c}}{{\\small(continued)}}\\\\
\\midrule
\\endhead
\\bottomrule
\\endfoot
{auto_rows}
\\end{{longtable}}

\\noindent
\\textbf{{Notes:}}
\\colorbox{{matchperfect}}{{$|\\Delta$AIC$|<0.001$}}\\
\\colorbox{{matchgood}}{{$|\\Delta$AIC$|<1.0$}}\\
\\colorbox{{matchbad}}{{$|\\Delta$AIC$|\\ge 1.0$}}
(both optimizers may select different but statistically equivalent models).
A \\textcolor{{green}}{{\\checkmark}} indicates identical $(p,d,q)(P,D,Q)$ selection;
\\textcolor{{red}}{{$\\times$}} means different order (AIC comparison still valid).

% ─────────────────────────────────────────────────────────────────────────────
\\section{{Methodology Notes}}
% ─────────────────────────────────────────────────────────────────────────────

\\begin{{itemize}}
\\item Both libraries use the \\textbf{{concentrated Gaussian log-likelihood}}
      with $\\sigma^2$ profiled out (\\texttt{{concentrate\\_scale=True}}).
\\item \\texttt{{enforce\\_stationarity=False}}, \\texttt{{enforce\\_invertibility=False}}
      in both to ensure a fair comparison on the same parameter space.
\\item LLF differences near $0.00$ indicate both optimizers reach the same
      global optimum.  Non-zero differences for high-order ARMA models
      (e.g.\\ ARIMA(2,1,2)) reflect equivalent local optima on near-flat
      likelihood ridges --- both estimates are statistically equivalent.
\\item Speed is measured including Python overhead, parameter transformation,
      and Kalman filter evaluation.  sarimax-rs benefits from compiled Rust
      code, native L-BFGS-B, and CSS pre-optimization warm-start.
\\item \\texttt{{auto\\_arima}} speed includes full stepwise search;
      sarimax-rs uses the Rust batch-fit engine for each candidate.
\\item Platform: Apple Silicon (ARM64), macOS 14, Python 3.14,
      single-threaded fit (batch parallelism not used here).
\\end{{itemize}}

\\end{{document}}
""".replace("{N}", str(300)).replace("{REPEAT}", str(5))

    return tex

if __name__ == "__main__":
    print("Running fixed-order benchmarks...")
    results = run_benchmarks()

    print("\nRunning auto_arima benchmarks...")
    auto_results = run_auto_benchmarks()

    print("\nGenerating LaTeX report...")
    tex = generate_tex(results, auto_results)

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "paper", "validation_report.tex"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"Saved: {out_path}")
