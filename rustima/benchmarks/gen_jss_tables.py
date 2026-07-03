"""Generate JSS §4 LaTeX tables and figures from sec4_1/sec4_2 result CSVs.

Inputs:
  paper/jss_results/sec4_1_fixed_order_parity.csv
  paper/jss_results/sec4_2_auto_arima.csv

Outputs (paper/jss_results/):
  sec4_1_fixed_order_parity.tex   — fixed-order parity table
  sec4_2_auto_arima.tex           — auto_arima cross-engine table
  sec4_runtime_memory.pdf         — bar chart: wall time + peak RSS
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "..", "paper", "jss_results")

ENGINE_LABEL = {
    "rustima": r"\textbf{rustima}",
    "statsmodels": "statsmodels",
    "r_stats_arima": r"R \texttt{stats::arima}",
    "pmdarima": "pmdarima",
    "r_forecast": r"R \texttt{forecast::auto.arima}",
}

STATUS_LABEL = {
    "ok": "ok",
    "oom": r"\textbf{OOM (RSS)}",
    "oom_swap": r"\textbf{OOM (swap)}",
    "aborted_system_pressure": r"\textbf{aborted (swap)}",
    "not_attempted": r"\textit{not attempted}",
    "timeout": r"\textbf{TIMEOUT}",
    "error": r"\textit{error}",
}


def _fmt_float(x, fmt=".3f"):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "--"
    return format(float(x), fmt)


def _fmt_int(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "--"
    return str(int(x))


EXOG_NAMES = ["ta", "hm"]
ENGINE_ORDER = ["rustima", "r_stats_arima", "statsmodels"]


def _positional_labels(order, sorder, n_exog: int = 2) -> list[str]:
    """Label rustima/statsmodels positional param vectors.

    Layout: [exog(k) | ar(p) | ma(q) | sar(P) | sma(Q)].
    rustima additionally appends sigma² as the trailing element; that's
    handled by the caller by length comparison.
    """
    p, _d, q = order
    P, _D, Q, _s = sorder
    labels = list(EXOG_NAMES[:n_exog])
    labels += [f"ar{i}" for i in range(1, p + 1)]
    labels += [f"ma{i}" for i in range(1, q + 1)]
    labels += [f"sar{i}" for i in range(1, P + 1)]
    labels += [f"sma{i}" for i in range(1, Q + 1)]
    return labels


def _normalize_params(entry: dict) -> dict:
    params = entry["params"]
    if isinstance(params, dict):
        # R returns a named coefficient dict already.
        out = dict(params)
    else:
        base = _positional_labels(entry["order"], entry["seasonal_order"])
        labels = base + (["sigma2"] if len(params) == len(base) + 1 else [])
        out = dict(zip(labels, params))
    out.setdefault("sigma2", entry.get("scale"))
    return out


def coef_table(json_path: str, out_tex: str, print_preview: bool = True) -> None:
    """Per-engine coefficient table for the §4.1 fixed-order fits."""
    with open(json_path) as f:
        full = json.load(f)

    by_tag: dict[str, dict[str, dict]] = {}
    by_tag_order: dict[str, tuple] = {}
    for entry in full:
        tag = entry["model_tag"]
        by_tag.setdefault(tag, {})[entry["engine"]] = _normalize_params(entry)
        by_tag_order.setdefault(tag, (entry["order"], entry["seasonal_order"]))

    def model_label(tag: str) -> str:
        order, sorder = by_tag_order[tag]
        p, d, q = order
        P, D, Q, s = sorder
        return f"$({p},{d},{q})({P},{D},{Q})_{{{s}}}$"

    def fmt(v, key):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "--"
        if key == "sigma2":
            return f"{int(round(float(v))):,}"
        return f"{float(v):.4f}"

    if print_preview:
        print("\n========== §4.1 COEFFICIENT COMPARISON ==========")
        for tag, engines in by_tag.items():
            print(f"\n[{tag}] order={engines.get('rustima', {}).get('_order_str', '')}")
            rs_keys = list(engines.get("rustima", {}).keys())
            keys = [k for k in rs_keys if k != "sigma2"] + ["sigma2"]
            header = f"{'param':>8s}" + "".join(f" {e:>14s}" for e in ENGINE_ORDER)
            print(header)
            print("-" * len(header))
            for k in keys:
                cells = [fmt(engines.get(e, {}).get(k), k) for e in ENGINE_ORDER]
                print(f"{k:>8s}" + "".join(f" {c:>14s}" for c in cells))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Section 4.1 (continued): Estimated parameters per engine. "
        r"On $(2,0,0)(0,1,1)_{24}$ \texttt{rustima} and R \texttt{stats::arima} "
        r"return invertible solutions that agree on AR to $\le 5\times 10^{-3}$, "
        r"on sma$_1$ to $\le 5\times 10^{-3}$, and on $\sigma^2$ to $0.5\%$; "
        r"\texttt{statsmodels} settles at sma$_1=-1.11$ outside the "
        r"invertibility circle (we keep \texttt{enforce\_invertibility=false} "
        r"to match R's default), which gives its lower $\sigma^2$. On "
        r"$(3,0,0)(1,1,0)_{24}$ all three engines agree on AR to $\le 0.04$, "
        r"with R and \texttt{statsmodels} matching closely on sar$_1$ and "
        r"$\sigma^2$ while \texttt{rustima} reaches a nearby local optimum "
        r"with $|\Delta\sigma^2|/\sigma^2 \approx 4.7\%$ and a log-likelihood "
        r"deficit of only $20$ units relative to R on $n = 26{,}280$.}",
        r"\label{tab:sec4-1-coefs}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Model & Parameter & \textbf{rustima} & R \texttt{stats::arima} "
        r"& statsmodels \\",
        r"\midrule",
    ]

    for tag, engines in by_tag.items():
        rs_keys = list(engines.get("rustima", {}).keys())
        # Force sigma2 to the bottom for readability.
        keys = [k for k in rs_keys if k != "sigma2"] + ["sigma2"]
        first = True
        for k in keys:
            mtag = model_label(tag) if first else ""
            first = False
            display_key = r"$\sigma^2$" if k == "sigma2" else k
            cells = [fmt(engines.get(e, {}).get(k), k) for e in ENGINE_ORDER]
            lines.append(
                f"{mtag} & {display_key} & {cells[0]} & {cells[1]} & {cells[2]} \\\\"
            )
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[tex] wrote {out_tex}")


def _model_label(row) -> str:
    """Render the order/seasonal_order pair as a compact LaTeX math label."""
    order = eval(row["order"]) if isinstance(row["order"], str) else row["order"]
    sorder = eval(row["seasonal_order"]) if isinstance(row["seasonal_order"], str) else row["seasonal_order"]
    p, d, q = order
    P, D, Q, s = sorder
    return f"$({p},{d},{q})({P},{D},{Q})_{{{s}}}$"


def fixed_order_table(csv_path: str, out_tex: str) -> None:
    df = pd.read_csv(csv_path)
    df = df.sort_values(["model_tag", "engine"]).reset_index(drop=True)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Section 4.1: Fixed-order SARIMAX parity on 3 years of Korean "
        r"hourly power demand ($n=26{,}280$, exog $=\{ta, hm\}$). Each engine "
        r"fits the same model in a separate subprocess under a 16 GB RSS and "
        r"3 GB swap-delta watchdog. \texttt{rustima} matches "
        r"\texttt{R \texttt{stats::arima}} on log-likelihood to within $0.05\%$ "
        r"of the absolute value while running $3.5$--$13\times$ faster, and "
        r"uses $26$--$31\times$ less peak resident memory than "
        r"\texttt{statsmodels}. The second specification "
        r"$(3,0,0)(1,1,0)_{24}$ is the order R \texttt{forecast::auto.arima} "
        r"selects on this dataset in Section~\ref{sec:rustima:swap}; "
        r"re-using it here gives a fixed-order parity check on a model that "
        r"R's stepwise search has already vetted.}",
        r"\label{tab:sec4-1-fixed}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Model & Engine & log-lik & AIC & BIC & Wall (s) & Peak RSS (GB) \\",
        r"\midrule",
    ]

    for tag, group in df.groupby("model_tag", sort=False):
        first = True
        for _, row in group.iterrows():
            mtag = _model_label(row) if first else ""
            first = False
            lines.append(
                f"{mtag} & "
                f"{ENGINE_LABEL.get(row['engine'], row['engine'])} & "
                f"{_fmt_float(row.get('loglike'), '.2f')} & "
                f"{_fmt_float(row.get('aic'), '.2f')} & "
                f"{_fmt_float(row.get('bic'), '.2f')} & "
                f"{_fmt_float(row.get('wall_time_s'), '.2f')} & "
                f"{_fmt_float(row.get('peak_rss_gb'), '.2f')} \\\\"
            )
        lines.append(r"\midrule")
    # drop trailing midrule
    if lines[-1] == r"\midrule":
        lines.pop()

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[tex] wrote {out_tex}")


def auto_arima_table(csv_path: str, out_tex: str) -> None:
    df = pd.read_csv(csv_path)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Section 4.2: \texttt{auto\_arima} cross-engine comparison on "
        r"3 years of Korean hourly power demand ($n=26{,}280$, $s=24$, "
        r"search space $\{p,q\}\le 3$, $\{P,Q\}\le 1$, stepwise AIC) on a "
        r"24 GB host. \texttt{rustima} and R \texttt{forecast::auto.arima} "
        r"both complete the search at $0.44$--$0.96$ GB peak RSS with no "
        r"swap allocation; \texttt{rustima} runs $9.9\times$ faster but "
        r"reaches a richer specification with $\Delta\mathrm{AIC}=-7{,}576$. "
        r"\texttt{pmdarima} drove system swap past the safe envelope (8 GB) "
        r"on two attempts and was aborted in both. See "
        r"\texttt{paper/jss\_results/sec4\_2\_run\_notes.md} for details.}",
        r"\label{tab:sec4-2-auto}",
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"Engine & Status & Selected order & AIC & Wall (s) & Peak RSS (GB) \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        if row.get("status") == "ok":
            order_disp = f"{row.get('order', '')}\\,{row.get('seasonal_order', '')}"
        else:
            order_disp = "--"
        lines.append(
            f"{ENGINE_LABEL.get(row['engine'], row['engine'])} & "
            f"{STATUS_LABEL.get(row['status'], row['status'])} & "
            f"{order_disp} & "
            f"{_fmt_float(row.get('aic'), '.2f')} & "
            f"{_fmt_float(row.get('wall_time_s'), '.2f')} & "
            f"{_fmt_float(row.get('peak_rss_gb'), '.2f')} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[tex] wrote {out_tex}")


def runtime_memory_figure(
    fixed_csv: str,
    auto_csv: str,
    out_pdf: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df_fixed = pd.read_csv(fixed_csv)
    df_auto = pd.read_csv(auto_csv).fillna({"wall_time_s": 0.0, "peak_rss_gb": 0.0})

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    # ----- §4.1 wall time -----
    ax = axes[0, 0]
    pivot = df_fixed.pivot_table(
        index="model_tag", columns="engine", values="wall_time_s", aggfunc="first"
    )
    pivot.plot(kind="bar", ax=ax, edgecolor="black")
    ax.set_title("§4.1 Wall time (fixed-order fit)")
    ax.set_ylabel("seconds")
    ax.set_xlabel("model")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    # ----- §4.1 peak RSS -----
    ax = axes[0, 1]
    pivot = df_fixed.pivot_table(
        index="model_tag", columns="engine", values="peak_rss_gb", aggfunc="first"
    )
    pivot.plot(kind="bar", ax=ax, edgecolor="black")
    ax.set_title("§4.1 Peak RSS (fixed-order fit)")
    ax.set_ylabel("GB")
    ax.set_xlabel("model")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.axhline(16.0, color="red", linestyle="--", lw=1, label="16 GB OOM")

    def _status_color(s):
        if s == "ok":
            return "tab:blue"
        if s == "not_attempted":
            return "lightgray"
        return "tab:red"

    # ----- §4.2 wall time -----
    ax = axes[1, 0]
    colors = [_status_color(s) for s in df_auto["status"]]
    ax.bar(df_auto["engine"], df_auto["wall_time_s"], color=colors, edgecolor="black")
    for i, row in df_auto.reset_index(drop=True).iterrows():
        if row["status"] != "ok":
            label = row["status"].replace("_", " ")
            ax.text(i, max(row["wall_time_s"], 0) + 1, label,
                    ha="center", va="bottom", fontsize=8, color="darkred",
                    rotation=0)
    ax.set_title("§4.2 Wall time (auto_arima)")
    ax.set_ylabel("seconds (lower bound for aborted runs)")
    ax.set_xlabel("engine")
    ax.grid(axis="y", alpha=0.3)

    # ----- §4.2 peak RSS -----
    ax = axes[1, 1]
    ax.bar(df_auto["engine"], df_auto["peak_rss_gb"], color=colors, edgecolor="black")
    ax.axhline(16.0, color="red", linestyle="--", lw=1)
    ax.text(len(df_auto) - 0.5, 16.2, "16 GB RSS watchdog", color="red",
            ha="right", va="bottom", fontsize=8)
    for i, row in df_auto.reset_index(drop=True).iterrows():
        if row["status"] != "ok":
            label = row["status"].replace("_", " ")
            ax.text(i, max(row["peak_rss_gb"], 0) + 0.3, label,
                    ha="center", va="bottom", fontsize=8, color="darkred")
    ax.set_title("§4.2 Peak RSS (auto_arima)")
    ax.set_ylabel("GB (lower bound for aborted runs)")
    ax.set_xlabel("engine")
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "JSS §4 — rustima vs. statsmodels / pmdarima / R on 3y hourly Korean power demand",
        fontsize=12, fontweight="bold", y=1.00,
    )
    plt.tight_layout()
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out_pdf}")


def main():
    fixed_csv = os.path.join(OUT_DIR, "sec4_1_fixed_order_parity.csv")
    auto_csv = os.path.join(OUT_DIR, "sec4_2_auto_arima.csv")

    if os.path.exists(fixed_csv):
        fixed_order_table(
            fixed_csv,
            os.path.join(OUT_DIR, "sec4_1_fixed_order_parity.tex"),
        )
    else:
        print(f"[skip] {fixed_csv} not found")

    fixed_json = os.path.join(OUT_DIR, "sec4_1_fixed_order_parity.json")
    if os.path.exists(fixed_json):
        coef_table(
            fixed_json,
            os.path.join(OUT_DIR, "sec4_1_coefficients.tex"),
        )
    else:
        print(f"[skip] {fixed_json} not found")

    if os.path.exists(auto_csv):
        auto_arima_table(
            auto_csv,
            os.path.join(OUT_DIR, "sec4_2_auto_arima.tex"),
        )
    else:
        print(f"[skip] {auto_csv} not found")

    if os.path.exists(fixed_csv) and os.path.exists(auto_csv):
        runtime_memory_figure(
            fixed_csv, auto_csv,
            os.path.join(OUT_DIR, "sec4_runtime_memory.pdf"),
        )


if __name__ == "__main__":
    main()
