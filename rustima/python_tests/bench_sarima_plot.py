#!/usr/bin/env python3
"""
SARIMA(s=24) 스케일링 2-panel 플롯 (rustima vs pmdarima).
- 기존 paper/bench_2019_2023/scaling_with_pmdarima.png 와 동일한 레이아웃.
- pmdarima가 swap에 의해 KILLED된 연도는 X 마커 + 'KILLED'로 표시.

단독 실행 (CSV에서 다시 그리기):
    cd rustima/
    .venv/bin/python python_tests/bench_sarima_plot.py
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(PROJECT_ROOT, "..", "paper", "bench_2019_2023")


def _runs_from_csv(csv_path):
    import csv
    runs = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            runs.append(type("R", (), {
                "engine": row["engine"],
                "years": int(row["years"]),
                "n_obs": int(row["n_obs"] or 0),
                "time_s": float(row["time_s"] or 0),
                "peak_rss_mb": float(row["peak_rss_mb"] or 0),
                "peak_swap_delta_mb": float(row.get("peak_swap_delta_mb") or 0),
                "status": row["status"],
                "order": row.get("order") or "",
                "seasonal": row.get("seasonal") or "",
                "aic": float(row["aic"]) if row.get("aic") else None,
            })())
    return runs


def save_plot(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["axes.unicode_minus"] = False
    try:
        plt.rcParams["font.family"] = "Apple SD Gothic Neo"
    except Exception:
        pass

    rs_ok = [r for r in runs if r.engine == "rustima" and r.status == "OK"]
    pm_ok = [r for r in runs if r.engine == "pmdarima" and r.status == "OK"]
    pm_kl = [r for r in runs if r.engine == "pmdarima" and r.status != "OK"]

    rs_ok.sort(key=lambda r: r.years)
    pm_ok.sort(key=lambda r: r.years)
    pm_kl.sort(key=lambda r: r.years)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Wall time ─────────────────────────────────────────────────────────
    ax = axes[0]
    if rs_ok:
        xs = [r.years for r in rs_ok]
        ys = [r.time_s for r in rs_ok]
        ax.plot(xs, ys, "o-", color="#DE6B35", lw=2.5, ms=10, label="rustima")
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.0f}s", xy=(x, y), xytext=(0, 12),
                        textcoords="offset points", ha="center",
                        fontsize=10, color="#DE6B35", fontweight="bold")
    if pm_ok:
        xs = [r.years for r in pm_ok]
        ys = [r.time_s for r in pm_ok]
        ax.plot(xs, ys, "o", color="#7A1F1F", ms=11, label="pmdarima")
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.0f}s", xy=(x, y), xytext=(0, 12),
                        textcoords="offset points", ha="center",
                        fontsize=10, color="#7A1F1F", fontweight="bold")
    if pm_kl:
        xs = [r.years for r in pm_kl]
        ymax = ax.get_ylim()[1] if rs_ok else 100
        ys = [ymax * 0.05] * len(xs)
        pm_label = None if pm_ok else "pmdarima"
        ax.plot(xs, ys, "x", color="#7A1F1F", ms=7, mew=1.8,
                label=pm_label)
        for x in xs:
            ax.annotate("KILLED\n(swap)", xy=(x, ymax * 0.05),
                        xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=8, color="#7A1F1F")

    ax.set_xlabel("Years of data", fontweight="bold")
    ax.set_ylabel("seconds", fontweight="bold")
    ax.set_title("Wall time")
    ax.grid(alpha=0.3)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend(loc="best", frameon=True)

    # ── Peak RSS (log) ────────────────────────────────────────────────────
    ax = axes[1]
    if rs_ok:
        xs = [r.years for r in rs_ok]
        ys = [r.peak_rss_mb for r in rs_ok]
        ax.plot(xs, ys, "s-", color="#3FA0FF", lw=2.5, ms=10, label="rustima")
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.0f} MB", xy=(x, y), xytext=(0, -16),
                        textcoords="offset points", ha="center",
                        fontsize=10, color="#1F70C9", fontweight="bold")
    if pm_ok:
        xs = [r.years for r in pm_ok]
        ys = [r.peak_rss_mb for r in pm_ok]
        ax.plot(xs, ys, "o", color="#1F2E5C", ms=11, label="pmdarima")
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:,.0f} MB", xy=(x, y), xytext=(0, 12),
                        textcoords="offset points", ha="center",
                        fontsize=10, color="#1F2E5C", fontweight="bold")
    if pm_kl:
        xs = [r.years for r in pm_kl]
        ys = [max((r.peak_rss_mb or 1) for r in pm_kl)] * len(xs)
        pm_label = None if pm_ok else "pmdarima"
        ax.plot(xs, ys, "x", color="#1F2E5C", ms=7, mew=1.8,
                label=pm_label)
        for r in pm_kl:
            label = f"KILLED\n(Δswap≈{r.peak_swap_delta_mb:.0f}MB)"
            ax.annotate(label, xy=(r.years, r.peak_rss_mb or 1),
                        xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=8, color="#1F2E5C")

    ax.set_xlabel("Years of data", fontweight="bold")
    ax.set_ylabel("MB (log scale)", fontweight="bold")
    ax.set_title("Peak resident memory")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend(loc="best", frameon=True)

    plt.suptitle(
        "rustima vs pmdarima  —  SARIMA(s=24)  Korean power demand "
        "(hourly, no exog)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    csv = os.path.join(OUTDIR, "sarima_scaling_results.csv")
    if not os.path.exists(csv):
        print(f"missing CSV: {csv}", file=sys.stderr)
        sys.exit(1)
    runs = _runs_from_csv(csv)
    out = os.path.join(OUTDIR, "sarima_scaling_with_pmdarima.png")
    save_plot(runs, out)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
