"""JSS §4.1 — Fixed-order SARIMAX parity benchmark (3 engines).

For each (order, seasonal_order) we launch three subprocesses under the
OOM watchdog:

  * rustima (rustima.model.SARIMAXModel)
  * statsmodels (statsmodels.tsa.statespace.sarimax.SARIMAX)
  * R::stats::arima (CSS-ML)

and record loglike / AIC / BIC / params / runtime / peak RSS / status.

Output:
  paper/jss_results/sec4_1_fixed_order_parity.csv
  paper/jss_results/sec4_1_fixed_order_parity.json  (full params)

Run from rustima/ root:
  uv run python benchmarks/exp_jss_fixed_order.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from jss_common import run_with_oom_watchdog  # noqa: E402

RUNNERS_DIR = os.path.join(THIS_DIR, "jss_runners")
OUT_DIR = os.path.join(THIS_DIR, "..", "paper", "jss_results")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = [
    # (order, seasonal_order, tag)
    # Both have D=1 so the seasonal unit root is absorbed by differencing.
    # baseline is the parsimonious case; the second specification is the
    # one selected by R::forecast::auto.arima on the same 3-year window
    # in §4.2 — re-using R's pick as a fixed-order parity check avoids
    # the optimization-path divergence we saw with (2,1,2)(1,1,1)[24].
    ((2, 0, 0), (0, 1, 1, 24), "baseline"),
    ((3, 0, 0), (1, 1, 0, 24), "r_auto_pick"),
]

ENGINES = [
    (
        "rustima",
        ["uv", "run", "python", os.path.join(RUNNERS_DIR, "fit_py.py"),
         "--engine", "rustima"],
    ),
    (
        "statsmodels",
        ["uv", "run", "python", os.path.join(RUNNERS_DIR, "fit_py.py"),
         "--engine", "statsmodels"],
    ),
    (
        "r_stats_arima",
        ["Rscript", os.path.join(RUNNERS_DIR, "fit_r.R")],
    ),
]


def run_one(engine_name, base_cmd, order, sorder):
    fd, out_path = tempfile.mkstemp(suffix=".json", prefix="jss_fit_")
    os.close(fd)
    cmd = base_cmd + [
        "--order", ",".join(str(x) for x in order),
        "--seasonal-order", ",".join(str(x) for x in sorder),
        "--out", out_path,
    ]

    r = run_with_oom_watchdog(
        cmd,
        timeout_s=3600.0,
        cwd=os.path.dirname(THIS_DIR),  # = rustima/
    )

    row = {
        "engine": engine_name,
        "order": str(tuple(order)),
        "seasonal_order": str(tuple(sorder)),
        "status": r.status,
        "wall_time_s": round(r.wall_time_s, 3),
        "peak_rss_gb": round(r.peak_rss_gb, 3),
        "peak_swap_delta_gb": round(r.peak_swap_delta_gb, 3),
        "returncode": r.returncode,
    }

    fit_data = None
    if r.status == "ok" and os.path.exists(out_path):
        try:
            with open(out_path) as f:
                fit_data = json.load(f)
            row.update({
                "loglike": fit_data.get("loglike"),
                "aic": fit_data.get("aic"),
                "bic": fit_data.get("bic"),
                "scale": fit_data.get("scale"),
                "n_obs": fit_data.get("n_obs"),
                "runtime_inner_s": fit_data.get("runtime_inner_s"),
                "n_params": (
                    len(fit_data.get("params") or [])
                    if not isinstance(fit_data.get("params"), dict)
                    else len(fit_data["params"])
                ),
            })
        except Exception as e:
            row["parse_error"] = str(e)

    if r.status != "ok" or fit_data is None:
        # capture tail of stderr for diagnosis
        row["stderr_tail"] = (r.stderr or "")[-800:]

    try:
        os.unlink(out_path)
    except OSError:
        pass

    return row, fit_data


def main():
    rows = []
    full = []
    for order, sorder, tag in MODELS:
        print(f"\n=== model [{tag}]: {order} x {sorder} ===", flush=True)
        for engine_name, base_cmd in ENGINES:
            print(f"  [{engine_name}] launching ...", flush=True)
            row, fit_data = run_one(engine_name, base_cmd, order, sorder)
            row["model_tag"] = tag
            rows.append(row)

            if row["status"] == "ok":
                print(
                    f"    OK  wall={row['wall_time_s']:>7.2f}s  "
                    f"peak={row['peak_rss_gb']:>5.2f}GB  "
                    f"ll={row.get('loglike', float('nan')):>12.3f}  "
                    f"aic={row.get('aic', float('nan')):>12.3f}  "
                    f"k={row.get('n_params')}",
                    flush=True,
                )
            else:
                print(
                    f"    FAIL status={row['status']} rc={row['returncode']}\n"
                    f"    --- stderr tail ---\n{row.get('stderr_tail', '')}",
                    flush=True,
                )

            if fit_data is not None:
                full.append({"model_tag": tag, **fit_data})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "sec4_1_fixed_order_parity.csv")
    df.to_csv(csv_path, index=False)

    json_path = os.path.join(OUT_DIR, "sec4_1_fixed_order_parity.json")
    with open(json_path, "w") as f:
        json.dump(full, f, indent=2)

    print(f"\n[CSV]  {csv_path}")
    print(f"[JSON] {json_path}")

    # ---- pretty summary table ----
    print("\n========== §4.1 SUMMARY ==========")
    pivot = df.pivot_table(
        index=["model_tag", "engine"],
        values=["loglike", "aic", "bic", "wall_time_s", "peak_rss_gb"],
        aggfunc="first",
    )
    print(pivot.to_string())


if __name__ == "__main__":
    main()
