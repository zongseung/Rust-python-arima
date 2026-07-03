"""JSS §4.2 — auto_arima cross-engine benchmark (3 engines, OOM watchdog).

For each engine we launch a single subprocess and record:
  * selected (order, seasonal_order)
  * AIC
  * wall_time_s
  * peak_rss_gb (incl. all descendants)
  * status: 'ok' | 'oom' | 'timeout' | 'error'

The watchdog SIGKILLs the process group when peak RSS exceeds 16 GB so
that swap thrashing doesn't pollute wall-clock measurements. Expected
behavior:
  - rustima: completes within a few minutes, well under 16 GB
  - pmdarima: likely OOM (full-pipeline retains many models in memory)
  - R::forecast::auto.arima: likely OOM or very slow

Output:
  paper/jss_results/sec4_2_auto_arima.csv
  paper/jss_results/sec4_2_auto_arima.json
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

ENGINES = [
    (
        "rustima",
        ["uv", "run", "python", os.path.join(RUNNERS_DIR, "auto_py.py"),
         "--engine", "rustima"],
    ),
    (
        "pmdarima",
        ["uv", "run", "python", os.path.join(RUNNERS_DIR, "auto_py.py"),
         "--engine", "pmdarima"],
    ),
    (
        "r_forecast",
        ["Rscript", os.path.join(RUNNERS_DIR, "auto_r.R")],
    ),
]


def run_one(engine_name, base_cmd):
    fd, out_path = tempfile.mkstemp(suffix=".json", prefix="jss_auto_")
    os.close(fd)
    cmd = base_cmd + ["--out", out_path]

    r = run_with_oom_watchdog(
        cmd,
        timeout_s=7200.0,  # 2h hard cap
        cwd=os.path.dirname(THIS_DIR),
    )

    row = {
        "engine": engine_name,
        "status": r.status,
        "wall_time_s": round(r.wall_time_s, 3),
        "peak_rss_gb": round(r.peak_rss_gb, 3),
        "peak_swap_delta_gb": round(r.peak_swap_delta_gb, 3),
        "returncode": r.returncode,
    }

    auto_data = None
    if r.status == "ok" and os.path.exists(out_path):
        try:
            with open(out_path) as f:
                auto_data = json.load(f)
            row.update({
                "order": str(tuple(auto_data.get("order") or [])),
                "seasonal_order": str(tuple(auto_data.get("seasonal_order") or [])),
                "aic": auto_data.get("aic"),
                "n_models": auto_data.get("n_models"),
                "runtime_inner_s": auto_data.get("runtime_inner_s"),
            })
        except Exception as e:
            row["parse_error"] = str(e)

    if r.status != "ok" or auto_data is None:
        row["stderr_tail"] = (r.stderr or "")[-1200:]
        row["stdout_tail"] = (r.stdout or "")[-400:]

    try:
        os.unlink(out_path)
    except OSError:
        pass

    return row, auto_data


def main():
    rows = []
    full = []
    for engine_name, base_cmd in ENGINES:
        print(f"\n=== engine [{engine_name}] launching ===", flush=True)
        row, auto_data = run_one(engine_name, base_cmd)
        rows.append(row)

        if row["status"] == "ok":
            print(
                f"  OK  wall={row['wall_time_s']:>8.2f}s  "
                f"rss={row['peak_rss_gb']:>5.2f}GB  "
                f"swapΔ={row['peak_swap_delta_gb']:>5.2f}GB  "
                f"order={row.get('order')}  sorder={row.get('seasonal_order')}  "
                f"aic={row.get('aic', float('nan')):.3f}",
                flush=True,
            )
        else:
            print(
                f"  {row['status'].upper()}  wall={row['wall_time_s']:.2f}s  "
                f"rss={row['peak_rss_gb']:.2f}GB  "
                f"swapΔ={row['peak_swap_delta_gb']:.2f}GB  "
                f"rc={row['returncode']}\n"
                f"  --- stderr tail ---\n{row.get('stderr_tail', '')}",
                flush=True,
            )

        if auto_data is not None:
            full.append(auto_data)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "sec4_2_auto_arima.csv")
    df.to_csv(csv_path, index=False)

    json_path = os.path.join(OUT_DIR, "sec4_2_auto_arima.json")
    with open(json_path, "w") as f:
        json.dump(full, f, indent=2)

    print(f"\n[CSV]  {csv_path}")
    print(f"[JSON] {json_path}")

    print("\n========== §4.2 SUMMARY ==========")
    cols = ["engine", "status", "wall_time_s", "peak_rss_gb",
            "peak_swap_delta_gb", "order", "seasonal_order", "aic"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
