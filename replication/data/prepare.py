"""Stage the South Korean hourly electricity-demand dataset into data/.

Copies power_demand_final.csv from the repository root (hourly, 2019-01-01
onward, 51,144 rows, columns include `power demand(MW)`, `ta` temperature,
`hm` humidity) and validates its shape. Simulated series used by the scaling
benchmarks are generated in-process with fixed seeds and need no files.
"""
from __future__ import annotations

import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import DATA_CSV_LOCAL, DATA_CSV_ROOT  # noqa: E402


def main() -> None:
    if not os.path.exists(DATA_CSV_LOCAL):
        if not os.path.exists(DATA_CSV_ROOT):
            raise SystemExit(
                f"source dataset missing: {DATA_CSV_ROOT}\n"
                "Place power_demand_final.csv at the repository root."
            )
        shutil.copy2(DATA_CSV_ROOT, DATA_CSV_LOCAL)
        print(f"[data] copied -> {DATA_CSV_LOCAL}")

    import pandas as pd

    df = pd.read_csv(DATA_CSV_LOCAL)
    n = len(df)
    assert n >= 50000, f"expected >=50000 hourly rows, got {n}"
    for col in ("일시", "power demand(MW)", "ta", "hm"):
        assert col in df.columns, f"missing column: {col}"
    dates = pd.to_datetime(df["일시"])
    print(
        f"[data] OK n={n:,}  range={dates.iloc[0]} .. {dates.iloc[-1]}  "
        f"nan(power)={int(df['power demand(MW)'].isna().sum())}"
    )


if __name__ == "__main__":
    main()
