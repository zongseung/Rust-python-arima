# Replication archive — rustima JSS manuscript

Regenerates every benchmark table in the manuscript from a single entry
point:

```bash
python run_all.py            # full suite (several hours on a 24 GB host)
python run_all.py --smoke    # reduced-scale wiring check (a few minutes)
python test_smoke.py         # asserts the smoke outputs have the right schema
```

## Layout

```
replication/
├── README.md
├── run_all.py               # one-command reproduction
├── common.py                # env pinning + measured-subprocess harness reuse
├── test_smoke.py            # end-to-end wiring check
├── gen_tables.py            # outputs/raw/*.csv -> outputs/tables/*.tex
├── data/
│   └── prepare.py           # stages + validates the hourly demand CSV
├── benchmark/
│   ├── parallel_scaling.py  # fair parallel matrix (time + peak RSS)
│   └── longseries_scaling.py# n x s scaling with DNF reporting
├── accuracy/
│   ├── auto_fourway.py      # rustima / pmdarima / StatsForecast / R forecast
│   ├── sf_auto_worker.py
│   └── grid_auto_worker.py  # matched 64-model parallel grid search
├── application/
│   └── demand_extended.py   # 2019–2023 train, 48 h holdout
└── outputs/
    ├── raw/                 # machine-readable results (CSV)
    ├── tables/              # LaTeX fragments used in the paper
    └── figures/
```

## Environment

* Python environment: the repository's `rustima/uv.lock` (run everything via
  `uv run python …` from `rustima/`, or activate that venv). Additional
  benchmark-only packages: `pmdarima`, `statsforecast`, `joblib`, `psutil`.
* R with the `forecast` package (only for the `auto` stage's
  `forecast::auto.arima` engine).
* Build the extension before benchmarking:
  `cd ../rustima && uv run maturin develop --release`.

## Measurement protocol

Every measured condition runs in a fresh subprocess. A watchdog samples the
whole process tree and records wall time and peak RSS, and kills the group on
RSS > 16 GB, system-swap growth > 3 GB, or timeout — killed cells are
reported as `oom` / `oom_swap` / `timeout` rather than dropped. BLAS, OpenMP,
and Numba intra-op threading is pinned to 1 in all conditions so that only
the task-level parallelism under study (Rayon threads, joblib processes,
StatsForecast `n_jobs`) varies. Simulated inputs use fixed seeds; timings are
reported from in-process counters that exclude interpreter startup and data
generation.

## Data

`data/prepare.py` stages `power_demand_final.csv` (hourly South Korean
electricity demand, 2019-01-01 onward, 51,144 observations; columns include
`power demand(MW)`, temperature `ta`, humidity `hm`) from the repository root
into `data/` and validates its shape.
