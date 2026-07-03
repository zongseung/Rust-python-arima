"""
Comprehensive benchmark: sarimax_rs vs statsmodels
Generates comparison reports for ARIMA, SARIMA, ARIMAX, SARIMAX models.
"""
import json
import time
import warnings
import numpy as np
import sys
import os

warnings.filterwarnings("ignore")

# ── helpers ──────────────────────────────────────────────────────────────────

def generate_arma_data(n, ar_coeffs, ma_coeffs, seed=42):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1, n)
    y = np.zeros(n)
    p, q = len(ar_coeffs), len(ma_coeffs)
    for t in range(max(p, q), n):
        y[t] = e[t]
        for i, a in enumerate(ar_coeffs):
            y[t] += a * y[t - 1 - i]
        for j, m in enumerate(ma_coeffs):
            y[t] += m * e[t - 1 - j]
    return y

def generate_seasonal_data(n, s, ar=0.5, sar=0.3, ma=0.3, sma=-0.4, seed=42):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1, n)
    y = np.zeros(n)
    for t in range(s + 1, n):
        y[t] = e[t] + ar * y[t-1] + sar * y[t-s] + ma * e[t-1] + sma * e[t-s]
    # integrate once
    y_int = np.cumsum(y)
    # add seasonal integration
    result = np.zeros(n)
    for t in range(s, n):
        result[t] = y_int[t] - y_int[t-s]
    return np.cumsum(result[s:])

def generate_exog(n, k, seed=123):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n, k))

# ── sarimax_rs wrapper ──────────────────────────────────────────────────────

def fit_rs(y, order, seasonal_order=(0,0,0,0), exog=None, trend='n'):
    from rustima import SARIMAXModel
    p, d, q = order
    P, D, Q, s = seasonal_order
    model = SARIMAXModel(y, order=(p, d, q), seasonal_order=(P, D, Q, s),
                         exog=exog, trend=trend)
    t0 = time.perf_counter()
    result = model.fit()
    elapsed = time.perf_counter() - t0
    return {
        'loglike': result.llf,
        'aic': result.aic,
        'bic': result.bic,
        'params': result.params.tolist() if hasattr(result.params, 'tolist') else list(result.params),
        'time_ms': elapsed * 1000,
        'converged': True,
    }

# ── statsmodels wrapper ─────────────────────────────────────────────────────

def fit_sm(y, order, seasonal_order=(0,0,0,0), exog=None, trend='n'):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    p, d, q = order
    P, D, Q, s = seasonal_order
    model = SARIMAX(y, order=(p, d, q), seasonal_order=(P, D, Q, s),
                    exog=exog, trend=trend, enforce_stationarity=False,
                    enforce_invertibility=False)
    t0 = time.perf_counter()
    result = model.fit(disp=False, maxiter=500)
    elapsed = time.perf_counter() - t0
    return {
        'loglike': result.llf,
        'aic': result.aic,
        'bic': result.bic,
        'params': result.params.tolist(),
        'time_ms': elapsed * 1000,
        'converged': result.mle_retvals.get('converged', False) if hasattr(result, 'mle_retvals') and result.mle_retvals else True,
    }

# ── benchmark runner ─────────────────────────────────────────────────────────

def run_comparison(model_name, y, order, seasonal_order=(0,0,0,0), exog=None, trend='n'):
    """Fit both engines, return comparison dict."""
    try:
        rs = fit_rs(y, order, seasonal_order, exog, trend)
    except Exception as e:
        rs = {'loglike': float('nan'), 'aic': float('nan'), 'bic': float('nan'),
               'params': [], 'time_ms': 0, 'converged': False, 'error': str(e)}
    try:
        sm = fit_sm(y, order, seasonal_order, exog, trend)
    except Exception as e:
        sm = {'loglike': float('nan'), 'aic': float('nan'), 'bic': float('nan'),
               'params': [], 'time_ms': 0, 'converged': False, 'error': str(e)}

    delta_ll = abs(rs['loglike'] - sm['loglike']) if np.isfinite(rs['loglike']) and np.isfinite(sm['loglike']) else float('nan')
    speedup = sm['time_ms'] / rs['time_ms'] if rs['time_ms'] > 0 else float('nan')

    return {
        'model': model_name,
        'order': list(order),
        'seasonal_order': list(seasonal_order),
        'rs': rs, 'sm': sm,
        'delta_ll': delta_ll,
        'speedup': speedup,
    }


# ── ARIMA benchmark ─────────────────────────────────────────────────────────

def bench_arima(max_pq=3):
    print("=" * 60)
    print("ARIMA Benchmark (d=1, n=500)")
    print("=" * 60)
    n = 500
    y = generate_arma_data(n, [0.7, -0.3], [0.4], seed=42)
    y = np.cumsum(y)  # integrate for d=1

    results = []
    for p in range(max_pq + 1):
        for q in range(max_pq + 1):
            if p == 0 and q == 0:
                continue
            name = f"ARIMA({p},1,{q})"
            print(f"  {name}...", end=" ", flush=True)
            r = run_comparison(name, y, (p, 1, q))
            results.append(r)
            print(f"LL_rs={r['rs']['loglike']:.2f}  LL_sm={r['sm']['loglike']:.2f}  "
                  f"Δ={r['delta_ll']:.3f}  speed={r['speedup']:.1f}x  "
                  f"rs={r['rs']['time_ms']:.0f}ms  sm={r['sm']['time_ms']:.0f}ms")
    return results


# ── SARIMA benchmark ─────────────────────────────────────────────────────────

def bench_sarima(s_values=(7, 12, 24), max_pq=2):
    all_results = {}
    for s in s_values:
        print(f"\n{'=' * 60}")
        print(f"SARIMA Benchmark (d=1, D=1, s={s}, n=500)")
        print("=" * 60)
        n = max(500, s * 20)
        y = generate_seasonal_data(n, s, seed=42)
        if len(y) > 500:
            y = y[-500:]

        results = []
        for p in range(max_pq + 1):
            for q in range(max_pq + 1):
                for P in range(max_pq + 1):
                    for Q in range(max_pq + 1):
                        if p + q + P + Q == 0:
                            continue
                        name = f"SARIMA({p},1,{q})({P},1,{Q},{s})"
                        print(f"  {name}...", end=" ", flush=True)
                        r = run_comparison(name, y, (p, 1, q), (P, 1, Q, s))
                        results.append(r)
                        delta_s = f"Δ={r['delta_ll']:.3f}" if np.isfinite(r['delta_ll']) else "Δ=NaN"
                        speed_s = f"speed={r['speedup']:.1f}x" if np.isfinite(r['speedup']) else "speed=N/A"
                        print(f"LL_rs={r['rs']['loglike']:.2f}  LL_sm={r['sm']['loglike']:.2f}  "
                              f"{delta_s}  {speed_s}")
        all_results[s] = results
    return all_results


# ── ARIMAX benchmark ─────────────────────────────────────────────────────────

def bench_arimax(max_pq=3, k_exog=2):
    print(f"\n{'=' * 60}")
    print(f"ARIMAX Benchmark (d=1, k_exog={k_exog}, n=500)")
    print("=" * 60)
    n = 500
    X = generate_exog(n, k_exog, seed=123)
    y = generate_arma_data(n, [0.7, -0.3], [0.4], seed=42)
    y = np.cumsum(y + X @ np.array([0.5, -0.3])[:k_exog])

    exog_rs = np.asfortranarray(X)

    results = []
    for p in range(max_pq + 1):
        for q in range(max_pq + 1):
            if p == 0 and q == 0:
                continue
            name = f"ARIMAX({p},1,{q})+k{k_exog}"
            print(f"  {name}...", end=" ", flush=True)
            r = run_comparison(name, y, (p, 1, q), exog=exog_rs)
            results.append(r)
            print(f"LL_rs={r['rs']['loglike']:.2f}  LL_sm={r['sm']['loglike']:.2f}  "
                  f"Δ={r['delta_ll']:.3f}  speed={r['speedup']:.1f}x")
    return results


# ── SARIMAX benchmark ────────────────────────────────────────────────────────

def bench_sarimax(s_values=(7, 12, 24), max_pq=2, k_exog=2):
    all_results = {}
    for s in s_values:
        print(f"\n{'=' * 60}")
        print(f"SARIMAX Benchmark (d=1, D=1, s={s}, k_exog={k_exog}, n=500)")
        print("=" * 60)
        n = max(500, s * 20)
        X = generate_exog(n, k_exog, seed=123)
        y = generate_seasonal_data(n, s, seed=42)
        if len(y) > n:
            y = y[-n:]
        y = y + X[:len(y)] @ np.array([0.5, -0.3])[:k_exog]
        exog_rs = np.asfortranarray(X[:len(y)])

        results = []
        for p in range(max_pq + 1):
            for q in range(max_pq + 1):
                for P in range(max_pq + 1):
                    for Q in range(max_pq + 1):
                        if p + q + P + Q == 0:
                            continue
                        name = f"SARIMAX({p},1,{q})({P},1,{Q},{s})+k{k_exog}"
                        print(f"  {name}...", end=" ", flush=True)
                        r = run_comparison(name, y, (p, 1, q), (P, 1, Q, s), exog=exog_rs)
                        results.append(r)
                        delta_s = f"Δ={r['delta_ll']:.3f}" if np.isfinite(r['delta_ll']) else "Δ=NaN"
                        speed_s = f"speed={r['speedup']:.1f}x" if np.isfinite(r['speedup']) else "speed=N/A"
                        print(f"LL_rs={r['rs']['loglike']:.2f}  LL_sm={r['sm']['loglike']:.2f}  "
                              f"{delta_s}  {speed_s}")
        all_results[s] = results
    return all_results


# ── Report generation ────────────────────────────────────────────────────────

def gen_summary_stats(results):
    """Compute summary statistics from a list of comparison results."""
    valid = [r for r in results if np.isfinite(r['delta_ll'])]
    if not valid:
        return {'n_total': len(results), 'n_valid': 0}

    deltas = [r['delta_ll'] for r in valid]
    speedups = [r['speedup'] for r in valid if np.isfinite(r['speedup'])]
    rs_times = [r['rs']['time_ms'] for r in valid]
    sm_times = [r['sm']['time_ms'] for r in valid]

    n_close = sum(1 for d in deltas if d < 1.0)
    n_very_close = sum(1 for d in deltas if d < 0.1)
    n_medium = sum(1 for d in deltas if 1.0 <= d < 5.0)
    n_far = sum(1 for d in deltas if d >= 5.0)

    return {
        'n_total': len(results),
        'n_valid': len(valid),
        'n_close_1': n_close,
        'n_very_close_01': n_very_close,
        'n_medium': n_medium,
        'n_far': n_far,
        'pct_close_1': n_close / len(valid) * 100,
        'pct_very_close_01': n_very_close / len(valid) * 100,
        'mean_delta': np.mean(deltas),
        'median_delta': np.median(deltas),
        'p95_delta': np.percentile(deltas, 95),
        'max_delta': np.max(deltas),
        'mean_speedup': np.mean(speedups) if speedups else float('nan'),
        'median_speedup': np.median(speedups) if speedups else float('nan'),
        'mean_rs_ms': np.mean(rs_times),
        'mean_sm_ms': np.mean(sm_times),
    }


def write_md_report(filepath, title, results, stats):
    """Write a markdown comparison report."""
    with open(filepath, 'w') as f:
        f.write(f"# {title}\n\n")
        f.write(f"Generated: 2026-02-25\n\n")

        # Summary
        f.write("## Summary\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Total models | {stats['n_total']} |\n")
        f.write(f"| Valid comparisons | {stats['n_valid']} |\n")
        if stats['n_valid'] > 0:
            f.write(f"| Very close (Δ<0.1) | {stats['n_very_close_01']} ({stats['pct_very_close_01']:.1f}%) |\n")
            f.write(f"| Close (Δ<1.0) | {stats['n_close_1']} ({stats['pct_close_1']:.1f}%) |\n")
            f.write(f"| Medium (1.0≤Δ<5.0) | {stats['n_medium']} |\n")
            f.write(f"| Far (Δ≥5.0) | {stats['n_far']} |\n")
            f.write(f"| Mean Δ(loglike) | {stats['mean_delta']:.4f} |\n")
            f.write(f"| Median Δ(loglike) | {stats['median_delta']:.4f} |\n")
            f.write(f"| P95 Δ(loglike) | {stats['p95_delta']:.4f} |\n")
            f.write(f"| Max Δ(loglike) | {stats['max_delta']:.4f} |\n")
            f.write(f"| Mean speedup (sm/rs) | {stats['mean_speedup']:.1f}x |\n")
            f.write(f"| Mean time (rs) | {stats['mean_rs_ms']:.0f} ms |\n")
            f.write(f"| Mean time (sm) | {stats['mean_sm_ms']:.0f} ms |\n")

        # Detailed table
        f.write("\n## Detailed Results\n\n")
        f.write("| Model | LL_rs | LL_sm | Δ(LL) | AIC_rs | AIC_sm | rs(ms) | sm(ms) | Speedup |\n")
        f.write("|-------|-------|-------|-------|--------|--------|--------|--------|--------|\n")
        for r in results:
            ll_rs = f"{r['rs']['loglike']:.2f}" if np.isfinite(r['rs']['loglike']) else "N/A"
            ll_sm = f"{r['sm']['loglike']:.2f}" if np.isfinite(r['sm']['loglike']) else "N/A"
            delta = f"{r['delta_ll']:.3f}" if np.isfinite(r['delta_ll']) else "N/A"
            aic_rs = f"{r['rs']['aic']:.2f}" if np.isfinite(r['rs']['aic']) else "N/A"
            aic_sm = f"{r['sm']['aic']:.2f}" if np.isfinite(r['sm']['aic']) else "N/A"
            t_rs = f"{r['rs']['time_ms']:.0f}"
            t_sm = f"{r['sm']['time_ms']:.0f}"
            spd = f"{r['speedup']:.1f}x" if np.isfinite(r['speedup']) else "N/A"
            f.write(f"| {r['model']} | {ll_rs} | {ll_sm} | {delta} | {aic_rs} | {aic_sm} | {t_rs} | {t_sm} | {spd} |\n")

        # Large differences
        far = [r for r in results if np.isfinite(r['delta_ll']) and r['delta_ll'] >= 5.0]
        if far:
            f.write("\n## Models with Large Differences (Δ≥5.0)\n\n")
            for r in sorted(far, key=lambda x: -x['delta_ll']):
                f.write(f"- **{r['model']}**: Δ={r['delta_ll']:.3f} (rs={r['rs']['loglike']:.2f}, sm={r['sm']['loglike']:.2f})\n")

        f.write("\n---\n*Comparison: sarimax_rs vs statsmodels.tsa.statespace.SARIMAX*\n")
        f.write("*Note: statsmodels uses exact diffuse initialization; sarimax_rs uses approximate diffuse (kappa=1e6).*\n")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    all_data = {}

    # 1. ARIMA
    print("\n[1/4] ARIMA benchmark...")
    arima_results = bench_arima(max_pq=3)
    arima_stats = gen_summary_stats(arima_results)
    write_md_report(os.path.join(out_dir, "01_ARIMA_comparison.md"),
                    "ARIMA Comparison: sarimax_rs vs statsmodels", arima_results, arima_stats)
    all_data['arima'] = {'results': arima_results, 'stats': arima_stats}

    # 2. SARIMA
    print("\n[2/4] SARIMA benchmark...")
    sarima_all = bench_sarima(s_values=(7, 12, 24), max_pq=2)
    for s, results in sarima_all.items():
        stats = gen_summary_stats(results)
        write_md_report(os.path.join(out_dir, f"02_SARIMA_s{s}_comparison.md"),
                        f"SARIMA Comparison (s={s}): sarimax_rs vs statsmodels", results, stats)
        all_data[f'sarima_s{s}'] = {'stats': stats}

    # 3. ARIMAX
    print("\n[3/4] ARIMAX benchmark...")
    arimax_results = bench_arimax(max_pq=3, k_exog=2)
    arimax_stats = gen_summary_stats(arimax_results)
    write_md_report(os.path.join(out_dir, "03_ARIMAX_comparison.md"),
                    "ARIMAX Comparison: sarimax_rs vs statsmodels", arimax_results, arimax_stats)
    all_data['arimax'] = {'stats': arimax_stats}

    # 4. SARIMAX
    print("\n[4/4] SARIMAX benchmark...")
    sarimax_all = bench_sarimax(s_values=(7, 12, 24), max_pq=2, k_exog=2)
    for s, results in sarimax_all.items():
        stats = gen_summary_stats(results)
        write_md_report(os.path.join(out_dir, f"04_SARIMAX_s{s}_comparison.md"),
                        f"SARIMAX Comparison (s={s}): sarimax_rs vs statsmodels", results, stats)
        all_data[f'sarimax_s{s}'] = {'stats': stats}

    # 5. Summary report
    print("\n\nWriting summary report...")
    with open(os.path.join(out_dir, "00_SUMMARY.md"), 'w') as f:
        f.write("# sarimax_rs v1 Benchmark Summary\n\n")
        f.write("Generated: 2026-02-25\n\n")
        f.write("## Overview\n\n")
        f.write("Comprehensive comparison of `sarimax_rs` (Rust) vs `statsmodels` (Python) across\n")
        f.write("ARIMA, SARIMA, ARIMAX, and SARIMAX model families.\n\n")
        f.write("| Category | Models | Close (Δ<1) | Very Close (Δ<0.1) | Mean Δ | P95 Δ | Max Δ | Mean Speedup |\n")
        f.write("|----------|--------|-------------|---------------------|--------|-------|-------|-------------|\n")
        for key, data in all_data.items():
            s = data['stats']
            if s['n_valid'] == 0:
                continue
            f.write(f"| {key} | {s['n_valid']} | {s['pct_close_1']:.0f}% | {s['pct_very_close_01']:.0f}% | "
                    f"{s['mean_delta']:.3f} | {s['p95_delta']:.3f} | {s['max_delta']:.3f} | {s['mean_speedup']:.1f}x |\n")

        f.write("\n## Files\n\n")
        f.write("- `01_ARIMA_comparison.md` - ARIMA(p,1,q) for p,q in 0..3\n")
        f.write("- `02_SARIMA_s7_comparison.md` - SARIMA with s=7\n")
        f.write("- `02_SARIMA_s12_comparison.md` - SARIMA with s=12\n")
        f.write("- `02_SARIMA_s24_comparison.md` - SARIMA with s=24\n")
        f.write("- `03_ARIMAX_comparison.md` - ARIMAX(p,1,q) with 2 exog\n")
        f.write("- `04_SARIMAX_s7_comparison.md` - SARIMAX with s=7, 2 exog\n")
        f.write("- `04_SARIMAX_s12_comparison.md` - SARIMAX with s=12, 2 exog\n")
        f.write("- `04_SARIMAX_s24_comparison.md` - SARIMAX with s=24, 2 exog\n")

        f.write("\n## Notes\n\n")
        f.write("- **Initialization**: sarimax_rs uses approximate diffuse (kappa=1e6), statsmodels uses exact diffuse\n")
        f.write("- **Optimizer**: sarimax_rs uses L-BFGS-B + CSS pre-optimization, statsmodels uses L-BFGS-B only\n")
        f.write("- **Speedup**: Ratio of statsmodels time to sarimax_rs time (higher = rs is faster)\n")
        f.write("- **Δ(LL)**: Absolute difference in log-likelihood between the two engines\n")

    print("\nDone! Reports saved to result_v1/")
