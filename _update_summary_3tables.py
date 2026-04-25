"""Update sweep cells: add loglik_own capture, add sweep-refit, rewrite sweep-summary (3 tables)."""
import json
import sys
from pathlib import Path

nb_path = Path(r"c:/Rust-python-arima/rustima/prac_readme_jihun.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))
cells = nb["cells"]


def find_cell(cell_id):
    return next((i for i, c in enumerate(cells) if c.get("id") == cell_id), None)


def to_lines(text):
    if text.startswith("\n"):
        text = text[1:]
    return text.splitlines(keepends=True)


# =====================================================================
# (1) Surgical patch of sweep-run — add loglik_own capture
# =====================================================================
idx_run = find_cell("sweep-run")
assert idx_run is not None, "sweep-run cell not found"
src = "".join(cells[idx_run]["source"])


def _patch(src, old, new, label):
    if new in src:
        print(f"  [skip] {label} — already applied")
        return src
    if old not in src:
        print(f"  [WARN] {label} — old pattern not found, skipping")
        return src
    print(f"  [ok]   {label}")
    return src.replace(old, new, 1)


print("Patching sweep-run:")

src = _patch(src,
    '    "status", "order", "seasonal_order", "ic_value",\n',
    '    "status", "order", "seasonal_order", "ic_value", "loglik_own",\n',
    "FIELDS += loglik_own")

src = _patch(src,
    '    order = seasonal_order = ic_val = None\n',
    '    order = seasonal_order = ic_val = loglik_own = None\n',
    "worker init vars")

src = _patch(src,
    '                ic_val = getattr(res.result, criterion, None)\n            else:',
    '                ic_val = getattr(res.result, criterion, None)\n'
    '                loglik_own = (getattr(res.result, "loglik", None)\n'
    '                              or getattr(res.result, "llf", None)\n'
    '                              or getattr(res.result, "log_likelihood", None))\n'
    '            else:',
    "worker rustima loglik")

src = _patch(src,
    '                getter = getattr(res, criterion, None)\n'
    '                ic_val = getter() if callable(getter) else None\n'
    '            elapsed',
    '                getter = getattr(res, criterion, None)\n'
    '                ic_val = getter() if callable(getter) else None\n'
    '                ar = getattr(res, "arima_res_", None)\n'
    '                if ar is not None:\n'
    '                    loglik_own = getattr(ar, "llf", None)\n'
    '                if loglik_own is None:\n'
    '                    llf_fn = getattr(res, "llf", None)\n'
    '                    loglik_own = llf_fn() if callable(llf_fn) else getattr(res, "loglik", None)\n'
    '            elapsed',
    "worker pmdarima loglik")

src = _patch(src,
    '"ic_value": ic_val if isinstance(ic_val, (int, float)) else None,\n'
    '                   "time_s": elapsed,',
    '"ic_value": ic_val if isinstance(ic_val, (int, float)) else None,\n'
    '                   "loglik_own": loglik_own if isinstance(loglik_own, (int, float)) else None,\n'
    '                   "time_s": elapsed,',
    "worker result JSON")

src = _patch(src,
    '                ic = d.get("ic_value")\n'
    '                row["ic_value"] = f"{ic:.4f}" if isinstance(ic, (int, float)) else ""\n'
    '                t_s = d.get("time_s")',
    '                ic = d.get("ic_value")\n'
    '                row["ic_value"] = f"{ic:.4f}" if isinstance(ic, (int, float)) else ""\n'
    '                ll = d.get("loglik_own")\n'
    '                row["loglik_own"] = f"{ll:.4f}" if isinstance(ll, (int, float)) else ""\n'
    '                t_s = d.get("time_s")',
    "parent row parsing")

src = _patch(src,
    "f\"ic={row['ic_value']} time={row['time_s']}s \"\n"
    "              f\"peak={row['mem_peak_mb']}MB\"",
    "f\"ic={row['ic_value']} loglik={row['loglik_own']} \"\n"
    "              f\"time={row['time_s']}s peak={row['mem_peak_mb']}MB\"",
    "parent print")

cells[idx_run]["source"] = src.splitlines(keepends=True)
cells[idx_run]["outputs"] = []
cells[idx_run]["execution_count"] = None


# =====================================================================
# (2) Insert/replace sweep-refit
# =====================================================================
SWEEP_REFIT = r"""
# 선택된 order 를 sarimax_rs 엔진으로 재적합 — 공정 비교용 (aic / bic / hqic / loglik)
# - sweep_results.csv 에서 각 (idx, library) 의 order 와 seasonal_order 를 읽고
# - sarimax_rs 로 재적합한 후 sweep_refit.csv 에 append
# - 이미 저장된 (idx, library_source) 는 자동 스킵 (resume)
import csv
import pickle
import ast
import time
from pathlib import Path

from rustima import SARIMAXModel

csvs = sorted(Path("bench_results").glob("sweep_*/sweep_results.csv"))
assert csvs, "sweep-run 먼저 실행하세요."
OUT_DIR = csvs[-1].parent
main_csv = OUT_DIR / "sweep_results.csv"
refit_csv = OUT_DIR / "sweep_refit.csv"
y_pkl = OUT_DIR / "y.pkl"
assert y_pkl.exists(), f"{y_pkl} 없음. sweep-run 을 다시 실행하세요."

with open(y_pkl, "rb") as f:
    y_data = pickle.load(f)
print(f"y 로드: len={len(y_data) if hasattr(y_data,'__len__') else '?'}")

REFIT_FIELDS = [
    "idx", "library_source", "order", "seasonal_order", "trend", "s",
    "criterion", "status", "aic_refit", "bic_refit", "hqic_refit",
    "loglik_refit", "error",
]

done = set()
if refit_csv.exists():
    with open(refit_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                done.add((int(r["idx"]), r["library_source"]))
            except (ValueError, TypeError):
                pass
else:
    with open(refit_csv, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=REFIT_FIELDS).writeheader()
print(f"이미 재적합된 건: {len(done)}")


def _parse_tup(s):
    if not s or s == "None":
        return None
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, tuple) else None
    except Exception:
        return None


def _refit(order, seasonal_order, trend):
    try:
        kw = dict(order=order, trend=trend)
        if seasonal_order is not None and len(seasonal_order) == 4:
            kw["seasonal_order"] = seasonal_order
        m = SARIMAXModel(y_data, **kw)
        r = m.fit()
        return {
            "status": "ok",
            "aic_refit": getattr(r, "aic", None),
            "bic_refit": getattr(r, "bic", None),
            "hqic_refit": getattr(r, "hqic", None),
            "loglik_refit": (getattr(r, "loglik", None)
                             or getattr(r, "llf", None)
                             or getattr(r, "log_likelihood", None)),
            "error": "",
        }
    except Exception as e:
        return {"status": "fail", "aic_refit": None, "bic_refit": None,
                "hqic_refit": None, "loglik_refit": None,
                "error": f"{type(e).__name__}: {e}"}


rows_by_idx = {}
with open(main_csv, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r.get("status") != "ok":
            continue
        try:
            idx = int(r["idx"])
        except (ValueError, TypeError):
            continue
        rows_by_idx.setdefault(idx, {})[r["library"]] = r

total = sum(len(v) for v in rows_by_idx.values())
print(f"재적합 대상: {total} 건 (sweep_results.csv 중 status=ok 만)")

t_start = time.perf_counter()
for idx in sorted(rows_by_idx.keys()):
    for lib, src_row in rows_by_idx[idx].items():
        if (idx, lib) in done:
            continue
        order = _parse_tup(src_row.get("order"))
        seasonal_order = _parse_tup(src_row.get("seasonal_order"))
        trend = src_row.get("trend", "n")
        if order is None:
            continue
        print(f"[refit] idx={idx:02d} lib={lib:8s} order={order} "
              f"seasonal={seasonal_order} trend={trend}", end="  ")
        res = _refit(order, seasonal_order, trend)
        row = {
            "idx": idx, "library_source": lib,
            "order": str(order),
            "seasonal_order": str(seasonal_order) if seasonal_order else "",
            "trend": trend, "s": src_row.get("s", ""),
            "criterion": src_row.get("criterion", ""),
            "status": res["status"],
            "error": (res["error"] or "")[:300],
        }
        for k in ("aic_refit", "bic_refit", "hqic_refit", "loglik_refit"):
            v = res[k]
            row[k] = f"{v:.4f}" if isinstance(v, (int, float)) else ""
        with open(refit_csv, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=REFIT_FIELDS).writerow(row)
        print(f"→ [{res['status']}] aic={row['aic_refit']} loglik={row['loglik_refit']}")
        done.add((idx, lib))

elapsed = time.perf_counter() - t_start
print(f"\n✅ 재적합 완료 — 누적 {len(done)} 건, 이번 세션 {elapsed/60:.1f}분")
print(f"   CSV : {refit_csv}")
"""

idx_refit = find_cell("sweep-refit")
refit_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "sweep-refit",
    "metadata": {},
    "outputs": [],
    "source": to_lines(SWEEP_REFIT),
}
if idx_refit is not None:
    cells[idx_refit] = refit_cell
    print(f"Replaced sweep-refit at position {idx_refit}")
else:
    cells.insert(idx_run + 1, refit_cell)
    print(f"Inserted sweep-refit after sweep-run (position {idx_run + 1})")


# =====================================================================
# (3) Replace sweep-summary — 3 tables
# =====================================================================
SWEEP_SUMMARY = r"""
# 3 테이블 요약
#   Table 1 — criterion 값 + 속도 (각 패키지 자체 보고값)
#   Table 2 — 선택된 order + 각 패키지 자체 loglik
#   Table 3 — 동일 엔진(sarimax_rs) 재적합 criterion 공정 비교
import pandas as pd
import numpy as np
from pathlib import Path

csvs = sorted(Path("bench_results").glob("sweep_*/sweep_results.csv"))
assert csvs, "sweep-run 먼저 실행하세요."
latest = csvs[-1]
out_dir = latest.parent
print(f"Source : {latest}")

df = pd.read_csv(latest)
for c in ("time_s", "mem_baseline_mb", "mem_peak_mb", "mem_delta_mb",
          "ic_value", "loglik_own"):
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

ok = df[df.status == "ok"].copy()
key_cols = ["idx", "seasonal", "s", "stepwise", "criterion", "trend"]


def _pivot(col):
    p = (ok.pivot_table(index=key_cols, columns="library",
                        values=col, aggfunc="first").reset_index())
    return p.rename(columns={"rustima": f"rs_{col}", "pmdarima": f"pm_{col}"})


merged = _pivot("order")
for col in ("seasonal_order", "ic_value", "time_s", "mem_peak_mb", "loglik_own"):
    if col in ok.columns:
        merged = merged.merge(_pivot(col), on=key_cols, how="outer")
merged = merged.sort_values("idx").reset_index(drop=True)

# ---------------- Table 1 — criterion + 속도 ----------------
t1 = merged[key_cols + ["rs_ic_value", "pm_ic_value",
                         "rs_time_s", "pm_time_s"]].copy()
t1["delta_ic"]   = t1["pm_ic_value"] - t1["rs_ic_value"]
t1["delta_time"] = t1["pm_time_s"]   - t1["rs_time_s"]
t1["speedup"]    = t1["pm_time_s"]   / t1["rs_time_s"]
t1 = t1[key_cols + ["rs_ic_value", "pm_ic_value", "delta_ic",
                     "rs_time_s", "pm_time_s", "delta_time", "speedup"]]

# ---------------- Table 2 — order + 자체 loglik ----------------
t2 = merged[key_cols + ["rs_order", "pm_order",
                         "rs_loglik_own", "pm_loglik_own"]].copy()
t2["same_order"]   = (t2["rs_order"] == t2["pm_order"])
t2["delta_loglik"] = t2["rs_loglik_own"] - t2["pm_loglik_own"]
t2 = t2[key_cols + ["rs_order", "pm_order", "same_order",
                     "rs_loglik_own", "pm_loglik_own", "delta_loglik"]]

# ---------------- Table 3 — sarimax_rs 재적합 criterion ----------------
t3 = None
refit_path = out_dir / "sweep_refit.csv"
if refit_path.exists():
    rf = pd.read_csv(refit_path)
    for c in ("aic_refit", "bic_refit", "hqic_refit", "loglik_refit"):
        if c in rf.columns:
            rf[c] = pd.to_numeric(rf[c], errors="coerce")

    rs_rf = rf[rf.library_source == "rustima"][
        ["idx", "aic_refit", "bic_refit", "hqic_refit", "loglik_refit"]
    ].rename(columns=lambda c: c if c == "idx" else "rs_" + c)
    pm_rf = rf[rf.library_source == "pmdarima"][
        ["idx", "aic_refit", "bic_refit", "hqic_refit", "loglik_refit"]
    ].rename(columns=lambda c: c if c == "idx" else "pm_" + c)

    t3 = (merged[key_cols + ["rs_order", "pm_order"]]
          .merge(rs_rf, on="idx", how="left")
          .merge(pm_rf, on="idx", how="left"))

    def _pick(row, prefix):
        col = f"{prefix}_{row['criterion']}_refit"
        return row[col] if col in row.index else np.nan

    t3["rs_crit_refit"]    = t3.apply(lambda r: _pick(r, "rs"), axis=1)
    t3["pm_crit_refit"]    = t3.apply(lambda r: _pick(r, "pm"), axis=1)
    t3["delta_crit_refit"] = t3["pm_crit_refit"] - t3["rs_crit_refit"]
    t3 = t3[key_cols + ["rs_order", "pm_order",
                         "rs_crit_refit", "pm_crit_refit", "delta_crit_refit"]]

# ---------------- 저장 ----------------
p1 = out_dir / "table1_criterion_speed.csv"
p2 = out_dir / "table2_order_loglik.csv"
p3 = out_dir / "table3_refit_criterion.csv"
t1.to_csv(p1, index=False)
t2.to_csv(p2, index=False)
if t3 is not None:
    t3.to_csv(p3, index=False)

# ---------------- 출력 포맷 ----------------
FMT = {
    "rs_ic_value": "{:.2f}", "pm_ic_value": "{:.2f}", "delta_ic": "{:+.2f}",
    "rs_time_s": "{:.3f}",   "pm_time_s": "{:.3f}",   "delta_time": "{:+.3f}",
    "speedup": "{:.2f}x",
    "rs_loglik_own": "{:.2f}", "pm_loglik_own": "{:.2f}", "delta_loglik": "{:+.2f}",
    "rs_crit_refit": "{:.2f}", "pm_crit_refit": "{:.2f}", "delta_crit_refit": "{:+.2f}",
}


def _fmt(df_):
    out = df_.copy()
    for c, f in FMT.items():
        if c in out.columns:
            out[c] = out[c].apply(lambda v: f.format(v) if pd.notna(v) else "")
    return out


print("\n" + "=" * 100)
print("Table 1 — criterion 값 + 속도   (Δ = pm - rs,  speedup = pm_time / rs_time)")
print("=" * 100)
with pd.option_context("display.max_rows", 200, "display.width", 240):
    print(_fmt(t1).to_string(index=False))

print("\n" + "=" * 100)
print("Table 2 — 선택된 order + 각 패키지 자체 loglik   (Δloglik = rs - pm,  양수면 rs 가 데이터에 더 맞음)")
print("=" * 100)
with pd.option_context("display.max_rows", 200, "display.width", 240):
    print(_fmt(t2).to_string(index=False))

if t3 is not None:
    print("\n" + "=" * 100)
    print("Table 3 — 동일 엔진(sarimax_rs) 재적합 criterion   (Δ = pm - rs,  음수면 rs 가 선택한 order 가 더 우수)")
    print("=" * 100)
    with pd.option_context("display.max_rows", 200, "display.width", 240):
        print(_fmt(t3).to_string(index=False))
else:
    print("\n⚠️  Table 3 생략: sweep-refit 셀을 먼저 실행하세요.")

print(f"\n저장 파일:")
print(f"  T1  {p1}")
print(f"  T2  {p2}")
if t3 is not None:
    print(f"  T3  {p3}")
"""

idx_sum = find_cell("sweep-summary")
assert idx_sum is not None, "sweep-summary cell not found"
cells[idx_sum]["source"] = to_lines(SWEEP_SUMMARY)
cells[idx_sum]["outputs"] = []
cells[idx_sum]["execution_count"] = None
print(f"Replaced sweep-summary at position {idx_sum}")


# =====================================================================
# Save
# =====================================================================
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("\nNotebook saved.")
