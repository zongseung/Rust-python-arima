"""Rewrite sweep-run/refit/summary: retry logic, per-attempt timeline cleanup, reason keywords, light→heavy order."""
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
# (1) sweep-run — full rewrite: retry + light→heavy + attempt cleanup
# =====================================================================
SWEEP_RUN = r'''
# ======================================================================
# 전체 조합 스위프 — subprocess 격리 + Resume + 재시도
#
# 방어선:
#   1) 서브프로세스 격리 → OOM/segfault 에도 메인 커널 생존
#   2) CSV append + completed set → resume 자동 스킵
#   3) y.pkl → 커널 재시작에도 이 셀만 재실행
#   4) per-run timeout
#   5) 워커가 10샘플마다 mem_json flush
#   6) ★ 실패 시 MAX_ATTEMPTS 회까지 재시도
#        - 중간 실패 attempt 의 mem/log 파일은 삭제
#        - 최종 attempt (성공이든 실패든) 의 파일만 보존
#        - attempt_history 컬럼에 각 시도 결과 기록
#   7) ★ 가벼운 조합부터 실행 (seasonal=False → True, stepwise=True → False)
# ======================================================================
import itertools
import subprocess
import pickle
import csv
import json
import time
import sys
import os
import textwrap
from pathlib import Path
from datetime import datetime

# ---- 설정 ------------------------------------------------------------
# 가벼운 값 먼저 → 무거운 값 나중에 (itertools.product 는 첫 인자가 가장 느리게 순환)
SEASONAL_OPTS = [(False, 0), (True, 24)]   # 비계절 먼저 → 계절 s=24 나중
STEPWISE_OPTS = [True, False]              # stepwise 먼저 → grid search 나중
CRITERION_OPTS = ["aic", "bic", "hqic"]
TREND_OPTS = ["n", "c", "t", "ct"]

PER_RUN_TIMEOUT_S = 1800
MEM_INTERVAL_S = 0.05
MAX_ATTEMPTS = 3          # 초기 1회 + 재시도 2회
RETRY_BACKOFF_S = 2.0     # 재시도 전 대기
RESUME = True

combos = list(itertools.product(SEASONAL_OPTS, STEPWISE_OPTS, CRITERION_OPTS, TREND_OPTS))
RUNNERS = ["rustima", "pmdarima"]
total_runs = len(combos) * len(RUNNERS)

# ---- 출력 디렉터리 (resume or new) ----------------------------------
base = Path("bench_results")
base.mkdir(exist_ok=True)

FIELDS = [
    "idx", "library", "seasonal", "s", "stepwise", "criterion", "trend",
    "status", "order", "seasonal_order", "ic_value", "loglik_own",
    "time_s", "mem_baseline_mb", "mem_peak_mb", "mem_delta_mb",
    "attempts", "attempt_history",
    "error", "log_file", "mem_file",
]

OUT_DIR = None
if RESUME:
    for d in sorted(base.glob("sweep_*"), reverse=True):
        if (d / "sweep_results.csv").exists():
            OUT_DIR = d
            print(f"🔁 Resume — 기존 디렉터리: {OUT_DIR}")
            break
if OUT_DIR is None:
    STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR = base / f"sweep_{STAMP}"
    print(f"🆕 신규 디렉터리: {OUT_DIR}")

LOG_DIR = OUT_DIR / "logs"
MEM_DIR = OUT_DIR / "mem_samples"
TMP_DIR = OUT_DIR / "_tmp"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MEM_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

csv_path = OUT_DIR / "sweep_results.csv"
if not csv_path.exists():
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()

# ---- 완료 집합 -------------------------------------------------------
completed = set()
if csv_path.exists():
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status"):
                try:
                    completed.add((int(r["idx"]), r["library"]))
                except (TypeError, ValueError):
                    pass
print(f"이미 완료: {len(completed)} / {total_runs}")

# ---- y pickle --------------------------------------------------------
Y_PKL = OUT_DIR / "y.pkl"
if "y" in dir():
    with open(Y_PKL, "wb") as f:
        pickle.dump(y, f)
    print(f"y pickled → {Y_PKL}  (len={len(y) if hasattr(y,'__len__') else '?'})")
elif Y_PKL.exists():
    print(f"⚠️  커널에 y 미정의. 기존 y.pkl 재사용: {Y_PKL}")
else:
    raise RuntimeError(
        f"y 가 커널에도 없고 {Y_PKL} 도 없습니다. 데이터 로딩 셀을 먼저 실행하세요.")

# ---- 워커 스크립트 ---------------------------------------------------
WORKER = OUT_DIR / "_worker.py"
WORKER.write_text(textwrap.dedent(r"""
    import sys, os, time, json, pickle, threading, traceback, gc
    (tag, lib, s, stepwise, criterion, trend, y_pkl, result_json,
     mem_json, log_txt, interval) = sys.argv[1:12]
    s = int(s); stepwise = stepwise == "1"; interval = float(interval)

    import psutil
    proc = psutil.Process(os.getpid())

    stop = threading.Event()
    samples = []
    baseline = proc.memory_info().rss / 1024 ** 2
    peak = [baseline]

    def _flush_mem():
        with open(mem_json, "w", encoding="utf-8") as f:
            json.dump({"tag": tag, "baseline_mb": baseline,
                       "peak_mb": peak[0], "delta_mb": peak[0] - baseline,
                       "interval_s": interval, "samples": samples,
                       "partial": not stop.is_set()}, f)

    def _sampler():
        t0 = time.perf_counter()
        samples.append((0.0, baseline))
        i = 0
        while not stop.is_set():
            m = proc.memory_info().rss / 1024 ** 2
            samples.append((time.perf_counter() - t0, m))
            if m > peak[0]: peak[0] = m
            i += 1
            if i % 10 == 0:
                try: _flush_mem()
                except Exception: pass
            stop.wait(interval)

    th = threading.Thread(target=_sampler, daemon=True)
    th.start()

    with open(y_pkl, "rb") as f:
        y = pickle.load(f)

    import io, contextlib
    buf = io.StringIO()
    status, err = "ok", ""
    order = seasonal_order = ic_val = loglik_own = None
    elapsed = None
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            t0 = time.perf_counter()
            if lib == "rustima":
                from rustima import auto_arima as rs_auto_arima
                res = rs_auto_arima(y, s=s, trend=trend, stepwise=stepwise,
                                    trace=True, criterion=criterion)
                order = str(tuple(res.order))
                so = getattr(res, "seasonal_order", None)
                seasonal_order = str(tuple(so)) if so is not None else ""
                ic_val = getattr(res.result, criterion, None)
                loglik_own = (getattr(res.result, "loglik", None)
                              or getattr(res.result, "llf", None)
                              or getattr(res.result, "log_likelihood", None))
            else:
                import pmdarima as pm
                res = pm.auto_arima(
                    y, seasonal=(s > 0), m=s if s > 0 else 1, trend=trend,
                    stepwise=stepwise, suppress_warnings=True,
                    information_criterion=criterion, trace=True)
                order = str(tuple(res.order))
                seasonal_order = str(tuple(res.seasonal_order))
                getter = getattr(res, criterion, None)
                ic_val = getter() if callable(getter) else None
                ar = getattr(res, "arima_res_", None)
                if ar is not None:
                    loglik_own = getattr(ar, "llf", None)
                if loglik_own is None:
                    llf_fn = getattr(res, "llf", None)
                    loglik_own = llf_fn() if callable(llf_fn) else getattr(res, "loglik", None)
            elapsed = time.perf_counter() - t0
    except Exception as e:
        status = "fail"
        err = f"{type(e).__name__}: {e}"
        traceback.print_exc(file=buf)

    stop.set(); th.join()
    _flush_mem()
    with open(log_txt, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump({"status": status, "error": err,
                   "order": order or "", "seasonal_order": seasonal_order or "",
                   "ic_value": ic_val if isinstance(ic_val, (int, float)) else None,
                   "loglik_own": loglik_own if isinstance(loglik_own, (int, float)) else None,
                   "time_s": elapsed,
                   "mem_baseline_mb": baseline,
                   "mem_peak_mb": peak[0],
                   "mem_delta_mb": peak[0] - baseline}, f)
"""), encoding="utf-8")


def _run_one_attempt(cmd, result_json, log_path, tmp_log_path):
    """한 번의 subprocess 시도. (row_dict, log_txt_for_caller_append) 반환."""
    out = {"status": "", "error": "", "order": "", "seasonal_order": "",
           "ic_value": "", "loglik_own": "", "time_s": "",
           "mem_baseline_mb": "", "mem_peak_mb": "", "mem_delta_mb": ""}
    try:
        cp = subprocess.run(cmd, timeout=PER_RUN_TIMEOUT_S,
                            capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
        if result_json.exists():
            d = json.loads(result_json.read_text(encoding="utf-8"))
            out["status"] = d.get("status", "ok")
            out["error"] = (d.get("error") or "")[:500]
            out["order"] = d.get("order", "")
            out["seasonal_order"] = d.get("seasonal_order", "")
            for src_k, dst_k, fmt in (
                ("ic_value", "ic_value", "{:.4f}"),
                ("loglik_own", "loglik_own", "{:.4f}"),
                ("time_s", "time_s", "{:.4f}"),
                ("mem_baseline_mb", "mem_baseline_mb", "{:.2f}"),
                ("mem_peak_mb", "mem_peak_mb", "{:.2f}"),
                ("mem_delta_mb", "mem_delta_mb", "{:.2f}"),
            ):
                v = d.get(src_k)
                out[dst_k] = fmt.format(v) if isinstance(v, (int, float)) else ""
            try: result_json.unlink()
            except OSError: pass
        else:
            out["status"] = "crashed"
            tail = (cp.stderr or cp.stdout or "").splitlines()[-5:]
            out["error"] = (" | ".join(tail))[:500]
            with open(tmp_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n--- SUBPROCESS CRASH (exit={cp.returncode}) ---\n")
                f.write("STDOUT:\n" + (cp.stdout or "") + "\n")
                f.write("STDERR:\n" + (cp.stderr or "") + "\n")
    except subprocess.TimeoutExpired:
        out["status"] = "timeout"
        out["error"] = f"exceeded {PER_RUN_TIMEOUT_S}s"
        with open(tmp_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n--- TIMEOUT after {PER_RUN_TIMEOUT_S}s ---\n")
    except Exception as e:
        out["status"] = "launch_fail"
        out["error"] = f"{type(e).__name__}: {e}"
    return out


# ---- 메인 루프 -------------------------------------------------------
sweep_t0 = time.perf_counter()
run_i = 0
for idx, ((seasonal, s), stepwise, criterion, trend) in enumerate(combos):
    for lib_name in RUNNERS:
        run_i += 1
        if (idx, lib_name) in completed:
            continue
        tag = f"{idx:02d}_{lib_name}_s{s}_sw{int(stepwise)}_{criterion}_{trend}"
        canonical_mem = MEM_DIR / f"{tag}.json"
        canonical_log = LOG_DIR / f"{tag}.log"

        print(f"\n{'=' * 80}")
        print(f"[{run_i:03d}/{total_runs}] {lib_name}  seasonal={seasonal}(s={s}) "
              f"stepwise={stepwise} crit={criterion} trend={trend}")
        print("=" * 80)

        # 시도별 임시 파일
        attempt_mem_paths = []
        attempt_log_paths = []
        attempt_histories = []
        final_out = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            tmp_mem = TMP_DIR / f"{tag}_try{attempt}.mem.json"
            tmp_log = TMP_DIR / f"{tag}_try{attempt}.log"
            tmp_result = TMP_DIR / f"{tag}_try{attempt}.result.json"

            cmd = [sys.executable, str(WORKER), tag, lib_name, str(s),
                   "1" if stepwise else "0", criterion, trend,
                   str(Y_PKL), str(tmp_result), str(tmp_mem), str(tmp_log),
                   str(MEM_INTERVAL_S)]

            print(f"  ▶ attempt {attempt}/{MAX_ATTEMPTS}")
            out = _run_one_attempt(cmd, tmp_result, canonical_log, tmp_log)
            attempt_mem_paths.append(tmp_mem)
            attempt_log_paths.append(tmp_log)
            attempt_histories.append(out["status"])
            final_out = out

            if out["status"] == "ok":
                print(f"    → ok")
                break
            print(f"    → {out['status']}: {(out['error'] or '')[:140]}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_S)

        # 최종 시도의 mem/log → canonical 경로, 나머지는 삭제
        final_mem = attempt_mem_paths[-1]
        final_log = attempt_log_paths[-1]
        try:
            if canonical_mem.exists(): canonical_mem.unlink()
            if final_mem.exists(): final_mem.replace(canonical_mem)
        except OSError: pass
        try:
            if canonical_log.exists(): canonical_log.unlink()
            if final_log.exists(): final_log.replace(canonical_log)
        except OSError: pass
        for mp, lp in zip(attempt_mem_paths[:-1], attempt_log_paths[:-1]):
            for p in (mp, lp):
                try:
                    if p.exists(): p.unlink()
                except OSError: pass

        # CSV row
        row = {k: "" for k in FIELDS}
        row.update({
            "idx": idx, "library": lib_name,
            "seasonal": seasonal, "s": s, "stepwise": stepwise,
            "criterion": criterion, "trend": trend,
            "log_file": str(canonical_log.relative_to(OUT_DIR)),
            "mem_file": str(canonical_mem.relative_to(OUT_DIR)),
            "attempts": len(attempt_histories),
            "attempt_history": ";".join(attempt_histories),
        })
        row.update(final_out)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)

        print(f"  ▣ final [{row['status']}] attempts={len(attempt_histories)} "
              f"history={row['attempt_history']}")
        if row["status"] == "ok":
            print(f"    order={row['order']} ic={row['ic_value']} "
                  f"loglik={row['loglik_own']} time={row['time_s']}s "
                  f"peak={row['mem_peak_mb']}MB")
        elif row["error"]:
            print(f"    err: {row['error'][:200]}")

        completed.add((idx, lib_name))

elapsed = time.perf_counter() - sweep_t0
print(f"\n✅ 세션 종료 — 누적 완료 {len(completed)}/{total_runs}, "
      f"이번 세션 {elapsed/60:.1f}분 소요")
print(f"   CSV : {csv_path}")
print(f"   로그 : {LOG_DIR}")
print(f"   메모리 : {MEM_DIR}")
'''

idx_run = find_cell("sweep-run")
assert idx_run is not None
cells[idx_run]["source"] = to_lines(SWEEP_RUN)
cells[idx_run]["outputs"] = []
cells[idx_run]["execution_count"] = None
print(f"Rewrote sweep-run (position {idx_run})")


# =====================================================================
# (2) sweep-refit — add retry
# =====================================================================
SWEEP_REFIT = r'''
# 선택된 order 를 sarimax_rs 로 재적합 — 공정 비교용 (aic/bic/hqic/loglik)
# 재시도 로직 포함: MAX_ATTEMPTS 회까지 재시도 후 최종 결과 기록
import csv
import pickle
import ast
import time
import gc
from pathlib import Path

from rustima import SARIMAXModel

MAX_REFIT_ATTEMPTS = 3
RETRY_BACKOFF_S = 1.0

csvs = sorted(Path("bench_results").glob("sweep_*/sweep_results.csv"))
assert csvs, "sweep-run 먼저 실행하세요."
OUT_DIR = csvs[-1].parent
main_csv = OUT_DIR / "sweep_results.csv"
refit_csv = OUT_DIR / "sweep_refit.csv"
y_pkl = OUT_DIR / "y.pkl"
assert y_pkl.exists(), f"{y_pkl} 없음. sweep-run 재실행 필요."

with open(y_pkl, "rb") as f:
    y_data = pickle.load(f)
print(f"y 로드: len={len(y_data) if hasattr(y_data,'__len__') else '?'}")

REFIT_FIELDS = [
    "idx", "library_source", "order", "seasonal_order", "trend", "s",
    "criterion", "status", "aic_refit", "bic_refit", "hqic_refit",
    "loglik_refit", "attempts", "attempt_history", "error",
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
            continue  # 스위프 자체가 실패한 건 refit 대상 X
        try:
            idx = int(r["idx"])
        except (ValueError, TypeError):
            continue
        rows_by_idx.setdefault(idx, {})[r["library"]] = r

total = sum(len(v) for v in rows_by_idx.values())
print(f"재적합 대상: {total} 건 (sweep_results.csv 중 status=ok)")

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
              f"seasonal={seasonal_order} trend={trend}")

        histories = []
        final_res = None
        for attempt in range(1, MAX_REFIT_ATTEMPTS + 1):
            gc.collect()
            res = _refit(order, seasonal_order, trend)
            histories.append(res["status"])
            final_res = res
            print(f"  ▶ attempt {attempt}: [{res['status']}] "
                  f"{(res['error'] or '')[:120]}")
            if res["status"] == "ok":
                break
            if attempt < MAX_REFIT_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_S)

        row = {
            "idx": idx, "library_source": lib,
            "order": str(order),
            "seasonal_order": str(seasonal_order) if seasonal_order else "",
            "trend": trend, "s": src_row.get("s", ""),
            "criterion": src_row.get("criterion", ""),
            "status": final_res["status"],
            "attempts": len(histories),
            "attempt_history": ";".join(histories),
            "error": (final_res["error"] or "")[:300],
        }
        for k in ("aic_refit", "bic_refit", "hqic_refit", "loglik_refit"):
            v = final_res[k]
            row[k] = f"{v:.4f}" if isinstance(v, (int, float)) else ""
        with open(refit_csv, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=REFIT_FIELDS).writerow(row)
        print(f"  ▣ final [{row['status']}] aic={row['aic_refit']} "
              f"loglik={row['loglik_refit']} history={row['attempt_history']}")
        done.add((idx, lib))

elapsed = time.perf_counter() - t_start
print(f"\n✅ 재적합 완료 — 누적 {len(done)} 건, 이번 세션 {elapsed/60:.1f}분")
print(f"   CSV : {refit_csv}")
'''

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
    print(f"Replaced sweep-refit (position {idx_refit})")
else:
    cells.insert(idx_run + 1, refit_cell)
    print(f"Inserted sweep-refit (position {idx_run + 1})")


# =====================================================================
# (3) sweep-summary — 3 tables with reason keywords
# =====================================================================
SWEEP_SUMMARY = r'''
# 3 테이블 요약 + 실패 조합 키워드
#   Table 1 — criterion 값 + 속도 (각 패키지 자체 보고값)
#   Table 2 — 선택된 order + 각 패키지 자체 loglik
#   Table 3 — 동일 엔진(sarimax_rs) 재적합 criterion 공정 비교
#
# 실패한 런은 해당 셀에 [OOM] / [TIMEOUT] / [CRASH] / [NOCONV] / [SINGULAR] /
# [LINALG] / [OVERFLOW] / [NAN] / [FAIL] 같은 키워드로 채워집니다.
import pandas as pd
import numpy as np
from pathlib import Path

csvs = sorted(Path("bench_results").glob("sweep_*/sweep_results.csv"))
assert csvs, "sweep-run 먼저 실행하세요."
latest = csvs[-1]
out_dir = latest.parent
print(f"Source : {latest}")


def _reason(status, error):
    """status + error 메시지에서 짧은 키워드 추출."""
    if status == "ok":
        return ""
    if status == "timeout":
        return "TIMEOUT"
    if status == "crashed":
        return "CRASH"
    if status == "launch_fail":
        return "LAUNCH"
    e = (error or "")
    el = e.lower()
    if "memoryerror" in el or ("memory" in el and ("allocat" in el or "alloc" in el)):
        return "OOM"
    if "overflow" in el:
        return "OVERFLOW"
    if "did not converge" in el or "convergence" in el or "converge" in el:
        return "NOCONV"
    if "singular" in el:
        return "SINGULAR"
    if "linalg" in el or "linear algebra" in el:
        return "LINALG"
    if "nan" in el or "inf" in el:
        return "NAN"
    if "timeout" in el:
        return "TIMEOUT"
    return "FAIL"


# ---- 로드 ------------------------------------------------------------
df = pd.read_csv(latest)
for c in ("time_s", "mem_baseline_mb", "mem_peak_mb", "mem_delta_mb",
          "ic_value", "loglik_own"):
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
df["error"] = df.get("error", "").fillna("") if "error" in df.columns else ""
df["reason"] = df.apply(lambda r: _reason(r.get("status", ""), r.get("error", "")), axis=1)

key_cols = ["idx", "seasonal", "s", "stepwise", "criterion", "trend"]


def _lib_frame(lib, prefix):
    value_cols = [c for c in ("order", "seasonal_order", "ic_value", "time_s",
                               "mem_peak_mb", "loglik_own", "status", "error",
                               "reason", "attempts", "attempt_history")
                  if c in df.columns]
    sub = df[df.library == lib][key_cols + value_cols].copy()
    rename = {c: f"{prefix}_{c}" for c in value_cols}
    return sub.rename(columns=rename)


merged = _lib_frame("rustima", "rs").merge(
    _lib_frame("pmdarima", "pm"), on=key_cols, how="outer"
).sort_values("idx").reset_index(drop=True)


# ---- 포맷 헬퍼 --------------------------------------------------------
def _fmt_cell(val, reason, fmt):
    if reason:
        return f"[{reason}]"
    if pd.isna(val):
        return ""
    try:
        return fmt.format(val)
    except Exception:
        return str(val)


def _format_table(tbl, value_fmts):
    """value_fmts = {col: (fmt_str, reason_col)}  — reason_col 이 truthy 면 키워드로 대체."""
    out = tbl.copy()
    for col, (fmt, reason_col) in value_fmts.items():
        if col not in out.columns:
            continue
        if reason_col and reason_col in out.columns:
            out[col] = [_fmt_cell(v, r, fmt) for v, r in zip(tbl[col], tbl[reason_col])]
        else:
            out[col] = out[col].apply(lambda v: fmt.format(v) if pd.notna(v) else "")
    return out


# ---- Table 1 — criterion + 속도 --------------------------------------
t1 = merged[key_cols + ["rs_ic_value", "pm_ic_value", "rs_time_s", "pm_time_s",
                         "rs_reason", "pm_reason"]].copy()
t1["delta_ic"]   = t1["pm_ic_value"] - t1["rs_ic_value"]
t1["delta_time"] = t1["pm_time_s"]   - t1["rs_time_s"]
t1["speedup"]    = t1["pm_time_s"]   / t1["rs_time_s"]

t1_display = _format_table(t1, {
    "rs_ic_value":   ("{:.2f}",  "rs_reason"),
    "pm_ic_value":   ("{:.2f}",  "pm_reason"),
    "delta_ic":      ("{:+.2f}", None),
    "rs_time_s":     ("{:.3f}",  "rs_reason"),
    "pm_time_s":     ("{:.3f}",  "pm_reason"),
    "delta_time":    ("{:+.3f}", None),
    "speedup":       ("{:.2f}x", None),
})
t1_display = t1_display[key_cols + [
    "rs_ic_value", "pm_ic_value", "delta_ic",
    "rs_time_s", "pm_time_s", "delta_time", "speedup",
    "rs_reason", "pm_reason",
]]

# ---- Table 2 — order + 자체 loglik ----------------------------------
t2 = merged[key_cols + ["rs_order", "pm_order", "rs_loglik_own", "pm_loglik_own",
                         "rs_reason", "pm_reason"]].copy()
t2["same_order"]   = (t2["rs_order"].fillna("") == t2["pm_order"].fillna("")) & \
                     (t2["rs_order"].notna() & t2["pm_order"].notna())
t2["delta_loglik"] = t2["rs_loglik_own"] - t2["pm_loglik_own"]

t2_display = _format_table(t2, {
    "rs_loglik_own": ("{:.2f}",  "rs_reason"),
    "pm_loglik_own": ("{:.2f}",  "pm_reason"),
    "delta_loglik":  ("{:+.2f}", None),
})
# order 컬럼은 문자열이지만 실패 시 키워드로 대체
t2_display["rs_order"] = [f"[{r}]" if r else (str(o) if pd.notna(o) else "")
                          for o, r in zip(t2["rs_order"], t2["rs_reason"])]
t2_display["pm_order"] = [f"[{r}]" if r else (str(o) if pd.notna(o) else "")
                          for o, r in zip(t2["pm_order"], t2["pm_reason"])]
t2_display = t2_display[key_cols + [
    "rs_order", "pm_order", "same_order",
    "rs_loglik_own", "pm_loglik_own", "delta_loglik",
    "rs_reason", "pm_reason",
]]

# ---- Table 3 — sarimax_rs 재적합 criterion --------------------------
t3 = None
t3_display = None
refit_path = out_dir / "sweep_refit.csv"
if refit_path.exists():
    rf = pd.read_csv(refit_path)
    for c in ("aic_refit", "bic_refit", "hqic_refit", "loglik_refit"):
        if c in rf.columns:
            rf[c] = pd.to_numeric(rf[c], errors="coerce")
    rf["error"] = rf.get("error", "").fillna("") if "error" in rf.columns else ""
    rf["reason"] = rf.apply(
        lambda r: _reason(r.get("status", ""), r.get("error", "")), axis=1)

    def _refit_lib(lib, prefix):
        cols = [c for c in ("aic_refit", "bic_refit", "hqic_refit",
                             "loglik_refit", "reason")
                if c in rf.columns]
        sub = rf[rf.library_source == lib][["idx"] + cols].copy()
        return sub.rename(columns={c: f"{prefix}_{c}" for c in cols})

    t3 = (merged[key_cols + ["rs_order", "pm_order", "rs_reason", "pm_reason"]]
          .merge(_refit_lib("rustima", "rs"), on="idx", how="left")
          .merge(_refit_lib("pmdarima", "pm"), on="idx", how="left"))

    def _pick_crit(row, prefix):
        col = f"{prefix}_{row['criterion']}_refit"
        return row[col] if col in row.index else np.nan

    t3["rs_crit_refit"] = t3.apply(lambda r: _pick_crit(r, "rs"), axis=1)
    t3["pm_crit_refit"] = t3.apply(lambda r: _pick_crit(r, "pm"), axis=1)
    t3["delta_crit_refit"] = t3["pm_crit_refit"] - t3["rs_crit_refit"]

    # refit 자체 실패도 반영: 스위프에서는 ok 였어도 refit 실패면 키워드 표시
    t3["rs_final_reason"] = t3.apply(
        lambda r: r["rs_reason"] if r["rs_reason"]
        else (r.get("rs_reason_refit", "") or ""), axis=1)
    t3["pm_final_reason"] = t3.apply(
        lambda r: r["pm_reason"] if r["pm_reason"]
        else (r.get("pm_reason_refit", "") or ""), axis=1)

    t3_display = _format_table(t3, {
        "rs_crit_refit":    ("{:.2f}",  "rs_final_reason"),
        "pm_crit_refit":    ("{:.2f}",  "pm_final_reason"),
        "delta_crit_refit": ("{:+.2f}", None),
    })
    t3_display["rs_order"] = [f"[{r}]" if r else (str(o) if pd.notna(o) else "")
                              for o, r in zip(t3["rs_order"], t3["rs_final_reason"])]
    t3_display["pm_order"] = [f"[{r}]" if r else (str(o) if pd.notna(o) else "")
                              for o, r in zip(t3["pm_order"], t3["pm_final_reason"])]
    t3_display = t3_display[key_cols + [
        "rs_order", "pm_order",
        "rs_crit_refit", "pm_crit_refit", "delta_crit_refit",
        "rs_final_reason", "pm_final_reason",
    ]].rename(columns={"rs_final_reason": "rs_reason", "pm_final_reason": "pm_reason"})

# ---- 저장 ------------------------------------------------------------
p1 = out_dir / "table1_criterion_speed.csv"
p2 = out_dir / "table2_order_loglik.csv"
p3 = out_dir / "table3_refit_criterion.csv"
t1_display.to_csv(p1, index=False)
t2_display.to_csv(p2, index=False)
if t3_display is not None:
    t3_display.to_csv(p3, index=False)

# ---- 실패 요약 -------------------------------------------------------
fail_df = df[df["reason"] != ""][
    ["idx", "library", "seasonal", "stepwise", "criterion", "trend",
     "status", "reason", "attempts", "attempt_history", "error"]
].copy() if "attempts" in df.columns else df[df["reason"] != ""].copy()
fail_path = out_dir / "failures.csv"
fail_df.to_csv(fail_path, index=False)

# ---- 출력 ------------------------------------------------------------
with pd.option_context("display.max_rows", 200, "display.width", 260):
    print("\n" + "=" * 100)
    print("Table 1 — criterion 값 + 속도   (Δ = pm - rs,  speedup = pm_time / rs_time)")
    print("=" * 100)
    print(t1_display.to_string(index=False))

    print("\n" + "=" * 100)
    print("Table 2 — 선택된 order + 각 패키지 자체 loglik   (Δloglik = rs - pm)")
    print("=" * 100)
    print(t2_display.to_string(index=False))

    if t3_display is not None:
        print("\n" + "=" * 100)
        print("Table 3 — 동일 엔진(sarimax_rs) 재적합 criterion   (Δ = pm - rs)")
        print("=" * 100)
        print(t3_display.to_string(index=False))
    else:
        print("\n⚠️  Table 3 생략: sweep-refit 셀을 먼저 실행하세요.")

    if len(fail_df):
        print("\n" + "=" * 100)
        print(f"실패/스킵 목록 ({len(fail_df)}건) — keyword 별 구분")
        print("=" * 100)
        print(fail_df.to_string(index=False))
    else:
        print("\n✨ 실패 런 없음.")

print(f"\n저장 파일:")
print(f"  T1  {p1}")
print(f"  T2  {p2}")
if t3_display is not None:
    print(f"  T3  {p3}")
print(f"  F   {fail_path}")
'''

idx_sum = find_cell("sweep-summary")
assert idx_sum is not None
cells[idx_sum]["source"] = to_lines(SWEEP_SUMMARY)
cells[idx_sum]["outputs"] = []
cells[idx_sum]["execution_count"] = None
print(f"Rewrote sweep-summary (position {idx_sum})")


nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("\nNotebook saved.")
