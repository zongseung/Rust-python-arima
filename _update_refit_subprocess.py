"""Upgrade sweep-refit: subprocess isolation + timeout (no memory sampling)."""
import json
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


SWEEP_REFIT = r'''
# ======================================================================
# sweep-refit — sarimax_rs 엔진으로 재적합해 공정 비교 (aic/bic/hqic/loglik)
#
# 방어선:
#   1) 서브프로세스 격리 → refit OOM/segfault 에도 메인 커널 생존
#   2) per-refit timeout → hang 방지
#   3) 재시도 MAX_REFIT_ATTEMPTS 회, 중간 attempt 로그 삭제, 최종만 보존
#   4) CSV append + resume
#
# 메모리 타임라인은 기록하지 않음 — refit 은 criterion 비교용이지
# 속도/메모리 비교가 아니기 때문.
# ======================================================================
import csv
import pickle
import ast
import time
import json
import subprocess
import sys
import textwrap
from pathlib import Path

MAX_REFIT_ATTEMPTS = 3
PER_REFIT_TIMEOUT_S = 600      # refit 은 auto_arima 보다 훨씬 빨라야 정상
RETRY_BACKOFF_S = 1.0

csvs = sorted(Path("bench_results").glob("sweep_*/sweep_results.csv"))
assert csvs, "sweep-run 먼저 실행하세요."
OUT_DIR = csvs[-1].parent
main_csv = OUT_DIR / "sweep_results.csv"
refit_csv = OUT_DIR / "sweep_refit.csv"
y_pkl = OUT_DIR / "y.pkl"
assert y_pkl.exists(), f"{y_pkl} 없음. sweep-run 재실행 필요."

LOG_DIR = OUT_DIR / "logs_refit"
TMP_DIR = OUT_DIR / "_tmp_refit"
LOG_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

REFIT_FIELDS = [
    "idx", "library_source", "order", "seasonal_order", "trend", "s",
    "criterion", "status",
    "aic_refit", "bic_refit", "hqic_refit", "loglik_refit",
    "time_s", "attempts", "attempt_history",
    "error", "log_file",
]

if not refit_csv.exists():
    with open(refit_csv, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=REFIT_FIELDS).writeheader()

done = set()
with open(refit_csv, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        try:
            done.add((int(r["idx"]), r["library_source"]))
        except (ValueError, TypeError):
            pass
print(f"이미 재적합된 건: {len(done)}")


# ---- 워커 스크립트 생성 ---------------------------------------------
WORKER_REFIT = OUT_DIR / "_worker_refit.py"
WORKER_REFIT.write_text(textwrap.dedent(r"""
    import sys, time, json, pickle, traceback, ast

    (y_pkl, result_json, log_txt,
     order_s, seasonal_order_s, trend) = sys.argv[1:7]

    def _parse(s):
        if not s or s == "None":
            return None
        try:
            v = ast.literal_eval(s)
            return v if isinstance(v, tuple) else None
        except Exception:
            return None

    order = _parse(order_s)
    seasonal_order = _parse(seasonal_order_s)

    with open(y_pkl, "rb") as f:
        y_data = pickle.load(f)

    import io, contextlib
    buf = io.StringIO()
    status, err = "ok", ""
    aic = bic = hqic = loglik = None
    elapsed = None
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            from rustima import SARIMAXModel
            kw = dict(order=order, trend=trend)
            if seasonal_order is not None and len(seasonal_order) == 4:
                kw["seasonal_order"] = seasonal_order
            t0 = time.perf_counter()
            m = SARIMAXModel(y_data, **kw)
            r = m.fit()
            elapsed = time.perf_counter() - t0
            aic = getattr(r, "aic", None)
            bic = getattr(r, "bic", None)
            hqic = getattr(r, "hqic", None)
            loglik = (getattr(r, "loglik", None)
                      or getattr(r, "llf", None)
                      or getattr(r, "log_likelihood", None))
    except Exception as e:
        status = "fail"
        err = f"{type(e).__name__}: {e}"
        traceback.print_exc(file=buf)

    with open(log_txt, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump({
            "status": status, "error": err,
            "aic_refit":    aic    if isinstance(aic,    (int, float)) else None,
            "bic_refit":    bic    if isinstance(bic,    (int, float)) else None,
            "hqic_refit":   hqic   if isinstance(hqic,   (int, float)) else None,
            "loglik_refit": loglik if isinstance(loglik, (int, float)) else None,
            "time_s": elapsed,
        }, f)
"""), encoding="utf-8")


def _run_refit_attempt(order, seasonal_order, trend, tmp_result, tmp_log):
    out = {"status": "", "error": "",
           "aic_refit": "", "bic_refit": "", "hqic_refit": "", "loglik_refit": "",
           "time_s": ""}
    cmd = [sys.executable, str(WORKER_REFIT), str(y_pkl),
           str(tmp_result), str(tmp_log),
           str(order), str(seasonal_order), trend]
    try:
        cp = subprocess.run(cmd, timeout=PER_REFIT_TIMEOUT_S,
                            capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
        if tmp_result.exists():
            d = json.loads(tmp_result.read_text(encoding="utf-8"))
            out["status"] = d.get("status", "ok")
            out["error"] = (d.get("error") or "")[:500]
            for src_k, fmt in (
                ("aic_refit", "{:.4f}"), ("bic_refit", "{:.4f}"),
                ("hqic_refit", "{:.4f}"), ("loglik_refit", "{:.4f}"),
                ("time_s", "{:.4f}"),
            ):
                v = d.get(src_k)
                out[src_k] = fmt.format(v) if isinstance(v, (int, float)) else ""
            try: tmp_result.unlink()
            except OSError: pass
        else:
            out["status"] = "crashed"
            tail = (cp.stderr or cp.stdout or "").splitlines()[-5:]
            out["error"] = (" | ".join(tail))[:500]
            with open(tmp_log, "a", encoding="utf-8") as f:
                f.write(f"\n--- SUBPROCESS CRASH (exit={cp.returncode}) ---\n")
                f.write("STDOUT:\n" + (cp.stdout or "") + "\n")
                f.write("STDERR:\n" + (cp.stderr or "") + "\n")
    except subprocess.TimeoutExpired:
        out["status"] = "timeout"
        out["error"] = f"exceeded {PER_REFIT_TIMEOUT_S}s"
        with open(tmp_log, "a", encoding="utf-8") as f:
            f.write(f"\n--- TIMEOUT after {PER_REFIT_TIMEOUT_S}s ---\n")
    except Exception as e:
        out["status"] = "launch_fail"
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def _parse_tup(s):
    if not s or s == "None":
        return None
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, tuple) else None
    except Exception:
        return None


# ---- 소스 테이블 로드 ------------------------------------------------
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
print(f"로그 : {LOG_DIR}")

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

        tag = f"{idx:02d}_{lib}_refit"
        canonical_log = LOG_DIR / f"{tag}.log"

        print(f"\n[refit] idx={idx:02d} lib={lib:8s} order={order} "
              f"seasonal={seasonal_order} trend={trend}")

        attempt_log_paths = []
        histories = []
        final_out = None

        for attempt in range(1, MAX_REFIT_ATTEMPTS + 1):
            tmp_log = TMP_DIR / f"{tag}_try{attempt}.log"
            tmp_result = TMP_DIR / f"{tag}_try{attempt}.result.json"

            print(f"  ▶ attempt {attempt}/{MAX_REFIT_ATTEMPTS}")
            out = _run_refit_attempt(order, seasonal_order, trend,
                                     tmp_result, tmp_log)
            attempt_log_paths.append(tmp_log)
            histories.append(out["status"])
            final_out = out

            if out["status"] == "ok":
                print(f"    → ok  aic={out['aic_refit']} loglik={out['loglik_refit']} "
                      f"time={out['time_s']}s")
                break
            print(f"    → {out['status']}: {(out['error'] or '')[:140]}")
            if attempt < MAX_REFIT_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_S)

        # 최종 attempt 의 로그만 canonical 경로로, 나머지는 삭제
        final_log = attempt_log_paths[-1]
        try:
            if canonical_log.exists(): canonical_log.unlink()
            if final_log.exists(): final_log.replace(canonical_log)
        except OSError: pass
        for lp in attempt_log_paths[:-1]:
            try:
                if lp.exists(): lp.unlink()
            except OSError: pass

        row = {k: "" for k in REFIT_FIELDS}
        row.update({
            "idx": idx, "library_source": lib,
            "order": str(order),
            "seasonal_order": str(seasonal_order) if seasonal_order else "",
            "trend": trend, "s": src_row.get("s", ""),
            "criterion": src_row.get("criterion", ""),
            "attempts": len(histories),
            "attempt_history": ";".join(histories),
            "log_file": str(canonical_log.relative_to(OUT_DIR)),
        })
        row.update(final_out)

        with open(refit_csv, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=REFIT_FIELDS).writerow(row)
        print(f"  ▣ final [{row['status']}] history={row['attempt_history']}")
        done.add((idx, lib))

elapsed = time.perf_counter() - t_start
print(f"\n✅ 재적합 완료 — 누적 {len(done)} 건, 이번 세션 {elapsed/60:.1f}분")
print(f"   CSV : {refit_csv}")
'''

idx_refit = find_cell("sweep-refit")
assert idx_refit is not None, "sweep-refit cell not found"
cells[idx_refit]["source"] = to_lines(SWEEP_REFIT)
cells[idx_refit]["outputs"] = []
cells[idx_refit]["execution_count"] = None
print(f"Rewrote sweep-refit (position {idx_refit})")

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook saved.")
