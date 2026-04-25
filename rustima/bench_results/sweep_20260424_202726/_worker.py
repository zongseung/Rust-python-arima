
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
