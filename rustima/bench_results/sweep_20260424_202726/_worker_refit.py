
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
