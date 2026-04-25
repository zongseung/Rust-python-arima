"""Insert bench helpers + combinatorial sweep after mem-bench-header."""
import json
import sys
from pathlib import Path

nb_path = Path(r"c:/Rust-python-arima/rustima/prac_readme_jihun.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

ANCHOR_ID = "mem-bench-header"


def md_cell(cell_id, lines):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": lines}


def code_cell(cell_id, lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


# ---- (재)추가: 이전 단계에서 유실된 bench helper 3종 + compare ----
setup = code_cell(
    "mem-bench-setup",
    [
        "# 공통 설정: 메모리 측정 유틸 (단일-실행 벤치용)\n",
        "import time\n",
        "import gc\n",
        "import os\n",
        "import psutil  # 설치 필요: pip install psutil\n",
        "\n",
        "process = psutil.Process(os.getpid())\n",
        "\n",
        "def mem_mb():\n",
        "    return process.memory_info().rss / (1024 ** 2)\n",
        "\n",
        "def bench(label, fn):\n",
        "    gc.collect()\n",
        "    time.sleep(1)\n",
        "    mem_before = mem_mb()\n",
        "    t0 = time.perf_counter()\n",
        "    result = fn()\n",
        "    elapsed = time.perf_counter() - t0\n",
        "    mem_after = mem_mb()\n",
        "    print(f\"[{label}]\")\n",
        "    print(f\"  time   : {elapsed:.3f}s\")\n",
        "    print(f\"  mem Δ  : {mem_after - mem_before:+.1f} MB \"\n",
        "          f\"(before={mem_before:.1f}, after={mem_after:.1f})\")\n",
        "    return result, elapsed"
    ],
)

rustima_cell = code_cell(
    "mem-bench-rustima",
    [
        "# rustima(sarimax_rs) 단독 실행 — btop 에서 피크 메모리 관찰\n",
        "from rustima import auto_arima as rs_auto_arima\n",
        "\n",
        "def _run_rustima():\n",
        "    return rs_auto_arima(y, s=0, trend='c', stepwise=True,\n",
        "                         trace=True, criterion='bic')\n",
        "\n",
        "auto_rs_bic, t_rs = bench(\"sarimax_rs auto_arima\", _run_rustima)\n",
        "print(f\"  order  : {auto_rs_bic.order}\")\n",
        "print(f\"  BIC    : {auto_rs_bic.result.bic:.2f}\")\n",
        "\n",
        "del auto_rs_bic\n",
        "gc.collect()"
    ],
)

pmdarima_cell = code_cell(
    "mem-bench-pmdarima",
    [
        "# pmdarima 단독 실행 — btop 에서 피크 메모리 관찰\n",
        "import pmdarima as pm\n",
        "\n",
        "def _run_pmdarima():\n",
        "    return pm.auto_arima(y, seasonal=False, trend='c', stepwise=True,\n",
        "                         suppress_warnings=True, information_criterion='bic')\n",
        "\n",
        "auto_pm_bic, t_pm = bench(\"pmdarima auto_arima\", _run_pmdarima)\n",
        "print(f\"  order  : {auto_pm_bic.order}\")\n",
        "print(f\"  BIC    : {auto_pm_bic.bic():.2f}\")\n",
        "\n",
        "del auto_pm_bic\n",
        "gc.collect()"
    ],
)

compare = code_cell(
    "mem-bench-compare",
    [
        "# 속도 요약\n",
        "print(f\"속도 비교: sarimax_rs 가 pmdarima 대비 {t_pm/t_rs:.1f}x faster\")"
    ],
)

# ---- 신규: 전체 조합 스위프 --------------------------------------
sweep_header = md_cell(
    "sweep-header",
    [
        "### 전체 조합 스위프 — rustima vs pmdarima\n",
        "\n",
        "`seasonal × stepwise × criterion × trend` = **2 × 2 × 3 × 4 = 48 조합**, 라이브러리 2개까지 합쳐 **총 96런** 을 자동 실행합니다.\n",
        "\n",
        "- 각 런마다 50 ms 간격으로 RSS 를 샘플링한 **메모리 타임라인(JSON)** 과 `trace=True` **stdout 로그(.log)** 를 저장합니다.\n",
        "- 전체 결과는 `bench_results/sweep_<timestamp>/sweep_results.csv` 로 누적 저장되므로, 중간에 중단돼도 일부 결과가 유지됩니다.\n",
        "- btop 은 시각 관찰용이므로 96 런을 수동 캡처하는 대신 **psutil 샘플러**가 btop 역할을 대신합니다. 원하면 별도 터미널의 btop 도 함께 띄우세요.\n",
        "- 파라미터 조합에 따라 수렴 실패가 날 수 있어 각 런을 `try/except` 로 감싸고, 실패 런도 CSV 에 `status=fail` 로 기록합니다.\n",
        "- **주의:** `y` 시퀀스 크기/조합에 따라 실행이 수십 분 ~ 수 시간까지 걸릴 수 있습니다."
    ],
)

sweep_run = code_cell(
    "sweep-run",
    [
        "# 전체 조합 스위프 실행 — 결과는 bench_results/sweep_<timestamp>/ 에 저장\n",
        "import itertools\n",
        "import threading\n",
        "import io\n",
        "import contextlib\n",
        "import traceback\n",
        "import csv\n",
        "import gc\n",
        "import time\n",
        "import os\n",
        "import json\n",
        "from pathlib import Path\n",
        "from datetime import datetime\n",
        "\n",
        "import psutil\n",
        "import pmdarima as pm\n",
        "from rustima import auto_arima as rs_auto_arima\n",
        "\n",
        "# ---- 출력 디렉터리 -------------------------------------------------\n",
        "STAMP = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n",
        "OUT_DIR = Path(\"bench_results\") / f\"sweep_{STAMP}\"\n",
        "LOG_DIR = OUT_DIR / \"logs\"\n",
        "MEM_DIR = OUT_DIR / \"mem_samples\"\n",
        "LOG_DIR.mkdir(parents=True, exist_ok=True)\n",
        "MEM_DIR.mkdir(parents=True, exist_ok=True)\n",
        "print(f\"결과 저장 경로: {OUT_DIR.resolve()}\")\n",
        "\n",
        "_proc = psutil.Process(os.getpid())\n",
        "\n",
        "\n",
        "class MemSampler:\n",
        "    \"\"\"50 ms 간격으로 RSS 를 기록하는 백그라운드 샘플러 (btop 대체).\"\"\"\n",
        "    def __init__(self, interval=0.05):\n",
        "        self.interval = interval\n",
        "        self._stop = threading.Event()\n",
        "        self._thread = None\n",
        "        self.baseline = 0.0\n",
        "        self.peak = 0.0\n",
        "        self.samples = []  # (elapsed_s, rss_mb)\n",
        "    def __enter__(self):\n",
        "        gc.collect()\n",
        "        time.sleep(0.3)\n",
        "        self.baseline = self.peak = _proc.memory_info().rss / 1024 ** 2\n",
        "        self.samples = [(0.0, self.baseline)]\n",
        "        t0 = time.perf_counter()\n",
        "        def _run():\n",
        "            while not self._stop.is_set():\n",
        "                m = _proc.memory_info().rss / 1024 ** 2\n",
        "                self.samples.append((time.perf_counter() - t0, m))\n",
        "                if m > self.peak:\n",
        "                    self.peak = m\n",
        "                self._stop.wait(self.interval)\n",
        "        self._thread = threading.Thread(target=_run, daemon=True)\n",
        "        self._thread.start()\n",
        "        return self\n",
        "    def __exit__(self, *exc):\n",
        "        self._stop.set()\n",
        "        self._thread.join()\n",
        "\n",
        "\n",
        "# ---- 조합 정의 ------------------------------------------------------\n",
        "SEASONAL_OPTS = [(True, 24), (False, 0)]\n",
        "STEPWISE_OPTS = [True, False]\n",
        "CRITERION_OPTS = [\"aic\", \"bic\", \"hqic\"]\n",
        "TREND_OPTS = [\"n\", \"c\", \"t\", \"ct\"]\n",
        "combos = list(itertools.product(SEASONAL_OPTS, STEPWISE_OPTS, CRITERION_OPTS, TREND_OPTS))\n",
        "print(f\"조합 수: {len(combos)}, 총 실행 수: {len(combos) * 2}  (rustima + pmdarima)\")\n",
        "\n",
        "\n",
        "# ---- 라이브러리별 러너 --------------------------------------------\n",
        "def _run_rustima(y, s, stepwise, criterion, trend):\n",
        "    return rs_auto_arima(y, s=s, trend=trend, stepwise=stepwise,\n",
        "                         trace=True, criterion=criterion)\n",
        "\n",
        "def _run_pmdarima(y, s, stepwise, criterion, trend):\n",
        "    return pm.auto_arima(\n",
        "        y,\n",
        "        seasonal=(s > 0),\n",
        "        m=s if s > 0 else 1,\n",
        "        trend=trend,\n",
        "        stepwise=stepwise,\n",
        "        suppress_warnings=True,\n",
        "        information_criterion=criterion,\n",
        "        trace=True,\n",
        "    )\n",
        "\n",
        "RUNNERS = [(\"rustima\", _run_rustima), (\"pmdarima\", _run_pmdarima)]\n",
        "\n",
        "# ---- CSV 헤더 ------------------------------------------------------\n",
        "FIELDS = [\n",
        "    \"idx\", \"library\", \"seasonal\", \"s\", \"stepwise\", \"criterion\", \"trend\",\n",
        "    \"status\", \"order\", \"seasonal_order\", \"ic_value\",\n",
        "    \"time_s\", \"mem_baseline_mb\", \"mem_peak_mb\", \"mem_delta_mb\",\n",
        "    \"error\", \"log_file\", \"mem_file\",\n",
        "]\n",
        "csv_path = OUT_DIR / \"sweep_results.csv\"\n",
        "with open(csv_path, \"w\", newline=\"\", encoding=\"utf-8\") as f:\n",
        "    csv.DictWriter(f, fieldnames=FIELDS).writeheader()\n",
        "\n",
        "# ---- 메인 루프 -----------------------------------------------------\n",
        "total = len(combos) * len(RUNNERS)\n",
        "run_i = 0\n",
        "sweep_t0 = time.perf_counter()\n",
        "for idx, ((seasonal, s), stepwise, criterion, trend) in enumerate(combos):\n",
        "    for lib_name, runner in RUNNERS:\n",
        "        run_i += 1\n",
        "        tag = f\"{idx:02d}_{lib_name}_s{s}_sw{int(stepwise)}_{criterion}_{trend}\"\n",
        "        print(f\"\\n{'=' * 80}\")\n",
        "        print(f\"[{run_i:03d}/{total}] {lib_name}  seasonal={seasonal}(s={s}) \"\n",
        "              f\"stepwise={stepwise} crit={criterion} trend={trend}\")\n",
        "        print(\"=\" * 80)\n",
        "        log_path = LOG_DIR / f\"{tag}.log\"\n",
        "        mem_path = MEM_DIR / f\"{tag}.json\"\n",
        "        buf = io.StringIO()\n",
        "        status, err = \"ok\", \"\"\n",
        "        order = seasonal_order = ic_val = elapsed = None\n",
        "        mem_base = mem_peak = mem_delta = None\n",
        "        samples = []\n",
        "        try:\n",
        "            with MemSampler(0.05) as ms, \\\n",
        "                 contextlib.redirect_stdout(buf), \\\n",
        "                 contextlib.redirect_stderr(buf):\n",
        "                t0 = time.perf_counter()\n",
        "                res = runner(y, s, stepwise, criterion, trend)\n",
        "                elapsed = time.perf_counter() - t0\n",
        "            mem_base, mem_peak = ms.baseline, ms.peak\n",
        "            mem_delta = mem_peak - mem_base\n",
        "            samples = ms.samples\n",
        "            if lib_name == \"rustima\":\n",
        "                order = tuple(res.order)\n",
        "                seasonal_order = getattr(res, \"seasonal_order\", None)\n",
        "                ic_val = getattr(res.result, criterion, None)\n",
        "            else:\n",
        "                order = tuple(res.order)\n",
        "                seasonal_order = tuple(res.seasonal_order)\n",
        "                getter = getattr(res, criterion, None)\n",
        "                ic_val = getter() if callable(getter) else None\n",
        "        except Exception as e:\n",
        "            status = \"fail\"\n",
        "            err = f\"{type(e).__name__}: {e}\"\n",
        "            traceback.print_exc(file=buf)\n",
        "        finally:\n",
        "            log_path.write_text(buf.getvalue(), encoding=\"utf-8\")\n",
        "            with open(mem_path, \"w\", encoding=\"utf-8\") as f:\n",
        "                json.dump({\n",
        "                    \"tag\": tag,\n",
        "                    \"baseline_mb\": mem_base,\n",
        "                    \"peak_mb\": mem_peak,\n",
        "                    \"delta_mb\": mem_delta,\n",
        "                    \"interval_s\": 0.05,\n",
        "                    \"samples\": samples,\n",
        "                }, f)\n",
        "            row = {\n",
        "                \"idx\": idx, \"library\": lib_name,\n",
        "                \"seasonal\": seasonal, \"s\": s,\n",
        "                \"stepwise\": stepwise, \"criterion\": criterion, \"trend\": trend,\n",
        "                \"status\": status,\n",
        "                \"order\": str(order) if order else \"\",\n",
        "                \"seasonal_order\": str(seasonal_order) if seasonal_order else \"\",\n",
        "                \"ic_value\": f\"{ic_val:.4f}\" if isinstance(ic_val, (int, float)) else \"\",\n",
        "                \"time_s\": f\"{elapsed:.4f}\" if elapsed is not None else \"\",\n",
        "                \"mem_baseline_mb\": f\"{mem_base:.2f}\" if mem_base is not None else \"\",\n",
        "                \"mem_peak_mb\": f\"{mem_peak:.2f}\" if mem_peak is not None else \"\",\n",
        "                \"mem_delta_mb\": f\"{mem_delta:.2f}\" if mem_delta is not None else \"\",\n",
        "                \"error\": err,\n",
        "                \"log_file\": str(log_path.relative_to(OUT_DIR)),\n",
        "                \"mem_file\": str(mem_path.relative_to(OUT_DIR)),\n",
        "            }\n",
        "            with open(csv_path, \"a\", newline=\"\", encoding=\"utf-8\") as f:\n",
        "                csv.DictWriter(f, fieldnames=FIELDS).writerow(row)\n",
        "            print(f\"-> [{status}] order={order} ic={ic_val} \"\n",
        "                  f\"time={elapsed}s peak={mem_peak}MB (Δ{mem_delta}MB)\")\n",
        "            if err:\n",
        "                print(f\"   err: {err}\")\n",
        "            del buf\n",
        "            gc.collect()\n",
        "            time.sleep(0.3)\n",
        "\n",
        "sweep_elapsed = time.perf_counter() - sweep_t0\n",
        "print(f\"\\n✅ 완료 — 총 {run_i}런, {sweep_elapsed/60:.1f}분 소요\")\n",
        "print(f\"   CSV : {csv_path}\")\n",
        "print(f\"   로그 : {LOG_DIR}\")\n",
        "print(f\"   메모리 타임라인: {MEM_DIR}\")"
    ],
)

sweep_summary = code_cell(
    "sweep-summary",
    [
        "# 스위프 결과 요약 — 가장 최근 sweep_*/sweep_results.csv 로드\n",
        "import pandas as pd\n",
        "from pathlib import Path\n",
        "\n",
        "csvs = sorted(Path(\"bench_results\").glob(\"sweep_*/sweep_results.csv\"))\n",
        "assert csvs, \"bench_results/sweep_*/sweep_results.csv 가 없습니다. 스위프를 먼저 실행하세요.\"\n",
        "latest = csvs[-1]\n",
        "df = pd.read_csv(latest)\n",
        "print(f\"Loaded: {latest}\")\n",
        "print(f\"shape : {df.shape}  (성공 {(df.status=='ok').sum()} / 실패 {(df.status=='fail').sum()})\")\n",
        "\n",
        "ok = df[df.status == \"ok\"].copy()\n",
        "ok[\"time_s\"] = ok[\"time_s\"].astype(float)\n",
        "ok[\"mem_peak_mb\"] = ok[\"mem_peak_mb\"].astype(float)\n",
        "ok[\"mem_delta_mb\"] = ok[\"mem_delta_mb\"].astype(float)\n",
        "\n",
        "key_cols = [\"seasonal\", \"s\", \"stepwise\", \"criterion\", \"trend\"]\n",
        "\n",
        "pivot_time = (ok.pivot_table(index=key_cols, columns=\"library\",\n",
        "                             values=\"time_s\", aggfunc=\"first\")\n",
        "                .reset_index())\n",
        "if {\"rustima\", \"pmdarima\"}.issubset(pivot_time.columns):\n",
        "    pivot_time[\"speedup_rs_over_pm\"] = pivot_time[\"pmdarima\"] / pivot_time[\"rustima\"]\n",
        "\n",
        "pivot_peak = (ok.pivot_table(index=key_cols, columns=\"library\",\n",
        "                             values=\"mem_peak_mb\", aggfunc=\"first\")\n",
        "                .reset_index())\n",
        "if {\"rustima\", \"pmdarima\"}.issubset(pivot_peak.columns):\n",
        "    pivot_peak[\"pm_over_rs_peak\"] = pivot_peak[\"pmdarima\"] / pivot_peak[\"rustima\"]\n",
        "\n",
        "print(\"\\n=== 실행 시간 비교 (초) ===\")\n",
        "print(pivot_time.to_string(index=False))\n",
        "print(\"\\n=== 피크 메모리 비교 (MB) ===\")\n",
        "print(pivot_peak.to_string(index=False))\n",
        "\n",
        "out_dir = latest.parent\n",
        "pivot_time.to_csv(out_dir / \"pivot_time.csv\", index=False)\n",
        "pivot_peak.to_csv(out_dir / \"pivot_peak_mem.csv\", index=False)\n",
        "print(f\"\\n피벗 CSV 저장: {out_dir}\")"
    ],
)

new_cells = [setup, rustima_cell, pmdarima_cell, compare, sweep_header, sweep_run, sweep_summary]

# ---- 삽입 -----------------------------------------------------------
cells = nb["cells"]
idx = next((i for i, c in enumerate(cells) if c.get("id") == ANCHOR_ID), None)
if idx is None:
    print(f"ERROR: anchor cell id '{ANCHOR_ID}' not found", file=sys.stderr)
    sys.exit(1)

existing_ids = {c.get("id") for c in cells}
dups = [nc["id"] for nc in new_cells if nc["id"] in existing_ids]
if dups:
    print(f"ERROR: cell ids already exist: {dups}; aborting to avoid duplicates.", file=sys.stderr)
    sys.exit(2)

cells[idx + 1 : idx + 1] = new_cells
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Inserted {len(new_cells)} cells after cell id='{ANCHOR_ID}' (position {idx}).")
