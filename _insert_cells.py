import json
import sys
from pathlib import Path

nb_path = Path(r"c:/Rust-python-arima/rustima/prac_readme_jihun.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

TARGET_ID = "04087484"

# Build new cells
def code_cell(cell_id, source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source,
    }

def md_cell(cell_id, source):
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source,
    }

header = md_cell(
    "mem-bench-header",
    [
        "### btop / psutil 로 rustima vs pmdarima 메모리 비교\n",
        "\n",
        "아래 셀들을 **하나씩 개별 실행**하며, 별도 PowerShell 창에서 `btop`을 띄워 python 프로세스를 관찰하세요.\n",
        "- 같은 셀에서 두 라이브러리를 연달아 돌리면 피크 메모리가 섞여 비교가 오염됩니다.\n",
        "- `psutil`로 RSS 델타를 찍어 정량 비교도 함께 남깁니다."
    ],
)

setup = code_cell(
    "mem-bench-setup",
    [
        "# 공통 설정: 메모리 측정 유틸\n",
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
        "    time.sleep(1)  # btop에서 베이스라인을 확인할 여유 시간\n",
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
        "# rustima(sarimax_rs) 단독 실행 — btop에서 피크 메모리 관찰\n",
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
        "# 다음 셀 측정이 오염되지 않도록 참조 정리\n",
        "del auto_rs_bic\n",
        "gc.collect()"
    ],
)

pmdarima_cell = code_cell(
    "mem-bench-pmdarima",
    [
        "# pmdarima 단독 실행 — btop에서 피크 메모리 관찰\n",
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

compare_cell = code_cell(
    "mem-bench-compare",
    [
        "# 속도 요약\n",
        "print(f\"속도 비교: sarimax_rs 가 pmdarima 대비 {t_pm/t_rs:.1f}x faster\")"
    ],
)

new_cells = [header, setup, rustima_cell, pmdarima_cell, compare_cell]

# Locate target and insert
cells = nb["cells"]
idx = next((i for i, c in enumerate(cells) if c.get("id") == TARGET_ID), None)
if idx is None:
    print(f"ERROR: target cell id {TARGET_ID} not found", file=sys.stderr)
    sys.exit(1)

# Avoid duplicate insertion if script is re-run
existing_ids = {c.get("id") for c in cells}
if any(nc["id"] in existing_ids for nc in new_cells):
    print("ERROR: one of the new cell ids already exists; aborting to avoid duplicates", file=sys.stderr)
    sys.exit(2)

cells[idx + 1 : idx + 1] = new_cells
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Inserted {len(new_cells)} cells after cell id={TARGET_ID} (position {idx}).")
