"""Insert a cleanup cell (sweep-clean) directly above sweep-run."""
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


SWEEP_CLEAN = r'''
# ======================================================================
# sweep-clean — sweep-run 실행 전 기존 산출물 정리
#
# MODE 를 바꿔서 원하는 수준의 정리를 실행하세요.
#   "none"  → 아무것도 안 함 (현재 상태만 출력)
#   "pkl"   → sweep_*/y.pkl 만 삭제 (sweep-run 이 y 를 새로 pickle)
#   "dir"   → sweep_* 디렉터리 전체 삭제 (완전 초기화; RESUME 무효)
#   "latest"→ 가장 최근 sweep_* 디렉터리만 통째로 삭제
#
# 실수 방지를 위해 DRY_RUN=True 이면 삭제 대상만 출력하고 실행 안 함.
# ======================================================================
import shutil
from pathlib import Path

MODE = "pkl"       # "none" | "pkl" | "dir" | "latest"
DRY_RUN = False    # True 이면 삭제 대상만 표시

base = Path("bench_results")
if not base.exists():
    print("bench_results 디렉터리 없음 — 정리할 것 없음")
else:
    sweep_dirs = sorted(base.glob("sweep_*"))
    print(f"현재 sweep 디렉터리: {len(sweep_dirs)}개")
    for d in sweep_dirs:
        csv_p = d / "sweep_results.csv"
        pkl_p = d / "y.pkl"
        n_rows = 0
        if csv_p.exists():
            try:
                n_rows = sum(1 for _ in open(csv_p, encoding="utf-8")) - 1
            except Exception:
                pass
        print(f"  {d.name}  rows={n_rows:3d}  y.pkl={'O' if pkl_p.exists() else '-'}")

    def _rm_file(p):
        print(f"  rm  {p}")
        if not DRY_RUN:
            try: p.unlink()
            except OSError as e: print(f"     (failed: {e})")

    def _rm_tree(p):
        print(f"  rmtree  {p}")
        if not DRY_RUN:
            try: shutil.rmtree(p)
            except OSError as e: print(f"     (failed: {e})")

    if MODE == "none":
        print("\nMODE=none — 건너뜀")
    elif MODE == "pkl":
        targets = [p for d in sweep_dirs for p in d.glob("*.pkl")]
        print(f"\n[MODE=pkl] {len(targets)}개 pkl 삭제 (DRY_RUN={DRY_RUN})")
        for p in targets:
            _rm_file(p)
    elif MODE == "dir":
        print(f"\n[MODE=dir] {len(sweep_dirs)}개 sweep 디렉터리 전체 삭제 (DRY_RUN={DRY_RUN})")
        for d in sweep_dirs:
            _rm_tree(d)
    elif MODE == "latest":
        if sweep_dirs:
            print(f"\n[MODE=latest] 가장 최근 1개 삭제 (DRY_RUN={DRY_RUN})")
            _rm_tree(sweep_dirs[-1])
        else:
            print("\n[MODE=latest] 대상 없음")
    else:
        raise ValueError(f"알 수 없는 MODE: {MODE!r}")

    if not DRY_RUN and MODE != "none":
        print(f"\n✅ 정리 완료. 남은 sweep 디렉터리: "
              f"{len(list(base.glob('sweep_*')))}개")
'''

idx_run = find_cell("sweep-run")
assert idx_run is not None, "sweep-run cell not found"

idx_clean = find_cell("sweep-clean")
clean_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "sweep-clean",
    "metadata": {},
    "outputs": [],
    "source": to_lines(SWEEP_CLEAN),
}
if idx_clean is not None:
    cells[idx_clean] = clean_cell
    print(f"Replaced sweep-clean (position {idx_clean})")
else:
    cells.insert(idx_run, clean_cell)
    print(f"Inserted sweep-clean before sweep-run (position {idx_run})")

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook saved.")
