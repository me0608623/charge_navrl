"""Task 3 pytest suite — corridor_crossing_fraction knob in e2e_final20_v1.

Brief 要求優先使用完整套件路徑匯入。實際執行時 isaaclab_tasks.__init__ 會觸發
pxr (Isaac Sim) 而失敗，因此改用 sys.path 插入 + importlib 直接以套件方式
載入 phases 目錄，讓相對 import 正確解析（sim-free 保證）。
所有 brief 指定的 assert 完全保留不變。

Import strategy:
- sys.path に curriculum/ を追加し、`phases` を sub-package として import できるように
- `phases.__init__` の実行は conftest で noop にしているので pxr は不要
"""
from __future__ import annotations

import importlib
import pathlib
import sys

# ── sim-free path setup ─────────────────────────────────────────────────────
# Add the `curriculum/` directory to sys.path so that `phases` is importable
# as a package without going through the full isaaclab_tasks hierarchy.
_CURRICULUM_DIR = str(pathlib.Path(__file__).resolve().parents[2])
if _CURRICULUM_DIR not in sys.path:
    sys.path.insert(0, _CURRICULUM_DIR)

# Import as a package so relative imports inside phase files resolve correctly.
# conftest's noop Package.setup ensures __init__.py chains don't pull in Isaac Sim.
m = importlib.import_module("phases.e2e_final20_v1")


# ── helpers ──────────────────────────────────────────────────────────────────


def _stage(name: str) -> dict:
    return next(s for s in m.CONFIG["stages"] if s["name"] == name)


# ── tests (verbatim from brief) ───────────────────────────────────────────────


def test_corridor_fraction_on_for_sa3_sa5():
    for name in ("SA3_walls_crossing", "SA4_spatial_plan", "SA5_endurance"):
        assert _stage(name)["corridor_crossing_fraction"] == 0.12


def test_corridor_fraction_off_elsewhere():
    for name in ("SA1_nav_bootstrap", "SA2_nav_static", "SA6_dense_avoid",
                 "SA7_high_pressure", "SA8_final"):
        assert _stage(name)["corridor_crossing_fraction"] == 0.0
