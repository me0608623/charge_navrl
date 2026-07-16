"""Conftest for sim-free unit tests in events/tests/.

Isaac Sim (pxr) 沒有安裝於純 pytest 環境中。pytest 的 Package 收集器會在測試項目
的 setup 階段呼叫所有父套件目錄的 Package.setup()，而 isaaclab_tasks/__init__.py
鏈結到 pxr 因而崩潰。

解法：monkeypatch _pytest.python.Package.setup，讓父套件 __init__.py 的匯入
不去執行（只有在有 pxr 時才需要真正跑）。這樣測試本身透過 file-path loader 載入
corridor_crossing_geometry.py，完全不碰 Isaac Sim。
"""
from __future__ import annotations

import _pytest.python


def _noop_package_setup(self) -> None:  # type: ignore[no-untyped-def]
    """Skip Package.__init__.py setup to avoid Isaac Sim import in sim-free pytest."""
    pass


# Patch before collection begins.
_pytest.python.Package.setup = _noop_package_setup  # type: ignore[method-assign]
