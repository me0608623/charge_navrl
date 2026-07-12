"""warp_drive_single_agent_v3e_deploy_dense — 高密度部署重訓 curriculum (2026-07-12).

★ 目的：建立「高密度障礙導航部署重訓」pipeline 的 curriculum 骨架。
  在 v3e_vdec2 基礎上，以非線性密度 ramp 把靜態從 2 拉到 15（SA8）、
  動態從 0 拉到 6-8（SA8 min6/max8），並引入 arena_shape DR 骨架
  （SA1-3 正方形、SA4+ 正方/走廊混合）。

  ┌─ 設計重點 ─────────────────────────────────────────────────────────┐
  │ 靜態 ramp: 2, 4, 6, 8, 10, 12, 14, 15（SA1→SA8）                 │
  │ 動態 ramp: 0, 0, 1, 2,  3,  4,  6,  6-8（SA8 min6/max8）         │
  │ SA8 總障礙 15+8=23 ≤ cap 24（max_active_obstacles 需設 24）        │
  │ arena_shape: SA1-3=square、SA4+=mix（走廊/正方 per-env DR）         │
  │   ⚠ arena_shape 是未來 arena 生成邏輯的佔位鍵，本檔只設定，       │
  │     消費端（boundary 生成）由後續 Task 1.2 實作。                  │
  └────────────────────────────────────────────────────────────────────┘

⚠️ 搭配 config 需要 max_active_obstacles ≥ 24，否則 SA8 動態會被靜默截斷。
⚠️ obs 佈局同 vdec2（109D LV-DOT），需 CHARGE_USE_ACT_HIST=0 CHARGE_USE_LVDOT_OBS=1。
"""

from __future__ import annotations

import copy

from .wd_single_agent_v3 import GLOBAL, STAGES as _V3_STAGES, _flatten_phase
from .wd_single_agent_v3e_vdec2 import STAGES as _VDEC2_STAGES

# ── 密度 ramp 表（所有 stage 完整列出）──────────────────────────────────────
# 靜態障礙：非線性 ramp，SA8 達到真密集目標
_STATIC_RAMP = {
    "SA1_nav_bootstrap":   2,
    "SA2_nav_static":      4,
    "SA3_walls_crossing":  6,
    "SA4_spatial_plan":    8,
    "SA5_endurance":      10,
    "SA6_dense_avoid":    12,
    "SA7_high_pressure":  14,
    "SA8_final":          15,
}

# 動態障礙：SA1-2 維持 0（純靜態學習），之後線性遞增
_DYNAMIC_RAMP = {
    "SA1_nav_bootstrap":   0,
    "SA2_nav_static":      0,
    "SA3_walls_crossing":  1,
    "SA4_spatial_plan":    2,
    "SA5_endurance":       3,
    "SA6_dense_avoid":     4,
    "SA7_high_pressure":   6,
    "SA8_final":           8,
}

# 動態障礙下限（只覆寫非 min==max 的 stage）
# 其餘 stage: dynamic_obstacles_min = dynamic_obstacles（固定值）
_DYNAMIC_MIN = {
    "SA8_final": 6,  # SA8: min6 max8（允許少量浮動）
}

# arena 形狀 DR 骨架（SA4+ 引入 mix = 正方/走廊 per-env 隨機）
# ⚠ 值 "square"/"mix" 是佔位字串，boundary 生成邏輯由 Task 1.2 實作
_ARENA_SHAPE = {
    "SA1_nav_bootstrap":  "square",
    "SA2_nav_static":     "square",
    "SA3_walls_crossing": "square",
    "SA4_spatial_plan":   "mix",
    "SA5_endurance":      "mix",
    "SA6_dense_avoid":    "mix",
    "SA7_high_pressure":  "mix",
    "SA8_final":          "mix",
}


def _build_deploy_dense_stages() -> list[dict]:
    """以 vdec2 STAGES 為基底，套入 deploy_dense 密度 ramp + arena DR。

    vdec2 已含：SA2 重設計、head_on 平滑、SA7/SA8 速度還原、obs_near_goal 統一、
    靜態 ramp（2-13）等 5 項改良。本函數在其基礎上僅替換密度設定與
    加入 arena_shape，不改其他 behavior 設計。
    """
    stages = [copy.deepcopy(s) for s in _VDEC2_STAGES]
    by_name = {s["name"]: s for s in stages}

    for name in _STATIC_RAMP:
        sc = by_name[name]["scene"]

        # 靜態障礙：直接用 deploy_dense 的 ramp（覆寫 vdec2 的設定）
        sc["static_obstacles"] = _STATIC_RAMP[name]

        # 動態障礙：用 deploy_dense ramp 覆寫
        sc["dynamic_obstacles"] = _DYNAMIC_RAMP[name]

        # 動態下限：有明確指定則用，否則等於上限（固定值）
        sc["dynamic_obstacles_min"] = _DYNAMIC_MIN.get(name, _DYNAMIC_RAMP[name])

        # arena_shape：佔位鍵（boundary 生成由 Task 1.2 消費）
        sc["arena_shape"] = _ARENA_SHAPE[name]

    return stages


# deploy_dense stage 列表（巢狀 schema，供人類閱讀/再調整）
STAGES = _build_deploy_dense_stages()

# 供 phases registry 匯出的標準格式（與 vdec2 / v3 結構完全一致）
CONFIG = {
    "obstacle_mode": GLOBAL["obstacle_mode"],
    "upgrade_pass_required": GLOBAL["upgrade_pass_required"],
    "clear_window_on_promote": GLOBAL["clear_window_on_promote"],
    "trainer_sync_allowlist": GLOBAL["trainer_sync_allowlist"],
    "stages": [_flatten_phase(s) for s in STAGES],
}
