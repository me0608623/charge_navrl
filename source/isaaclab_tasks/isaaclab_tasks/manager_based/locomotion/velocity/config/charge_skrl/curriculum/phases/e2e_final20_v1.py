"""Smooth curriculum ending at the real 12x12 m, 20-obstacle deployment load."""

from __future__ import annotations

import copy

from .wd_single_agent_v3 import GLOBAL, _flatten_phase
from .wd_single_agent_v3e_deploy_dense import (
    STAGES as _DEPLOY_STAGES,
    _deploy_behavior_mix,
)


# Total obstacle count rises monotonically: 2, 4, 7, 10, 13, 16, 18, 20.
_COUNTS = {
    "SA1_nav_bootstrap": (2, 0),
    "SA2_nav_static": (4, 0),
    "SA3_walls_crossing": (6, 1),
    "SA4_spatial_plan": (8, 2),
    "SA5_endurance": (10, 3),
    "SA6_dense_avoid": (12, 4),
    "SA7_high_pressure": (13, 5),
    "SA8_final": (14, 6),
}

# Corridor-crossing injection fraction per stage (deployment near-wall trap).
# ON for the stages that first introduce crossing pedestrians (SA3-5).
_CORRIDOR_FRACTION = {
    "SA1_nav_bootstrap": 0.0,
    "SA2_nav_static": 0.0,
    "SA3_walls_crossing": 0.12,
    "SA4_spatial_plan": 0.12,
    "SA5_endurance": 0.12,
    "SA6_dense_avoid": 0.0,
    "SA7_high_pressure": 0.0,
    "SA8_final": 0.0,
}

# ★內牆縮減(用戶 2026-07-18):SA6-8 held-out 場景過密——長內牆(v3 繼承 walls_max=3 / wall_length
#   4.5~5.0m)+ 密障礙切割空間造死路 → goal 不可解假陽性 CR + 訓練污染(用戶 seed303 實測)。
#   部署有牆但少 → walls_max 3→2 + wall_length →3.5(縮~25%,牆長是主阻擋)。SA7/8 未訓 + SA6 未
#   frozen → 不用重訓已畢業 SA1-5。見 docs/gate2_held_out_unsolvable_scene_issues.md 根因③。
_WALL_TRIM = {
    "SA6_dense_avoid":   {"walls_max": 2, "wall_length": 3.5},
    "SA7_high_pressure": {"walls_max": 2, "wall_length": 3.5},
    "SA8_final":         {"walls_max": 2, "wall_length": 3.5},
}


def _build_stages() -> list[dict]:
    stages = [copy.deepcopy(stage) for stage in _DEPLOY_STAGES]
    for stage in stages:
        name = stage["name"]
        n_static, n_dynamic = _COUNTS[name]
        scene = stage["scene"]
        scene["static_obstacles"] = n_static
        scene["dynamic_obstacles"] = n_dynamic
        scene["dynamic_obstacles_min"] = n_dynamic
        scene["corridor_crossing_fraction"] = _CORRIDOR_FRACTION[name]
        stage.setdefault("behavior", {})["behavior_mix"] = _deploy_behavior_mix(
            n_static, n_dynamic
        )
        # ★升級 SR 門檻全階段 ≥0.75（用戶 2026-07-15）：每個密度真正精熟（75% SR）才晉級，
        # 不再隨難度降門檻（舊 v3 表 0.72→0.45 會把未精熟政策往更難場景推）。
        # 用 max() 只抬升低於 0.75 的階段；SA8_final 的 1.0（終點不可升階）保持不變。
        transition = stage.setdefault("transition", {})
        transition["upgrade_sr"] = max(0.75, float(transition.get("upgrade_sr", 0.75)))
        # ★內牆縮減(SA6-8,見上方 _WALL_TRIM):降 walls_max + 縮 wall_length,解 held-out 死路
        if name in _WALL_TRIM:
            scene.update(_WALL_TRIM[name])
    return stages


STAGES = _build_stages()

CONFIG = {
    "obstacle_mode": GLOBAL["obstacle_mode"],
    "upgrade_pass_required": GLOBAL["upgrade_pass_required"],
    "clear_window_on_promote": GLOBAL["clear_window_on_promote"],
    "trainer_sync_allowlist": GLOBAL["trainer_sync_allowlist"],
    "stages": [_flatten_phase(stage) for stage in STAGES],
}
