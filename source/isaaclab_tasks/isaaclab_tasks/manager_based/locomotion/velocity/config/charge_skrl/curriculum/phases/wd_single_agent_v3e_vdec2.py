"""warp_drive_single_agent_v3e_vdec2 — vdec + 課程平滑度 5 項改良 (2026-07-11).

★ 目的：在保留 v3e_vdec「速度決定性」設計的前提下，修掉用戶點出的 5 個課程
  設計問題（SA1≈SA2 浪費、SA2→SA3 跳太陡、head_on 斷層、SA7/8 速度被壓低、
  obs_near_goal 不一致）。base v3 / v3e / v3h / v3f 完全不動，零風險隔離。

★ 隔離原則：本檔「站在 vdec STAGES 之上」只加明確 delta（DRY + 好稽核）：
    vdec2 = vdec (已含 v3e smoothness=0 + ent_ang floor 0.02 + head_on0.25 SA3-5
                  + dynamic+1 + obstacle_speed→0.9 + speed_range×1.10 cap0.8)
            + 以下 5 項改良

  ┌─ 改良清單（對應用戶問題編號）────────────────────────────────────────┐
  │ ① SA2 重設計（★★★★★，問題1+2）：SA1≈SA2 → 讓 SA2 成為「SA1 加強版 +      │
  │    SA3 前味」。static 1→2、dynamic 2(min1)→3(min2)、goal_dist (2,6)→(2,7)、 │
  │    behavior 加 head_on 0.10（從 patrol 扣）。同時平滑 SA2→SA3 跳躍。       │
  │ ② head_on 平滑（★★★★☆，問題3）：SA6 保留 head_on 0.10（原本 SA6 突然移除）  │
  │    + near_miss 提前到 SA5 出現 0.05（原本 SA6 才首見）。雙向平滑行為斷層。   │
  │ ③ 恢復 SA7/SA8 速度（★★★★☆，問題4）：vdec 把 obstacle_speed 統一壓成 0.9、  │
  │    又用 ×1.10 cap0.8 壓扁高速行為 → 後期挑戰性下降。整個 behavior block 從   │
  │    base v3 還原（obstacle_speed 1.00/1.15 + 原始 speed_range），一次撤兩層。 │
  │ ④ SA2→SA3 過渡（★★★☆☆，問題2）：見①的 head_on 0.10 + static 1→2。          │
  │ ⑤ obs_near_goal 統一（★★☆☆☆，問題5）：radius 全部 2.0m（SA3/4 原 2.5）、     │
  │    count 從 SA3 起一律 2（SA5 原本 1 → 2，補回單調性）。用意：把障礙強制生在  │
  │    goal 周圍環形區 [min_dist, radius]，逼 policy 處理目標旁雜物、不能直衝。   │
  │ ⑥ 密度 ramp（2026-07-11，密集障礙導航）：base v3 靜態 SA3-8 全卡 3 = 密度缺陷。 │
  │    靜態拉成單調 ramp 2→13、動態保留 vdec，後期到真密集(SA8 22顆≈11/100㎡)。    │
  │    面積校準 196㎡(±7 非外牆400㎡)。⚠️需 config max_active_obstacles≥22(建議24)。│
  └────────────────────────────────────────────────────────────────────────┘

⚠️ 這是「未來跑完整課程（SA1→SA8）」用的 curriculum_version，需搭配 initial_stage=1
   + fixed_stage=false 的 config。對現行 fixed_stage=SA3 的 ttc02 run 無影響。
⚠️ obs 佈局同 vdec（109D LV-DOT），仍需 CHARGE_USE_ACT_HIST=0 CHARGE_USE_LVDOT_OBS=1。
"""

from __future__ import annotations

import copy

from .wd_single_agent_v3 import GLOBAL, STAGES as _V3_STAGES, _flatten_phase
from .wd_single_agent_v3e_vdec import STAGES as _VDEC_STAGES

# base v3 原始 behavior block（供 ③ 還原 SA7/SA8 速度用；index = SA-1）
_V3_BY_NAME = {s["name"]: s for s in _V3_STAGES}


def _build_vdec2_stages() -> list[dict]:
    """deep-copy vdec STAGES 後套 5 項改良（按 name 定位，明確逐 stage 改）。"""
    stages = [copy.deepcopy(s) for s in _VDEC_STAGES]
    by_name = {s["name"]: s for s in stages}

    # ── ① + ④  SA2 重設計：SA1 加強版 + SA3 前味（含 head_on 0.10 平滑跳躍）──
    #    註：static 由 ⑥ 密度 ramp 統一設定，這裡只管 dynamic / dist / head_on。
    sa2 = by_name["SA2_nav_static"]
    sa2["scene"]["dynamic_obstacles"] = 3         # 2→3（比 SA1 的 2 多）
    sa2["scene"]["dynamic_obstacles_min"] = 2     # 1→2
    sa2["scene"]["goal_distance"] = (2.0, 7.0)    # (2,6)→(2,7) 稍拉長
    sa2["behavior"]["behavior_mix"] = {           # 引入少量 head_on
        "patrol": 0.50,                           # 0.60→0.50（讓出 0.10 給 head_on）
        "random_walk": 0.40,
        "head_on": 0.10,                          # 新增：SA3 前的迎面前味
    }
    # SA2 head_on 速度較 SA3（0.33~0.605）溫和，直接指定（不再過 vdec ×1.10）
    sa2["behavior"].setdefault("speed_overrides", {})["head_on"] = {"speed_range": (0.25, 0.45)}

    # ── ② head_on 平滑：SA5 提前引入 near_miss、SA6 保留 head_on ──
    sa5 = by_name["SA5_endurance"]
    sa5_mix = sa5["behavior"]["behavior_mix"]
    sa5_mix["corridor_crossing"] = round(sa5_mix.get("corridor_crossing", 0.20) - 0.05, 6)  # 0.20→0.15
    sa5_mix["near_miss"] = 0.05                   # 新增：SA6 才首見 → 提前到 SA5 前味
    sa5["behavior"].setdefault("speed_overrides", {})["near_miss"] = {"speed_range": (0.30, 0.55)}

    sa6 = by_name["SA6_dense_avoid"]
    sa6_mix = sa6["behavior"]["behavior_mix"]
    sa6_mix["patrol"] = round(sa6_mix.get("patrol", 0.15) - 0.05, 6)          # 0.15→0.10
    sa6_mix["random_walk"] = round(sa6_mix.get("random_walk", 0.15) - 0.05, 6)  # 0.15→0.10
    sa6_mix["head_on"] = 0.10                     # 保留 head_on（原本 SA6 突然移除 → 斷層）
    sa6["behavior"].setdefault("speed_overrides", {})["head_on"] = {"speed_range": (0.35, 0.70)}

    # ── ③ 恢復 SA7/SA8 obstacle speed（整個 behavior block 從 base v3 還原）──
    #    撤銷 vdec 的 obstacle_speed→0.9 與 speed_range ×1.10 cap0.8 雙層壓縮。
    for name in ("SA7_high_pressure", "SA8_final"):
        by_name[name]["behavior"] = copy.deepcopy(_V3_BY_NAME[name]["behavior"])

    # ── ⑤ obs_near_goal 統一：radius 一律 2.0、count 從 SA3 起一律 2 ──
    for name in ("SA3_walls_crossing", "SA4_spatial_plan", "SA5_endurance",
                 "SA6_dense_avoid", "SA7_high_pressure", "SA8_final"):
        sc = by_name[name]["scene"]
        if sc.get("obs_near_goal_count", 0) > 0:
            sc["obs_near_goal_count"] = 2         # SA5 原 1 → 2（補回單調性）
            sc["obs_near_goal_radius"] = 2.0      # SA3/4 原 2.5 → 2.0（統一語義）

    # ── ⑥ 密度 ramp（2026-07-11，用戶要求「密集障礙導航」）──
    #    ★ 面積校準：障礙 spawn/移動範圍 = scene_bound_base ±7 = 14×14 ≈ 196㎡
    #      （非外牆 20×20=400㎡；robot/goal 也都在內圈）。密度以 196㎡ 計。
    #    真正缺陷：原 base v3 靜態 SA3-8 全卡 3 不 ramp → 密度上不去。這裡把靜態
    #    拉成單調 ramp，動態保留 vdec 設計，後期到「真密集」(SA8 ~11/100㎡, 近 dense 目標)。
    #    ⚠️ 需搭配 config max_active_obstacles ≥ 最大總數(SA8=22) → 建議設 24，
    #       否則 train_rnn_car_wdclip.py:2689 N_obs 硬 cap 會靜默截斷。prim 上限 MAX_OBSTACLES=50 足夠。
    #    ⚠️ 動態 >5 時 LV-DOT channel(K=5) 只回報最近 5 顆(其餘仍在 LiDAR)；TTC 稅作用於最近 5。
    #    格式: stage_name -> static_obstacles（動態沿用上面各 patch 後的值）
    _STATIC_RAMP = {
        "SA1_nav_bootstrap": 2,   # 1→2
        "SA2_nav_static":    3,   # (①原設2) → 3
        "SA3_walls_crossing": 5,  # 3→5
        "SA4_spatial_plan":  6,   # 3→6
        "SA5_endurance":     8,   # 3→8
        "SA6_dense_avoid":   9,   # 3→9
        "SA7_high_pressure": 11,  # 3→11
        "SA8_final":         13,  # 3→13
    }
    for name, n_static in _STATIC_RAMP.items():
        by_name[name]["scene"]["static_obstacles"] = n_static

    return stages


# v3e_vdec2 的 stage 列表（巢狀 schema，供人類閱讀/再調整）
STAGES = _build_vdec2_stages()


# 供 phases registry 匯出的標準格式（與 v3 結構一致）
CONFIG = {
    "obstacle_mode": GLOBAL["obstacle_mode"],
    "upgrade_pass_required": GLOBAL["upgrade_pass_required"],
    "clear_window_on_promote": GLOBAL["clear_window_on_promote"],
    "trainer_sync_allowlist": GLOBAL["trainer_sync_allowlist"],
    "stages": [_flatten_phase(stage) for stage in STAGES],
}
