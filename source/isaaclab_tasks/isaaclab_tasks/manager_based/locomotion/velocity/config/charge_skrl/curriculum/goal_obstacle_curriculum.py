"""Goal-Obstacle 聯動課程學習 — 資料驅動多版本設計

支援多個 curriculum version，透過 CLI --curriculum_version 切換：
- baseline_v1: 原始 v12 線性 8 階段（每階 +1S+1D, -1G）
- goal_first_v1: 先導航再避障（Stage 1-4 無 dynamic）

獎勵函數在所有階段、所有版本完全不變。課程只改變環境參數。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ============================================================================
# Ablation 參數（由 CLI 設定）
# ============================================================================

_ablation_ss_scale = 1.0  # safety 權重縮放因子 (1.0=v8 baseline, 0.6=lower, 1.4=raise)


def set_ablation_params(ss_lower_mode: str | None = None, ss_raise: bool = False):
    """設定 Ablation Family 5 參數。

    Args:
        ss_lower_mode: "aggressive" (ss*0.6), "moderate" (ss*0.8), None (不調整)
        ss_raise: True (ss*1.4), False (不調整)
    """
    global _ablation_ss_scale
    if ss_lower_mode == "aggressive":
        _ablation_ss_scale = 0.6
    elif ss_lower_mode == "moderate":
        _ablation_ss_scale = 0.8
    elif ss_raise:
        _ablation_ss_scale = 1.4
    else:
        _ablation_ss_scale = 1.0


def _apply_ss_scale(value: float) -> float:
    """套用 safety 權重縮放因子，保留 2 位小數。"""
    return round(value * _ablation_ss_scale, 2)


# ============================================================================
# Curriculum Version Configs — 資料驅動
# ============================================================================

def _make_stage(
    num_goals, n_static, n_dynamic, min_walls, max_walls,
    gamma, episode_s, empty_ratio,
    upgrade_sr, upgrade_max_cr=0.40, upgrade_max_to=0.30,
    upgrade_min_dyn_sr=0.0, min_stage_updates=50,
    downgrade_sr=0.15, downgrade_min_cr=0.70, downgrade_min_to=0.65,
    name="",
    reward_weights=None,
    obs_size_rand=0.0,
    scene_bound_rand=0.0,
    spot_penalty_hit=-5.0,
    spot_reward_get_goal=40.0,
    spot_cost_operate=0.0,
    ent_coeff_linear=0.30,
    ent_coeff_angular=0.375,
    obstacle_speed_rate=0.8,
):
    """建立單一 stage config dict。

    reward_weights: dict of {reward_term_name: weight} 用於 stage-dependent 獎勵權重。
                   None = 不修改權重（使用 config 預設值）。

    Warp Drive reward params (per-phase):
        spot_penalty_hit: 碰撞懲罰（Phase 1: -5 → Phase 8+: -200）
        spot_reward_get_goal: 到達目標獎勵（固定 40）
        spot_cost_operate: 動作成本（Phase 1: 0.03，Phase 2+: 0）

    Warp Drive entropy params (per-phase, A2CK style):
        ent_coeff_linear:  head1 entropy coeff (WD: spot_entropy_coeff × 2.5)
        ent_coeff_angular: head2 entropy coeff (WD: spot_action2_entropy_coeff × 2.5)
        Phase 1: 0.10 / 0.375 (低 linear entropy → 先學基本走)
        Phase 2+: 0.30 / 0.375 (高 linear entropy → 鼓勵探索)

    Warp Drive obstacle speed (per-phase):
        obstacle_speed_rate: 障礙物最大速度（relative to charge）
        Phase 1: 0.8, Phase 2-6: 0.85, Phase 7-8: 1.15
    """
    if n_static + n_dynamic > 0:
        remaining = round(1.0 - empty_ratio, 2)
        total_obs = n_static + n_dynamic
        s_ratio = round(remaining * n_static / total_obs, 2) if total_obs > 0 else 0.0
        d_ratio = round(remaining - s_ratio, 2)
    else:
        s_ratio = 0.0
        d_ratio = 0.0

    stage = {
        "name": name,
        "num_goals": num_goals,
        "goal_distance": (2.0, 13.0),
        "num_obstacles_static": n_static,
        "num_obstacles_dynamic": n_dynamic,
        "empty_ratio": empty_ratio,
        "static_ratio": s_ratio,
        "dynamic_ratio": d_ratio,
        "gamma": gamma,
        "episode_length_s": float(episode_s),
        "min_walls": min_walls,
        "max_walls": max_walls,
        "upgrade_sr": upgrade_sr,
        "upgrade_max_cr": upgrade_max_cr,
        "upgrade_max_to": upgrade_max_to,
        "upgrade_min_dyn_sr": upgrade_min_dyn_sr,
        "min_stage_updates": min_stage_updates,
        "downgrade_sr": downgrade_sr,
        "downgrade_min_cr": downgrade_min_cr,
        "downgrade_min_to": downgrade_min_to,
    }
    # WD reward params — always include (training loop reads these)
    stage["spot_penalty_hit"] = spot_penalty_hit
    stage["spot_reward_get_goal"] = spot_reward_get_goal
    stage["spot_cost_operate"] = spot_cost_operate
    # WD entropy params (A2CK per-head)
    stage["ent_coeff_linear"] = ent_coeff_linear
    stage["ent_coeff_angular"] = ent_coeff_angular
    # WD obstacle speed per-phase
    stage["obstacle_speed_rate"] = obstacle_speed_rate

    if reward_weights is not None:
        stage["reward_weights"] = reward_weights
    if obs_size_rand > 0.0:
        stage["obs_size_rand"] = obs_size_rand
    if scene_bound_rand > 0.0:
        stage["scene_bound_rand"] = scene_bound_rand
    return stage


CURRICULUM_CONFIGS = {
    # ==================================================================
    # baseline_v1: 精確重現原 v12 線性 8 階段
    # 每階 -1G, +1S, +1D。Stage 2 就同時有 static+dynamic。
    # ==================================================================
    "baseline_v1": {
        "upgrade_pass_required": 5,
        "clear_window_on_promote": True,
        "stages": [
            _make_stage(8, 0, 0, 0, 2, 0.990, 45, 1.00,
                        upgrade_sr=0.72, upgrade_max_cr=1.0, upgrade_max_to=0.30,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=50,
                        downgrade_sr=0.0, downgrade_min_cr=1.0, downgrade_min_to=1.0,
                        name="純導航"),
            _make_stage(7, 1, 1, 0, 3, 0.991, 51, 0.87,
                        upgrade_sr=0.65, min_stage_updates=65, name="1S+1D"),
            _make_stage(6, 2, 2, 1, 4, 0.993, 56, 0.74,
                        upgrade_sr=0.65, upgrade_min_dyn_sr=0.35, min_stage_updates=80, name="2S+2D"),
            _make_stage(5, 3, 3, 1, 5, 0.994, 61, 0.61,
                        upgrade_sr=0.65, upgrade_min_dyn_sr=0.35, min_stage_updates=95, name="3S+3D"),
            _make_stage(4, 4, 4, 2, 6, 0.996, 67, 0.48,
                        upgrade_sr=0.65, upgrade_min_dyn_sr=0.35, min_stage_updates=110, name="4S+4D"),
            _make_stage(3, 5, 5, 2, 7, 0.997, 72, 0.35,
                        upgrade_sr=0.65, upgrade_min_dyn_sr=0.35, min_stage_updates=125, name="5S+5D"),
            _make_stage(2, 6, 6, 3, 8, 0.998, 78, 0.22,
                        upgrade_sr=0.65, upgrade_min_dyn_sr=0.35, min_stage_updates=140, name="6S+6D"),
            _make_stage(1, 7, 7, 4, 8, 0.998, 90, 0.15,
                        upgrade_sr=1.0, upgrade_max_cr=0.0, upgrade_max_to=0.0,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=0,
                        downgrade_sr=0.15, name="終極挑戰"),
        ],
    },

    # ==================================================================
    # goal_first_v1: 先導航再避障
    #
    # | Stage | 學習重點         | Goals | Static | Dynamic | Walls | SR門檻 |
    # |-------|-----------------|-------|--------|---------|-------|--------|
    # | 1     | 純 goal-reaching | 6~8   | 0      | 0       | 0~1   | >85%   |
    # | 2     | 近距離 + 少量牆   | 5~7   | 0      | 0       | 1~2   | >85%   |
    # | 3     | 開始學繞路        | 4~6   | 1~2    | 0       | 2~3   | >80%   |
    # | 4     | 增加靜態擁擠度    | 3~5   | 2~3    | 0       | 3~4   | >80%   |
    # | 5     | 輕度動態干擾      | 3~4   | 3~4    | 1~2     | 3~4   | >75%   |
    # | 6     | 中度動態避障      | 2~3   | 4~5    | 2~3     | 4~5   | >75%   |
    # | 7     | 高擁擠 + 多障礙   | 1~2   | 5~6    | 4~5     | 5~6   | >70%   |
    # | 8     | 最終挑戰          | 1     | 6~7    | 6~7     | 6~8   | 固定    |
    #
    # 表中 "~" 表示 mixed_parallel 的隨機採樣範圍，
    # 下面 num_obstacles_static/dynamic 設為該範圍的中位數或上限。
    # ==================================================================
    # ==================================================================
    # goal_first_v1: 先導航再避障 + stage-dependent reward weights
    #
    # Stage-dependent 權重設計理念：
    # - Phase A (1-2): 無障礙物 → safety 權重低，goal 權重高
    # - Phase B (3-4): 有 static → 提高 static_safety
    # - Phase C (5-8): 有 dynamic → 提高 dynamic_safety
    #
    # | Stage | vel | prog | ss  | ds  | 理由 |
    # |-------|-----|------|-----|-----|------|
    # | 1-2   | 4.0 | 5.0  | 0.5 | 0   | 專注 goal-reaching |
    # | 3-4   | 3.0 | 4.0  | 2.0 | 0.5 | 開始學 static 避障 |
    # | 5-6   | 2.0 | 3.0  | 2.0 | 2.0 | 加入 dynamic |
    # | 7-8   | 2.0 | 3.0  | 2.0 | 2.0 | 全功能 |
    # ==================================================================
    "goal_first_v1": {
        "upgrade_pass_required": 5,
        "clear_window_on_promote": True,
        "stages": [
            # --- Phase A: Goal-reaching (Stage 1-2) ---
            _make_stage(7, 0, 0, 0, 1, 0.990, 45, 1.00,
                        upgrade_sr=0.85, upgrade_max_cr=1.0, upgrade_max_to=0.20,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=50,
                        downgrade_sr=0.0, downgrade_min_cr=1.0, downgrade_min_to=1.0,
                        name="goal_reaching_open",
                        reward_weights={"goal_velocity": 4.0, "goal_progress": 5.0,
                                        "static_safety": 0.5, "dynamic_safety": 0.0}),
            _make_stage(6, 0, 0, 1, 2, 0.990, 50, 1.00,
                        upgrade_sr=0.85, upgrade_max_cr=1.0, upgrade_max_to=0.20,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=60,
                        downgrade_sr=0.30, downgrade_min_cr=1.0, downgrade_min_to=0.80,
                        name="goal_walls",
                        reward_weights={"goal_velocity": 4.0, "goal_progress": 5.0,
                                        "static_safety": 0.5, "dynamic_safety": 0.0}),

            # --- Phase B: Static obstacles (Stage 3-4) ---
            _make_stage(5, 2, 0, 2, 3, 0.992, 55, 0.55,
                        upgrade_sr=0.80, upgrade_max_cr=0.35, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=70,
                        name="static_intro",
                        reward_weights={"goal_velocity": 3.0, "goal_progress": 4.0,
                                        "static_safety": 2.0, "dynamic_safety": 0.5}),
            _make_stage(4, 3, 0, 3, 4, 0.993, 60, 0.40,
                        upgrade_sr=0.80, upgrade_max_cr=0.35, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=85,
                        name="static_dense",
                        reward_weights={"goal_velocity": 3.0, "goal_progress": 4.0,
                                        "static_safety": 2.0, "dynamic_safety": 0.5}),

            # --- Phase C: Dynamic obstacles (Stage 5-8) ---
            _make_stage(4, 3, 1, 3, 4, 0.994, 65, 0.30,
                        upgrade_sr=0.75, upgrade_max_cr=0.40, upgrade_max_to=0.30,
                        upgrade_min_dyn_sr=0.30, min_stage_updates=100,
                        name="dynamic_intro",
                        reward_weights={"goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 2.0, "dynamic_safety": 2.0}),
            _make_stage(3, 4, 3, 4, 5, 0.995, 72, 0.20,
                        upgrade_sr=0.75, upgrade_max_cr=0.40, upgrade_max_to=0.30,
                        upgrade_min_dyn_sr=0.35, min_stage_updates=115,
                        name="dynamic_medium",
                        reward_weights={"goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 2.0, "dynamic_safety": 2.0}),
            _make_stage(2, 5, 4, 5, 6, 0.997, 80, 0.15,
                        upgrade_sr=0.70, upgrade_max_cr=0.40, upgrade_max_to=0.30,
                        upgrade_min_dyn_sr=0.35, min_stage_updates=130,
                        name="crowded",
                        reward_weights={"goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 2.0, "dynamic_safety": 2.0}),
            _make_stage(1, 7, 7, 6, 8, 0.998, 90, 0.15,
                        upgrade_sr=1.0, upgrade_max_cr=0.0, upgrade_max_to=0.0,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=0,
                        downgrade_sr=0.15, name="final_challenge",
                        reward_weights={"goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 2.0, "dynamic_safety": 2.0}),
        ],
    },

    # ==================================================================
    # goal_first_v2: 密度自適應權重 + 開放式 12 Stage (MAX_OBSTACLES=20)
    #
    # 設計原則:
    # 1. ss/ds weight 跟障礙物密度掛鉤（少障礙=低權重，密集=高權重）
    # 2. vel/prog weight 前期高後期低（先學 goal-reaching）
    # 3. 開放式：Stage 8 不是終點，可繼續增加到 Stage 12
    #
    # 公式: ss_w = ss_base × max(n_static/10, 0.1)
    #        ds_w = ds_base × max(n_dynamic/10, 0.0)
    #        vel_w = lerp(5.0, 2.0, stage_progress)
    #        prog_w = lerp(6.0, 3.0, stage_progress)
    # ==================================================================
    "goal_first_v2": {
        "upgrade_pass_required": 5,
        "clear_window_on_promote": True,
        "stages": [
            # --- Phase A: Goal-reaching (Stage 1-2) ---
            # ss=0.1(floor), ds=0, vel=5, prog=6
            _make_stage(7, 0, 0, 0, 1, 0.990, 45, 1.00,
                        upgrade_sr=0.85, upgrade_max_cr=1.0, upgrade_max_to=0.20,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=50,
                        downgrade_sr=0.0, downgrade_min_cr=1.0, downgrade_min_to=1.0,
                        name="goal_open",
                        reward_weights={"goal_velocity": 5.0, "goal_progress": 6.0,
                                        "static_safety": 0.2, "dynamic_safety": 0.0}),
            _make_stage(6, 0, 0, 1, 2, 0.990, 50, 1.00,
                        upgrade_sr=0.85, upgrade_max_cr=1.0, upgrade_max_to=0.20,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=60,
                        downgrade_sr=0.30, downgrade_min_cr=1.0, downgrade_min_to=0.80,
                        name="goal_walls",
                        reward_weights={"goal_velocity": 5.0, "goal_progress": 6.0,
                                        "static_safety": 0.2, "dynamic_safety": 0.0}),

            # --- Phase B: Static obstacles (Stage 3-5) ---
            # ss 隨密度上升: 2/10=0.4, 4/10=0.8, 6/10=1.2
            _make_stage(5, 2, 0, 2, 3, 0.992, 55, 0.55,
                        upgrade_sr=0.80, upgrade_max_cr=0.35, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=70,
                        name="static_light",
                        reward_weights={"goal_velocity": 4.5, "goal_progress": 5.5,
                                        "static_safety": 0.4, "dynamic_safety": 0.0}),
            _make_stage(4, 4, 0, 3, 4, 0.993, 60, 0.40,
                        upgrade_sr=0.80, upgrade_max_cr=0.35, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=85,
                        name="static_medium",
                        reward_weights={"goal_velocity": 4.0, "goal_progress": 5.0,
                                        "static_safety": 0.8, "dynamic_safety": 0.0}),
            _make_stage(3, 6, 0, 3, 5, 0.994, 65, 0.30,
                        upgrade_sr=0.78, upgrade_max_cr=0.35, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=100,
                        name="static_dense",
                        reward_weights={"goal_velocity": 3.5, "goal_progress": 4.5,
                                        "static_safety": 1.2, "dynamic_safety": 0.0}),

            # --- Phase C: Dynamic obstacles (Stage 6-8) ---
            # reaching_goal 隨 γ/episode 增大: 1000 確保 V(reach) > V(stay)
            _make_stage(3, 5, 2, 4, 5, 0.995, 72, 0.20,
                        upgrade_sr=0.75, upgrade_max_cr=0.40, upgrade_max_to=0.30,
                        upgrade_min_dyn_sr=0.30, min_stage_updates=115,
                        name="dynamic_intro",
                        reward_weights={"reaching_goal": 1000,
                                        "goal_velocity": 3.0, "goal_progress": 4.0,
                                        "static_safety": 1.0, "dynamic_safety": 0.4}),
            _make_stage(2, 6, 4, 5, 6, 0.996, 78, 0.15,
                        upgrade_sr=0.72, upgrade_max_cr=0.40, upgrade_max_to=0.30,
                        upgrade_min_dyn_sr=0.35, min_stage_updates=130,
                        name="dynamic_medium",
                        reward_weights={"reaching_goal": 1000,
                                        "goal_velocity": 2.5, "goal_progress": 3.5,
                                        "static_safety": 1.2, "dynamic_safety": 0.8}),
            _make_stage(1, 7, 6, 6, 7, 0.997, 85, 0.15,
                        upgrade_sr=0.70, upgrade_max_cr=0.40, upgrade_max_to=0.30,
                        upgrade_min_dyn_sr=0.35, min_stage_updates=140,
                        name="crowded",
                        reward_weights={"reaching_goal": 1000,
                                        "goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 1.4, "dynamic_safety": 1.2}),

            # --- Phase D: Open-ended (Stage 9-12, 超過 10 個障礙物) ---
            # reaching_goal=1500: γ=0.998 + 450 步 → V(stay) 可達 240+
            _make_stage(1, 8, 8, 7, 8, 0.998, 90, 0.15,
                        upgrade_sr=0.68, upgrade_max_cr=0.40, upgrade_max_to=0.35,
                        upgrade_min_dyn_sr=0.30, min_stage_updates=150,
                        name="dense_9",
                        reward_weights={"reaching_goal": 1500,
                                        "goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 1.6, "dynamic_safety": 1.6}),
            _make_stage(1, 9, 9, 7, 8, 0.998, 90, 0.10,
                        upgrade_sr=0.65, upgrade_max_cr=0.45, upgrade_max_to=0.35,
                        upgrade_min_dyn_sr=0.30, min_stage_updates=160,
                        name="dense_10",
                        reward_weights={"reaching_goal": 1500,
                                        "goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 1.8, "dynamic_safety": 1.8}),
            _make_stage(1, 10, 10, 8, 8, 0.998, 90, 0.10,
                        upgrade_sr=0.60, upgrade_max_cr=0.45, upgrade_max_to=0.40,
                        upgrade_min_dyn_sr=0.25, min_stage_updates=170,
                        name="dense_11",
                        reward_weights={"reaching_goal": 1500,
                                        "goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 2.0, "dynamic_safety": 2.0}),
            _make_stage(1, 10, 10, 8, 8, 0.998, 90, 0.10,
                        upgrade_sr=1.0, upgrade_max_cr=0.0, upgrade_max_to=0.0,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=0,
                        downgrade_sr=0.15, name="ultimate",
                        reward_weights={"reaching_goal": 1500,
                                        "goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 2.0, "dynamic_safety": 2.0}),
        ],
    },

    # ==================================================================
    # goal_first_v3: v2 基礎 + Stage 3-6 局部提高 static_safety
    #
    # 修正: v2 Stage 3-6 的安全側 (ss) 太弱，agent 偏向硬闖。
    # 只改 Stage 3-6，其餘 stage 與 v2 完全相同，確保可診斷。
    #
    # Stage 3: ss 0.4→0.8  | Stage 4: ss 0.8→1.4
    # Stage 5: ss 1.2→1.8  | Stage 6: ss 1.0→1.6
    # ==================================================================
    "goal_first_v3": {
        "upgrade_pass_required": 5,
        "clear_window_on_promote": True,
        "stages": [
            # --- Phase A: Goal-reaching (Stage 1-2) — 與 v2 相同 ---
            _make_stage(7, 0, 0, 0, 1, 0.990, 45, 1.00,
                        upgrade_sr=0.85, upgrade_max_cr=1.0, upgrade_max_to=0.20,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=50,
                        downgrade_sr=0.0, downgrade_min_cr=1.0, downgrade_min_to=1.0,
                        name="goal_open",
                        reward_weights={"goal_velocity": 5.0, "goal_progress": 6.0,
                                        "static_safety": 0.2, "dynamic_safety": 0.0}),
            _make_stage(6, 0, 0, 1, 2, 0.990, 50, 1.00,
                        upgrade_sr=0.85, upgrade_max_cr=1.0, upgrade_max_to=0.20,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=60,
                        downgrade_sr=0.30, downgrade_min_cr=1.0, downgrade_min_to=0.80,
                        name="goal_walls",
                        reward_weights={"goal_velocity": 5.0, "goal_progress": 6.0,
                                        "static_safety": 0.2, "dynamic_safety": 0.0}),

            # --- Phase B: Static obstacles (Stage 3-5) — ss 局部提高 ---
            # v2: 0.4, 0.8, 1.2 → v3: 0.8, 1.4, 1.8
            _make_stage(5, 2, 0, 2, 3, 0.992, 55, 0.55,
                        upgrade_sr=0.80, upgrade_max_cr=0.35, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=70,
                        name="static_light",
                        reward_weights={"goal_velocity": 4.5, "goal_progress": 5.5,
                                        "static_safety": 0.8, "dynamic_safety": 0.0}),
            _make_stage(4, 4, 0, 3, 4, 0.993, 60, 0.40,
                        upgrade_sr=0.80, upgrade_max_cr=0.35, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=85,
                        name="static_medium",
                        reward_weights={"goal_velocity": 4.0, "goal_progress": 5.0,
                                        "static_safety": 1.4, "dynamic_safety": 0.0}),
            _make_stage(3, 6, 0, 3, 5, 0.994, 65, 0.30,
                        upgrade_sr=0.78, upgrade_max_cr=0.35, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=100,
                        name="static_dense",
                        reward_weights={"goal_velocity": 3.5, "goal_progress": 4.5,
                                        "static_safety": 1.8, "dynamic_safety": 0.0}),

            # --- Phase C: Dynamic obstacles (Stage 6-8) — Stage 6 ss 提高，7-8 同 v2 ---
            _make_stage(3, 5, 2, 4, 5, 0.995, 72, 0.20,
                        upgrade_sr=0.75, upgrade_max_cr=0.40, upgrade_max_to=0.30,
                        upgrade_min_dyn_sr=0.30, min_stage_updates=115,
                        name="dynamic_intro",
                        reward_weights={"reaching_goal": 1000,
                                        "goal_velocity": 3.0, "goal_progress": 4.0,
                                        "static_safety": 1.6, "dynamic_safety": 0.4}),
            # Stage 7-8: 與 v2 相同
            _make_stage(2, 6, 4, 5, 6, 0.996, 78, 0.15,
                        upgrade_sr=0.72, upgrade_max_cr=0.40, upgrade_max_to=0.30,
                        upgrade_min_dyn_sr=0.35, min_stage_updates=130,
                        name="dynamic_medium",
                        reward_weights={"reaching_goal": 1000,
                                        "goal_velocity": 2.5, "goal_progress": 3.5,
                                        "static_safety": 1.2, "dynamic_safety": 0.8}),
            _make_stage(1, 7, 6, 6, 7, 0.997, 85, 0.15,
                        upgrade_sr=0.70, upgrade_max_cr=0.40, upgrade_max_to=0.30,
                        upgrade_min_dyn_sr=0.35, min_stage_updates=140,
                        name="crowded",
                        reward_weights={"reaching_goal": 1000,
                                        "goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 1.4, "dynamic_safety": 1.2}),

            # --- Phase D: Open-ended (Stage 9-12) — 與 v2 相同 ---
            _make_stage(1, 8, 8, 7, 8, 0.998, 90, 0.15,
                        upgrade_sr=0.68, upgrade_max_cr=0.40, upgrade_max_to=0.35,
                        upgrade_min_dyn_sr=0.30, min_stage_updates=150,
                        name="dense_9",
                        reward_weights={"reaching_goal": 1500,
                                        "goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 1.6, "dynamic_safety": 1.6}),
            _make_stage(1, 9, 9, 7, 8, 0.998, 90, 0.10,
                        upgrade_sr=0.65, upgrade_max_cr=0.45, upgrade_max_to=0.35,
                        upgrade_min_dyn_sr=0.30, min_stage_updates=160,
                        name="dense_10",
                        reward_weights={"reaching_goal": 1500,
                                        "goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 1.8, "dynamic_safety": 1.8}),
            _make_stage(1, 10, 10, 8, 8, 0.998, 90, 0.10,
                        upgrade_sr=0.60, upgrade_max_cr=0.45, upgrade_max_to=0.40,
                        upgrade_min_dyn_sr=0.25, min_stage_updates=170,
                        name="dense_11",
                        reward_weights={"reaching_goal": 1500,
                                        "goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 2.0, "dynamic_safety": 2.0}),
            _make_stage(1, 10, 10, 8, 8, 0.998, 90, 0.10,
                        upgrade_sr=1.0, upgrade_max_cr=0.0, upgrade_max_to=0.0,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=0,
                        downgrade_sr=0.15, name="ultimate",
                        reward_weights={"reaching_goal": 1500,
                                        "goal_velocity": 2.0, "goal_progress": 3.0,
                                        "static_safety": 2.0, "dynamic_safety": 2.0}),
        ],
    },

    # ==================================================================
    # open_ended_v1: 6 bootstrap stages + 無上限 open-ended curriculum
    #
    # Bootstrap (B1-B6): 固定 stages，先學 goal-reaching + 基本避障
    # Open-ended: difficulty_level 無上限，參數化映射場景難度
    #   static clamp 20, dynamic clamp 5, walls=0
    # ==================================================================
    "open_ended_v1": {
        "upgrade_pass_required": 5,
        "clear_window_on_promote": True,
        "open_ended": True,  # 標記此版本使用 open-ended 模式
        "stages": [
            # B1: goal_open — 純導航 (collision=-5, 前期靠死亡機制)
            _make_stage(6, 0, 0, 0, 0, 0.990, 45, 1.00,
                        upgrade_sr=0.85, upgrade_max_cr=1.0, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=50,
                        downgrade_sr=0.0, downgrade_min_cr=1.0, downgrade_min_to=1.0,
                        name="B1_goal_open",
                        reward_weights={"goal_velocity": 5.0, "goal_progress": 6.0,
                                        "static_safety": 0.0, "dynamic_safety": 0.0,
                                        "collision_ground": -5}),
            # B2: goal_sparse_static (collision=-5)
            _make_stage(5, 2, 0, 0, 0, 0.992, 50, 0.60,
                        upgrade_sr=0.82, upgrade_max_cr=0.35, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=60,
                        downgrade_sr=0.30, downgrade_min_cr=1.0, downgrade_min_to=0.80,
                        name="B2_sparse_static",
                        reward_weights={"goal_velocity": 5.0, "goal_progress": 5.8,
                                        "static_safety": 0.2, "dynamic_safety": 0.0,
                                        "collision_ground": -5}),
            # B3: static_light (collision=-5, ss 降低確保 goal 主導)
            _make_stage(4, 4, 0, 0, 0, 0.993, 55, 0.40,
                        upgrade_sr=0.80, upgrade_max_cr=0.35, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=70,
                        name="B3_static_light",
                        reward_weights={"goal_velocity": 4.8, "goal_progress": 5.5,
                                        "static_safety": 0.3, "dynamic_safety": 0.0,
                                        "collision_ground": -5}),
            # B4: static_medium (collision=-5)
            _make_stage(3, 6, 0, 0, 0, 0.994, 60, 0.25,
                        upgrade_sr=0.78, upgrade_max_cr=0.35, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=85,
                        name="B4_static_medium",
                        reward_weights={"goal_velocity": 4.5, "goal_progress": 5.0,
                                        "static_safety": 0.4, "dynamic_safety": 0.0,
                                        "collision_ground": -5}),
            # B5: dynamic_intro (collision=-10, vel 提高保持前進主導)
            _make_stage(3, 6, 2, 0, 0, 0.995, 68, 0.10,
                        upgrade_sr=0.75, upgrade_max_cr=0.35, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=100,
                        name="B5_dynamic_intro",
                        reward_weights={"goal_velocity": 4.5, "goal_progress": 4.5,
                                        "static_safety": 0.5, "dynamic_safety": 0.2,
                                        "collision_ground": -10}),
            # B6: dynamic_bridge (collision=-20, 3G 7S 2D, ss+ds 壓低)
            _make_stage(3, 7, 2, 0, 0, 0.996, 75, 0.0,
                        upgrade_sr=0.72, upgrade_max_cr=0.35, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=115,
                        name="B6_dynamic_bridge",
                        reward_weights={"goal_velocity": 4.0, "goal_progress": 4.0,
                                        "static_safety": 0.5, "dynamic_safety": 0.3,
                                        "collision_ground": -20}),
        ],
    },
    # navrl_hybrid: HEIGHT-friendly 混合課程
    # 關鍵設計：static 達到 6 之後固定不減，dynamic 逐步 ramp 0→2→4→6
    # 對齊 NavRL 論文實際做法 (static fixed + dynamic ramp + SR>80% gate)
    # 解決 v24/v25 symmetric nS=nD 課程下 static 壓力不足的問題
    "navrl_hybrid": {
        "upgrade_pass_required": 5,
        "clear_window_on_promote": True,
        "open_ended": True,
        "stages": [
            # H0: goal warmup (0s+0d)
            _make_stage(6, 0, 0, 0, 0, 0.990, 45, 1.00,
                        upgrade_sr=0.85, upgrade_max_cr=1.0, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=50,
                        downgrade_sr=0.0, downgrade_min_cr=1.0, downgrade_min_to=1.0,
                        name="H0_goal_warmup",
                        reward_weights={"goal_velocity": 5.0, "goal_progress": 6.0,
                                        "static_safety": 0.0, "dynamic_safety": 0.0,
                                        "collision_ground": -5}),
            # H1: static ramp 2
            _make_stage(5, 2, 0, 0, 0, 0.992, 50, 0.60,
                        upgrade_sr=0.85, upgrade_max_cr=0.30, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=60,
                        downgrade_sr=0.30, downgrade_min_cr=1.0, downgrade_min_to=0.80,
                        name="H1_static2",
                        reward_weights={"goal_velocity": 5.0, "goal_progress": 5.8,
                                        "static_safety": 0.2, "dynamic_safety": 0.0,
                                        "collision_ground": -5}),
            # H2: static ramp 4
            _make_stage(4, 4, 0, 0, 0, 0.993, 55, 0.35,
                        upgrade_sr=0.83, upgrade_max_cr=0.30, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=70,
                        name="H2_static4",
                        reward_weights={"goal_velocity": 4.8, "goal_progress": 5.5,
                                        "static_safety": 0.3, "dynamic_safety": 0.0,
                                        "collision_ground": -5}),
            # H3: static plateau 6 (最終 static 密度)
            _make_stage(3, 6, 0, 0, 0, 0.994, 60, 0.20,
                        upgrade_sr=0.82, upgrade_max_cr=0.30, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=85,
                        name="H3_static6",
                        reward_weights={"goal_velocity": 4.5, "goal_progress": 5.0,
                                        "static_safety": 0.4, "dynamic_safety": 0.0,
                                        "collision_ground": -5}),
            # H4: dynamic intro (6s + 2d) — static 不減！
            _make_stage(3, 6, 2, 0, 0, 0.995, 68, 0.10,
                        upgrade_sr=0.80, upgrade_max_cr=0.32, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.60, min_stage_updates=100,
                        name="H4_dyn2",
                        reward_weights={"goal_velocity": 4.5, "goal_progress": 4.5,
                                        "static_safety": 0.5, "dynamic_safety": 0.2,
                                        "collision_ground": -10}),
            # H5: dynamic mid (6s + 4d)
            _make_stage(3, 6, 4, 0, 0, 0.996, 72, 0.05,
                        upgrade_sr=0.78, upgrade_max_cr=0.32, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.60, min_stage_updates=115,
                        name="H5_dyn4",
                        reward_weights={"goal_velocity": 4.2, "goal_progress": 4.2,
                                        "static_safety": 0.5, "dynamic_safety": 0.3,
                                        "collision_ground": -15}),
            # H6: dynamic final (6s + 6d)
            _make_stage(3, 6, 6, 0, 0, 0.997, 75, 0.0,
                        upgrade_sr=0.75, upgrade_max_cr=0.35, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.65, min_stage_updates=130,
                        name="H6_dyn6",
                        reward_weights={"goal_velocity": 4.0, "goal_progress": 4.0,
                                        "static_safety": 0.5, "dynamic_safety": 0.35,
                                        "collision_ground": -20}),
        ],
    },
    # navrl_hybrid_dense: v27 warm-start 用
    # 從 v26 navrl_hybrid H6 (6s+6d) best_model 接續
    # 逐步推高密度到最終目標 19s+15d
    "navrl_hybrid_dense": {
        "upgrade_pass_required": 5,
        "clear_window_on_promote": True,
        "open_ended": True,
        "stages": [
            # D0: 起始 = v26 H6 最終條件 (6s+6d)，warm start 過渡
            _make_stage(3, 6, 6, 0, 0, 0.997, 75, 0.0,
                        upgrade_sr=0.75, upgrade_max_cr=0.35, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.60, min_stage_updates=80,
                        downgrade_sr=0.30, downgrade_min_cr=1.0, downgrade_min_to=0.80,
                        name="D0_warmstart_6s6d",
                        reward_weights={"goal_velocity": 4.0, "goal_progress": 4.0,
                                        "static_safety": 0.5, "dynamic_safety": 0.35,
                                        "collision_ground": -20}),
            # D1: 8s+8d — 開始推密度
            _make_stage(3, 8, 8, 0, 0, 0.997, 80, 0.0,
                        upgrade_sr=0.70, upgrade_max_cr=0.38, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.55, min_stage_updates=100,
                        name="D1_8s8d",
                        reward_weights={"goal_velocity": 3.8, "goal_progress": 3.8,
                                        "static_safety": 0.5, "dynamic_safety": 0.4,
                                        "collision_ground": -25}),
            # D2: 10s+8d — 靜態先推
            _make_stage(2, 10, 8, 0, 0, 0.997, 85, 0.0,
                        upgrade_sr=0.65, upgrade_max_cr=0.42, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.45, min_stage_updates=120,
                        name="D2_10s8d",
                        reward_weights={"goal_velocity": 3.5, "goal_progress": 3.5,
                                        "static_safety": 0.5, "dynamic_safety": 0.4,
                                        "collision_ground": -25}),
            # D3: 12s+10d
            _make_stage(2, 12, 10, 0, 0, 0.997, 90, 0.0,
                        upgrade_sr=0.60, upgrade_max_cr=0.45, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.45, min_stage_updates=130,
                        name="D3_12s10d",
                        reward_weights={"goal_velocity": 3.2, "goal_progress": 3.2,
                                        "static_safety": 0.5, "dynamic_safety": 0.45,
                                        "collision_ground": -30}),
            # D4: 15s+12d
            _make_stage(2, 15, 12, 0, 0, 0.998, 95, 0.0,
                        upgrade_sr=0.55, upgrade_max_cr=0.50, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.40, min_stage_updates=150,
                        name="D4_15s12d",
                        reward_weights={"goal_velocity": 3.0, "goal_progress": 3.0,
                                        "static_safety": 0.5, "dynamic_safety": 0.5,
                                        "collision_ground": -30}),
            # D5: 19s+15d — 最終目標密度
            _make_stage(2, 19, 15, 0, 0, 0.998, 105, 0.0,
                        upgrade_sr=0.50, upgrade_max_cr=0.55, upgrade_max_to=0.25,
                        upgrade_min_dyn_sr=0.35, min_stage_updates=180,
                        name="D5_19s15d_final",
                        reward_weights={"goal_velocity": 2.8, "goal_progress": 2.8,
                                        "static_safety": 0.5, "dynamic_safety": 0.5,
                                        "collision_ground": -35}),
        ],
    },

    # ==================================================================
    # warp_drive_v1: 完全照 Warp Drive train_rnn_car.py Phase 1-8 設計
    #
    # Warp Drive 原始 → Isaac Lab 映射:
    #   num_spot → 1 (Isaac Lab 只能 1 robot/env)
    #   num_goals → num_goals (直接對應)
    #   num_obstacle → num_obstacles_dynamic (全部 policy-controlled)
    #   num_stairs → 0 (Isaac Lab 無樓梯)
    #   num_wall → walls
    #
    # 核心差異: obstacles 從 Phase 1 就有 3 個 dynamic，obstacle policy 一開始就訓練。
    # Static obstacles = 0（Warp Drive 沒有 static obstacle 概念）。
    #
    # | Phase | Spot→Charge | Goals | Obstacles(dynamic) | Walls | Episode(s) |
    # |-------|-------------|-------|--------------------|-------|------------|
    # | 1     | 1 (WD=6)   | 8     | 3                  | 0     | 60         |
    # | 2     | 1 (WD=3)   | 8     | 3                  | 0     | 60         |
    # | 3     | 1 (WD=3)   | 1     | 2                  | 0     | 90         |
    # | 4     | 1 (WD=3)   | 1     | 2                  | 1     | 60         |
    # | 5     | 1 (WD=2)   | 1     | 2                  | 1     | 100        |
    # | 6     | 1 (WD=3)   | 1     | 10                 | 1     | 210        |
    # | 7     | 1 (WD=2)   | 1     | 10                 | 2     | 210        |
    # | 8+    | 1 (WD=2)   | 1     | 6                  | 2     | 210        |
    # ==================================================================
    "warp_drive_v1": {
        "upgrade_pass_required": 5,
        "clear_window_on_promote": True,
        "stages": [
            # Phase 1: 基本導航 + 3 dynamic obstacles（scripted warm-up or policy）
            # WD: spot=6, goals=20, obs=3, stairs=1, rl_fps=4, episode=60s
            # WD reward: penalty_hit=-5, cost_operate=0.03, get_goal=40
            # WD entropy: spot_entropy=0.04×2.5=0.10 (低: 先學基本走)
            # WD obs_speed: 0.8
            _make_stage(8, 0, 3, 0, 0, 0.990, 60, 0.50,
                        upgrade_sr=0.72, upgrade_max_cr=1.0, upgrade_max_to=0.30,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=50,
                        downgrade_sr=0.0, downgrade_min_cr=1.0, downgrade_min_to=1.0,
                        name="WD1_nav_3obs",
                        spot_penalty_hit=-5.0, spot_cost_operate=0.03,
                        ent_coeff_linear=0.10,
                        obstacle_speed_rate=0.8),

            # Phase 2: spot 減少 → 更高存活期望值
            # WD: spot=3, goals=16, obs=3, obs_speed=0.85×spot
            # WD reward: penalty_hit=-8
            _make_stage(8, 0, 3, 0, 0, 0.991, 60, 0.50,
                        upgrade_sr=0.65, min_stage_updates=65,
                        name="WD2_3obs_fast",
                        spot_penalty_hit=-8.0,
                        obstacle_speed_rate=0.85),

            # Phase 3: 單目標追蹤 + cooperate mode
            # WD: spot=3, goals=1, obs=2, episode=90s, obs_size_rand=0.1
            # WD reward: penalty_hit=-12, obs_speed=0.85
            _make_stage(1, 0, 2, 0, 0, 0.993, 90, 0.40,
                        upgrade_sr=0.65, upgrade_min_dyn_sr=0.35, min_stage_updates=80,
                        name="WD3_single_goal",
                        obs_size_rand=0.1, scene_bound_rand=0.5,
                        spot_penalty_hit=-12.0,
                        obstacle_speed_rate=0.85),

            # Phase 4: 加入牆壁
            # WD: spot=3, goals=1, obs=2, wall=1, obs_size_rand=0.2, obs_speed=0.85
            # WD reward: penalty_hit=-12
            _make_stage(1, 0, 2, 0, 1, 0.994, 60, 0.40,
                        upgrade_sr=0.65, upgrade_min_dyn_sr=0.35, min_stage_updates=95,
                        name="WD4_wall",
                        obs_size_rand=0.2, scene_bound_rand=1.0,
                        spot_penalty_hit=-12.0,
                        obstacle_speed_rate=0.85),

            # Phase 5: 長 episode
            # WD: spot=2, goals=1, obs=2, wall=1, obs_size_rand=0.3, obs_speed=0.85
            # WD reward: penalty_hit=-12
            _make_stage(1, 0, 2, 0, 1, 0.995, 100, 0.35,
                        upgrade_sr=0.65, upgrade_min_dyn_sr=0.35, min_stage_updates=110,
                        name="WD5_long",
                        obs_size_rand=0.3, scene_bound_rand=1.0,
                        spot_penalty_hit=-12.0,
                        obstacle_speed_rate=0.85),

            # Phase 6: 大量動態障礙
            # WD: spot=3, goals=1, obs=10, wall=1, obs_size_rand=0.4, obs_speed=0.85
            # WD reward: penalty_hit=-15
            _make_stage(1, 0, 10, 0, 1, 0.997, 210, 0.10,
                        upgrade_sr=0.60, upgrade_min_dyn_sr=0.30, min_stage_updates=130,
                        name="WD6_10obs",
                        obs_size_rand=0.4, scene_bound_rand=1.0,
                        spot_penalty_hit=-15.0,
                        obstacle_speed_rate=0.85),

            # Phase 7: 主要訓練階段（保守策略訓練）
            # WD: spot=2, goals=1, obs=10, wall=2, obs_size_rand=0.4, obs_speed=1.15
            # WD reward: penalty_hit=-85（大幅增加避障壓力）
            _make_stage(1, 0, 10, 0, 2, 0.998, 210, 0.10,
                        upgrade_sr=0.55, upgrade_min_dyn_sr=0.30, min_stage_updates=150,
                        name="WD7_main",
                        obs_size_rand=0.4, scene_bound_rand=1.0,
                        spot_penalty_hit=-85.0,
                        obstacle_speed_rate=1.15),

            # Phase 8+: 穩定化（極端碰撞懲罰）
            # WD: spot=2, goals=1, obs=6, wall=2, obs_size_rand=0.4, obs_speed=1.15
            # WD reward: penalty_hit=-200
            _make_stage(1, 0, 6, 0, 2, 0.998, 210, 0.15,
                        upgrade_sr=1.0, upgrade_max_cr=0.0, upgrade_max_to=0.0,
                        upgrade_min_dyn_sr=0.0, min_stage_updates=0,
                        downgrade_sr=0.15, name="WD8_stable",
                        obs_size_rand=0.4, scene_bound_rand=1.0,
                        spot_penalty_hit=-200.0,
                        obstacle_speed_rate=1.15),
        ],
    },
}


# ============================================================================
# Open-ended curriculum 參數映射
# ============================================================================

def _open_ended_params(level: int) -> dict:
    """difficulty_level → 場景參數映射。level=0 從 B6 延續。

    障礙物: total = min(50, 11 + level*3), dynamic 比例從 27% 漸增到 40%
    L0=11(8S+3D), L10=41(28S+13D), L13=50(32S+18D, cap)
    """
    # 障礙物: 總數線性增長，dynamic 比例漸增（上限 50 = MAX_OBSTACLES）
    total_obs = min(50, 11 + level * 3)
    dyn_ratio = min(0.40, 0.27 + 0.005 * level)
    dynamic_count = int(total_obs * dyn_ratio)
    static_count = total_obs - dynamic_count

    goal_count = 1 if level >= 2 else 2
    episode_length_s = min(95, 75 + level * 2)
    gamma = min(0.998, 0.996 + level * 0.0002)

    # 物件數到頂後 (L30+)，透過隨機性增加難度
    beyond = max(0, level - 30)
    dynamic_speed_scale = min(1.4, 1.0 + 0.03 * beyond)
    spawn_compactness = min(1.3, 1.0 + 0.02 * beyond)
    goal_distance_scale = min(1.25, 1.0 + 0.02 * beyond)

    # 有動態障礙 → mixed mode（靜態不動 + 動態移動）
    # 注意：dynamic_ratio=1.0 會讓所有障礙物都移動，不區分靜態/動態
    # mixed_ratio=1.0 才能正確區分：i < num_static 靜止，i >= num_static 移動
    if dynamic_count > 0 and static_count > 0:
        empty_ratio = 0.0
        static_ratio = 0.0
        dynamic_ratio = 0.0
        mixed_ratio = 1.0
    elif dynamic_count > 0:
        # 純動態（無靜態）→ dynamic mode OK
        empty_ratio = 0.0
        static_ratio = 0.0
        dynamic_ratio = 1.0
        mixed_ratio = 0.0
    else:
        empty_ratio = 0.0
        static_ratio = 1.0
        dynamic_ratio = 0.0
        mixed_ratio = 0.0

    return {
        "name": f"OE_L{level}",
        "num_goals": goal_count,
        "num_obstacles_static": static_count,
        "num_obstacles_dynamic": dynamic_count,
        "empty_ratio": empty_ratio,
        "static_ratio": static_ratio,
        "dynamic_ratio": dynamic_ratio,
        "mixed_ratio": mixed_ratio,
        "min_walls": 0,
        "max_walls": 0,
        "episode_length_s": float(episode_length_s),
        "gamma": gamma,
        "goal_distance": (3.0, min(13.0, 8.0 * goal_distance_scale)),
        # 進階難度參數
        "speed_range": 1.2 * dynamic_speed_scale,
        "boundary": max(5.5, 7.5 / spawn_compactness),
    }


def _open_ended_reward_weights(level: int) -> dict:
    """difficulty_level → reward 權重映射。

    v8 修正: 降低 safety 上限、提高 velocity 下限。
    確保 (ss+ds)/goal < 30%，防止安全 reward 搶走核心主導。

    Ablation 5: 可透過 set_ablation_params() 調整 safety 權重。
    """
    base_ss = min(0.5, 0.3 + 0.02 * level)
    base_ds = min(0.4, 0.2 + 0.02 * level)
    return {
        "goal_velocity": max(3.5, 4.0 - 0.05 * level),
        "goal_progress": max(3.5, 4.0 - 0.05 * level),
        "static_safety": _apply_ss_scale(base_ss),
        "dynamic_safety": _apply_ss_scale(base_ds),
    }


def _open_ended_target_sr(level: int) -> float:
    """升級 SR 門檻 — 固定 80%（文獻共識：navigation 任務標準門檻）。"""
    return 0.80

def _open_ended_target_cr(level: int) -> float:
    """升級 CR 上限 — 固定 15%（碰撞是嚴重失敗，需嚴格控制）。"""
    return 0.15

def _open_ended_target_to(level: int) -> float:
    """升級 TO 上限 — 固定 20%。"""
    return 0.20

def _open_ended_min_stage_updates(level: int) -> int:
    """open-ended level → min_stage_updates（rollout cycles）。

    歷史數據: metrics 在 window 填滿 (6 cycles) 後即穩定達標。
    前期多留 buffer（場景初次改變需適應），後期收斂到下限。
    下限 8 ≈ 剛好填滿 window + min_stage_episodes。
    """
    return max(8, 20 - level)


def _apply_open_ended(env, level: int):
    """套用 open-ended 難度等級的場景參數 + reward 權重。"""
    cfg = _open_ended_params(level)

    # Goal command
    try:
        cmd = env.command_manager.get_term("goal_command")
        cmd.cfg.num_goals = cfg["num_goals"]
        cmd.cfg.ranges.distance = cfg["goal_distance"]
        cmd.cfg.num_obstacles = cfg["num_obstacles_static"] + cfg["num_obstacles_dynamic"]
    except Exception:
        pass

    # Obstacle event params
    try:
        evt = env.event_manager
        for name in ["randomize_obstacles", "randomize_obstacles_startup"]:
            try:
                ec = evt.get_term_cfg(name)
                ec.params["empty_ratio"] = cfg["empty_ratio"]
                ec.params["static_ratio"] = cfg["static_ratio"]
                ec.params["dynamic_ratio"] = cfg["dynamic_ratio"]
                ec.params["mixed_ratio"] = cfg.get("mixed_ratio", 0.0)
                ec.params["num_obstacles_static"] = cfg["num_obstacles_static"]
                ec.params["num_obstacles_dynamic"] = cfg["num_obstacles_dynamic"]
                ec.params["boundary"] = cfg["boundary"]
                evt.set_term_cfg(name, ec)
            except Exception:
                continue
        # 更新 env._num_obstacles 讓 move_obstacles_vectorized 的 for loop
        # 只跑實際用量（N），不跑全局 max_obstacles(100)
        env._num_obstacles = cfg["num_obstacles_static"] + cfg["num_obstacles_dynamic"]
        # 清除 obstacle cache，讓 move_obstacles_vectorized 下次重建
        if hasattr(env, "_obstacle_cache"):
            del env._obstacle_cache
        # 動態速度
        try:
            ec = evt.get_term_cfg("move_dynamic_obstacles")
            ec.params["speed_max"] = cfg["speed_range"]
            evt.set_term_cfg("move_dynamic_obstacles", ec)
        except Exception:
            pass
    except Exception:
        pass

    # Walls = 0
    try:
        evt = env.event_manager
        ec = evt.get_term_cfg("randomize_wall_positions")
        ec.params["min_walls"] = 0
        ec.params["max_walls"] = 0
        evt.set_term_cfg("randomize_wall_positions", ec)
    except Exception:
        pass

    # Episode length + gamma
    env.cfg.episode_length_s = cfg["episode_length_s"]
    env._target_discount_factor = cfg["gamma"]

    # Reward weights (含碰撞成本遞增)
    rw = _open_ended_reward_weights(level)
    # 碰撞成本遞增: -30 + level * -5, cap at -80
    collision_w = max(-80, -30 + level * -5)
    rw["collision_ground"] = collision_w
    try:
        rm = env.reward_manager
        for term_name, weight in rw.items():
            try:
                tc = rm.get_term_cfg(term_name)
                tc.weight = weight
                rm.set_term_cfg(term_name, tc)
            except Exception:
                pass
        weights_str = " ".join(f"{k}={v:.2f}" for k, v in rw.items())
        print(f"[OpenEnded] Level {level} reward weights: {weights_str}", flush=True)
    except Exception:
        pass

    print(
        f"[OpenEnded] Level {level}: "
        f"{cfg['num_goals']}G {cfg['num_obstacles_static']}S+{cfg['num_obstacles_dynamic']}D "
        f"walls=0 ep={cfg['episode_length_s']:.0f}s γ={cfg['gamma']:.4f} "
        f"speed={cfg['speed_range']:.2f} boundary={cfg['boundary']:.1f} "
        f"collision_w={collision_w}",
        flush=True,
    )


def _load_stages(version: str) -> tuple[dict, int, dict]:
    """根據版本載入 stages → (stages_dict, max_stage, version_config)"""
    if version not in CURRICULUM_CONFIGS:
        print(f"[Curriculum] WARNING: unknown version '{version}', fallback to baseline_v1")
        version = "baseline_v1"

    config = CURRICULUM_CONFIGS[version]
    stages = {}
    for i, stage_cfg in enumerate(config["stages"]):
        stages[i + 1] = stage_cfg
    return stages, len(config["stages"]), config


# ============================================================================
# 全局 STAGES（預設 baseline_v1 用於向後相容）
# ============================================================================
STAGES, MAX_STAGE, _DEFAULT_CONFIG = _load_stages("baseline_v1")


# ============================================================================
# 主函數
# ============================================================================

def goal_obstacle_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
    window_size: int = 2000,
    min_stage_episodes: int = 5000,
    initial_stage: int = 1,
    curriculum_version: str = "baseline_v1",
) -> dict[str, float]:
    """資料驅動課程學習 — 支援多版本切換。

    Args:
        curriculum_version: "baseline_v1" (原 v12) 或 "goal_first_v1" (先導航再避障)
    """
    global STAGES, MAX_STAGE

    if not hasattr(env, "_goal_obs_curriculum"):
        # 根據 version 載入 stages
        stages, max_stage, ver_config = _load_stages(curriculum_version)
        STAGES = stages
        MAX_STAGE = max_stage
        upgrade_pass_required = ver_config.get("upgrade_pass_required", 5)
        is_open_ended = ver_config.get("open_ended", False)

        n = env.num_envs
        if is_open_ended:
            effective_window = max(4000, n * 6)
        else:
            effective_window = max(window_size, n * 5)
        effective_min = max(min_stage_episodes, n * 8)
        env._goal_obs_curriculum = {
            "stage": initial_stage,
            "outcome_window": deque(maxlen=effective_window),
            "effective_window_size": effective_window,
            "min_stage_episodes": effective_min,
            "total_episodes": 0,
            "stage_episodes": 0,
            "stage_transitions": 0,
            "upgrade_pass_count": 0,
            "upgrade_pass_required": upgrade_pass_required,
            "curriculum_version": curriculum_version,
            "clear_window_on_promote": ver_config.get("clear_window_on_promote", True),
            # Open-ended 狀態
            "is_open_ended": is_open_ended,
            "curriculum_mode": "bootstrap",  # "bootstrap" | "open_ended"
            "difficulty_level": 0,
            "cooldown_remaining": 0,
            "downgrade_streak": 0,  # 連續降級 window 計數
        }
        _apply_stage(env, initial_stage)
        s = STAGES[initial_stage]
        print(
            f"\n{'='*70}\n"
            f"[Curriculum {curriculum_version}] 初始化 — Stage {initial_stage}: "
            f"{s.get('name', '')}\n"
            f"  num_envs={n}  |  窗口={effective_window}  |  "
            f"最低停留={effective_min} episodes\n"
            f"  Goals={s['num_goals']}  "
            f"障礙物={s['num_obstacles_static']}S+{s['num_obstacles_dynamic']}D  "
            f"牆壁={s['min_walls']}-{s['max_walls']}\n"
            f"  γ={s['gamma']}  episode={s['episode_length_s']}s\n"
            f"  升級: SR>{s['upgrade_sr']:.0%}"
            f"{'  CR<' + format(s['upgrade_max_cr'], '.0%') if s['upgrade_max_cr'] < 1.0 else ''}"
            f"  TO<{s['upgrade_max_to']:.0%}"
            f"  需連續通過 {upgrade_pass_required} 次\n"
            f"{'='*70}",
            flush=True,
        )

    state = env._goal_obs_curriculum
    upgrade_pass_required = state.get("upgrade_pass_required", 5)

    if env_ids is not None and not isinstance(env_ids, slice):
        _collect_episode_results(env, env_ids, state)

    window = state["outcome_window"]
    effective_window = state["effective_window_size"]

    # 計算整體 rates
    if len(window) >= 10:
        total = len(window)
        success_rate = sum(1 for o, _ in window if o == 1) / total
        collision_rate = sum(1 for o, _ in window if o == -1) / total
        timeout_rate = sum(1 for o, _ in window if o == 0) / total
    else:
        success_rate = 0.0
        collision_rate = 0.0
        timeout_rate = 0.0

    # dynamic-only SR
    dyn_outcomes = [(o, t) for o, t in window if t == 2]
    dynamic_sr = sum(1 for o, _ in dyn_outcomes if o == 1) / max(len(dyn_outcomes), 1)

    current_stage = state["stage"]
    stage_cfg = STAGES[current_stage]

    # DEBUG: 每 50000 episodes 輸出完整狀態
    if state["total_episodes"] % 50000 < env.num_envs:
        print(
            f"\n[DEBUG Curriculum State] "
            f"episodes={state['total_episodes']} stage={current_stage} "
            f"mode={state.get('curriculum_mode')} level={state.get('difficulty_level')} "
            f"is_oe={state.get('is_open_ended')} pass={state['upgrade_pass_count']}/{upgrade_pass_required} "
            f"cooldown={state.get('cooldown_remaining')} window={len(window)}/{effective_window}",
            flush=True,
        )

    approx_rollout_cycles = state["stage_episodes"] // max(env.num_envs, 1)
    # open-ended 模式使用自適應 min_stage_updates（前期長、後期短）
    if state.get("curriculum_mode") == "open_ended":
        min_stage_updates = _open_ended_min_stage_updates(state.get("difficulty_level", 0))
    else:
        min_stage_updates = stage_cfg.get("min_stage_updates", 0)

    can_transition = (
        len(window) >= effective_window
        and state["stage_episodes"] >= state["min_stage_episodes"]
        and approx_rollout_cycles >= min_stage_updates
    )

    if can_transition:
        up_sr = stage_cfg["upgrade_sr"]
        up_cr = stage_cfg["upgrade_max_cr"]
        up_to = stage_cfg["upgrade_max_to"]
        up_dyn_sr = stage_cfg.get("upgrade_min_dyn_sr", 0.0)
        down_sr = stage_cfg["downgrade_sr"]
        down_cr = stage_cfg["downgrade_min_cr"]
        down_to = stage_cfg["downgrade_min_to"]

        total = len(window)
        success_count = sum(1 for o, _ in window if o == 1)
        collision_count = sum(1 for o, _ in window if o == -1)
        timeout_count = sum(1 for o, _ in window if o == 0)

        ver = state["curriculum_version"]

        def _debug_log(direction: str, old_stage: int, new_stage: int):
            s = STAGES[new_stage]
            print(
                f"\n{'='*70}\n"
                f"[Curriculum {ver}] {direction} Stage {old_stage} → "
                f"Stage {new_stage}: {s.get('name', '')}\n"
                f"  SR={success_rate:.4f} CR={collision_rate:.4f} TO={timeout_rate:.4f} "
                f"dynSR={dynamic_sr:.4f}\n"
                f"  累計 {state['total_episodes']} ep, "
                f"第 {state['stage_transitions']} 次轉換\n"
                f"  Goals={s['num_goals']}  "
                f"障礙物={s['num_obstacles_static']}S+{s['num_obstacles_dynamic']}D  "
                f"牆壁={s['min_walls']}-{s['max_walls']}\n"
                f"  γ={s['gamma']}  episode={s['episode_length_s']}s\n"
                f"{'='*70}",
                flush=True,
            )

        # ============================================================
        # Open-ended 模式的升降級邏輯
        # ============================================================
        if state.get("curriculum_mode") == "open_ended":
            level = state["difficulty_level"]
            cooldown = state.get("cooldown_remaining", 0)
            cooldown_size = max(2000, env.num_envs * 4)

            # DEBUG: 每 10000 episodes 輸出一次狀態
            if state["total_episodes"] % 10000 < env.num_envs:
                print(
                    f"[DEBUG OpenEnded] mode={state.get('curriculum_mode')} level={level} "
                    f"pass={state['upgrade_pass_count']}/{upgrade_pass_required} "
                    f"cooldown={cooldown} SR={success_rate:.3f} CR={collision_rate:.3f} TO={timeout_rate:.3f}",
                    flush=True,
                )

            # Cooldown 遞減
            new_episodes = state["stage_episodes"]  # 本 level 的 episode 數
            if cooldown > 0:
                state["cooldown_remaining"] = max(0, cooldown - len(window))
            else:
                t_sr = _open_ended_target_sr(level)
                t_cr = _open_ended_target_cr(level)
                t_to = _open_ended_target_to(level)

                sr_ok = success_rate > t_sr
                cr_ok = collision_rate < t_cr
                to_ok = timeout_rate < t_to

                # DEBUG: 輸出升級條件檢查
                if state["total_episodes"] % 10000 < env.num_envs:
                    print(
                        f"[DEBUG OpenEnded Check] t_sr={t_sr:.2f} t_cr={t_cr:.2f} t_to={t_to:.2f} "
                        f"sr_ok={sr_ok} cr_ok={cr_ok} to_ok={to_ok}",
                        flush=True,
                    )

                if sr_ok and cr_ok and to_ok:
                    state["upgrade_pass_count"] += 1
                    state["downgrade_streak"] = 0
                    if state["upgrade_pass_count"] >= upgrade_pass_required:
                        old_level = level
                        level += 1
                        state["difficulty_level"] = level
                        state["outcome_window"].clear()
                        state["stage_episodes"] = 0
                        state["stage_transitions"] += 1
                        state["upgrade_pass_count"] = 0
                        state["cooldown_remaining"] = cooldown_size
                        _apply_open_ended(env, level)
                        p = _open_ended_params(level)
                        print(
                            f"\n{'='*70}\n"
                            f"[OpenEnded] ▲ Level {old_level} → {level} | "
                            f"SR={success_rate:.3f} CR={collision_rate:.3f} TO={timeout_rate:.3f}\n"
                            f"{'='*70}", flush=True,
                        )
                else:
                    state["upgrade_pass_count"] = 0

                    # 降級檢查 (需要連續 2 個 window 觸發)
                    should_down = (
                        success_rate < t_sr - 0.12 or
                        collision_rate > t_cr + 0.12
                    )
                    if should_down:
                        state["downgrade_streak"] += 1
                    else:
                        state["downgrade_streak"] = 0

                    if state["downgrade_streak"] >= 2 and level > 0:
                        old_level = level
                        level -= 1
                        state["difficulty_level"] = level
                        state["outcome_window"].clear()
                        state["stage_episodes"] = 0
                        state["stage_transitions"] += 1
                        state["cooldown_remaining"] = cooldown_size
                        state["downgrade_streak"] = 0
                        _apply_open_ended(env, level)
                        print(
                            f"\n{'='*70}\n"
                            f"[OpenEnded] ▼ Level {old_level} → {level} | "
                            f"SR={success_rate:.3f} CR={collision_rate:.3f} TO={timeout_rate:.3f}\n"
                            f"{'='*70}", flush=True,
                        )

            # 回傳 open-ended metrics
            oe_cfg = _open_ended_params(level)
            t_sr = _open_ended_target_sr(level)
            t_cr = _open_ended_target_cr(level)
            t_to = _open_ended_target_to(level)
            return {
                "stage": float(MAX_STAGE + level),  # bootstrap stages + level
                "difficulty_level": float(level),
                "stage_name": oe_cfg["name"],
                "success_rate": success_rate,
                "collision_rate": collision_rate,
                "timeout_rate": timeout_rate,
                "dynamic_sr": dynamic_sr,
                "num_goals": float(oe_cfg["num_goals"]),
                "num_obstacles_static": float(oe_cfg["num_obstacles_static"]),
                "num_obstacles_dynamic": float(oe_cfg["num_obstacles_dynamic"]),
                "min_walls": 0.0,
                "max_walls": 0.0,
                "gamma": float(oe_cfg["gamma"]),
                "episode_length_s": float(oe_cfg["episode_length_s"]),
                "num_episodes": float(state["total_episodes"]),
                "stage_episodes": float(state["stage_episodes"]),
                "window_fill": float(len(window)) / float(effective_window),
                "upgrade_pass_count": float(state["upgrade_pass_count"]),
                "approx_rollout_cycles": float(approx_rollout_cycles),
                "upgrade_sr_target": t_sr,
                "upgrade_cr_target": t_cr,
                "upgrade_to_target": t_to,
                "sr_gap": success_rate - t_sr,
                "cr_gap": t_cr - collision_rate,
                "to_gap": t_to - timeout_rate,
            }

        # ============================================================
        # Bootstrap / 固定 stage 模式的升降級邏輯（原有邏輯）
        # ============================================================
        sr_ok = success_rate > up_sr
        cr_ok = collision_rate < up_cr
        to_ok = timeout_rate < up_to
        dyn_ok = dynamic_sr > up_dyn_sr if (up_dyn_sr > 0 and len(dyn_outcomes) > 0) else True
        all_pass = sr_ok and cr_ok and to_ok and dyn_ok

        # DEBUG: 每 10000 episodes 輸出 bootstrap 狀態
        if state["total_episodes"] % 10000 < env.num_envs:
            print(
                f"[DEBUG Bootstrap] mode={state.get('curriculum_mode')} stage={current_stage}/{MAX_STAGE} "
                f"is_oe={state.get('is_open_ended')} pass={state['upgrade_pass_count']}/{upgrade_pass_required} "
                f"all_pass={all_pass} SR={success_rate:.3f} CR={collision_rate:.3f} TO={timeout_rate:.3f}",
                flush=True,
            )

        if all_pass and current_stage < MAX_STAGE:
            state["upgrade_pass_count"] += 1
            if state["upgrade_pass_count"] >= upgrade_pass_required:
                old = current_stage
                current_stage += 1
                state["stage"] = current_stage
                if state.get("clear_window_on_promote", True):
                    state["outcome_window"].clear()
                state["stage_episodes"] = 0
                state["stage_transitions"] += 1
                state["upgrade_pass_count"] = 0

                # 檢查是否從 bootstrap 最後一級升級到 open-ended
                if state.get("is_open_ended") and current_stage > MAX_STAGE:
                    # 不會到這裡，因為 current_stage < MAX_STAGE 才進 if
                    pass
                elif state.get("is_open_ended") and current_stage == MAX_STAGE:
                    # 完成 bootstrap 最後一級 — 下次升級將進入 open-ended
                    _apply_stage(env, current_stage)
                    _debug_log(f"▲ 升級 (bootstrap)", old, current_stage)
                    stage_cfg = STAGES[current_stage]
                else:
                    _apply_stage(env, current_stage)
                    _debug_log(f"▲ 升級", old, current_stage)
                    stage_cfg = STAGES[current_stage]
            else:
                print(
                    f"[Curriculum {ver}] 升級待確認 "
                    f"{state['upgrade_pass_count']}/{upgrade_pass_required} — "
                    f"SR={success_rate:.3f} CR={collision_rate:.3f} TO={timeout_rate:.3f}",
                    flush=True,
                )
        elif all_pass and current_stage == MAX_STAGE and state.get("is_open_ended"):
            # Bootstrap 最後一級達標 → 進入 open-ended
            # DEBUG: 每 10000 episodes 輸出 MAX_STAGE 分支檢查
            if state["total_episodes"] % 10000 < env.num_envs:
                print(
                    f"[DEBUG MAX_STAGE branch] stage={current_stage}==MAX_STAGE={MAX_STAGE} "
                    f"is_oe={state.get('is_open_ended')} all_pass={all_pass} "
                    f"pass_before={state['upgrade_pass_count']}",
                    flush=True,
                )
            state["upgrade_pass_count"] += 1
            if state["upgrade_pass_count"] >= upgrade_pass_required:
                state["curriculum_mode"] = "open_ended"
                state["difficulty_level"] = 0
                state["outcome_window"].clear()
                state["stage_episodes"] = 0
                state["stage_transitions"] += 1
                state["upgrade_pass_count"] = 0
                cooldown_size = max(2000, env.num_envs * 4)
                state["cooldown_remaining"] = cooldown_size
                _apply_open_ended(env, 0)
                print(
                    f"\n{'='*70}\n"
                    f"[Curriculum {ver}] ★ Bootstrap 完成 → Open-ended Mode (Level 0)\n"
                    f"  SR={success_rate:.3f} CR={collision_rate:.3f} TO={timeout_rate:.3f}\n"
                    f"{'='*70}", flush=True,
                )
                oe_cfg = _open_ended_params(0)
                t_sr = _open_ended_target_sr(0)
                t_cr = _open_ended_target_cr(0)
                t_to = _open_ended_target_to(0)
                return {
                    "stage": float(MAX_STAGE),
                    "stage_name": "open_ended_L0",
                    "success_rate": success_rate,
                    "collision_rate": collision_rate,
                    "timeout_rate": timeout_rate,
                    "dynamic_sr": dynamic_sr,
                    "num_goals": float(oe_cfg["num_goals"]),
                    "num_obstacles_static": float(oe_cfg["num_obstacles_static"]),
                    "num_obstacles_dynamic": float(oe_cfg["num_obstacles_dynamic"]),
                    "min_walls": 0.0, "max_walls": 0.0,
                    "gamma": float(oe_cfg["gamma"]),
                    "episode_length_s": float(oe_cfg["episode_length_s"]),
                    "num_episodes": float(state["total_episodes"]),
                    "stage_episodes": 0.0,
                    "window_fill": 0.0,
                    "upgrade_pass_count": 0.0,
                    "approx_rollout_cycles": 0.0,
                    "upgrade_sr_target": t_sr, "upgrade_cr_target": t_cr, "upgrade_to_target": t_to,
                    "sr_gap": 0.0, "cr_gap": 0.0, "to_gap": 0.0,
                }
        else:
            if state["upgrade_pass_count"] > 0:
                print(
                    f"[Curriculum {ver}] 升級計數歸零 "
                    f"(was {state['upgrade_pass_count']}/{upgrade_pass_required})",
                    flush=True,
                )
            state["upgrade_pass_count"] = 0

            # 降級
            if current_stage > 1:
                reason = None
                if success_rate < down_sr:
                    reason = f"SR={success_rate:.4f}<{down_sr}"
                elif collision_rate > down_cr:
                    reason = f"CR={collision_rate:.4f}>{down_cr}"
                elif timeout_rate > down_to:
                    reason = f"TO={timeout_rate:.4f}>{down_to}"

                if reason is not None:
                    old = current_stage
                    current_stage -= 1
                    state["stage"] = current_stage
                    if state.get("clear_window_on_promote", True):
                        state["outcome_window"].clear()
                    state["stage_episodes"] = 0
                    state["stage_transitions"] += 1
                    state["upgrade_pass_count"] = 0
                    _apply_stage(env, current_stage)
                    _debug_log(f"▼ 降級({reason})", old, current_stage)
                    stage_cfg = STAGES[current_stage]

    # open-ended 模式：回報正確的 stage (= MAX_STAGE + difficulty_level)
    if state.get("curriculum_mode") == "open_ended":
        level = state.get("difficulty_level", 0)
        oe_cfg = _open_ended_params(level)
        t_sr = _open_ended_target_sr(level)
        t_cr = _open_ended_target_cr(level)
        t_to = _open_ended_target_to(level)
        return {
            "stage": float(MAX_STAGE + level),
            "difficulty_level": float(level),
            "stage_name": oe_cfg["name"],
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "timeout_rate": timeout_rate,
            "dynamic_sr": dynamic_sr,
            "num_goals": float(oe_cfg["num_goals"]),
            "num_obstacles_static": float(oe_cfg["num_obstacles_static"]),
            "num_obstacles_dynamic": float(oe_cfg["num_obstacles_dynamic"]),
            "min_walls": 0.0, "max_walls": 0.0,
            "gamma": float(oe_cfg["gamma"]),
            "episode_length_s": float(oe_cfg["episode_length_s"]),
            "num_episodes": float(state["total_episodes"]),
            "stage_episodes": float(state["stage_episodes"]),
            "window_fill": float(len(window)) / float(effective_window),
            "upgrade_pass_count": float(state["upgrade_pass_count"]),
            "approx_rollout_cycles": float(approx_rollout_cycles),
            "upgrade_sr_target": t_sr, "upgrade_cr_target": t_cr, "upgrade_to_target": t_to,
            "sr_gap": success_rate - t_sr,
            "cr_gap": t_cr - collision_rate,
            "to_gap": t_to - timeout_rate,
        }

    up_sr_target = stage_cfg["upgrade_sr"]
    up_cr_target = stage_cfg["upgrade_max_cr"]
    up_to_target = stage_cfg["upgrade_max_to"]

    return {
        "stage": float(current_stage),
        "difficulty_level": 0.0,
        "stage_name": stage_cfg.get("name", ""),
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "timeout_rate": timeout_rate,
        "dynamic_sr": dynamic_sr,
        "num_goals": float(stage_cfg["num_goals"]),
        "num_obstacles_static": float(stage_cfg["num_obstacles_static"]),
        "num_obstacles_dynamic": float(stage_cfg["num_obstacles_dynamic"]),
        "min_walls": float(stage_cfg["min_walls"]),
        "max_walls": float(stage_cfg["max_walls"]),
        "gamma": float(stage_cfg["gamma"]),
        "episode_length_s": float(stage_cfg["episode_length_s"]),
        "num_episodes": float(state["total_episodes"]),
        "stage_episodes": float(state["stage_episodes"]),
        "window_fill": float(len(window)) / float(effective_window),
        "upgrade_pass_count": float(state["upgrade_pass_count"]),
        "approx_rollout_cycles": float(approx_rollout_cycles),
        "upgrade_sr_target": up_sr_target,
        "upgrade_cr_target": up_cr_target,
        "upgrade_to_target": up_to_target,
        "sr_gap": success_rate - up_sr_target,
        "cr_gap": up_cr_target - collision_rate,
        "to_gap": up_to_target - timeout_rate,
        "obs_size_rand": float(stage_cfg.get("obs_size_rand", 0.0)),
        "scene_bound_rand": float(stage_cfg.get("scene_bound_rand", 0.0)),
        "spot_penalty_hit": float(stage_cfg.get("spot_penalty_hit", -5.0)),
        "spot_reward_get_goal": float(stage_cfg.get("spot_reward_get_goal", 40.0)),
        "spot_cost_operate": float(stage_cfg.get("spot_cost_operate", 0.0)),
        "ent_coeff_linear": float(stage_cfg.get("ent_coeff_linear", 0.30)),
        "ent_coeff_angular": float(stage_cfg.get("ent_coeff_angular", 0.375)),
        "obstacle_speed_rate": float(stage_cfg.get("obstacle_speed_rate", 0.8)),
    }


def _stage_label(stage: int) -> str:
    return str(stage)


def _phase_name(stage: int) -> str:
    cfg = STAGES[stage]
    name = cfg.get("name", "")
    g = cfg["num_goals"]
    s = cfg["num_obstacles_static"]
    d = cfg["num_obstacles_dynamic"]
    label = f"{g}G/{s}S+{d}D"
    return f"{name} ({label})" if name else label


def _collect_episode_results(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    state: dict,
):
    """收集 per-env episode 結局：(outcome, env_type)"""
    import torch

    if isinstance(env_ids, torch.Tensor):
        ids = env_ids
    else:
        ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    if ids.numel() == 0:
        return

    difficulty = None
    if hasattr(env, '_env_difficulty') and env._env_difficulty is not None:
        difficulty = env._env_difficulty

    try:
        tm = env.termination_manager
        goal_buf = None
        coll_bufs = []
        for name in tm._term_names:
            if "goal_reached" in name and goal_buf is None:
                goal_buf = tm.get_term(name)
            elif "collision" in name:
                coll_bufs.append(tm.get_term(name))

        for idx in ids:
            eid = idx.item() if isinstance(idx, torch.Tensor) else int(idx)
            if goal_buf is not None and goal_buf[eid].item():
                outcome = 1
            elif any(buf[eid].item() for buf in coll_bufs):
                outcome = -1
            else:
                outcome = 0

            env_type = int(difficulty[eid].item()) if difficulty is not None else -1
            state["outcome_window"].append((outcome, env_type))
            state["total_episodes"] += 1
            state["stage_episodes"] += 1

    except Exception:
        n = ids.numel() if hasattr(ids, 'numel') else len(ids)
        for _ in range(n):
            state["outcome_window"].append((0, -1))
            state["total_episodes"] += 1
            state["stage_episodes"] += 1


def _apply_stage(env: ManagerBasedRLEnv, stage: int):
    """套用指定階段的環境參數 + reward 權重。"""
    cfg = STAGES[stage]

    try:
        cmd = env.command_manager.get_term("goal_command")
        cmd.cfg.num_goals = cfg["num_goals"]
        cmd.cfg.ranges.distance = cfg["goal_distance"]
        cmd.cfg.num_obstacles = cfg["num_obstacles_static"] + cfg["num_obstacles_dynamic"]
    except Exception:
        pass

    try:
        evt = env.event_manager
        for name in ["randomize_obstacles", "randomize_obstacles_startup"]:
            try:
                ec = evt.get_term_cfg(name)
                ec.params["empty_ratio"] = cfg["empty_ratio"]
                ec.params["static_ratio"] = cfg["static_ratio"]
                ec.params["dynamic_ratio"] = cfg["dynamic_ratio"]
                ec.params["mixed_ratio"] = cfg.get("mixed_ratio", 0.0)
                ec.params["num_obstacles_static"] = cfg["num_obstacles_static"]
                ec.params["num_obstacles_dynamic"] = cfg["num_obstacles_dynamic"]
                evt.set_term_cfg(name, ec)
            except Exception:
                continue
        # 同步 env._num_obstacles 讓 move_obstacles for loop 只跑實際用量
        env._num_obstacles = cfg["num_obstacles_static"] + cfg["num_obstacles_dynamic"]
        if hasattr(env, "_obstacle_cache"):
            del env._obstacle_cache
    except Exception:
        pass

    try:
        evt = env.event_manager
        ec = evt.get_term_cfg("randomize_wall_positions")
        ec.params["min_walls"] = cfg["min_walls"]
        ec.params["max_walls"] = cfg["max_walls"]
        evt.set_term_cfg("randomize_wall_positions", ec)
    except Exception:
        pass

    env.cfg.episode_length_s = cfg["episode_length_s"]
    env._target_discount_factor = cfg["gamma"]

    # --- Stage-dependent reward weights ---
    rw = cfg.get("reward_weights")
    if rw is not None:
        # Ablation 5: 套用 safety 權重縮放因子
        rw_scaled = {}
        for term_name, weight in rw.items():
            if term_name in ("static_safety", "dynamic_safety"):
                rw_scaled[term_name] = _apply_ss_scale(weight)
            else:
                rw_scaled[term_name] = weight
        try:
            rm = env.reward_manager
            for term_name, weight in rw_scaled.items():
                try:
                    tc = rm.get_term_cfg(term_name)
                    tc.weight = weight
                    rm.set_term_cfg(term_name, tc)
                except Exception:
                    pass
            weights_str = " ".join(f"{k}={v}" for k, v in rw_scaled.items())
            print(f"[Curriculum] Stage {stage} reward weights: {weights_str}", flush=True)
        except Exception:
            pass


__all__ = ["goal_obstacle_curriculum", "CURRICULUM_CONFIGS", "STAGES", "MAX_STAGE"]
