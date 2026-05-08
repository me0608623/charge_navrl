"""NavRL-Ground V8 — 自包含 reward module（公式 + 權重 + 參數）。

═══════════════════════════════════════════════════════════════
V8 繼承鏈: V1 → V2 → V3 → V6 → V7 → V8
本檔案凍結 V8 的最終有效值，不再依賴 IsaacLab reward_manager。
修改權重/參數 → 改下方 REWARD_TERMS dict 即可。
修改公式 → 覆寫 _compute_* 方法或建新版 V9。
═══════════════════════════════════════════════════════════════

8 項 reward:
  goal_velocity    (w= 2.0)  朝目標速度 × soft gate（無 gate, V2 起）
  goal_progress    (w= 3.0)  PBRS 進度 × soft scale（無 scale, V2 起）
  static_safety    (w= 2.0)  72-bin log clearance + 前向堵塞
  dynamic_safety   (w= 2.0)  per-obstacle log clearance
  smoothness       (w=-0.1)  dv² + dw²（w_coeff=0.6, V7 起）
  time_penalty     (w=-0.2)  每步恆定懲罰（V8 恢復）
  reaching_goal    (w=500.0) 到達目標終端獎勵
  collision_ground (w=-5.0)  碰撞終端懲罰（遞增由 curriculum 控制）

dt 乘法:
  與 IsaacLab reward_manager 一致: reward = func() * weight * dt
  dt = env.step_dt (通常 0.2s = physics_dt 0.01 × decimation 20)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


# ─────────────────────────────────────────────────────────────
# V8 權重 + 參數定義（修改這裡即可調整實驗）
# ─────────────────────────────────────────────────────────────

@dataclass
class RewardTermConfig:
    """單一 reward 項的設定。"""
    weight: float
    params: dict[str, Any] = field(default_factory=dict)


# V8 最終有效值（已展開全部繼承）
REWARD_TERMS: dict[str, RewardTermConfig] = {
    # --- 核心目標項 ---
    "goal_velocity": RewardTermConfig(
        weight=2.0,
        params={
            "body_radius": 0.35,
            "bottom_k": 36,            # V2: 10→36（更穩定 d_safe 估計）
            "v_max": 1.0,
            "min_goal_dist": 0.1,      # V2: 0.5→0.1（消除 0.35~0.5m 死區）
            "use_soft_gate": False,     # V2: 移除 gate（對齊 NavRL 原論文）
            # gate 參數保留但不生效（use_soft_gate=False）
            "gate_beta": 0.15,
            "gate_dmin": 0.6,
            "gate_dmax": 2.0,
            "directional_gate": False,
        },
    ),
    "goal_progress": RewardTermConfig(
        weight=3.0,
        params={
            "body_radius": 0.35,
            "bottom_k": 36,            # V2: 與 goal_velocity 一致
            "progress_clip": 0.25,     # V2: 1.0→0.25（max 0.2m/step, 留 25% margin）
            "use_soft_scale": False,   # V2: 移除 scale（對齊 NavRL 原論文）
            "scale_gamma": 0.15,
            "scale_dmin": 0.5,
            "scale_dmax": 2.0,
            "directional_scale": False,
        },
    ),
    # --- 安全項 ---
    "static_safety": RewardTermConfig(
        weight=2.0,
        params={
            "body_radius": 0.35,
            "a_global": 1.0,
            "a_front_block": 1.0,
            "front_half_angle_deg": 30.0,   # V2: 20→30°（覆蓋側碰風險）
            "front_nearest_k": 5,
            "front_warn_dist": 1.2,
        },
    ),
    "dynamic_safety": RewardTermConfig(
        weight=2.0,
        params={
            "body_radius": 0.35,
            "max_obstacles": 50,
            "mode": "log_distance",
            "risk_sigma": 2.0,              # V2: 1.0→2.0（更平緩衰減）
            "b_log": 1.0,
            "b_risk": 1.0,
        },
    ),
    # --- 平滑 / 時間 ---
    "smoothness": RewardTermConfig(
        weight=-0.1,                        # V7: -0.05→-0.1
        params={
            "smooth_v_coeff": 1.0,
            "smooth_w_coeff": 0.6,          # V7: 1.0→0.6（降低角速度懲罰）
        },
    ),
    "time_penalty": RewardTermConfig(
        weight=-0.2,                        # V8 恢復 V1 的 -0.2（V2~V7 曾歸零）
        params={},
    ),
    # --- 終端 reward ---
    "reaching_goal": RewardTermConfig(
        weight=500.0,                       # V3: 100→500
        params={
            "threshold": 0.35,              # GOAL_REACH_THRESHOLD
            "body_radius": 0.35,
        },
    ),
    "collision_ground": RewardTermConfig(
        weight=-5.0,                        # V7: 遞增由 curriculum 控制（初始 -5）
        params={
            "threshold": 0.3,               # COLLISION_THRESHOLD (LiDAR ≤ 0.3m)
        },
    ),
}


class NavRLDenseV8Reward:
    """NavRL-Ground V8 自包含 reward module。

    外部計算（與 WD sparse 一致），不依賴 IsaacLab reward_manager。
    公式來自 navrl_ground_rewards.py + goal_rewards.py + potential_based_rewards.py。
    """

    name: str = "navrl_dense_v8"

    def __init__(
        self,
        terms: dict[str, RewardTermConfig] | None = None,
    ) -> None:
        self.terms = terms if terms is not None else dict(REWARD_TERMS)
        # Lazy-loaded function references (avoid importing isaaclab before AppLauncher)
        self._funcs: dict[str, Any] | None = None
        self._scene_cfg_cache: dict[str, Any] = {}

    def _init_funcs(self) -> None:
        """Lazy import reward functions (must be called after AppLauncher)."""
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.rewards.navrl_ground_rewards import (
            goal_velocity_reward,
            goal_progress_reward,
            static_safety_reward,
            dynamic_safety_reward,
            control_smoothness_penalty,
        )
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.rewards.goal_rewards import (
            reaching_goal,
        )
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.rewards.potential_based_rewards import (
            collision_terminal_penalty,
            per_step_time_penalty,
        )
        self._funcs = {
            "goal_velocity": goal_velocity_reward,
            "goal_progress": goal_progress_reward,
            "static_safety": static_safety_reward,
            "dynamic_safety": dynamic_safety_reward,
            "smoothness": control_smoothness_penalty,
            "time_penalty": per_step_time_penalty,
            "reaching_goal": reaching_goal,
            "collision_ground": collision_terminal_penalty,
        }

    def _get_scene_cfg(self, name: str) -> object:
        """Create and cache SceneEntityCfg objects."""
        if name not in self._scene_cfg_cache:
            from isaaclab.managers import SceneEntityCfg
            self._scene_cfg_cache[name] = SceneEntityCfg(name)
        return self._scene_cfg_cache[name]

    def _build_call_params(self, term_name: str, params: dict) -> dict:
        """Expand params dict, injecting SceneEntityCfg where needed."""
        call_params = dict(params)

        # Functions that need robot_cfg / sensor_cfg
        needs_robot = {"goal_velocity", "goal_progress", "static_safety", "dynamic_safety", "reaching_goal"}
        needs_sensor = {"goal_velocity", "goal_progress", "static_safety", "collision_ground"}

        if term_name in needs_robot and "robot_cfg" not in call_params:
            call_params["robot_cfg"] = self._get_scene_cfg("robot")
        if term_name == "reaching_goal" and "asset_cfg" not in call_params:
            call_params["asset_cfg"] = self._get_scene_cfg("robot")
            call_params.pop("robot_cfg", None)  # reaching_goal uses asset_cfg, not robot_cfg
        if term_name in needs_sensor and "sensor_cfg" not in call_params:
            call_params["sensor_cfg"] = self._get_scene_cfg("lidar")

        return call_params

    def update_params(self, curriculum_info: dict) -> None:
        """Update reward parameters from curriculum phase config.

        NavRL V8 的 collision_ground weight 可由 curriculum 遞增（-5 → -80）。
        Expected keys:
            spot_penalty_hit → collision_ground weight (取絕對值的負數)
        """
        if "spot_penalty_hit" in curriculum_info:
            penalty = curriculum_info["spot_penalty_hit"]
            if "collision_ground" in self.terms:
                self.terms["collision_ground"] = RewardTermConfig(
                    weight=penalty if penalty <= 0 else -abs(penalty),
                    params=self.terms["collision_ground"].params,
                )

    def compute(
        self,
        env_unwrapped: object,
        actions: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        context: dict | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute NavRL-Ground V8 reward externally.

        Mirrors reward_manager.compute(): reward = Σ(func * weight * dt)

        Returns:
            reward: [N] total reward.
            breakdown: dict of term_name → [N] raw values (before weight × dt).
        """
        if self._funcs is None:
            self._init_funcs()

        N = env_unwrapped.num_envs
        device = env_unwrapped.device
        dt = env_unwrapped.step_dt

        total = torch.zeros(N, device=device)
        breakdown: dict[str, torch.Tensor] = {}

        for term_name, term_cfg in self.terms.items():
            if term_cfg.weight == 0.0:
                breakdown[term_name] = torch.zeros(N, device=device)
                continue

            func = self._funcs[term_name]
            call_params = self._build_call_params(term_name, term_cfg.params)

            raw = func(env_unwrapped, **call_params)             # [N] 原始值
            value = raw * term_cfg.weight * dt                    # 加權 × dt
            total += value

            breakdown[term_name] = raw  # 記錄原始值供 WandB logging

        return total, breakdown
