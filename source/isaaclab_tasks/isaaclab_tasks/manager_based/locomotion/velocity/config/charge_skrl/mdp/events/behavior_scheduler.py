"""BehaviorScheduler — Rule-based obstacle 向量化調度器。

負責:
  1. 根據 curriculum stage 的 behavior_mix 為每個 env 的每個 obstacle 分配行為
  2. 每步向量化 step() 移動所有 obstacle
  3. Episode reset 時重新分配
  4. 產出 WandB 診斷指標
  5. 邊界反彈 + safety constraint 檢查

不含神經網路、不含 gradient、完全 deterministic (given seed)。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import sys
import os
_skrl_dir = os.path.join(os.path.dirname(__file__), "../../../../../../scripts/reinforcement_learning/skrl")
if _skrl_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_skrl_dir))

from obstacle_agent.behavior_config import (
    BehaviorConfig,
    BEHAVIOR_INACTIVE, BEHAVIOR_STATIC, BEHAVIOR_PATROL, BEHAVIOR_RANDOM_WALK,
    BEHAVIOR_HORIZONTAL_CROSSING, BEHAVIOR_PATH_CROSSING,
    BEHAVIOR_NEAR_MISS, BEHAVIOR_CORRIDOR_CROSSING, BEHAVIOR_OCCLUSION,
    BEHAVIOR_NAMES, NUM_BEHAVIOR_TYPES,
)
try:
    from .rule_behaviors import (
        step_static, spawn_static,
        step_patrol, spawn_patrol,
        step_random_walk, spawn_random_walk,
        step_horizontal_crossing, spawn_horizontal_crossing,
        step_path_crossing, spawn_path_crossing,
        step_near_miss, spawn_near_miss,
        step_corridor_crossing, spawn_corridor_crossing,
        step_occlusion, spawn_occlusion,
    )
except ImportError:
    from rule_behaviors import (
        step_static, spawn_static,
        step_patrol, spawn_patrol,
        step_random_walk, spawn_random_walk,
        step_horizontal_crossing, spawn_horizontal_crossing,
        step_path_crossing, spawn_path_crossing,
        step_near_miss, spawn_near_miss,
        step_corridor_crossing, spawn_corridor_crossing,
        step_occlusion, spawn_occlusion,
    )


class BehaviorScheduler:
    """Rule-based obstacle behavior 調度器。

    Usage:
        scheduler = BehaviorScheduler(stage_config, num_envs=512, max_obstacles=10, device="cuda:0")
        scheduler.reset(all_env_ids, env)       # 初始分配
        ...
        scheduler.step(env, dt=0.2)             # 每步呼叫
        metrics = scheduler.get_metrics()       # WandB logging
    """

    def __init__(
        self,
        stage_config: dict,
        num_envs: int,
        max_obstacles: int,
        device: str,
        boundary: float = 8.5,
    ):
        """
        Args:
            stage_config: curriculum stage dict, 必須包含 "behavior_mix" 欄位
            num_envs: 並行環境數
            max_obstacles: 每個 env 最大 obstacle slots
            device: "cuda:0" 等
            boundary: obstacle 活��邊界 (m)
        """
        self.num_envs = num_envs
        self.max_obstacles = max_obstacles
        self.device = device
        self.boundary = boundary

        # 讀取 behavior_mix 和 speed overrides
        self.behavior_mix = stage_config.get("behavior_mix", {"random_walk": 1.0})
        speed_overrides = stage_config.get("speed_overrides", {})
        safety_overrides = stage_config.get("safety_overrides", {})

        # Near-goal obstacle placement (from curriculum config)
        self.obs_near_goal_count = int(stage_config.get("obs_near_goal_count", 0))
        self.obs_near_goal_radius = float(stage_config.get("obs_near_goal_radius", 2.0))

        # 建立 config + apply overrides
        self.cfg = BehaviorConfig()
        all_overrides = {**speed_overrides, **safety_overrides}
        if all_overrides:
            self.cfg.apply_stage_overrides(all_overrides)

        # ════════════════════════════════════���═════════════════════════════
        # Core State Tensors
        # ═══════════════════════════��══════════════════════════════════════

        E, N = num_envs, max_obstacles

        # 行為類型: 0=inactive, 1=static, 2=patrol, ...
        self.behavior_type = torch.zeros(E, N, dtype=torch.long, device=device)
        # XY 位置 (env local frame)
        self.positions = torch.zeros(E, N, 2, device=device)
        # XY 速度
        self.velocities = torch.zeros(E, N, 2, device=device)
        # 各 behavior 的 phase timer (通用計數器)
        self.phase_timer = torch.zeros(E, N, dtype=torch.long, device=device)

        # ���═════════════════════════════���═══════════════════════════════════
        # Patrol State
        # ══════���═══════════════════════════════���═══════════════════════════
        MAX_WAYPOINTS = 4
        self.patrol_waypoints = torch.zeros(E, N, MAX_WAYPOINTS, 2, device=device)
        self.patrol_num_waypoints = torch.zeros(E, N, dtype=torch.long, device=device)
        self.patrol_wp_index = torch.zeros(E, N, dtype=torch.long, device=device)
        self.patrol_speed = torch.zeros(E, N, device=device)
        self.patrol_pause_remaining = torch.zeros(E, N, dtype=torch.long, device=device)

        # ══════════════════════════════════════════════════════════════════
        # Random Walk State
        # ═════════════════��═══════════════════════════════════���════════════
        self.rw_heading = torch.zeros(E, N, device=device)          # current heading (rad)
        self.rw_target_heading = torch.zeros(E, N, device=device)   # target heading (turning to)
        self.rw_speed = torch.zeros(E, N, device=device)
        self.rw_timer = torch.zeros(E, N, dtype=torch.long, device=device)
        self.rw_change_interval = torch.zeros(E, N, dtype=torch.long, device=device)
        self.rw_turn_remaining = torch.zeros(E, N, dtype=torch.long, device=device)

        # ══════════════════════════════════════════════════════════════════
        # Horizontal Crossing State
        # ══════════════════════════════════════════════════════════════════
        self.hc_velocity = torch.zeros(E, N, 2, device=device)
        self.hc_cross_y = torch.zeros(E, N, device=device)
        self.hc_cooldown = torch.zeros(E, N, dtype=torch.long, device=device)
        self.hc_spawn_pos = torch.zeros(E, N, 2, device=device)

        # ══════════════════════════════════════════════════════════════════
        # Path Crossing State
        # ══════════════════════════════════════════════════════════════════
        self.pc_velocity = torch.zeros(E, N, 2, device=device)
        self.pc_spawn_pos = torch.zeros(E, N, 2, device=device)
        self.pc_travel_dist = torch.zeros(E, N, device=device)
        self.pc_activation_delay = torch.zeros(E, N, dtype=torch.long, device=device)
        self.pc_done = torch.zeros(E, N, dtype=torch.bool, device=device)

        # ══════════════════════════════════════════════════════════════════
        # Near-Miss State
        # ══════════════════════════════════════════════════════════════════
        self.nm_velocity = torch.zeros(E, N, 2, device=device)
        self.nm_spawn_pos = torch.zeros(E, N, 2, device=device)
        self.nm_travel_dist = torch.zeros(E, N, device=device)
        self.nm_done = torch.zeros(E, N, dtype=torch.bool, device=device)
        self.nm_clearance = torch.zeros(E, N, device=device)

        # ══════════════════════════════════════════════════════════════════
        # Occlusion State
        # ══════════════════════════════════════════════════════════════════
        self.occ_velocity = torch.zeros(E, N, 2, device=device)
        self.occ_frame_counter = torch.zeros(E, N, dtype=torch.long, device=device)

        # ══════════════════════════════════════════════════════════════════
        # Metrics
        # ══════════════════════════════════════════════════════════════════
        self._collision_by_type = torch.zeros(NUM_BEHAVIOR_TYPES, device=device)
        self._step_count = 0

    # ═══════════════��══════════════════════════════════════════════════════
    # Public API
    # ═════════════════════════════════��════════════════════════���═══════════

    def reset(self, env_ids: Tensor, env: ManagerBasedRLEnv) -> None:
        """Episode reset 時重新分配 behaviors。

        1. 清空指定 env 的所有 state
        2. 根據 behavior_mix 比例分配 behavior type
        3. ��每個 obstacle spawn 到合法位置
        """
        if not isinstance(env_ids, Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        N_envs = len(env_ids)
        if N_envs == 0:
            return

        # 清空 state
        self.behavior_type[env_ids] = BEHAVIOR_INACTIVE
        self.positions[env_ids] = 0.0
        self.velocities[env_ids] = 0.0
        self.phase_timer[env_ids] = 0

        # 計算每種 behavior 分配幾個 slot
        counts = self._allocate_counts(self.behavior_mix, self.max_obstacles)

        # 依序分配 slot — static 排前面，與 mixed_parallel 一致
        # （robot_state.py 用 i < num_static_mixed 判斷 static vs dynamic）
        sorted_names = sorted(counts.keys(),
                              key=lambda n: 0 if n == "static" else 1)
        slot_idx = 0
        for btype_name in sorted_names:
            count = counts[btype_name]
            if count == 0:
                continue

            btype_id = self._name_to_id(btype_name)
            for _ in range(count):
                if slot_idx >= self.max_obstacles:
                    break
                # 設定 behavior type
                self.behavior_type[env_ids, slot_idx] = btype_id
                # Spawn
                self._spawn_single(env_ids, slot_idx, btype_id)
                slot_idx += 1

        # Near-goal placement: 覆寫前 N 個 active slot 位置到 goal 附近
        if self.obs_near_goal_count > 0:
            self._place_near_goal(env_ids, env)

        # Rejection sampling: 確保 safety constraints
        self._enforce_spawn_constraints(env_ids, env)

        # ★ 立刻寫入 sim（避免等到第一個 step() 才顯示）
        self._write_positions_to_sim(env)

        # 診斷: 印出分配結果
        active_count = (self.behavior_type[env_ids] != BEHAVIOR_INACTIVE).sum().item()
        if N_envs <= 4:  # play 模式少 env 才印
            print(f"[BehaviorScheduler] reset {N_envs} envs → {active_count} active obstacles "
                  f"(max_slots={self.max_obstacles}, mix={self.behavior_mix}, "
                  f"near_goal={self.obs_near_goal_count})")

    def step(self, env: ManagerBasedRLEnv, dt: float = 0.2) -> None:
        """每個 env step 呼叫一��，向量化移動��有 obstacle。

        依 behavior_type 用 mask 分組 dispatch 到各 behavior 的 step 函式。
        """
        self._step_count += 1

        # 各 behavior mask: [num_envs, max_obstacles] bool
        static_mask = (self.behavior_type == BEHAVIOR_STATIC)
        patrol_mask = (self.behavior_type == BEHAVIOR_PATROL)
        rw_mask = (self.behavior_type == BEHAVIOR_RANDOM_WALK)

        hc_mask = (self.behavior_type == BEHAVIOR_HORIZONTAL_CROSSING)
        pc_mask = (self.behavior_type == BEHAVIOR_PATH_CROSSING)
        nm_mask = (self.behavior_type == BEHAVIOR_NEAR_MISS)

        # Dispatch
        step_static(self, static_mask, dt)
        step_patrol(self, patrol_mask, dt)
        step_random_walk(self, rw_mask, dt)
        cc_mask = (self.behavior_type == BEHAVIOR_CORRIDOR_CROSSING)
        occ_mask = (self.behavior_type == BEHAVIOR_OCCLUSION)

        step_horizontal_crossing(self, hc_mask, dt)
        step_path_crossing(self, pc_mask, dt)
        step_near_miss(self, nm_mask, dt)
        step_corridor_crossing(self, cc_mask, dt)
        step_occlusion(self, occ_mask, dt)

        # 邊界反彈 (所有 active obstacles)
        active_mask = (self.behavior_type != BEHAVIOR_INACTIVE)
        self._boundary_bounce(active_mask)

        # 牆壁反彈 (含 T 走廊 fill blocks 等 boundary walls)
        self._wall_bounce(active_mask, env, dt)

        # 遞增 phase timer
        self.phase_timer[active_mask] += 1

        # 寫入 Isaac Sim
        self._write_positions_to_sim(env)

    def get_metrics(self) -> dict:
        """產出 WandB 用指標 dict。"""
        result = {}

        # Per-behavior 數量統計
        for btype_id, name in BEHAVIOR_NAMES.items():
            count = (self.behavior_type == btype_id).sum().item()
            result[f"behavior/{name}_count"] = count

        # 平均速度 (active only)
        active = self.behavior_type != BEHAVIOR_INACTIVE
        if active.any():
            speeds = self.velocities[active].norm(dim=-1)
            result["behavior/avg_speed"] = speeds.mean().item()
            result["behavior/speed_std"] = speeds.std().item()
        else:
            result["behavior/avg_speed"] = 0.0
            result["behavior/speed_std"] = 0.0

        # Collision by type (由外部呼叫 record_collision 時累計)
        for btype_id, name in BEHAVIOR_NAMES.items():
            if btype_id == 0:
                continue
            result[f"behavior/collision_{name}"] = self._collision_by_type[btype_id].item()
        self._collision_by_type.zero_()

        return result

    def record_collision(self, env_idx: int, obs_idx: int) -> None:
        """外部碰撞偵測後呼叫，紀錄哪種 behavior 造成碰���。"""
        btype = self.behavior_type[env_idx, obs_idx].item()
        self._collision_by_type[btype] += 1

    def update_stage(self, stage_config: dict) -> None:
        """Curriculum 升階時呼叫，更新 behavior_mix 和 overrides。"""
        self.behavior_mix = stage_config.get("behavior_mix", self.behavior_mix)
        speed_overrides = stage_config.get("speed_overrides", {})
        safety_overrides = stage_config.get("safety_overrides", {})
        all_overrides = {**speed_overrides, **safety_overrides}
        if all_overrides:
            self.cfg.apply_stage_overrides(all_overrides)
        # Near-goal placement params
        if "obs_near_goal_count" in stage_config:
            self.obs_near_goal_count = int(stage_config["obs_near_goal_count"])
        if "obs_near_goal_radius" in stage_config:
            self.obs_near_goal_radius = float(stage_config["obs_near_goal_radius"])

    # ══════════════���════════════════════════���══════════════════════════════
    # Private
    # ═══════════════════════════════���══════════════════════════════════���═══

    def _spawn_single(self, env_ids: Tensor, slot_idx: int, btype_id: int) -> None:
        """根據 behavior type dispatch 到對應 spawn 函式。"""
        slot_ids = torch.full((len(env_ids),), slot_idx, dtype=torch.long, device=self.device)

        if btype_id == BEHAVIOR_STATIC:
            spawn_static(self, env_ids, slot_ids, self.boundary)
        elif btype_id == BEHAVIOR_PATROL:
            spawn_patrol(self, env_ids, slot_ids, self.boundary)
        elif btype_id == BEHAVIOR_RANDOM_WALK:
            spawn_random_walk(self, env_ids, slot_ids, self.boundary)
        elif btype_id == BEHAVIOR_HORIZONTAL_CROSSING:
            spawn_horizontal_crossing(self, env_ids, slot_ids, self.boundary)
        elif btype_id == BEHAVIOR_PATH_CROSSING:
            spawn_path_crossing(self, env_ids, slot_ids, self.boundary)
        elif btype_id == BEHAVIOR_NEAR_MISS:
            spawn_near_miss(self, env_ids, slot_ids, self.boundary)
        elif btype_id == BEHAVIOR_CORRIDOR_CROSSING:
            spawn_corridor_crossing(self, env_ids, slot_ids, self.boundary)
        elif btype_id == BEHAVIOR_OCCLUSION:
            spawn_occlusion(self, env_ids, slot_ids, self.boundary)

    def _allocate_counts(self, behavior_mix: dict, total_slots: int) -> dict:
        """根據 behavior_mix 比例分配 slot 數量。

        Example:
            behavior_mix = {"static": 0.4, "patrol": 0.4, "random_walk": 0.2}
            total_slots = 5
            → {"static": 2, "patrol": 2, "random_walk": 1}
        """
        counts = {}
        remaining = total_slots

        # 按比例分配
        items = list(behavior_mix.items())
        for i, (name, ratio) in enumerate(items):
            if i == len(items) - 1:
                # 最後一個拿剩餘
                counts[name] = remaining
            else:
                n = round(ratio * total_slots)
                n = min(n, remaining)
                counts[name] = n
                remaining -= n

        return counts

    def _name_to_id(self, name: str) -> int:
        """Behavior name → id。"""
        name_to_id = {v: k for k, v in BEHAVIOR_NAMES.items()}
        return name_to_id.get(name, BEHAVIOR_INACTIVE)

    def _boundary_bounce(self, mask: Tensor) -> None:
        """邊界反彈: 超出 boundary 的 obstacle 反射。"""
        b = self.boundary
        pos = self.positions  # [E, N, 2]
        vel = self.velocities

        # X 方向
        over_x = pos[:, :, 0] > b
        under_x = pos[:, :, 0] < -b
        pos[:, :, 0] = torch.where(over_x & mask, 2 * b - pos[:, :, 0], pos[:, :, 0])
        pos[:, :, 0] = torch.where(under_x & mask, -2 * b - pos[:, :, 0], pos[:, :, 0])
        vel[:, :, 0] = torch.where((over_x | under_x) & mask, -vel[:, :, 0], vel[:, :, 0])

        # Y 方向
        over_y = pos[:, :, 1] > b
        under_y = pos[:, :, 1] < -b
        pos[:, :, 1] = torch.where(over_y & mask, 2 * b - pos[:, :, 1], pos[:, :, 1])
        pos[:, :, 1] = torch.where(under_y & mask, -2 * b - pos[:, :, 1], pos[:, :, 1])
        vel[:, :, 1] = torch.where((over_y | under_y) & mask, -vel[:, :, 1], vel[:, :, 1])

        # Final clamp
        pos[:, :, 0].clamp_(-b, b)
        pos[:, :, 1].clamp_(-b, b)

    def _wall_bounce(self, mask: Tensor, env: ManagerBasedRLEnv, dt: float) -> None:
        """牆壁反彈: 碰到 boundary walls / fill blocks 時回退 + 反轉速度。"""
        try:
            from ..wall_layout import get_combined_wall_data
            cw_centers, cw_sizes, cw_mask = get_combined_wall_data(env)
        except Exception:
            return  # 無牆壁資料 → 跳過

        pos = self.positions   # [E, N, 2] local frame
        vel = self.velocities  # [E, N, 2]
        E, N, _ = pos.shape
        W = cw_centers.shape[1]

        # [E, N, 1, 2] vs [E, 1, W, 2]
        pos_exp = pos.unsqueeze(2)                   # [E, N, 1, 2]
        wc = cw_centers[:E].unsqueeze(1)             # [E, 1, W, 2]
        ws = cw_sizes[:E].unsqueeze(1)               # [E, 1, W, 2]
        wm = cw_mask[:E].unsqueeze(1)                # [E, 1, W]

        delta = (pos_exp - wc).abs() - ws * 0.5      # [E, N, W, 2]
        delta = delta.clamp(min=0.0)
        dist = torch.norm(delta, dim=3)               # [E, N, W]
        dist = torch.where(wm, dist, torch.full_like(dist, 1e6))
        in_wall = (dist < 0.3).any(dim=2) & mask     # [E, N]

        if in_wall.any():
            # 回退一步 + 反轉速度
            pos[:, :, 0] = torch.where(in_wall, pos[:, :, 0] - vel[:, :, 0] * dt, pos[:, :, 0])
            pos[:, :, 1] = torch.where(in_wall, pos[:, :, 1] - vel[:, :, 1] * dt, pos[:, :, 1])
            vel[:, :, 0] = torch.where(in_wall, -vel[:, :, 0], vel[:, :, 0])
            vel[:, :, 1] = torch.where(in_wall, -vel[:, :, 1], vel[:, :, 1])

    def _place_near_goal(self, env_ids: Tensor, env: ManagerBasedRLEnv) -> None:
        """將前 obs_near_goal_count 個 active slot 放到最近 goal 附近。

        在 goal 周圍 [min_near_dist, obs_near_goal_radius] 環形區域隨機放置。
        """
        K = len(env_ids)
        n_place = self.obs_near_goal_count
        radius = self.obs_near_goal_radius
        min_near_dist = 0.8  # 不要太貼 goal 中心

        # 取得 goal 位置並轉成 local frame（相對 env origin）
        try:
            goal_cmd = env.command_manager.get_command("goal_command")
            goal_world = goal_cmd[env_ids, :2]  # [K, 2] world frame
            env_origins_xy = env.scene.env_origins[env_ids, :2]  # [K, 2]
            goal_xy = goal_world - env_origins_xy  # → local frame
        except (AttributeError, KeyError, IndexError):
            return  # 沒有 goal command → 跳過

        placed = 0
        for slot_idx in range(self.max_obstacles):
            if placed >= n_place:
                break
            # 只覆寫 active slot
            active = self.behavior_type[env_ids, slot_idx] != BEHAVIOR_INACTIVE
            if not active.any():
                continue

            # 環形隨機：角度 uniform，距離 uniform in [min_near_dist, radius]
            angle = torch.rand(K, device=self.device) * 2.0 * math.pi
            dist = (torch.rand(K, device=self.device)
                    * (radius - min_near_dist) + min_near_dist)
            new_x = (goal_xy[:, 0] + dist * torch.cos(angle)).clamp(-self.boundary, self.boundary)
            new_y = (goal_xy[:, 1] + dist * torch.sin(angle)).clamp(-self.boundary, self.boundary)

            # 只覆寫 active 的 env
            self.positions[env_ids[active], slot_idx, 0] = new_x[active]
            self.positions[env_ids[active], slot_idx, 1] = new_y[active]
            placed += 1

    def _enforce_spawn_constraints(self, env_ids: Tensor, env: ManagerBasedRLEnv) -> None:
        """Spawn 後檢查 safety constraints，不合法的重新採樣。

        檢查:
        - 離 robot 太近
        - 離 goal 太近
        - obstacle 之間太近
        - 離牆壁太近
        """
        # TODO: 實作 rejection sampling (與現有 reset_obstacles 類似邏輯)
        # 暫時跳過 — spawn 位置的 boundary 已經提供基本安全距離
        pass

    def _write_positions_to_sim(self, env: ManagerBasedRLEnv) -> None:
        """將 positions 寫入 Isaac Sim obstacle rigid bodies。"""
        env_origins = env.scene.env_origins  # [E, 3]
        HIDDEN_Z = -10.0
        ACTIVE_Z = 0.85  # obstacle center height

        for slot_idx in range(self.max_obstacles):
            name = f"obstacle_{slot_idx}"
            try:
                obstacle = env.scene[name]
            except KeyError:
                continue

            active = self.behavior_type[:, slot_idx] != BEHAVIOR_INACTIVE  # [E]

            # Build pose [E, 7] = [x, y, z, qw, qx, qy, qz]
            pose = torch.zeros(self.num_envs, 7, device=self.device)
            # Active: local pos → world pos
            pose[:, 0] = torch.where(
                active,
                self.positions[:, slot_idx, 0] + env_origins[:, 0],
                env_origins[:, 0],
            )
            pose[:, 1] = torch.where(
                active,
                self.positions[:, slot_idx, 1] + env_origins[:, 1],
                env_origins[:, 1],
            )
            pose[:, 2] = torch.where(
                active,
                torch.full((self.num_envs,), ACTIVE_Z, device=self.device),
                torch.full((self.num_envs,), HIDDEN_Z, device=self.device),
            )
            pose[:, 3] = 1.0  # identity quaternion (w=1)

            obstacle.write_root_pose_to_sim(pose)
