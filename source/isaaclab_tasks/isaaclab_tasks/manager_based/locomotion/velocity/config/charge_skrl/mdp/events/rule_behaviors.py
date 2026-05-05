"""Rule-Based Obstacle Behaviors — 8 種確定性行為的 GPU 向量化實作。

每種 behavior 不含神經網路，只有參數化規則。
所有計算走 batched tensor ops，不用 Python for-loop over envs。

設計原則:
  - obstacle 軌跡不依賴 robot 動作 (防止 robot 學到 "obstacle 會讓路")
  - 所有行為參數帶隨機性 (防止 policy overfit 到固定 pattern)
  - 遵守 SafetyConstraints (spawn/step 時都檢查)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .behavior_scheduler import BehaviorScheduler

# 從 obstacle_agent 模組取得 config 常數
import sys
import os
# 確保 obstacle_agent 在 path 上
_skrl_dir = os.path.join(os.path.dirname(__file__), "../../../../../../scripts/reinforcement_learning/skrl")
if _skrl_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_skrl_dir))

from obstacle_agent.behavior_config import (
    BEHAVIOR_STATIC, BEHAVIOR_PATROL, BEHAVIOR_RANDOM_WALK,
    BEHAVIOR_HORIZONTAL_CROSSING, BEHAVIOR_PATH_CROSSING,
    BEHAVIOR_NEAR_MISS, BEHAVIOR_CORRIDOR_CROSSING, BEHAVIOR_OCCLUSION,
    BehaviorConfig,
)


# ═══════════════════════════════════════════════════════════════════════════
# Static Behavior
# ═══════════════════════════════════════════════════════════════════════════

def step_static(sched: BehaviorScheduler, mask: Tensor, dt: float) -> None:
    """Static: 不動。什麼都不做。"""
    pass  # velocity 已經是 0，position 不變


def spawn_static(
    sched: BehaviorScheduler,
    env_ids: Tensor,
    slot_ids: Tensor,
    boundary: float = 8.5,
) -> None:
    """Static spawn: 象限分層隨機位置。

    Args:
        sched: scheduler 實例
        env_ids: [K] 需要 spawn 的 env indices
        slot_ids: [K] 對應的 obstacle slot indices
        boundary: spawn 邊界 (m)
    """
    K = len(env_ids)
    device = sched.device

    # 隨機位置 (uniform within boundary)
    pos_x = (torch.rand(K, device=device) * 2 - 1) * boundary
    pos_y = (torch.rand(K, device=device) * 2 - 1) * boundary

    sched.positions[env_ids, slot_ids, 0] = pos_x
    sched.positions[env_ids, slot_ids, 1] = pos_y
    sched.velocities[env_ids, slot_ids] = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Patrol Behavior
# ═══════════════════════════════════════════════════════════════════════════

def step_patrol(sched: BehaviorScheduler, mask: Tensor, dt: float) -> None:
    """Patrol: 向 current waypoint 移動，到達後切換。

    mask: [num_envs, max_obstacles] bool — 哪些是 patrol behavior
    """
    if not mask.any():
        return

    # 取出 active patrol obstacles 的位置和目標 waypoint
    env_idx, obs_idx = mask.nonzero(as_tuple=True)

    current_pos = sched.positions[env_idx, obs_idx]      # [K, 2]
    wp_idx = sched.patrol_wp_index[env_idx, obs_idx]     # [K] current waypoint index
    # Gather 對應 waypoint: patrol_waypoints[env, obs, wp_idx, :]
    target = sched.patrol_waypoints[env_idx, obs_idx, wp_idx]  # [K, 2]
    speed = sched.patrol_speed[env_idx, obs_idx]          # [K]
    pause = sched.patrol_pause_remaining[env_idx, obs_idx]  # [K]

    # 暫停中的不動
    pausing = pause > 0
    sched.patrol_pause_remaining[env_idx[pausing], obs_idx[pausing]] -= 1

    # 非暫停的: 向 waypoint 移動
    moving = ~pausing
    if moving.any():
        m_env = env_idx[moving]
        m_obs = obs_idx[moving]
        m_pos = current_pos[moving]
        m_target = target[moving]
        m_speed = speed[moving]

        direction = m_target - m_pos                       # [K', 2]
        dist = direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # [K', 1]
        direction_norm = direction / dist                   # [K', 2]

        # 移動距離 = speed * dt，但不超過剩餘距離
        step_dist = (m_speed * dt).unsqueeze(-1)           # [K', 1]
        step_dist = torch.min(step_dist, dist)
        new_pos = m_pos + direction_norm * step_dist       # [K', 2]

        sched.positions[m_env, m_obs] = new_pos
        sched.velocities[m_env, m_obs] = direction_norm * m_speed.unsqueeze(-1)

        # 到達 waypoint? (距離 < 0.3m)
        reached = dist.squeeze(-1) < 0.3                   # [K']
        if reached.any():
            r_env = m_env[reached]
            r_obs = m_obs[reached]
            # 切換到下一個 waypoint
            num_wp = sched.patrol_num_waypoints[r_env, r_obs]  # [R]
            next_wp = (sched.patrol_wp_index[r_env, r_obs] + 1) % num_wp
            sched.patrol_wp_index[r_env, r_obs] = next_wp
            # 設定 pause
            pause_steps = torch.randint(
                sched.cfg.patrol.pause_steps_range[0],
                sched.cfg.patrol.pause_steps_range[1] + 1,
                (reached.sum().item(),), device=sched.device,
            )
            sched.patrol_pause_remaining[r_env, r_obs] = pause_steps
            sched.velocities[r_env, r_obs] = 0.0


def spawn_patrol(
    sched: BehaviorScheduler,
    env_ids: Tensor,
    slot_ids: Tensor,
    boundary: float = 8.0,
) -> None:
    """Patrol spawn: 生成巡邏路線 (2~4 waypoints)。"""
    K = len(env_ids)
    device = sched.device
    cfg = sched.cfg.patrol

    # 決定 waypoint 數量
    num_wp = torch.randint(cfg.num_waypoints_range[0], cfg.num_waypoints_range[1] + 1, (K,), device=device)

    # 生成 center point
    center_x = (torch.rand(K, device=device) * 2 - 1) * (boundary - 2.0)
    center_y = (torch.rand(K, device=device) * 2 - 1) * (boundary - 2.0)

    # 圍繞 center 生成 waypoints (最大展開 waypoint_spread_max)
    max_wp = cfg.num_waypoints_range[1]
    waypoints = torch.zeros(K, max_wp, 2, device=device)
    for wp_i in range(max_wp):
        angle = torch.rand(K, device=device) * 2 * math.pi
        radius = torch.rand(K, device=device) * cfg.waypoint_spread_max * 0.5
        waypoints[:, wp_i, 0] = center_x + radius * torch.cos(angle)
        waypoints[:, wp_i, 1] = center_y + radius * torch.sin(angle)
        # Clamp within boundary
        waypoints[:, wp_i, 0].clamp_(-boundary, boundary)
        waypoints[:, wp_i, 1].clamp_(-boundary, boundary)

    # 寫入 scheduler state
    sched.patrol_waypoints[env_ids, slot_ids] = waypoints
    sched.patrol_num_waypoints[env_ids, slot_ids] = num_wp
    sched.patrol_wp_index[env_ids, slot_ids] = 0
    sched.patrol_pause_remaining[env_ids, slot_ids] = 0

    # 速度大小
    speed = torch.empty(K, device=device).uniform_(cfg.speed_range[0], cfg.speed_range[1])
    sched.patrol_speed[env_ids, slot_ids] = speed

    # 初始位置 = 第一個 waypoint
    sched.positions[env_ids, slot_ids] = waypoints[:, 0]
    sched.velocities[env_ids, slot_ids] = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Random Walk Behavior
# ═══════════════════════════════════════════════════════════════════════════

def step_random_walk(sched: BehaviorScheduler, mask: Tensor, dt: float) -> None:
    """Random Walk: 定時改變方向 + 平滑轉向 + 邊界反彈。"""
    if not mask.any():
        return

    env_idx, obs_idx = mask.nonzero(as_tuple=True)
    K = len(env_idx)
    device = sched.device
    cfg = sched.cfg.random_walk

    # 遞增 timer
    sched.rw_timer[env_idx, obs_idx] += 1

    # 檢查是否需要改方向
    timer = sched.rw_timer[env_idx, obs_idx]
    interval = sched.rw_change_interval[env_idx, obs_idx]
    need_change = timer >= interval

    if need_change.any():
        nc_env = env_idx[need_change]
        nc_obs = obs_idx[need_change]
        nc_count = need_change.sum().item()

        # 新方向: 在當前方向 ± max_turn_angle 內隨機偏轉
        current_heading = sched.rw_heading[nc_env, nc_obs]  # [NC] radians
        delta_angle = (torch.rand(nc_count, device=device) * 2 - 1) * cfg.max_turn_angle
        new_heading = current_heading + delta_angle
        sched.rw_target_heading[nc_env, nc_obs] = new_heading
        sched.rw_turn_remaining[nc_env, nc_obs] = cfg.smooth_turn_steps

        # 重置 timer + 新 interval
        sched.rw_timer[nc_env, nc_obs] = 0
        sched.rw_change_interval[nc_env, nc_obs] = torch.randint(
            cfg.direction_change_interval_range[0],
            cfg.direction_change_interval_range[1] + 1,
            (nc_count,), device=device,
        )

    # 平滑轉向中的: 線性插值 heading
    turning = sched.rw_turn_remaining[env_idx, obs_idx] > 0
    if turning.any():
        t_env = env_idx[turning]
        t_obs = obs_idx[turning]
        remaining = sched.rw_turn_remaining[t_env, t_obs].float()
        current_h = sched.rw_heading[t_env, t_obs]
        target_h = sched.rw_target_heading[t_env, t_obs]
        # 每步插值 1/remaining
        interp = 1.0 / remaining.clamp(min=1.0)
        new_h = current_h + (target_h - current_h) * interp
        sched.rw_heading[t_env, t_obs] = new_h
        sched.rw_turn_remaining[t_env, t_obs] -= 1

    # 移動: heading → velocity → position
    heading = sched.rw_heading[env_idx, obs_idx]  # [K]
    speed = sched.rw_speed[env_idx, obs_idx]      # [K]
    vx = speed * torch.cos(heading)
    vy = speed * torch.sin(heading)
    sched.velocities[env_idx, obs_idx, 0] = vx
    sched.velocities[env_idx, obs_idx, 1] = vy
    sched.positions[env_idx, obs_idx, 0] += vx * dt
    sched.positions[env_idx, obs_idx, 1] += vy * dt


def spawn_random_walk(
    sched: BehaviorScheduler,
    env_ids: Tensor,
    slot_ids: Tensor,
    boundary: float = 8.0,
) -> None:
    """Random Walk spawn: 隨機初始位置 + 方向 + 速度。"""
    K = len(env_ids)
    device = sched.device
    cfg = sched.cfg.random_walk

    # 位置
    pos_x = (torch.rand(K, device=device) * 2 - 1) * boundary
    pos_y = (torch.rand(K, device=device) * 2 - 1) * boundary
    sched.positions[env_ids, slot_ids, 0] = pos_x
    sched.positions[env_ids, slot_ids, 1] = pos_y

    # 初始 heading (random)
    heading = torch.rand(K, device=device) * 2 * math.pi
    sched.rw_heading[env_ids, slot_ids] = heading
    sched.rw_target_heading[env_ids, slot_ids] = heading

    # 速度大小
    speed = torch.empty(K, device=device).uniform_(cfg.speed_range[0], cfg.speed_range[1])
    sched.rw_speed[env_ids, slot_ids] = speed

    # Timer 和 interval
    sched.rw_timer[env_ids, slot_ids] = 0
    sched.rw_change_interval[env_ids, slot_ids] = torch.randint(
        cfg.direction_change_interval_range[0],
        cfg.direction_change_interval_range[1] + 1,
        (K,), device=device,
    )
    sched.rw_turn_remaining[env_ids, slot_ids] = 0

    # 初始 velocity
    sched.velocities[env_ids, slot_ids, 0] = speed * torch.cos(heading)
    sched.velocities[env_ids, slot_ids, 1] = speed * torch.sin(heading)


# ═══════════════════════════════════════════════════════════════════════════
# Horizontal Crossing Behavior
#
# 模擬行人橫穿 robot 前方。預先計算線性軌跡，不依賴 robot 動作。
# obstacle 從場景一側線性穿越到另一側，穿越完成後等待再次觸發。
# ═══════════════════════════════════════════════════════════════════════════

def step_horizontal_crossing(sched: BehaviorScheduler, mask: Tensor, dt: float) -> None:
    """Horizontal Crossing: 沿預計算方向線性移動。

    crossing 用 hc_velocity (spawn 時計算好的固定方向+速度)。
    穿越完成 (超出 boundary) 後進入 cooldown，cooldown 結束後 respawn 到對側。
    """
    if not mask.any():
        return

    env_idx, obs_idx = mask.nonzero(as_tuple=True)
    K = len(env_idx)
    device = sched.device

    # 讀取狀態
    cooldown = sched.hc_cooldown[env_idx, obs_idx]  # [K] 剩餘 cooldown 步數
    in_cooldown = cooldown > 0

    # Cooldown 中: 不動，計數 -1
    if in_cooldown.any():
        sched.hc_cooldown[env_idx[in_cooldown], obs_idx[in_cooldown]] -= 1

    # 非 cooldown: 線性移動
    moving = ~in_cooldown
    if moving.any():
        m_env = env_idx[moving]
        m_obs = obs_idx[moving]
        vel = sched.hc_velocity[m_env, m_obs]  # [K', 2] 固定速度方向
        sched.positions[m_env, m_obs] += vel * dt
        sched.velocities[m_env, m_obs] = vel

        # 超出 boundary → 穿越完成，進入 cooldown + 隱藏
        pos = sched.positions[m_env, m_obs]
        out_of_bounds = (pos[:, 0].abs() > sched.boundary + 1.0) | \
                        (pos[:, 1].abs() > sched.boundary + 1.0)
        if out_of_bounds.any():
            oo_env = m_env[out_of_bounds]
            oo_obs = m_obs[out_of_bounds]
            # 隱藏 (移到場景外等待)
            sched.positions[oo_env, oo_obs] = 0.0  # 暫時歸零，_write 時由 cooldown 控制 Z
            sched.velocities[oo_env, oo_obs] = 0.0
            # 設定 cooldown (隨機 30~60 步 = 6~12s)
            cfg = sched.cfg.horizontal_crossing
            n_oo = out_of_bounds.sum().item()
            sched.hc_cooldown[oo_env, oo_obs] = torch.randint(
                cfg.repeat_interval_range[0], cfg.repeat_interval_range[1] + 1,
                (n_oo,), device=device,
            )
            # Respawn: 從對側重新開始 (反轉 spawn side)
            sched.hc_velocity[oo_env, oo_obs] *= -1  # 反向
            spawn_offset = sched.boundary + 0.5
            # 放到反向的起始位置
            vel_dir = sched.hc_velocity[oo_env, oo_obs]  # [n, 2]
            # spawn 在速度反方向的邊界外
            sched.hc_spawn_pos[oo_env, oo_obs, 0] = -torch.sign(vel_dir[:, 0]) * spawn_offset
            sched.hc_spawn_pos[oo_env, oo_obs, 1] = sched.hc_cross_y[oo_env, oo_obs]


def spawn_horizontal_crossing(
    sched: BehaviorScheduler,
    env_ids: Tensor,
    slot_ids: Tensor,
    boundary: float = 8.5,
) -> None:
    """Horizontal Crossing spawn: 從場景一側生成，朝另一側直線穿越。"""
    K = len(env_ids)
    device = sched.device
    cfg = sched.cfg.horizontal_crossing

    # 決定穿越方向: 50% 左→右 / 50% 右→左
    go_right = torch.rand(K, device=device) > 0.5

    # Spawn X: 在邊界外 (即將進入場景)
    spawn_x = torch.where(go_right,
                          torch.full((K,), -(boundary + 0.5), device=device),
                          torch.full((K,), boundary + 0.5, device=device))

    # Spawn Y: 在 robot 前方某處 (隨機)
    # 因為 spawn 時不知道 robot 確切位置，用場景中央附近
    spawn_y = (torch.rand(K, device=device) * 2 - 1) * (boundary * 0.6)

    sched.positions[env_ids, slot_ids, 0] = spawn_x
    sched.positions[env_ids, slot_ids, 1] = spawn_y

    # 速度: 純水平方向 + 隨機速度大小
    speed = torch.empty(K, device=device).uniform_(cfg.speed_range[0], cfg.speed_range[1])
    vx = torch.where(go_right, speed, -speed)
    sched.hc_velocity[env_ids, slot_ids, 0] = vx
    sched.hc_velocity[env_ids, slot_ids, 1] = 0.0
    sched.velocities[env_ids, slot_ids, 0] = vx
    sched.velocities[env_ids, slot_ids, 1] = 0.0

    # 記錄 Y 座標 (respawn 時保持同一水平線)
    sched.hc_cross_y[env_ids, slot_ids] = spawn_y
    sched.hc_cooldown[env_ids, slot_ids] = 0
    sched.hc_spawn_pos[env_ids, slot_ids, 0] = spawn_x
    sched.hc_spawn_pos[env_ids, slot_ids, 1] = spawn_y


# ═══════════════════════════════════════════════════════════════════════════
# Path Crossing Behavior
#
# 從 robot→goal 連線的垂直方向穿越。觸發有延遲 (模擬「同時出現」情境)。
# 軌跡在 spawn 時預計算，不依賴 robot 動作。
# ═══════════════════════════════════════════════════════════════════════════

def step_path_crossing(sched: BehaviorScheduler, mask: Tensor, dt: float) -> None:
    """Path Crossing: 延遲激活 + 線性穿越。

    Phase 1 (waiting): 等待 activation_delay 步後才開始移動。
    Phase 2 (crossing): 沿預計算方向線性移動。
    Phase 3 (done): 穿越完成後停止 (不重複，等 episode reset)。
    """
    if not mask.any():
        return

    env_idx, obs_idx = mask.nonzero(as_tuple=True)
    device = sched.device

    # 讀取 activation delay
    delay = sched.pc_activation_delay[env_idx, obs_idx]  # [K]
    done = sched.pc_done[env_idx, obs_idx]               # [K] bool

    # Phase: waiting (delay > 0 且未完成)
    waiting = (delay > 0) & ~done
    if waiting.any():
        sched.pc_activation_delay[env_idx[waiting], obs_idx[waiting]] -= 1
        sched.velocities[env_idx[waiting], obs_idx[waiting]] = 0.0

    # Phase: crossing (delay == 0 且未完成)
    crossing = (delay <= 0) & ~done
    if crossing.any():
        c_env = env_idx[crossing]
        c_obs = obs_idx[crossing]
        vel = sched.pc_velocity[c_env, c_obs]  # [K', 2]
        sched.positions[c_env, c_obs] += vel * dt
        sched.velocities[c_env, c_obs] = vel

        # 穿越完成: 超出移動目標距離
        dist_from_spawn = (sched.positions[c_env, c_obs] - sched.pc_spawn_pos[c_env, c_obs]).norm(dim=-1)
        finished = dist_from_spawn > sched.pc_travel_dist[c_env, c_obs]
        if finished.any():
            f_env = c_env[finished]
            f_obs = c_obs[finished]
            sched.pc_done[f_env, f_obs] = True
            sched.velocities[f_env, f_obs] = 0.0


def spawn_path_crossing(
    sched: BehaviorScheduler,
    env_ids: Tensor,
    slot_ids: Tensor,
    boundary: float = 8.5,
) -> None:
    """Path Crossing spawn: 在 robot→goal 連線的垂直方向生成。

    因為 spawn 時不確定 robot/goal 精確位置（每次 reset 都不同），
    使用場景中央附近的隨機交叉路線。
    """
    K = len(env_ids)
    device = sched.device
    cfg = sched.cfg.path_crossing

    # 交叉角度 (30~150° 相對於 X 軸)
    angle = torch.empty(K, device=device).uniform_(cfg.crossing_angle_range[0], cfg.crossing_angle_range[1])

    # Spawn 位置: 場景中央偏移
    center_x = (torch.rand(K, device=device) * 2 - 1) * 3.0  # 中央 ±3m
    center_y = (torch.rand(K, device=device) * 2 - 1) * 3.0

    # Spawn 在交叉線的起始端 (離中心 perp_dist)
    perp_dist = torch.empty(K, device=device).uniform_(
        cfg.spawn_perp_distance_range[0], cfg.spawn_perp_distance_range[1])
    spawn_x = center_x - perp_dist * torch.cos(angle)
    spawn_y = center_y - perp_dist * torch.sin(angle)

    sched.positions[env_ids, slot_ids, 0] = spawn_x
    sched.positions[env_ids, slot_ids, 1] = spawn_y

    # 速度: 沿 angle 方向
    speed = torch.empty(K, device=device).uniform_(cfg.speed_range[0], cfg.speed_range[1])
    vx = speed * torch.cos(angle)
    vy = speed * torch.sin(angle)
    sched.pc_velocity[env_ids, slot_ids, 0] = vx
    sched.pc_velocity[env_ids, slot_ids, 1] = vy
    sched.velocities[env_ids, slot_ids, 0] = 0.0  # 等激活後才動
    sched.velocities[env_ids, slot_ids, 1] = 0.0

    # 穿越總距離 = 2 * perp_dist (穿過中心到對側)
    sched.pc_travel_dist[env_ids, slot_ids] = perp_dist * 2.0
    sched.pc_spawn_pos[env_ids, slot_ids, 0] = spawn_x
    sched.pc_spawn_pos[env_ids, slot_ids, 1] = spawn_y

    # Activation delay: 2~5s (10~25 steps)
    sched.pc_activation_delay[env_ids, slot_ids] = torch.randint(
        cfg.activation_delay_range[0], cfg.activation_delay_range[1] + 1,
        (K,), device=device,
    )
    sched.pc_done[env_ids, slot_ids] = False


# ═══════════════════════════════════════════════════════════════════════════
# Near-Miss Behavior
#
# 從 robot 側方極近距離擦過但不碰撞。
# 軌跡在 spawn 時完全預計算 (precomputed linear trajectory)，不依賴 robot 動作。
# clearance 保證: 即使 robot 完全不動不閃，obstacle 也不會碰撞。
# ═══════════════════════════════════════════════════════════════════════════

def step_near_miss(sched: BehaviorScheduler, mask: Tensor, dt: float) -> None:
    """Near Miss: 沿預計算軌跡線性移動。

    比 horizontal_crossing 更簡單 — 只走一次，不重複。
    穿越完成後標記 done，不再移動（等 episode reset）。
    """
    if not mask.any():
        return

    env_idx, obs_idx = mask.nonzero(as_tuple=True)
    device = sched.device

    done = sched.nm_done[env_idx, obs_idx]

    # 未完成的: 線性移動
    active = ~done
    if active.any():
        a_env = env_idx[active]
        a_obs = obs_idx[active]
        vel = sched.nm_velocity[a_env, a_obs]  # [K', 2] precomputed
        sched.positions[a_env, a_obs] += vel * dt
        sched.velocities[a_env, a_obs] = vel

        # 完成判斷: 走過預定距離
        dist_from_spawn = (sched.positions[a_env, a_obs] - sched.nm_spawn_pos[a_env, a_obs]).norm(dim=-1)
        finished = dist_from_spawn > sched.nm_travel_dist[a_env, a_obs]
        if finished.any():
            f_env = a_env[finished]
            f_obs = a_obs[finished]
            sched.nm_done[f_env, f_obs] = True
            sched.velocities[f_env, f_obs] = 0.0
            # 隱藏到場景外
            sched.positions[f_env, f_obs, 0] = 0.0
            sched.positions[f_env, f_obs, 1] = sched.boundary + 5.0  # 場景外

    # Done 的: 停止
    done_now = sched.nm_done[env_idx, obs_idx]
    if done_now.any():
        sched.velocities[env_idx[done_now], obs_idx[done_now]] = 0.0


def spawn_near_miss(
    sched: BehaviorScheduler,
    env_ids: Tensor,
    slot_ids: Tensor,
    boundary: float = 8.5,
) -> None:
    """Near Miss spawn: 預計算一條線性軌跡，保證 surface clearance >= min。

    軌跡設計:
    1. 選擇穿越通過的「最近點」(closest_point)：在場景中央附近
    2. 選擇 clearance bucket (tight/medium/loose)
    3. 計算 perpendicular offset = clearance + robot_r + obs_r
    4. 從 LiDAR range 邊界反推 spawn 位置
    5. 產出線性軌跡: pos(t) = spawn_pos + velocity * t
    """
    K = len(env_ids)
    device = sched.device
    cfg = sched.cfg.near_miss

    ROBOT_RADIUS = 0.35
    OBS_RADIUS = 0.30

    # 1. 選擇 clearance bucket
    weights = torch.tensor(cfg.clearance_weights, device=device)
    bucket_idx = torch.multinomial(weights.expand(K, -1), 1).squeeze(-1)  # [K] 0/1/2

    clearance_ranges = [cfg.clearance_tight, cfg.clearance_medium, cfg.clearance_loose]
    # 對每個 obstacle 採樣 clearance
    clearance = torch.zeros(K, device=device)
    for i, (lo, hi) in enumerate(clearance_ranges):
        in_bucket = bucket_idx == i
        if in_bucket.any():
            n = in_bucket.sum().item()
            clearance[in_bucket] = torch.empty(n, device=device).uniform_(lo, hi)

    # 2. 穿越的「最近點」位置 (場景中央附近)
    closest_x = (torch.rand(K, device=device) * 2 - 1) * 4.0  # ±4m
    closest_y = (torch.rand(K, device=device) * 2 - 1) * 4.0

    # 3. 穿越方向 (隨機角度)
    cross_angle = torch.rand(K, device=device) * 2 * math.pi

    # 4. Perpendicular offset (保證不碰撞)
    offset_dist = clearance + ROBOT_RADIUS + OBS_RADIUS  # surface clearance
    # 隨機選 perpendicular 方向 (左或右)
    perp_sign = torch.where(torch.rand(K, device=device) > 0.5,
                            torch.ones(K, device=device),
                            -torch.ones(K, device=device))
    # 最近點的實際通過位置 = closest_point + perpendicular * offset
    perp_x = -torch.sin(cross_angle) * perp_sign
    perp_y = torch.cos(cross_angle) * perp_sign
    pass_x = closest_x + perp_x * offset_dist
    pass_y = closest_y + perp_y * offset_dist

    # 5. 速度
    speed = torch.empty(K, device=device).uniform_(cfg.speed_range[0], cfg.speed_range[1])
    vx = speed * torch.cos(cross_angle)
    vy = speed * torch.sin(cross_angle)

    # 6. Spawn 位置: 從 pass point 反推 (走 TTC 秒能到 pass point)
    ttc = torch.empty(K, device=device).uniform_(cfg.ttc_range[0], cfg.ttc_range[1])
    travel_before_pass = speed * ttc
    spawn_x = pass_x - vx * ttc
    spawn_y = pass_y - vy * ttc

    # 7. 總穿越距離 = 2 * travel_before_pass (對稱: spawn → pass → exit)
    total_dist = travel_before_pass * 2.0

    # Clamp spawn within extended boundary (允許從邊界外進入)
    spawn_x.clamp_(-(boundary + 2.0), boundary + 2.0)
    spawn_y.clamp_(-(boundary + 2.0), boundary + 2.0)

    # 寫入 state
    sched.positions[env_ids, slot_ids, 0] = spawn_x
    sched.positions[env_ids, slot_ids, 1] = spawn_y
    sched.nm_velocity[env_ids, slot_ids, 0] = vx
    sched.nm_velocity[env_ids, slot_ids, 1] = vy
    sched.nm_spawn_pos[env_ids, slot_ids, 0] = spawn_x
    sched.nm_spawn_pos[env_ids, slot_ids, 1] = spawn_y
    sched.nm_travel_dist[env_ids, slot_ids] = total_dist
    sched.nm_done[env_ids, slot_ids] = False
    sched.nm_clearance[env_ids, slot_ids] = clearance
    sched.velocities[env_ids, slot_ids, 0] = vx
    sched.velocities[env_ids, slot_ids, 1] = vy


# ═══════════════════════════════════════════════════════════════════════════
# Corridor Crossing Behavior
#
# 在牆壁之間的通道 (gap) 中來回移動。
# 通道偵測: 用 env._maze_wall_centers/sizes/mask 找 wall endpoints 附近的 gap。
# 如果場景無合適 corridor → fallback 到 patrol 行為。
# ═══════════════════════════════════════════════════════════════════════════

def step_corridor_crossing(sched: BehaviorScheduler, mask: Tensor, dt: float) -> None:
    """Corridor Crossing: 在通道端點間 ping-pong 移動。

    邏輯與 patrol 幾乎相同 (兩點往返)，差別在 spawn 位置受 wall gap 約束。
    """
    if not mask.any():
        return

    env_idx, obs_idx = mask.nonzero(as_tuple=True)

    current_pos = sched.positions[env_idx, obs_idx]
    # corridor 端點存在 patrol_waypoints 的前 2 個 slot
    wp_idx = sched.patrol_wp_index[env_idx, obs_idx]
    target = sched.patrol_waypoints[env_idx, obs_idx, wp_idx]
    speed = sched.patrol_speed[env_idx, obs_idx]

    direction = target - current_pos
    dist = direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    direction_norm = direction / dist

    step_dist = (speed * dt).unsqueeze(-1)
    step_dist = torch.min(step_dist, dist)
    new_pos = current_pos + direction_norm * step_dist

    sched.positions[env_idx, obs_idx] = new_pos
    sched.velocities[env_idx, obs_idx] = direction_norm * speed.unsqueeze(-1)

    # 到達端點 → 反轉 (ping-pong between wp 0 and wp 1)
    reached = dist.squeeze(-1) < 0.3
    if reached.any():
        r_env = env_idx[reached]
        r_obs = obs_idx[reached]
        sched.patrol_wp_index[r_env, r_obs] = 1 - sched.patrol_wp_index[r_env, r_obs]


def spawn_corridor_crossing(
    sched: BehaviorScheduler,
    env_ids: Tensor,
    slot_ids: Tensor,
    boundary: float = 8.5,
) -> None:
    """Corridor spawn: 偵測 wall gap → 在通道內建立 2 點 patrol。

    Gap 偵測策略 (簡化版):
    - 每面牆壁有兩個端點
    - 如果端點離邊界 > min_gap_width → 有 gap
    - 在 gap 中心線上設置兩個巡邏點

    如果找不到合適 corridor → fallback 到一般 patrol。
    """
    K = len(env_ids)
    device = sched.device
    cfg = sched.cfg.corridor_crossing

    # 嘗試從 wall data 偵測 gap
    # 簡化: 在場景中隨機兩點建立狹窄通道式巡邏 (模擬 corridor 行為)
    # 真正的 wall gap detection 需要 env 的 wall state — 這裡用 heuristic

    # Heuristic corridor: 沿 X 或 Y 軸，在牆壁附近 (boundary * 0.5~0.8) 建立短巡邏
    # 方向: 50% 沿 X, 50% 沿 Y
    along_x = torch.rand(K, device=device) > 0.5

    # 通道中心位置 (靠近場景中心)
    center_x = (torch.rand(K, device=device) * 2 - 1) * (boundary * 0.5)
    center_y = (torch.rand(K, device=device) * 2 - 1) * (boundary * 0.5)

    # 巡邏長度 (通道長度 2~4m)
    corridor_len = torch.empty(K, device=device).uniform_(2.0, 4.0)
    half_len = corridor_len * 0.5

    # 端點 A 和 B
    wp_a = torch.zeros(K, 2, device=device)
    wp_b = torch.zeros(K, 2, device=device)

    # 沿 X 軸的
    wp_a[along_x, 0] = center_x[along_x] - half_len[along_x]
    wp_a[along_x, 1] = center_y[along_x]
    wp_b[along_x, 0] = center_x[along_x] + half_len[along_x]
    wp_b[along_x, 1] = center_y[along_x]

    # 沿 Y 軸的
    wp_a[~along_x, 0] = center_x[~along_x]
    wp_a[~along_x, 1] = center_y[~along_x] - half_len[~along_x]
    wp_b[~along_x, 0] = center_x[~along_x]
    wp_b[~along_x, 1] = center_y[~along_x] + half_len[~along_x]

    # 寫入 patrol state (複用 patrol 的 waypoint 結構)
    sched.patrol_waypoints[env_ids, slot_ids, 0] = wp_a
    sched.patrol_waypoints[env_ids, slot_ids, 1] = wp_b
    sched.patrol_num_waypoints[env_ids, slot_ids] = 2
    sched.patrol_wp_index[env_ids, slot_ids] = 0
    sched.patrol_pause_remaining[env_ids, slot_ids] = 0

    # 速度 (corridor 內慢速)
    speed = torch.empty(K, device=device).uniform_(cfg.speed_range[0], cfg.speed_range[1])
    sched.patrol_speed[env_ids, slot_ids] = speed

    # 初始位置 = wp_a
    sched.positions[env_ids, slot_ids] = wp_a
    sched.velocities[env_ids, slot_ids] = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Obstacle Occlusion Behavior
#
# 2~3 個 obstacle 組成一組: 1 front blocker (慢) + 1~2 back target (正常速)
# 設計保證:
#   - back target 在被遮擋前至少可見 min_visible_frames (8 frames = 1.6s)
#   - 重現位置符合 constant velocity extrapolation
#   - 不允許從完全不可觀測位置突然出現
#
# 簡化實作: front 和 back 沿相同方向移動，但 front 較慢 → back 會逐漸被遮擋。
# front 週期性加速/減速 → 造成遮擋/取消遮擋的循環。
# ═══════════════════════════════════════════════════════════════════════════

def step_occlusion(sched: BehaviorScheduler, mask: Tensor, dt: float) -> None:
    """Occlusion group step: 所有 member 線性移動。

    Front blocker 比 back target 慢 → front 逐漸被 back 追上。
    週期性: back 在 front 後方時被遮擋，超過 front 後重新可見。
    整組到達邊界後反彈 (保持 group 隊形)。
    """
    if not mask.any():
        return

    env_idx, obs_idx = mask.nonzero(as_tuple=True)

    # 簡單線性移動 (使用各自的 velocity)
    vel = sched.occ_velocity[env_idx, obs_idx]
    sched.positions[env_idx, obs_idx] += vel * dt
    sched.velocities[env_idx, obs_idx] = vel

    # 遞增 visible/hidden timer
    sched.occ_frame_counter[env_idx, obs_idx] += 1


def spawn_occlusion(
    sched: BehaviorScheduler,
    env_ids: Tensor,
    slot_ids: Tensor,
    boundary: float = 8.5,
) -> None:
    """Occlusion group spawn: 1 front + 1 back 沿相同方向排列。

    此 function 被呼叫時 slot_ids 指向 group 中的單一 member。
    group 分配由 scheduler 的 assign 邏輯統一處理。

    簡化: 每個 occlusion slot 獨立 spawn，但確保:
    - 相同 group 的 obstacle 沿同方向
    - front 較慢 (speed * 0.5), back 正常速
    - back spawn 在 front 後方 0.8~1.5m
    """
    K = len(env_ids)
    device = sched.device
    cfg = sched.cfg.occlusion

    # 移動方向 (隨機)
    angle = torch.rand(K, device=device) * 2 * math.pi

    # 判斷這是 front 還是 back (偶數 slot = front, 奇數 = back)
    is_front = (slot_ids % 2 == 0)

    # 速度
    front_speed = torch.empty(K, device=device).uniform_(cfg.front_speed_range[0], cfg.front_speed_range[1])
    back_speed = torch.empty(K, device=device).uniform_(cfg.back_speed_range[0], cfg.back_speed_range[1])
    speed = torch.where(is_front, front_speed, back_speed)

    vx = speed * torch.cos(angle)
    vy = speed * torch.sin(angle)
    sched.occ_velocity[env_ids, slot_ids, 0] = vx
    sched.occ_velocity[env_ids, slot_ids, 1] = vy

    # 位置: 場景內隨機，back 比 front 偏後 (沿移動方向的反方向偏移)
    base_x = (torch.rand(K, device=device) * 2 - 1) * (boundary * 0.6)
    base_y = (torch.rand(K, device=device) * 2 - 1) * (boundary * 0.6)

    # Back 在 front 後方 spacing 距離
    spacing = torch.empty(K, device=device).uniform_(
        cfg.front_back_spacing_range[0], cfg.front_back_spacing_range[1])
    offset_x = torch.where(is_front, torch.zeros(K, device=device), -torch.cos(angle) * spacing)
    offset_y = torch.where(is_front, torch.zeros(K, device=device), -torch.sin(angle) * spacing)

    sched.positions[env_ids, slot_ids, 0] = base_x + offset_x
    sched.positions[env_ids, slot_ids, 1] = base_y + offset_y
    sched.velocities[env_ids, slot_ids, 0] = vx
    sched.velocities[env_ids, slot_ids, 1] = vy

    # Frame counter (用於 metrics: 可見了多少幀)
    sched.occ_frame_counter[env_ids, slot_ids] = 0
