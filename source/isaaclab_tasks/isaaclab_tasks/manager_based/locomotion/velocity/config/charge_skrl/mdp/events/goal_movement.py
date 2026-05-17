"""goal_movement.py — Goal 隨機移動 EventTerm

在訓練期間讓 goal 緩慢移動，模擬真實環境中 target 不固定的情境。

移動行為：
  - random_walk:  隨機方向 + 平滑轉向，最常用
  - drift:        緩慢線性飄移，方向變化少
  - patrol:       在初始位置附近的隨機 waypoint 間巡邏

安全保證：
  - 不穿過牆壁（AABB 碰撞 + 反彈）
  - 不進入障礙物範圍（距離檢測 + 偏轉）
  - 不超出場景邊界（boundary clamp）
  - 不離初始位置太遠（max_radius leash）

使用方式：
  在 EventCfg 註冊為 mode="interval", interval_range_s=(0.2, 0.2)。
  參數由 curriculum phase 動態同步（goal_move_speed=0 時不移動）。
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from ..wall_layout import check_wall_proximity_perenv, get_combined_wall_data


# ---------------------------------------------------------------------------
# Helper: 取得所有可見障礙物位置（批量）
# ---------------------------------------------------------------------------

def _gather_obstacle_positions(
    env: ManagerBasedRLEnv,
    max_obs: int = 50,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """取得所有障礙物位置 [N, K, 2] 和 visibility [N, K]。

    只收集 Z > 0 的可見障礙物。回傳 None 表示無障礙物。
    """
    num_obs = min(getattr(env, '_num_obstacles', 0), max_obs)
    if num_obs <= 0:
        return None, None

    positions = []
    visibilities = []
    for i in range(num_obs):
        try:
            e = env.scene[f"obstacle_{i}"]
            pos_w = e.data.root_pos_w  # [N, 3]
            positions.append(pos_w[:, :2])
            visibilities.append(pos_w[:, 2] > 0)
        except Exception:
            break

    if not positions:
        return None, None

    return torch.stack(positions, dim=1), torch.stack(visibilities, dim=1)


# ---------------------------------------------------------------------------
# Helper: 障礙物近距離檢測
# ---------------------------------------------------------------------------

def _check_obstacle_proximity(
    env: ManagerBasedRLEnv,
    positions: torch.Tensor,    # [N, 2]
    margin: float,
) -> torch.Tensor:
    """檢查 positions 是否太靠近任何可見障礙物。

    Returns:
        [N] boolean — True = 太靠近某個障礙物
    """
    N = positions.shape[0]
    device = positions.device

    obs_xy, obs_vis = _gather_obstacle_positions(env)
    if obs_xy is None:
        return torch.zeros(N, dtype=torch.bool, device=device)

    # [N, K] distances
    dists = torch.norm(positions.unsqueeze(1) - obs_xy, dim=2)
    dists[~obs_vis] = 1e6  # 不可見 → 無限遠
    min_dist = dists.min(dim=1).values
    return min_dist < margin


# ---------------------------------------------------------------------------
# Main event function
# ---------------------------------------------------------------------------

def move_goal_positions(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    # --- 移動控制 ---
    goal_move_speed: float = 0.0,
    goal_move_max_radius: float = 3.0,
    goal_move_behavior: str = "random_walk",
    goal_move_angular_speed: float = 0.5,
    goal_move_dt: float = 0.2,
    # --- 安全邊距 ---
    goal_move_wall_margin: float = 0.5,
    goal_move_obs_margin: float = 0.8,
    # --- 方向變換 (random_walk / drift) ---
    goal_move_dir_steps_min: int = 15,
    goal_move_dir_steps_max: int = 40,
):
    """移動 goal 位置（interval event，每 goal_move_dt 秒呼叫一次）。

    Args:
        goal_move_speed:         線速度上限 (m/s)。0 = 不移動（功能關閉）。
        goal_move_max_radius:    goal 可離初始 spawn 位置的最大距離 (m)。
        goal_move_behavior:      移動行為類型:
                                 "random_walk" — 隨機方向 + 平滑轉向
                                 "drift"       — 緩慢飄移，少量方向變化
                                 "patrol"      — 隨機 waypoint 間巡邏
        goal_move_angular_speed: 方向變換的角速度 (rad/s)。
        goal_move_dt:            時間步長 (s)，須與 interval_range_s 一致。
        goal_move_wall_margin:   與牆壁的最小安全距離 (m)。
        goal_move_obs_margin:    與障礙物的最小安全距離 (m)。
        goal_move_dir_steps_min: random_walk/drift 方向切換的最短間隔 (步)。
        goal_move_dir_steps_max: random_walk/drift 方向切換的最長間隔 (步)。

    移動邏輯:
        1. 根據 behavior 計算位移量 (dx, dy)
        2. 安全檢查: max_radius → boundary → walls → obstacles
        3. 若碰撞: 不移動 + 反轉/偏轉方向
        4. 若通過: 更新 goal_pos_w + 刷新 marker
    """
    # ── 速度 = 0 → 功能關閉 ──
    if goal_move_speed <= 0:
        return

    # ── 取得 GoalCommand ──
    try:
        goal_cmd = env.command_manager.get_term("goal_command")
    except Exception:
        return

    goal_pos = goal_cmd.goal_pos_w  # [N, 3] world frame
    N = goal_pos.shape[0]
    device = goal_pos.device

    # ── Lazy init: 首次呼叫時建立狀態 tensors ──
    if not hasattr(env, '_goal_move_origin_w'):
        env._goal_move_origin_w = goal_pos.clone()
        env._goal_move_heading = (
            torch.rand(N, device=device) * 2 * math.pi - math.pi
        )
        env._goal_move_timer = torch.randint(
            goal_move_dir_steps_min,
            goal_move_dir_steps_max + 1,
            (N,), device=device,
        )
        env._goal_move_waypoint = goal_pos[:, :2].clone()
        env._goal_move_step = torch.zeros(N, dtype=torch.long, device=device)

    origin = env._goal_move_origin_w
    heading = env._goal_move_heading
    timer = env._goal_move_timer
    waypoint = env._goal_move_waypoint
    step_count = env._goal_move_step

    # ── 偵測 goal resample（episode reset 導致 goal 跳位） ──
    # 正常每步最多移動 speed*dt ≈ 0.06m，超過 0.5m 表示 resample 發生
    displacement = torch.norm(goal_pos[:, :2] - origin[:, :2], dim=1)
    resampled = displacement > (goal_move_max_radius + 1.0)
    if resampled.any():
        n_rs = resampled.sum().item()
        origin[resampled] = goal_pos[resampled].clone()
        heading[resampled] = (
            torch.rand(n_rs, device=device) * 2 * math.pi - math.pi
        )
        timer[resampled] = torch.randint(
            goal_move_dir_steps_min,
            goal_move_dir_steps_max + 1,
            (n_rs,), device=device,
        )
        waypoint[resampled] = goal_pos[resampled, :2].clone()
        step_count[resampled] = 0

    # ── 計算位移 ──
    step_count += 1

    if goal_move_behavior == "random_walk":
        # ---- random_walk: 平滑隨機轉向 ----
        # 每 timer 步做一次隨機方向偏轉
        expired = step_count >= timer
        if expired.any():
            n_exp = expired.sum().item()
            # 隨機偏轉量 ∈ [-angular_speed*timer*dt, +angular_speed*timer*dt]
            delta_angle = (
                (torch.rand(n_exp, device=device) - 0.5)
                * 2.0
                * goal_move_angular_speed
                * goal_move_dt
                * timer[expired].float()
            )
            heading[expired] += delta_angle
            timer[expired] = torch.randint(
                goal_move_dir_steps_min,
                goal_move_dir_steps_max + 1,
                (n_exp,), device=device,
            )
            step_count[expired] = 0

        dx = goal_move_speed * goal_move_dt * torch.cos(heading)
        dy = goal_move_speed * goal_move_dt * torch.sin(heading)

    elif goal_move_behavior == "drift":
        # ---- drift: 緩慢線性飄移 ----
        # 方向變換間隔 = random_walk 的 3 倍
        expired = step_count >= (timer * 3)
        if expired.any():
            n_exp = expired.sum().item()
            heading[expired] += (
                (torch.rand(n_exp, device=device) - 0.5)
                * goal_move_angular_speed
                * goal_move_dt
            )
            step_count[expired] = 0

        # 速度減半 → 更慢的飄移
        dx = goal_move_speed * 0.5 * goal_move_dt * torch.cos(heading)
        dy = goal_move_speed * 0.5 * goal_move_dt * torch.sin(heading)

    elif goal_move_behavior == "patrol":
        # ---- patrol: waypoint 間巡邏 ----
        to_wp = waypoint - goal_pos[:, :2]
        dist_to_wp = torch.norm(to_wp, dim=1)
        reached = dist_to_wp < 0.5

        # 到達 waypoint → 隨機選新 waypoint（在 origin 附近 max_radius*0.8 內）
        if reached.any():
            n_reached = reached.sum().item()
            angle = torch.rand(n_reached, device=device) * 2 * math.pi - math.pi
            radius = torch.rand(n_reached, device=device) * goal_move_max_radius * 0.8
            waypoint[reached, 0] = origin[reached, 0] + radius * torch.cos(angle)
            waypoint[reached, 1] = origin[reached, 1] + radius * torch.sin(angle)

        # 朝 waypoint 方向移動
        safe_dist = torch.clamp(dist_to_wp, min=1e-6)
        dir_x = to_wp[:, 0] / safe_dist
        dir_y = to_wp[:, 1] / safe_dist

        dx = goal_move_speed * goal_move_dt * dir_x
        dy = goal_move_speed * goal_move_dt * dir_y

    elif goal_move_behavior == "toward_obstacle":
        # ---- toward_obstacle: 朝最近的 obstacle 移動 ----
        # 讓 goal 主動靠近障礙物，迫使 agent 在 obstacle 附近導航
        obs_xy, obs_vis = _gather_obstacle_positions(env)
        if obs_xy is not None and obs_xy.shape[0] > 0:
            # 找每個 goal 最近的 visible obstacle
            goal_2d = goal_pos[:, :2]  # [N_goals, 2]
            # obs_xy: [N_obs, 2] — 取所有可見的
            vis_mask = obs_vis.any(dim=0) if obs_vis.dim() > 1 else obs_vis
            vis_obs = obs_xy[vis_mask] if vis_mask.any() else obs_xy[:1]

            # 距離計算：每個 goal 到每個 visible obs
            diff = vis_obs.unsqueeze(0) - goal_2d.unsqueeze(1)  # [G, O, 2]
            dists = diff.norm(dim=2)  # [G, O]
            nearest_idx = dists.argmin(dim=1)  # [G]
            nearest_obs = vis_obs[nearest_idx]  # [G, 2]

            # 朝 nearest obstacle 方向移動
            to_obs = nearest_obs - goal_2d
            dist_to_obs = to_obs.norm(dim=1, keepdim=True).clamp(min=1e-6)
            direction = to_obs / dist_to_obs

            # 到了 obstacle 附近 (< 1.5m) 就停下或繞行
            too_close = dist_to_obs.squeeze() < 1.5
            if too_close.any():
                # 太近時轉為隨機繞行（避免重疊）
                n_close = too_close.sum().item()
                random_angle = torch.rand(n_close, device=device) * 2 * math.pi
                direction[too_close, 0] = torch.cos(random_angle)
                direction[too_close, 1] = torch.sin(random_angle)

            dx = goal_move_speed * goal_move_dt * direction[:, 0]
            dy = goal_move_speed * goal_move_dt * direction[:, 1]
        else:
            # 沒有 obstacle → fallback to random_walk
            dx = goal_move_speed * goal_move_dt * torch.cos(heading)
            dy = goal_move_speed * goal_move_dt * torch.sin(heading)

    else:
        # unknown behavior → 不移動
        return

    # ── Candidate 新位置 ──
    candidate = goal_pos[:, :2].clone()
    candidate[:, 0] += dx
    candidate[:, 1] += dy

    # ══════════════════════════════════════════════════════════════════════
    # 安全檢查 1: max_radius leash（不離初始位置太遠）
    # ══════════════════════════════════════════════════════════════════════
    dist_from_origin = torch.norm(candidate - origin[:, :2], dim=1)
    too_far = dist_from_origin > goal_move_max_radius
    if too_far.any():
        # 超出 leash → 將 candidate 拉回到 max_radius 邊界上
        scale = goal_move_max_radius / dist_from_origin[too_far].clamp(min=1e-6)
        offset = candidate[too_far] - origin[too_far, :2]
        candidate[too_far] = origin[too_far, :2] + offset * scale.unsqueeze(1)
        # 反轉朝 origin 方向
        toward_origin = origin[too_far, :2] - candidate[too_far]
        heading[too_far] = torch.atan2(toward_origin[:, 1], toward_origin[:, 0])

    # ══════════════════════════════════════════════════════════════════════
    # 安全檢查 2: 場景邊界 clamp
    # ══════════════════════════════════════════════════════════════════════
    env_origins_xy = env.scene.env_origins[:, :2]  # [N, 2]
    candidate_local = candidate - env_origins_xy
    boundary = getattr(env, '_room_boundary', 8.5)
    limit = boundary - goal_move_wall_margin
    clamped = (candidate_local.abs() > limit).any(dim=1)
    candidate_local = torch.clamp(candidate_local, -limit, limit)
    candidate = candidate_local + env_origins_xy
    # 碰邊界 → 反轉方向
    if clamped.any():
        heading[clamped] += math.pi
        heading[clamped] = torch.remainder(heading[clamped] + math.pi, 2 * math.pi) - math.pi

    # ══════════════════════════════════════════════════════════════════════
    # 安全檢查 3: 牆壁碰撞（AABB proximity）
    # ══════════════════════════════════════════════════════════════════════
    wall_blocked = torch.zeros(N, dtype=torch.bool, device=device)
    try:
        wall_centers, wall_sizes, wall_mask = get_combined_wall_data(env)
        # check_wall_proximity_perenv 需要 local frame positions
        in_wall = check_wall_proximity_perenv(
            candidate_local, wall_centers, wall_sizes, wall_mask,
            goal_move_wall_margin,
        )
        wall_blocked = in_wall
    except Exception:
        pass

    if wall_blocked.any():
        # 碰牆 → 不移動 + 反轉 180 度
        candidate[wall_blocked] = goal_pos[wall_blocked, :2]
        heading[wall_blocked] += math.pi
        heading[wall_blocked] = torch.remainder(
            heading[wall_blocked] + math.pi, 2 * math.pi
        ) - math.pi

    # ══════════════════════════════════════════════════════════════════════
    # 安全檢查 4: 障礙物碰撞
    # ══════════════════════════════════════════════════════════════════════
    if goal_move_obs_margin > 0:
        obs_blocked = _check_obstacle_proximity(env, candidate, goal_move_obs_margin)
        if obs_blocked.any():
            # 碰障礙物 → 不移動 + 偏轉 90 度
            candidate[obs_blocked] = goal_pos[obs_blocked, :2]
            heading[obs_blocked] += math.pi * 0.5
            heading[obs_blocked] = torch.remainder(
                heading[obs_blocked] + math.pi, 2 * math.pi
            ) - math.pi

    # ══════════════════════════════════════════════════════════════════════
    # 寫入新位置 + 更新 marker
    # ══════════════════════════════════════════════════════════════════════
    goal_pos[:, 0] = candidate[:, 0]
    goal_pos[:, 1] = candidate[:, 1]

    # 更新綠色箭頭 marker
    try:
        if hasattr(goal_cmd, '_update_goal_markers') and goal_cmd.cfg.debug_vis:
            goal_cmd._update_goal_markers()
    except Exception:
        pass
