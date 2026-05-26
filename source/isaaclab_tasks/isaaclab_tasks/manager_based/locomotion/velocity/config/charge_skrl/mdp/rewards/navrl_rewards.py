"""NavRL-Style Dense Rewards for VLP16 Curriculum

3 個密集獎勵函數，提供「方向性的安全導航信號」：
1. velocity_to_goal_reward: 鼓勵朝目標方向高效前進
2. safety_log_distance_reward: LiDAR bottom-K log 距離（近距離梯度陡，遠距離梯度平）
3. safe_progress_reward: PBRS × 分段線性安全 gate（含負區：危險前進扣分）

設計動機：
v9 純死亡機制缺乏密集梯度信號，agent 學會「不動=不死」。
本模組引入 NavRL 風格的密集獎勵，讓 agent 同時學會前進和避障。

References:
  - NavRL (Xu et al., 2025) for log-distance safety
  - Ng et al. (1999) for potential-based reward shaping
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_lidar_safety_stats(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    body_radius: float = 0.35,
    bottom_k: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """計算 LiDAR bottom-K 安全統計量。

    不做快取，直接計算（GPU 上 5760 ray sort 只需 ~0.1ms）。

    Args:
        env: 環境實例
        sensor_cfg: LiDAR 感測器配置
        body_radius: 機器人車體半徑 (m)
        bottom_k: 取最近的 K 條 ray

    Returns:
        (bottom_k_mean, d_safe):
            bottom_k_mean: [N] 最近 K 條 ray 的平均 2D 距離
            d_safe: [N] 安全餘裕 = bottom_k_mean - body_radius，clamp >= ε
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    # 2D 距離計算（與 _get_lidar_min_distance 一致）
    sensor_pos_2d = sensor.data.pos_w[:, :2]  # [N, 2]
    hit_points_2d = sensor.data.ray_hits_w[:, :, :2]  # [N, num_rays, 2]
    distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)  # [N, num_rays]

    # NaN/inf 保護：未命中的 ray 設為 max_distance
    distances_2d = torch.nan_to_num(
        distances_2d,
        nan=sensor.cfg.max_distance,
        posinf=sensor.cfg.max_distance,
    )

    # Bottom-K：取最近 K 條 ray（比 global min 更穩定）
    # topk(largest=False) 返回最小的 K 個值
    actual_k = min(bottom_k, distances_2d.shape[1])
    bottom_k_vals = torch.topk(distances_2d, k=actual_k, dim=1, largest=False).values  # [N, K]
    bottom_k_mean = bottom_k_vals.mean(dim=1)  # [N]

    # 安全餘裕 = 到最近障礙物表面的距離
    d_safe = (bottom_k_mean - body_radius).clamp(min=1e-4)  # [N]

    return bottom_k_mean, d_safe


def velocity_to_goal_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    min_goal_dist: float = 0.5,
    max_reward_speed: float = 1.0,
    body_radius: float = 0.35,
    bottom_k: int = 10,
    d_attenuate: float = 1.0,
    d_full: float = 2.5,
    v_gate_floor: float = 0.0,
) -> torch.Tensor:
    """朝目標方向的速度獎勵（雙向：正向獎勵、背離懲罰 + LiDAR 安全衰減）。

    Math:
        goal_dir = normalize(goal_pos - robot_pos)      # 2D unit vector
        v_toward = dot(vel_2d, goal_dir)                 # 含正負：正=朝向，負=背離
        reward = clamp(v_toward / v_max, -1, 1)          # 歸一化到 [-1, 1]
        reward = reward * (goal_dist > min_dist).float()  # 太近目標時不給

        # 安全衰減：靠近障礙物時削弱 v_to_goal，防止撞牆
        v_gate = clamp((d_safe - d_attenuate) / (d_full - d_attenuate), 0, 1)
        reward = reward * v_gate

    物理意義：朝目標前進得正獎勵，背離目標得負懲罰。
    靠近障礙物時獎勵被衰減，避免目標在牆後時推 agent 撞牆。
    範圍：[-1, 1]，有效 per-step: -3.0 ~ +3.0（×15×0.2）

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        sensor_cfg: LiDAR 感測器配置（用於安全衰減）
        min_goal_dist: 低於此距離不給獎勵 (m)
        max_reward_speed: 歸一化用的最大速度 (m/s)
        body_radius: 機器人車體半徑 (m)
        bottom_k: 取最近的 K 條 ray
        d_attenuate: 低於此距離開始削弱 v_to_goal (m)
        d_full: 高於此距離完整獎勵 (m)

    Returns:
        [num_envs] in [-1, 1]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    device = env.device

    # Robot position & velocity (2D)
    robot_pos = torch.nan_to_num(robot.data.root_pos_w[:, :2], nan=0.0)  # [N, 2]
    vel_2d = torch.nan_to_num(robot.data.root_lin_vel_w[:, :2], nan=0.0)  # [N, 2]

    # Goal position（統一來源）
    if hasattr(env, "_local_goal_world") and env._local_goal_world is not None:
        goal_pos = env._local_goal_world  # [N, 2]
    else:
        goal_pos = env.command_manager.get_command("goal_command")[:, :2]

    # Goal direction
    diff = goal_pos - robot_pos  # [N, 2]
    goal_dist = torch.norm(diff, dim=1, keepdim=True).clamp(min=1e-6)  # [N, 1]
    goal_dir = diff / goal_dist  # [N, 2] unit vector

    # 朝目標方向的速度分量（雙向：正=朝向，負=背離）
    v_toward = (vel_2d * goal_dir).sum(dim=1)  # [N]

    # 歸一化到 [-1, 1]
    reward = (v_toward / max_reward_speed).clamp(-1.0, 1.0)  # [N]

    # 太近目標時不給（避免到達後繼續加速）
    far_enough = (goal_dist.squeeze(1) > min_goal_dist).float()  # [N]
    reward = reward * far_enough

    # 安全衰減：靠近障礙物時削弱 v_to_goal，防止目標在牆後時推 agent 撞牆
    # d_safe < d_attenuate → v_gate=0（完全關閉）
    # d_attenuate < d_safe < d_full → v_gate ∈ (0, 1)（線性過渡）
    # d_safe > d_full → v_gate=1.0（完整信號）
    _, d_safe = _get_lidar_safety_stats(env, sensor_cfg, body_radius, bottom_k)
    v_gate = ((d_safe - d_attenuate) / (d_full - d_attenuate + 1e-6)).clamp(v_gate_floor, 1.0)
    reward = reward * v_gate

    # NaN 保護
    reward = torch.nan_to_num(reward, nan=0.0)

    return reward


def safety_log_distance_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    body_radius: float = 0.35,
    bottom_k: int = 10,
    max_distance: float = 8.0,
    speed_threshold: float = 0.1,
    min_speed_factor: float = 0.1,
) -> torch.Tensor:
    """LiDAR bottom-K log 距離安全獎勵（速度耦合）。

    Math:
        d_safe = clamp(mean(bottom_k) - body_radius, min=ε)
        raw = clamp(log(d_safe), -3, 3)

        speed_factor = clamp(speed / speed_threshold, min_sf, 1.0)
        reward = raw × speed_factor

    速度耦合解決「安全地不動」局部最優：
    - 靜止時 speed_factor=0.1 → 正向安全獎勵衰減 90%
    - 移動時 speed_factor=1.0 → 完整安全信號
    - min_sf=0.1 確保靜止時仍保留 10% 的危險警告

    範圍：[-3, 3]，有效 per-step: -1.8 ~ +1.8（×3×0.2）

    Args:
        env: 環境實例
        robot_cfg: 機器人配置（用於讀取速度）
        sensor_cfg: LiDAR 感測器配置
        body_radius: 機器人車體半徑 (m)
        bottom_k: 取最近的 K 條 ray
        max_distance: 超過此距離視為等效安全（限制 log 上界）
        speed_threshold: 速度因子飽和閾值 (m/s)
        min_speed_factor: 靜止時的最低速度因子（保留部分危險警告）

    Returns:
        [num_envs] in [-6, 2]
    """
    robot: Articulation = env.scene[robot_cfg.name]

    _, d_safe = _get_lidar_safety_stats(env, sensor_cfg, body_radius, bottom_k)

    # Log 變換：近距離梯度陡，遠距離梯度平
    raw = torch.log(d_safe).clamp(-6.0, 2.0)  # [N]

    # 速度因子：靜止時衰減安全正獎勵，移動時完整信號
    speed = torch.norm(
        torch.nan_to_num(robot.data.root_lin_vel_w[:, :2], nan=0.0), dim=-1
    )  # [N]
    speed_factor = (speed / speed_threshold).clamp(min_speed_factor, 1.0)  # [N]

    reward = raw * speed_factor  # [N]

    # NaN 保護
    reward = torch.nan_to_num(reward, nan=0.0)

    return reward


def safe_progress_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    body_radius: float = 0.35,
    bottom_k: int = 10,
    safety_threshold: float = 1.0,
    safety_temperature: float = 0.3,
    d_danger: float = 0.8,
    d_comfort: float = 2.0,
    negative_scale: float = 0.5,
) -> torch.Tensor:
    """安全耦合的 PBRS 進度獎勵（分段線性 gate，含負區）。

    Math:
        # PBRS 部分
        progress = d_prev - d_curr                          # 距離縮減
        progress = clamp(progress, -1, 1)

        # 安全耦合（分段線性 gate，含負區）
        d_safe = bottom_k_mean - body_radius
        gate:
          d_safe < d_danger   → -η         (懲罰前進)
          d_danger ≤ d_safe ≤ d_comfort → [0, 1]  (線性過渡)
          d_safe > d_comfort  → 1.0        (完整獎勵)

        # 最終獎勵
        reward = progress × gate

    關鍵性質：gate < 0 且 progress > 0 → reward < 0（危險區前進扣分）。
    反之 gate < 0 且 progress < 0 → reward > 0（獎勵撤退！）

    使用 env._prev_goal_dist_navrl（獨立於舊版 _prev_goal_dist）。

    範圍：[-1, 1]，有效 per-step: -12 ~ +12（×60×0.2）

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        sensor_cfg: LiDAR 感測器配置
        body_radius: 機器人車體半徑 (m)
        bottom_k: 取最近的 K 條 ray
        safety_threshold: 安全 gate 中心點 (m)（保留向後相容）
        safety_temperature: 安全 gate sigmoid 溫度（保留向後相容）
        d_danger: 低於此距離 gate 為負 (m)
        d_comfort: 高於此距離 gate = 1.0 (m)
        negative_scale: η，danger zone 內 gate = -η

    Returns:
        [num_envs] in [-1, 1]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    device = env.device

    # Robot position (2D)
    robot_pos = torch.nan_to_num(robot.data.root_pos_w[:, :2], nan=0.0)  # [N, 2]

    # Goal position（統一來源）
    if hasattr(env, "_local_goal_world") and env._local_goal_world is not None:
        goal_pos = env._local_goal_world  # [N, 2]
    else:
        goal_pos = env.command_manager.get_command("goal_command")[:, :2]

    # Current distance to goal
    d_curr = torch.norm(goal_pos - robot_pos, dim=1)  # [N]

    # === PBRS 部分 ===
    # 使用獨立的 _prev_goal_dist_navrl（不與舊版衝突）
    if not hasattr(env, "_prev_goal_dist_navrl") or env._prev_goal_dist_navrl is None:
        env._prev_goal_dist_navrl = d_curr.clone()
        return torch.zeros(env.num_envs, device=device)

    # Detect episode resets
    just_reset = env.episode_length_buf == 0
    env._prev_goal_dist_navrl[just_reset] = d_curr[just_reset]

    d_prev = env._prev_goal_dist_navrl

    # Progress: positive when approaching goal
    progress = (d_prev - d_curr).clamp(-1.0, 1.0)  # [N]

    # Update for next step
    env._prev_goal_dist_navrl = d_curr.clone()

    # === 安全耦合部分（分段線性 gate，含負區）===
    _, d_safe = _get_lidar_safety_stats(env, sensor_cfg, body_radius, bottom_k)

    # 分段線性 gate:
    #   d_safe < d_danger   → gate = -η        (懲罰「危險區前進」)
    #   d_danger ≤ d_safe ≤ d_comfort → gate ∈ [0, 1]  (線性過渡)
    #   d_safe > d_comfort  → gate = 1.0       (完整獎勵)
    gate = torch.where(
        d_safe < d_danger,
        torch.full_like(d_safe, -negative_scale),
        ((d_safe - d_danger) / (d_comfort - d_danger + 1e-6)).clamp(0.0, 1.0),
    )

    # 最終獎勵
    # Fix: 當 gate < 0（危險區）且 progress < 0（後退），原本 reward > 0
    # 這會獎勵倒車行為。改為：危險區後退時 reward = 0（中性），不獎勵也不懲罰。
    # 只保留 gate < 0 懲罰「危險區前進」的效果。
    raw_reward = progress * gate  # [N]
    # Clamp: 只允許 gate<0 造成負獎勵（懲罰危險前進），不允許正獎勵（獎勵倒車）
    in_danger = d_safe < d_danger
    reward = torch.where(
        in_danger & (raw_reward > 0),  # 危險區且 reward > 0 → 後退被獎勵的情況
        torch.zeros_like(raw_reward),   # 改為 0（不獎勵倒車）
        raw_reward,
    )

    # NaN 保護
    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

    return reward


def reverse_near_dynamic_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    body_radius: float = 0.35,
    front_arc_half_angle: float = 1.047,  # 60° = π/3 rad
    detection_distance: float = 3.0,
    reverse_speed_threshold: float = -0.02,
    bottom_k: int = 6,
) -> torch.Tensor:
    """Penalize reversing when dynamic obstacles are detected in the forward sector.

    設計動機：
    ORCA 或 policy 在遇到正面動態障礙時傾向直接倒車，而非側向繞行。
    此獎勵函數在「前方有動態障礙 + 機器人正在倒車」時施加懲罰，
    引導 policy 學習 forward-moving detour 而非 passive retreat。

    判斷邏輯：
    1. 前方扇形區（±60°）內的 LiDAR bottom-K 距離 < detection_distance → 前方有障礙
    2. body-frame 線速度 < reverse_speed_threshold → 正在倒車
    3. 兩者同時成立 → penalty = |v_body| / v_max（倒車越快罰越重）

    排除情況：
    - 前方完全暢通（d_front > detection_distance）→ 不罰（自由倒車調整姿態）
    - 機器人往前走（v_body ≥ 0）→ 不罰

    範圍：[0, 1]，建議 weight = -5.0 ~ -10.0

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        sensor_cfg: LiDAR 感測器配置
        body_radius: 機器人車體半徑 (m)
        front_arc_half_angle: 前方扇形半角 (rad)，預設 60° = π/3
        detection_distance: 前方障礙偵測距離 (m)
        reverse_speed_threshold: 低於此速度視為倒車 (m/s)，負值
        bottom_k: 前方扇形內取最近 K 條 ray

    Returns:
        [num_envs] in [0, 1]，倒車越快、前方越近 → 值越大
    """
    robot: Articulation = env.scene[robot_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    # --- Body-frame velocity: forward component ---
    # 用 quaternion 將 world-frame vel 投影到 body-frame x 軸
    robot_quat_w = robot.data.root_quat_w  # [N, 4]
    vel_w = robot.data.root_lin_vel_w  # [N, 3]
    # Inverse rotate world vel to body frame
    quat_inv = torch.stack([robot_quat_w[:, 0], -robot_quat_w[:, 1],
                            -robot_quat_w[:, 2], -robot_quat_w[:, 3]], dim=1)
    from isaaclab.utils.math import quat_apply
    vel_body = quat_apply(quat_inv, vel_w)  # [N, 3]
    v_forward = vel_body[:, 0]  # [N] body x = forward

    # --- Front-arc LiDAR distance ---
    # Compute 2D distances for all rays
    sensor_pos_2d = sensor.data.pos_w[:, :2]  # [N, 2]
    hit_points_2d = sensor.data.ray_hits_w[:, :, :2]  # [N, R, 2]
    distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)  # [N, R]
    distances_2d = torch.nan_to_num(distances_2d, nan=sensor.cfg.max_distance,
                                    posinf=sensor.cfg.max_distance)

    # Determine which rays fall within the front arc
    # Ray angles: evenly spaced around 360°, relative to robot heading
    num_rays = distances_2d.shape[1]
    ray_angles = torch.linspace(0, 2 * torch.pi, num_rays + 1, device=env.device)[:num_rays]

    # Front arc mask: rays within ±front_arc_half_angle of 0° (forward)
    # Rays near 0° or near 2π are "forward"
    front_mask = (ray_angles < front_arc_half_angle) | (ray_angles > (2 * torch.pi - front_arc_half_angle))

    # Apply mask to select front-arc rays only
    front_distances = distances_2d.clone()
    front_distances[:, ~front_mask] = sensor.cfg.max_distance  # mask out non-front rays

    # Bottom-K of front-arc distances
    actual_k = min(bottom_k, int(front_mask.sum().item()))
    if actual_k == 0:
        return torch.zeros(env.num_envs, device=env.device)
    front_bottom_k = torch.topk(front_distances, k=max(1, actual_k), dim=1, largest=False).values
    d_front_min = front_bottom_k.mean(dim=1)  # [N]

    # --- Penalty logic ---
    # Condition 1: obstacle in front (within detection distance)
    obstacle_ahead = (d_front_min - body_radius) < detection_distance  # [N] bool

    # Condition 2: robot is reversing
    is_reversing = v_forward < reverse_speed_threshold  # [N] bool

    # Penalty magnitude: proportional to reverse speed (faster reverse = larger penalty)
    # Normalized by max_speed (1.0 m/s)
    reverse_magnitude = (-v_forward / 1.0).clamp(0.0, 1.0)  # [N] in [0, 1]

    # Combined penalty: only when both conditions are met
    penalty = (obstacle_ahead & is_reversing).float() * reverse_magnitude  # [N]

    return torch.nan_to_num(penalty, nan=0.0)


def forward_detour_bonus(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    body_radius: float = 0.35,
    bottom_k: int = 10,
    d_activation: float = 2.0,
    min_forward_speed: float = 0.1,
) -> torch.Tensor:
    """Bonus for maintaining forward speed while near obstacles (lateral detour behavior).

    設計動機：
    鼓勵 agent 在靠近障礙物時保持前進（繞行），而非停下或倒車。
    只在 d_safe < d_activation 且 v_forward > min_forward_speed 時才給獎勵，
    引導 policy 學習「邊閃邊前進」的平滑繞行策略。

    範圍：[0, 1]，建議 weight = 2.0 ~ 5.0

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        sensor_cfg: LiDAR 感測器配置
        body_radius: 機器人車體半徑 (m)
        bottom_k: 取最近的 K 條 ray
        d_activation: 低於此距離才啟動獎勵 (m)
        min_forward_speed: 最低前進速度才算 detour (m/s)

    Returns:
        [num_envs] in [0, 1]
    """
    robot: Articulation = env.scene[robot_cfg.name]

    # Body-frame forward velocity
    robot_quat_w = robot.data.root_quat_w
    vel_w = robot.data.root_lin_vel_w
    quat_inv = torch.stack([robot_quat_w[:, 0], -robot_quat_w[:, 1],
                            -robot_quat_w[:, 2], -robot_quat_w[:, 3]], dim=1)
    from isaaclab.utils.math import quat_apply
    vel_body = quat_apply(quat_inv, vel_w)
    v_forward = vel_body[:, 0]

    # Safety distance
    _, d_safe = _get_lidar_safety_stats(env, sensor_cfg, body_radius, bottom_k)

    # Activation: only near obstacles
    near_obstacle = (d_safe < d_activation).float()

    # Forward bonus: speed normalized to [0, 1]
    forward_speed_normalized = (v_forward / 1.0).clamp(0.0, 1.0)

    # Only reward if actually moving forward above threshold
    is_forward = (v_forward > min_forward_speed).float()

    # Proximity scaling: closer obstacles → stronger bonus (incentivize active dodging)
    proximity_scale = (1.0 - d_safe / d_activation).clamp(0.0, 1.0)

    bonus = near_obstacle * is_forward * forward_speed_normalized * proximity_scale

    return torch.nan_to_num(bonus, nan=0.0)


__all__ = [
    "velocity_to_goal_reward",
    "safety_log_distance_reward",
    "safe_progress_reward",
    "reverse_near_dynamic_penalty",
    "forward_detour_bonus",
]
