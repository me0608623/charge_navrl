"""Gap-Seeking Rewards — 引導 agent 朝可通行間隙轉向

NavRL03 消融實驗用。只在障礙物附近啟用，遠離障礙物時為零。

1. heading_to_gap_reward: 鼓勵 heading 朝向最大可通行弧段
2. forward_clearance_improvement_reward: 鼓勵轉向後前方空間變大
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_lidar_distances_2d(env, sensor_cfg):
    """取得 LiDAR 每條 ray 的 2D 距離 [N, num_rays]。"""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    sensor_pos_2d = sensor.data.pos_w[:, :2]
    hit_points_2d = sensor.data.ray_hits_w[:, :, :2]
    distances = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
    return torch.nan_to_num(distances, nan=sensor.cfg.max_distance, posinf=sensor.cfg.max_distance)


def _get_d_safe(env, sensor_cfg, body_radius, bottom_k=10):
    """計算 d_safe = mean(bottom_k) - body_radius。"""
    distances = _get_lidar_distances_2d(env, sensor_cfg)
    k = min(bottom_k, distances.shape[1])
    bottom_vals = torch.topk(distances, k=k, dim=1, largest=False).values
    return (bottom_vals.mean(dim=1) - body_radius).clamp(min=1e-4)


def heading_to_gap_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    body_radius: float = 0.35,
    min_gap_width: float = 0.9,
    activation_d_safe: float = 2.0,
    speed_threshold: float = 0.05,
) -> torch.Tensor:
    """鼓勵 agent heading 朝向最大可通行間隙。

    LiDAR 72 bins (5° 分辨率) 覆蓋 360°。
    找到最大連續「可通行」弧段的中心方向，獎勵 heading 對齊。

    只在 d_safe < activation_d_safe 時啟用。

    Returns:
        [N] in [-1, 1]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    device = env.device
    N = env.num_envs

    # 只在近障礙物時啟用
    d_safe = _get_d_safe(env, sensor_cfg, body_radius)
    active = d_safe < activation_d_safe  # [N]
    if not active.any():
        return torch.zeros(N, device=device)

    # LiDAR 72-bin 2D 距離
    # obs_functions.py 的 lidar_vlp16_to_2d_bins 做了 min-pooling + 歸一化
    # 這裡直接從 sensor raw data 計算，取每 bin 的 min distance
    distances = _get_lidar_distances_2d(env, sensor_cfg)  # [N, num_rays]
    num_rays = distances.shape[1]
    num_bins = 72
    rays_per_bin = num_rays // num_bins  # 5760 / 72 = 80

    # Min-pool to 72 bins
    if rays_per_bin > 1:
        # Reshape [N, 72, rays_per_bin] → min over last dim
        trimmed = distances[:, :num_bins * rays_per_bin]
        binned = trimmed.reshape(N, num_bins, rays_per_bin).min(dim=2).values  # [N, 72]
    else:
        binned = distances[:, :num_bins]  # [N, 72]

    # 找可通行方向: distance > min_gap_width
    passable = binned > min_gap_width  # [N, 72] bool

    # 找最大連續可通行弧段（環形處理）
    # 將 passable 左右拼接處理環形
    passable_extended = torch.cat([passable, passable], dim=1)  # [N, 144]

    # 對每個 env 找最大連續 True 段的中心 bin
    gap_center_bin = torch.zeros(N, device=device, dtype=torch.long)
    gap_found = torch.zeros(N, device=device, dtype=torch.bool)

    # 向量化：用 cumsum trick 找最長連續段
    # 標記 False→True 邊界
    not_passable = ~passable_extended  # [N, 144]
    # 計算每個 True run 的長度
    # 在每個 False 位置重置計數
    cumsum = passable_extended.float().cumsum(dim=1)  # [N, 144]
    # 每遇到 False 就記錄該位置的 cumsum 作為重置基準
    reset_vals = cumsum * not_passable.float()  # [N, 144] — 只在 False 位置有值
    # 前向填充 reset_vals（每個 True 段用其前面最後一個 False 的 cumsum）
    reset_base = reset_vals.cummax(dim=1).values  # [N, 144]
    run_lengths = cumsum - reset_base  # [N, 144] — 每個位置的連續 True 長度

    # 找最長 run 的結束位置
    max_run_len, max_run_end = run_lengths.max(dim=1)  # [N], [N]
    # run 中心 = end - length/2 (取模 72)
    gap_center_idx = (max_run_end.float() - max_run_len.float() / 2).long() % num_bins  # [N]
    gap_found = max_run_len > 0

    # 計算 gap 中心的角度（相對於 robot body frame）
    # bin i 對應角度: i * 5° - 180° (從 -180° 到 +175°)
    bin_angle = (gap_center_idx.float() * 5.0 - 180.0) * (3.14159265 / 180.0)  # [N] rad

    # heading error = gap 中心角度（在 body frame 中，0 = 正前方）
    # cos(heading_error): 正前方=1, 正後方=-1
    reward = torch.cos(bin_angle)  # [N]

    # 只在活躍時給獎勵
    reward = reward * active.float() * gap_found.float()

    # 速度門檻：靜止時不給獎勵
    speed = torch.norm(
        torch.nan_to_num(robot.data.root_lin_vel_w[:, :2], nan=0.0), dim=-1
    )
    reward = reward * (speed > speed_threshold).float()

    return torch.nan_to_num(reward, nan=0.0)


def forward_clearance_improvement_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    body_radius: float = 0.35,
    front_arc_bins: int = 12,
    activation_d_safe: float = 2.0,
) -> torch.Tensor:
    """鼓勵轉向後前方可通行空間變大。

    比較當前步與前一步的前方扇區平均 clearance。
    正 = 前方空間增大（好的轉向），負 = 前方空間減小（壞的轉向）。

    只在 d_safe < activation_d_safe 時啟用。

    Args:
        front_arc_bins: 前方扇區的 bin 數（12 bins = 60°）

    Returns:
        [N] in [-0.5, 0.5]
    """
    device = env.device
    N = env.num_envs

    d_safe = _get_d_safe(env, sensor_cfg, body_radius)
    active = d_safe < activation_d_safe

    # 取 LiDAR 原始距離並 bin 化
    distances = _get_lidar_distances_2d(env, sensor_cfg)
    num_rays = distances.shape[1]
    num_bins = 72
    rays_per_bin = num_rays // num_bins

    if rays_per_bin > 1:
        trimmed = distances[:, :num_bins * rays_per_bin]
        binned = trimmed.reshape(N, num_bins, rays_per_bin).min(dim=2).values
    else:
        binned = distances[:, :num_bins]

    # 前方扇區: bin 0 = -180°, bin 36 = 0° (正前方)
    # 前方 ±30° = bin (36 - front_arc_bins//2) to (36 + front_arc_bins//2)
    center = num_bins // 2  # 36
    half = front_arc_bins // 2
    start = center - half
    end = center + half
    front_clearance = binned[:, start:end].mean(dim=1)  # [N]

    # PBRS: 比較前一步的 front clearance
    if not hasattr(env, "_prev_front_clearance") or env._prev_front_clearance is None:
        env._prev_front_clearance = front_clearance.clone()
        return torch.zeros(N, device=device)

    # Episode reset 處理
    just_reset = env.episode_length_buf == 0
    env._prev_front_clearance[just_reset] = front_clearance[just_reset]

    delta = (front_clearance - env._prev_front_clearance).clamp(-0.5, 0.5)  # [N]
    env._prev_front_clearance = front_clearance.clone()

    # 只在近障礙物時啟用
    reward = delta * active.float()

    return torch.nan_to_num(reward, nan=0.0)


__all__ = [
    "heading_to_gap_reward",
    "forward_clearance_improvement_reward",
]
