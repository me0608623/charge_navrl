"""感測器域隨機化 (Sensor Domain Randomization)

模擬真實 LiDAR 感測器的非理想行為，提高策略對感測器雜訊的魯棒性。

功能：
    1. 射線丟失 (Ray Dropout): 真實 LiDAR 5-10% 的射線因表面反射率低/
       角度過大/遮擋而丟失，返回 max_range。
    2. 距離高斯雜訊: 真實測距精度 ±3cm (VLP-16 規格)。
    3. 系統性偏差: 每個通道有固定安裝偏差 (每次 reset 重新採樣)。
    4. 幽靈點: 多路徑反射產生的虛假近距離讀數。

參數:
    dropout_rate: float = 0.05    — 射線丟失概率 (5%)
    distance_noise_std: float = 0.03  — 距離雜訊標準差 [m]
    bias_range: tuple = (-0.05, 0.05)  — 通道偏差範圍 [m]
    ghost_rate: float = 0.002     — 幽靈點概率
    ghost_distance: tuple = (0.2, 2.0)  — 幽靈點距離範圍 [m]

訓練流程整合:
    這些函數設計為在 obs_functions.py 的 lidar_vlp16_to_2d_bins 內部調用，
    或作為獨立的 EventTerm 在 reset/interval 模式下使用。

參考:
    - VLP-16 Datasheet: ±3cm 精度, 5-10% 丟失率
    - Dosovitskiy et al. (2017) "CARLA" 模擬器的感測器雜訊模型
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def apply_lidar_ray_dropout(
    distances: torch.Tensor,
    max_distance: float = 20.0,
    dropout_rate: float = 0.05,
) -> torch.Tensor:
    """LiDAR 射線丟失 — 隨機將部分射線設為 max_distance。

    物理意義: 真實 LiDAR 在黑色/反光表面、極端角度、煙霧環境中
    會丟失射線，返回零值或最大量程。此函數模擬這種行為。

    Args:
        distances: [num_envs, num_rays] 原始距離讀數
        max_distance: 最大量程 [m]，丟失的射線設為此值
        dropout_rate: 每條射線的丟失概率 (0.05 = 5%)

    Returns:
        [num_envs, num_rays] 加入丟失後的距離讀數
    """
    dropout_mask = torch.rand_like(distances) < dropout_rate
    return torch.where(dropout_mask, max_distance, distances)


def apply_lidar_distance_noise(
    distances: torch.Tensor,
    noise_std: float = 0.03,
    min_distance: float = 0.1,
) -> torch.Tensor:
    """LiDAR 距離高斯雜訊 — 模擬測距精度限制。

    物理意義: VLP-16 的測距精度為 ±3cm (1σ)。距離越遠，
    雜訊影響越大，但相對誤差保持在合理範圍。

    Args:
        distances: [num_envs, num_rays] 原始距離讀數
        noise_std: 高斯雜訊標準差 [m] (VLP-16 典型值 0.03)
        min_distance: 加噪後的最小距離 [m]（避免負值）

    Returns:
        [num_envs, num_rays] 加噪後的距離讀數
    """
    noise = torch.randn_like(distances) * noise_std
    noisy = distances + noise
    return noisy.clamp(min=min_distance)


def apply_lidar_systematic_bias(
    env: ManagerBasedRLEnv,
    distances: torch.Tensor,
    bias_range: tuple[float, float] = (-0.05, 0.05),
) -> torch.Tensor:
    """LiDAR 系統性偏差 — 每個通道有固定偏移量。

    物理意義: LiDAR 安裝時的微小角度/位置偏差會導致每個通道
    有固定的距離偏移。這在每次 episode reset 時重新採樣。

    Args:
        env: 環境實例（用於存取/生成 per-env 偏差）
        distances: [num_envs, num_rays] 原始距離讀數
        bias_range: 偏差範圍 [m]

    Returns:
        [num_envs, num_rays] 加入偏差後的距離讀數
    """
    num_rays = distances.shape[1]

    # 每個 env 有自己的通道偏差（在 reset 時生成一次）
    if not hasattr(env, "_lidar_channel_bias") or env._lidar_channel_bias is None:
        env._lidar_channel_bias = torch.empty(
            env.num_envs, num_rays, device=env.device
        ).uniform_(bias_range[0], bias_range[1])

    # 偵測 reset 的 env 並重新採樣
    just_reset = env.episode_length_buf == 0
    if just_reset.any():
        env._lidar_channel_bias[just_reset] = torch.empty(
            just_reset.sum().item(), num_rays, device=env.device
        ).uniform_(bias_range[0], bias_range[1])

    return (distances + env._lidar_channel_bias).clamp(min=0.1)


def apply_lidar_ghost_points(
    distances: torch.Tensor,
    max_distance: float = 20.0,
    ghost_rate: float = 0.002,
    ghost_distance_range: tuple[float, float] = (0.2, 2.0),
) -> torch.Tensor:
    """LiDAR 幽靈點 — 多路徑反射產生的虛假近距離讀數。

    物理意義: 在玻璃/鏡面/金屬表面附近，LiDAR 光束可能經過
    多次反射後返回，產生比真實距離更近的虛假讀數。

    Args:
        distances: [num_envs, num_rays] 原始距離讀數
        max_distance: 最大量程 [m]
        ghost_rate: 幽靈點概率 (0.002 = 0.2%)
        ghost_distance_range: 幽靈點距離範圍 [m]

    Returns:
        [num_envs, num_rays] 加入幽靈點的距離讀數
    """
    ghost_mask = torch.rand_like(distances) < ghost_rate
    ghost_distances = torch.empty_like(distances).uniform_(
        ghost_distance_range[0], ghost_distance_range[1]
    )
    # 幽靈點只會比真實距離更近
    ghost_values = torch.min(distances, ghost_distances)
    return torch.where(ghost_mask, ghost_values, distances)


__all__ = [
    "apply_lidar_ray_dropout",
    "apply_lidar_distance_noise",
    "apply_lidar_systematic_bias",
    "apply_lidar_ghost_points",
]
