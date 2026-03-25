"""
域隨機化事件 (Domain Randomization Events)

將域隨機化集成到 Isaac Lab 的事件系統中。
"""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import SceneEntityCfg


def apply_domain_randomization(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    enable_physics: bool = True,
    enable_sensor_noise: bool = True,
    enable_external_force: bool = True,
) -> None:
    """應用域隨機化（事件處理器版本）

    在環境重置時應用域隨機化。

    Args:
        env: 環境實例
        env_ids: 需要隨機化的環境 ID
        enable_physics: 是否啟用物理參數隨機化
        enable_sensor_noise: 是否啟用傳感器噪聲
        enable_external_force: 是否啟用外部擾動
    """
    from isaaclab.assets import Articulation

    asset: Articulation = env.scene["robot"]

    # ========== 1. 初始速度隨機 ==========
    # 機器人重置時給予隨機初始速度（模擬真實環境中的非靜止狀態）
    num_envs = len(env_ids)
    if num_envs > 0:
        # 線速度隨機：±0.5 m/s
        random_lin_vel = torch.zeros(
            (num_envs, 3),
            device=env.device,
            dtype=torch.float32,
        )
        random_lin_vel[:, 0] = torch.zeros(
            num_envs, device=env.device
        ).uniform_(-0.5, 0.5)  # X 方向：前進/後退
        random_lin_vel[:, 1] = torch.zeros(
            num_envs, device=env.device
        ).uniform_(-0.15, 0.15)  # Y 方向：側向（較小）

        # 角速度隨機：±0.5 rad/s
        random_ang_vel = torch.zeros(
            (num_envs, 3),
            device=env.device,
            dtype=torch.float32,
        )
        random_ang_vel[:, 2] = torch.zeros(
            num_envs, device=env.device
        ).uniform_(-0.5, 0.5)  # Z 軸旋轉

        # 寫入速度
        # 🔥 修復：write_root_velocity_to_sim 只接受一個 root_velocity 張量
        # root_velocity = [lin_vel, ang_vel]，需要合併為 [N, 6]
        root_velocity = torch.cat([random_lin_vel, random_ang_vel], dim=1)
        asset.write_root_velocity_to_sim(
            root_velocity=root_velocity,
            env_ids=env_ids,
        )

    # ========== 2. 物理參數隨機化 ==========
    # 註：Isaac Sim 中無法在運行時直接修改質量和摩擦力
    # 這需要在 USD 層面實現，這裡僅作為接口預留
    if enable_physics:
        # 質量隨機化：±10%
        # 摩擦力隨機化：±20%
        # TODO: 在 USD 層面實現
        pass

    # ========== 3. 傳感器噪聲增強 ==========
    # LiDAR 噪聲已在觀測配置中通過 noise 參數實現
    if enable_sensor_noise:
        # 噪聲在觀測階段添加，不需要在這裡處理
        pass

    # ========== 4. 外部擾動 ==========
    # 隨機施加外力模擬不平整地面
    if enable_external_force and num_envs > 0:
        # 10% 的環境施加外力
        num_forces = max(1, num_envs // 10)
        force_env_ids = env_ids[torch.randint(0, num_envs, (num_forces,), device=env.device)]

        # 隨機力大小：5-20N
        force_magnitude = torch.zeros(num_forces, device=env.device).uniform_(5.0, 20.0)

        # 隨機方向（水平面）
        angles = torch.zeros(num_forces, device=env.device).uniform_(0, 2 * np.pi)
        force_direction = torch.stack([
            torch.cos(angles),
            torch.sin(angles),
            torch.zeros_like(angles),
        ], dim=1)

        # 施加力（在質心處）
        # 🔥 修復：使用 Isaac Lab 正確的 API
        forces = force_direction * force_magnitude.unsqueeze(1)
        torques = torch.zeros_like(forces)  # 不施加力矩

        # 使用 permanent_wrench_composer 設置外力和外力矩
        asset.permanent_wrench_composer.set_forces_and_torques(
            forces=forces.unsqueeze(1),  # [N, 1, 3] - 應用到根物體
            torques=torques.unsqueeze(1),
            body_ids=None,  # None 表示根物體
            env_ids=force_env_ids,
        )


def domain_randomization_pre_step(
    env: ManagerBasedRLEnv,
) -> None:
    """域隨機化 - Pre Step 回調

    在每個 step 之前應用域隨機化。

    Args:
        env: 環境實例
    """
    # 獲取域隨機化配置（從環境屬性中）
    if not hasattr(env, "_domain_rand_config"):
        return

    cfg = env._domain_rand_config
    if not cfg.enable:
        return

    # 按頻率隨機化
    if env.common_step_counter % cfg.randomize_frequency != 0:
        return

    # 低概率施加外部擾動（模擬不平整地面）
    if np.random.random() < cfg.external_force_prob:
        _apply_random_external_force(env, cfg)


def _apply_random_external_force(
    env: ManagerBasedRLEnv,
    cfg,
) -> None:
    """施加隨機外部擾動"""
    from isaaclab.assets import Articulation

    asset: Articulation = env.scene["robot"]

    # 隨機選擇環境
    num_forces = max(1, env.num_envs // 10)
    env_ids = torch.randint(0, env.num_envs, (num_forces,), device=env.device)

    # 隨機力
    force_magnitude = torch.zeros(num_forces, device=env.device).uniform_(
        cfg.external_force_range[0],
        cfg.external_force_range[1],
    )

    # 隨機方向
    angles = torch.zeros(num_forces, device=env.device).uniform_(0, 2 * np.pi)
    force_direction = torch.stack([
        torch.cos(angles),
        torch.sin(angles),
        torch.zeros_like(angles),
    ], dim=1)

    # 🔥 修復：使用 Isaac Lab 正確的 API
    forces = force_direction * force_magnitude.unsqueeze(1)
    torques = torch.zeros_like(forces)  # 不施加力矩

    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces.unsqueeze(1),  # [N, 1, 3] - 應用到根物體
        torques=torques.unsqueeze(1),
        body_ids=None,  # None 表示根物體
        env_ids=env_ids,
    )


__all__ = [
    "apply_domain_randomization",
    "domain_randomization_pre_step",
]
