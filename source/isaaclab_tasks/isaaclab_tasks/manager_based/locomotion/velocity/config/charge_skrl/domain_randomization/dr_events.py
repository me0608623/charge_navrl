"""域隨機化事件整合 (Domain Randomization Event Integration)

將所有域隨機化技術整合為 Isaac Lab EventTerm 兼容的函數。
在環境配置的 EventCfg 中使用。

函數：
    apply_domain_randomization: 每次 reset 時調用，初始化該 episode 的隨機參數
    domain_randomization_pre_step: 每步調用，施加持續性擾動

使用方式：
    class EventCfg:
        domain_randomization = EventTerm(
            func=apply_domain_randomization,
            mode="reset",
            params={
                "enable_physics": True,
                "enable_sensor_noise": True,
                "enable_external_force": True,
                "enable_actuator_dr": False,  # Phase 1+ 啟用
            },
        )

參數預設值（保守配置，適合 Phase 0）：
    - 初始速度: ±0.5 m/s, ±0.5 rad/s
    - 推力: 10-20N, 10% 環境
    - 物理/致動器 DR: 預設關閉（Phase 1+ 漸進啟用）
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def apply_domain_randomization(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    enable_physics: bool = True,
    enable_sensor_noise: bool = True,
    enable_external_force: bool = True,
    enable_actuator_dr: bool = False,
    # 初始速度範圍
    init_lin_vel_range: float = 0.5,
    init_ang_vel_range: float = 0.5,
    # 推力參數
    push_force_range: tuple[float, float] = (5.0, 20.0),
    push_env_ratio: float = 0.10,
    # 物理參數
    mass_scale: tuple[float, float] = (0.85, 1.15),
    friction_scale: tuple[float, float] = (0.7, 1.3),
    com_offset: float = 0.05,
    # 風力
    wind_force_range: tuple[float, float] = (0.0, 3.0),
) -> None:
    """應用域隨機化 — 每次 episode reset 時調用。

    為每個重置的環境採樣新的隨機參數：
    1. 初始速度隨機化（必定啟用）
    2. 物理參數隨機化（質量/摩擦/質心）
    3. 外部擾動初始化（風力方向/大小）
    4. 致動器隨機化（動作延遲/速度縮放）

    Args:
        env: Isaac Lab 環境實例
        env_ids: 需要重置的環境 ID 張量
        enable_physics: 啟用物理參數隨機化（質量/摩擦/質心）
        enable_sensor_noise: 啟用感測器噪聲（由 obs 層處理，這裡僅標記）
        enable_external_force: 啟用外部擾動（推力/風力）
        enable_actuator_dr: 啟用致動器隨機化（動作延遲/速度縮放）
        init_lin_vel_range: 初始線速度範圍 [m/s]
        init_ang_vel_range: 初始角速度範圍 [rad/s]
        push_force_range: 推力範圍 [N]
        push_env_ratio: 被推環境比例
        mass_scale: 質量縮放範圍
        friction_scale: 摩擦力縮放範圍
        com_offset: 質心偏移範圍 [m]
        wind_force_range: 風力範圍 [N]
    """
    from isaaclab.assets import Articulation

    asset: Articulation = env.scene["robot"]
    num_envs = len(env_ids)
    if num_envs == 0:
        return

    # ===== 1. 初始速度隨機化 (必定執行) =====
    random_lin_vel = torch.zeros(num_envs, 3, device=env.device)
    random_lin_vel[:, 0] = torch.empty(num_envs, device=env.device).uniform_(
        -init_lin_vel_range, init_lin_vel_range
    )
    random_lin_vel[:, 1] = torch.empty(num_envs, device=env.device).uniform_(
        -init_lin_vel_range * 0.3, init_lin_vel_range * 0.3
    )

    random_ang_vel = torch.zeros(num_envs, 3, device=env.device)
    random_ang_vel[:, 2] = torch.empty(num_envs, device=env.device).uniform_(
        -init_ang_vel_range, init_ang_vel_range
    )

    root_velocity = torch.cat([random_lin_vel, random_ang_vel], dim=1)
    asset.write_root_velocity_to_sim(root_velocity=root_velocity, env_ids=env_ids)

    # ===== 2. 物理參數隨機化 =====
    if enable_physics:
        try:
            from .physics_dr import randomize_robot_mass, randomize_ground_friction, randomize_com_offset
            randomize_robot_mass(env, env_ids, mass_scale=mass_scale)
            randomize_ground_friction(env, env_ids, friction_scale=friction_scale)
            randomize_com_offset(env, env_ids, com_offset_range=com_offset)
        except Exception:
            pass  # API 不可用時靜默跳過

    # ===== 3. 外部擾動初始化 =====
    if enable_external_force:
        try:
            from .disturbance_dr import apply_continuous_wind
            apply_continuous_wind(env, env_ids, wind_force_range=wind_force_range)
        except Exception:
            pass

    # ===== 4. 感測器噪聲標記 =====
    if enable_sensor_noise:
        env._sensor_dr_enabled = True

    # ===== 5. 致動器 DR 初始化 =====
    if enable_actuator_dr:
        env._actuator_dr_enabled = True


def domain_randomization_pre_step(
    env: ManagerBasedRLEnv,
) -> None:
    """域隨機化 Pre-Step 回調 — 每個物理步之前調用。

    施加持續性擾動（如風力）和隨機推力。
    通常作為 EventTerm(mode="interval") 使用。

    Args:
        env: 環境實例
    """
    # 持續風力已由 permanent_wrench_composer 處理（在 reset 時設定，跨 step 自動施加）
    # 無需在 pre_step 中重複施加
    pass


__all__ = [
    "apply_domain_randomization",
    "domain_randomization_pre_step",
]
