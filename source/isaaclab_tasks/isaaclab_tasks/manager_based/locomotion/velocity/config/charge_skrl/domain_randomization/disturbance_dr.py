"""外部擾動域隨機化 (External Disturbance Domain Randomization)

模擬真實環境中不可控的外部力，提高策略的物理魯棒性。

功能：
    1. 隨機推力 (Random Push): 瞬間衝擊力模擬碰撞/地面不平
    2. 持續風力 (Continuous Wind): 恆定方向力模擬通風/坡度

參數:
    push_force_range: tuple = (10, 30)   — 推力範圍 [N]
    push_interval_s: tuple = (5, 15)     — 推力間隔 [s]
    wind_force_range: tuple = (0, 5)     — 風力範圍 [N]
    wind_direction_range: tuple = (0, 2π) — 風向範圍 [rad]

訓練流程整合:
    - Random Push: EventTerm(mode="interval", interval_range_s=(5,15))
    - Continuous Wind: EventTerm(mode="reset") 生成, pre_step 施加

物理量級分析 (Charge 機器人 ~20kg):
    - 10N push → 0.5 m/s² 加速度，0.1s 內偏移 2.5cm
    - 30N push → 1.5 m/s²，0.1s 內偏移 7.5cm
    - 5N wind → 0.25 m/s² 持續加速（需要策略主動補償）

參考:
    - Kumar et al. (2021) "RMA: Rapid Motor Adaptation" — 外部擾動 DR
    - Lee et al. (2020) "Learning Quadrupedal Locomotion over Challenging Terrain"
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def apply_random_push(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float] = (10.0, 30.0),
    push_ratio: float = 0.15,
    asset_name: str = "robot",
) -> None:
    """隨機推力 — 模擬突然的外部衝擊。

    物理意義: 真實環境中機器人可能被行人碰撞、經過門檻/
    地面凹凸、或受到其他移動物體的推擠。
    這些瞬間衝擊要求策略具備快速恢復平衡的能力。

    實現: 以 interval 模式調用，每次隨機選擇 15% 的 env
    施加一個隨機方向的水平推力，持續一個物理步。

    Args:
        env: 環境實例
        env_ids: 當前活躍的環境 ID
        force_range: 推力大小範圍 [N]
        push_ratio: 被推的環境比例 (0.15 = 15%)
        asset_name: 機器人資產名稱
    """
    from isaaclab.assets import Articulation

    asset: Articulation = env.scene[asset_name]
    num_envs = len(env_ids)
    if num_envs == 0:
        return

    # 隨機選擇要推的環境
    num_push = max(1, int(num_envs * push_ratio))
    push_indices = torch.randint(0, num_envs, (num_push,), device=env.device)
    push_env_ids = env_ids[push_indices]

    # 隨機力大小
    force_mag = torch.empty(num_push, device=env.device).uniform_(
        force_range[0], force_range[1]
    )

    # 隨機方向（水平面 360 度）
    angles = torch.empty(num_push, device=env.device).uniform_(0, 2 * math.pi)
    forces = torch.zeros(num_push, 1, 3, device=env.device)
    forces[:, 0, 0] = torch.cos(angles) * force_mag
    forces[:, 0, 1] = torch.sin(angles) * force_mag

    torques = torch.zeros_like(forces)

    # 施加瞬間推力（僅作用一個 sim step）
    # body_ids=[0] = base_link；forces shape (N,1,3) 必須與 body_ids 長度一致
    asset.instantaneous_wrench_composer.add_forces_and_torques(
        forces=forces,
        torques=torques,
        body_ids=[0],
        env_ids=push_env_ids,
    )


def apply_continuous_wind(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    wind_force_range: tuple[float, float] = (0.0, 5.0),
    asset_name: str = "robot",
) -> None:
    """持續風力 — 模擬恆定方向的外部力。

    物理意義: 室外環境有風（1-5 m/s 微風 → 1-5N 阻力），
    室內有通風系統氣流。斜坡也等效於重力分量的持續力。
    策略需要學會在持續外力下維持航向。

    實現: 每次 reset 時為每個 env 採樣風力大小和方向，
    整個 episode 保持不變。在 pre_physics_step 中施加。

    Args:
        env: 環境實例
        env_ids: 需要重新採樣的環境 ID
        wind_force_range: 風力範圍 [N]
        asset_name: 機器人資產名稱
    """
    from isaaclab.assets import Articulation

    asset: Articulation = env.scene[asset_name]
    num_envs = len(env_ids)
    if num_envs == 0:
        return

    # 生成或重新採樣風力
    if not hasattr(env, "_wind_forces"):
        env._wind_forces = torch.zeros(env.num_envs, 3, device=env.device)

    # 為指定 env 採樣新的風力
    wind_mag = torch.empty(num_envs, device=env.device).uniform_(
        wind_force_range[0], wind_force_range[1]
    )
    wind_angles = torch.empty(num_envs, device=env.device).uniform_(0, 2 * math.pi)
    env._wind_forces[env_ids, 0] = torch.cos(wind_angles) * wind_mag
    env._wind_forces[env_ids, 1] = torch.sin(wind_angles) * wind_mag

    # 施加持續風力（permanent wrench，跨 step 持續作用直到下次 reset）
    all_forces = env._wind_forces.unsqueeze(1)  # [N, 1, 3]
    all_torques = torch.zeros_like(all_forces)

    # body_ids=[0] = base_link；forces shape (N,1,3) 必須與 body_ids 長度一致
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=all_forces,
        torques=all_torques,
        body_ids=[0],
    )


__all__ = [
    "apply_random_push",
    "apply_continuous_wind",
]
