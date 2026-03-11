"""動力學域隨機化 (Physics Domain Randomization)

模擬真實機器人的物理參數不確定性，提高策略對動力學變化的魯棒性。

功能：
    1. 質量隨機化: ±15% 模擬載重變化（電池/貨物）
    2. 地面摩擦力隨機化: ±30% 模擬不同地面（瓷磚/地毯/水泥）
    3. 質心偏移: ±5cm 模擬組裝誤差或載重不均

參數:
    mass_scale: tuple = (0.85, 1.15)      — 質量縮放範圍
    friction_scale: tuple = (0.7, 1.3)    — 摩擦力縮放範圍
    com_offset_range: float = 0.05        — 質心偏移最大值 [m]

訓練流程整合:
    在 EventCfg 中以 mode="reset" 使用，每次 episode 開始時
    重新採樣物理參數，讓 agent 在不同動力學下學習。

Isaac Lab API:
    - asset.root_physx_view.set_masses(): 修改質量
    - asset.root_physx_view.set_coms(): 修改質心
    - env.scene.terrain.physics_material: 修改摩擦力

參考:
    - Peng et al. (2018) "Sim-to-Real Transfer with Dynamics Randomization"
    - OpenAI (2019) "Solving Rubik's Cube" — 質量/摩擦 DR
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def randomize_robot_mass(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    mass_scale: tuple[float, float] = (0.85, 1.15),
    asset_name: str = "robot",
) -> None:
    """隨機化機器人質量 — 模擬載重變化。

    物理意義: 真實機器人的有效質量會因電池電量（鋰電池密度變化）、
    攜帶貨物、附件增減而改變。±15% 覆蓋典型場景。

    實現: 通過 Isaac Lab 的 PhysX view API 直接修改剛體質量。
    每次 episode reset 時重新採樣。

    Args:
        env: 環境實例
        env_ids: 需要隨機化的環境 ID
        mass_scale: 質量縮放範圍 (min, max)，1.0 = 原始質量
        asset_name: 機器人資產名稱
    """
    from isaaclab.assets import Articulation

    asset: Articulation = env.scene[asset_name]
    num_envs = len(env_ids)
    if num_envs == 0:
        return

    # 獲取原始質量（首次調用時保存）
    if not hasattr(env, "_original_robot_masses"):
        env._original_robot_masses = asset.root_physx_view.get_masses().clone()

    # 按環境採樣隨機縮放因子
    scales = torch.empty(num_envs, 1, device=env.device).uniform_(
        mass_scale[0], mass_scale[1]
    )

    # 修改質量
    original = env._original_robot_masses[env_ids]
    new_masses = original * scales
    asset.root_physx_view.set_masses(new_masses, env_ids)


def randomize_ground_friction(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    friction_scale: tuple[float, float] = (0.7, 1.3),
    asset_name: str = "robot",
) -> None:
    """隨機化地面摩擦力 — 模擬不同地面材質。

    物理意義: 真實環境地面從光滑瓷磚 (μ≈0.3) 到粗糙水泥 (μ≈0.8)
    差異很大。通過隨機化摩擦力讓 agent 學會適應不同地面。

    實現: 修改機器人輪子與地面接觸的摩擦係數。
    Isaac Lab 中通過 material properties 修改。

    Args:
        env: 環境實例
        env_ids: 需要隨機化的環境 ID
        friction_scale: 摩擦力縮放範圍
        asset_name: 機器人資產名稱
    """
    from isaaclab.assets import Articulation

    asset: Articulation = env.scene[asset_name]
    num_envs = len(env_ids)
    if num_envs == 0:
        return

    # 獲取當前材質屬性
    if not hasattr(env, "_original_friction"):
        try:
            mat_props = asset.root_physx_view.get_material_properties()
            env._original_friction = mat_props.clone()
        except Exception:
            # 如果 API 不可用，跳過
            return

    # 採樣縮放因子
    scales = torch.empty(num_envs, 1, 1, device=env.device).uniform_(
        friction_scale[0], friction_scale[1]
    )

    # 修改摩擦力 (static_friction, dynamic_friction, restitution)
    original = env._original_friction[env_ids]
    new_props = original.clone()
    new_props[..., :2] *= scales  # 只縮放 static 和 dynamic friction
    asset.root_physx_view.set_material_properties(new_props, env_ids)


def randomize_com_offset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    com_offset_range: float = 0.05,
    asset_name: str = "robot",
) -> None:
    """隨機化質心偏移 — 模擬組裝誤差或載重不均。

    物理意義: 真實機器人的質心不會精確在幾何中心。
    電池位置、線纜佈局、附件安裝都會造成質心偏移。
    ±5cm 的偏移會導致轉彎時的不對稱行為。

    實現: 通過 PhysX view 的 set_coms API 修改質心位置。

    Args:
        env: 環境實例
        env_ids: 需要隨機化的環境 ID
        com_offset_range: XY 方向最大偏移量 [m]
        asset_name: 機器人資產名稱
    """
    from isaaclab.assets import Articulation

    asset: Articulation = env.scene[asset_name]
    num_envs = len(env_ids)
    if num_envs == 0:
        return

    # 獲取原始質心
    if not hasattr(env, "_original_coms"):
        env._original_coms = asset.root_physx_view.get_coms().clone()

    # 隨機 XY 偏移
    offsets = torch.zeros(num_envs, 1, 3, device=env.device)
    offsets[:, 0, 0] = torch.empty(num_envs, device=env.device).uniform_(
        -com_offset_range, com_offset_range
    )
    offsets[:, 0, 1] = torch.empty(num_envs, device=env.device).uniform_(
        -com_offset_range, com_offset_range
    )
    # Z 偏移很小（高度方向影響傾斜穩定性）
    offsets[:, 0, 2] = torch.empty(num_envs, device=env.device).uniform_(
        -com_offset_range * 0.5, com_offset_range * 0.5
    )

    original = env._original_coms[env_ids]
    new_coms = original + offsets
    asset.root_physx_view.set_coms(new_coms, env_ids)


__all__ = [
    "randomize_robot_mass",
    "randomize_ground_friction",
    "randomize_com_offset",
]
