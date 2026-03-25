"""TTC 懲罰函數 (純 PyTorch 張量操作實現)

基於「特權物理蒸餾 (Privileged Physics Distillation)」概念，
使用純 PyTorch 張量操作實現的防禦性駕駛懲罰函數。

特點：
- 嚴禁使用 Python for 迴圈遍歷環境
- 使用 unsqueeze(1) 對齊機器人與障礙物的維度
- 沿著 dim=-1 計算物理量
- 使用遮罩 (mask) 過濾條件
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .utils import _check_reward_term


def ttc_penalty_pure(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tau: float = 1.5,
) -> torch.Tensor:
    """TTC 防禦性駕駛懲罰 (純 PyTorch 張量操作)

    使用純 PyTorch 張量操作實現，嚴禁使用 Python for 迴圈遍歷環境。

    物理計算：
        1. 相對位置 dp = p_o - p_r，相對速度 dv = v_o - v_r
        2. 距離 dist = ||dp||（沿 dim=-1）
        3. 接近速率 v_app = -(dp · dv) / (dist + ε)
        4. 碰撞時間 ttc = dist / (v_app + ε)

    遮罩與懲罰：
        - danger_mask = (z_o > 0) & (v_app > 0) & (ttc < tau)
        - 懲罰值 = -(tau - ttc) 只對 danger_mask 為 True 的元素生效
        - 最後對障礙物維度 dim=1 進行 sum 加總

    註：z_o > 0 是為了忽略被藏在地底下的 Empty 環境障礙物

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        tau: 安全時間閾值（秒），TTC < tau 時給予懲罰

    Returns:
        懲罰張量 [num_envs]，負值表示懲罰
    """
    robot: Articulation = env.scene[robot_cfg.name]
    num_envs = env.num_envs
    device = env.device

    # ========================================================================
    # 1. 取得機器人狀態 [num_envs, 2]
    # ========================================================================
    # p_r: 機器人 2D 座標, v_r: 機器人 2D 速度
    p_r = robot.data.root_pos_w[:, :2]  # [num_envs, 2]
    v_r = robot.data.root_lin_vel_w[:, :2]  # [num_envs, 2]

    # ========================================================================
    # 2. 取得障礙物狀態 [num_envs, num_obstacles, 2/1]
    # ========================================================================
    # 檢查是否有動態障礙物資料
    if not hasattr(env, '_dynamic_obstacle_pos') or env._dynamic_obstacle_pos is None:
        return torch.zeros(num_envs, device=device)

    p_o = env._dynamic_obstacle_pos  # [num_envs, num_obstacles, 2]
    v_o = env._dynamic_obstacle_vel  # [num_envs, num_obstacles, 2]
    z_o = env._dynamic_obstacle_pos_w[:, :, 2] if hasattr(env, '_dynamic_obstacle_pos_w') else torch.ones_like(p_o[..., :1])  # [num_envs, num_obstacles, 1]

    # 如果沒有障礙物，返回 0
    if p_o.shape[1] == 0:
        return torch.zeros(num_envs, device=device)

    num_obstacles = p_o.shape[1]

    # ========================================================================
    # 3. 對齊維度：使用 unsqueeze(1) 擴展機器人維度
    # ========================================================================
    # [num_envs, 2] -> [num_envs, 1, 2] -> [num_envs, num_obstacles, 2]
    p_r_exp = p_r.unsqueeze(1).expand(-1, num_obstacles, -1)
    v_r_exp = v_r.unsqueeze(1).expand(-1, num_obstacles, -1)

    # ========================================================================
    # 4. 物理計算 (沿著 dim=-1)
    # ========================================================================
    # 相對位置與相對速度
    dp = p_o - p_r_exp  # [num_envs, num_obstacles, 2]
    dv = v_o - v_r_exp  # [num_envs, num_obstacles, 2]

    # 距離 (沿著 dim=-1 計算範數)
    dist = torch.norm(dp, dim=-1, keepdim=True)  # [num_envs, num_obstacles, 1]

    # 接近速率 v_app = -(dp · dv) / (dist + ε)
    # 點積沿著 dim=-1
    dot_product = (dp * dv).sum(dim=-1, keepdim=True)  # [num_envs, num_obstacles, 1]
    epsilon = 1e-5
    v_app = -dot_product / (dist + epsilon)  # [num_envs, num_obstacles, 1]

    # 碰撞時間 TTC = dist / (v_app + ε)
    ttc = dist / (v_app + epsilon)  # [num_envs, num_obstacles, 1]

    # ========================================================================
    # 5. 建立遮罩
    # ========================================================================
    # danger_mask = (z_o > 0) & (v_app > 0) & (ttc < tau)
    # - z_o > 0: 忽略被藏在地底下的 Empty 環境障礙物
    # - v_app > 0: 只考慮正在靠近的障礙物
    # - ttc < tau: TTC 小於安全閾值
    danger_mask = (z_o > 0) & (v_app.squeeze(-1) > 0) & (ttc.squeeze(-1) < tau)  # [num_envs, num_obstacles]

    # ========================================================================
    # 6. 計算懲罰
    # ========================================================================
    # 懲罰值 = -(tau - ttc)，只對 danger_mask 為 True 的元素生效
    penalty_tensor = torch.zeros_like(ttc.squeeze(-1))  # [num_envs, num_obstacles]
    penalty_tensor = torch.where(
        danger_mask,
        -(tau - ttc.squeeze(-1)),
        torch.zeros_like(ttc.squeeze(-1))
    )

    # ========================================================================
    # 7. 對障礙物維度 dim=1 進行 sum 加總
    # ========================================================================
    penalty = penalty_tensor.sum(dim=1)  # [num_envs]

    # 檢查 reward term
    penalty = _check_reward_term("ttc_penalty", penalty, env, raise_on_error=True)

    return penalty


def ttc_penalty_pure_weighted(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tau: float = 1.5,
    weight: float = 1.0,
    max_penalty: float = 10.0,
) -> torch.Tensor:
    """帶權重的 TTC 防禦性駕駛懲罰

    與 ttc_penalty 相同，但增加了權重參數和最大懲罰限制。

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        tau: 安全時間閾值（秒）
        weight: 懲罰權重
        max_penalty: 最大懲罰值（絕對值）

    Returns:
        懲罰張量 [num_envs]，範圍 [-max_penalty, 0]
    """
    penalty = ttc_penalty_pure(env, robot_cfg, tau)

    # 應用權重
    penalty = penalty * weight

    # 裁剪到合理範圍
    penalty = torch.clamp(penalty, -max_penalty, 0.0)

    # 安全處理
    penalty = torch.nan_to_num(penalty, nan=0.0, posinf=0.0, neginf=-max_penalty)

    return penalty
