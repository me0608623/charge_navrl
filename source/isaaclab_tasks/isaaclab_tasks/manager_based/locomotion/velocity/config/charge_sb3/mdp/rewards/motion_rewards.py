"""運動相關獎勵函數

包含所有與運動、速度、時間相關的獎勵函數：
- 前進速度獎勵
- 超時懲罰
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .utils import _check_reward_term


def forward_velocity_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_speed: float = 0.0,
    max_speed: float | None = None,
) -> torch.Tensor:
    """鼓勵機器人向前走的速度獎勵
    
    獎勵機器人「朝自己前方」的速度分量（機器人座標系 X 軸）。
    只獎勵前進速度，不獎勵後退或側向。
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        min_speed: 最小速度門檻（低於此速度不給獎勵）
        max_speed: 最大速度截斷（避免獎勵過大），None 表示不截斷
    
    Returns:
        shape [num_envs]：獎勵值 [0, max_speed]
    """
    # 獲取機器人狀態
    asset: Articulation = env.scene[asset_cfg.name]
    robot_vel_w = asset.data.root_lin_vel_w[:, :3]  # 世界座標速度
    robot_quat_w = asset.data.root_quat_w  # 世界座標姿態
    
    # 計算機器人前方向量（世界座標）
    num_envs = robot_vel_w.shape[0]
    forward_vec = torch.zeros(num_envs, 3, device=env.device)
    forward_vec[:, 0] = 1.0  # 機器人座標系 X 軸是前方
    forward_w = math_utils.quat_apply(robot_quat_w, forward_vec)
    
    # 計算前進速度分量（dot product）
    forward_speed = torch.sum(robot_vel_w * forward_w, dim=1)
    forward_speed = torch.clamp(forward_speed, min=0.0)  # 只獎勵前進
    
    # 最小速度門檻
    if min_speed > 0.0:
        forward_speed = torch.where(
            forward_speed >= min_speed,
            forward_speed,
            torch.zeros_like(forward_speed),
        )
    
    # 最大速度截斷
    if max_speed is not None:
        forward_speed = torch.clamp(forward_speed, 0.0, max_speed)
    
    # 清理 NaN/Inf
    forward_speed = torch.nan_to_num(forward_speed, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 檢查 reward term
    forward_speed = _check_reward_term("forward_velocity_reward", forward_speed, env, raise_on_error=True)
    
    return forward_speed


def time_out_penalty(env: ManagerBasedRLEnv, weight: float = -1.0) -> torch.Tensor:
    """超時懲罰
    
    只對超時的環境施加懲罰，不對成功到達目標的環境懲罰。
    
    Args:
        env: 環境實例
        weight: 懲罰權重（已棄用，由 charge_env_cfg.py 控制）
    
    Returns:
        shape [num_envs]：0 或 weight
        只有超時終止的環境才會得到懲罰，成功到達目標的環境不會被懲罰
    
    修復說明：
    - 使用 time_outs 屬性而不是 terminated 屬性
    - time_outs 只包含超時終止，不包含成功/失敗終止
    - 這樣可以避免對成功完成的任務也施加懲罰
    """
    # 獲取超時狀態（只包含超時終止，不包含成功/失敗終止）
    time_outs = env.termination_manager.time_outs
    # shape: [num_envs]：布林張量
    # True = 環境因為超時而終止
    # False = 環境未超時（可能還在運行，或因為其他原因終止）
    
    # 創建懲罰（只有超時才懲罰）
    reward = torch.where(
        time_outs,  # 只有超時的環境才給懲罰
        torch.full_like(time_outs, weight, dtype=torch.float32),  # 給懲罰
        torch.zeros_like(time_outs, dtype=torch.float32)  # 不給懲罰
    )
    
    # 安全處理：清理 NaN/Inf 並限制範圍
    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
    reward = torch.clamp(reward, -100.0, 0.0)  # 限制超時懲罰範圍
    
    # 檢查 reward term（防止 PPO std>=0 錯誤）
    reward = _check_reward_term("time_out_penalty", reward, env, raise_on_error=True)
    
    return reward


def forward_motion_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """正向前進獎勵 (姿態控制能力 - 差速驅動版本)
    
    實現論文的方向性引導,針對差速驅動機器人調整:
    - 前進 (vx > 0): 正獎勵 (最高優先級)
    - 後退 (vx < 0): 中等懲罰
    - 過度旋轉 (|ω| > threshold): 輕微懲罰 (避免原地打轉)
    
    注意: 差速驅動機器人沒有橫向速度 (vy),只有線性速度 (vx) 和角速度 (ω)
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        
    Returns:
        shape [num_envs]: 獎勵值 [-0.8, 1.0]
        - 正向前進: [0, 1.0]
        - 倒退: [-0.5, 0]
        - 過度旋轉: [-0.3, 0]
    """
    # 獲取機器人本體坐標系的速度
    asset: Articulation = env.scene[asset_cfg.name]
    velocity_b = asset.data.root_lin_vel_b  # 機器人本體坐標系線性速度
    angular_vel_b = asset.data.root_ang_vel_b  # 機器人本體坐標系角速度
    
    vx = velocity_b[:, 0]  # 前進方向 (X 軸)
    omega_z = angular_vel_b[:, 2]  # 繞 Z 軸旋轉 (yaw)
    
    # V_vx: 前進獎勵 (0 to 1.0)
    # 只獎勵正向前進,速度越快獎勵越高 (但限制在 1.0 m/s)
    forward_reward = torch.clamp(vx, 0.0, 1.0)
    
    # V_v-x: 後退懲罰 (0 to -0.5)
    # 倒退會受到中等懲罰,但不會太嚴重 (允許短暫倒退調整姿態)
    backward_penalty = torch.clamp(-vx, 0.0, 0.5)
    
    # V_ω: 過度旋轉懲罰 (0 to -0.3)
    # 懲罰過度的原地旋轉 (|ω| > 1.0 rad/s),鼓勵平滑轉向
    # 差速驅動應該「邊走邊轉」而非「停下來原地打轉」
    excessive_rotation_threshold = 1.0  # rad/s
    excessive_rotation = torch.clamp(torch.abs(omega_z) - excessive_rotation_threshold, 0.0, 2.0)
    rotation_penalty = excessive_rotation * 0.15  # 最大懲罰 0.3
    
    # V_spin: 原地旋轉懲罰 (Non-Holonomic Penalty)
    # 懲罰 "原地打轉" 行為：當線速度很低 (|vx| < 0.2) 但角速度很高 (|ω| > 0.5)
    # spin_penalty = (|ω| - 0.5) * (1.0 - |vx|/0.2)
    min_move_speed = 0.2
    spin_threshold = 0.5
    
    # 計算靜止系數 (0.0=動, 1.0=靜)
    stillness = torch.clamp(1.0 - torch.abs(vx) / min_move_speed, 0.0, 1.0)
    # 計算旋轉強度
    spinning = torch.clamp(torch.abs(omega_z) - spin_threshold, 0.0, 5.0)
    
    # 懲罰項：只有在 "又不動又轉" 時才會觸發
    spin_penalty = spinning * stillness * 1.0  # 權重 1.0，確保足夠痛
    
    # 總獎勵: V_vx - V_v-x - V_ω - V_spin
    reward = forward_reward - backward_penalty - rotation_penalty - spin_penalty

    # [GATED REWARD IMPLEMENTATION]
    # 檢查機器人是否 "Upright" (Z 軸朝上)
    # 如果機器人翻車（Z 軸分量小），則不給予任何前進獎勵
    # 這防止 Agent 學會 "衝刺然後翻車滑行" 的作弊策略
    robot_quat_w = asset.data.root_quat_w
    
    # 簡單方法：檢查 root_quat_w 轉換後的 Z 軸向量的 Z 分量
    # 構造局部 Z 軸向量 (0, 0, 1)
    vec_z = torch.zeros_like(asset.data.root_lin_vel_w)
    vec_z[:, 2] = 1.0
    # 旋轉到世界坐標系
    vec_z_w = math_utils.quat_apply(robot_quat_w, vec_z)
    # 檢查 Z 分量 (upward component)
    uprightness = vec_z_w[:, 2]
    
    # 定義閾值範圍
    # lower: 0.5 (約 60度) - 低於此值視為完全翻車，獎勵歸零
    # upper: 0.8 (約 37度) - 高於此值視為穩定，全額獎勵
    upright_threshold_min = 0.5
    upright_threshold_max = 0.8
    
    # 計算 Soft Gate 係數 (0.0 to 1.0)
    # scale = (x - min) / (max - min)
    gate_scale = (uprightness - upright_threshold_min) / (upright_threshold_max - upright_threshold_min)
    gate_scale = torch.clamp(gate_scale, 0.0, 1.0)
    
    # 應用 Soft Gate：將獎勵乘以係數
    # 這樣在輕微傾斜時 (0.5 < x < 0.8) 仍能獲得部分獎勵，保留梯度指引 Agent 恢復姿態
    reward = reward * gate_scale
    
    # 清理 NaN/Inf
    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 檢查 reward term
    reward = _check_reward_term("forward_motion_reward", reward, env, raise_on_error=True)
    
    return reward


def action_rate_penalty(
    env: ManagerBasedRLEnv,
    angular_weight: float = 2.0,
) -> torch.Tensor:
    """動作變化率懲罰（Δaction penalty）

    懲罰連續時間步的動作變化，鼓勵平滑控制。
    特別針對角速度變化（action[:, 1]）進行加重懲罰，以消除蛇行行為。

    設計理念：
    - 蛇行行為的根本原因是策略頻繁切換旋轉方向
    - 透過懲罰 Δaction（特別是角速度變化），強制策略輸出平滑
    - angular_weight > 1.0 表示角速度變化比線速度變化受到更重的懲罰

    Args:
        env: 環境實例
        angular_weight: 角速度變化的懲罰權重倍數（默認 2.0，即角速度變化懲罰是線速度的 2 倍）

    Returns:
        shape [num_envs]：懲罰值 [0, +∞)
        值越大表示動作變化越劇烈

    使用建議：
        weight=-0.5 到 -1.0，配合 angular_weight=2.0
    """
    # 獲取當前動作
    current_action = env.action_manager.action  # [num_envs, 2]

    # 初始化上一步動作緩存
    if not hasattr(env, "_prev_actions_for_smoothness"):
        env._prev_actions_for_smoothness = torch.zeros_like(current_action)

    # 計算動作變化
    action_diff = current_action - env._prev_actions_for_smoothness

    # 線速度變化懲罰（action[:, 0]）
    linear_diff_sq = action_diff[:, 0] ** 2

    # 角速度變化懲罰（action[:, 1]），加重權重
    angular_diff_sq = action_diff[:, 1] ** 2 * angular_weight

    # 總懲罰
    penalty = linear_diff_sq + angular_diff_sq

    # 更新緩存
    env._prev_actions_for_smoothness = current_action.clone().detach()

    # 重置時清除緩存（避免跨 episode 的錯誤懲罰）
    reset_mask = env.episode_length_buf == 0
    if reset_mask.any():
        env._prev_actions_for_smoothness[reset_mask] = current_action[reset_mask].detach()
        penalty[reset_mask] = 0.0  # 重置後第一步不懲罰

    # 安全處理
    penalty = torch.nan_to_num(penalty, nan=0.0, posinf=10.0, neginf=0.0)
    penalty = torch.clamp(penalty, 0.0, 10.0)

    # 檢查 reward term
    penalty = _check_reward_term("action_rate_penalty", penalty, env, raise_on_error=True)

    return penalty


def move_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    k: float = 1.0,
    b: float = 0.0,
) -> torch.Tensor:
    """移動獎勵 (R_move)：鼓勵車輛保持移動，避免因恐懼碰撞而停滯
    
    設計公式：k × (speed_x + b)
    - speed_x：機器人本體座標系 X 軸正向速度（前進方向）
    - k：係數（默認 1.0）
    - b：偏移量（默認 0.0，可設為負值以鼓勵更高速度）
    
    目的：
    - 鼓勵車輛保持移動，避免因恐懼碰撞而停滯
    - 確保 X 軸正向速度越高分越高
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        k: 係數（默認 1.0）
        b: 偏移量（默認 0.0）
    
    Returns:
        shape [num_envs]：獎勵值 [k×b, k×(max_speed + b)]
        只獎勵正向速度（vx > 0），後退速度不給獎勵
    """
    # 獲取機器人本體座標系速度
    asset: Articulation = env.scene[asset_cfg.name]
    velocity_b = asset.data.root_lin_vel_b  # 機器人本體座標系線性速度
    
    vx = velocity_b[:, 0]  # 前進方向 (X 軸)
    
    # 只獎勵正向速度（前進），不獎勵後退
    speed_x = torch.clamp(vx, min=0.0)
    
    # 計算獎勵：k × (speed_x + b)
    reward = k * (speed_x + b)
    
    # 安全處理
    reward = torch.nan_to_num(reward, nan=0.0, posinf=k * 10.0, neginf=0.0)
    reward = torch.clamp(reward, 0.0, k * 10.0)  # 限制最大獎勵
    
    # 檢查 reward term
    reward = _check_reward_term("move_reward", reward, env, raise_on_error=True)

    return reward


def action_smoothness_linear(
    env: ManagerBasedRLEnv,
    angular_weight: float = 1.0,
) -> torch.Tensor:
    """動作平滑懲罰 - 線性版本 (Action Smoothness - Linear)

    實現 Pull-Push Theory 中的平滑懲罰：P_smooth = -0.1 × ||Δv||

    與 action_rate_penalty 的區別：
    - action_rate_penalty: 使用平方項 (Δv)²，對大變化懲罰更重
    - action_smoothness_linear: 使用線性項 |Δv|，懲罰更平緩

    Args:
        env: 環境實例
        angular_weight: 角速度變化的懲罰權重倍數（默認 1.0）

    Returns:
        shape [num_envs]：懲罰值 [0, +∞)
        值越大表示動作變化越劇烈

    使用建議：
        weight=-0.1，配合 angular_weight=1.0
    """
    # 獲取當前動作
    current_action = env.action_manager.action  # [num_envs, 2]

    # 初始化上一步動作緩存
    if not hasattr(env, "_prev_actions_for_smoothness_linear"):
        env._prev_actions_for_smoothness_linear = torch.zeros_like(current_action)

    # 計算動作變化（使用絕對值而非平方）
    action_diff = current_action - env._prev_actions_for_smoothness_linear

    # 線速度變化懲罰（action[:, 0]）- 使用絕對值
    linear_diff = torch.abs(action_diff[:, 0])

    # 角速度變化懲罰（action[:, 1]）- 使用絕對值並加權
    angular_diff = torch.abs(action_diff[:, 1]) * angular_weight

    # 總懲罰（線性相加）
    penalty = linear_diff + angular_diff

    # 更新緩存
    env._prev_actions_for_smoothness_linear = current_action.clone().detach()

    # 重置時清除緩存（避免跨 episode 的錯誤懲罰）
    reset_mask = env.episode_length_buf == 0
    if reset_mask.any():
        env._prev_actions_for_smoothness_linear[reset_mask] = current_action[reset_mask].detach()
        penalty[reset_mask] = 0.0  # 重置後第一步不懲罰

    # 安全處理
    penalty = torch.nan_to_num(penalty, nan=0.0, posinf=10.0, neginf=0.0)
    penalty = torch.clamp(penalty, 0.0, 10.0)

    # 檢查 reward term
    penalty = _check_reward_term("action_smoothness_linear", penalty, env, raise_on_error=True)

    return penalty


