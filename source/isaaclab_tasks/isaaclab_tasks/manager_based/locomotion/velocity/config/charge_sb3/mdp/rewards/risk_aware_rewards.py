"""風險感知獎勵函數 (Risk-Aware Reward Functions)

基於論文「技術貢獻」設計的獎勵函數，解決兩大核心問題：
1. 消除「冰凍」與「震盪」行為
2. 引導穩定收斂至「防禦性且具備推進力」的行為

技術貢獻：
────────────────────────────────────────────────────────────────────────────────
1. 風險感知之平滑推進獎勵 (Risk-Aware Smooth Progress Reward)
   - 公式：R_prog = v_proj × (1 - e^(-α × TTC_min))
   - 物理意義：安全時鼓勵全速前進，危險時獎勵衰減

2. 動作平滑性懲罰 (Action Smoothness Penalty)
   - 公式：R_smooth = -λ × ||a_t - a_{t-1}||²
   - 目的：減少高頻震盪，輸出符合車輛運動學的平滑控制

學術價值：
────────────────────────────────────────────────────────────────────────────────
- 證明了時間維度上的動作正則化 (Temporal Action Regularization) 能迫使策略網路
  輸出連續且符合車輛運動學的平滑控制指令
- 動態獎勵調變機制能引導機器人學會「在安全時加速，在危險時減速繞行」
- 極大提升了收斂後的行駛流暢度，避免「原地煞車」或「原地打轉」
"""

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .utils import _check_reward_term
from .ttc_defensive_driving import compute_ttc


def risk_aware_progress_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    alpha: float = 1.0,
    ttc_threshold: float = 3.0,
    max_velocity: float = 1.5,
    goal_weight: float = 1.0,
) -> torch.Tensor:
    """風險感知之平滑推進獎勵 (Risk-Aware Smooth Progress Reward)

    ═══════════════════════════════════════════════════════════════════════════
                    技術貢獻 1：風險感知之平滑推進獎勵
    ═══════════════════════════════════════════════════════════════════════════

    核心思想：
    ────────────────────────────────────────────────────────────────────────────
    不要只懲罰碰撞，必須將「前進的動力」與「當前環境的風險」掛鉤。

    數學公式：
    ────────────────────────────────────────────────────────────────────────────
    R_prog = v_proj × (1 - e^(-α × TTC_min))

    其中：
    - v_proj：機器人速度在目標方向上的投影（前進速度）
    - TTC_min：最危險障礙物的碰撞時間（取最小值）
    - α：風險敏感係數（控制衰減速度）
    - (1 - e^(-α × TTC_min))：風險調節因子，範圍 [0, 1)

    物理意義：
    ────────────────────────────────────────────────────────────────────────────
    1. 當 TTC_min 很大（環境安全）時：
       - e^(-α × TTC_min) → 0
       - 風險因子 → 1
       - R_prog ≈ v_proj（鼓勵全速前進）

    2. 當 TTC_min 逼近安全閾值（環境危險）時：
       - e^(-α × TTC_min) → 1
       - 風險因子 → 0
       - R_prog → 0（前進獎勵衰減）

    3. 這樣能引導機器人學會「在安全時加速，在危險時減速繞行」

    學術貢獻：
    ────────────────────────────────────────────────────────────────────────────
    - 解決「冰凍機器人 (Freezing Robot)」問題
    - 避免粗暴急煞，提升行駛流暢度
    - 實現動態風險感知的推進策略

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        alpha: 風險敏感係數（控制衰減速度，越大衰減越快）
        ttc_threshold: TTC 閾值（秒），超過此值視為安全
        max_velocity: 最大速度（用於歸一化）
        goal_weight: 目標權重

    Returns:
        獎勵張量 [num_envs]，範圍 [0, max_velocity]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    num_envs = env.num_envs
    device = env.device

    # 獲取機器人狀態
    robot_pos = robot.data.root_pos_w[:, :2]  # [num_envs, 2]
    robot_vel_w = robot.data.root_lin_vel_w[:, :2]  # [num_envs, 2]
    robot_quat_w = robot.data.root_quat_w

    # 獲取目標位置
    if hasattr(env, '_local_goal_world') and env._local_goal_world is not None:
        goal_pos = env._local_goal_world
    else:
        goal_pos = env.command_manager.get_command("goal_command")[:, :2]

    # 計算目標方向（單位向量）
    goal_dir = goal_pos - robot_pos  # [num_envs, 2]
    goal_dist = torch.norm(goal_dir, dim=1, keepdim=True) + 1e-6  # [num_envs, 1]
    goal_unit = goal_dir / goal_dist  # [num_envs, 2]

    # 計算速度在目標方向上的投影 (v_proj)
    # v_proj = v · goal_unit（點積）
    v_proj = (robot_vel_w * goal_unit).sum(dim=1)  # [num_envs]

    # 歸一化到 [0, 1]
    v_proj_normalized = (v_proj / max_velocity).clamp(0.0, 1.0)

    # 計算 TTC_min（最危險障礙物的碰撞時間）
    ttc_min = torch.full((num_envs,), ttc_threshold, device=device)  # 默認為安全

    # 檢查是否有動態障礙物資料
    if hasattr(env, '_dynamic_obstacle_pos') and env._dynamic_obstacle_pos is not None:
        obstacle_pos = env._dynamic_obstacle_pos
        obstacle_vel = env._dynamic_obstacle_vel

        if obstacle_pos.shape[1] > 0:
            # 計算 TTC
            ttc = compute_ttc(robot_pos, robot_vel_w, obstacle_pos, obstacle_vel)

            # 過濾無效值（inf, nan）並取最小值
            ttc_valid = torch.where(
                (ttc > 0) & (ttc < float('inf')) & torch.isfinite(ttc),
                ttc,
                torch.full_like(ttc, ttc_threshold)
            )
            ttc_min, _ = ttc_valid.min(dim=1)  # [num_envs]

    # 裁剪 TTC 到合理範圍
    ttc_min = ttc_min.clamp(0.0, ttc_threshold)

    # 計算風險調節因子：(1 - e^(-α × TTC_min))
    # 當 TTC_min 很大時，因子接近 1；當 TTC_min 很小時，因子接近 0
    risk_factor = 1.0 - torch.exp(-alpha * ttc_min)  # [num_envs]

    # 計算最終獎勵：R_prog = v_proj × risk_factor
    reward = v_proj_normalized * risk_factor * goal_weight

    # 安全處理
    reward = torch.nan_to_num(reward, nan=0.0, posinf=max_velocity, neginf=0.0)
    reward = reward.clamp(0.0, max_velocity)

    # 檢查 reward term
    reward = _check_reward_term("risk_aware_progress_reward", reward, env, raise_on_error=True)

    return reward


def temporal_action_smoothness_penalty(
    env: ManagerBasedRLEnv,
    lambda_weight: float = 0.1,
    separate_components: bool = False,
) -> torch.Tensor:
    """動作平滑性懲罰 (Temporal Action Smoothness Penalty)

    ═══════════════════════════════════════════════════════════════════════════
                    技術貢獻 2：動作平滑性懲罰
    ═══════════════════════════════════════════════════════════════════════════

    核心思想：
    ────────────────────────────────────────────────────────────────────────────
    RL 輸出的控制指令往往呈現高頻震盪（Bang-Bang Control），這在實車部署時
    會損壞馬達，也會讓訓練難以收斂。

    數學公式：
    ────────────────────────────────────────────────────────────────────────────
    R_smooth = -λ × ||a_t - a_{t-1}||²

    其中：
    - a_t：當前動作
    - a_{t-1}：上一幀動作
    - λ：懲罰權重（控制平滑程度）

    學術貢獻：
    ────────────────────────────────────────────────────────────────────────────
    證明了施加時間維度上的動作正則化 (Temporal Action Regularization)，
    能迫使策略網路輸出連續且符合車輛運動學 (Kinematics) 的平滑控制指令，
    大幅降低收斂過程中的策略震盪。

    設計要點：
    ────────────────────────────────────────────────────────────────────────────
    1. 使用 L2 範數計算動作變化量
    2. 懲罰權重 λ 需要適當調整：
       - 過大：動作過於保守，影響響應速度
       - 過小：無法抑制震盪
    3. 可以分別計算線速度和角速度的平滑度

    Args:
        env: 環境實例
        lambda_weight: 懲罰權重（默認 0.1）
        separate_components: 是否分別計算線速度和角速度的平滑度

    Returns:
        懲罰張量 [num_envs]，範圍 [-inf, 0]
    """
    # 獲取當前動作
    current_actions = env.action_manager.action  # [num_envs, action_dim]

    # 獲取上一幀動作
    if hasattr(env.action_manager, 'previous_action'):
        previous_actions = env.action_manager.previous_action
    else:
        # 如果沒有上一幀動作，初始化為 0
        previous_actions = torch.zeros_like(current_actions)

    # 計算動作變化量
    action_diff = current_actions - previous_actions  # [num_envs, action_dim]

    if separate_components and action_diff.shape[1] >= 2:
        # 分別計算線速度和角速度的平滑度
        linear_diff = action_diff[:, 0]  # 線速度變化
        angular_diff = action_diff[:, 1]  # 角速度變化

        # 使用不同的權重
        linear_penalty = linear_diff ** 2
        angular_penalty = angular_diff ** 2 * 2.0  # 角速度震盪更嚴重

        penalty = -(linear_penalty + angular_penalty) * lambda_weight
    else:
        # 計算 L2 範數的平方
        # ||a_t - a_{t-1}||²
        action_change_sq = (action_diff ** 2).sum(dim=1)  # [num_envs]
        penalty = -lambda_weight * action_change_sq

    # 安全處理
    penalty = torch.nan_to_num(penalty, nan=0.0, posinf=0.0, neginf=-1.0)

    # 檢查 reward term
    penalty = _check_reward_term("temporal_action_smoothness_penalty", penalty, env, raise_on_error=True)

    return penalty


def risk_aware_smooth_navigation_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # 風險感知推進參數
    alpha: float = 1.0,
    ttc_threshold: float = 3.0,
    max_velocity: float = 1.5,
    goal_weight: float = 1.0,
    # 動作平滑參數
    lambda_smooth: float = 0.1,
    # 總體權重
    progress_weight: float = 1.0,
    smooth_weight: float = 0.5,
) -> torch.Tensor:
    """風險感知平滑導航總獎勵 (Risk-Aware Smooth Navigation Total Reward)

    結合兩個技術貢獻的總獎勵函數：
    1. 風險感知之平滑推進獎勵
    2. 動作平滑性懲罰

    總獎勵公式：
    ────────────────────────────────────────────────────────────────────────────
    R_total = w_prog × R_prog + w_smooth × R_smooth

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        alpha: 風險敏感係數
        ttc_threshold: TTC 閾值
        max_velocity: 最大速度
        goal_weight: 目標權重
        lambda_smooth: 平滑懲罰權重
        progress_weight: 推進獎勵權重
        smooth_weight: 平滑懲罰權重

    Returns:
        總獎勵張量 [num_envs]
    """
    # 計算風險感知推進獎勵
    progress_reward = risk_aware_progress_reward(
        env,
        robot_cfg=robot_cfg,
        alpha=alpha,
        ttc_threshold=ttc_threshold,
        max_velocity=max_velocity,
        goal_weight=goal_weight,
    )

    # 計算動作平滑懲罰
    smooth_penalty = temporal_action_smoothness_penalty(
        env,
        lambda_weight=lambda_smooth,
    )

    # 總獎勵
    total_reward = progress_weight * progress_reward + smooth_weight * smooth_penalty

    return total_reward


def adaptive_action_smoothness_penalty(
    env: ManagerBasedRLEnv,
    base_lambda: float = 0.1,
    velocity_threshold: float = 0.3,
    high_speed_multiplier: float = 2.0,
) -> torch.Tensor:
    """自適應動作平滑性懲罰 (Adaptive Action Smoothness Penalty)

    根據當前速度自適應調整平滑懲罰的強度：
    - 低速時：允許較大的動作變化（方便轉向）
    - 高速時：嚴格限制動作變化（避免危險）

    物理意義：
    ────────────────────────────────────────────────────────────────────────────
    高速行駛時，突然的轉向或煞車更危險，因此需要更嚴格的平滑約束。

    Args:
        env: 環境實例
        base_lambda: 基礎懲罰權重
        velocity_threshold: 速度閾值（區分低速/高速）
        high_speed_multiplier: 高速時的懲罰倍數

    Returns:
        懲罰張量 [num_envs]
    """
    # 獲取當前動作和上一幀動作
    current_actions = env.action_manager.action
    previous_actions = getattr(env.action_manager, 'previous_action', torch.zeros_like(current_actions))

    # 計算動作變化量
    action_diff = current_actions - previous_actions
    action_change_sq = (action_diff ** 2).sum(dim=1)

    # 獲取當前速度（用於自適應調整）
    # 從動作推斷速度（線速度在 action[0]）
    current_velocity = torch.abs(current_actions[:, 0])

    # 自適應權重：高速時懲罰更重
    adaptive_lambda = torch.where(
        current_velocity > velocity_threshold,
        base_lambda * high_speed_multiplier,
        base_lambda
    )

    # 計算懲罰
    penalty = -adaptive_lambda * action_change_sq

    # 安全處理
    penalty = torch.nan_to_num(penalty, nan=0.0, posinf=0.0, neginf=-1.0)

    return penalty


def exponential_decay_progress_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    alpha: float = 0.5,
    ttc_threshold: float = 2.0,
    max_reward: float = 1.0,
) -> torch.Tensor:
    """指數衰減推進獎勵 (Exponential Decay Progress Reward)

    簡化版本的風險感知推進獎勵，使用指數衰減函數：
    R = max_reward × e^(-α / max(TTC_min, ε))

    當 TTC 很大時，獎勵接近 max_reward
    當 TTC 很小時，獎勵快速衰減

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        alpha: 衰減係數
        ttc_threshold: TTC 閾值
        max_reward: 最大獎勵值

    Returns:
        獎勵張量 [num_envs]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    num_envs = env.num_envs
    device = env.device

    # 計算 TTC_min
    ttc_min = torch.full((num_envs,), ttc_threshold, device=device)

    if hasattr(env, '_dynamic_obstacle_pos') and env._dynamic_obstacle_pos is not None:
        obstacle_pos = env._dynamic_obstacle_pos
        obstacle_vel = env._dynamic_obstacle_vel

        if obstacle_pos.shape[1] > 0:
            robot_pos = robot.data.root_pos_w[:, :2]
            robot_vel_w = robot.data.root_lin_vel_w[:, :2]

            ttc = compute_ttc(robot_pos, robot_vel_w, obstacle_pos, obstacle_vel)

            ttc_valid = torch.where(
                (ttc > 0) & (ttc < float('inf')) & torch.isfinite(ttc),
                ttc,
                torch.full_like(ttc, ttc_threshold)
            )
            ttc_min, _ = ttc_valid.min(dim=1)

    # 指數衰減獎勵
    reward = max_reward * torch.exp(-alpha / (ttc_min.clamp(min=0.1) + 1e-6))

    # 安全處理
    reward = torch.nan_to_num(reward, nan=0.0, posinf=max_reward, neginf=0.0)
    reward = reward.clamp(0.0, max_reward)

    return reward


__all__ = [
    "risk_aware_progress_reward",
    "temporal_action_smoothness_penalty",
    "risk_aware_smooth_navigation_reward",
    "adaptive_action_smoothness_penalty",
    "exponential_decay_progress_reward",
]
