"""TTC 防禦性駕駛獎勵函數 (Time-to-Collision Defensive Driving Reward)

基於「特權物理蒸餾 (Privileged Physics Distillation)」概念實現：

核心思想：
-----------
傳統 RL 避障使用「距離」作為獎勵：
    - 距離 < 0.5m → 扣分
    - 這是「反應式避障」：車子要等到很近才煞車
    - 對於快速移動的障礙物，這樣太慢了！

特權物理蒸餾的做法：
    1. Critic 擁有「上帝視角」(那 50 維的障礙物位置+速度)
    2. 使用物理公式計算「碰撞時間 (TTC)」
    3. 如果 TTC < 安全閾值，給予嚴厲懲罰
    4. Actor 被迫從 LiDAR 歷史幀中學會辨識危險特徵
    5. 結果：提早 3 公尺就開始減速或繞路（防禦性駕駛！）

數學公式：
----------
假設機器人位置為 p_r，速度為 v_r；障礙物位置為 p_o，速度為 v_o

相對位置向量：Δp = p_o - p_r
相對速度向量：Δv = v_o - v_r

接近速率 (Approach Speed)：
    v_app = - (Δp · Δv) / ||Δp||
    - v_app > 0 表示雙方正在靠近
    - v_app <= 0 表示雙方正在遠離（安全）

碰撞時間 (TTC)：
    TTC = ||Δp|| / v_app

防禦性駕駛懲罰 (Defensive Penalty)：
    R_defensive = -k * (τ - TTC)  if v_app > 0 and TTC < τ
                = 0               otherwise

    其中 τ 是安全時間閾值（例如 1.5 秒）

參考論文：
---------
- NavRL: Learning Safe Flight in Dynamic Environments (IEEE RA-L 2025)
- Deep Imitation Learning of Sequential Comoving for Autonomous Driving
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .utils import _check_reward_term


def compute_ttc(
    robot_pos: torch.Tensor,
    robot_vel: torch.Tensor,
    obstacle_pos: torch.Tensor,
    obstacle_vel: torch.Tensor,
    min_distance: float = 0.1,
) -> torch.Tensor:
    """計算碰撞時間 (Time-to-Collision)

    使用精確的物理動力學公式計算機器人與障礙物的 TTC。

    數學公式：
    - 相對位置：Δp = p_o - p_r
    - 相對速度：Δv = v_o - v_r
    - 接近速率：v_app = - (Δp · Δv) / ||Δp||
    - 碰撞時間：TTC = ||Δp|| / v_app

    Args:
        robot_pos: 機器人位置 [num_envs, 2]
        robot_vel: 機器人速度 [num_envs, 2] (世界座標系)
        obstacle_pos: 障礙物位置 [num_envs, num_obstacles, 2]
        obstacle_vel: 障礙物速度 [num_envs, num_obstacles, 2] (世界座標系)
        min_distance: 最小距離閾值，避免除以零

    Returns:
        TTC 張量 [num_envs, num_obstacles]
        - 正值：預計 TTC 秒後會碰撞
        - inf：正在遠離，不會碰撞
        - nan：計算錯誤
    """
    num_envs = robot_pos.shape[0]
    num_obstacles = obstacle_pos.shape[1]
    device = robot_pos.device

    # 擴展機器人位置和速度以匹配障礙物維度
    # [num_envs, 2] -> [num_envs, num_obstacles, 2]
    robot_pos_exp = robot_pos.unsqueeze(1).expand(-1, num_obstacles, -1)
    robot_vel_exp = robot_vel.unsqueeze(1).expand(-1, num_obstacles, -1)

    # 計算相對位置和相對速度
    delta_p = obstacle_pos - robot_pos_exp  # [num_envs, num_obstacles, 2]
    delta_v = obstacle_vel - robot_vel_exp  # [num_envs, num_obstacles, 2]

    # 計算距離
    distance = torch.norm(delta_p, dim=-1)  # [num_envs, num_obstacles]

    # 計算接近速率 (Approach Speed)
    # v_app = - (Δp · Δv) / ||Δp||
    # 負號是因為：當障礙物「朝向」機器人移動時，delta_p 和 delta_v 的方向相反
    # 所以 dot product 會是負的，加負號讓它變成正的「接近速率」
    dot_product = (delta_p * delta_v).sum(dim=-1)  # [num_envs, num_obstacles]
    approach_speed = -dot_product / (distance.clamp(min=min_distance))  # [num_envs, num_obstacles]

    # 計算 TTC
    # 只有當 approach_speed > 0（正在靠近）時，TTC 才有意義
    # 否則設為 inf（永遠不會碰撞）
    ttc = torch.where(
        approach_speed > 0,
        distance / approach_speed.clamp(min=1e-6),  # 避免除以零
        torch.full_like(approach_speed, float('inf'))
    )

    return ttc


def ttc_defensive_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    safety_time_threshold: float = 1.5,
    penalty_weight: float = 2.0,
    max_penalty: float = 5.0,
    min_obstacle_distance: float = 0.3,
) -> torch.Tensor:
    """TTC 防禦性駕駛懲罰

    基於碰撞時間 (Time-to-Collision) 的防禦性駕駛獎勵函數。
    使用「特權物理蒸餾」概念，利用 Critic 擁有的障礙物速度資訊。

    核心邏輯：
    1. 計算機器人與所有動態障礙物的 TTC
    2. 找出最小 TTC（最危險的障礙物）
    3. 如果 TTC < safety_time_threshold，給予懲罰
    4. TTC 越小，懲罰越大

    獎勵公式：
    R = -penalty_weight * (τ - TTC)  if TTC < τ and TTC > 0
    R = 0                            otherwise

    例如：
    - τ = 1.5 秒, penalty_weight = 2.0
    - TTC = 1.0 秒 → R = -2.0 * (1.5 - 1.0) = -1.0
    - TTC = 0.5 秒 → R = -2.0 * (1.5 - 0.5) = -2.0
    - TTC = 0.1 秒 → R = -2.0 * (1.5 - 0.1) = -2.8

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        safety_time_threshold: 安全時間閾值（秒），低於此值開始懲罰
        penalty_weight: 懲罰權重
        max_penalty: 最大懲罰值
        min_obstacle_distance: 最小障礙物距離（米），用於過濾已經碰撞的

    Returns:
        懲罰張量 [num_envs]，範圍 [-max_penalty, 0]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    num_envs = env.num_envs
    device = env.device

    # 獲取機器人位置和速度（世界座標系）
    robot_pos = robot.data.root_pos_w[:, :2]  # [num_envs, 2]
    robot_vel_w = robot.data.root_lin_vel_w[:, :2]  # [num_envs, 2]

    # 初始化懲罰為 0
    penalty = torch.zeros(num_envs, device=device)

    # 檢查是否有動態障礙物資料
    if not hasattr(env, '_dynamic_obstacle_pos') or env._dynamic_obstacle_pos is None:
        return penalty

    obstacle_pos = env._dynamic_obstacle_pos  # [num_envs, num_obstacles, 2]
    obstacle_vel = env._dynamic_obstacle_vel  # [num_envs, num_obstacles, 2]

    # 如果沒有障礙物，返回 0
    if obstacle_pos.shape[1] == 0:
        return penalty

    # 計算 TTC
    ttc = compute_ttc(robot_pos, robot_vel_w, obstacle_pos, obstacle_vel)

    # 計算距離（用於過濾已經碰撞的）
    robot_pos_exp = robot_pos.unsqueeze(1).expand(-1, obstacle_pos.shape[1], -1)
    distances = torch.norm(obstacle_pos - robot_pos_exp, dim=-1)  # [num_envs, num_obstacles]

    # 過濾：只考慮還沒碰撞的障礙物（距離 > min_obstacle_distance）
    valid_mask = distances > min_obstacle_distance
    ttc_filtered = torch.where(valid_mask, ttc, torch.full_like(ttc, float('inf')))

    # 找出每個環境中最小的 TTC
    min_ttc, _ = ttc_filtered.min(dim=1)  # [num_envs]

    # 計算防禦性駕駛懲罰
    # R = -penalty_weight * (τ - TTC) if TTC < τ and TTC > 0
    dangerous_mask = (min_ttc < safety_time_threshold) & (min_ttc > 0) & (min_ttc < float('inf'))

    penalty = torch.where(
        dangerous_mask,
        -penalty_weight * (safety_time_threshold - min_ttc),
        torch.zeros_like(min_ttc)
    )

    # 裁剪到合理範圍
    penalty = torch.clamp(penalty, -max_penalty, 0.0)

    # 安全處理
    penalty = torch.nan_to_num(penalty, nan=0.0, posinf=0.0, neginf=-max_penalty)

    # 檢查 reward term
    penalty = _check_reward_term("ttc_defensive_penalty", penalty, env, raise_on_error=True)

    return penalty


def ttc_early_warning_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    warning_time_threshold: float = 2.5,
    reward_weight: float = 0.5,
) -> torch.Tensor:
    """TTC 早期預警獎勵

    當機器人在 TTC 接近危險區域時，如果主動減速或轉向，
    給予獎勵。這鼓勵「防禦性駕駛」行為。

    核心邏輯：
    1. 計算 TTC
    2. 如果 TTC 在 warning_time_threshold 附近（1.5s ~ 2.5s）
    3. 且機器人正在減速（速度降低）
    4. 給予獎勵

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        warning_time_threshold: 預警時間閾值（秒）
        reward_weight: 獎勵權重

    Returns:
        獎勵張量 [num_envs]，範圍 [0, reward_weight]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    num_envs = env.num_envs
    device = env.device

    # 獲取機器人位置和速度
    robot_pos = robot.data.root_pos_w[:, :2]
    robot_vel_w = robot.data.root_lin_vel_w[:, :2]
    robot_speed = torch.norm(robot_vel_w, dim=1)  # [num_envs]

    # 初始化獎勵為 0
    reward = torch.zeros(num_envs, device=device)

    # 檢查是否有動態障礙物資料
    if not hasattr(env, '_dynamic_obstacle_pos') or env._dynamic_obstacle_pos is None:
        return reward

    obstacle_pos = env._dynamic_obstacle_pos
    obstacle_vel = env._dynamic_obstacle_vel

    if obstacle_pos.shape[1] == 0:
        return reward

    # 計算 TTC
    ttc = compute_ttc(robot_pos, robot_vel_w, obstacle_pos, obstacle_vel)

    # 找出每個環境中最小的 TTC
    min_ttc, _ = ttc.min(dim=1)  # [num_envs]

    # 早期預警區域：TTC 在 1.5s ~ warning_time_threshold 之間
    warning_zone = (min_ttc > 1.5) & (min_ttc < warning_time_threshold) & (min_ttc < float('inf'))

    # 如果在預警區域，且速度較低（正在減速），給予獎勵
    # 速度越低，獎勵越高
    max_speed = 1.0  # 預期最大速度
    speed_ratio = 1.0 - (robot_speed / max_speed).clamp(0.0, 1.0)  # [0, 1]，速度低時接近 1

    reward = torch.where(
        warning_zone,
        reward_weight * speed_ratio,
        torch.zeros_like(min_ttc)
    )

    # 安全處理
    reward = torch.nan_to_num(reward, nan=0.0, posinf=reward_weight, neginf=0.0)

    return reward


def ttc_progressive_safety_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    time_thresholds: tuple = (0.5, 1.0, 1.5, 2.0),
    penalty_weights: tuple = (5.0, 3.0, 2.0, 1.0),
) -> torch.Tensor:
    """漸進式 TTC 安全獎勵

    根據 TTC 的不同級別，給予不同強度的懲罰。
    TTC 越小，懲罰越大，形成「漸進式」的安全邊界。

    級別設計：
    - TTC < 0.5s：極度危險，最大懲罰
    - TTC < 1.0s：危險，重懲罰
    - TTC < 1.5s：警告，中等懲罰
    - TTC < 2.0s：注意，輕微懲罰
    - TTC >= 2.0s：安全，無懲罰

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        time_thresholds: 時間閾值元組（從小到大）
        penalty_weights: 對應的懲罰權重元組（從大到小）

    Returns:
        懲罰張量 [num_envs]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    num_envs = env.num_envs
    device = env.device

    # 獲取機器人位置和速度
    robot_pos = robot.data.root_pos_w[:, :2]
    robot_vel_w = robot.data.root_lin_vel_w[:, :2]

    # 初始化懲罰為 0
    penalty = torch.zeros(num_envs, device=device)

    # 檢查是否有動態障礙物資料
    if not hasattr(env, '_dynamic_obstacle_pos') or env._dynamic_obstacle_pos is None:
        return penalty

    obstacle_pos = env._dynamic_obstacle_pos
    obstacle_vel = env._dynamic_obstacle_vel

    if obstacle_pos.shape[1] == 0:
        return penalty

    # 計算 TTC
    ttc = compute_ttc(robot_pos, robot_vel_w, obstacle_pos, obstacle_vel)

    # 找出每個環境中最小的 TTC
    min_ttc, _ = ttc.min(dim=1)  # [num_envs]

    # 過濾無效值
    valid_mask = (min_ttc > 0) & (min_ttc < float('inf'))

    # 漸進式懲罰
    for i, (threshold, weight) in enumerate(zip(time_thresholds, penalty_weights)):
        if i == 0:
            # 最危險級別：TTC < threshold
            level_mask = valid_mask & (min_ttc < threshold)
            penalty = torch.where(level_mask, -weight, penalty)
        else:
            # 其他級別：上一個閾值 <= TTC < 當前閾值
            prev_threshold = time_thresholds[i - 1]
            level_mask = valid_mask & (min_ttc >= prev_threshold) & (min_ttc < threshold)
            penalty = torch.where(level_mask, -weight, penalty)

    # 安全處理
    penalty = torch.nan_to_num(penalty, nan=0.0, posinf=0.0, neginf=0.0)

    return penalty


def privileged_physics_total_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # TTC 參數
    ttc_safety_threshold: float = 1.5,
    ttc_penalty_weight: float = 2.0,
    # 漸進式參數
    use_progressive: bool = True,
    progressive_thresholds: tuple = (0.5, 1.0, 1.5, 2.0),
    progressive_weights: tuple = (5.0, 3.0, 2.0, 1.0),
    # 早期預警參數
    early_warning_threshold: float = 2.5,
    early_warning_weight: float = 0.5,
    # 總體參數
    max_penalty: float = 5.0,
) -> torch.Tensor:
    """特權物理蒸餾總獎勵

    結合所有 TTC 相關的獎勵函數，形成完整的防禦性駕駛獎勵系統。

    組成部分：
    1. TTC 防禦性駕駛懲罰（主要）
    2. 漸進式 TTC 安全獎勵（可選）
    3. TTC 早期預警獎勵（可選）

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        ttc_safety_threshold: TTC 安全閾值
        ttc_penalty_weight: TTC 懲罰權重
        use_progressive: 是否使用漸進式懲罰
        progressive_thresholds: 漸進式閾值
        progressive_weights: 漸進式權重
        early_warning_threshold: 早期預警閾值
        early_warning_weight: 早期預警權重
        max_penalty: 最大懲罰

    Returns:
        總獎勵張量 [num_envs]
    """
    # TTC 防禦性駕駛懲罰
    defensive_penalty = ttc_defensive_penalty(
        env,
        robot_cfg=robot_cfg,
        safety_time_threshold=ttc_safety_threshold,
        penalty_weight=ttc_penalty_weight,
        max_penalty=max_penalty,
    )

    # 漸進式安全懲罰（可選）
    if use_progressive:
        progressive_penalty = ttc_progressive_safety_reward(
            env,
            robot_cfg=robot_cfg,
            time_thresholds=progressive_thresholds,
            penalty_weights=progressive_weights,
        )
    else:
        progressive_penalty = torch.zeros_like(defensive_penalty)

    # 早期預警獎勵
    early_reward = ttc_early_warning_reward(
        env,
        robot_cfg=robot_cfg,
        warning_time_threshold=early_warning_threshold,
        reward_weight=early_warning_weight,
    )

    # 總獎勵
    total = defensive_penalty + progressive_penalty + early_reward

    # 裁剪
    total = torch.clamp(total, -max_penalty, early_warning_weight)

    return total


def ttc_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    safety_time_threshold: float = 1.5,
    penalty_weight: float = 1.0,
    max_penalty: float = 10.0,
    min_distance: float = 0.1,
    num_obstacles: int = 10,
) -> torch.Tensor:
    """TTC 防禦性駕駛懲罰（新版：加總所有危險障礙物）

    基於「特權物理蒸餾 (Privileged Physics Distillation)」概念實現。

    ═══════════════════════════════════════════════════════════════════════════
                    特權物理蒸餾之防禦性駕駛獎勵
    ═══════════════════════════════════════════════════════════════════════════

    核心思想：
    - Critic 擁有「上帝視角」（障礙物的真實座標與速度）
    - 計算「碰撞時間 (Time-To-Collision, TTC)」
    - 如果 TTC < 安全閾值 τ，給予懲罰
    - Actor 被迫從 LiDAR 歷史幀學會辨識危險特徵
    - 結果：提早 3 公尺就開始減速或繞路（防禦性駕駛！）

    ────────────────────────────────────────────────────────────────────────────────
    數學公式：
    ────────────────────────────────────────────────────────────────────────────────

    取得狀態張量（只取 X, Y 軸維度）：
        - 機器人座標 p_r 與速度 v_r：shape [num_envs, 2]
        - 障礙物座標 p_o 與速度 v_o：shape [num_envs, num_obstacles, 2]
        - 障礙物高度 Z：shape [num_envs, num_obstacles]

    計算相對物理量：
        - 相對位置：Δp = p_o - p_r
        - 相對速度：Δv = v_o - v_r
        - 距離：D = ||Δp||

    計算接近速率 (Approach Speed) 與 TTC：
        - v_app = - (Δp · Δv) / D  （加上 1e-5 避免除以零）
        - TTC = D / v_app

    過濾與懲罰邏輯 (PyTorch Masking)：
        - 定義安全時間閾值 τ（例如 1.5 秒）
        - 建立危險遮罩 danger_mask，必須同時滿足：
            1. Z > 0（不在地底下）
            2. v_app > 0（正在互相靠近）
            3. TTC < τ（即將碰撞）
        - 懲罰值計算：對於觸發危險遮罩的障礙物，給予 -(τ - TTC) 的懲罰
        - 將每個環境中所有障礙物的懲罰加總

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        safety_time_threshold: 安全時間閾值 τ（秒），低於此值開始懲罰
        penalty_weight: 懲罰權重（乘數）
        max_penalty: 最大懲罰值
        min_distance: 最小距離閾值，避免除以零
        num_obstacles: 場景中障礙物數量

    Returns:
        懲罰張量 [num_envs]，範圍 [-max_penalty, 0]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    num_envs = env.num_envs
    device = env.device

    # 獲取機器人位置和速度（世界座標系，只取 X, Y）
    robot_pos = robot.data.root_pos_w[:, :2]  # [num_envs, 2]
    robot_vel = robot.data.root_lin_vel_w[:, :2]  # [num_envs, 2]

    # 初始化懲罰為 0
    penalty = torch.zeros(num_envs, device=device)

    # 準備障礙物資料
    obstacle_positions = []  # [num_envs, num_obstacles, 2]
    obstacle_velocities = []  # [num_envs, num_obstacles, 2]
    obstacle_z_coords = []  # [num_envs, num_obstacles]

    for i in range(num_obstacles):
        obstacle_name = f"obstacle_{i}"
        # 🔥 修正：InteractiveScene 只支援 scene["name"]，不支援 hasattr/getattr
        if obstacle_name not in env.scene.keys():
            continue

        obstacle = env.scene[obstacle_name]
        obs_pos_w = obstacle.data.root_pos_w  # [num_envs, 3]

        # 位置（X, Y）
        obstacle_positions.append(obs_pos_w[:, :2])

        # Z 座標（用於地底遮蔽檢查）
        obstacle_z_coords.append(obs_pos_w[:, 2])

        # 速度（從環境緩存獲取，或使用 0）
        if hasattr(env, '_obstacle_velocities') and env._obstacle_velocities is not None:
            if env._obstacle_velocities.shape[1] > i:
                obstacle_velocities.append(env._obstacle_velocities[:, i, :])
            else:
                obstacle_velocities.append(torch.zeros(num_envs, 2, device=device))
        else:
            # 嘗試從障礙物獲取速度
            if hasattr(obstacle.data, 'root_lin_vel_w'):
                obstacle_velocities.append(obstacle.data.root_lin_vel_w[:, :2])
            else:
                obstacle_velocities.append(torch.zeros(num_envs, 2, device=device))

    # 如果沒有障礙物，返回 0
    if len(obstacle_positions) == 0:
        return penalty

    # 堆疊成張量
    obstacle_pos = torch.stack(obstacle_positions, dim=1)  # [num_envs, num_obstacles, 2]
    obstacle_vel = torch.stack(obstacle_velocities, dim=1)  # [num_envs, num_obstacles, 2]
    obstacle_z = torch.stack(obstacle_z_coords, dim=1)  # [num_envs, num_obstacles]

    actual_num_obstacles = obstacle_pos.shape[1]

    # ═══════════════════════════════════════════════════════════════════════════
    # 計算相對物理量（全程 Batch 操作，不使用 for 迴圈）
    # ═══════════════════════════════════════════════════════════════════════════

    # 使用 unsqueeze(1) 對齊維度
    # robot_pos: [num_envs, 2] -> [num_envs, 1, 2]
    robot_pos_exp = robot_pos.unsqueeze(1)  # [num_envs, 1, 2]
    robot_vel_exp = robot_vel.unsqueeze(1)  # [num_envs, 1, 2]

    # 相對位置：Δp = p_o - p_r
    delta_p = obstacle_pos - robot_pos_exp  # [num_envs, num_obstacles, 2]

    # 相對速度：Δv = v_o - v_r
    delta_v = obstacle_vel - robot_vel_exp  # [num_envs, num_obstacles, 2]

    # 距離：D = ||Δp||
    distance = torch.norm(delta_p, dim=-1)  # [num_envs, num_obstacles]

    # ═══════════════════════════════════════════════════════════════════════════
    # 計算接近速率 (Approach Speed) 與 TTC
    # ═══════════════════════════════════════════════════════════════════════════

    # v_app = - (Δp · Δv) / D
    # 負號是因為：當雙方靠近時，Δp 和 Δv 方向相反，dot product 為負
    dot_product = (delta_p * delta_v).sum(dim=-1)  # [num_envs, num_obstacles]
    approach_speed = -dot_product / (distance.clamp(min=min_distance) + 1e-5)  # [num_envs, num_obstacles]

    # TTC = D / v_app（只在 v_app > 0 時有意義）
    ttc = torch.where(
        approach_speed > 1e-6,
        distance / (approach_speed + 1e-6),
        torch.full_like(approach_speed, float('inf'))
    )  # [num_envs, num_obstacles]

    # ═══════════════════════════════════════════════════════════════════════════
    # 過濾與懲罰邏輯 (PyTorch Masking)
    # ═══════════════════════════════════════════════════════════════════════════

    # 建立危險遮罩 danger_mask：
    # 1. Z > 0（不在地底下）
    # 2. v_app > 0（正在互相靠近）
    # 3. TTC < τ（即將碰撞）
    z_valid = obstacle_z > 0.0  # [num_envs, num_obstacles]
    approaching = approach_speed > 0.0  # [num_envs, num_obstacles]
    ttc_dangerous = ttc < safety_time_threshold  # [num_envs, num_obstacles]

    danger_mask = z_valid & approaching & ttc_dangerous  # [num_envs, num_obstacles]

    # 懲罰值計算：-(τ - TTC) 對於觸發危險遮罩的障礙物
    # 只對危險障礙物計算懲罰，其他設為 0
    individual_penalty = torch.where(
        danger_mask,
        -(safety_time_threshold - ttc),
        torch.zeros_like(ttc)
    )  # [num_envs, num_obstacles]

    # 將每個環境中所有障礙物的懲罰加總
    total_penalty = individual_penalty.sum(dim=1)  # [num_envs]

    # 應用懲罰權重
    total_penalty = total_penalty * penalty_weight

    # 裁剪到合理範圍
    total_penalty = torch.clamp(total_penalty, -max_penalty, 0.0)

    # 安全處理
    total_penalty = torch.nan_to_num(total_penalty, nan=0.0, posinf=0.0, neginf=-max_penalty)

    # 檢查 reward term
    total_penalty = _check_reward_term("ttc_penalty", total_penalty, env, raise_on_error=True)

    return total_penalty


__all__ = [
    "compute_ttc",
    "ttc_defensive_penalty",
    "ttc_early_warning_reward",
    "ttc_progressive_safety_reward",
    "privileged_physics_total_reward",
    "ttc_penalty",  # 新增：用戶要求的 TTC 懲罰函數
]
