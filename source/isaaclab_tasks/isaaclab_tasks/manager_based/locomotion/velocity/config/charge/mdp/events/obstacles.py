"""障礙物重置和移動事件

包含障礙物相關的事件函數：
- reset_obstacles: 重置障礙物位置
- move_obstacles: 動態移動障礙物
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 從 mdp.core 導入狀態管理功能
from ..core import (
    get_obstacle_num as _get_obstacle_num,
    get_obstacle_sizes as _get_obstacle_sizes,
)


def reset_obstacles(
    env,
    env_ids,
    speed_range: float = 0.5,
    min_speed: float = 0.05,
    min_robot_distance: float = 1.5,
    min_goal_distance: float = 1.0,
    min_obstacle_spacing: float = 1.0,
    max_spawn_attempts: int = 50,
    boundary: float = 5.0,  # 場景邊界（米），默認 5.0 米
):
    """重置障礙物位置（Phase 1：完全靜態，含碰撞檢查）

    在環境重置時，將所有障礙物隨機重新擺放到新位置。
    Phase 1 設計：障礙物完全靜態（速度設為 0），只在每個 episode 開始時移動。
    
    2024-01 修正：加入碰撞檢查，確保障礙物不會與機器人、目標或其他障礙物重疊。

    Args:
        env: 環境實例
        env_ids: 需要重置的環境 ID 列表
                 例如：[0, 5, 12] = 第 0、5、12 個環境需要重置
        speed_range: 速度範圍（Phase 1 不使用，保留用於 Phase 2）
        min_speed: 最小速度（Phase 1 不使用，保留用於 Phase 2）
        min_robot_distance: 障礙物與機器人的最小距離（米）
        min_goal_distance: 障礙物與目標的最小距離（米）
        min_obstacle_spacing: 障礙物之間的最小距離（米）
        max_spawn_attempts: 每個障礙物最大嘗試生成次數
        boundary: 場景邊界（米），默認 5.0 米（Phase 1/2），Phase 3 為 8.0 米

    Phase 1 設計說明：
    - 障礙物設為 kinematic（固定不動）
    - 速度設為 0（與物理引擎一致）
    - 只在環境重置時隨機擺放位置
    - 避免「瞬移」問題（不會在 episode 中間突然移動）
    - 碰撞檢查確保合理的初始配置

    Phase 2 升級（動態障礙物）：
    - 將 rigid_props.kinematic_enabled 設為 False
    - 使用 move_obstacles 事件進行連續移動
    - 此時速度才會被實際使用
    """
    # reset 模式可能傳入 env_ids=None，代表所有環境
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    num_resets = len(env_ids)  # 需要重置的環境數量
    device = env.device  # 設備（CPU 或 GPU）

    # 遍歷所有可能的障礙物
    # Phase 2.5：優先使用環境中已設置的 _num_obstacles（由課程學習設置）
    # 只有在未設置時才使用全局配置
    num_obstacles = getattr(env, "_num_obstacles", None)
    if num_obstacles is None:
        num_obstacles = _get_obstacle_num()
        env._num_obstacles = num_obstacles
    # 注意：如果 env._num_obstacles 已經設置（例如由 reset_obstacles_with_adaptive_params 設置），
    # 則使用該值，不覆蓋它
    if not hasattr(env, "_obstacle_sizes"):
        env._obstacle_sizes = _get_obstacle_sizes()

    # 初始化障礙物速度緩存（用於觀測）
    if not hasattr(env, "_obstacle_velocities") or env._obstacle_velocities.shape[1] != num_obstacles:
        env._obstacle_velocities = torch.zeros(env.num_envs, num_obstacles, 2, device=device)

    # Phase 1：速度設為 0（障礙物完全靜態）
    # 這樣觀測中的速度資訊會正確反映物理引擎的狀態（靜止）
    env._obstacle_velocities[env_ids, :, :] = 0.0

    # ------------------------------------------------------------------------
    # 獲取機器人和目標位置（用於碰撞檢查）
    # ------------------------------------------------------------------------
    robot_pos_xy = env.scene["robot"].data.root_pos_w[env_ids, :2]  # [num_resets, 2]
    
    # 嘗試獲取目標位置（可能還未初始化）
    try:
        goal_pos = env.command_manager.get_command("goal_command")
        goal_pos_xy = goal_pos[env_ids, :2]  # [num_resets, 2]
        has_goal = True
    except (AttributeError, KeyError, IndexError):
        goal_pos_xy = None
        has_goal = False

    # 儲存已放置的障礙物位置（用於障礙物間碰撞檢查）
    # shape: [num_resets, num_obstacles, 2]
    placed_positions = torch.full(
        (num_resets, num_obstacles, 2), float('inf'), device=device, dtype=torch.float32
    )

    # ------------------------------------------------------------------------
    # 分層採樣設計：確保障礙物均勻分布在場景各區域
    # ------------------------------------------------------------------------
    # 將場景分成 4 個象限，每個象限分配固定數量的障礙物
    # 這樣可以避免障礙物集中在某一側
    #
    # 象限劃分（以原點為中心）：
    #   象限 0: X > 0, Y > 0（右上）
    #   象限 1: X < 0, Y > 0（左上）
    #   象限 2: X < 0, Y < 0（左下）
    #   象限 3: X > 0, Y < 0（右下）
    #
    # 邊界值：使用傳入的 boundary 參數（但留安全邊距，實際生成範圍 boundary - safe_margin）
    # ------------------------------------------------------------------------
    
    # boundary 已作為參數傳入（默認 5.0 米，Phase 3 為 8.0 米）
    # boundary 已作為參數傳入（默認 5.0 米，Phase 3 為 8.0 米）
    # 安全邊距 = 用戶請求 (1.0m) + 障礙物半徑 (0.5m) = 1.5m
    safe_margin = 1.5
    spawn_range = boundary - safe_margin  # 實際生成範圍
    
    # 定義 4 個象限的範圍
    # 格式：(x_min, x_max, y_min, y_max)
    quadrants = [
        (0.3, spawn_range, 0.3, spawn_range),      # 象限 0: 右上（避開原點附近）
        (-spawn_range, -0.3, 0.3, spawn_range),    # 象限 1: 左上
        (-spawn_range, -0.3, -spawn_range, -0.3),  # 象限 2: 左下
        (0.3, spawn_range, -spawn_range, -0.3),    # 象限 3: 右下
    ]
    
    # 計算每個象限分配多少障礙物
    # 例如：10 個障礙物 → 每象限 2-3 個
    obstacles_per_quadrant = num_obstacles // 4  # 基礎數量
    remainder = num_obstacles % 4  # 餘數分配給前幾個象限
    
    # 建立障礙物到象限的映射
    obstacle_to_quadrant = []
    for q in range(4):
        count = obstacles_per_quadrant + (1 if q < remainder else 0)
        obstacle_to_quadrant.extend([q] * count)
    
    for i in range(num_obstacles):
        obstacle_name = f"obstacle_{i}"

        # 檢查障礙物是否存在
        if not hasattr(env.scene, obstacle_name):
            continue

        # ------------------------------------------------------------------------
        # 取得此障礙物應該生成的象限
        # ------------------------------------------------------------------------
        quadrant_idx = obstacle_to_quadrant[i] if i < len(obstacle_to_quadrant) else i % 4
        x_min, x_max, y_min, y_max = quadrants[quadrant_idx]

        # ------------------------------------------------------------------------
        # 生成隨機位置（含碰撞檢查）
        # ------------------------------------------------------------------------
        pos = torch.zeros(num_resets, 3, device=device, dtype=torch.float32)
        pos[:, 2] = 0.5  # Z 座標：固定高度

        # 追蹤哪些環境還需要找到有效位置
        needs_position = torch.ones(num_resets, dtype=torch.bool, device=device)
        
        for attempt in range(max_spawn_attempts):
            if not needs_position.any():
                break  # 所有環境都找到有效位置了
            
            # 為需要位置的環境生成隨機候選位置（在指定象限內）
            num_need = needs_position.sum().item()
            
            # 在指定象限範圍內生成隨機位置
            candidate_x = torch.rand(num_need, device=device) * (x_max - x_min) + x_min
            candidate_y = torch.rand(num_need, device=device) * (y_max - y_min) + y_min
            
            # 暫存候選位置
            pos[needs_position, 0] = candidate_x
            pos[needs_position, 1] = candidate_y
            
            # ------------------------------------------------------------------------
            # 碰撞檢查 1：與機器人距離
            # ------------------------------------------------------------------------
            dist_to_robot = torch.norm(pos[:, :2] - robot_pos_xy, dim=1)
            valid_robot = dist_to_robot >= min_robot_distance
            
            # ------------------------------------------------------------------------
            # 碰撞檢查 2：與目標距離
            # ------------------------------------------------------------------------
            if has_goal:
                dist_to_goal = torch.norm(pos[:, :2] - goal_pos_xy, dim=1)
                valid_goal = dist_to_goal >= min_goal_distance
            else:
                valid_goal = torch.ones(num_resets, dtype=torch.bool, device=device)
            
            # ------------------------------------------------------------------------
            # 碰撞檢查 3：與其他已放置障礙物的距離
            # ------------------------------------------------------------------------
            valid_obstacles = torch.ones(num_resets, dtype=torch.bool, device=device)
            for j in range(i):  # 只檢查已放置的障礙物
                dist_to_other = torch.norm(pos[:, :2] - placed_positions[:, j, :], dim=1)
                valid_obstacles &= dist_to_other >= min_obstacle_spacing
            
            # ------------------------------------------------------------------------
            # 綜合判斷：所有檢查都通過才算有效
            # ------------------------------------------------------------------------
            valid_this_attempt = valid_robot & valid_goal & valid_obstacles & needs_position
            
            # 更新 needs_position：有效的不再需要
            needs_position = needs_position & ~valid_this_attempt
        
        # 如果達到最大嘗試次數仍有環境找不到有效位置
        # 允許在整個場景範圍內重新嘗試（降級策略）
        if needs_position.any():
            num_failed = needs_position.sum().item()
            # 使用全場景範圍重新生成
            fallback_x = torch.rand(num_failed, device=device) * (2 * spawn_range) - spawn_range
            fallback_y = torch.rand(num_failed, device=device) * (2 * spawn_range) - spawn_range
            pos[needs_position, 0] = fallback_x
            pos[needs_position, 1] = fallback_y
        
        # 記錄已放置的位置
        placed_positions[:, i, :] = pos[:, :2]

        # ------------------------------------------------------------------------
        # 生成姿態（不旋轉）
        # ------------------------------------------------------------------------
        quat = torch.zeros(num_resets, 4, device=device, dtype=torch.float32)
        quat[:, 0] = 1.0  # (w=1, x=0, y=0, z=0) = 不旋轉

        # ------------------------------------------------------------------------
        # 應用到模擬器
        # ------------------------------------------------------------------------
        getattr(env.scene, obstacle_name).write_root_pose_to_sim(
            torch.cat([pos, quat], dim=-1),  # 拼接位置和姿態 [7]
            env_ids=env_ids  # 只更新指定的環境
        )
        # write_root_pose_to_sim：直接設定物體的位置和姿態
        # env_ids：指定哪些環境需要更新（避免影響其他環境）


def move_obstacles(
    env,
    env_ids,
    move_dt: float = 0.2,
    speed_range: float = 0.5,
    min_speed: float = 0.05,
    speed_jitter: float = 0.05,
    area_limit: float = 6.0,
):
    """動態移動障礙物（interval 事件）
    
    透過「連續速度 + 小擾動」讓障礙物平滑移動。
    
    Args:
        env: 環境實例
        env_ids: 需要更新的環境 ID（可能是 None、張量或列表）
        move_dt: 每次更新的時間步長（秒）
        speed_range: 最大速度（m/s）
        min_speed: 最小速度（m/s）
        speed_jitter: 速度小擾動（m/s）
        area_limit: 障礙物活動範圍限制（正負範圍）
    """
    # 處理 env_ids 參數（interval 模式可能傳入 None 或張量）
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, (list, tuple)):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor([env_ids], device=env.device, dtype=torch.long)
    
    num_resets = len(env_ids)
    device = env.device
    
    num_obstacles = getattr(env, "_num_obstacles", None)
    if num_obstacles is None:
        num_obstacles = _get_obstacle_num()
    if not hasattr(env, "_obstacle_velocities") or env._obstacle_velocities.shape[1] != num_obstacles:
        env._obstacle_velocities = torch.zeros(env.num_envs, num_obstacles, 2, device=device)
    
    for i in range(num_obstacles):
        obstacle_name = f"obstacle_{i}"
        if not hasattr(env.scene, obstacle_name):
            continue
        
        obstacle = getattr(env.scene, obstacle_name)
        
        # 取出當前位置與速度
        pos = obstacle.data.root_pos_w[env_ids, :3].clone()
        vel = env._obstacle_velocities[env_ids, i, :].clone()
        
        # 如果速度為 0（初始狀態），初始化隨機速度
        speed = torch.linalg.norm(vel, dim=1, keepdim=True)
        zero_speed_mask = (speed.squeeze(1) < 1e-6)
        if zero_speed_mask.any():
            # 為速度為 0 的障礙物初始化隨機速度
            num_zero = zero_speed_mask.sum().item()
            angle = torch.rand(num_zero, device=device) * 2.0 * math.pi - math.pi  # [-π, π]
            init_speed = torch.rand(num_zero, device=device) * (speed_range - min_speed) + min_speed
            vel[zero_speed_mask, 0] = init_speed * torch.cos(angle)
            vel[zero_speed_mask, 1] = init_speed * torch.sin(angle)
        
        # 速度小擾動（讓方向慢慢變化）
        if speed_jitter > 0.0:
            vel += (torch.rand(num_resets, 2, device=device) - 0.5) * 2.0 * speed_jitter
        
        # 速度限制
        speed = torch.linalg.norm(vel, dim=1, keepdim=True)
        speed = torch.clamp(speed, min_speed, speed_range)
        vel = vel / (torch.linalg.norm(vel, dim=1, keepdim=True) + 1e-6) * speed
        
        # 平滑位移
        pos[:, 0:2] += vel * move_dt
        
        # 邊界反彈
        hit_x = (pos[:, 0] <= -area_limit) | (pos[:, 0] >= area_limit)
        hit_y = (pos[:, 1] <= -area_limit) | (pos[:, 1] >= area_limit)
        vel[hit_x, 0] *= -1.0
        vel[hit_y, 1] *= -1.0
        
        pos[:, 0] = torch.clamp(pos[:, 0], -area_limit, area_limit)
        pos[:, 1] = torch.clamp(pos[:, 1], -area_limit, area_limit)
        
        # 取出當前姿態（保持不變）
        if hasattr(obstacle.data, "root_quat_w"):
            quat = obstacle.data.root_quat_w[env_ids, :4].clone()
        else:
            quat = torch.zeros(num_resets, 4, device=device, dtype=torch.float32)
            quat[:, 0] = 1.0
        
        # 應用到模擬器
        obstacle.write_root_pose_to_sim(
            torch.cat([pos, quat], dim=-1),
            env_ids=env_ids,
        )
        
        # 回寫速度
        env._obstacle_velocities[env_ids, i, :] = vel
