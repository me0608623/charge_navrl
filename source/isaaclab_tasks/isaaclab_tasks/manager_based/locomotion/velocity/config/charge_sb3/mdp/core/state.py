# ============================================================================
# 狀態管理模組
# ============================================================================
"""
狀態管理

負責管理障礙物元數據等狀態信息，包括：
- 障礙物元數據（數量、尺寸）
- 動態障礙物移動狀態（起始位置、移動方向）
"""

from __future__ import annotations

import math
import torch
from typing import Optional

# ============================================================================
# 障礙物元數據管理
# ============================================================================
# 全局變量：障礙物數量與尺寸（避免寫入 scene cfg）
_OBSTACLE_NUM: int | None = None
_OBSTACLE_SIZES: list[float] | None = None


def set_obstacle_metadata(num_obstacles: int, obstacle_sizes: list[float]) -> None:
    """保存障礙物數量與尺寸（供觀測與事件使用）
    
    Args:
        num_obstacles: 初始/當前障礙物數量
        obstacle_sizes: 所有障礙物的尺寸列表（包含未啟用的）
    """
    global _OBSTACLE_NUM, _OBSTACLE_SIZES
    _OBSTACLE_NUM = num_obstacles
    _OBSTACLE_SIZES = obstacle_sizes


def _get_obstacle_num(default: int = 8) -> int:
    """獲取障礙物數量（內部使用）"""
    if _OBSTACLE_NUM is not None:
        return _OBSTACLE_NUM
    return default


def _get_obstacle_sizes() -> list[float] | None:
    """獲取障礙物尺寸列表（內部使用）"""
    return _OBSTACLE_SIZES


def get_obstacle_metadata():
    """獲取障礙物元數據（用於動態障礙物）"""
    return (_OBSTACLE_NUM or 3, _OBSTACLE_SIZES or [])


def get_obstacle_num(default: int = 8) -> int:
    """獲取障礙物數量（公開接口）"""
    return _get_obstacle_num(default)


def get_obstacle_sizes() -> list[float] | None:
    """獲取障礙物尺寸列表（公開接口）"""
    return _get_obstacle_sizes()


# ============================================================================
# 動態障礙物移動邏輯（Phase 2）
# ============================================================================
# 全局變量：記錄每個障礙物的移動狀態
_obstacle_start_positions: torch.Tensor | None = None  # 記錄移動起始位置
_obstacle_directions: torch.Tensor | None = None  # 記錄移動方向（角度，弧度）


def reset_obstacle_targets(env, env_ids: torch.Tensor, num_obstacles: int, boundary: float = 5.0):
    """重置障礙物的移動狀態（來回移動模式）"""
    global _obstacle_start_positions, _obstacle_directions
    
    if _obstacle_start_positions is None:
        _obstacle_start_positions = torch.zeros(env.num_envs, num_obstacles, 2, device=env.device)
    
    if _obstacle_directions is None:
        _obstacle_directions = torch.zeros(env.num_envs, num_obstacles, device=env.device)
    
    # 為每個障礙物隨機生成移動方向（0 到 2π）
    if _obstacle_start_positions is not None and _obstacle_directions is not None:
        for i in range(num_obstacles):
            # 隨機方向（0 到 2π）
            _obstacle_directions[env_ids, i] = torch.empty(len(env_ids), device=env.device).uniform_(0, 2 * math.pi)
            # 記錄起始位置（當前位置會在 reset_obstacles 中設置）


def reset_obstacles_with_targets(
    env,
    env_ids,
    num_obstacles: int = 10,
    boundary: float = 5.0,
    min_robot_distance: float = 1.5,
    min_goal_distance: float = 1.0,
    min_obstacle_spacing: float = 1.0,
    max_spawn_attempts: int = 50,
    reset_obstacles_func=None,  # 依賴注入：reset_obstacles 函數
):
    """
    重置障礙物位置並初始化移動狀態（Phase 2 動態障礙物）
    
    注意：此函數依賴 reset_obstacles 函數，需要通過 reset_obstacles_func 參數傳入。
    這是為了避免循環依賴，reset_obstacles 將在 events 模組中實現。
    """
    global _obstacle_start_positions
    
    # 如果提供了 reset_obstacles 函數，先調用它重置障礙物位置
    if reset_obstacles_func is not None:
        reset_obstacles_func(
            env,
            env_ids,
            speed_range=0.5,
            min_speed=0.05,
            min_robot_distance=min_robot_distance,
            min_goal_distance=min_goal_distance,
            min_obstacle_spacing=min_obstacle_spacing,
            max_spawn_attempts=max_spawn_attempts,
        )
    
    # 初始化移動方向
    reset_obstacle_targets(env, env_ids, num_obstacles, boundary)
    
    # 記錄重置後的起始位置
    if _obstacle_start_positions is not None:
        for i in range(num_obstacles):
            try:
                obstacle = env.scene[f"obstacle_{i}"]
                _obstacle_start_positions[env_ids, i] = obstacle.data.root_pos_w[env_ids, :2]
            except KeyError:
                continue


def move_obstacles_toward_target(
    env,
    env_ids: torch.Tensor | None,
    num_obstacles: int = 10,
    max_velocity: float = 0.5,
    boundary: float = 5.0,
    travel_distance: float = 3.0,
):
    """
    讓障礙物水平來回移動
    
    邏輯：
    1. 每個障礙物有一個移動方向（角度）
    2. 障礙物沿該方向移動，記錄從起始位置的距離
    3. 移動 3 米後，反轉方向（角度 + π）
    4. 繼續移動 3 米後，再次反轉
    5. 形成來回移動的模式
    
    Args:
        env: 環境實例
        env_ids: 需要更新的環境 ID（None 表示所有環境）
        num_obstacles: 障礙物數量
        max_velocity: 移動速度（m/s）
        boundary: 活動邊界（米），超出邊界時反轉方向
        travel_distance: 單向移動距離（米），移動此距離後反轉方向
    """
    global _obstacle_start_positions, _obstacle_directions
    
    # 處理 env_ids 參數
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, (list, tuple)):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor([env_ids], device=env.device, dtype=torch.long)
    
    # 類型斷言：確保 env_ids 是 Tensor（消除 linter 警告）
    assert isinstance(env_ids, torch.Tensor)
    
    actual_count, _ = get_obstacle_metadata()
    num_obstacles = min(num_obstacles, actual_count)
    
    # 初始化移動狀態（如果還沒有）
    if _obstacle_start_positions is None or _obstacle_directions is None:
        reset_obstacle_targets(env, torch.arange(env.num_envs, device=env.device), actual_count, boundary)
        # 記錄起始位置
        if _obstacle_start_positions is not None:
            for i in range(actual_count):
                try:
                    obstacle = env.scene[f"obstacle_{i}"]
                    _obstacle_start_positions[:, i] = obstacle.data.root_pos_w[:, :2]
                except KeyError:
                    continue
    
    if _obstacle_start_positions is None or _obstacle_directions is None:
        return  # 如果初始化失敗，直接返回
    
    # 類型斷言：確保變量不為 None（消除 linter 警告）
    # 在上面的檢查後，這些變量必定不為 None
    assert _obstacle_start_positions is not None
    assert _obstacle_directions is not None
    
    # 創建局部引用並明確類型（解決 pyright 類型推斷問題）
    obstacle_start_positions: torch.Tensor = _obstacle_start_positions
    obstacle_directions: torch.Tensor = _obstacle_directions
    
    # ========================================================================
    # 獲取環境原點（用於將世界座標轉換為環境內相對座標）
    # ========================================================================
    # 重要：root_pos_w 是「世界座標」，但邊界檢查需要「環境內相對座標」
    # 例如：env_0 中心在 (0,0)，env_1 中心在 (15,0)（env_spacing=15）
    # 如果不轉換，env_1 的障礙物會被誤判為超出邊界
    env_origins = env.scene.env_origins[env_ids, :2]  # (len(env_ids), 2)
    
    for i in range(num_obstacles):
        try:
            obstacle = env.scene[f"obstacle_{i}"]
            
            # 獲取當前位置（世界座標）
            pos_world = obstacle.data.root_pos_w[env_ids, :2]  # (len(env_ids), 2)
            
            # 轉換為環境內相對座標（減去環境原點）
            pos_local = pos_world - env_origins  # (len(env_ids), 2)
            
            # 獲取移動狀態（使用局部引用）
            start_pos = obstacle_start_positions[env_ids, i]  # (len(env_ids), 2)
            direction_angle = obstacle_directions[env_ids, i]  # (len(env_ids),)
            
            # 計算從起始位置的位移（使用世界座標，因為起始位置也是世界座標）
            displacement = pos_world - start_pos
            
            # 計算沿移動方向的距離（投影）
            # 移動方向單位向量
            dir_vec = torch.stack([
                torch.cos(direction_angle),
                torch.sin(direction_angle)
            ], dim=-1)  # (len(env_ids), 2)
            
            # 投影距離
            travel_dist = torch.sum(displacement * dir_vec, dim=-1)  # (len(env_ids),)
            
            # 檢查是否需要反轉方向：
            # 1. 移動距離超過 travel_distance
            # 2. 超出邊界（使用環境內相對座標！）
            # 重要：使用 OR 合併條件，避免雙重反轉（方向加兩次 π 會導致方向不變）
            need_reverse_travel = (travel_dist.abs() >= travel_distance)
            out_of_bounds = (pos_local.abs() > boundary).any(dim=-1)  # 使用相對座標
            need_reverse = need_reverse_travel | out_of_bounds  # 合併條件，只反轉一次
            
            # 反轉方向（角度 + π）
            if need_reverse.any():
                obstacle_directions[env_ids[need_reverse], i] += math.pi
                # 更新起始位置為當前位置（新的移動起點，使用世界座標）
                obstacle_start_positions[env_ids[need_reverse], i] = pos_world[need_reverse]
            
            # 邊界保護：將超出邊界的障礙物位置 clamp 回邊界內
            # 重要：必須寫回模擬器，否則下次檢查仍然超出邊界
            if out_of_bounds.any():
                # 獲取超出邊界的環境 ID
                oob_env_ids = env_ids[out_of_bounds]
                oob_env_origins = env_origins[out_of_bounds]
                
                # 獲取當前完整位置和姿態（世界座標）
                full_pos = obstacle.data.root_pos_w[oob_env_ids, :3].clone()
                full_quat = obstacle.data.root_quat_w[oob_env_ids, :].clone()
                
                # 將世界座標轉換為相對座標，clamp，再轉回世界座標
                local_xy = full_pos[:, :2] - oob_env_origins
                local_xy = torch.clamp(local_xy, -boundary, boundary)
                full_pos[:, :2] = local_xy + oob_env_origins
                
                # 寫回模擬器
                obstacle.write_root_pose_to_sim(
                    torch.cat([full_pos, full_quat], dim=-1),
                    env_ids=oob_env_ids
                )
                # 更新起始位置為 clamp 後的位置（世界座標）
                obstacle_start_positions[oob_env_ids, i] = full_pos[:, :2]
            
            # 獲取當前方向（可能已經更新）
            current_direction = obstacle_directions[env_ids, i]
            
            # 計算速度向量
            velocity = torch.stack([
                torch.cos(current_direction) * max_velocity,
                torch.sin(current_direction) * max_velocity
            ], dim=-1)  # (len(env_ids), 2)
            
            # 設置速度 (vx, vy, vz, wx, wy, wz)
            vel_3d = torch.zeros(len(env_ids), 6, device=env.device)
            vel_3d[:, 0] = velocity[:, 0]  # vx
            vel_3d[:, 1] = velocity[:, 1]  # vy
            # vz, wx, wy, wz 保持為 0
            
            obstacle.write_root_velocity_to_sim(vel_3d, env_ids=env_ids)
            
        except KeyError:
            continue


def get_obstacle_start_positions() -> torch.Tensor | None:
    """獲取障礙物起始位置（公開接口）"""
    return _obstacle_start_positions


def get_obstacle_directions() -> torch.Tensor | None:
    """獲取障礙物移動方向（公開接口）"""
    return _obstacle_directions


def update_obstacle_start_position(env_ids: torch.Tensor, obstacle_idx: int, positions: torch.Tensor):
    """更新指定障礙物的起始位置"""
    global _obstacle_start_positions
    if _obstacle_start_positions is not None:
        _obstacle_start_positions[env_ids, obstacle_idx] = positions


def update_obstacle_direction(env_ids: torch.Tensor, obstacle_idx: int, directions: torch.Tensor):
    """更新指定障礙物的移動方向"""
    global _obstacle_directions
    if _obstacle_directions is not None:
        _obstacle_directions[env_ids, obstacle_idx] = directions
