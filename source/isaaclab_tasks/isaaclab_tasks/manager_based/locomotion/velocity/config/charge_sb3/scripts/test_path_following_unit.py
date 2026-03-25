#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
路徑跟隨獎勵函數單元測試

不需要完整的 Isaac Lab 環境，直接測試獎勵函數的數值計算邏輯。
使用 Mock 環境模擬必要的接口。
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from typing import Any

# 導入要測試的函數
import sys
sys.path.insert(0, '/home/aa/IsaacLab/source')

# 這裡我們直接實現測試邏輯，避免需要完整的 isaaclab 導入
# 我們將複製核心計算邏輯進行測試


@dataclass
class MockRobotData:
    """模擬機器人數據"""
    root_pos_w: torch.Tensor  # [num_envs, 3]
    root_lin_vel_w: torch.Tensor  # [num_envs, 3]


@dataclass
class MockRobot:
    """模擬機器人對象"""
    data: MockRobotData


@dataclass
class MockEnv:
    """模擬環境對象"""
    robot: MockRobot
    num_envs: int

    def __getitem__(self, key):
        if key == "robot":
            return self.robot
        return None


def create_straight_path_points(start: np.ndarray, goal: np.ndarray, num_points: int = 10) -> torch.Tensor:
    """創建直線路徑點

    Args:
        start: [2] 起點坐標
        goal: [2] 終點坐標
        num_points: 路徑點數量

    Returns:
        [1, num_points, 2] 路徑點張量
    """
    t = np.linspace(0, 1, num_points)
    path = np.zeros((num_points, 2))
    path[:, 0] = start[0] + t * (goal[0] - start[0])
    path[:, 1] = start[1] + t * (goal[1] - start[1])
    return torch.from_numpy(path).float().unsqueeze(0)  # [1, num_points, 2]


def compute_progress_along_path(
    robot_pos: torch.Tensor,
    path_points: torch.Tensor,
) -> torch.Tensor:
    """計算沿路徑前進的距離獎勵

    Args:
        robot_pos: [num_envs, 2] 機器人位置
        path_points: [num_envs, num_points, 2] 路徑點

    Returns:
        [num_envs] 前進距離獎勵
    """
    num_envs = robot_pos.shape[0]
    device = robot_pos.device
    rewards = torch.zeros(num_envs, device=device)

    for env_id in range(num_envs):
        path = path_points[env_id]
        distances = torch.norm(path - robot_pos[env_id], dim=-1)
        closest_idx = torch.argmin(distances)

        # 計算到最近點的距離
        distance_to_path = distances[closest_idx]

        # 簡化版本：根據距離給獎勵
        # 越接近路徑終點，獎勵越高
        progress_ratio = closest_idx.item() / (path.shape[0] - 1)
        rewards[env_id] = progress_ratio * 10.0  # 最大獎勵 10

    return rewards


def compute_cross_track_error(
    robot_pos: torch.Tensor,
    path_points: torch.Tensor,
    safe_distance: float = 0.5,
) -> torch.Tensor:
    """計算偏離路徑的垂直距離懲罰

    Args:
        robot_pos: [num_envs, 2] 機器人位置
        path_points: [num_envs, num_points, 2] 路徑點
        safe_distance: 安全距離

    Returns:
        [num_envs] cross-track error (負值 = 懲罰)
    """
    num_envs = robot_pos.shape[0]
    device = robot_pos.device
    cte = torch.zeros(num_envs, device=device)

    for env_id in range(num_envs):
        path = path_points[env_id]
        min_cte = float('inf')

        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]

            segment = p2 - p1
            segment_length_sq = torch.sum(segment ** 2)

            if segment_length_sq < 1e-6:
                continue

            robot_to_p1 = robot_pos[env_id] - p1
            t = torch.clamp(
                torch.sum(robot_to_p1 * segment) / segment_length_sq,
                min=0.0, max=1.0
            )

            projection = p1 + t * segment
            dist_to_segment = torch.norm(robot_pos[env_id] - projection)

            if dist_to_segment > safe_distance:
                adjusted_dist = dist_to_segment - safe_distance
            else:
                adjusted_dist = torch.tensor(0.0, device=device)

            min_cte = min(min_cte, adjusted_dist.item())

        cte[env_id] = -min_cte

    return cte


def compute_path_direction_reward(
    robot_pos: torch.Tensor,
    robot_vel: torch.Tensor,
    path_points: torch.Tensor,
) -> torch.Tensor:
    """計算路徑方向一致性獎勵

    Args:
        robot_pos: [num_envs, 2] 機器人位置
        robot_vel: [num_envs, 2] 機器人速度
        path_points: [num_envs, num_points, 2] 路徑點

    Returns:
        [num_envs] 方向一致性獎勵 [-1, 1]
    """
    num_envs = robot_pos.shape[0]
    device = robot_pos.device

    # 歸一化速度
    speed = torch.norm(robot_vel, dim=-1, keepdim=True)
    speed = torch.clamp(speed, min=1e-6)
    velocity_direction = robot_vel / speed

    rewards = torch.zeros(num_envs, device=device)

    for env_id in range(num_envs):
        path = path_points[env_id]
        min_dist = float('inf')
        best_direction = None

        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]

            segment = p2 - p1
            segment_length = torch.norm(segment)

            if segment_length < 1e-6:
                continue

            segment_direction = segment / segment_length

            robot_to_p1 = robot_pos[env_id] - p1
            dist_to_segment = torch.norm(
                robot_to_p1 - torch.clamp(
                    torch.sum(robot_to_p1 * segment) / (segment_length ** 2),
                    min=0.0, max=1.0
                ) * segment
            )

            if dist_to_segment < min_dist:
                min_dist = dist_to_segment.item()
                best_direction = segment_direction

        if best_direction is not None:
            direction_similarity = torch.sum(
                velocity_direction[env_id] * best_direction
            )
            rewards[env_id] = direction_similarity

    return rewards


class TestPathFollowingRewards:
    """路徑跟隨獎勵函數測試類"""

    def __init__(self):
        self.device = torch.device('cpu')

    def test_progress_along_path(self):
        """測試沿路徑前進獎勵"""
        print("\n" + "=" * 60)
        print("測試 1: progress_along_path（沿路徑前進獎勵）")
        print("=" * 60)

        # 創建直線路徑：從 (0, 0) 到 (10, 0)
        path = create_straight_path_points(
            np.array([0.0, 0.0]),
            np.array([10.0, 0.0]),
            num_points=11
        )  # [1, 11, 2]

        # 測試不同位置的機器人
        test_cases = [
            ("起點", np.array([0.0, 0.0])),
            ("1/4 處", np.array([2.5, 0.0])),
            ("中點", np.array([5.0, 0.0])),
            ("3/4 處", np.array([7.5, 0.0])),
            ("終點", np.array([10.0, 0.0])),
            ("偏離路徑", np.array([5.0, 2.0])),
        ]

        print(f"\n路徑: (0, 0) -> (10, 0)")
        print(f"路徑點數量: {path.shape[1]}\n")

        for name, pos in test_cases:
            robot_pos = torch.from_numpy(pos).float().unsqueeze(0)  # [1, 2]
            reward = compute_progress_along_path(robot_pos, path)
            print(f"  {name:>10}: pos=({pos[0]:>5.1f}, {pos[1]:>5.1f}) -> reward={reward[0].item():>6.2f}")

        print("\n✅ 預期：越接近終點，獎勵越高")

    def test_cross_track_error(self):
        """測試 cross-track error"""
        print("\n" + "=" * 60)
        print("測試 2: cross_track_error（偏離路徑懲罰）")
        print("=" * 60)

        # 創建直線路徑：從 (0, 0) 到 (10, 0)
        path = create_straight_path_points(
            np.array([0.0, 0.0]),
            np.array([10.0, 0.0]),
            num_points=11
        )

        safe_distance = 0.5

        # 測試不同偏離距離
        test_cases = [
            ("在路徑上", np.array([5.0, 0.0])),
            ("安全範圍內", np.array([5.0, 0.3])),
            ("安全邊界", np.array([5.0, 0.5])),
            ("超出安全距離", np.array([5.0, 1.0])),
            ("嚴重偏離", np.array([5.0, 2.0])),
        ]

        print(f"\n路徑: (0, 0) -> (10, 0)")
        print(f"安全距離: {safe_distance}m\n")

        for name, pos in test_cases:
            robot_pos = torch.from_numpy(pos).float().unsqueeze(0)
            cte = compute_cross_track_error(robot_pos, path, safe_distance)
            deviation = abs(pos[1])
            print(f"  {name:>10}: pos=({pos[0]:>5.1f}, {pos[1]:>5.1f}) "
                  f"deviation={deviation:>4.1f}m -> cte={cte[0].item():>6.2f}")

        print("\n✅ 預期：偏離越大，CTE 越小（懲罰越大）")

    def test_path_direction_reward(self):
        """測試路徑方向一致性獎勵"""
        print("\n" + "=" * 60)
        print("測試 3: path_direction_reward（方向一致性獎勵）")
        print("=" * 60)

        # 創建直線路徑：從 (0, 0) 到 (10, 0)
        path = create_straight_path_points(
            np.array([0.0, 0.0]),
            np.array([10.0, 0.0]),
            num_points=11
        )

        # 測試不同速度方向
        speed = 1.0
        test_cases = [
            ("與路徑同向", np.array([speed, 0.0])),
            ("與路徑反向", np.array([-speed, 0.0])),
            ("垂直路徑", np.array([0.0, speed])),
            ("45° 夾角", np.array([speed * 0.707, speed * 0.707])),
            ("-45° 夾角", np.array([speed * 0.707, -speed * 0.707])),
        ]

        print(f"\n路徑方向: (1, 0) [向右]")
        print(f"速度大小: {speed} m/s\n")

        for name, vel in test_cases:
            robot_pos = torch.from_numpy(np.array([5.0, 0.0])).float().unsqueeze(0)
            robot_vel = torch.from_numpy(vel).float().unsqueeze(0)
            reward = compute_path_direction_reward(robot_pos, robot_vel, path)

            # 計算夾角
            expected_cos = vel[0] / speed
            print(f"  {name:>10}: vel=({vel[0]:>6.2f}, {vel[1]:>6.2f}) -> "
                  f"reward={reward[0].item():>6.2f} (expected={expected_cos:>6.2f})")

        print("\n✅ 預期：方向一致時 reward=1，相反時=-1")

    def test_combined_reward(self):
        """測試綜合獎勵"""
        print("\n" + "=" * 60)
        print("測試 4: path_following_reward（綜合獎勵）")
        print("=" * 60)

        path = create_straight_path_points(
            np.array([0.0, 0.0]),
            np.array([10.0, 0.0]),
            num_points=11
        )

        # 權重配置
        weights = {
            "progress": 1.0,
            "cross_track": 0.5,
            "direction": 0.3,
        }

        print(f"\n權重配置:")
        print(f"  progress:     {weights['progress']}")
        print(f"  cross_track:  {weights['cross_track']}")
        print(f"  direction:    {weights['direction']}")

        # 模擬機器人沿路徑移動的軌跡
        print(f"\n模擬機器人沿路徑移動...")

        trajectory = []
        rewards_history = []

        num_steps = 10
        for step in range(num_steps):
            # 機器人位置：沿 X 軸移動
            x = step * 1.0
            y = 0.1 * np.sin(step * 0.5)  # 添加輕微偏離

            robot_pos = torch.from_numpy(np.array([x, y])).float().unsqueeze(0)

            # 速度方向
            vx = 1.0
            vy = 0.1 * np.cos(step * 0.5)
            robot_vel = torch.from_numpy(np.array([vx, vy])).float().unsqueeze(0)

            # 計算各項獎勵
            progress = compute_progress_along_path(robot_pos, path)
            cte = compute_cross_track_error(robot_pos, path, safe_distance=0.5)
            direction = compute_path_direction_reward(robot_pos, robot_vel, path)

            # 綜合獎勵
            total = (
                weights["progress"] * progress +
                weights["cross_track"] * cte +
                weights["direction"] * direction
            )

            trajectory.append((x, y))
            rewards_history.append({
                'progress': progress[0].item(),
                'cte': cte[0].item(),
                'direction': direction[0].item(),
                'total': total[0].item(),
            })

            if step % 2 == 0:
                print(f"  Step {step:2d}: pos=({x:>4.1f}, {y:>5.2f}) | "
                      f"prog={progress[0].item():>5.2f} cte={cte[0].item():>5.2f} "
                      f"dir={direction[0].item():>5.2f} total={total[0].item():>6.2f}")

        # 統計
        progress_sum = sum(r['progress'] for r in rewards_history)
        cte_sum = sum(r['cte'] for r in rewards_history)
        direction_sum = sum(r['direction'] for r in rewards_history)
        total_sum = sum(r['total'] for r in rewards_history)

        print(f"\n累積獎勵:")
        print(f"  Progress:   {progress_sum:>8.2f}")
        print(f"  CTE:        {cte_sum:>8.2f}")
        print(f"  Direction:  {direction_sum:>8.2f}")
        print(f"  " + "-" * 30)
        print(f"  TOTAL:      {total_sum:>8.2f}")

        print("\n✅ 綜合獎勵測試完成")

    def run_all_tests(self):
        """運行所有測試"""
        print("\n" + "=" * 60)
        print("     路徑跟隨獎勵函數單元測試")
        print("=" * 60)

        self.test_progress_along_path()
        self.test_cross_track_error()
        self.test_path_direction_reward()
        self.test_combined_reward()

        print("\n" + "=" * 60)
        print("     ✅ 所有測試完成")
        print("=" * 60)


def main():
    """主函數"""
    tester = TestPathFollowingRewards()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
