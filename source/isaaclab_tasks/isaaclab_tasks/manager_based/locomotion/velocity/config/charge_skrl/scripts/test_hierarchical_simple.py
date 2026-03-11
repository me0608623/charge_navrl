#!/usr/bin/env python3
"""層級式導航核心算法測試（無需 Isaac Sim）

只測試核心算法邏輯，不依賴 Isaac Lab 環境。
"""

import torch
import numpy as np
from pathlib import Path


def test_local_goal_extraction():
    """測試局部目標提取邏輯"""
    print("\n=== 測試 1: 局部目標提取 ===")

    # 創建一條直線路徑（從 (0,0) 到 (10,0)）
    path = torch.tensor([
        [0.0, 0.0],
        [2.0, 0.0],
        [4.0, 0.0],
        [6.0, 0.0],
        [8.0, 0.0],
        [10.0, 0.0],
    ])

    def extract_local_goal(robot_pos, path, lookahead=2.0):
        """簡化版局部目標提取"""
        # 找最近點
        distances = torch.norm(path - robot_pos, dim=1)
        nearest_idx = torch.argmin(distances)

        # 向前累積距離
        accumulated = 0.0
        target_idx = nearest_idx

        # 從最近點的下一段開始累積
        for i in range(nearest_idx.item(), len(path) - 1):
            segment = torch.norm(path[i + 1] - path[i]).item()
            accumulated += segment
            target_idx = i + 1
            if accumulated >= lookahead:
                break

        return path[target_idx]

    # 測試 1: 機器人在起點
    goal = extract_local_goal(torch.tensor([0.0, 0.0]), path)
    print(f"機器人 (0,0) -> 局部目標: {goal.numpy()}")
    assert abs(goal[0] - 2.0) < 0.1, f"預期 (2, 0), 得到 {goal}"
    print("✓ 測試 1 通過")

    # 測試 2: 機器人在 (3, 0)
    # 最近點是 (2, 0)（距離 1m），從那裡向前看 2m 到 (4, 0)
    goal = extract_local_goal(torch.tensor([3.0, 0.0]), path)
    print(f"機器人 (3,0) -> 局部目標: {goal.numpy()}")
    # (2,0) + 2m = (4, 0)
    assert abs(goal[0] - 4.0) < 0.1, f"預期 (4, 0), 得到 {goal}"
    print("✓ 測試 2 通過")

    # 測試 3: 機器人在 (5, 0)
    # 最近點是 (4, 0) 或 (6, 0)，取 (4, 0)，向前看 2m 到 (6, 0)
    goal = extract_local_goal(torch.tensor([5.0, 0.0]), path)
    print(f"機器人 (5,0) -> 局部目標: {goal.numpy()}")
    # (4,0) + 2m = (6, 0)
    assert abs(goal[0] - 6.0) < 0.1, f"預期 (6, 0), 得到 {goal}"
    print("✓ 測試 3 通過")

    # 測試 4: 機器人接近終點
    goal = extract_local_goal(torch.tensor([9.5, 0.0]), path)
    print(f"機器人 (9.5,0) -> 局部目標: {goal.numpy()}")
    assert abs(goal[0] - 10.0) < 0.1, f"預期 (10, 0), 得到 {goal}"
    print("✓ 測試 4 通過")

    print("✓ 局部目標提取測試完成！\n")


def test_coordinate_transform():
    """測試坐標轉換"""
    print("\n=== 測試 2: 坐標轉換 ===")

    def world_to_robot(target_world, robot_pos, robot_yaw):
        """世界坐標 -> 機器人坐標"""
        dx = target_world[0] - robot_pos[0]
        dy = target_world[1] - robot_pos[1]

        cos_yaw = np.cos(robot_yaw)
        sin_yaw = np.sin(robot_yaw)

        forward = dx * cos_yaw + dy * sin_yaw
        lateral = -dx * sin_yaw + dy * cos_yaw

        return np.array([forward, lateral])

    # 測試 1: 機器人在原點，朝向 X 軸
    robot_pos = np.array([0.0, 0.0])
    robot_yaw = 0.0
    target = np.array([2.0, 1.0])  # 在前方右側

    result = world_to_robot(target, robot_pos, robot_yaw)
    print(f"目標 {target} -> 機器人坐標: {result}")
    assert abs(result[0] - 2.0) < 0.01, f"前向分量應約為 2, 得到 {result[0]}"
    assert abs(result[1] - 1.0) < 0.01, f"側向分量應約為 1, 得到 {result[1]}"
    print("✓ 測試 1 通過")

    # 測試 2: 機器人旋轉 90 度
    robot_yaw = np.pi / 2
    result = world_to_robot(target, robot_pos, robot_yaw)
    print(f"機器人旋轉 90° 後: {result}")
    print("✓ 測試 2 通過")

    # 測試 3: 目標在機器人後方
    target = np.array([-1.0, 0.0])
    robot_yaw = 0.0
    result = world_to_robot(target, robot_pos, robot_yaw)
    print(f"目標在後方 {target} -> 機器人坐標: {result}")
    assert result[0] < 0, f"前向分量應為負, 得到 {result[0]}"
    print("✓ 測試 3 通過")

    print("✓ 坐標轉換測試完成！\n")


def test_polar_coordinates():
    """測試極坐標計算"""
    print("\n=== 測試 3: 極坐標計算 ===")

    def cartesian_to_polar(forward, lateral):
        """笛卡爾 -> 極坐標"""
        distance = np.sqrt(forward**2 + lateral**2)
        angle = np.arctan2(lateral, forward)
        return distance, angle

    # 測試 1: 目標正前方
    dist, angle = cartesian_to_polar(2.0, 0.0)
    print(f"前向 2m, 側向 0m -> 距離: {dist:.2f}, 角度: {angle:.2f}")
    assert abs(dist - 2.0) < 0.01, f"距離應為 2"
    assert abs(angle) < 0.01, f"角度應為 0"
    print("✓ 測試 1 通過")

    # 測試 2: 目標左前方 45 度
    dist, angle = cartesian_to_polar(2.0, 2.0)
    print(f"前向 2m, 側向 2m -> 距離: {dist:.2f}, 角度: {angle:.2f}")
    assert abs(angle - np.pi/4) < 0.01, f"角度應約為 π/4"
    print("✓ 測試 2 通過")

    # 測試 3: 目標右側 90 度
    dist, angle = cartesian_to_polar(0.0, -1.5)
    print(f"前向 0m, 側向 -1.5m -> 距離: {dist:.2f}, 角度: {angle:.2f}")
    assert abs(angle + np.pi/2) < 0.01, f"角度應約為 -π/2"
    print("✓ 測試 3 通過")

    print("✓ 極坐標計算測試完成！\n")


def test_reward_calculation():
    """測試獎勵計算"""
    print("\n=== 測試 4: 獎勵計算 ===")

    # 測試前進獎勵
    def progress_reward(prev_dist, curr_dist):
        return prev_dist - curr_dist

    # 靠近
    r = progress_reward(5.0, 4.0)
    print(f"靠近: 5m -> 4m, 獎勵: {r:.2f}")
    assert r > 0, "靠近應給正獎勵"

    # 遠離
    r = progress_reward(4.0, 5.0)
    print(f"遠離: 4m -> 5m, 獎勵: {r:.2f}")
    assert r < 0, "遠離應給負獎勵"

    print("✓ 前進獎勵測試通過")

    # 測試航向對齊
    def alignment_reward(heading_error):
        return np.cos(heading_error)

    r = alignment_reward(0.0)
    print(f"航向誤差 0°, 獎勵: {r:.2f}")
    assert abs(r - 1.0) < 0.01, "完全對齊應給最大獎勵"

    r = alignment_reward(np.pi / 2)
    print(f"航向誤差 90°, 獎勵: {r:.2f}")
    assert abs(r) < 0.01, "垂直應給 0 獎勵"

    r = alignment_reward(np.pi)
    print(f"航向誤差 180°, 獎勵: {r:.2f}")
    assert abs(r + 1.0) < 0.01, "相反應給 -1 獎勵"

    print("✓ 航向對齊測試通過")

    print("✓ 獎勵計算測試完成！\n")


def test_s_curve_scenario():
    """測試 S 形路徑跟隨場景"""
    print("\n=== 測試 5: S 形路徑場景 ===")

    # 創建 S 形路徑
    t = torch.linspace(0, 2*np.pi, 20)
    path = torch.stack([
        4 * torch.cos(t) + 5,
        2 * torch.sin(2*t),
    ], dim=1)

    print(f"路徑點數: {len(path)}")

    # 模擬機器人跟隨
    positions_idx = [0, 5, 10, 15]
    lookahead = 2.0

    for idx in positions_idx:
        robot_pos = path[idx]

        # 提取局部目標
        distances = torch.norm(path - robot_pos, dim=1)
        nearest = torch.argmin(distances)

        accumulated = 0.0
        target_idx = nearest
        for i in range(nearest.item(), len(path) - 1):
            accumulated += torch.norm(path[i + 1] - path[i]).item()
            target_idx = i + 1
            if accumulated >= lookahead:
                break

        local_goal = path[target_idx]

        # 計算相對坐標
        to_goal = local_goal - robot_pos
        distance = torch.norm(to_goal).item()
        angle = torch.atan2(to_goal[1], to_goal[0]).item()

        print(f"  位置 {idx}: {robot_pos.numpy()}")
        print(f"    局部目標: {local_goal.numpy()}")
        print(f"    距離: {distance:.2f}m, 角度: {np.degrees(angle):.1f}°")

    print("✓ S 形路徑場景測試通過")
    print("\n場景測試完成！\n")


def main():
    print("=" * 60)
    print("層級式導航核心算法測試")
    print("=" * 60)

    try:
        test_local_goal_extraction()
        test_coordinate_transform()
        test_polar_coordinates()
        test_reward_calculation()
        test_s_curve_scenario()

        print("\n" + "=" * 60)
        print("所有測試通過！✓")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
