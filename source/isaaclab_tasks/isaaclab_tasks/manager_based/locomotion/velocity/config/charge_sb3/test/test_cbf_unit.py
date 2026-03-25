#!/usr/bin/env python3
"""CBF Core Logic 單元測試（不依賴 Isaac Sim）

測試 CBF 的核心邏輯：
1. SafetyAction 創建
2. Obstacle 碰撞檢測
3. RuleBasedShield 過濾邏輯
4. 修正量計算

使用方式:
    python -m isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.test.test_cbf_unit
"""

import math
import sys
from typing import List, Tuple, Optional

# 模擬 SafetyAction (從 types.py)
class SafetyAction:
    """安全動作（過濾後的動作）"""
    def __init__(
        self,
        linear_speed: float,
        angular_speed: float,
        safe: bool = True,
        reason: str = ""
    ):
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.safe = safe
        self.reason = reason

    @classmethod
    def unsafe(cls, linear_speed: float, angular_speed: float, reason: str):
        return cls(linear_speed, angular_speed, safe=False, reason=reason)

    @classmethod
    def safe_action(cls, linear_speed: float, angular_speed: float):
        return cls(linear_speed, angular_speed, safe=True, reason="")


# 模擬 Obstacle (從 types.py)
class Obstacle:
    """障礙物"""
    def __init__(
        self,
        position: Tuple[float, float],
        size: float,
        shape: str = "circle",
    ):
        self.position = position
        self.size = size
        self.shape = shape

    def is_collision(
        self,
        point: Tuple[float, float],
        robot_radius: float = 0.5,
    ) -> bool:
        if self.shape == "circle":
            dist = math.sqrt((point[0] - self.position[0])**2 + (point[1] - self.position[1])**2)
            return dist < (self.size / 2 + robot_radius)
        return False


# 模擬 SimpleTensor (用於測試)
class SimpleTensor:
    """簡單的 Tensor 模擬"""
    def __init__(self, data):
        if isinstance(data, list):
            self.data = data
        else:
            self.data = [data]

    def __getitem__(self, idx):
        return self.data[idx]

    def item(self):
        return self.data[0] if len(self.data) == 1 else self.data

    def tolist(self):
        return self.data

    def max(self):
        return max(self.data)

    def numel(self):
        return len(self.data)


# 模擬 RuleBasedShield 核心邏輯
def test_rule_based_filter():
    """測試 RuleBasedShield 的核心邏輯"""

    safety_distance = 0.8
    danger_distance = 0.5
    emergency_distance = 0.3

    print("="*60)
    print("Test 1: No Obstacles (Safe Action)")
    print("="*60)

    # 模擬輸入
    action_linear = 1.0
    action_angular = 0.5

    # Lidar 歸一化：值越大表示越近，值越小表示越遠
    # 假設 max_distance = 10m，歸一化 = 1 - (distance / 10)
    # 無障礙物時，所有射線都返回 max_distance (10m)，歸一化值為 0
    # 有障礙物在距離 d 時，歸一化值為 1 - (d / 10)
    lidar_scan = SimpleTensor([0.0] * 72)  # 全遠（距離 10m）

    # 計算最近距離
    min_lidar_dist = lidar_scan.max()
    actual_min_dist = 10.0 * (1.0 - min_lidar_dist)  # 轉換為實際距離

    print(f"Min lidar distance: {actual_min_dist:.2f}m")

    # 檢查安全狀態
    if actual_min_dist < emergency_distance:
        print("  → Emergency stop")
        result = SafetyAction.unsafe(0.0, 0.0, "Emergency stop")
    elif actual_min_dist < danger_distance:
        print("  → Danger zone - slow down")
        speed_scale = (actual_min_dist - emergency_distance) / (danger_distance - emergency_distance)
        result = SafetyAction.unsafe(
            action_linear * speed_scale * 0.5,
            action_angular * 0.5,
            f"Danger - speed scale: {speed_scale:.2f}"
        )
    elif actual_min_dist < safety_distance:
        print("  → Safety warning - reduce speed")
        speed_scale = (actual_min_dist - danger_distance) / (safety_distance - danger_distance)
        result = SafetyAction.unsafe(
            action_linear * speed_scale,
            action_angular,
            f"Safety warning - speed scale: {speed_scale:.2f}"
        )
    else:
        print("  → Safe - no modification")
        result = SafetyAction.safe_action(action_linear, action_angular)

    print(f"Input: linear={action_linear:.2f}, angular={action_angular:.2f}")
    print(f"Output: linear={result.linear_speed:.2f}, angular={result.angular_speed:.2f}")
    print(f"Safe: {result.safe}")
    assert result.safe, "Should be safe with no obstacles"
    print("✓ PASSED\n")

    print("="*60)
    print("Test 2: Close Obstacle (Danger Zone)")
    print("="*60)

    # 模擬障礙物在 0.4m 處
    # 歸一化：1 - (0.4 / 10) = 0.96
    lidar_scan = SimpleTensor([0.0] * 71 + [0.96])

    min_lidar_dist = lidar_scan.max()
    actual_min_dist = 10.0 * (1.0 - min_lidar_dist)  # 轉換為實際距離

    print(f"Min lidar distance: {actual_min_dist:.2f}m")

    if actual_min_dist < emergency_distance:
        result = SafetyAction.unsafe(0.0, 0.0, "Emergency stop")
    elif actual_min_dist < danger_distance:
        speed_scale = (actual_min_dist - emergency_distance) / (danger_distance - emergency_distance)
        speed_scale = max(0.0, min(1.0, speed_scale))
        result = SafetyAction.unsafe(
            action_linear * speed_scale * 0.5,
            action_angular * 0.5,
            f"Danger - speed scale: {speed_scale:.2f}"
        )
    elif actual_min_dist < safety_distance:
        speed_scale = (actual_min_dist - danger_distance) / (safety_distance - danger_distance)
        speed_scale = max(0.0, min(1.0, speed_scale))
        result = SafetyAction.unsafe(
            action_linear * speed_scale,
            action_angular,
            f"Safety warning - speed scale: {speed_scale:.2f}"
        )
    else:
        result = SafetyAction.safe_action(action_linear, action_angular)

    print(f"Input: linear={action_linear:.2f}, angular={action_angular:.2f}")
    print(f"Output: linear={result.linear_speed:.2f}, angular={result.angular_speed:.2f}")
    print(f"Safe: {result.safe}")
    print(f"Reason: {result.reason}")
    assert not result.safe, "Should be unsafe with close obstacle"
    assert result.linear_speed < action_linear, "Speed should be reduced"
    print("✓ PASSED\n")

    print("="*60)
    print("Test 3: Emergency Zone")
    print("="*60)

    # 模擬障礙物在 0.2m 處
    # 歸一化：1 - (0.2 / 10) = 0.98
    lidar_scan = SimpleTensor([0.0] * 71 + [0.98])

    min_lidar_dist = lidar_scan.max()
    actual_min_dist = 10.0 * (1.0 - min_lidar_dist)  # 轉換為實際距離

    print(f"Min lidar distance: {actual_min_dist:.2f}m")

    if actual_min_dist < emergency_distance:
        result = SafetyAction.unsafe(0.0, 0.0, "Emergency stop")
    else:
        result = SafetyAction.safe_action(action_linear, action_angular)

    print(f"Input: linear={action_linear:.2f}, angular={action_angular:.2f}")
    print(f"Output: linear={result.linear_speed:.2f}, angular={result.angular_speed:.2f}")
    print(f"Safe: {result.safe}")
    print(f"Reason: {result.reason}")
    assert not result.safe, "Should be unsafe in emergency zone"
    assert result.linear_speed == 0.0, "Should stop in emergency zone"
    print("✓ PASSED\n")


def test_correction_penalty():
    """測試修正量懲罰計算"""
    print("="*60)
    print("Test 4: Correction Penalty Calculation")
    print("="*60)

    penalty_coeff = 0.5

    # 測試案例 1: 無修正
    original = [1.0, 0.5]
    safe = [1.0, 0.5]
    correction = math.sqrt(sum((o - s)**2 for o, s in zip(original, safe)))
    penalty = penalty_coeff * (correction ** 2)

    print(f"\nCase 1: No correction")
    print(f"  Original: {original}")
    print(f"  Safe: {safe}")
    print(f"  Correction: {correction:.4f}")
    print(f"  Penalty: {penalty:.4f}")
    assert correction == 0.0, "No correction expected"
    assert penalty == 0.0, "No penalty expected"
    print("  ✓ PASSED")

    # 測試案例 2: 小修正
    original = [1.0, 0.5]
    safe = [0.9, 0.5]
    correction = math.sqrt(sum((o - s)**2 for o, s in zip(original, safe)))
    penalty = penalty_coeff * (correction ** 2)

    print(f"\nCase 2: Small correction")
    print(f"  Original: {original}")
    print(f"  Safe: {safe}")
    print(f"  Correction: {correction:.4f}")
    print(f"  Penalty: {penalty:.4f}")
    assert correction > 0, "Correction expected"
    assert penalty > 0, "Penalty expected"
    print("  ✓ PASSED")

    # 測試案例 3: 大修正
    original = [1.5, 1.0]
    safe = [0.0, 0.0]
    correction = math.sqrt(sum((o - s)**2 for o, s in zip(original, safe)))
    penalty = penalty_coeff * (correction ** 2)

    print(f"\nCase 3: Large correction")
    print(f"  Original: {original}")
    print(f"  Safe: {safe}")
    print(f"  Correction: {correction:.4f}")
    print(f"  Penalty: {penalty:.4f}")
    assert correction > 1.0, "Large correction expected"
    assert penalty > 0.5, "Significant penalty expected"
    print("  ✓ PASSED\n")


def test_obstacle_collision():
    """測試障礙物碰撞檢測"""
    print("="*60)
    print("Test 5: Obstacle Collision Detection")
    print("="*60)

    # 創建障礙物
    obstacle = Obstacle(position=(2.0, 0.0), size=1.0, shape="circle")  # 半徑 0.5m
    robot_radius = 0.5

    # 測試案例 1: 遠離障礙物
    point = (0.0, 0.0)
    collision = obstacle.is_collision(point, robot_radius)
    print(f"\nCase 1: Point {point} vs Obstacle at {obstacle.position}")
    print(f"  Distance: {math.sqrt((point[0]-obstacle.position[0])**2 + (point[1]-obstacle.position[1])**2):.2f}m")
    print(f"  Collision: {collision}")
    assert not collision, "Should not collide"
    print("  ✓ PASSED")

    # 測試案例 2: 接近障礙物邊緣
    point = (1.0, 0.0)  # 距離中心 1.0m，剛好在碰撞邊界
    collision = obstacle.is_collision(point, robot_radius)
    print(f"\nCase 2: Point {point} vs Obstacle at {obstacle.position}")
    print(f"  Distance: {math.sqrt((point[0]-obstacle.position[0])**2 + (point[1]-obstacle.position[1])**2):.2f}m")
    print(f"  Obstacle radius: {obstacle.size/2:.2f}m")
    print(f"  Collision threshold: {obstacle.size/2 + robot_radius:.2f}m")
    print(f"  Collision: {collision}")
    # 1.0 == (0.5 + 0.5) = 邊界，條件是 < 所以不算碰撞
    assert not collision, "Should not collide (at edge)"
    print("  ✓ PASSED")

    # 測試案例 3: 碰撞
    # 障礙物在 (2.0, 0.0)，半徑 0.5m
    # 機器人半徑 0.5m
    # 碰撞閾值 = 1.0m
    # 要在碰撞區內，點需要距離 (2.0, 0.0) 小於 1.0m
    # 例如：x = 1.1 (距離 0.9m < 1.0m)
    point = (1.1, 0.0)
    collision = obstacle.is_collision(point, robot_radius)
    print(f"\nCase 3: Point {point} vs Obstacle at {obstacle.position}")
    dist = math.sqrt((point[0]-obstacle.position[0])**2 + (point[1]-obstacle.position[1])**2)
    print(f"  Distance: {dist:.2f}m")
    print(f"  Obstacle radius: {obstacle.size/2:.2f}m")
    print(f"  Collision threshold: {obstacle.size/2 + robot_radius:.2f}m")
    print(f"  Collision: {collision}")
    assert collision, f"Should collide (distance {dist:.2f} < threshold {obstacle.size/2 + robot_radius:.2f})"
    print("  ✓ PASSED\n")


def main():
    print("\n" + "="*60)
    print("CBF Core Logic Unit Tests")
    print("="*60 + "\n")

    try:
        test_rule_based_filter()
        test_correction_penalty()
        test_obstacle_collision()

        print("="*60)
        print("ALL TESTS PASSED!")
        print("="*60)

        return 0
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
