#!/usr/bin/env python3
"""層級式導航測試腳本

測試 AIT* + RL (PPO) 層級式導航系統的核心組件：
1. 局部目標提取器（Carrot-on-stick）
2. 層級式導航觀測函數
3. 層級式導航獎勵函數
"""

import torch
import numpy as np
from pathlib import Path

# 添加路徑
import sys
charge_skrl_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(charge_skrl_path))

from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.path_planner.local_goal_extractor import (
    LocalGoalExtractor,
    LocalGoalConfig,
    extract_local_goals_batch,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.observations.hierarchical_navigation import (
    local_goal_polar,
    local_goal_cartesian,
    navigation_features,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.rewards.hierarchical_rewards import (
    local_goal_reached_reward,
    progress_to_local_goal,
    heading_alignment_reward,
    hierarchical_navigation_reward,
)


def create_mock_env(num_envs: int = 4):
    """創建模擬環境（用於測試）"""
    class MockAsset:
        def __init__(self):
            self.data = MockData()

    class MockData:
        def __init__(self):
            self.root_pos_w = torch.zeros(num_envs, 3)
            self.root_quat_w = torch.zeros(num_envs, 4)
            self.root_quat_w[:, 0] = 1.0  # 單位四元數
            self.root_lin_vel_b = torch.zeros(num_envs, 3)
            self.root_ang_vel_b = torch.zeros(num_envs, 3)

    class MockSensor:
        def __init__(self):
            self.data = MockSensorData()

    class MockSensorData:
        def __init__(self):
            self.pos_w = torch.zeros(num_envs, 3)
            self.ray_hits_w = torch.ones(num_envs, 72, 3) * 10.0

    class MockCommandManager:
        def __init__(self):
            self.goals = torch.zeros(num_envs, 3)

        def get_command(self, name):
            return self.goals

    class MockActionManager:
        def __init__(self):
            self.action = torch.zeros(num_envs, 2)

        def get_current_action(self):
            return self.action

    class MockScene:
        def __init__(self):
            self.robot = MockAsset()
            self.sensors = {"lidar": MockSensor()}

    class MockEnv:
        def __init__(self):
            self.num_envs = num_envs
            self.device = torch.device("cpu")
            self.scene = MockScene()
            self.command_manager = MockCommandManager()
            self.action_manager = MockActionManager()
            self._local_goal_world = torch.zeros(num_envs, 2)

        @property
        def unwrapped(self):
            return self

    return MockEnv()


def test_local_goal_extractor():
    """測試局部目標提取器"""
    print("\n=== 測試 1: 局部目標提取器 ===")

    # 創建一條直線路徑（從 (0,0) 到 (10,0)）
    path = torch.tensor([
        [0.0, 0.0],
        [2.0, 0.0],
        [4.0, 0.0],
        [6.0, 0.0],
        [8.0, 0.0],
        [10.0, 0.0],
    ])

    # 配置
    cfg = LocalGoalConfig(
        lookahead_distance=2.0,
        min_update_distance=0.3,
    )

    extractor = LocalGoalExtractor(cfg, num_envs=1)

    # 測試 1: 機器人在起點
    robot_pos = torch.tensor([0.0, 0.0])
    goal = extractor.extract_local_goal(robot_pos, path, env_id=0)
    print(f"機器人在 (0, 0): 局部目標 = {goal.numpy()}")
    assert goal[0].item() == 2.0, f"應該返回 (2, 0)，但得到 {goal}"
    print("✓ 測試 1 通過")

    # 測試 2: 機器人在 (3, 0)
    robot_pos = torch.tensor([3.0, 0.0])
    goal = extractor.extract_local_goal(robot_pos, path, env_id=0)
    print(f"機器人在 (3, 0): 局部目標 = {goal.numpy()}")
    # 預期: 前方 2 米 = (5, 0)
    assert abs(goal[0].item() - 5.0) < 0.1, f"應該接近 (5, 0)，但得到 {goal}"
    print("✓ 測試 2 通過")

    # 測試 3: 機器人接近終點
    robot_pos = torch.tensor([9.5, 0.0])
    goal = extractor.extract_local_goal(robot_pos, path, env_id=0)
    print(f"機器人在 (9.5, 0): 局部目標 = {goal.numpy()}")
    # 預期: 返回終點 (10, 0)
    assert goal[0].item() == 10.0, f"應該返回終點 (10, 0)，但得到 {goal}"
    print("✓ 測試 3 通過")

    print("\n局部目標提取器測試完成！\n")


def test_navigation_observations():
    """測試導航觀測函數"""
    print("\n=== 測試 2: 導航觀測函數 ===")

    from isaaclab.managers import SceneEntityCfg

    env = create_mock_env(num_envs=4)

    # 設置機器人位置（4 個環境）
    env.scene.robot.data.root_pos_w = torch.tensor([
        [0.0, 0.0, 0.0],   # env 0
        [2.0, 0.0, 0.0],   # env 1
        [4.0, 0.0, 0.0],   # env 2
        [6.0, 0.0, 0.0],   # env 3
    ])

    # 設置目標位置
    env.command_manager.goals = torch.tensor([
        [10.0, 0.0, 0.0],  # env 0: 目標在前方 10m
        [10.0, 0.0, 0.0],  # env 1
        [10.0, 0.0, 0.0],  # env 2
        [10.0, 0.0, 0.0],  # env 3
    ])

    # 設置局部目標
    env._local_goal_world = torch.tensor([
        [2.0, 0.0],   # env 0: 局部目標 2m
        [4.0, 0.0],   # env 1: 局部目標 2m
        [6.0, 0.0],   # env 2: 局部目標 2m
        [8.0, 0.0],   # env 3: 局部目標 2m
    ])

    robot_cfg = SceneEntityCfg("robot")
    goal_cfg = SceneEntityCfg("goal_command")

    # 測試極坐標觀測
    polar = local_goal_polar(env, robot_cfg, goal_cfg)
    print(f"極坐標觀測: {polar}")
    print(f"  形狀: {polar.shape}")
    assert polar.shape == (4, 2), f"預期 (4, 2)，得到 {polar.shape}"
    print("✓ 極坐標觀測測試通過")

    # 測試笛卡爾坐標觀測
    cartesian = local_goal_cartesian(env, robot_cfg, goal_cfg)
    print(f"笛卡爾坐標觀測: {cartesian}")
    assert cartesian.shape == (4, 2), f"預期 (4, 2)，得到 {cartesian.shape}"
    print("✓ 笛卡爾坐標觀測測試通過")

    # 測試導航特征
    nav_features = navigation_features(env, robot_cfg, goal_cfg)
    print(f"導航特征: {nav_features}")
    print(f"  形狀: {nav_features.shape}")
    assert nav_features.shape == (4, 6), f"預期 (4, 6)，得到 {nav_features.shape}"
    print("✓ 導航特征測試通過")

    print("\n導航觀測測試完成！\n")


def test_navigation_rewards():
    """測試導航獎勵函數"""
    print("\n=== 測試 3: 導航獎勵函數 ===")

    from isaaclab.managers import SceneEntityCfg

    env = create_mock_env(num_envs=4)

    # 設置機器人位置
    env.scene.robot.data.root_pos_w = torch.tensor([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
    ])

    # 設置目標位置
    env.command_manager.goals = torch.tensor([
        [10.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
    ])

    # 設置局部目標
    env._local_goal_world = torch.tensor([
        [2.0, 0.0],
        [4.0, 0.0],
        [6.0, 0.0],
        [8.0, 0.0],
    ])

    # 初始化前一次距離
    env._prev_goal_distance = torch.tensor([3.0, 3.0, 3.0, 3.0])

    robot_cfg = SceneEntityCfg("robot")
    goal_cfg = SceneEntityCfg("goal_command")
    sensor_cfg = SceneEntityCfg("lidar")

    # 測試抵達獎勵
    reach_reward = local_goal_reached_reward(env, robot_cfg, goal_cfg, threshold=0.5, reward=10.0)
    print(f"抵達獎勵: {reach_reward}")
    assert reach_reward.shape == (4,), f"預期 (4,)，得到 {reach_reward.shape}"
    print("✓ 抵達獎勵測試通過")

    # 測試前進獎勵
    progress_reward = progress_to_local_goal(env, robot_cfg, goal_cfg)
    print(f"前進獎勵: {progress_reward}")
    print("  (正值表示靠近目標，負值表示遠離)")
    assert progress_reward.shape == (4,), f"預期 (4,)，得到 {progress_reward.shape}"
    print("✓ 前進獎勵測試通過")

    # 測試航向對齊獎勵
    alignment_reward = heading_alignment_reward(env, robot_cfg, goal_cfg)
    print(f"航向對齊獎勵: {alignment_reward}")
    assert alignment_reward.shape == (4,), f"預期 (4,)，得到 {alignment_reward.shape}"
    print("✓ 航向對齊獎勵測試通過")

    # 測試完整獎勵函數
    total_reward = hierarchical_navigation_reward(env, robot_cfg, goal_cfg, sensor_cfg)
    print(f"總獎勵: {total_reward}")
    assert total_reward.shape == (4,), f"預期 (4,)，得到 {total_reward.shape}"
    print("✓ 完整獎勵函數測試通過")

    print("\n導航獎勵測試完成！\n")


def test_path_scenario():
    """測試完整路徑跟隨場景"""
    print("\n=== 測試 4: 完整路徑跟隨場景 ===")

    from isaaclab.managers import SceneEntityCfg

    env = create_mock_env(num_envs=1)

    # 創建一條 S 形路徑
    t = torch.linspace(0, 2*np.pi, 20)
    path = torch.stack([
        4 * torch.cos(t) + 5,
        2 * torch.sin(2*t),
    ], dim=1)

    print(f"路徑點數: {len(path)}")

    # 模擬機器人沿著路徑移動
    cfg = LocalGoalConfig(lookahead_distance=2.0)
    extractor = LocalGoalExtractor(cfg, num_envs=1)

    positions = [0, 5, 10, 15]
    robot_cfg = SceneEntityCfg("robot")
    goal_cfg = SceneEntityCfg("goal_command")

    for idx in positions:
        robot_pos = path[idx]
        env.scene.robot.data.root_pos_w[0, :2] = robot_pos

        # 提取局部目標
        local_goal = extractor.extract_local_goal(robot_pos, path, env_id=0)
        env._local_goal_world = local_goal.unsqueeze(0)
        env.command_manager.goals[0, :2] = path[-1]

        # 計算觀測
        nav = navigation_features(env, robot_cfg, goal_cfg)
        print(f"  位置 {idx}: {robot_pos.numpy()}")
        print(f"    局部目標: {local_goal.numpy()}")
        print(f"    導航觀測: {nav[0].numpy()}")

    print("✓ 完整路徑跟隨場景測試通過")
    print("\n場景測試完成！\n")


def main():
    """運行所有測試"""
    print("="*60)
    print("層級式導航系統測試")
    print("="*60)

    try:
        test_local_goal_extractor()
        test_navigation_observations()
        test_navigation_rewards()
        test_path_scenario()

        print("\n" + "="*60)
        print("所有測試通過！✓")
        print("="*60)

    except AssertionError as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
