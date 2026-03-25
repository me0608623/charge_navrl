#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""AIT* 路徑視覺化測試腳本

展示如何在 Isaac Sim 中視覺化 AIT* 規劃的路徑。

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/test_aitstar_visualization.py

"""

import argparse

from isaaclab.app import AppLauncher

# 添加 argparse 參數
parser = argparse.ArgumentParser(description="AIT* 路徑視覺化測試")
parser.add_argument(
    "--test",
    type=str,
    default="all",
    choices=["basic", "curved", "aitstar", "env", "hierarchical", "all"],
    help="要執行的測試"
)
# 添加 AppLauncher 參數
AppLauncher.add_app_launcher_args(parser)
args_cli, unknown_args = parser.parse_known_args()

# 啟動 omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# Isaac Lab 導入需要在 SimulationApp 啟動後
from isaaclab.envs import ManagerBasedRLEnv
import gymnasium as gym
import isaaclab_tasks
import numpy as np
import torch

# 導入 AIT* 相關模組
from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.mdp.path_planner import (
    AITStarPathVisualizer,
    MultiEnvPathVisualizer,
    create_aitstar_planner,
)


def test_basic_visualization():
    """測試基本路徑視覺化（直線）"""
    print("\n=== 測試 1: 基本直線視覺化 ===")

    # 創建視覺化器
    visualizer = AITStarPathVisualizer(
        prim_path="/World/TestPath",
        path_color=(0.0, 1.0, 0.0),  # 綠色
    )

    # 創建一條簡單的直線路徑
    path_points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
    ])

    # 視覺化
    visualizer.visualize(
        path_points=path_points,
        start_pos=np.array([0.0, 0.0]),
        goal_pos=np.array([4.0, 0.0]),
    )

    print("✓ 直線路徑已視覺化（請查看 Isaac Sim 視窗）")
    return visualizer


def test_curved_path():
    """測試曲線路徑視覺化"""
    print("\n=== 測試 2: 曲線路徑視覺化 ===")

    # 創建視覺化器
    visualizer = AITStarPathVisualizer(
        prim_path="/World/CurvedPath",
        path_color=(0.0, 0.5, 1.0),  # 藍色
    )

    # 創建一條曲線路徑（S 形）
    t = np.linspace(0, 2*np.pi, 20)
    path_points = np.column_stack([
        4 * np.cos(t) + 5,  # x: 偏移到右側
        2 * np.sin(2*t),    # y: S 形
    ])

    # 視覺化
    visualizer.visualize(
        path_points=path_points,
        start_pos=path_points[0],
        goal_pos=path_points[-1],
    )

    print("✓ 曲線路徑已視覺化（請查看 Isaac Sim 視窗）")
    return visualizer


def test_aitstar_planning():
    """測試 AIT* 規劃 + 視覺化"""
    print("\n=== 測試 3: AIT* 規劃與視覺化 ===")

    try:
        # 創建 AIT* 規劃器
        planner = create_aitstar_planner(
            map_size=(20, 20),
            grid_resolution=0.1,
            robot_radius=0.3,
        )

        # 添加一些障礙物
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.mdp.path_planner import EnvironmentMap

        # 添加障礙物到地圖
        planner.map.add_obstacle(
            pos=torch.tensor([5.0, 5.0]),
            size=torch.tensor([1.0, 1.0]),
            shape="cuboid"
        )
        planner.map.add_obstacle(
            pos=torch.tensor([8.0, 3.0]),
            size=torch.tensor([2.0, 0.5]),
            shape="cuboid"
        )

        # 規劃路徑
        start = torch.tensor([2.0, 2.0])
        goal = torch.tensor([15.0, 15.0])

        path = planner.plan_path(start, goal)

        if len(path) > 0:
            # 視覺化路徑
            visualizer = AITStarPathVisualizer(
                prim_path="/World/AITStarPath",
                path_color=(1.0, 0.5, 0.0),  # 橙色
            )
            visualizer.visualize(
                path_points=path,
                start_pos=start.numpy(),
                goal_pos=goal.numpy(),
            )

            print(f"✓ AIT* 規劃成功！路徑包含 {len(path)} 個路徑點")
            print(f"  起點: {start.numpy()}")
            print(f"  終點: {goal.numpy()}")
            return visualizer
        else:
            print("⚠ AIT* 規劃失敗（找不到路徑）")
            return None

    except Exception as e:
        print(f"⚠ AIT* 測試失敗: {e}")
        return None


def test_with_env():
    """在實際環境中測試視覺化"""
    print("\n=== 測試 4: 在環境中視覺化 ===")

    try:
        # 創建環境
        env = ManagerBasedRLEnv(
            task_name="Isaac-Navigation-Charge-SB3-v0",
            num_envs=1,
        )

        # 重置環境
        env.reset()

        # 獲取機器人位置和目標位置
        robot = env.scene["robot"]
        robot_pos = robot.data.root_pos_w[0, :2].cpu().numpy()

        # 從命令中獲取目標位置
        if hasattr(env, 'command_manager'):
            goal_pos = env.command_manager.get_command("goal_command")
            goal_pos = goal_pos[0, :2].cpu().numpy() if goal_pos is not None else np.array([3.0, 0.0])
        else:
            goal_pos = np.array([3.0, 0.0])

        # 創建視覺化器
        visualizer = AITStarPathVisualizer(
            prim_path="/World/EnvPath",
            path_color=(1.0, 0.0, 1.0),  # 紫色
        )

        # 創建一條簡單的直線路徑（模擬 AIT*）
        num_points = 20
        path_points = np.linspace(robot_pos, goal_pos, num_points)

        visualizer.visualize(
            path_points=path_points,
            start_pos=robot_pos,
            goal_pos=goal_pos,
        )

        print(f"✓ 環境路徑已視覺化")
        print(f"  機器人位置: {robot_pos}")
        print(f"  目標位置: {goal_pos}")

        # 保持模擬運行
        print("\n按 Ctrl+C 退出...")
        for i in range(100):
            env.step(env.action_manager.get_current_action())
            simulation_app.render()

        env.close()
        return visualizer

    except Exception as e:
        print(f"⚠ 環境測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_hierarchical_env():
    """測試層級式導航環境（帶 AIT* 視覺化）"""
    print("\n=== 測試 5: 層級式導航環境 ===")

    try:
        import gymnasium as gym

        # 創建層級式導航環境
        task_id = "Isaac-Navigation-Charge-Hierarchical-v0"
        print(f"  創建環境: {task_id}")

        env = gym.make(
            task_id,
            num_envs=1,
            headless=False,
        )

        print(f"  ✓ 環境創建成功")

        # 檢查是否有 AIT* 和視覺化器
        if hasattr(env.unwrapped, '_aitstar_planner'):
            if env.unwrapped._aitstar_planner is not None:
                print(f"  ✓ AIT* 規劃器已初始化")
            else:
                print(f"  ! AIT* 規劃器不可用")

        if hasattr(env.unwrapped, '_path_visualizer'):
            if env.unwrapped._path_visualizer is not None:
                print(f"  ✓ 路徑視覺化器已初始化")
            else:
                print(f"  ! 路徑視覺化器不可用")

        # 重置環境
        obs, info = env.reset()
        print(f"  ✓ 環境已重置")

        # 運行幾步
        print(f"\n  運行 100 步...")
        for step in range(100):
            actions = torch.zeros(1, 2, device=env.unwrapped.device)
            actions[0, 0] = 0.5  # 前進
            actions[0, 1] = 0.0  # 不旋轉

            obs, reward, terminated, truncated, info = env.step(actions)

            if step % 25 == 0:
                print(f"    步驟 {step}: 獎勵 = {reward.mean().item():.3f}")

                # 檢查路徑
                if hasattr(env.unwrapped, '_current_paths'):
                    path = env.unwrapped._current_paths[0]
                    if path is not None and len(path) > 0:
                        print(f"    AIT* 路徑: {len(path)} 個點")

        print(f"  ✓ 測試完成")

        # 保持運行以便查看
        print(f"\n  按 Ctrl+C 退出...")
        while True:
            simulation_app.render()

        env.close()

    except Exception as e:
        print(f"⚠ 層級式環境測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    visualizers = []

    test = args_cli.test if hasattr(args_cli, 'test') else 'all'

    try:
        if test in ["basic", "all"]:
            viz = test_basic_visualization()
            if viz:
                visualizers.append(viz)

        if test in ["curved", "all"]:
            viz = test_curved_path()
            if viz:
                visualizers.append(viz)

        if test in ["aitstar", "all"]:
            viz = test_aitstar_planning()
            if viz:
                visualizers.append(viz)

        if test in ["env", "all"]:
            viz = test_with_env()
            if viz:
                visualizers.append(viz)

        if test in ["hierarchical", "all"]:
            viz = test_hierarchical_env()
            if viz:
                visualizers.append(viz)

        # 如果不是環境測試或層級式測試，保持模擬運行以供查看
        if test not in ["env", "hierarchical"]:
            print("\n" + "="*50)
            print("視覺化完成！請在 Isaac Sim 視窗中查看結果")
            print("按 Ctrl+C 退出...")
            print("="*50)

            # 保持運行
            while True:
                simulation_app.update()

    except KeyboardInterrupt:
        print("\n\n用戶中斷，正在退出...")
    finally:
        print("清理資源...")
        simulation_app.close()


if __name__ == "__main__":
    main()
