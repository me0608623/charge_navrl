#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""測試層級式導航環境 (Test Hierarchical Navigation Environment)

展示 AIT* 全局路徑規劃和視覺化。

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/test_hierarchical_env.py

"""

import argparse

from isaaclab.app import AppLauncher

# 添加 argparse 參數
parser = argparse.ArgumentParser(description="測試層級式導航環境")
parser.add_argument("--num_envs", type=int, default=1, help="環境數量")
# 添加 AppLauncher 參數
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 啟動 omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401


def main():
    print("=" * 60)
    print("層級式導航環境測試")
    print("=" * 60)

    # 使用層級式環境
    task_id = "Isaac-Navigation-Charge-Hierarchical-v0"

    print(f"\n創建環境: {task_id}")
    print(f"環境數量: {args_cli.num_envs}")

    try:
        env = gym.make(task_id, num_envs=args_cli.num_envs)
    except Exception as e:
        print(f"無法創建層級式環境: {e}")
        print(f"嘗試使用標準 SB3 環境...")
        task_id = "Isaac-Navigation-Charge-SB3-v0"
        env = gym.make(task_id, num_envs=args_cli.num_envs)

    # 檢查環境屬性
    print(f"\n環境屬性檢查:")
    print(f"  環境類型: {type(env.unwrapped).__name__}")
    print(f"  有 AIT* 規劃器: {hasattr(env.unwrapped, '_aitstar_planner')}")
    print(f"  有路徑視覺化器: {hasattr(env.unwrapped, '_path_visualizer')}")
    print(f"  有當前路徑: {hasattr(env.unwrapped, '_current_paths')}")

    if hasattr(env.unwrapped, '_aitstar_planner'):
        planner = env.unwrapped._aitstar_planner
        if planner is not None:
            print(f"  ✓ AIT* 規劃器已初始化")
        else:
            print(f"  ! AIT* 規劃器為 None")

    if hasattr(env.unwrapped, '_path_visualizer'):
        visualizer = env.unwrapped._path_visualizer
        if visualizer is not None:
            print(f"  ✓ 路徑視覺化器已初始化")
        else:
            print(f"  ! 路徑視覺化器為 None")

    # 重置環境
    print(f"\n重置環境...")
    obs, info = env.reset()

    # 獲取機器人和目標位置
    robot = env.unwrapped.scene["robot"]
    robot_pos = robot.data.root_pos_w[0, :2].cpu().numpy()
    print(f"  機器人位置: {robot_pos}")

    if hasattr(env.unwrapped, 'command_manager'):
        goal_pos = env.unwrapped.command_manager.get_command("goal_command")
        goal_pos = goal_pos[0, :2].cpu().numpy()
        print(f"  目標位置: {goal_pos}")
    else:
        goal_pos = None
        print(f"  ! 無 command_manager")

    # 檢查路徑
    if hasattr(env.unwrapped, '_current_paths'):
        path = env.unwrapped._current_paths[0]
        if path is not None:
            print(f"  ✓ 當前路徑: {len(path)} 個點")
        else:
            print(f"  ! 當前路徑為 None")

    # 運行測試
    print(f"\n運行 100 步...")
    for step in range(100):
        actions = torch.zeros(args_cli.num_envs, 2, device=env.unwrapped.device)
        actions[:, 0] = 0.5  # 前進
        actions[:, 1] = 0.0  # 不旋轉

        obs, reward, terminated, truncated, info = env.step(actions)

        if step % 25 == 0:
            print(f"  步驟 {step}: 獎勵 = {reward.mean().item():.3f}")

            # 檢查路徑
            if hasattr(env.unwrapped, '_current_paths'):
                path = env.unwrapped._current_paths[0]
                if path is not None and len(path) > 0:
                    print(f"    AIT* 路徑: {len(path)} 個點")

    print(f"\n測試完成！")
    print(f"\n提示:")
    print(f"  - 綠色小球 = AIT* 路徑點")
    print(f"  - 綠色線條 = 路徑連接線")
    print(f"  - 綠色大球 = 起點")
    print(f"  - 紅色大球 = 終點")

    # 保持運行以供查看
    if not args_cli.headless:
        print(f"\n按 Ctrl+C 退出...")
        try:
            while True:
                simulation_app.update()
        except KeyboardInterrupt:
            pass

    env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用戶中斷")
    except Exception as e:
        print(f"錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
