#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 0 測試腳本 - 驗證基礎運動學環境

用法:
    ./isaaclab.sh -p scripts/reinforcement_learning/sb3/test_phase0.py
"""

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# 添加 argparse 參數
parser = argparse.ArgumentParser(description="測試 Phase 0 環境")
parser.add_argument("--test_steps", type=int, default=100, help="測試步數")
parser.add_argument("--num_envs", type=int, default=None, help="環境數量（可選，會覆蓋配置）")
AppLauncher.add_app_launcher_args(parser)
args_cli, unknown_args = parser.parse_known_args()

# 啟動 omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main():
    import sys
    import gymnasium as gym
    import isaaclab_tasks
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    sys.stdout.flush()

    print("=" * 60, flush=True)
    print("Phase 0: 基礎運動學環境測試", flush=True)
    print("=" * 60, flush=True)

    # 創建環境
    print("\n[1/4] 創建環境...", flush=True)

    # 從 registry 載入配置
    task_name = "Isaac-Navigation-Charge-Phase0"
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")

    # 如果提供了 num_envs，覆蓋配置
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    print(f"  使用配置: num_envs={env_cfg.scene.num_envs}", flush=True)

    # 創建環境（需要傳入 cfg）
    env = gym.make(task_name, cfg=env_cfg)
    print(f"✓ 環境創建成功: {env.unwrapped.num_envs} 個並行環境", flush=True)

    # 重置環境
    print("\n[2/4] 重置環境...", flush=True)
    obs, info = env.reset()
    print(f"✓ 環境已重置", flush=True)
    print(f"  觀測空間形狀: {obs.shape}", flush=True)
    print(f"  動作空間形狀: {env.action_space.shape}", flush=True)

    # 檢查觀測內容
    print("\n[3/4] 檢查觀測內容...", flush=True)
    print(f"  觀測範圍 (min): {obs.min(axis=0)}", flush=True)
    print(f"  觀測範圍 (max): {obs.max(axis=0)}", flush=True)
    print(f"  觀測是否有 NaN: {np.isnan(obs).any()}", flush=True)
    print(f"  觀測是否有 Inf: {np.isinf(obs).any()}", flush=True)

    # 解析觀測（根據 Phase 0 配置）
    # 觀測順序：goal_pos(2) + goal_dist(1) + vel_xy(2) + angular_vel(1) + time(1) + alive(1) + actions(2) = 10
    obs_dict = {
        "goal_position": obs[0, 0:2],
        "goal_distance": obs[0, 2:3],
        "base_velocity_xy": obs[0, 3:5],
        "angular_velocity_z": obs[0, 5:6],
        "time_remaining": obs[0, 6:7],
        "alive_flag": obs[0, 7:8],
        "last_action": obs[0, 8:10],
    }
    print(f"\n  觀測分解:", flush=True)
    for name, value in obs_dict.items():
        print(f"    {name}: {value}", flush=True)

    # 運行幾步
    print(f"\n[4/4] 運行 {args_cli.test_steps} 步...", flush=True)
    total_reward = 0.0

    num_envs = env.unwrapped.num_envs
    for step in range(args_cli.test_steps):
        # 隨機動作（實際訓練時由 PPO 提供）
        actions = np.random.uniform(-0.5, 0.5, (num_envs, 2))

        # 執行一步
        obs, reward, terminated, truncated, info = env.step(actions)

        total_reward += reward.mean()

        if step % 25 == 0:
            print(f"  步驟 {step}: 獎勵 = {reward.mean():.3f}, "
                  f"目標距離 = {obs[0, 2]:.2f}, "
                  f"速度 = ({obs[0, 3]:.2f}, {obs[0, 4]:.2f})", flush=True)

    print(f"\n✓ 測試完成!", flush=True)
    print(f"  平均獎勵: {total_reward / args_cli.test_steps:.3f}", flush=True)
    if hasattr(terminated, 'sum'):
        print(f"  最終終止數: {terminated.sum()}", flush=True)
    if hasattr(truncated, 'sum'):
        print(f"  最終截斷數: {truncated.sum()}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("Phase 0 測試摘要", flush=True)
    print("=" * 60, flush=True)
    print("觀測維度: 10 維（無 LiDAR）", flush=True)
    print("  - goal_position_in_robot_frame: 2 維", flush=True)
    print("  - goal_distance: 1 維", flush=True)
    print("  - base_velocity_xy: 2 維", flush=True)
    print("  - base_angular_velocity_z: 1 維", flush=True)
    print("  - time_remaining_ratio: 1 維", flush=True)
    print("  - alive_flag: 1 維", flush=True)
    print("  - last_action: 2 維", flush=True)
    print(flush=True)
    print("訓練命令:", flush=True)
    print("  ./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \\", flush=True)
    print("      --task Isaac-Navigation-Charge-Phase0 \\", flush=True)
    print("      --num_envs 256 \\", flush=True)
    print("      --headless \\", flush=True)
    print("      --agent sb3_cfg_entry_point", flush=True)
    print("=" * 60, flush=True)

    # 關閉環境
    env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用戶中斷", flush=True)
    except Exception as e:
        print(f"\n錯誤: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
