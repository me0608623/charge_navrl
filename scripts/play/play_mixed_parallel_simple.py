#!/usr/bin/env python3
"""GUI 模式演示腳本 - 展示混合平行環境效果

運行方式:
    ./isaaclab.sh -p scripts/play_mixed_parallel_simple.py

展示內容:
    - 10 個並排環境
    - 2 個空環境 (無障礙物)
    - 5 個靜態障礙物環境
    - 3 個動態障礙物環境
"""

import argparse

from isaaclab.app import AppLauncher

# 創建參數
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# 啟動模擬器
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

# 創建環境
env = gym.make("Isaac-Navigation-Charge-Phase0", num_envs=args_cli.num_envs)

print("=" * 80)
print("混合平行環境 GUI 演示")
print("=" * 80)
print()
print(f"環境數量: {args_cli.num_envs}")
print()
print("混合平行環境特點:")
print("  ~20%: 空環境 (無障礙物)")
print("  ~50%: 靜態障礙物 (5 個)")
print("  ~30%: 動態障礙物 (8 個, 移動中)")
print()
print("按 Ctrl+C 退出")
print("=" * 80)
print()

# 重置環境
obs, info = env.reset()

# 檢查難度分佈
if hasattr(env.unwrapped, "_env_difficulty"):
    difficulties = env.unwrapped._env_difficulty.cpu().numpy()
    print("實際環境分佈:")
    for i, diff in enumerate(difficulties):
        if diff == 0:
            print(f"  環境 {i}: 空環境 (無障礙物)")
        elif diff == 1:
            num_vis = env.unwrapped._env_num_visible_obstacles[i].item()
            print(f"  環境 {i}: 靜態障礙物 ({num_vis} 個)")
        else:
            num_vis = env.unwrapped._env_num_visible_obstacles[i].item()
            print(f"  環境 {i}: 動態障礙物 ({num_vis} 個, 移動中)")
    print()

# 檢查障礙物位置
if hasattr(env.scene, "obstacle_0"):
    obstacle_z = env.scene.obstacle_0.data.root_pos_w[:, 2]
    visible = (obstacle_z >= 0).sum().item()
    hidden = (obstacle_z < 0).sum().item()
    print(f"障礙物 0: {visible} 可見, {hidden} 隱藏")
    print()

# 模擬循環
step_count = 0
try:
    while True:
        # 隨機動作
        actions = env.action_space.sample()
        obs, rewards, terminated, truncated, info = env.step(actions)

        # 每 100 步報告一次
        if step_count % 100 == 0:
            print(f"  步數: {step_count}, 平均獎勵: {rewards.mean():.3f}")

        # 處理終止
        if terminated.any() or truncated.any():
            # 只重置終止的環境
            reset_ids = torch.where(terminated | truncated)[0].tolist()
            if reset_ids:
                obs, info = env.reset(reset_ids)

        step_count += 1

except KeyboardInterrupt:
    print("\n退出模擬")

env.close()
simulation_app.close()
