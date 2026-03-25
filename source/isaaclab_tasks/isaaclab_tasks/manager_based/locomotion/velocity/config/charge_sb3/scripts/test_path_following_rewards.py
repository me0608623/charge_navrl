#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
路徑跟隨獎勵函數測試腳本

測試 A* + RL 層級式導航的獎勵函數：
- progress_along_path: 沿路徑前進獎勵
- cross_track_error: 偏離路徑懲罰
- path_direction_reward: 方向一致性獎勵
- path_following_reward: 綜合獎勵

使用 generate_straight_path 模擬 A* 輸出（Phase 0 環境，無障礙物）
"""

from __future__ import annotations

import argparse
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection

import gymnasium as gym

# 導入獎勵函數
from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.mdp.rewards import (
    progress_along_path,
    cross_track_error,
    path_direction_reward,
    path_following_reward,
    generate_straight_path,
)


def visualize_test(
    robot_positions: np.ndarray,
    goal_positions: np.ndarray,
    path_points_list: list,
    rewards_dict: dict,
    save_path: str = None,
):
    """可視化測試結果

    Args:
        robot_positions: [num_steps, 2] 機器人軌跡
        goal_positions: [num_steps, 2] 目標位置（每步相同）
        path_points_list: 每一步的路徑點列表
        rewards_dict: 各獎勵組成的歷史記錄
        save_path: 保存圖像的路徑（可選）
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 軌跡圖（左上）
    ax = axes[0, 0]
    ax.set_title("Robot Trajectory & Path Following", fontsize=12)

    # 繪製 A* 路徑（使用最後一條路徑作為參考）
    if path_points_list:
        path = path_points_list[-1].cpu().numpy() if isinstance(path_points_list[-1], torch.Tensor) else path_points_list[-1]
        if len(path.shape) == 2:  # [num_points, 2]
            ax.plot(path[:, 0], path[:, 1], 'g--', linewidth=2, alpha=0.7, label='A* Path')
            # 標記路徑點
            ax.scatter(path[:, 0], path[:, 1], c='green', s=30, alpha=0.5)

    # 繪製機器人軌跡
    ax.plot(robot_positions[:, 0], robot_positions[:, 1], 'b-', linewidth=2, label='Robot Trajectory')
    ax.scatter(robot_positions[0, 0], robot_positions[0, 1], c='blue', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(robot_positions[-1, 0], robot_positions[-1, 1], c='red', s=100, marker='*', label='End', zorder=5)
    ax.scatter(goal_positions[0, 0], goal_positions[0, 1], c='orange', s=150, marker='X', label='Goal', zorder=5)

    # 繪製速度方向箭頭（每隔幾步）
    step = max(1, len(robot_positions) // 10)
    for i in range(0, len(robot_positions) - 1, step):
        if i < len(robot_positions) - 1:
            dx = robot_positions[i + 1, 0] - robot_positions[i, 0]
            dy = robot_positions[i + 1, 1] - robot_positions[i, 1]
            if abs(dx) > 1e-4 or abs(dy) > 1e-4:
                arrow = FancyArrowPatch(
                    (robot_positions[i, 0], robot_positions[i, 1]),
                    (robot_positions[i, 0] + dx * 5, robot_positions[i, 1] + dy * 5),
                    arrowstyle='->', color='cyan', alpha=0.6, mutation_scale=15
                )
                ax.add_patch(arrow)

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # 2. 獎勵隨時間變化（右上）
    ax = axes[0, 1]
    ax.set_title("Reward Components Over Time", fontsize=12)

    steps = np.arange(len(rewards_dict['progress']))
    ax.plot(steps, rewards_dict['progress'], label='Progress Along Path', linewidth=2)
    ax.plot(steps, rewards_dict['cte'], label='Cross-Track Error (penalty)', linewidth=2)
    ax.plot(steps, rewards_dict['direction'], label='Direction Consistency', linewidth=2)
    ax.plot(steps, rewards_dict['total'], label='Total Reward', linewidth=2, linestyle='--')

    ax.set_xlabel("Step")
    ax.set_ylabel("Reward Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 累積獎勵（左下）
    ax = axes[1, 0]
    ax.set_title("Cumulative Rewards", fontsize=12)

    cumulative = {
        'progress': np.cumsum(rewards_dict['progress']),
        'cte': np.cumsum(rewards_dict['cte']),
        'direction': np.cumsum(rewards_dict['direction']),
        'total': np.cumsum(rewards_dict['total']),
    }

    for name, values in cumulative.items():
        ax.plot(steps, values, label=name.capitalize(), linewidth=2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 統計摘要（右下）
    ax = axes[1, 1]
    ax.axis('off')
    ax.set_title("Statistics Summary", fontsize=12)

    # 計算統計量
    final_dist_to_goal = np.linalg.norm(robot_positions[-1] - goal_positions[-1])

    stats_text = f"""
    ╔════════════════════════════════════════╗
    ║  Path Following Rewards Test Summary   ║
    ╠════════════════════════════════════════╣
    ║ Total Steps: {len(rewards_dict['progress']):>26} ║
    ║ Final Distance to Goal: {final_dist_to_goal:>19.2f}m ║
    ╠────────────────────────────────────────╣
    ║ Reward Component  │  Total    │  Mean  ║
    ╠────────────────────────────────────────╣
    ║ Progress          │ {rewards_dict['progress'].sum():>8.2f}  │ {rewards_dict['progress'].mean():>6.3f} ║
    ║ Cross-Track Error │ {rewards_dict['cte'].sum():>8.2f}  │ {rewards_dict['cte'].mean():>6.3f} ║
    ║ Direction         │ {rewards_dict['direction'].sum():>8.2f}  │ {rewards_dict['direction'].mean():>6.3f} ║
    ╠────────────────────────────────────────╣
    ║ TOTAL             │ {rewards_dict['total'].sum():>8.2f}  │ {rewards_dict['total'].mean():>6.3f} ║
    ╚════════════════════════════════════════╝
    """

    ax.text(0.1, 0.5, stats_text, family='monospace', fontsize=10, va='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 圖表已保存至: {save_path}")

    plt.show()


def test_reward_functions(env, num_steps: int = 100):
    """測試路徑跟隨獎勵函數

    Args:
        env: Isaac Lab 環境實例
        num_steps: 測試步數

    Returns:
        dict: 包含軌跡和獎勵歷史的字典
    """
    print("\n" + "=" * 50)
    print("開始測試路徑跟隨獎勵函數")
    print("=" * 50)

    # Reset 環境
    obs, _ = env.reset()
    print(f"✅ 環境已重置，num_envs = {env.unwrapped.num_envs}")

    # 獲取機器人和目標資訊
    robot = env.unwrapped.scene["robot"]
    robot_pos = robot.data.root_pos_w[:, :2]  # [num_envs, 2]

    # 假設環境中有 goal_pos（需要從 observation 中提取或從配置中獲取）
    # 這裡我們使用 generate_straight_path 生成測試路徑

    # 設定起點和終點（使用第一個環境）
    start_pos = robot_pos[0:1]  # [1, 2]
    # 假設目標在 (3, 3) 位置（實際應從環境配置中獲取）
    goal_pos = torch.tensor([[3.0, 3.0]], device=robot_pos.device)

    print(f"📍 起點位置: ({start_pos[0, 0].item():.2f}, {start_pos[0, 1].item():.2f})")
    print(f"🎯 目標位置: ({goal_pos[0, 0].item():.2f}, {goal_pos[0, 1].item():.2f})")

    # 生成直線路徑（模擬 A* 輸出）
    path_points = generate_straight_path(env.unwrapped, start_pos, goal_pos)
    print(f"✅ A* 路徑已生成，shape = {path_points.shape}")

    # 記錄歷史
    history = {
        'robot_positions': [],
        'goal_positions': [],
        'progress': [],
        'cte': [],
        'direction': [],
        'total': [],
    }

    # 擴展 path_points 到所有環境
    num_envs = env.unwrapped.num_envs
    if path_points.shape[0] == 1:
        # 如果只有一條路徑，複製到所有環境
        path_points = path_points.repeat(num_envs, 1, 1)

    print(f"\n開始執行 {num_steps} 步測試...")
    print("-" * 50)

    for step in range(num_steps):
        # 獲取當前機器人位置
        robot_pos = robot.data.root_pos_w[:, :2]

        # 計算各個獎勵組成部分
        progress = progress_along_path(env.unwrapped, path_points)
        cte = cross_track_error(env.unwrapped, path_points, safe_distance=0.5)
        direction = path_direction_reward(env.unwrapped, path_points)

        # 綜合獎勵（使用默認權重）
        total = path_following_reward(env.unwrapped, path_points)

        # 記錄第一個環境的數據
        history['robot_positions'].append(robot_pos[0].cpu().numpy())
        history['goal_positions'].append(goal_pos[0].cpu().numpy())
        history['progress'].append(progress[0].item())
        history['cte'].append(cte[0].item())
        history['direction'].append(direction[0].item())
        history['total'].append(total[0].item())

        # 每 20 步打印一次
        if (step + 1) % 20 == 0:
            print(f"Step {step + 1:3d} | Progress: {progress[0].item():>6.2f} | "
                  f"CTE: {cte[0].item():>6.2f} | Dir: {direction[0].item():>6.2f} | "
                  f"Total: {total[0].item():>6.2f}")

        # 執行隨機動作（僅用於測試）
        actions = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(actions)

        # 檢查是否結束
        if terminated.any() or truncated.any():
            print(f"\n⚠️  環境在第 {step + 1} 步結束")
            break

    print("-" * 50)
    print(f"✅ 測試完成，共執行 {len(history['progress'])} 步\n")

    return history, path_points[0]


def main():
    parser = argparse.ArgumentParser(description="測試路徑跟隨獎勵函數")
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Navigation-Charge-v0",
        choices=[
            "Isaac-Navigation-Charge-v0",
            "Isaac-Navigation-Charge-v1",
            "Isaac-Navigation-Charge-v2",
            "Isaac-Navigation-Charge-v3",
            "Isaac-Navigation-Charge-SB3-v0",
            "Isaac-Navigation-Charge-SB3-v1",
            "Isaac-Navigation-Charge-SB3-v2",
            "Isaac-Navigation-Charge-SB3-v3",
        ],
        help="環境 ID（默認：Isaac-Navigation-Charge-v0，無障礙物）"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="測試步數（默認：100）"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="並行環境數量（默認：1）"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="不顯示可視化圖表"
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="保存圖表到指定路徑"
    )

    args = parser.parse_args()

    # 創建環境
    print(f"\n創建環境: {args.task}")
    print(f"並行環境數量: {args.num_envs}")

    env = gym.make(
        args.task,
        num_envs=args.num_envs,
        headless=False,  # 顯示可視化
    )

    try:
        # 運行測試
        history, path_points = test_reward_functions(env, num_steps=args.num_steps)

        # 轉換為 numpy 數組
        robot_positions = np.array(history['robot_positions'])
        goal_positions = np.array(history['goal_positions'])
        rewards_dict = {
            'progress': np.array(history['progress']),
            'cte': np.array(history['cte']),
            'direction': np.array(history['direction']),
            'total': np.array(history['total']),
        }

        # 可視化結果
        if not args.no_plot:
            print("生成可視化圖表...")
            visualize_test(
                robot_positions,
                goal_positions,
                [path_points],
                rewards_dict,
                save_path=args.save_plot,
            )

    finally:
        env.close()
        print("\n✅ 測試完成，環境已關閉")


if __name__ == "__main__":
    main()
