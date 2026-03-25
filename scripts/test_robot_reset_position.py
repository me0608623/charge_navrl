#!/usr/bin/env python3
"""
診斷機器人重置後的位置問題

運行命令：
./isaaclab.sh -p scripts/test_robot_reset_position.py --num_envs 5
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diagnose robot reset position issue")
parser.add_argument("--num_envs", type=int, default=5, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.cfg.charge_env_cfg_phase0 import ChargeNavigationEnvCfgPhase0


def main():
    print(f"\n創建 {args_cli.num_envs} 個環境並診斷機器人位置...")

    # 創建配置並覆蓋 num_envs
    env_cfg = ChargeNavigationEnvCfgPhase0()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make("Isaac-Navigation-Charge-Phase0", cfg=env_cfg)

    # Reset 環境
    print("\n第一次 reset...")
    obs, info = env.reset()

    scene = env.unwrapped.scene
    robot = scene.articulations["robot"]

    # 打印機器人初始位置
    print("\n" + "="*60)
    print("機器人初始位置（第一次 reset 後）")
    print("="*60)
    positions = robot.data.root_pos_w.cpu().numpy()
    print(f"形狀: {positions.shape}")
    for i in range(args_cli.num_envs):
        x, y, z = positions[i]
        print(f"  env_{i}: X={x:7.3f}, Y={y:7.3f}, Z={z:7.3f}")

    # 打印環境原點
    print("\n環境原點 (env_origins):")
    origins = scene.env_origins.cpu().numpy()
    for i in range(args_cli.num_envs):
        x, y, z = origins[i]
        print(f"  env_{i}: X={x:7.3f}, Y={y:7.3f}, Z={z:7.3f}")

    # 打印 default_root_state
    print("\n" + "="*60)
    print("機器人 default_root_state")
    print("="*60)
    default_state = robot.data.default_root_state.cpu().numpy()
    print(f"形狀: {default_state.shape}")
    for i in range(min(args_cli.num_envs, 3)):  # 只打印前 3 個
        print(f"  env_{i}:")
        print(f"    pos: {default_state[i, 0:3]}")
        print(f"    quat: {default_state[i, 3:7]}")
        print(f"    vel: {default_state[i, 7:13]}")

    # 檢查機器人是否在地面上
    print("\n" + "="*60)
    print("機器人高度檢查")
    print("="*60)
    z_positions = positions[:, 2]
    print(f"Z 位置範圍: [{z_positions.min():.3f}, {z_positions.max():.3f}]")
    print(f"Z 位置平均值: {z_positions.mean():.3f}")

    if z_positions.min() < 0.05:
        print("⚠️  警告：機器人 Z 位置過低（可能在地面以下）！")
    if z_positions.min() < 0.0:
        print("❌ 錯誤：機器人 Z 位置為負數（絕對在地面以下）！")
    if z_positions.max() > 1.0:
        print("⚠️  警告：機器人 Z 位置過高（可能飄浮）！")

    # 模擬一些步數，然後觸發重置
    print("\n" + "="*60)
    print("模擬 50 步後檢查機器人位置...")
    print("="*60)
    for step in range(50):
        actions = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(actions)

        # 檢查是否有環境終止
        done = terminated | truncated
        if done.any():
            print(f"\n步 {step}: 有環境終止")
            for i in range(args_cli.num_envs):
                if done[i]:
                    print(f"  env_{i} 終止")

    # 打印 50 步後的位置
    positions_after = robot.data.root_pos_w.cpu().numpy()
    print(f"\n50 步後的機器人位置:")
    for i in range(args_cli.num_envs):
        x, y, z = positions_after[i]
        print(f"  env_{i}: X={x:7.3f}, Y={y:7.3f}, Z={z:7.3f}")

    # 檢查固定位置存儲
    print("\n" + "="*60)
    print("檢查固定位置存儲 (_fixed_robot_poses)")
    print("="*60)
    if hasattr(env.unwrapped, "_fixed_robot_poses"):
        fixed_poses = env.unwrapped._fixed_robot_poses.cpu().numpy()
        print(f"形狀: {fixed_poses.shape}")
        for i in range(args_cli.num_envs):
            x, y, z = fixed_poses[i, 0:3]
            print(f"  env_{i}: X={x:7.3f}, Y={y:7.3f}, Z={z:7.3f}")

        initialized = env.unwrapped._fixed_robot_initialized.cpu().numpy()
        print(f"\n初始化狀態: {initialized}")
    else:
        print("❌ 沒有 _fixed_robot_poses 屬性！")

    print("\n按 Ctrl+C 退出...")
    try:
        import time
        while True:
            env.step(env.action_space.sample())
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n退出...")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
