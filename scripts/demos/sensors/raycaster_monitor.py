# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
示範腳本：Ray Caster Monitor 的使用

此腳本展示如何在訓練或測試環境中使用 RayCasterMonitor 來監測和記錄
Ray Caster 感測器的數據。

使用方法：
    ./isaaclab.sh -p scripts/demos/sensors/raycaster_monitor.py --num_envs 4 --num_steps 500

此腳本會：
1. 創建一個包含 Ray Caster 感測器的環境（使用 Carter 導航環境作為範例）
2. 運行指定數量的步驟
3. 使用 RayCasterMonitor 記錄感測器數據
4. 將統計資訊保存到日誌文件
"""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Example on using the Ray Caster Monitor.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
parser.add_argument("--num_steps", type=int, default=500, help="Number of simulation steps to run.")
parser.add_argument("--log_dir", type=str, default="logs/ray_caster_monitor_demo", help="Directory to save logs.")
parser.add_argument("--sensor_name", type=str, default="lidar", help="Name of the ray caster sensor in the scene.")
parser.add_argument("--log_interval", type=int, default=50, help="Interval (in steps) for logging data.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import gymnasium as gym

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.sensors.ray_caster import RayCasterMonitor


def main():
    """Main function to demonstrate Ray Caster Monitor usage."""
    # Try to create Carter environment (or any environment with ray caster sensor)
    try:
        env_cfg = parse_env_cfg(
            "Isaac-Navigation-Carter-v0",
            device=args_cli.device,
            num_envs=args_cli.num_envs,
        )
        env = gym.make("Isaac-Navigation-Carter-v0", cfg=env_cfg)
        print("[INFO] Successfully created Carter navigation environment.")
    except Exception as e:
        print(f"[ERROR] Failed to create Carter environment: {e}")
        print("[INFO] Please ensure the Carter environment is properly registered.")
        print("[INFO] You can modify this script to use any environment with a ray caster sensor.")
        simulation_app.close()
        return

    # Initialize Ray Caster Monitor
    print(f"\n{'='*70}")
    print("Ray Caster Monitor Demo")
    print(f"{'='*70}")
    print(f"Number of environments: {args_cli.num_envs}")
    print(f"Number of steps: {args_cli.num_steps}")
    print(f"Sensor name: {args_cli.sensor_name}")
    print(f"Log directory: {args_cli.log_dir}")
    print(f"Log interval: {args_cli.log_interval} steps")
    print(f"{'='*70}\n")

    monitor = RayCasterMonitor(
        log_dir=args_cli.log_dir,
        sensor_name=args_cli.sensor_name,
        log_interval=args_cli.log_interval,
        save_to_file=True,
        save_to_tensorboard=False,
    )

    # Check if sensor exists
    try:
        sensor = env.unwrapped.scene[args_cli.sensor_name]
        print(f"[INFO] Found sensor: {args_cli.sensor_name}")
        print(f"       Sensor type: {type(sensor).__name__}")
        print(f"       Number of sensors: {sensor.num_instances}")
        print(f"       Rays per sensor: {sensor.num_rays}")
        print(f"       Total rays: {sensor.num_rays * sensor.num_instances}\n")
    except KeyError:
        print(f"[ERROR] Sensor '{args_cli.sensor_name}' not found!")
        print(f"        Available sensors: {list(env.unwrapped.scene.sensors.keys())}")
        env.close()
        simulation_app.close()
        return

    # Reset environment
    obs, _ = env.reset()
    print("[INFO] Environment reset successfully.\n")

    # Run simulation loop
    print(f"[INFO] Starting simulation loop...")
    print(f"       Logging data every {args_cli.log_interval} steps.\n")

    episode_count = 0

    for step in range(args_cli.num_steps):
        # Generate random actions for demonstration
        # Convert numpy array to torch tensor and move to correct device
        action_np = env.action_space.sample()
        action = torch.from_numpy(action_np).to(device=env.unwrapped.device)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Log Ray Caster data
        if step % monitor.log_interval == 0:
            stats = monitor.log_step(env.unwrapped, step)
            if stats:  # Ensure stats is not empty
                monitor.print_current_stats(stats)

        # Handle episode resets
        if terminated.any() or truncated.any():
            episode_count += 1
            monitor.log_episode_reset(episode_count)
            # Note: Environment handles reset automatically, we just log episode count

    # Save summary
    monitor.save_summary(final_step=args_cli.num_steps)

    print(f"\n{'='*70}")
    print("Demo completed successfully!")
    print(f"{'='*70}")
    print(f"Log files saved to:")
    print(f"  - Detailed log: {args_cli.log_dir}/ray_caster_monitor.log")
    print(f"  - CSV data: {args_cli.log_dir}/ray_caster_stats.csv")
    print(f"  - Summary: {args_cli.log_dir}/ray_caster_summary.txt")
    print(f"{'='*70}\n")

    # Close environment and app
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

