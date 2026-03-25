#!/usr/bin/env python3
"""
設置相機位置以查看所有環境

運行命令：
./isaaclab.sh -p scripts/test_multi_env_camera.py --num_envs 5
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Set camera to view all environments")
parser.add_argument("--num_envs", type=int, default=5, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.cfg.charge_env_cfg_phase0 import ChargeNavigationEnvCfgPhase0


def main():
    print(f"\n創建 {args_cli.num_envs} 個環境並設置相機...")

    # 創建配置並覆蓋 num_envs
    env_cfg = ChargeNavigationEnvCfgPhase0()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make("Isaac-Navigation-Charge-Phase0", cfg=env_cfg)

    # Reset 環境以觸發場景初始化
    obs, info = env.reset()

    # 獲取場景信息
    scene = env.unwrapped.scene
    env_spacing = scene.cfg.env_spacing
    num_envs = scene.cfg.num_envs

    print(f"env_spacing: {env_spacing}m")
    print(f"num_envs: {num_envs}")

    # 計算相機位置（頂視圖，俯視所有環境）
    # GridCloner 使用網格布局，計算中心點
    n_cols = math.ceil(math.sqrt(num_envs))
    n_rows = math.ceil(num_envs / n_cols)

    # 計算整個場景的範圍
    total_width = (n_cols - 1) * env_spacing
    total_height = (n_rows - 1) * env_spacing

    # 相機位置（俯視，Z 軸向上）
    cam_x = total_width / 2
    cam_y = total_height / 2
    cam_z = max(total_width, total_height) * 1.5 + 20  # 足夠高以看到所有環境

    print(f"\n環境布局: {n_rows} 行 x {n_cols} 列")
    print(f"場景範圍: {total_width:.1f}m x {total_height:.1f}m")
    print(f"相機位置: ({cam_x:.1f}, {cam_y:.1f}, {cam_z:.1f})")

    # 設置相機
    import omni.isaac.core.utils.viewports as viewports
    from omni.isaac.core.utils.stage import get_current_stage

    # 獲取當前視口
    viewport = viewports.get_current_viewport()

    # 設置相機位置和目標
    import omni.isaac.core.utils.prims as prims_utils
    import omni

    # 使用 Camera API 設置相機
    from pxr import Gf

    # 方法 1: 使用 set_camera_position
    viewports.set_camera_position(eye=(cam_x, cam_y, cam_z), target=(cam_x, cam_y, 0))

    print("\n✅ 相機已設置為頂視圖")
    print("提示：你可以使用鼠標/鍵盤控制相機：")
    print("  - 左鍵拖動：旋轉視圖")
    print("  - 右鍵拖動：平移視圖")
    print("  - 滾輪：縮放")
    print("  - ALT + 左鍵：旋轉視點")
    print("  - ALT + 中鍵：平移視點")
    print("  - ALT + 右鍵：縮放視點")

    # 保持模擬運行以便用戶可以查看
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
