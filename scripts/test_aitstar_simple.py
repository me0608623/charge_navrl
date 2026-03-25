#!/usr/bin/env python3
"""简单的 AIT* 调试测试"""

import argparse
from isaaclab.app import AppLauncher

# 添加 argparse 参数
parser = argparse.ArgumentParser(description="AIT* 调试测试")
parser.add_argument("--num_envs", type=int, default=1, help="环境数量")
AppLauncher.add_app_launcher_args(parser)
args_cli, unknown_args = parser.parse_known_args()

# 启动 omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main():
    import sys
    import gymnasium as gym
    import isaaclab_tasks
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    print("=" * 60, flush=True)
    print("AIT* 调试测试", flush=True)
    print("=" * 60, flush=True)

    # 创建环境
    print("\n[1/3] 创建环境...", flush=True)
    task_name = "Isaac-Navigation-Charge-Phase0"
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    print(f"  使用配置: num_envs={env_cfg.scene.num_envs}", flush=True)

    env = gym.make(task_name, cfg=env_cfg)
    print(f"✓ 环境创建成功", flush=True)

    # 重置环境（这里会触发 AIT* 规划）
    print("\n[2/3] 重置环境（触发 AIT* 规划）...", flush=True)
    obs, info = env.reset()
    print(f"✓ 环境已重置", flush=True)
    print(f"  观测空间形状: {obs.shape}", flush=True)

    # 运行一步
    print("\n[3/3] 运行一步...", flush=True)
    actions = [[0.0, 0.0]] * args_cli.num_envs
    obs, reward, terminated, truncated, info = env.step(actions)
    print(f"✓ 步骤完成", flush=True)
    print(f"  奖励: {reward}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("测试完成!", flush=True)
    print("=" * 60, flush=True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
