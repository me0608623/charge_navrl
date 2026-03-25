#!/usr/bin/env python3
"""
調試腳本：檢查多環境初始化時機器人是否正確生成

運行命令：
./isaaclab.sh -p scripts/test_multi_env_debug.py --num_envs 5
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test multi-environment initialization")
parser.add_argument("--num_envs", type=int, default=5, help="Number of environments")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, _ = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
from pxr import Usd, UsdGeom


def main():
    print("\n" + "="*60)
    print(f"多環境初始化調試 (num_envs={args_cli.num_envs})")
    print("="*60)

    # 使用 gym.make 創建環境
    print("\n創建環境...")
    import isaaclab_tasks  # noqa: F401

    # 先導入配置類，然後創建環境
    from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.cfg.charge_env_cfg_phase0 import ChargeNavigationEnvCfgPhase0

    # 創建配置並覆蓋 num_envs
    env_cfg = ChargeNavigationEnvCfgPhase0()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(
        "Isaac-Navigation-Charge-Phase0",
        cfg=env_cfg,
    )
    print(f"✅ 環境創建成功")

    # 獲取 unwrapped 環境以訪問場景
    env_unwrapped = env.unwrapped
    scene = env_unwrapped.scene

    # 打印場景配置
    print("\n" + "="*60)
    print("場景配置")
    print("="*60)
    print(f"num_envs: {scene.cfg.num_envs}")
    print(f"env_spacing: {scene.cfg.env_spacing}")
    print(f"replicate_physics: {scene.cfg.replicate_physics}")
    print(f"clone_in_fabric: {scene.cfg.clone_in_fabric}")
    print(f"env_ns: {scene.env_ns}")
    print(f"env_regex_ns: {scene.env_regex_ns}")
    print(f"env_prim_paths: {len(scene.env_prim_paths)} 個")
    for i, path in enumerate(scene.env_prim_paths):
        print(f"  [{i}] {path}")

    # 打印環境原點
    print("\n" + "="*60)
    print("環境原點 (env_origins)")
    print("="*60)
    origins = scene.env_origins.cpu().numpy()
    print(f"形狀: {origins.shape}")
    for i in range(len(origins)):
        print(f"  env_{i}: ({origins[i, 0]:.2f}, {origins[i, 1]:.2f}, {origins[i, 2]:.2f})")

    # 檢查每個環境中的機器人
    print("\n" + "="*60)
    print("機器人檢查")
    print("="*60)

    if "robot" not in scene.articulations:
        print("❌ 沒有找到 'robot' articulation")
        env.close()
        return

    robot = scene.articulations["robot"]
    print(f"✅ Robot articulation 存在")
    print(f"   num_instances: {robot.num_instances}")

    # 檢查每個環境中的機器人 prims
    stage = scene.stage
    print("\n檢查每個環境中的 Robot prim:")
    for i in range(args_cli.num_envs):
        robot_path = f"{scene.env_prim_paths[i]}/Robot"
        prim = stage.GetPrimAtPath(robot_path)
        if prim.IsValid():
            print(f"  ✅ env_{i}: {robot_path} 存在")
            # 檢查機器人的子 prims
            children = [child.GetPath().pathString for child in prim.GetAllChildren()]
            print(f"     子 prims: {len(children)} 個")
            if children:
                print(f"     例如: {children[0].split('/')[-1]}")
        else:
            print(f"  ❌ env_{i}: {robot_path} 不存在")

    # 檢查每個環境中的牆壁
    print("\n檢查每個環境中的牆壁:")
    walls = ["Wall_North", "Wall_South", "Wall_East", "Wall_West"]
    for i in range(args_cli.num_envs):
        print(f"  env_{i}:")
        for wall in walls:
            wall_path = f"{scene.env_prim_paths[i]}/{wall}"
            prim = stage.GetPrimAtPath(wall_path)
            if prim.IsValid():
                print(f"    ✅ {wall}")
            else:
                print(f"    ❌ {wall}")

    # 使用 Isaac Lab 的 find_matching_prims 檢查
    import isaaclab.sim as sim_utils
    robot_prims = sim_utils.find_matching_prims(scene.env_regex_ns + "/Robot")
    print(f"\n使用 find_matching_prims 找到的 Robot prims: {len(robot_prims)} 個")
    for prim in robot_prims:
        print(f"  - {prim}")

    # 打印所有環境下的 USD prims
    print("\n" + "="*60)
    print("USD Stage 結構 (前 50 層)")
    print("="*60)
    count = 0
    for prim in stage.Traverse():
        print(f"  {prim.GetPath()}")
        count += 1
        if count >= 100:
            print(f"  ... (省略其餘)")
            break

    # 檢查可見性
    print("\n" + "="*60)
    print("機器人可見性檢查")
    print("="*60)
    for i in range(args_cli.num_envs):
        robot_path = f"{scene.env_prim_paths[i]}/Robot"
        prim = stage.GetPrimAtPath(robot_path)
        if prim.IsValid():
            imageable = UsdGeom.Imageable(prim)
            visibility = imageable.ComputeVisibility()
            print(f"  env_{i}: {robot_path} -> visibility={visibility}")

    # 總結
    print("\n" + "="*60)
    print("總結")
    print("="*60)
    print(f"配置的環境數: {scene.cfg.num_envs}")
    print(f"實際的 env_prim_paths: {len(scene.env_prim_paths)}")
    print(f"Robot articulation instances: {robot.num_instances}")
    print(f"找到的 Robot prims: {len(robot_prims)}")

    if robot.num_instances == args_cli.num_envs:
        print("\n✅ 機器人數量正確！問題可能在於可視化或相機角度。")
    else:
        print(f"\n❌ 機器人數量不正確！預期 {args_cli.num_envs} 個，實際 {robot.num_instances} 個。")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
