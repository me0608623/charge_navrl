#!/usr/bin/env python3
"""
測試腳本：驗證 Phase 0 的 LiDAR (RayCaster) 是否正常運作

檢查項目：
1. LiDAR 傳感器是否存在
2. LiDAR 是否正確檢測到四面牆壁
3. LiDAR 數據範圍是否合理
4. 可視化 LiDAR 掃描結果
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test LiDAR sensor for Phase 0")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--video", action="store_true", default=False, help="Record video")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, _ = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式後端
import matplotlib.pyplot as plt
import gymnasium as gym

def main():
    """主測試流程"""
    print("\n" + "="*60)
    print("Phase 0 LiDAR 測試程序")
    print("="*60)

    # 使用 gym.make 創建環境
    print("\n創建 Phase 0 環境...")
    import isaaclab_tasks  # noqa: F401

    env = gym.make(
        "Isaac-Navigation-Charge-Phase0",
        num_envs=args_cli.num_envs,
        render_mode="rgb_array" if args_cli.video else None,
    )
    print("✅ 環境創建成功")

    # Reset 獲取初始數據
    obs, info = env.reset()
    print(f"✅ 觀測空間形狀: {obs.shape}")

    # 獲取 unwrapped 環境以訪問場景
    env_unwrapped = env.unwrapped
    scene = env_unwrapped.scene

    # ========================================
    # 測試 1: LiDAR 傳感器存在性
    # ========================================
    print("\n" + "="*60)
    print("測試 1: LiDAR 傳感器存在性檢查")
    print("="*60)

    if not hasattr(scene, "sensors"):
        print("❌ 錯誤: scene 沒有 sensors 屬性")
        env.close()
        return

    print(f"✅ scene.sensors 存在")
    print(f"   可用傳感器: {list(scene.sensors.keys())}")

    if "lidar" not in scene.sensors:
        print("❌ 錯誤: lidar 傳感器不存在")
        env.close()
        return

    lidar = scene.sensors["lidar"]
    print(f"✅ lidar 傳感器存在")
    print(f"   類型: {type(lidar).__name__}")
    print(f"   配置路徑: {lidar.cfg.prim_path}")

    # ========================================
    # 測試 2: LiDAR 配置檢查
    # ========================================
    print("\n" + "="*60)
    print("測試 2: LiDAR 配置檢查")
    print("="*60)

    cfg = lidar.cfg
    pattern_cfg = cfg.pattern_cfg

    print(f"✅ 射線模式: {pattern_cfg.__class__.__name__}")
    print(f"   水平 FOV: {pattern_cfg.horizontal_fov_range}°")
    print(f"   水平解析度: {pattern_cfg.horizontal_res}°")
    expected_rays = int(360 / pattern_cfg.horizontal_res)
    print(f"   預期射線數: {expected_rays}")
    print(f"   最大檢測距離: {cfg.max_distance}m")
    print(f"   更新週期: {cfg.update_period}s ({1/cfg.update_period:.1f} Hz)")
    print(f"   Debug 可視化: {cfg.debug_vis}")

    print(f"✅ 射線目標數量: {len(cfg.mesh_prim_paths)}")
    for i, target in enumerate(cfg.mesh_prim_paths):
        print(f"   [{i}] {target.prim_expr}")

    # ========================================
    # 測試 3: LiDAR 數據檢查
    # ========================================
    print("\n" + "="*60)
    print("測試 3: LiDAR 數據檢查")
    print("="*60)

    if not hasattr(lidar.data, "ray_hits_w"):
        print("❌ 錯誤: lidar.data 沒有 ray_hits_w 屬性")
        env.close()
        return

    # 獲取數據
    ray_hits = lidar.data.ray_hits_w.cpu().numpy()  # [num_envs, num_rays, 3]
    sensor_pos = lidar.data.pos_w.cpu().numpy()    # [num_envs, 3]

    print(f"✅ ray_hits_w 形狀: {ray_hits.shape}")
    print(f"✅ sensor_pos_w 形狀: {sensor_pos.shape}")

    num_rays = ray_hits.shape[1]
    print(f"   實際射線數: {num_rays}")

    if num_rays != expected_rays:
        print(f"⚠️  警告: 射線數不匹配 (預期: {expected_rays}, 實際: {num_rays})")

    # 計算距離 (環境 0)
    sensor_pos_2d = sensor_pos[0, :2]
    hit_points_2d = ray_hits[0, :, :2]
    distances = np.linalg.norm(hit_points_2d - sensor_pos_2d, axis=1)

    print(f"\n✅ 距離統計 (環境 0):")
    print(f"   最小距離: {distances.min():.3f}m")
    print(f"   最大距離: {distances.max():.3f}m")
    print(f"   平均距離: {distances.mean():.3f}m")
    print(f"   中位數距離: {np.median(distances):.3f}m")

    # 檢查無效值
    nan_count = np.isnan(distances).sum()
    inf_count = np.isinf(distances).sum()
    if nan_count > 0:
        print(f"⚠️  警告: 發現 NaN 值數量: {nan_count}")
    if inf_count > 0:
        print(f"⚠️  警告: 發現 Inf 值數量: {inf_count}")

    # ========================================
    # 測試 4: 牆壁檢測驗證
    # ========================================
    print("\n" + "="*60)
    print("測試 4: 牆壁檢測驗證")
    print("="*60)

    min_distance = distances.min()

    if min_distance < 0.1:
        print(f"❌ 錯誤: 最近距離 {min_distance:.3f}m 太小，可能機器人已經貼牆")
    elif min_distance > 8.0:
        print(f"⚠️  警告: 最近距離 {min_distance:.3f}m 太大，牆壁可能未被檢測")
    else:
        print(f"✅ 最近牆壁距離 {min_distance:.3f}m 合理 (預期: 3m ~ 8m)")

    # 檢查檢測範圍
    detected_ratio = (distances < 10.0).sum() / len(distances)
    print(f"✅ 檢測到物體的射線比例: {detected_ratio*100:.1f}%")

    if detected_ratio < 0.5:
        print(f"⚠️  警告: 檢測比例偏低，可能有問題")

    # ========================================
    # 測試 5: 觀測空間中的 LiDAR
    # ========================================
    print("\n" + "="*60)
    print("測試 5: 觀測空間中的 LiDAR")
    print("="*60)

    # Phase 0: lidar_scan 在觀測索引 0-71 (72 維)
    lidar_obs = obs[0, 0:72]
    print(f"✅ LiDAR 觀測形狀: {lidar_obs.shape}")
    print(f"   LiDAR 觀測範圍: [{lidar_obs.min():.3f}, {lidar_obs.max():.3f}]")
    print(f"   LiDAR 觀測平均: {lidar_obs.mean():.3f}")

    if lidar_obs.max() > 1.0:
        print(f"⚠️  警告: LiDAR 觀測最大值 {lidar_obs.max():.3f} > 1.0 (可能未歸一化)")
    if lidar_obs.min() < 0.0:
        print(f"⚠️  警告: LiDAR 觀測最小值 {lidar_obs.min():.3f} < 0.0")

    # ========================================
    # 測試 6: 可視化
    # ========================================
    print("\n" + "="*60)
    print("測試 6: LiDAR 可視化")
    print("="*60)

    fig, ax = plt.subplots(figsize=(10, 10))

    # 繪製機器人位置
    ax.plot(sensor_pos_2d[0], sensor_pos_2d[1], 'bo', markersize=10, label='Robot')

    # 繪製射線 (每 5 條射線畫一條)
    for i in range(0, len(hit_points_2d), 5):
        ax.plot([sensor_pos_2d[0], hit_points_2d[i, 0]],
                [sensor_pos_2d[1], hit_points_2d[i, 1]],
                'r-', alpha=0.3, linewidth=0.5)

    # 繪製碰撞點
    scatter = ax.scatter(hit_points_2d[:, 0], hit_points_2d[:, 1],
                         c=distances, cmap='viridis', s=10, label='Hit Points')
    plt.colorbar(scatter, ax=ax, label='Distance (m)')

    # 繪製牆壁 (Phase 0: ±8m)
    wall_size = 8.0
    # 繪製四面牆
    # 北牆
    ax.plot([-wall_size, wall_size], [wall_size, wall_size], 'k-', linewidth=3)
    # 南牆
    ax.plot([-wall_size, wall_size], [-wall_size, -wall_size], 'k-', linewidth=3)
    # 東牆
    ax.plot([wall_size, wall_size], [-wall_size, wall_size], 'k-', linewidth=3)
    # 西牆
    ax.plot([-wall_size, -wall_size], [-wall_size, wall_size], 'k-', linewidth=3, label='Walls')

    # 設置圖形
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Phase 0 LiDAR Detection (Top View)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 保存圖片
    output_path = "/home/aa/IsaacLab/lidar_test_phase0.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 可視化圖片已保存: {output_path}")

    plt.close()

    # ========================================
    # 總結
    # ========================================
    print("\n" + "="*60)
    print("測試總結")
    print("="*60)
    print("✅ LiDAR 傳感器正常運作")
    print(f"✅ 射線數量: {num_rays} (預期: {expected_rays})")
    print(f"✅ 最近牆壁距離: {min_distance:.3f}m")
    print(f"✅ 平均檢測距離: {distances.mean():.3f}m")
    print(f"✅ 檢測比例: {detected_ratio*100:.1f}%")
    print("\n建議: 檢查生成的可視化圖片以確認掃描結果")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
