# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ray Caster 實時監控腳本 - 獨立運行，無需修改訓練腳本

此腳本會創建一個獨立的環境實例來監控 ray_caster 感測器的狀態。
它不會干擾正在運行的訓練，只是創建一個相同的環境來檢查感測器配置和狀態。

使用方法：
    # 基本使用（創建環境並檢查 ray_caster）
    python scripts/tools/monitor_ray_caster_live.py --task Isaac-Navigation-Carter-v0 --num_envs 1
    
    # 持續監控模式（每 2 秒檢查一次）
    python scripts/tools/monitor_ray_caster_live.py --task Isaac-Navigation-Carter-v0 --num_envs 1 --watch --interval 2
"""

import argparse
import time
from pathlib import Path

import torch
import gymnasium as gym

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.sensors.ray_caster import RayCasterMonitor


def check_ray_caster_status(env, sensor_name: str = "lidar") -> dict:
    """檢查 ray_caster 感測器的狀態
    
    Args:
        env: 環境實例
        sensor_name: 感測器名稱
        
    Returns:
        包含狀態資訊的字典
    """
    status = {
        "sensor_exists": False,
        "sensor_type": None,
        "num_sensors": 0,
        "num_rays": 0,
        "total_rays": 0,
        "has_valid_data": False,
        "valid_hit_ratio": 0.0,
        "mean_distance": 0.0,
    }
    
    try:
        unwrapped = env.unwrapped
        if hasattr(unwrapped, "scene") and sensor_name in unwrapped.scene.sensors:
            sensor = unwrapped.scene[sensor_name]
            status["sensor_exists"] = True
            status["sensor_type"] = type(sensor).__name__
            status["num_sensors"] = sensor.num_instances
            status["num_rays"] = sensor.num_rays
            status["total_rays"] = sensor.num_rays * sensor.num_instances
            
            # 獲取感測器數據
            data = sensor.data
            if data.ray_hits_w is not None:
                ray_hits_w = data.ray_hits_w
                sensor_pos_w = data.pos_w
                
                # 計算有效碰撞
                is_valid = ~torch.isinf(ray_hits_w).any(dim=-1)
                num_valid_hits = is_valid.sum().item()
                status["valid_hit_ratio"] = num_valid_hits / status["total_rays"] if status["total_rays"] > 0 else 0.0
                
                # 計算距離
                if num_valid_hits > 0:
                    sensor_pos_expanded = sensor_pos_w.unsqueeze(1).expand_as(ray_hits_w)
                    distances = torch.norm(ray_hits_w - sensor_pos_expanded, dim=-1)
                    valid_distances = distances[is_valid]
                    status["mean_distance"] = valid_distances.mean().item()
                    status["has_valid_data"] = True
        else:
            available_sensors = list(unwrapped.scene.sensors.keys()) if hasattr(unwrapped, "scene") else []
            status["available_sensors"] = available_sensors
    except Exception as e:
        status["error"] = str(e)
    
    return status


def print_status(status: dict, sensor_name: str):
    """打印狀態資訊
    
    Args:
        status: 狀態字典
        sensor_name: 感測器名稱
    """
    print("\n" + "="*70)
    print(f"Ray Caster 監控狀態 - 感測器: {sensor_name}")
    print("="*70)
    
    if not status.get("sensor_exists", False):
        print(f"❌ 錯誤: 無法找到感測器 '{sensor_name}'")
        if "available_sensors" in status:
            print(f"   可用的感測器: {status['available_sensors']}")
        if "error" in status:
            print(f"   錯誤訊息: {status['error']}")
        print("="*70 + "\n")
        return
    
    print(f"✅ 感測器狀態: 正常")
    print(f"   感測器類型: {status['sensor_type']}")
    print(f"   感測器數量: {status['num_sensors']}")
    print(f"   每感測器射線數: {status['num_rays']}")
    print(f"   總射線數: {status['total_rays']}")
    
    if status.get("has_valid_data", False):
        print(f"\n📊 數據統計:")
        print(f"   有效碰撞比例: {status['valid_hit_ratio']*100:.2f}%")
        print(f"   平均距離: {status['mean_distance']:.3f} m")
        
        # 判斷是否正常運行
        ratio = status['valid_hit_ratio'] * 100
        if ratio < 1.0:
            print(f"\n   ⚠️  警告: 有效碰撞比例過低 ({ratio:.2f}%)，ray_caster 可能未正常工作！")
        elif ratio > 90:
            print(f"\n   ✅ 正常: 有效碰撞比例正常 ({ratio:.2f}%)")
        else:
            print(f"\n   ⚠️  注意: 有效碰撞比例中等 ({ratio:.2f}%)")
    else:
        print(f"\n⚠️  警告: 尚未有有效數據，請等待環境運行幾步")
    
    print("="*70 + "\n")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="實時監控 Ray Caster 感測器狀態")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="任務名稱（例如：Isaac-Navigation-Carter-v0）",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="環境數量（默認：1）",
    )
    parser.add_argument(
        "--sensor_name",
        type=str,
        default="lidar",
        help="感測器名稱（默認：lidar）",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="持續監控模式",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="監控更新間隔（秒），默認 2 秒",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="設備（默認：cuda:0）",
    )
    
    args = parser.parse_args()
    
    print(f"[INFO] 創建環境: {args.task}")
    print(f"[INFO] 環境數量: {args.num_envs}")
    print(f"[INFO] 監控感測器: {args.sensor_name}")
    
    # 創建環境配置
    try:
        env_cfg = parse_env_cfg(
            args.task,
            device=args.device,
            num_envs=args.num_envs,
        )
        env = gym.make(args.task, cfg=env_cfg)
        print("[INFO] 環境創建成功\n")
    except Exception as e:
        print(f"[ERROR] 無法創建環境: {e}")
        return
    
    # 重置環境
    try:
        obs, _ = env.reset()
        print("[INFO] 環境重置成功\n")
    except Exception as e:
        print(f"[ERROR] 環境重置失敗: {e}")
        env.close()
        return
    
    # 執行幾步以獲取數據
    print("[INFO] 執行環境步驟以獲取感測器數據...")
    for _ in range(10):
        action = torch.zeros(args.num_envs, env.action_space.shape[0], device=args.device)
        obs, reward, terminated, truncated, info = env.step(action)
    print("[INFO] 數據收集完成\n")
    
    if args.watch:
        print(f"[INFO] 持續監控模式（每 {args.interval} 秒更新）")
        print("[INFO] 按 Ctrl+C 停止監控\n")
        
        try:
            while True:
                # 執行一步以更新數據
                action = torch.zeros(args.num_envs, env.action_space.shape[0], device=args.device)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # 檢查狀態
                status = check_ray_caster_status(env, args.sensor_name)
                print_status(status, args.sensor_name)
                
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[INFO] 監控已停止")
    else:
        # 單次檢查
        status = check_ray_caster_status(env, args.sensor_name)
        print_status(status, args.sensor_name)
    
    # 關閉環境
    env.close()
    print("[INFO] 監控完成")


if __name__ == "__main__":
    main()

