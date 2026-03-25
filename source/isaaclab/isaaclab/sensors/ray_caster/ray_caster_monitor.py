# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ray Caster 監測類別 - 用於記錄訓練過程中的 Ray Caster 感測器數據

此類別參考 ray_caster.py 的輸出格式，記錄以下資訊：
- 感測器基本資訊（位置、姿態、更新週期等）
- 射線碰撞統計（有效碰撞數量、平均距離、最小/最大距離等）
- 感測器狀態資訊（網格數量、感測器數量、射線數量等）
"""

import os
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class RayCasterMonitor:
    """監測和記錄 Ray Caster 感測器數據的類別
    
    此類別用於在訓練過程中監測 RayCaster 或 MultiMeshRayCaster 感測器的狀態和數據，
    並將統計資訊記錄到文件或 TensorBoard。
    
    參考 ray_caster.py 的 __str__ 方法輸出格式：
    - view type
    - update period (s)
    - number of meshes
    - number of sensors
    - number of rays/sensor
    - total number of rays
    """

    def __init__(
        self,
        log_dir: str,
        sensor_name: str = "lidar",
        log_interval: int = 100,
        save_to_file: bool = True,
        save_to_tensorboard: bool = False,
        tensorboard_writer: Optional[object] = None,
    ):
        """初始化 Ray Caster 監測器
        
        Args:
            log_dir: 日誌保存目錄
            sensor_name: 場景中感測器的名稱（默認為 "lidar"）
            log_interval: 記錄間隔（每 N 步記錄一次）
            save_to_file: 是否保存到文件
            save_to_tensorboard: 是否保存到 TensorBoard
            tensorboard_writer: TensorBoard SummaryWriter 實例（如果使用 TensorBoard）
        """
        self.log_dir = Path(log_dir)
        self.sensor_name = sensor_name
        self.log_interval = log_interval
        self.save_to_file = save_to_file
        self.save_to_tensorboard = save_to_tensorboard
        self.tensorboard_writer = tensorboard_writer

        # 創建日誌目錄
        if self.save_to_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.log_dir / "ray_caster_monitor.log"
            self.stats_file = self.log_dir / "ray_caster_stats.csv"

        # 統計數據歷史
        self.stats_history = []
        self.step_count = 0
        self.episode_count = 0

        # 初始化 CSV 文件標題（如果保存到文件）
        if self.save_to_file and not self.stats_file.exists():
            with open(self.stats_file, "w") as f:
                f.write(
                    "step,episode,timestamp,"
                    "num_sensors,num_rays_per_sensor,total_rays,"
                    "num_valid_hits,valid_hit_ratio,"
                    "mean_distance,min_distance,max_distance,std_distance,"
                    "sensor_pos_mean_x,sensor_pos_mean_y,sensor_pos_mean_z,"
                    "update_period\n"
                )

    def log_step(self, env: "ManagerBasedRLEnv", step: int) -> dict:
        """記錄當前步驟的 Ray Caster 數據
        
        Args:
            env: ManagerBasedRLEnv 環境實例
            step: 當前訓練步驟
            
        Returns:
            包含統計資訊的字典
        """
        self.step_count = step

        # 獲取感測器
        try:
            sensor = env.scene[self.sensor_name]
        except KeyError:
            print(f"[WARNING] 無法找到感測器 '{self.sensor_name}'，跳過記錄")
            return {}

        # 獲取感測器數據
        data = sensor.data

        # 計算統計資訊
        stats = self._compute_statistics(sensor, data, step)

        # 記錄到文件
        if self.save_to_file and step % self.log_interval == 0:
            self._log_to_file(sensor, stats, step)

        # 記錄到 TensorBoard
        if self.save_to_tensorboard and self.tensorboard_writer is not None:
            self._log_to_tensorboard(stats, step)

        # 保存統計歷史
        self.stats_history.append(stats)

        return stats

    def _compute_statistics(
        self, sensor, data, step: int
    ) -> dict:
        """計算感測器統計資訊
        
        參考 ray_caster.py 的輸出格式計算統計數據
        
        Args:
            sensor: Ray Caster 感測器實例
            data: RayCasterData 或 MultiMeshRayCasterData 實例
            step: 當前步驟
            
        Returns:
            包含統計資訊的字典
        """
        # 基本感測器資訊（參考 ray_caster.py 的 __str__ 輸出）
        num_sensors = sensor.num_instances
        num_rays_per_sensor = sensor.num_rays
        total_rays = num_rays_per_sensor * num_sensors
        update_period = sensor.cfg.update_period

        # 獲取射線碰撞點
        ray_hits_w = data.ray_hits_w  # Shape: (N, B, 3)
        sensor_pos_w = data.pos_w  # Shape: (N, 3)

        # 計算有效碰撞（排除 inf 值）
        # 檢查哪些射線有有效碰撞
        is_valid = ~torch.isinf(ray_hits_w).any(dim=-1)  # Shape: (N, B)
        num_valid_hits = is_valid.sum().item()
        valid_hit_ratio = num_valid_hits / total_rays if total_rays > 0 else 0.0

        # 計算距離統計
        # 將射線碰撞點和感測器位置擴展到相同形狀以便計算距離
        sensor_pos_expanded = sensor_pos_w.unsqueeze(1).expand_as(ray_hits_w)  # (N, B, 3)
        
        # 計算每個射線的距離
        distances = torch.norm(ray_hits_w - sensor_pos_expanded, dim=-1)  # Shape: (N, B)
        
        # 只考慮有效碰撞的距離
        valid_distances = distances[is_valid]

        # 計算統計值
        if len(valid_distances) > 0:
            mean_distance = valid_distances.mean().item()
            min_distance = valid_distances.min().item()
            max_distance = valid_distances.max().item()
            std_distance = valid_distances.std().item()
        else:
            mean_distance = 0.0
            min_distance = float("inf")
            max_distance = 0.0
            std_distance = 0.0

        # 感測器位置統計（所有環境的平均位置）
        sensor_pos_mean = sensor_pos_w.mean(dim=0)  # Shape: (3,)

        # 構建統計字典
        stats = {
            "step": step,
            "episode": self.episode_count,
            "timestamp": time.time(),
            "num_sensors": num_sensors,
            "num_rays_per_sensor": num_rays_per_sensor,
            "total_rays": total_rays,
            "num_valid_hits": num_valid_hits,
            "valid_hit_ratio": valid_hit_ratio,
            "mean_distance": mean_distance,
            "min_distance": min_distance if min_distance != float("inf") else 0.0,
            "max_distance": max_distance,
            "std_distance": std_distance,
            "sensor_pos_mean_x": sensor_pos_mean[0].item(),
            "sensor_pos_mean_y": sensor_pos_mean[1].item(),
            "sensor_pos_mean_z": sensor_pos_mean[2].item(),
            "update_period": update_period,
        }

        return stats

    def _log_to_file(self, sensor, stats: dict, step: int):
        """將統計資訊記錄到文件
        
        參考 ray_caster.py 的 __str__ 輸出格式
        
        Args:
            sensor: Ray Caster 感測器實例
            stats: 統計資訊字典
            step: 當前步驟
        """
        # 記錄詳細資訊到日誌文件（參考 ray_caster.py 的輸出格式）
        with open(self.log_file, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Step {step} - Ray Caster Monitor\n")
            f.write(f"{'='*60}\n")
            
            # 感測器基本資訊（參考 ray_caster.py 的 __str__ 輸出）
            f.write(f"Ray-caster @ '{sensor.cfg.prim_path}':\n")
            f.write(f"\tview type            : {sensor._view.__class__}\n")
            f.write(f"\tupdate period (s)    : {sensor.cfg.update_period}\n")
            
            # 網格數量資訊
            if hasattr(sensor, "_num_meshes_per_env"):
                # MultiMeshRayCaster
                total_meshes = sum(sensor._num_meshes_per_env.values())
                f.write(f"\tnumber of meshes     : {sensor._num_envs} x {total_meshes}\n")
            else:
                # 基礎 RayCaster
                f.write(f"\tnumber of meshes     : {len(sensor.meshes)}\n")
            
            f.write(f"\tnumber of sensors    : {stats['num_sensors']}\n")
            f.write(f"\tnumber of rays/sensor: {stats['num_rays_per_sensor']}\n")
            f.write(f"\ttotal number of rays : {stats['total_rays']}\n")
            
            # 統計資訊
            f.write(f"\n統計資訊:\n")
            f.write(f"\t有效碰撞數量        : {stats['num_valid_hits']} / {stats['total_rays']}\n")
            f.write(f"\t有效碰撞比例        : {stats['valid_hit_ratio']:.2%}\n")
            f.write(f"\t平均距離 (m)        : {stats['mean_distance']:.3f}\n")
            f.write(f"\t最小距離 (m)        : {stats['min_distance']:.3f}\n")
            f.write(f"\t最大距離 (m)        : {stats['max_distance']:.3f}\n")
            f.write(f"\t距離標準差 (m)      : {stats['std_distance']:.3f}\n")
            f.write(f"\t感測器平均位置 (m)  : ({stats['sensor_pos_mean_x']:.3f}, "
                   f"{stats['sensor_pos_mean_y']:.3f}, {stats['sensor_pos_mean_z']:.3f})\n")
            f.write(f"{'='*60}\n")

        # 記錄 CSV 格式的統計數據
        with open(self.stats_file, "a") as f:
            f.write(
                f"{stats['step']},{stats['episode']},{stats['timestamp']},"
                f"{stats['num_sensors']},{stats['num_rays_per_sensor']},{stats['total_rays']},"
                f"{stats['num_valid_hits']},{stats['valid_hit_ratio']},"
                f"{stats['mean_distance']},{stats['min_distance']},{stats['max_distance']},"
                f"{stats['std_distance']},"
                f"{stats['sensor_pos_mean_x']},{stats['sensor_pos_mean_y']},"
                f"{stats['sensor_pos_mean_z']},{stats['update_period']}\n"
            )

    def _log_to_tensorboard(self, stats: dict, step: int):
        """將統計資訊記錄到 TensorBoard
        
        Args:
            stats: 統計資訊字典
            step: 當前步驟
        """
        if self.tensorboard_writer is None:
            return

        # 記錄標量數據
        self.tensorboard_writer.add_scalar(
            "RayCaster/num_valid_hits", stats["num_valid_hits"], step
        )
        self.tensorboard_writer.add_scalar(
            "RayCaster/valid_hit_ratio", stats["valid_hit_ratio"], step
        )
        self.tensorboard_writer.add_scalar(
            "RayCaster/mean_distance", stats["mean_distance"], step
        )
        self.tensorboard_writer.add_scalar(
            "RayCaster/min_distance", stats["min_distance"], step
        )
        self.tensorboard_writer.add_scalar(
            "RayCaster/max_distance", stats["max_distance"], step
        )
        self.tensorboard_writer.add_scalar(
            "RayCaster/std_distance", stats["std_distance"], step
        )
        self.tensorboard_writer.add_scalar(
            "RayCaster/sensor_pos_x", stats["sensor_pos_mean_x"], step
        )
        self.tensorboard_writer.add_scalar(
            "RayCaster/sensor_pos_y", stats["sensor_pos_mean_y"], step
        )
        self.tensorboard_writer.add_scalar(
            "RayCaster/sensor_pos_z", stats["sensor_pos_mean_z"], step
        )

    def log_episode_reset(self, episode: int):
        """記錄新 episode 開始
        
        Args:
            episode: episode 編號
        """
        self.episode_count = episode

    def save_summary(self, final_step: int):
        """保存訓練總結
        
        Args:
            final_step: 最終訓練步驟
        """
        if not self.stats_history:
            return

        summary_file = self.log_dir / "ray_caster_summary.txt"
        with open(summary_file, "w") as f:
            f.write("Ray Caster 監測總結\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"總訓練步驟: {final_step}\n")
            f.write(f"總記錄次數: {len(self.stats_history)}\n\n")

            # 計算整體統計
            valid_hit_ratios = [s["valid_hit_ratio"] for s in self.stats_history]
            mean_distances = [s["mean_distance"] for s in self.stats_history if s["mean_distance"] > 0]

            if valid_hit_ratios:
                f.write(f"平均有效碰撞比例: {np.mean(valid_hit_ratios):.2%}\n")
                f.write(f"最小有效碰撞比例: {np.min(valid_hit_ratios):.2%}\n")
                f.write(f"最大有效碰撞比例: {np.max(valid_hit_ratios):.2%}\n\n")

            if mean_distances:
                f.write(f"平均距離: {np.mean(mean_distances):.3f} m\n")
                f.write(f"最小平均距離: {np.min(mean_distances):.3f} m\n")
                f.write(f"最大平均距離: {np.max(mean_distances):.3f} m\n")

        print(f"[INFO] Ray Caster 監測總結已保存到: {summary_file}")

    def print_current_stats(self, stats: dict):
        """打印當前統計資訊（用於即時監控）
        
        Args:
            stats: 統計資訊字典
        """
        print(f"\n[Ray Caster Monitor] Step {stats['step']}")
        print(f"  有效碰撞: {stats['num_valid_hits']}/{stats['total_rays']} "
              f"({stats['valid_hit_ratio']:.2%})")
        print(f"  平均距離: {stats['mean_distance']:.3f} m")
        print(f"  距離範圍: [{stats['min_distance']:.3f}, {stats['max_distance']:.3f}] m")

