# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ray Caster Monitor 使用範例

此文件展示如何在訓練腳本中使用 RayCasterMonitor 來監測 ray_caster 感測器數據。

使用方式：
1. 在訓練腳本中導入 RayCasterMonitor
2. 初始化監測器
3. 在訓練循環中調用 log_step 方法
4. 訓練結束後調用 save_summary 方法
"""

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors.ray_caster import RayCasterMonitor


def train_with_monitor(env: ManagerBasedRLEnv, log_dir: str, num_steps: int = 10000):
    """訓練函數範例 - 展示如何使用 Ray Caster Monitor
    
    Args:
        env: ManagerBasedRLEnv 環境實例
        log_dir: 日誌保存目錄
        num_steps: 訓練總步數
    """
    # 方式 1: 僅保存到文件
    monitor = RayCasterMonitor(
        log_dir=log_dir,
        sensor_name="lidar",  # 對應環境配置中的感測器名稱
        log_interval=100,  # 每 100 步記錄一次
        save_to_file=True,
        save_to_tensorboard=False,
    )

    # 方式 2: 同時保存到文件和 TensorBoard
    # tensorboard_writer = SummaryWriter(log_dir=log_dir)
    # monitor = RayCasterMonitor(
    #     log_dir=log_dir,
    #     sensor_name="lidar",
    #     log_interval=100,
    #     save_to_file=True,
    #     save_to_tensorboard=True,
    #     tensorboard_writer=tensorboard_writer,
    # )

    # 訓練循環
    obs, _ = env.reset()
    
    for step in range(num_steps):
        # 執行環境步驟（這裡簡化，實際應該有動作）
        # action = agent.get_action(obs)
        # obs, reward, terminated, truncated, info = env.step(action)
        
        # 記錄 Ray Caster 數據
        if step % monitor.log_interval == 0:
            stats = monitor.log_step(env, step)
            # 可選：打印當前統計資訊
            monitor.print_current_stats(stats)
        
        # 處理 episode 重置
        # if terminated.any() or truncated.any():
        #     monitor.log_episode_reset(episode_count)
        #     obs, _ = env.reset()

    # 保存總結
    monitor.save_summary(final_step=num_steps)
    
    print(f"[INFO] Ray Caster 監測數據已保存到: {log_dir}")


# 在實際訓練腳本中的使用範例
def example_in_training_script():
    """在實際訓練腳本中的使用範例
    
    假設您有一個訓練腳本，可以這樣整合：
    """
    
    # 導入監測類
    # from isaaclab.sensors.ray_caster import RayCasterMonitor
    
    # 在訓練開始前初始化
    # log_dir = "logs/carter_training"
    # monitor = RayCasterMonitor(
    #     log_dir=log_dir,
    #     sensor_name="lidar",
    #     log_interval=100,
    #     save_to_file=True,
    # )
    
    # 在訓練循環中
    # for step in range(num_steps):
    #     # ... 您的訓練代碼 ...
    #     
    #     # 記錄 Ray Caster 數據
    #     if step % monitor.log_interval == 0:
    #         stats = monitor.log_step(env, step)
    #     
    #     # 處理 episode 重置
    #     if reset_occurred:
    #         monitor.log_episode_reset(episode_count)
    
    # 訓練結束後
    # monitor.save_summary(final_step=num_steps)
    
    pass


if __name__ == "__main__":
    print("這是 Ray Caster Monitor 的使用範例文件")
    print("請參考此文件中的範例代碼來整合到您的訓練腳本中")

