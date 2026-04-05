#!/usr/bin/env python3
"""
快速記錄訓練結果

使用方法：
    python quick_log.py

或者直接運行 Python：
    python -c "exec(open('quick_log.py').read())"
"""

import sys
from pathlib import Path

# 添加父目錄到路徑以便導入
sys.path.insert(0, str(Path(__file__).parent.parent))

from log.training_logger import quick_log

# ============================================================================
# 在這裡填入您的訓練結果
# ============================================================================

# 必需參數
final_mean_reward = None  # 請填入：最終平均獎勵
final_mean_episode_length = None  # 請填入：最終平均回合長度
success_rate = 0.0  # 請填入：目標達成率（0.0-1.0）
timeout_rate = 0.0  # 請填入：超時率（0.0-1.0）

# 可選參數
collision_rate = 0.0  # 碰撞率
final_value_loss = None  # 最終價值函數損失
final_surrogate_loss = None  # 最終代理損失
final_entropy_loss = None  # 最終熵損失
final_action_noise_std = None  # 最終動作噪聲標準差
total_timesteps = None  # 總時間步數
total_training_time_seconds = None  # 總訓練時間（秒）
steps_per_second = None  # 每秒步數
notes = "訓練完成"  # 備註

# ============================================================================
# 記錄結果
# ============================================================================

if __name__ == "__main__":
    if final_mean_reward is None or final_mean_episode_length is None:
        print("❌ 請先填入 final_mean_reward 和 final_mean_episode_length！")
        print("\n請編輯此文件，填入您的訓練結果數據。")
        sys.exit(1)
    
    print("正在記錄訓練結果...")
    quick_log(
        final_mean_reward=final_mean_reward,
        final_mean_episode_length=final_mean_episode_length,
        success_rate=success_rate,
        timeout_rate=timeout_rate,
        collision_rate=collision_rate,
        final_value_loss=final_value_loss,
        final_surrogate_loss=final_surrogate_loss,
        final_entropy_loss=final_entropy_loss,
        final_action_noise_std=final_action_noise_std,
        total_timesteps=total_timesteps,
        total_training_time_seconds=total_training_time_seconds,
        steps_per_second=steps_per_second,
        notes=notes,
    )
    print("✅ 訓練結果已成功記錄！")
