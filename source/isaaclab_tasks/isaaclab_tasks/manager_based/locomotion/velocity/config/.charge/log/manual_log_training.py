#!/usr/bin/env python3
# ============================================================================
# 手動記錄訓練結果腳本
# ============================================================================
"""
手動記錄訓練結果到 CSV 文件

使用方法：
    python manual_log_training.py

或者直接運行並輸入數據：
    python -c "from manual_log_training import quick_log_training; quick_log_training()"
"""

from .training_logger import TrainingLogger, quick_log


def quick_log_training():
    """快速記錄訓練結果的交互式腳本"""
    print("=" * 80)
    print("訓練結果記錄工具")
    print("=" * 80)
    print()
    
    # 獲取基本信息
    print("請輸入訓練結果（直接按 Enter 使用默認值或跳過）：")
    print()
    
    final_mean_reward = input("最終平均獎勵 (final_mean_reward): ").strip()
    final_mean_reward = float(final_mean_reward) if final_mean_reward else None
    
    final_mean_episode_length = input("最終平均回合長度 (final_mean_episode_length): ").strip()
    final_mean_episode_length = float(final_mean_episode_length) if final_mean_episode_length else None
    
    success_rate = input("目標達成率 (success_rate, 0.0-1.0): ").strip()
    success_rate = float(success_rate) if success_rate else 0.0
    
    timeout_rate = input("超時率 (timeout_rate, 0.0-1.0): ").strip()
    timeout_rate = float(timeout_rate) if timeout_rate else 0.0
    
    # 可選參數
    print("\n可選參數（直接按 Enter 跳過）：")
    
    collision_rate = input("碰撞率 (collision_rate): ").strip()
    collision_rate = float(collision_rate) if collision_rate else 0.0
    
    final_value_loss = input("最終價值函數損失 (final_value_loss): ").strip()
    final_value_loss = float(final_value_loss) if final_value_loss else None
    
    final_surrogate_loss = input("最終代理損失 (final_surrogate_loss): ").strip()
    final_surrogate_loss = float(final_surrogate_loss) if final_surrogate_loss else None
    
    final_entropy_loss = input("最終熵損失 (final_entropy_loss): ").strip()
    final_entropy_loss = float(final_entropy_loss) if final_entropy_loss else None
    
    total_timesteps = input("總時間步數 (total_timesteps): ").strip()
    total_timesteps = int(total_timesteps) if total_timesteps else None
    
    total_training_time_seconds = input("總訓練時間（秒）(total_training_time_seconds): ").strip()
    total_training_time_seconds = float(total_training_time_seconds) if total_training_time_seconds else None
    
    notes = input("備註 (notes): ").strip()
    notes = notes if notes else None
    
    # 檢查必需參數
    if final_mean_reward is None or final_mean_episode_length is None:
        print("\n❌ 錯誤：必須提供 final_mean_reward 和 final_mean_episode_length！")
        return
    
    # 記錄結果
    logger = TrainingLogger()
    
    print("\n正在記錄訓練結果...")
    logger.log_training_results(
        final_mean_reward=final_mean_reward,
        final_mean_episode_length=final_mean_episode_length,
        success_rate=success_rate,
        timeout_rate=timeout_rate,
        collision_rate=collision_rate,
        final_value_loss=final_value_loss,
        final_surrogate_loss=final_surrogate_loss,
        final_entropy_loss=final_entropy_loss,
        total_timesteps=total_timesteps,
        total_training_time_seconds=total_training_time_seconds,
        notes=notes,
    )
    
    print("\n✅ 訓練結果已成功記錄到 CSV 文件！")


if __name__ == "__main__":
    quick_log_training()
