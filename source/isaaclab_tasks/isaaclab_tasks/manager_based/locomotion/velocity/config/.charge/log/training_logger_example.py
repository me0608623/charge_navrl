# ============================================================================
# 訓練記錄器使用示例
# ============================================================================
"""
這個文件展示如何使用 TrainingLogger 來記錄訓練結果。

使用場景：
1. 訓練結束後手動記錄結果
2. 從訓練日誌中提取結果並記錄
3. 批量記錄多次訓練的結果
"""

from .training_logger import TrainingLogger, quick_log


# ============================================================================
# 示例 1：基本使用 - 手動記錄訓練結果
# ============================================================================

def example_basic_usage():
    """基本使用示例"""
    # 創建記錄器
    logger = TrainingLogger()
    
    # 記錄訓練結果
    logger.log_training_results(
        final_mean_reward=700.12,
        final_mean_episode_length=444.31,
        success_rate=0.25,  # 25% 成功率
        timeout_rate=0.75,  # 75% 超時率
        collision_rate=0.0,
        tipped_over_rate=0.0,
        final_value_loss=6.3465,
        final_surrogate_loss=0.0162,
        final_entropy_loss=3.1015,
        final_action_noise_std=1.17,
        total_timesteps=3686400,
        total_training_time_seconds=1970,  # 32分50秒
        steps_per_second=2186,
        reward_velocity_toward_goal_mean=0.0,
        reward_distance_to_goal_mean=0.0,
        reward_reaching_goal_mean=0.0,
        reward_collision_mean=0.0,
        reward_action_rate_l2_mean=-0.3193,
        reward_time_out_mean=0.0,
        episode_length_s=45.0,  # 手動指定（因為在 __post_init__ 中設置）
        num_envs=32,  # 手動指定環境數量
        notes="第一次訓練 - 調整後的超參數",
    )


# ============================================================================
# 示例 2：使用便捷函數快速記錄
# ============================================================================

def example_quick_log():
    """使用便捷函數快速記錄"""
    quick_log(
        final_mean_reward=700.12,
        final_mean_episode_length=444.31,
        success_rate=0.25,
        timeout_rate=0.75,
        final_value_loss=6.3465,
        notes="快速記錄示例",
    )


# ============================================================================
# 示例 3：從訓練輸出字典記錄
# ============================================================================

def example_from_dict():
    """從訓練輸出字典中提取並記錄"""
    logger = TrainingLogger()
    
    # 模擬訓練輸出
    training_output = {
        "mean_reward": 700.12,
        "mean_episode_length": 444.31,
        "success_rate": 0.25,
        "timeout_rate": 0.75,
        "collision_rate": 0.0,
        "tipped_over_rate": 0.0,
        "value_loss": 6.3465,
        "surrogate_loss": 0.0162,
        "entropy_loss": 3.1015,
        "action_noise_std": 1.17,
        "total_timesteps": 3686400,
        "training_time_seconds": 1970,
        "steps_per_second": 2186,
        "reward_velocity_toward_goal": 0.0,
        "reward_distance_to_goal": 0.0,
        "reward_reaching_goal": 0.0,
        "reward_collision": 0.0,
        "reward_action_rate_l2": -0.3193,
        "reward_time_out": 0.0,
    }
    
    logger.log_from_training_output(
        training_output=training_output,
        log_dir="/path/to/logs/2024-01-15_14-30-25",
        notes="從訓練輸出字典記錄",
    )


# ============================================================================
# 示例 4：在訓練腳本中集成使用
# ============================================================================

def example_integration():
    """在訓練腳本中集成使用的示例代碼"""
    
    # 在訓練開始時創建記錄器
    logger = TrainingLogger()
    
    # ... 訓練代碼 ...
    # runner.learn(num_learning_iterations=agent_cfg.max_iterations)
    
    # 訓練結束後，從最後一次迭代的統計中提取結果
    # 假設您有訪問 runner 或日誌的方式
    # final_stats = runner.get_final_stats()  # 這需要根據實際 API 調整
    
    # 記錄結果
    logger.log_training_results(
        final_mean_reward=700.12,  # 從訓練統計中獲取
        final_mean_episode_length=444.31,
        success_rate=0.25,  # 從終止條件統計中計算
        timeout_rate=0.75,
        final_value_loss=6.3465,
        final_surrogate_loss=0.0162,
        final_entropy_loss=3.1015,
        final_action_noise_std=1.17,
        total_timesteps=3686400,
        total_training_time_seconds=1970,
        steps_per_second=2186,
        episode_length_s=45.0,
        num_envs=32,
        log_dir="/path/to/logs",  # 實際的日誌目錄
        notes="訓練完成",
    )


# ============================================================================
# 示例 5：批量記錄多次訓練結果
# ============================================================================

def example_batch_logging():
    """批量記錄多次訓練結果"""
    logger = TrainingLogger()
    
    # 多次訓練的結果列表
    training_results = [
        {
            "final_mean_reward": 700.12,
            "final_mean_episode_length": 444.31,
            "success_rate": 0.25,
            "timeout_rate": 0.75,
            "notes": "第一次訓練",
        },
        {
            "final_mean_reward": 750.50,
            "final_mean_episode_length": 500.20,
            "success_rate": 0.35,
            "timeout_rate": 0.60,
            "notes": "第二次訓練 - 調整獎勵權重",
        },
        {
            "final_mean_reward": 800.30,
            "final_mean_episode_length": 550.10,
            "success_rate": 0.45,
            "timeout_rate": 0.50,
            "notes": "第三次訓練 - 增加探索係數",
        },
    ]
    
    # 批量記錄
    for i, result in enumerate(training_results, 1):
        logger.log_training_results(
            **result,
            episode_length_s=45.0,
            num_envs=32,
        )
        print(f"已記錄第 {i} 次訓練結果")


if __name__ == "__main__":
    # 運行示例
    print("運行基本使用示例...")
    example_basic_usage()
    
    print("\n運行快速記錄示例...")
    example_quick_log()
    
    print("\n運行批量記錄示例...")
    example_batch_logging()
    
    print("\n所有示例完成！請檢查 CSV 文件。")
