# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class ChargeNavigationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    Charge 導航 PPO 訓練配置類別
    
    定義了使用 PPO（Proximal Policy Optimization）算法訓練 Charge 機器人導航任務的完整配置。
    這個配置針對 Charge 差速驅動機器人的導航任務進行了優化。
    """
    
    # ========================================================================
    # 訓練超參數（Training Hyperparameters）
    # ========================================================================
    
    num_steps_per_env = 24
    """
    每個環境在每次更新前收集的步數
    
    說明：
    - 24 步表示每個並行環境在策略更新前會收集 24 個經驗樣本
    - 總經驗量 = num_steps_per_env × num_envs（環境數量）
    - 例如：24 步/環境 × 128 環境 = 3,072 步/迭代
    
    設計考量：
    - Episode 長度：45 秒 × 25 Hz（控制頻率）= 1,125 步
    - 每次更新使用約 2% 的經驗（24/1125）
    - 這個值在更新頻率和樣本效率之間取得平衡
    - 較小的值（如 16）：更新更頻繁，但樣本效率較低
    - 較大的值（如 32）：樣本效率更高，但更新頻率較低
    
    建議範圍：16-32（導航任務的常見範圍）
    """
    
    max_iterations = 8000
    """
    最大訓練迭代次數
    
    說明：
    - 一次迭代包括：收集經驗 → 策略更新 → 記錄日誌
    - 總訓練步數 = max_iterations × num_steps_per_env × num_envs
    - 例如：8000 × 24 × 128 = 24,576,000 步
    
    設計考量：
    - 從 4800 增加到 8000，考慮到：
      * 觀測空間增加：從 36 維增加到 72 維（raycaster 5 度解析度）
      * 複雜的獎勵結構：朝向目標獎勵、動態加速度限制
      * 動態加速度限制：根據距離調整加速度，增加策略複雜度
      * Episode 長度：45 秒（較長的訓練時間）
    
    訓練時間估算（假設 2000 steps/s）：
    - 24,576,000 步 ÷ 2000 steps/s ≈ 12,288 秒 ≈ 3.4 小時
    
    建議：
    - 如果訓練早期收斂：可以減少到 6000
    - 如果訓練不穩定或收斂慢：可以增加到 10000
    """
    
    save_interval = 50
    """
    模型檢查點保存間隔
    
    說明：
    - 每 50 次迭代保存一次模型檢查點
    - 總檢查點數量 = max_iterations / save_interval
    - 例如：8000 / 50 = 160 個檢查點
    
    設計考量：
    - 50 次迭代保存一次是合理的頻率
    - 可以及時保存訓練進度，避免訓練中斷導致進度丟失
    - 如果磁盤空間有限，可以增加到 100（減少檢查點數量）
    - 如果希望更頻繁保存，可以減少到 25（但會增加磁盤使用）
    
    檢查點用途：
    - 恢復訓練：從檢查點繼續訓練
    - 模型評估：測試不同迭代的模型性能
    - 最佳模型選擇：選擇訓練過程中表現最好的模型
    """
    
    experiment_name = "charge_navigation"
    """
    實驗名稱
    
    說明：
    - 用於日誌和模型保存路徑的標識符
    - 日誌路徑：logs/rsl_rl/charge_navigation/{timestamp}_{run_name}/
    - 模型保存路徑：logs/rsl_rl/charge_navigation/{timestamp}_{run_name}/model_*.pt
    
    設計考量：
    - 應該使用描述性的名稱，方便識別不同的實驗
    - 如果進行多個實驗變體，可以使用後綴：
      * "charge_navigation_v1"（基礎版本）
      * "charge_navigation_v2"（增加朝向獎勵）
      * "charge_navigation_v3"（動態加速度限制）
    """
    
    # Wandb 日誌記錄配置
    logger = "wandb"  # 使用 wandb 作為日誌記錄器（可選值: "tensorboard", "wandb", "neptune"）
    wandb_project = "charge_sb3"  # Wandb 項目名稱（Phase 0 訓練使用）
    
    # ✅ 使用经验归一化但设置裁剪
    empirical_normalization = True
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        
        # ✅ 建議一：收緊 policy clipping（鎖住已學會的好策略）
        clip_param=0.1,  # 從 0.2 降低到 0.1（更保守的更新）
        # 目的：防止後期被激進行為洗掉已學會的好策略
        # 效果：新舊策略比率被限制在 [0.9, 1.1]，更新更穩定
        
        # ✅ 建議一：後期降低 entropy（停止繼續「嘗試更快但更危險的路徑」）
        entropy_coef=0.001,  # 從 0.01 大幅降低到 0.001（後期收斂）
        # 目的：停止 PPO 繼續探索，專注於利用已學會的策略
        # 效果：策略更確定，減少隨機性，提高穩定性
        
        num_learning_epochs=5,
        num_mini_batches=4,
        
        # ✅ 修復：提高學習率（2024-01 修正）
        # 原問題：2e-5 太低，策略更新太慢，導致訓練後期退化
        # 修復：提高到 1e-4（標準值的 1/3，平衡穩定性和學習速度）
        learning_rate=1e-4,
        
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        
        # 梯度裁剪：保持 0.5（穩定訓練）
        max_grad_norm=0.5,
    )
