# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Carter 導航 PPO 訓練配置模組

此模組定義了使用 PPO（Proximal Policy Optimization）演算法訓練
Carter 導航任務的完整配置，包括：
- 訓練超參數（學習率、批次大小等）
- 網路架構（Actor-Critic 網路層數和大小）
- 演算法參數（裁剪係數、熵係數等）
"""

from isaaclab.utils import configclass  # 配置類別裝飾器
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,  # RSL-RL 策略梯度訓練器配置基類
    RslRlPpoActorCriticCfg,  # PPO Actor-Critic 網路配置
    RslRlPpoAlgorithmCfg,  # PPO 演算法配置
)


@configclass
class CarterNavigationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    Carter 導航 PPO 訓練配置類別

    定義了 PPO 演算法的所有訓練參數，針對 Carter 導航任務進行了優化。
    配置重點：
    - 較低的學習率以提高穩定性
    - 較小的梯度裁剪以穩定訓練
    - 經驗歸一化以提高樣本效率
    """

    # 訓練超參數
    num_steps_per_env = 24  # 每個環境的步數：24 步（每個回合收集 24 個樣本）
    # 24 步 × 128 環境 = 3072 步/迭代
    # Episode 長度約 1125 步（45秒 × 25Hz），所以每次更新使用約 2% 的經驗
    # 這個值對於導航任務是合理的（平衡更新頻率和樣本效率）
    
    max_iterations = 8000  # 最大迭代次數：4800 次（總共訓練 4800 個迭代）
    # 總訓練步數 = 4800 × 3072 = 14,745,600 步
    # 考慮到：
    #   - 觀測空間：72 維（raycaster 5 度解析度）
    #   - 複雜的獎勵結構（朝向獎勵、動態加速度限制）
    #   - Episode 長度：45 秒
    # 這個迭代次數應該是足夠的，但如果訓練不穩定或收斂慢，可以增加到 6000-8000
    
    save_interval = 50  # 模型保存間隔：每 50 次迭代保存一次模型
    # 4800 / 50 = 96 個檢查點（合理）
    # 如果訓練時間很長，可以增加到 100 以減少檢查點數量
    
    experiment_name = "carter_navigation"  # 實驗名稱：用於日誌和模型保存路徑

    # 經驗歸一化設定
    # 使用經驗歸一化但設置裁剪，提高樣本效率同時保持穩定性
    empirical_normalization = True  # 啟用經驗歸一化：使用收集到的經驗統計資訊歸一化觀測

    # Actor-Critic 網路配置
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # 初始動作雜訊標準差：1.0（用於探索）
        noise_std_type="log",  # 噪聲標準差類型：使用對數空間表示（確保標準差永遠為正）
        actor_obs_normalization=True,  # Actor 觀測歸一化：啟用（與 empirical_normalization 配合使用）
        critic_obs_normalization=True,  # Critic 觀測歸一化：啟用（與 empirical_normalization 配合使用）
        actor_hidden_dims=[256, 256, 128],  # Actor 網路隱藏層維度：[256, 256, 128]（3 層）
        critic_hidden_dims=[256, 256, 128],  # Critic 網路隱藏層維度：[256, 256, 128]（3 層）
        activation="elu",  # 激活函數：ELU（Exponential Linear Unit）
    )

    # PPO 演算法配置
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,  # 價值損失係數：1.0（價值函數損失的權重）
        use_clipped_value_loss=True,  # 使用裁剪的價值損失：True（提高穩定性）
        clip_param=0.2,  # 裁剪參數：0.2（PPO 的策略裁剪範圍）
        entropy_coef=0.01,  # 熵係數：0.01（鼓勵探索，但權重較小）
        num_learning_epochs=5,  # 學習輪數：5（每次更新時對同一批數據訓練 5 輪）
        num_mini_batches=4,  # 小批次數量：4（將數據分成 4 個小批次進行訓練）

        # 學習率設定（降低學習率以提高穩定性）
        learning_rate=2e-5,  # 學習率：2e-5（從 3e-4 大幅降低，提高訓練穩定性）

        schedule="adaptive",  # 學習率調度：自適應（根據 KL 散度自動調整）
        gamma=0.99,  # 折扣因子：0.99（未來獎勵的折扣率）
        lam=0.95,  # GAE（Generalized Advantage Estimation）參數：0.95
        desired_kl=0.01,  # 期望 KL 散度：0.01（用於自適應學習率調度）

        # 梯度裁剪設定（強制梯度裁剪以提高穩定性）
        max_grad_norm=0.5,  # 最大梯度範數：0.5（從 1.0 降到 0.5，更激進的梯度裁剪）
    )
