# ============================================================================
# 訓練記錄器（Training Logger）
# ============================================================================
"""
訓練超參數和結果記錄工具

這個模塊用於記錄每次訓練的超參數配置和訓練結果，並導出到 CSV 文件。
方便追蹤不同超參數配置下的訓練效果，進行超參數調優。

使用方法：
    1. 在訓練開始前，創建 TrainingLogger 實例
    2. 訓練結束後，調用 log_training_results() 記錄結果
    3. 數據會自動追加到 CSV 文件中

示例：
    from .training_logger import TrainingLogger
    
    # 創建記錄器
    logger = TrainingLogger()
    
    # 記錄訓練結果
    logger.log_training_results(
        final_reward=700.12,
        success_rate=0.25,
        timeout_rate=0.75,
        mean_episode_length=444.31,
        value_loss=6.3465,
        surrogate_loss=0.0162,
        entropy_loss=3.1015,
        action_noise_std=1.17,
        total_timesteps=3686400,
        training_time_seconds=1970,
        notes="第一次調整超參數"
    )
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# 導入配置類以提取超參數
try:
    # 作為模塊導入時使用相對導入（從 log 子目錄導入上一級的配置）
    from ..charge_env_cfg import ChargeNavigationEnvCfg, RewardsCfg
    from ..agents.rsl_rl_ppo_cfg import ChargeNavigationPPORunnerCfg
except ImportError:
    # 直接運行時使用絕對導入
    import sys
    from pathlib import Path
    # 添加父目錄到路徑
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    from charge_env_cfg import ChargeNavigationEnvCfg, RewardsCfg
    from agents.rsl_rl_ppo_cfg import ChargeNavigationPPORunnerCfg


class TrainingLogger:
    """訓練記錄器類
    
    用於記錄和導出訓練超參數與結果到 CSV 文件。
    """
    
    def __init__(self, csv_path: Optional[str] = None):
        """初始化訓練記錄器
        
        Args:
            csv_path: CSV 文件路徑。如果為 None，則使用默認路徑：
                     log/charge_navigation_training_log.csv
        """
        # 設置 CSV 文件路徑
        if csv_path is None:
            # 直接保存在 log 目錄下
            log_dir = Path(__file__).parent
            csv_path = log_dir / "charge_navigation_training_log.csv"
        else:
            csv_path = Path(csv_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = csv_path
        
        # 定義 CSV 列名（超參數 + 訓練結果）
        self.fieldnames = [
            # 時間戳
            "timestamp",
            "experiment_name",
            
            # ========== 環境超參數 ==========
            "episode_length_s",
            "num_envs",
            
            # ========== 獎勵超參數 ==========
            "reward_velocity_toward_goal_weight",
            "reward_velocity_toward_goal_min_dist",
            "reward_distance_to_goal_weight",
            "reward_reaching_goal_weight",
            "reward_reaching_goal_threshold",
            "reward_collision_weight",
            "reward_collision_threshold",
            "reward_action_rate_l2_weight",
            "reward_time_out_weight",
            
            # ========== 終止條件超參數 ==========
            "termination_goal_reached_threshold",
            "termination_collision_threshold",
            
            # ========== PPO 算法超參數 ==========
            "ppo_num_steps_per_env",
            "ppo_max_iterations",
            "ppo_learning_rate",
            "ppo_gamma",
            "ppo_lam",
            "ppo_clip_param",
            "ppo_entropy_coef",
            "ppo_value_loss_coef",
            "ppo_max_grad_norm",
            "ppo_num_learning_epochs",
            "ppo_num_mini_batches",
            "ppo_desired_kl",
            "ppo_schedule",
            "ppo_empirical_normalization",
            
            # ========== 網絡結構超參數 ==========
            "policy_init_noise_std",
            "policy_actor_hidden_dims",
            "policy_critic_hidden_dims",
            "policy_activation",
            
            # ========== 訓練結果指標 ==========
            "final_mean_reward",
            "final_mean_episode_length",
            "success_rate",  # 目標達成率
            "timeout_rate",  # 超時率
            "collision_rate",  # 碰撞率
            "tipped_over_rate",  # 翻倒率
            
            # ========== 訓練損失指標 ==========
            "final_value_loss",
            "final_surrogate_loss",
            "final_entropy_loss",
            "final_action_noise_std",
            
            # ========== 性能指標 ==========
            "total_timesteps",
            "total_training_time_seconds",
            "steps_per_second",
            
            # ========== 獎勵項統計（最後一次迭代的平均值）==========
            "reward_velocity_toward_goal_mean",
            "reward_distance_to_goal_mean",
            "reward_reaching_goal_mean",
            "reward_collision_mean",
            "reward_action_rate_l2_mean",
            "reward_time_out_mean",
            
            # ========== 其他信息 ==========
            "notes",
            "log_dir",  # 訓練日誌目錄路徑
        ]
        
        # 如果 CSV 文件不存在，創建並寫入表頭
        if not csv_path.exists():
            self._create_csv_file()
    
    def _create_csv_file(self):
        """創建 CSV 文件並寫入表頭"""
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
    
    def _extract_hyperparameters(self) -> Dict[str, Any]:
        """從配置文件中提取超參數
        
        Returns:
            包含所有超參數的字典
        """
        # 創建配置實例以獲取默認值
        env_cfg = ChargeNavigationEnvCfg()
        agent_cfg = ChargeNavigationPPORunnerCfg()
        rewards_cfg = RewardsCfg()
        
        # episode_length_s 在 __post_init__ 中設置，需要先初始化
        # 但為了避免副作用，我們直接讀取源文件或使用默認值
        # 這裡使用一個合理的默認值，實際值應該在訓練時記錄
        episode_length_s = 45.0  # 當前配置值
        
        # 提取超參數
        hyperparams = {
            # 時間戳
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_name": agent_cfg.experiment_name,
            
            # 環境超參數
            "episode_length_s": episode_length_s,
            "num_envs": getattr(env_cfg.scene, 'num_envs', None),
            
            # 獎勵超參數
            "reward_velocity_toward_goal_weight": rewards_cfg.velocity_toward_goal.weight,
            "reward_velocity_toward_goal_min_dist": rewards_cfg.velocity_toward_goal.params.get("min_dist", None),
            "reward_distance_to_goal_weight": rewards_cfg.distance_to_goal.weight,
            "reward_reaching_goal_weight": rewards_cfg.reaching_goal.weight,
            "reward_reaching_goal_threshold": rewards_cfg.reaching_goal.params.get("threshold", None),
            "reward_collision_weight": rewards_cfg.collision.weight,
            "reward_collision_threshold": rewards_cfg.collision.params.get("threshold", None),
            "reward_action_rate_l2_weight": rewards_cfg.action_rate_l2.weight,
            "reward_time_out_weight": rewards_cfg.time_out.weight,
            
            # 終止條件超參數
            "termination_goal_reached_threshold": env_cfg.terminations.goal_reached.params.get("threshold", None),
            "termination_collision_threshold": env_cfg.terminations.collision.params.get("threshold", None),
            
            # PPO 算法超參數
            "ppo_num_steps_per_env": agent_cfg.num_steps_per_env,
            "ppo_max_iterations": agent_cfg.max_iterations,
            "ppo_learning_rate": agent_cfg.algorithm.learning_rate,
            "ppo_gamma": agent_cfg.algorithm.gamma,
            "ppo_lam": agent_cfg.algorithm.lam,
            "ppo_clip_param": agent_cfg.algorithm.clip_param,
            "ppo_entropy_coef": agent_cfg.algorithm.entropy_coef,
            "ppo_value_loss_coef": agent_cfg.algorithm.value_loss_coef,
            "ppo_max_grad_norm": agent_cfg.algorithm.max_grad_norm,
            "ppo_num_learning_epochs": agent_cfg.algorithm.num_learning_epochs,
            "ppo_num_mini_batches": agent_cfg.algorithm.num_mini_batches,
            "ppo_desired_kl": agent_cfg.algorithm.desired_kl,
            "ppo_schedule": agent_cfg.algorithm.schedule,
            "ppo_empirical_normalization": agent_cfg.empirical_normalization,
            
            # 網絡結構超參數
            "policy_init_noise_std": agent_cfg.policy.init_noise_std,
            "policy_actor_hidden_dims": str(agent_cfg.policy.actor_hidden_dims),  # 轉為字符串以便 CSV 存儲
            "policy_critic_hidden_dims": str(agent_cfg.policy.critic_hidden_dims),
            "policy_activation": agent_cfg.policy.activation,
        }
        
        return hyperparams
    
    def log_training_results(
        self,
        final_mean_reward: float,
        final_mean_episode_length: float,
        success_rate: float,
        timeout_rate: float,
        collision_rate: float = 0.0,
        tipped_over_rate: float = 0.0,
        final_value_loss: Optional[float] = None,
        final_surrogate_loss: Optional[float] = None,
        final_entropy_loss: Optional[float] = None,
        final_action_noise_std: Optional[float] = None,
        total_timesteps: Optional[int] = None,
        total_training_time_seconds: Optional[float] = None,
        steps_per_second: Optional[float] = None,
        reward_velocity_toward_goal_mean: Optional[float] = None,
        reward_distance_to_goal_mean: Optional[float] = None,
        reward_reaching_goal_mean: Optional[float] = None,
        reward_collision_mean: Optional[float] = None,
        reward_action_rate_l2_mean: Optional[float] = None,
        reward_time_out_mean: Optional[float] = None,
        log_dir: Optional[str] = None,
        notes: Optional[str] = None,
        episode_length_s: Optional[float] = None,
        num_envs: Optional[int] = None,
    ):
        """記錄訓練結果到 CSV 文件
        
        Args:
            final_mean_reward: 最終平均獎勵
            final_mean_episode_length: 最終平均回合長度
            success_rate: 目標達成率（0.0-1.0）
            timeout_rate: 超時率（0.0-1.0）
            collision_rate: 碰撞率（0.0-1.0），默認 0.0
            tipped_over_rate: 翻倒率（0.0-1.0），默認 0.0
            final_value_loss: 最終價值函數損失
            final_surrogate_loss: 最終代理損失
            final_entropy_loss: 最終熵損失
            final_action_noise_std: 最終動作噪聲標準差
            total_timesteps: 總時間步數
            total_training_time_seconds: 總訓練時間（秒）
            steps_per_second: 每秒步數（性能指標）
            reward_velocity_toward_goal_mean: 速度獎勵平均值
            reward_distance_to_goal_mean: 距離獎勵平均值
            reward_reaching_goal_mean: 到達目標獎勵平均值
            reward_collision_mean: 碰撞懲罰平均值
            reward_action_rate_l2_mean: 動作變化率懲罰平均值
            reward_time_out_mean: 超時懲罰平均值
            log_dir: 訓練日誌目錄路徑
            notes: 備註信息
        """
        # 提取超參數
        hyperparams = self._extract_hyperparameters()
        
        # 覆蓋手動提供的參數
        if episode_length_s is not None:
            hyperparams["episode_length_s"] = episode_length_s
        if num_envs is not None:
            hyperparams["num_envs"] = num_envs
        
        # 組合所有數據
        row_data = {
            **hyperparams,
            # 訓練結果
            "final_mean_reward": final_mean_reward,
            "final_mean_episode_length": final_mean_episode_length,
            "success_rate": success_rate,
            "timeout_rate": timeout_rate,
            "collision_rate": collision_rate,
            "tipped_over_rate": tipped_over_rate,
            # 訓練損失
            "final_value_loss": final_value_loss,
            "final_surrogate_loss": final_surrogate_loss,
            "final_entropy_loss": final_entropy_loss,
            "final_action_noise_std": final_action_noise_std,
            # 性能指標
            "total_timesteps": total_timesteps,
            "total_training_time_seconds": total_training_time_seconds,
            "steps_per_second": steps_per_second,
            # 獎勵項統計
            "reward_velocity_toward_goal_mean": reward_velocity_toward_goal_mean,
            "reward_distance_to_goal_mean": reward_distance_to_goal_mean,
            "reward_reaching_goal_mean": reward_reaching_goal_mean,
            "reward_collision_mean": reward_collision_mean,
            "reward_action_rate_l2_mean": reward_action_rate_l2_mean,
            "reward_time_out_mean": reward_time_out_mean,
            # 其他信息
            "log_dir": log_dir,
            "notes": notes or "",
        }
        
        # 寫入 CSV 文件（追加模式）
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row_data)
        
        print(f"[INFO] 訓練結果已記錄到: {self.csv_path}")
    
    def log_from_training_output(
        self,
        training_output: Dict[str, Any],
        log_dir: Optional[str] = None,
        notes: Optional[str] = None,
    ):
        """從訓練輸出字典中提取並記錄結果
        
        這是一個便捷方法，可以從訓練過程中的統計信息直接記錄。
        
        Args:
            training_output: 包含訓練結果的字典，應包含以下鍵：
                - mean_reward: 平均獎勵
                - mean_episode_length: 平均回合長度
                - success_rate: 目標達成率
                - timeout_rate: 超時率
                - collision_rate: 碰撞率（可選）
                - tipped_over_rate: 翻倒率（可選）
                - value_loss: 價值函數損失（可選）
                - surrogate_loss: 代理損失（可選）
                - entropy_loss: 熵損失（可選）
                - action_noise_std: 動作噪聲標準差（可選）
                - total_timesteps: 總時間步數（可選）
                - training_time_seconds: 訓練時間（可選）
                - steps_per_second: 每秒步數（可選）
                - reward_*: 各獎勵項的平均值（可選）
            log_dir: 訓練日誌目錄路徑
            notes: 備註信息
        """
        self.log_training_results(
            final_mean_reward=training_output.get("mean_reward"),
            final_mean_episode_length=training_output.get("mean_episode_length"),
            success_rate=training_output.get("success_rate", 0.0),
            timeout_rate=training_output.get("timeout_rate", 0.0),
            collision_rate=training_output.get("collision_rate", 0.0),
            tipped_over_rate=training_output.get("tipped_over_rate", 0.0),
            final_value_loss=training_output.get("value_loss"),
            final_surrogate_loss=training_output.get("surrogate_loss"),
            final_entropy_loss=training_output.get("entropy_loss"),
            final_action_noise_std=training_output.get("action_noise_std"),
            total_timesteps=training_output.get("total_timesteps"),
            total_training_time_seconds=training_output.get("training_time_seconds"),
            steps_per_second=training_output.get("steps_per_second"),
            reward_velocity_toward_goal_mean=training_output.get("reward_velocity_toward_goal"),
            reward_distance_to_goal_mean=training_output.get("reward_distance_to_goal"),
            reward_reaching_goal_mean=training_output.get("reward_reaching_goal"),
            reward_collision_mean=training_output.get("reward_collision"),
            reward_action_rate_l2_mean=training_output.get("reward_action_rate_l2"),
            reward_time_out_mean=training_output.get("reward_time_out"),
            log_dir=log_dir,
            notes=notes,
        )


# ============================================================================
# 便捷函數
# ============================================================================

def quick_log(
    final_mean_reward: float,
    final_mean_episode_length: float,
    success_rate: float,
    timeout_rate: float,
    **kwargs
):
    """快速記錄訓練結果的便捷函數
    
    Args:
        final_mean_reward: 最終平均獎勵
        final_mean_episode_length: 最終平均回合長度
        success_rate: 目標達成率
        timeout_rate: 超時率
        **kwargs: 其他可選參數，傳遞給 log_training_results()
    """
    logger = TrainingLogger()
    logger.log_training_results(
        final_mean_reward=final_mean_reward,
        final_mean_episode_length=final_mean_episode_length,
        success_rate=success_rate,
        timeout_rate=timeout_rate,
        **kwargs
    )


if __name__ == "__main__":
    # 測試代碼
    logger = TrainingLogger()
    
    # 測試記錄
    logger.log_training_results(
        final_mean_reward=700.12,
        final_mean_episode_length=444.31,
        success_rate=0.25,
        timeout_rate=0.75,
        collision_rate=0.0,
        tipped_over_rate=0.0,
        final_value_loss=6.3465,
        final_surrogate_loss=0.0162,
        final_entropy_loss=3.1015,
        final_action_noise_std=1.17,
        total_timesteps=3686400,
        total_training_time_seconds=1970,
        steps_per_second=2186,
        reward_velocity_toward_goal_mean=0.0,
        reward_distance_to_goal_mean=0.0,
        reward_reaching_goal_mean=0.0,
        reward_collision_mean=0.0,
        reward_action_rate_l2_mean=-0.3193,
        reward_time_out_mean=0.0,
        notes="測試記錄 - 第一次訓練",
    )
    
    print("測試完成！請檢查 CSV 文件。")
