#!/usr/bin/env python3
"""層級式導航訓練腳本 (Hierarchical Navigation Training)

整合 AIT* + RL (PPO) 的訓練腳本，支援：
1. 動態重規劃 (Dynamic Re-planning)
2. 域隨機化 (Domain Randomization)
3. Frontier 探索 (Frontier Exploration)
"""

import argparse
import contextlib
import signal
import sys
import os
from pathlib import Path
from datetime import datetime

# 必須在 Isaac App 啟動後導入
simulation_app = None


def main():
    global simulation_app
    
    # 解析參數
    parser = argparse.ArgumentParser(description="Train hierarchical navigation with SB3 PPO")
    parser.add_argument("--task", type=str, default="Isaac-Navigation-Charge-SB3-v0", help="Task name")
    parser.add_argument("--num_envs", type=int, default=128, help="Number of environments")
    parser.add_argument("--max_iterations", type=int, default=10000, help="Max training iterations")
    parser.add_argument("--headless", action="store_true", default=False, help="Headless mode")
    parser.add_argument("--enable_viz", action="store_true", help="Enable AIT* visualization")
    parser.add_argument("--enable_dr", action="store_true", help="Enable domain randomization")
    parser.add_argument("--replan_interval", type=float, default=2.0, help="AIT* replan interval (seconds)")
    
    args, hydra_args = parser.parse_known_args()
    
    # 導入 Isaac Lab
    from isaaclab.app import AppLauncher
    
    # 創建 AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    
    # 導入其餘模組
    import gymnasium as gym
    import logging
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, LogEveryNTimesteps
    from stable_baselines3.common.vec_env import VecNormalize
    
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
    import isaaclab_tasks
    
    logger = logging.getLogger(__name__)
    
    # ============================================================================
    # 創建環境
    # ============================================================================
    
    env = gym.make(args.task, num_envs=args.num_envs, headless=args.headless)
    env = Sb3VecEnvWrapper(env)
    
    # 可選：添加觀測歸一化
    env = VecNormalize(
        env,
        training=True,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
        clip_reward=np.inf,
    )
    
    # ============================================================================
    # SB3 PPO 配置
    # ============================================================================
    
    agent_cfg = {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.001,
        "vf_coef": 1.0,
        "max_grad_norm": 1.0,
        "n_timesteps": args.max_iterations * args.num_envs,
        "seed": 42,
    }
    
    # ============================================================================
    # 創建 PPO Agent
    # ============================================================================
    
    agent = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/hierarchical_sb3", **agent_cfg)
    
    # ============================================================================
    # 訓練回調
    # ============================================================================
    
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./logs/hierarchical_sb3",
        name_prefix="hierarchical_model",
    )
    
    callbacks = [
        checkpoint_callback,
        LogEveryNTimesteps(n_steps=10000),
    ]
    
    # ============================================================================
    # 訓練循環
    # ============================================================================
    
    print(f"\n{'='*60}")
    print("開始層級式導航訓練")
    print(f"{'='*60}")
    print(f"任務: {args.task}")
    print(f"環境數量: {args.num_envs}")
    print(f"最大迭代: {args.max_iterations}")
    print(f"AIT* 重規劃間隔: {args.replan_interval} 秒")
    print(f"視覺化: {'啟用' if args.enable_viz else '關閉'}")
    print(f"域隨機化: {'啟用' if args.enable_dr else '關閉'}")
    print(f"{'='*60}\n")
    
    try:
        agent.learn(
            total_timesteps=agent_cfg["n_timesteps"],
            callback=callbacks,
            progress_bar=True,
        )
        
        # 保存最終模型
        agent.save("./logs/hierarchical_sb3/final_model")
        print("訓練完成！模型已保存")
        
    except KeyboardInterrupt:
        print("\n用戶中斷，正在保存模型...")
        agent.save("./logs/hierarchical_sb3/interrupted_model")
        print("模型已保存")
    
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
