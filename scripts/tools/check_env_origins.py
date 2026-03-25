#!/usr/bin/env python3
"""檢查 env_origins 的值"""

import sys
sys.path.insert(0, "source")

from isaaclab.app import AppLauncher

# 創建模擬器
simulation_app = AppLauncher(headless=True)

import torch
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab_tasks.isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.cfg.charge_env_cfg_phase0 as charge_cfg

# 創建環境配置
env_cfg = charge_cfg.ChargeNavigationEnvCfgPhase0(
    num_envs=4,
    env_spacing=18.0,
)

# 創建環境
env: ManagerBasedRLEnv = env_cfg.scene.class_type(
    cfg=env_cfg.scene,
    render_engine="",
)

# 獲取 env_origins
env_origins = env.scene.env_origins

print("=" * 60)
print("env_origins 檢查")
print("=" * 60)
print(f"num_envs: {env.num_envs}")
print(f"env_spacing: {env_cfg.scene.env_spacing}")
print(f"\nenv_origins shape: {env_origins.shape}")
print(f"env_origins:\n{env_origins}")
print()

for i in range(env.num_envs):
    print(f"env_{i}: origin = ({env_origins[i, 0]:.2f}, {env_origins[i, 1]:.2f}, {env_origins[i, 2]:.2f})")

print("=" * 60)

simulation_app.close()
