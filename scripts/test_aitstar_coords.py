#!/usr/bin/env python3
"""测试 AIT* 坐标系统"""

from isaaclab.app import AppLauncher

# 创建 Isaac Lab 应用
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
import numpy as np
from isaaclab.envs import ManagerBasedEnv

# 导入环境配置
import sys
sys.path.append("/home/aa/IsaacLab/source/isaaclab_tasks")
from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.cfg.charge_env_cfg_phase0 import (
    ChargeNavigationEnvCfgPhase0,
)

# 创建环境配置
env_cfg = ChargeNavigationEnvCfgPhase0()
env_cfg.scene.num_envs = 4  # 只用 4 个环境

# 创建环境
env = ManagerBasedEnv(cfg=env_cfg)

# 重置环境（触发 AIT* 规划）
print("\n" + "="*70)
print("重置环境...")
print("="*70)
env.reset()

print("\n" + "="*70)
print("运行 10 步...")
print("="*70)
for i in range(10):
    actions = torch.zeros(env.num_envs, 2, device=env.device)
    obs, rew, terminated, truncated, info = env.step(actions)

# 关闭环境
env.close()
simulation_app.close()

print("\n测试完成!")
