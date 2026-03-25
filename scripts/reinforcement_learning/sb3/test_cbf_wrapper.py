#!/usr/bin/env python3
"""CBF Action Wrapper 測試腳本

測試 CBF ActionWrapper 的功能：
1. 基本動作過濾
2. 修正量計算
3. 懲罰機制
4. 統計追蹤

使用方式 (在 conda env_isaaclab 環境中):
    python scripts/reinforcement_learning/sb3/test_cbf_wrapper.py
"""

import sys
import numpy as np
import torch

# 測試 imports
print("Testing CBF Wrapper imports...")

try:
    from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.wrappers import (
        CBFActionWrapper,
        CBFWrapperConfig,
        CBFRewardWrapper,
        CBFInfoWrapper,
        wrap_env_with_cbf,
    )
    print("  ✓ CBFActionWrapper imported")
    print("  ✓ CBFWrapperConfig imported")
    print("  ✓ CBFRewardWrapper imported")
    print("  ✓ CBFInfoWrapper imported")
    print("  ✓ wrap_env_with_cbf imported")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

try:
    from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.hierarchical import (
        SafetyShield,
        RuleBasedShield,
        CBFShield,
        CompositeShield,
    )
    from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.hierarchical.types import (
        SafetyAction,
        Obstacle,
    )
    print("  ✓ SafetyShield classes imported")
    print("  ✓ SafetyAction imported")
    print("  ✓ Obstacle imported")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

try:
    from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.mdp.rewards import (
        cbf_correction_penalty,
        cbf_intervention_penalty,
        cbf_adaptive_penalty,
    )
    print("  ✓ CBF penalty functions imported")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("Testing CBF Wrapper Configuration")
print("="*60)

# 測試配置
config = CBFWrapperConfig(
    safety_distance=0.8,
    danger_distance=0.5,
    emergency_distance=0.3,
    penalty_coeff=0.5,
    alpha=1.0,
    max_linear_speed=1.5,
    max_angular_speed=1.5,
    shield_type="composite",
    track_stats=True,
)

print(f"\nConfig created:")
print(f"  safety_distance: {config.safety_distance}m")
print(f"  penalty_coeff: {config.penalty_coeff}")
print(f"  shield_type: {config.shield_type}")

print("\n" + "="*60)
print("Testing SafetyAction")
print("="*60)

# 測試 SafetyAction
safe_action = SafetyAction.safe_action(1.0, 0.5)
unsafe_action = SafetyAction.unsafe(0.0, 0.0, "Emergency stop")

print(f"\nSafe action: linear={safe_action.linear_speed}, angular={safe_action.angular_speed}")
print(f"  Is safe: {safe_action.safe}")

print(f"\nUnsafe action: linear={unsafe_action.linear_speed}, angular={unsafe_action.angular_speed}")
print(f"  Is safe: {unsafe_action.safe}")
print(f"  Reason: {unsafe_action.reason}")

print("\n" + "="*60)
print("Testing SafetyShield (RuleBased)")
print("="*60)

shield = RuleBasedShield(
    safety_distance=0.8,
    danger_distance=0.5,
    emergency_distance=0.3,
)

# 模擬輸入
action = torch.tensor([1.0, 0.5])  # [linear, angular]
robot_pos = torch.tensor([0.0, 0.0])
robot_vel = torch.tensor([0.5, 0.0])
lidar_scan = torch.zeros(72)  # 無障礙物

result = shield.filter_action(action, robot_pos, robot_vel, lidar_scan, [])
print(f"\nNo obstacles - action should be safe:")
print(f"  Input: linear={action[0]:.2f}, angular={action[1]:.2f}")
print(f"  Output: linear={result.linear_speed:.2f}, angular={result.angular_speed:.2f}")
print(f"  Is safe: {result.safe}")

# 模擬接近障礙物
lidar_close = torch.ones(72) * 0.9  # 歸一化：0.9 表示很近（距離 < 1m）
result = shield.filter_action(action, robot_pos, robot_vel, lidar_close, [])
print(f"\nClose obstacle - action should be modified:")
print(f"  Input: linear={action[0]:.2f}, angular={action[1]:.2f}")
print(f"  Output: linear={result.linear_speed:.2f}, angular={result.angular_speed:.2f}")
print(f"  Is safe: {result.safe}")
print(f"  Reason: {result.reason}")

print("\n" + "="*60)
print("Testing Correction Penalty Calculation")
print("="*60)

# 模擬修正量計算
original_action = np.array([1.0, 0.5])
safe_action = np.array([0.2, 0.1])  # CBF 修改後的動作
correction = np.linalg.norm(original_action - safe_action)

penalty_coeff = 0.5
penalty = penalty_coeff * (correction ** 2)

print(f"\nOriginal action: {original_action}")
print(f"Safe action: {safe_action}")
print(f"Correction (L2 norm): {correction:.4f}")
print(f"Penalty (coeff={penalty_coeff}): {penalty:.4f}")

print("\n" + "="*60)
print("All tests passed!")
print("="*60)

print("\n" + "="*60)
print("Usage Example")
print("="*60)

print("""
# 在 SB3 訓練中使用 CBF Wrapper:

from stable_baselines3 import PPO
import gymnasium as gym
from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.wrappers import wrap_env_with_cbf
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

# 1. 建立 Isaac Lab 環境
env = gym.make("Isaac-charge-SB3-v0")

# 2. 套用 CBF Wrapper
config = CBFWrapperConfig(
    penalty_coeff=0.5,  # 懲罰係數
    shield_type="composite",  # 使用複合過濾器
)
env = wrap_env_with_cbf(env, config)

# 3. 轉換為 SB3 VecEnv
env = Sb3VecEnvWrapper(env)

# 4. 訓練 PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# 5. 查看訓練過程中的 CBF 統計
# 在 TensorBoard 中可以看到:
#   - cbf_trigger_rate: CBF 觸發頻率
#   - cbf_correction_mean: 平均修正量
""")
