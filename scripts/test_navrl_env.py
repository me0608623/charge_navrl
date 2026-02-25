#!/usr/bin/env python3
"""NavRL 環境測試腳本

驗證 NavRL 環境註冊和配置。
"""

from isaaclab.app import AppLauncher

# 創建應用（headless）
AppLauncher(headless=True).app

import gymnasium as gym

print("=" * 70)
print("    NavRL 環境測試")
print("=" * 70)

# 測試 1: 檢查環境註冊
print("\n[測試 1] 檢查環境註冊...")
env_id = "Isaac-Navigation-Charge-Phase0-NavRL"
all_envs = [e.id for e in gym.registry.values() if "Charge" in e.id]

if env_id in all_envs:
    print(f"  ✅ 環境已註冊: {env_id}")
    spec = gym.spec(env_id)
    print(f"     Entry point: {spec.entry_point}")
else:
    print(f"  ❌ 環境未註冊: {env_id}")
    print(f"     可用的 Charge 環境:")
    for env in sorted(all_envs)[:15]:
        print(f"       - {env}")

# 測試 2: 檢查 Phase0 環境（對比）
print("\n[測試 2] 檢查 Phase0 環境（對比）...")
phase0_id = "Isaac-Navigation-Charge-Phase0"
if phase0_id in all_envs:
    print(f"  ✅ Phase0 環境已註冊: {phase0_id}")
else:
    print(f"  ❌ Phase0 環境未註冊: {phase0_id}")

# 測試 3: 檢查 NavRL 獎勵函數
print("\n[測試 3] 檢查 NavRL 獎勵函數...")
try:
    from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.mdp.rewards import (
        navrl_velocity_reward,
        navrl_safety_reward_lidar,
        navrl_smoothness_penalty,
    )
    print("  ✅ NavRL 獎勵函數已導入")
    print("     - navrl_velocity_reward")
    print("     - navrl_safety_reward_lidar")
    print("     - navrl_smoothness_penalty")
except ImportError as e:
    print(f"  ❌ NavRL 獎勵函數導入失敗: {e}")

print("\n" + "=" * 70)
print("    測試完成")
print("=" * 70)
