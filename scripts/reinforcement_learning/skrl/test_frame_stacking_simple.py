#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Test Script for Frame Stacking and TTC Penalty

測試腳本：驗證 Frame Stacking 和 TTC Penalty 的實現

使用方法:
    python test_frame_stacking_simple.py
"""

import torch
import numpy as np


def test_frame_stacking():
    """測試 Frame Stacking 的滾動更新邏輯"""
    print("\n" + "=" * 70)
    print("🧪 Test 1: Frame Stacking 邏輯")
    print("=" * 70)

    # 模擬參數
    num_envs = 4
    obs_dim = 81
    num_stack = 3
    stacked_dim = obs_dim * num_stack  # 243

    # 初始化緩衝區
    stacked_obs = torch.zeros((num_envs, stacked_dim))
    print(f"初始狀態: stacked_obs.shape = {stacked_obs.shape}")

    # 第一次 reset (所有幀相同)
    initial_obs = torch.randn(num_envs, obs_dim)
    stacked_obs = initial_obs.repeat(1, num_stack)
    print(f"Reset 後: 所有幀相同")

    # 滾動更新 5 次
    for step in range(1, 6):
        new_obs = torch.randn(num_envs, obs_dim)
        stacked_obs[:, :-obs_dim] = stacked_obs[:, obs_dim:].clone()
        stacked_obs[:, -obs_dim:] = new_obs

    print(f"滾動 5 步後完成")

    # 模擬 Reset (部分環境)
    reset_mask = torch.tensor([False, True, False, True])
    reset_obs = torch.randn(num_envs, obs_dim)
    reset_indices = torch.where(reset_mask)[0]

    if len(reset_indices) > 0:
        stacked_obs[reset_indices] = reset_obs[reset_indices].repeat(1, num_stack)

    print(f"Reset {len(reset_indices)} 個環境，歷史已清除")

    # 驗證重置環境的所有幀相同
    for idx in reset_indices:
        f1 = stacked_obs[idx, :obs_dim]
        f2 = stacked_obs[idx, obs_dim:obs_dim*2]
        f3 = stacked_obs[idx, obs_dim*2:obs_dim*3]
        assert torch.allclose(f1, f2), f"環境 {idx} 重置後 Frame 1 ≠ Frame 2"
        assert torch.allclose(f2, f3), f"環境 {idx} 重置後 Frame 2 ≠ Frame 3"

    print("✅ Test 1 PASSED: Frame Stacking 邏輯正確")
    return True


def test_ttc_penalty():
    """測試 TTC Penalty 的純張量計算"""
    print("\n" + "=" * 70)
    print("🧪 Test 2: TTC Penalty 計算")
    print("=" * 70)

    tau = 1.5  # 安全時間閾值
    epsilon = 1e-5

    # 場景: 正在靠近的障礙物
    print("\n--- 場景: 正在靠近的障礙物 ---")
    num_envs = 2
    num_obstacles = 2

    # 機器人在原點
    p_r = torch.zeros(num_envs, 2)
    v_r = torch.zeros(num_envs, 2)

    # 障礙物位置和速度
    p_o = torch.tensor([
        [[2.0, 0.0], [3.0, 0.0]],  # 環境 0
        [[1.0, 0.0], [4.0, 0.0]],  # 環境 1
    ])
    v_o = torch.tensor([
        [[-1.0, 0.0], [-0.5, 0.0]],  # 環境 0
        [[-1.0, 0.0], [-0.3, 0.0]],  # 環境 1
    ])

    # 障礙物高度 (全部 > 0)
    z_o = torch.ones(num_envs, num_obstacles, 1)

    # 擴展機器人維度
    p_r_exp = p_r.unsqueeze(1).expand(-1, num_obstacles, -1)  # [2, 2, 2]
    v_r_exp = v_r.unsqueeze(1).expand(-1, num_obstacles, -1)  # [2, 2, 2]

    print(f"p_r_exp shape: {p_r_exp.shape}")
    print(f"p_o shape: {p_o.shape}")
    print(f"z_o shape: {z_o.shape}")

    # 計算 TTC
    dp = p_o - p_r_exp  # [2, 2, 2]
    dv = v_o - v_r_exp  # [2, 2, 2]
    dist = torch.norm(dp, dim=-1, keepdim=True)  # [2, 2, 1]
    dot_product = (dp * dv).sum(dim=-1, keepdim=True)  # [2, 2, 1]
    v_app = -dot_product / (dist + epsilon)  # [2, 2, 1]
    ttc = dist / (v_app + epsilon)  # [2, 2, 1]

    print(f"dist shape: {dist.shape}")
    print(f"v_app shape: {v_app.shape}")
    print(f"ttc shape: {ttc.shape}")

    # 計算懲罰 (確保所有張量都是 2D)
    danger_mask = (z_o.squeeze(-1) > 0) & (v_app.squeeze(-1) > 0) & (ttc.squeeze(-1) < tau)
    penalty_tensor = torch.where(
        danger_mask,
        -(tau - ttc.squeeze(-1)),
        torch.zeros_like(ttc.squeeze(-1))
    )
    penalty = penalty_tensor.sum(dim=-1)  # 對最後一維求和

    print(f"\n環境 0: dist={dist[0, 0, 0].item():.2f}, v_app={v_app[0, 0, 0].item():.2f}, ttc={ttc[0, 0, 0].item():.2f}")
    print(f"環境 1: dist={dist[1, 0, 0].item():.2f}, v_app={v_app[1, 0, 0].item():.2f}, ttc={ttc[1, 0, 0].item():.2f}")
    print(f"Penalty tensor shape: {penalty_tensor.shape}")
    print(f"Penalty shape: {penalty.shape}")
    print(f"Penalty values: {penalty.tolist()}")

    # 驗證 TTC 計算
    # 環境 0: TTC=2 > tau=1.5，無懲罰
    # 環境 1: TTC=1 < tau=1.5，有懲罰 -(1.5-1) = -0.5
    print(f"\n驗證結果:")
    print(f"  環境 0: TTC={ttc[0, 0, 0].item():.2f} > tau={tau} → 懲罰={penalty[0].item():.2f}")
    print(f"  環境 1: TTC={ttc[1, 0, 0].item():.2f} < tau={tau} → 懲罰={penalty[1].item():.2f}")

    assert penalty[0].item() == 0.0, "環境 0 不應該有懲罰 (TTC > tau)"
    assert penalty[1].item() < 0, "環境 1 應該有懲罰 (TTC < tau)"
    print("✅ Test 2 PASSED: TTC Penalty 計算正確")
    return True


def test_dimensions():
    """測試觀測和狀態維度是否正確"""
    print("\n" + "=" * 70)
    print("🧪 Test 3: 維度驗證")
    print("=" * 70)

    # 原始維度
    original_obs_dim = 81
    num_stack = 3
    privileged_dim = 50

    # 計算新維度
    actor_dim = original_obs_dim * num_stack  # 243
    critic_dim = actor_dim + privileged_dim  # 293

    print(f"原始 Actor 觀測: {original_obs_dim} 維")
    print(f"Frame Stacking:   {num_stack} 幀")
    print(f"新的 Actor 輸入:   {actor_dim} 維")
    print(f"新的 Critic 輸入:  {critic_dim} 維")

    # 驗證
    assert actor_dim == 243
    assert critic_dim == 293

    # 模擬張量操作
    batch_size = 512
    actor_obs = torch.randn(batch_size, actor_dim)
    privileged_info = torch.randn(batch_size, privileged_dim)
    critic_state = torch.cat([actor_obs, privileged_info], dim=-1)

    print(f"\nActor 觀測 shape: {actor_obs.shape}")
    print(f"Critic 狀態 shape: {critic_state.shape}")

    assert critic_state.shape == (batch_size, 293)

    print("✅ Test 3 PASSED: 維度正確")
    return True


def test_gradient_flow():
    """測試梯度流是否正常"""
    print("\n" + "=" * 70)
    print("🧪 Test 4: 梯度流測試")
    print("=" * 70)

    num_envs = 4
    obs_dim = 81

    # 模擬神經網路輸出
    policy_output = torch.randn(num_envs, obs_dim, requires_grad=True)

    # Frame Stacking 操作
    stacked = policy_output.repeat(1, 3)
    loss = stacked.sum()
    loss.backward()

    # 檢查梯度
    assert policy_output.grad is not None
    assert torch.all(policy_output.grad > 0)

    print(f"Policy output shape: {policy_output.shape}")
    print(f"Stacked shape: {stacked.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Gradient mean: {policy_output.grad.mean().item():.4f}")

    print("✅ Test 4 PASSED: 梯度流正常")
    return True


def main():
    """運行所有測試"""
    print("\n" + "=" * 70)
    print("🚀 Frame Stacking & TTC Penalty 測試套件")
    print("=" * 70)

    tests = [
        test_frame_stacking,
        test_ttc_penalty,
        test_dimensions,
        test_gradient_flow,
    ]

    results = {}
    for test_func in tests:
        try:
            results[test_func.__name__] = test_func()
        except Exception as e:
            print(f"\n❌ Test FAILED: {test_func.__name__}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            results[test_func.__name__] = False

    # 總結
    print("\n" + "=" * 70)
    print("📊 測試結果總結")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30s}: {status}")

    all_passed = all(results.values())
    print("=" * 70)

    if all_passed:
        print("\n🎉 所有測試通過！")
        return 0
    else:
        print("\n⚠️  部分測試失敗")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
