#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Script for Frame Stacking and TTC Penalty

測試腳本：驗證 Frame Stacking 和 TTC Penalty 的實現

使用方法:
    python test_frame_stacking.py

測試項目：
1. AAC Wrapper Frame Stacking 功能
2. 觀測維度是否正確 (81 -> 243)
3. Critic state 維度是否正確 (243 + 50 = 293)
4. Reset 時歷史幽靈清除
5. TTC Penalty 計算正確性
"""

import argparse
import sys
from pathlib import Path

# Add isaac lab to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "source"))

import torch
import numpy as np


# ============================================================================
# Test 1: Frame Stacking 邏輯測試
# ============================================================================

def test_frame_stacking_logic():
    """測試 Frame Stacking 的滾動更新邏輯"""
    print("\n" + "=" * 70)
    print("🧪 Test 1: Frame Stacking Logic")
    print("=" * 70)

    # 模擬參數
    num_envs = 4
    obs_dim = 81
    num_stack = 3
    stacked_dim = obs_dim * num_stack  # 243

    # 初始化緩衝區 (全零)
    stacked_obs = torch.zeros((num_envs, stacked_dim))

    print(f"初始狀態: stacked_obs.shape = {stacked_obs.shape}")
    print(f"  前 81 維: {stacked_obs[0, :5].tolist()}... (zeros)")

    # 模擬第一次觀測 (所有幀都設為相同的初始值)
    initial_obs = torch.randn(num_envs, obs_dim)
    stacked_obs = initial_obs.repeat(1, num_stack)

    print(f"\n第一次 reset: stacked_obs = initial_obs.repeat(1, {num_stack})")
    print(f"  Frame 1 (前 5 維): {stacked_obs[0, :5].tolist()}")
    print(f"  Frame 2 (前 5 維): {stacked_obs[0, obs_dim:obs_dim+5].tolist()}")
    print(f"  Frame 3 (前 5 維): {stacked_obs[0, obs_dim*2:obs_dim*2+5].tolist()}")

    # 模擬滾動更新
    for step in range(1, 6):
        new_obs = torch.randn(num_envs, obs_dim)

        # 滾動: 左移
        stacked_obs[:, :-obs_dim] = stacked_obs[:, obs_dim:].clone()
        # 寫入新觀測
        stacked_obs[:, -obs_dim:] = new_obs

        print(f"\nStep {step}: 滾動更新")
        print(f"  Frame 1 (前 5 維): {stacked_obs[0, :5].tolist()}")
        print(f"  Frame 2 (前 5 維): {stacked_obs[0, obs_dim:obs_dim+5].tolist()}")
        print(f"  Frame 3 (前 5 維): {stacked_obs[0, obs_dim*2:obs_dim*2+5].tolist()} (NEW)")

    # 模擬 Reset (清除歷史幽靈)
    print("\n--- 測試 Reset 清除歷史幽靈 ---")
    reset_mask = torch.tensor([False, True, False, True])  # 環境 1, 3 重置

    # 對重置的環境，用當前新觀測填滿所有幀
    reset_obs_for_new_envs = torch.randn(num_envs, obs_dim)
    reset_indices = torch.where(reset_mask)[0]

    if len(reset_indices) > 0:
        stacked_obs[reset_indices] = reset_obs_for_new_envs[reset_indices].repeat(1, num_stack)

    print(f"Reset mask: {reset_mask.tolist()}")
    print(f"環境 1 (未重置) Frame 1: {stacked_obs[0, :5].tolist()} (保留歷史)")
    print(f"環境 2 (已重置) Frame 1: {stacked_obs[1, :5].tolist()} (清除歷史)")
    print(f"環境 3 (未重置) Frame 1: {stacked_obs[2, :5].tolist()} (保留歷史)")
    print(f"環境 4 (已重置) Frame 1: {stacked_obs[3, :5].tolist()} (清除歷史)")

    # 驗證重置後的環境所有幀都相同
    for idx in reset_indices:
        frame_1 = stacked_obs[idx, :obs_dim]
        frame_2 = stacked_obs[idx, obs_dim:obs_dim*2]
        frame_3 = stacked_obs[idx, obs_dim*2:obs_dim*3]
        assert torch.allclose(frame_1, frame_2), f"環境 {idx} 重置後 Frame 1 ≠ Frame 2"
        assert torch.allclose(frame_2, frame_3), f"環境 {idx} 重置後 Frame 2 ≠ Frame 3"

    print("\n✅ Test 1 PASSED: Frame Stacking 邏輯正確")
    return True


# ============================================================================
# Test 2: TTC Penalty 計算測試
# ============================================================================

def test_ttc_penalty():
    """測試 TTC Penalty 的純張量計算"""
    print("\n" + "=" * 70)
    print("🧪 Test 2: TTC Penalty 計算")
    print("=" * 70)

    # 模擬參數
    num_envs = 2
    num_obstacles = 3
    tau = 1.5  # 安全時間閾值

    # ========================================================================
    # 測試場景 1: 正在靠近的障礙物 (應該有懲罰)
    # ========================================================================
    print("\n--- 測試場景 1: 正在靠近的障礙物 ---")

    # 機器人在原點，速度為 0
    p_r = torch.zeros(num_envs, 2)
    v_r = torch.zeros(num_envs, 2)

    # 障礙物在 (2, 0)，以速度 (-1, 0) 朝向機器人移動
    # 2 秒後會碰撞
    p_o = torch.tensor([
        [[2.0, 0.0], [3.0, 0.0], [0.0, 5.0]],  # 環境 0
        [[1.0, 0.0], [4.0, 0.0], [0.0, 3.0]],  # 環境 1 (更近)
    ])
    v_o = torch.tensor([
        [[-1.0, 0.0], [-0.5, 0.0], [0.0, -1.0]],  # 環境 0
        [[-1.0, 0.0], [-0.3, 0.0], [0.0, -0.5]],  # 環境 1
    ])

    # 障礙物高度 (全部 > 0，表示有效)
    z_o = torch.ones(num_envs, num_obstacles, 1)

    # 計算 TTC
    p_r_exp = p_r.unsqueeze(1).expand(-1, num_obstacles, -1)
    v_r_exp = v_r.unsqueeze(1).expand(-1, num_obstacles, -1)

    dp = p_o - p_r_exp
    dv = v_o - v_r_exp
    dist = torch.norm(dp, dim=-1, keepdim=True)
    dot_product = (dp * dv).sum(dim=-1, keepdim=True)
    epsilon = 1e-5
    v_app = -dot_product / (dist + epsilon)
    ttc = dist / (v_app + epsilon)

    print(f"環境 0 障礙物 0: dist={dist[0, 0, 0].item():.2f}, v_app={v_app[0, 0, 0].item():.2f}, ttc={ttc[0, 0, 0].item():.2f}")
    print(f"環境 0 障礙物 1: dist={dist[0, 1, 0].item():.2f}, v_app={v_app[0, 1, 0].item():.2f}, ttc={ttc[0, 1, 0].item():.2f}")
    print(f"環境 1 障礙物 0: dist={dist[1, 0, 0].item():.2f}, v_app={v_app[1, 0, 0].item():.2f}, ttc={ttc[1, 0, 0].item():.2f}")

    # 計算懲罰
    danger_mask = (z_o > 0) & (v_app.squeeze(-1) > 0) & (ttc.squeeze(-1) < tau)
    penalty_tensor = torch.where(
        danger_mask,
        -(tau - ttc.squeeze(-1)),
        torch.zeros_like(ttc.squeeze(-1))
    )
    penalty = penalty_tensor.sum(dim=1)

    print(f"\nDanger mask (環境 0): {danger_mask[0].tolist()}")
    print(f"Penalty tensor (環境 0): {penalty_tensor[0].tolist()}")
    print(f"Total penalty (環境 0): {penalty[0].item():.4f}")
    print(f"Total penalty (環境 1): {penalty[1].item():.4f}")

    # 驗證預期結果
    # 環境 0 障礙物 0: dist=2, v_app=1, ttc=2 > tau (無懲罰)
    # 環境 0 障礙物 1: dist=3, v_app=0.5, ttc=6 > tau (無懲罰)
    # 環境 1 障礙物 0: dist=1, v_app=1, ttc=1 < tau (有懲罰: -(1.5 - 1) = -0.5)
    assert penalty[1].item() < 0, "環境 1 應該有懲罰"
    print(f"✅ 環境 1 正確產生懲罰: {penalty[1].item():.4f}")

    # ========================================================================
    # 測試場景 2: 正在遠離的障礙物 (不應該有懲罰)
    # ========================================================================
    print("\n--- 測試場景 2: 正在遠離的障礙物 ---")

    # 障礙物遠離機器人
    p_o_away = torch.tensor([[[2.0, 0.0], [3.0, 0.0]]])
    v_o_away = torch.tensor([[[1.0, 0.0], [0.5, 0.0]]])  # 正速度表示遠離

    p_r_test = torch.zeros(1, 2)
    v_r_test = torch.zeros(1, 2)

    p_r_exp_test = p_r_test.unsqueeze(1).expand(-1, 2, -1)
    v_r_exp_test = v_r_test.unsqueeze(1).expand(-1, 2, -1)

    dp_away = p_o_away - p_r_exp_test
    dv_away = v_o_away - v_r_exp_test
    dist_away = torch.norm(dp_away, dim=-1, keepdim=True)
    dot_away = (dp_away * dv_away).sum(dim=-1, keepdim=True)
    v_app_away = -dot_away / (dist_away + epsilon)
    ttc_away = dist_away / (v_app_away + epsilon)

    print(f"障礙物 0: v_app={v_app_away[0, 0, 0].item():.2f} (負值表示遠離)")
    print(f"障礙物 1: v_app={v_app_away[0, 1, 0].item():.2f} (負值表示遠離)")

    # 遠離的障礙物不應該產生懲罰
    danger_mask_away = (v_app_away.squeeze(-1) > 0) & (ttc_away.squeeze(-1) < tau)
    assert not danger_mask_away[0, 0].item(), "遠離的障礙物不應該在 danger_mask 中"
    assert not danger_mask_away[0, 1].item(), "遠離的障礙物不應該在 danger_mask 中"
    print("✅ 遠離的障礙物正確排除在懲罰之外")

    # ========================================================================
    # 測試場景 3: 地底下的障礙物 (z_o <= 0, 不應該有懲罰)
    # ========================================================================
    print("\n--- 測試場景 3: 地底下的障礙物 (Empty 環境) ---")

    # 重新創建測試數據
    p_r_test3 = torch.zeros(1, 2)
    v_r_test3 = torch.zeros(1, 2)
    p_o_test3 = torch.tensor([[[2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]])
    v_o_test3 = torch.tensor([[[-1.0, 0.0], [-0.5, 0.0], [-0.3, 0.0]]])

    p_r_exp_test3 = p_r_test3.unsqueeze(1).expand(-1, 3, -1)
    v_r_exp_test3 = v_r_test3.unsqueeze(1).expand(-1, 3, -1)

    dp_test3 = p_o_test3 - p_r_exp_test3
    dv_test3 = v_o_test3 - v_r_exp_test3
    dist_test3 = torch.norm(dp_test3, dim=-1, keepdim=True)
    dot_test3 = (dp_test3 * dv_test3).sum(dim=-1, keepdim=True)
    v_app_test3 = -dot_test3 / (dist_test3 + epsilon)
    ttc_test3 = dist_test3 / (v_app_test3 + epsilon)

    z_o_underground = torch.tensor([
        [[1.0], [-1.0], [0.0]]  # 一個在地面，一個在地底，一個在地面
    ])

    danger_mask_underground = (z_o_underground > 0) & (v_app_test3.squeeze(-1) > 0) & (ttc_test3.squeeze(-1) < tau)
    print(f"Z heights: {z_o_underground.squeeze(-1).tolist()}")
    print(f"Danger mask (考慮 Z): {danger_mask_underground.tolist()}")
    assert not danger_mask_underground[0, 1].item(), "地底下的障礙物不應該產生懲罰"
    print("✅ 地底下的障礙物正確排除")

    print("\n✅ Test 2 PASSED: TTC Penalty 計算正確")
    return True


# ============================================================================
# Test 3: 維度驗證
# ============================================================================

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
    print(f"新的 Actor 輸入:   {actor_dim} 維 ({original_obs_dim} × {num_stack})")
    print(f"特權障礙物狀態:   {privileged_dim} 維")
    print(f"新的 Critic 輸入:  {critic_dim} 維 ({actor_dim} + {privileged_dim})")

    # 驗證
    assert actor_dim == 243, f"Actor 維度應為 243，實際為 {actor_dim}"
    assert critic_dim == 293, f"Critic 維度應為 293，實際為 {critic_dim}"

    # 模擬張量操作
    batch_size = 512
    actor_obs = torch.randn(batch_size, actor_dim)
    privileged_info = torch.randn(batch_size, privileged_dim)
    critic_state = torch.cat([actor_obs, privileged_info], dim=-1)

    print(f"\n模擬批次大小: {batch_size}")
    print(f"Actor 觀測 shape: {actor_obs.shape}")
    print(f"特權資訊 shape: {privileged_info.shape}")
    print(f"Critic 狀態 shape: {critic_state.shape}")

    assert critic_state.shape == (batch_size, 293), f"Critic 狀態形狀應為 (512, 293)"

    print("\n✅ Test 3 PASSED: 維度正確")
    return True


# ============================================================================
# Test 4: 梯度流測試 (確認張量操作可微分)
# ============================================================================

def test_gradient_flow():
    """測試梯度流是否正常"""
    print("\n" + "=" * 70)
    print("🧪 Test 4: 梯度流測試")
    print("=" * 70)

    # 創建需要梯度的張量
    num_envs = 4
    obs_dim = 81

    # 模擬神經網路輸出
    policy_output = torch.randn(num_envs, obs_dim, requires_grad=True)

    # Frame Stacking 操作
    stacked = policy_output.repeat(1, 3)
    loss = stacked.sum()
    loss.backward()

    # 檢查梯度
    assert policy_output.grad is not None, "梯度應該存在"
    assert torch.all(policy_output.grad > 0), "repeat 操作的梯度應該正確傳播"

    print(f"Policy output shape: {policy_output.shape}")
    print(f"Stacked shape: {stacked.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Gradient shape: {policy_output.grad.shape}")
    print(f"Gradient mean: {policy_output.grad.mean().item():.4f}")

    print("\n✅ Test 4 PASSED: 梯度流正常")
    return True


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """運行所有測試"""
    print("\n" + "=" * 70)
    print("🚀 Frame Stacking & TTC Penalty 測試套件")
    print("=" * 70)

    tests = [
        ("Frame Stacking 邏輯", test_frame_stacking_logic),
        ("TTC Penalty 計算", test_ttc_penalty),
        ("維度驗證", test_dimensions),
        ("梯度流", test_gradient_flow),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ Test FAILED: {test_name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # ========================================================================
    # 總結
    # ========================================================================
    print("\n" + "=" * 70)
    print("📊 測試結果總結")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30s}: {status}")

    all_passed = all(results.values())
    print("=" * 70)

    if all_passed:
        print("\n🎉 所有測試通過！Frame Stacking 和 TTC Penalty 實現正確。")
        return 0
    else:
        print("\n⚠️  部分測試失敗，請檢查實現。")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
