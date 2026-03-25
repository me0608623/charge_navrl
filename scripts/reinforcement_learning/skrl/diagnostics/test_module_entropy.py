"""Module Entropy 單元測試

驗證 4 個情境：
  a. actor >> critic → module_entropy > 1
  b. actor ≈ critic → module_entropy ≈ 0
  c. actor << critic → module_entropy 顯著為負
  d. 雙方都趨近 0 → 不會數值爆炸
"""

import math
import torch
import torch.nn as nn

from module_entropy import (
    compute_grad_norm,
    compute_module_entropy,
    classify_module_entropy,
    ModuleEntropyMonitor,
)


def _make_model_with_grads(grad_scale: float) -> nn.Module:
    """建立一個簡單模型並設定假梯度。"""
    model = nn.Linear(10, 5)
    # 模擬 backward 後的梯度
    for p in model.parameters():
        p.grad = torch.randn_like(p) * grad_scale
    return model


def test_actor_much_larger():
    """情境 a: actor 更新 >> critic 更新 → module_entropy > 1"""
    actor_gn = 10.0   # 大梯度
    critic_gn = 0.1    # 小梯度
    me, ratio, validity = compute_module_entropy(actor_gn, critic_gn)
    assert validity == "ok"
    assert me > 1.0, f"Expected > 1, got {me:.3f}"
    state, code, _ = classify_module_entropy(me)
    assert code == 2, f"Expected code 2, got {code}"
    print(f"[PASS] actor >> critic: ME={me:+.3f}, state={state}")


def test_balanced():
    """情境 b: actor ≈ critic → module_entropy ≈ 0"""
    actor_gn = 1.05
    critic_gn = 0.95
    me, ratio, validity = compute_module_entropy(actor_gn, critic_gn)
    assert validity == "ok"
    assert -0.5 <= me <= 0.5, f"Expected ≈0, got {me:.3f}"
    state, code, _ = classify_module_entropy(me)
    assert code == 0, f"Expected code 0, got {code}"
    print(f"[PASS] balanced: ME={me:+.3f}, state={state}")


def test_critic_much_larger():
    """情境 c: actor << critic → module_entropy 顯著為負"""
    actor_gn = 0.001
    critic_gn = 100.0
    me, ratio, validity = compute_module_entropy(actor_gn, critic_gn)
    assert validity == "ok"
    assert me < -3.0, f"Expected < -3, got {me:.3f}"
    state, code, _ = classify_module_entropy(me)
    assert code == -3, f"Expected code -3, got {code}"
    print(f"[PASS] critic >> actor: ME={me:+.3f}, state={state}")


def test_both_near_zero():
    """情境 d: 雙方都趨近 0 → 不會數值爆炸"""
    actor_gn = 1e-15
    critic_gn = 1e-15
    me, ratio, validity = compute_module_entropy(actor_gn, critic_gn)
    assert validity == "both_frozen"
    assert me == 0.0
    assert not math.isnan(me)
    assert not math.isinf(me)
    print(f"[PASS] both frozen: ME={me:+.3f}, validity={validity}")


def test_nan_handling():
    """NaN 輸入不爆炸"""
    me, ratio, validity = compute_module_entropy(float('nan'), 1.0)
    assert validity == "invalid_nan"
    assert me == 0.0
    print(f"[PASS] NaN handling: validity={validity}")


def test_inf_handling():
    """Inf 輸入不爆炸"""
    me, ratio, validity = compute_module_entropy(float('inf'), 1.0)
    assert validity == "invalid_inf"
    assert me == 0.0
    print(f"[PASS] Inf handling: validity={validity}")


def test_grad_norm_with_real_model():
    """使用真實 PyTorch 模型驗證 compute_grad_norm"""
    model = _make_model_with_grads(grad_scale=1.0)
    gn = compute_grad_norm(model)
    assert gn > 0.0, f"Expected > 0, got {gn}"
    assert not math.isnan(gn)
    print(f"[PASS] real model grad_norm: {gn:.4f}")


def test_monitor_integration():
    """ModuleEntropyMonitor 整合測試（step + flush 分離）"""
    policy = _make_model_with_grads(1.0)
    value = _make_model_with_grads(1.0)
    monitor = ModuleEntropyMonitor(policy, value, console_every_n_flush=1)

    # 模擬 8 epochs × 8 mini-batches = 64 次 step
    for _ in range(64):
        monitor.step()

    # flush 取平均
    result = monitor.flush()
    assert "train/module_entropy" in result
    assert "train/actor_grad_norm" in result
    assert "train/critic_grad_norm" in result
    assert not math.isnan(result["train/module_entropy"])
    print(f"[PASS] monitor integration (step+flush): {result}")


def test_classify_ranges():
    """驗證所有分類區間"""
    cases = [
        (2.0, 2),       # > 1 → code 2
        (0.0, 0),       # [-0.5, 0.5] → code 0
        (-2.0, -2),     # [-2.5, -1.5] → code -2
        (-4.0, -3),     # < -3 → code -3
        (0.8, -1),      # 未分類區間 → code -1
        (-1.0, -1),     # 未分類區間 → code -1
    ]
    for me_val, expected_code in cases:
        _, code, _ = classify_module_entropy(me_val)
        assert code == expected_code, (
            f"ME={me_val}: expected code={expected_code}, got {code}"
        )
    print("[PASS] all classification ranges correct")


if __name__ == "__main__":
    test_actor_much_larger()
    test_balanced()
    test_critic_much_larger()
    test_both_near_zero()
    test_nan_handling()
    test_inf_handling()
    test_grad_norm_with_real_model()
    test_monitor_integration()
    test_classify_ranges()
    print("\n=== ALL TESTS PASSED ===")
