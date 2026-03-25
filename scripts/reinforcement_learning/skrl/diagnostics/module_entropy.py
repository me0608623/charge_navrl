"""Module Entropy 訓練健康診斷模組

依使用者提供的論文描述實作。

公式：
    module_entropy = log10( (actor_grad_norm + eps) / (critic_grad_norm + eps) )

解讀：
    > 1    : Policy 更新過度 / Critic 遇到瓶頸
    ≈ 0    : Policy 與 Critic 處於最佳平衡訓練狀態
    ≈ -2   : Policy 已接近上限，Critic 仍在學習
    < -3   : 只有 Critic 在訓練，Policy 幾乎沒有有效更新

架構適配說明（SKRL PPO）：
    本專案使用 SKRL PPO，policy 和 value 共用一個 Adam optimizer，
    loss = policy_loss + entropy_loss + value_loss 合併 backward，
    只有一次 optimizer.step()。因此：
    - 無法用 parameter delta (θ_{t+1} - θ_t) 分開測量更新幅度
    - 改用 gradient norm proxy：在 backward 之後、step 之前，
      分別收集 policy 和 value 各自參數的梯度 L2 範數
    - 這是近似做法，不是真正的 parameter delta
    - gradient norm 受 grad_norm_clip 影響（本專案 clip=1.0），
      但 clip 是對 chain(policy, value) 整體做的，
      個別 grad_norm 仍然反映各自的梯度量級比例
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger("module_entropy")

# 數值穩定常數
EPS = 1e-12
# 雙方梯度都小於此值視為 frozen
FROZEN_THRESHOLD = 1e-10


def compute_grad_norm(model: nn.Module, only_requires_grad: bool = True) -> float:
    """計算模型所有參數的梯度 L2 範數。

    注意：這是 gradient norm proxy，不是真正的 parameter delta。
    在 loss.backward() 之後、optimizer.step() 之前呼叫。

    Args:
        model: PyTorch 模型（policy 或 value network）
        only_requires_grad: 只計算需要梯度的參數

    Returns:
        梯度 L2 範數 sqrt(sum(||g_i||^2))，若無梯度則回傳 0.0
    """
    total_norm_sq = 0.0
    param_count = 0
    for p in model.parameters():
        if only_requires_grad and not p.requires_grad:
            continue
        if p.grad is not None:
            total_norm_sq += p.grad.data.norm(2).item() ** 2
            param_count += 1

    if param_count == 0:
        return 0.0
    return math.sqrt(total_norm_sq)


def compute_module_entropy(
    actor_magnitude: float,
    critic_magnitude: float,
    eps: float = EPS,
) -> tuple[float, float, str]:
    """計算 module entropy。

    依使用者提供的論文描述實作：
        module_entropy = log10( (actor_magnitude + eps) / (critic_magnitude + eps) )

    Args:
        actor_magnitude: actor (policy) 的梯度 L2 範數
        critic_magnitude: critic (value) 的梯度 L2 範數
        eps: 數值穩定常數

    Returns:
        (module_entropy, ratio, validity_note)
        - module_entropy: log10 比值
        - ratio: actor/critic 原始比值
        - validity_note: "ok" / "both_frozen" / "invalid_nan" / "invalid_inf"
    """
    # 數值安全檢查
    if math.isnan(actor_magnitude) or math.isnan(critic_magnitude):
        logger.warning(
            f"[module_entropy] NaN detected: actor={actor_magnitude}, critic={critic_magnitude}"
        )
        return 0.0, 0.0, "invalid_nan"

    if math.isinf(actor_magnitude) or math.isinf(critic_magnitude):
        logger.warning(
            f"[module_entropy] Inf detected: actor={actor_magnitude}, critic={critic_magnitude}"
        )
        return 0.0, 0.0, "invalid_inf"

    # 雙方都接近 0 → frozen
    if actor_magnitude < FROZEN_THRESHOLD and critic_magnitude < FROZEN_THRESHOLD:
        return 0.0, 0.0, "both_frozen"

    # 計算 ratio 和 module_entropy
    ratio = (actor_magnitude + eps) / (critic_magnitude + eps)
    # clamp ratio 避免 log10(0)
    ratio_clamped = max(ratio, eps)
    module_entropy = math.log10(ratio_clamped)

    return module_entropy, ratio, "ok"


def classify_module_entropy(module_entropy: float) -> tuple[str, int, str]:
    """根據 module entropy 值分類訓練狀態。

    依使用者提供的論文描述實作。

    Args:
        module_entropy: module entropy 值

    Returns:
        (state_name, state_code, suggestion)
    """
    if module_entropy > 1.0:
        return (
            "policy_over_updating_or_critic_bottleneck",
            2,
            "檢查 policy learning rate 是否過高；"
            "若在後期持續發生，檢查 critic capacity / reward scale / target update",
        )
    elif -0.5 <= module_entropy <= 0.5:
        return (
            "balanced_training",
            0,
            "維持目前訓練設定",
        )
    elif -2.5 <= module_entropy <= -1.5:
        return (
            "policy_saturated_need_more_exploration",
            -2,
            "檢查是否需要提高 exploration；"
            "若是 SAC，檢查 entropy coefficient / target entropy / alpha autotune；"
            "PPO 可嘗試提高 entropy_loss_scale",
        )
    elif module_entropy < -3.0:
        return (
            "critic_only_training_possible_env_bug",
            -3,
            "檢查環境 reward / done / action scaling / detach / "
            "policy loss 是否正確連到 critic",
        )
    else:
        return (
            "intermediate_unclassified",
            -1,
            "訓練正常過渡中，持續觀察",
        )


class ModuleEntropyMonitor:
    """Module Entropy 訓練監控器。

    使用方式：
        monitor = ModuleEntropyMonitor(policy_model, value_model)

        # 在 loss.backward() 之後、optimizer.step() 之前呼叫：
        monitor.step()       # 每次 mini-batch 呼叫，累積數據

        # 在每次 rollout flush 時呼叫：
        result = monitor.flush()  # 回傳平均值 dict，印出 console log
    """

    # 狀態碼 → 顯示符號
    _STATE_ICONS = {
        2:  "!!",    # policy 過度更新
        0:  "OK",    # 平衡
        -1: "..",    # 過渡中
        -2: "??",    # policy 飽和
        -3: "XX",    # critic only
    }

    def __init__(
        self,
        policy_model: nn.Module,
        value_model: nn.Module,
        console_every_n_flush: int = 5,
    ):
        """初始化監控器。

        Args:
            policy_model: policy (actor) 網路
            value_model: value (critic) 網路
            console_every_n_flush: 每 N 次 flush 印一次 console log
        """
        self._policy = policy_model
        self._value = value_model
        self._console_every_n_flush = console_every_n_flush
        self._flush_count = 0

        # mini-batch 累積緩衝區（在 flush 時取平均）
        self._acc_me: list[float] = []
        self._acc_actor_gn: list[float] = []
        self._acc_critic_gn: list[float] = []
        self._acc_ratio: list[float] = []

        # 上一次 flush 的狀態（持續追蹤趨勢）
        self._last_me: float = 0.0
        self._last_state_code: int = 0
        self._consecutive_warn: int = 0

    def step(self) -> None:
        """每次 mini-batch 更新時呼叫，累積 gradient norm 數據。

        應在 loss.backward() 之後、optimizer.step() 之前。
        不印任何東西——數據累積到 flush() 時才輸出。
        """
        actor_gn = compute_grad_norm(self._policy)
        critic_gn = compute_grad_norm(self._value)
        me, ratio, validity = compute_module_entropy(actor_gn, critic_gn)

        if validity == "ok":
            self._acc_me.append(me)
            self._acc_actor_gn.append(actor_gn)
            self._acc_critic_gn.append(critic_gn)
            self._acc_ratio.append(ratio)

    def flush(self) -> dict[str, float]:
        """每次 rollout 結束時呼叫：取平均、分類、印 log、回傳 WandB dict。

        Returns:
            dict 包含所有監控指標，可直接送入 WandB tracking_data_snapshot。
        """
        self._flush_count += 1

        # 取平均
        if self._acc_me:
            n = len(self._acc_me)
            avg_me = sum(self._acc_me) / n
            avg_actor = sum(self._acc_actor_gn) / n
            avg_critic = sum(self._acc_critic_gn) / n
            avg_ratio = sum(self._acc_ratio) / n
        else:
            avg_me = 0.0
            avg_actor = 0.0
            avg_critic = 0.0
            avg_ratio = 0.0

        # 清空累積
        self._acc_me.clear()
        self._acc_actor_gn.clear()
        self._acc_critic_gn.clear()
        self._acc_ratio.clear()

        # 分類
        state_name, state_code, suggestion = classify_module_entropy(avg_me)
        icon = self._STATE_ICONS.get(state_code, "??")
        self._last_me = avg_me
        self._last_state_code = state_code

        # Console log（每 N 次 flush 印一次）
        should_print = (self._flush_count % self._console_every_n_flush == 0)

        # 異常狀態連續出現 → 強制印告警
        if state_code in (2, -2, -3):
            self._consecutive_warn += 1
        else:
            self._consecutive_warn = 0

        if should_print:
            print(
                f"[ME] [{icon}] entropy={avg_me:+.3f} | "
                f"actor={avg_actor:.3e} critic={avg_critic:.3e} | "
                f"ratio={avg_ratio:.2f} | {state_name}",
                flush=True,
            )

        # 異常持續告警（連續 3+ 次）
        if self._consecutive_warn >= 3 and self._consecutive_warn % 3 == 0:
            print(
                f"[ME 告警] {icon} {state_name} 已連續 {self._consecutive_warn} 次 | "
                f"ME={avg_me:+.3f} | 建議: {suggestion}",
                flush=True,
            )

        return {
            "train/module_entropy": avg_me,
            "train/actor_grad_norm": avg_actor,
            "train/critic_grad_norm": avg_critic,
            "train/update_ratio_actor_over_critic": avg_ratio,
            "train/module_entropy_state_code": float(state_code),
        }
