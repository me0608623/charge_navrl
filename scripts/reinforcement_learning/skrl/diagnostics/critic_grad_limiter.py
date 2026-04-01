"""動態 Critic 梯度限制器

根據 Policy 梯度的 EMA 基線，限制 Critic 梯度的最大幅度。
防止異常環境產生的極大 critic loss 破壞已學好的參數，
同時保留正常學習（發現新策略時的大 loss 更新）。

插入點: scaler.unscale_() 之後、clip_grad_norm_() 之前。
"""

from __future__ import annotations

import torch.nn as nn
from .module_entropy import compute_grad_norm


class CriticGradLimiter:
    """根據 actor 梯度 EMA 動態限制 critic 梯度幅度。

    演算法:
        每個 minibatch (unscale_ 後):
          1. actor_gn = grad_norm(policy)
          2. critic_gn = grad_norm(value)
          3. ema_actor = α * actor_gn + (1-α) * ema_actor
          4. if warmup 完成 and critic_gn > k * ema_actor:
               scale critic gradients by (k * ema_actor) / critic_gn
    """

    def __init__(
        self,
        policy_model: nn.Module,
        value_model: nn.Module,
        k: float = 2.0,
        ema_alpha: float = 0.01,
        warmup_steps: int = 100,
    ):
        self._policy = policy_model
        self._value = value_model
        self._k = k
        self._alpha = ema_alpha
        self._warmup = warmup_steps
        self._step_count = 0
        self._ema_actor_gn = 0.0

        # Metrics accumulators (flush per rollout)
        self._clip_count = 0
        self._total_count = 0
        self._clipped_ratios: list[float] = []

    def step(self):
        """每 minibatch 呼叫一次 (unscale_ 後、clip_grad_norm_ 前)。"""
        actor_gn = compute_grad_norm(self._policy)
        critic_gn = compute_grad_norm(self._value)

        self._step_count += 1
        self._total_count += 1

        # EMA 更新
        if self._step_count == 1:
            self._ema_actor_gn = actor_gn  # 首步直接設定
        else:
            self._ema_actor_gn = (
                self._alpha * actor_gn + (1 - self._alpha) * self._ema_actor_gn
            )

        # Warmup 期間不限制
        if self._step_count <= self._warmup or self._ema_actor_gn < 1e-8:
            return

        critic_max = self._k * self._ema_actor_gn
        if critic_gn > critic_max and critic_gn > 1e-8:
            scale = critic_max / critic_gn
            for p in self._value.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(scale)
            self._clip_count += 1
            self._clipped_ratios.append(critic_gn / self._ema_actor_gn)

    def flush(self) -> dict:
        """回傳指標並重置累計器。每 rollout flush 呼叫。"""
        total = max(self._total_count, 1)
        metrics = {
            "train/critic_grad_clip_rate": self._clip_count / total,
            "train/critic_grad_limit_ema_actor": self._ema_actor_gn,
        }
        if self._clipped_ratios:
            metrics["train/critic_grad_clip_max_ratio"] = max(self._clipped_ratios)

        self._clip_count = 0
        self._total_count = 0
        self._clipped_ratios.clear()
        return metrics
