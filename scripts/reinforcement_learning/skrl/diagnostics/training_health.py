"""Training Health Monitor — 觀測品質 + 學習健康度診斷

三層指標體系：
  Tier 1 (每 rollout): explained_variance, approx_kl, clip_fraction, dead_neuron
  Tier 2 (每 N rollout): erank, dormant_ratio, weight_magnitude, feature_correlation
  Tier 3 (可選): gradient_saliency, obs_erank

數學基礎：
  - Explained Variance: 1 - Var(R - V) / Var(R)  (Sutton & Barto)
  - Effective Rank: exp(-Σ p_k log p_k), p_k = σ_k/Σσ  (Roy & Vetterli, 2007)
  - Dormant Neuron: E[|h_i|] / mean_layer < τ  (Sokar et al., ICML 2023)
  - Dead Neuron: activation == 0 for all samples  (Dohare et al., Nature 2024)

Usage:
    monitor = TrainingHealthMonitor(policy, value, compute_tier2_every=5)

    # In PPO mini-batch loop (after forward pass):
    monitor.on_ppo_batch(returns, values, old_lp, new_lp, ratio_clip)

    # At rollout flush:
    metrics = monitor.flush()
    wandb.log(metrics)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class TrainingHealthMonitor:
    """Tier 1-2 訓練健康診斷，不修改 SKRL 源碼。"""

    def __init__(
        self,
        policy_model: nn.Module,
        value_model: nn.Module,
        compute_tier2_every: int = 5,
        dead_neuron_sample_size: int = 1024,
    ):
        self._policy = policy_model
        self._value = value_model
        self._tier2_every = compute_tier2_every
        self._dead_sample_size = dead_neuron_sample_size
        self._flush_count = 0

        # Tier 1 accumulators (per mini-batch)
        self._explained_variances: list[float] = []
        self._approx_kls: list[float] = []
        self._clip_fractions: list[float] = []
        self._td_errors_mean: list[float] = []
        self._td_errors_std: list[float] = []

        # 最近一批 states 用於 dead neuron / erank 計算
        self._last_states: Optional[torch.Tensor] = None

    # ──────────────────────────────────────────────────────────────
    # Per mini-batch collection (Tier 1)
    # ──────────────────────────────────────────────────────────────

    def on_ppo_batch(
        self,
        sampled_returns: torch.Tensor,
        predicted_values: torch.Tensor,
        old_log_prob: torch.Tensor,
        new_log_prob: torch.Tensor,
        ratio_clip: float = 0.2,
        sampled_states: Optional[torch.Tensor] = None,
    ) -> None:
        """每個 PPO mini-batch 呼叫，累積 Tier 1 指標。

        所有計算在 no_grad 下進行，不影響訓練。
        """
        with torch.no_grad():
            # --- Explained Variance ---
            returns = sampled_returns.view(-1)
            values = predicted_values.view(-1)
            var_returns = returns.var()
            if var_returns > 1e-8:
                ev = 1.0 - (returns - values).var() / var_returns
                self._explained_variances.append(ev.clamp(-1.0, 1.0).item())

            # --- TD Error Stats ---
            td_error = returns - values
            self._td_errors_mean.append(td_error.mean().item())
            self._td_errors_std.append(td_error.std().item())

            # --- Approx KL (k3 estimator) ---
            log_ratio = new_log_prob.view(-1) - old_log_prob.view(-1)
            kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
            self._approx_kls.append(kl.clamp(0.0, 10.0).item())

            # --- Clip Fraction ---
            ratio = torch.exp(log_ratio)
            clip_frac = ((ratio - 1.0).abs() > ratio_clip).float().mean()
            self._clip_fractions.append(clip_frac.item())

            # 保留一批 states 用於 Tier 2
            if sampled_states is not None:
                n = min(self._dead_sample_size, sampled_states.shape[0])
                self._last_states = sampled_states[:n].detach()

    # ──────────────────────────────────────────────────────────────
    # Flush (per rollout)
    # ──────────────────────────────────────────────────────────────

    def flush(self) -> dict[str, float]:
        """回傳所有累積指標，清空 buffer。"""
        self._flush_count += 1
        m: dict[str, float] = {}

        # Tier 1: 每次都算
        if self._explained_variances:
            m["diag/explained_variance"] = float(np.mean(self._explained_variances))
        if self._approx_kls:
            m["diag/approx_kl"] = float(np.mean(self._approx_kls))
        if self._clip_fractions:
            m["diag/clip_fraction"] = float(np.mean(self._clip_fractions))
        if self._td_errors_mean:
            m["diag/td_error_mean"] = float(np.mean(self._td_errors_mean))
            m["diag/td_error_std"] = float(np.mean(self._td_errors_std))

        # Tier 1: Dead neuron (需要 forward pass)
        if self._last_states is not None:
            m["diag/dead_neuron_actor"] = _compute_dead_neuron_ratio(
                self._policy, self._last_states
            )
            m["diag/dead_neuron_critic"] = _compute_dead_neuron_ratio(
                self._value, self._last_states
            )

        # Tier 1: Weight magnitude (便宜)
        m["diag/weight_mag_actor"] = _compute_weight_magnitude(self._policy)
        m["diag/weight_mag_critic"] = _compute_weight_magnitude(self._value)

        # Tier 2: 每 N 次
        if self._flush_count % self._tier2_every == 0:
            m["diag/erank_actor"] = _compute_erank(self._policy)
            m["diag/erank_critic"] = _compute_erank(self._value)

            if self._last_states is not None:
                m["diag/dormant_ratio_actor"] = _compute_dormant_ratio(
                    self._policy, self._last_states
                )

        self._clear()
        return m

    def _clear(self):
        self._explained_variances.clear()
        self._approx_kls.clear()
        self._clip_fractions.clear()
        self._td_errors_mean.clear()
        self._td_errors_std.clear()
        self._last_states = None


# ======================================================================
# Static computation functions
# ======================================================================

def _compute_weight_magnitude(model: nn.Module) -> float:
    """所有參數的平均絕對值 — 可塑性衰退指標。

    若 weight_mag 持續上升 → 網路可能正在失去可塑性。
    """
    total = 0.0
    count = 0
    for p in model.parameters():
        if p.requires_grad:
            total += p.data.abs().sum().item()
            count += p.data.numel()
    return total / max(count, 1)


def _compute_dead_neuron_ratio(model: nn.Module, sample_states: torch.Tensor) -> float:
    """ReLU 後對所有 sample 都輸出 0 的神經元比例。

    Dead ratio > 5% → 警告；> 20% → 嚴重可塑性損失。
    (Dohare et al., Nature 2024)
    """
    activations: list[torch.Tensor] = []
    hooks = []

    def _hook(module, input, output):
        if isinstance(output, torch.Tensor) and output.dim() >= 2:
            activations.append(output.detach())

    # 註冊 hook 在所有 ReLU 層
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(_hook))

    try:
        with torch.no_grad():
            if hasattr(model, 'act'):
                model.act({"states": sample_states}, role="policy")
            elif hasattr(model, 'forward'):
                model(sample_states)
    except Exception:
        pass
    finally:
        for h in hooks:
            h.remove()

    if not activations:
        return 0.0

    dead_count = 0
    total_count = 0
    for act in activations:
        # act: [B, D] or [B, D, L]
        if act.dim() == 2:
            # [B, D] → dead if all B samples are 0 for neuron d
            is_dead = (act.abs() < 1e-8).all(dim=0)  # [D]
            dead_count += is_dead.sum().item()
            total_count += is_dead.numel()
        elif act.dim() == 3:
            # [B, D, L] → flatten spatial, check per channel
            is_dead = (act.abs() < 1e-8).all(dim=0).all(dim=-1)  # [D]
            dead_count += is_dead.sum().item()
            total_count += is_dead.numel()

    return dead_count / max(total_count, 1)


def _compute_erank(model: nn.Module) -> float:
    """Linear 層 weight matrix 的 effective rank。

    eRank = exp(H(p)), p_k = σ_k / Σσ
    eRank 下降 → 特徵坍塌 (Kumar et al., ICLR 2021)

    注意: 對 scalar-output 模型 (如 Critic)，最後一層 [1, N] 永遠 rank=1，
    改用倒數第二層以獲得有意義的量測。
    """
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linear_layers:
        return 0.0

    target = linear_layers[-1]
    # scalar-output (e.g. critic Linear(32,1)) → weight [1,32] → SVD 只有 1 奇異值
    # 改用 penultimate layer 取得有意義的 eRank
    if target.weight.shape[0] == 1 and len(linear_layers) >= 2:
        target = linear_layers[-2]

    W = target.weight.data  # [out, in]
    try:
        S = torch.linalg.svdvals(W.float())  # [min(out, in)]
        S = S[S > 1e-10]
        if S.numel() == 0:
            return 0.0
        p = S / S.sum()
        entropy = -(p * torch.log(p)).sum().item()
        return math.exp(entropy)
    except Exception:
        return 0.0


def _compute_dormant_ratio(
    model: nn.Module,
    sample_states: torch.Tensor,
    tau: float = 0.1,
) -> float:
    """休眠神經元比例 — 平均活化值 < τ × 全層平均。

    Dormant ratio > 10% → 警告 (Sokar et al., ICML 2023)
    """
    activations: list[torch.Tensor] = []
    hooks = []

    def _hook(module, input, output):
        if isinstance(output, torch.Tensor) and output.dim() >= 2:
            activations.append(output.detach())

    for module in model.modules():
        if isinstance(module, (nn.ReLU, nn.LayerNorm)):
            hooks.append(module.register_forward_hook(_hook))

    try:
        with torch.no_grad():
            if hasattr(model, 'act'):
                model.act({"states": sample_states}, role="policy")
            elif hasattr(model, 'forward'):
                model(sample_states)
    except Exception:
        pass
    finally:
        for h in hooks:
            h.remove()

    if not activations:
        return 0.0

    dormant_count = 0
    total_count = 0
    for act in activations:
        if act.dim() == 2:
            # [B, D]
            mean_per_neuron = act.abs().mean(dim=0)  # [D]
            layer_mean = mean_per_neuron.mean()
            is_dormant = mean_per_neuron < tau * layer_mean
            dormant_count += is_dormant.sum().item()
            total_count += is_dormant.numel()

    return dormant_count / max(total_count, 1)


__all__ = ["TrainingHealthMonitor"]
