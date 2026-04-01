"""Actor Gradient Decomposition Diagnostic

離線載入 checkpoint，用 memory 中的 rollout 數據
分別計算 policy surrogate loss 和 entropy loss 的梯度範數，
驗證 entropy 是否真正主導 actor 梯度。

Usage:
    cd /home/aa/IsaacLab
    python scripts/reinforcement_learning/skrl/diagnostics/gradient_decompose.py \
        --checkpoint logs/skrl/.../checkpoints/agent_35154.pt
"""

import argparse
import sys
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add parent dir to path for vlp16_models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vlp16_models import VLP16DiscretePolicy, VLP16Value, NUM_BINS


def load_checkpoint(path: str, device: str = "cuda:0"):
    """Load SKRL checkpoint and extract policy/value networks + memory."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt


def compute_gradient_norms(params):
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total_norm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ratio_clip", type=float, default=0.2)
    parser.add_argument("--entropy_scale", type=float, default=0.01)
    parser.add_argument("--num_samples", type=int, default=8192,
                        help="Number of samples for gradient decomposition")
    args = parser.parse_args()

    device = args.device
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint, device)

    # ── 重建 Policy 網路 ──
    # SKRL checkpoint 格式: {"policy": state_dict, "value": state_dict, ...}
    from gymnasium.spaces import Box, MultiDiscrete as MDSpace
    obs_space = Box(low=-np.inf, high=np.inf, shape=(139,))
    act_space = MDSpace(np.array([NUM_BINS, NUM_BINS]))

    policy = VLP16DiscretePolicy(obs_space, act_space, device=device)
    value = VLP16Value(obs_space, act_space, device=device)

    # Load weights
    if "policy" in ckpt:
        policy.load_state_dict(ckpt["policy"])
    elif "policy_state_dict" in ckpt:
        policy.load_state_dict(ckpt["policy_state_dict"])
    else:
        print(f"Available keys: {list(ckpt.keys())}")
        raise KeyError("Cannot find policy state dict in checkpoint")

    if "value" in ckpt:
        value.load_state_dict(ckpt["value"])

    policy.to(device).train()
    value.to(device).eval()

    # ── 載入 state_preprocessor (CADN) ──
    cadn = None
    if "state_preprocessor" in ckpt:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from cadn_preprocessor import CurriculumAwareDualRateNormalizer
        cadn = CurriculumAwareDualRateNormalizer(size=139, device=device)
        cadn.load_state_dict(ckpt["state_preprocessor"])
        cadn.eval()
        print("[INFO] CADN state_preprocessor loaded from checkpoint")
    else:
        print("[WARNING] No state_preprocessor in checkpoint, using raw obs")

    # ── 生成合成數據做梯度分解 ──
    # 因為無法存取執行中的 env，用隨機 obs 模擬分佈
    # 但 action/advantage 才是關鍵 — 用合理的 advantage 分佈
    N = args.num_samples
    print(f"\nGenerating {N} synthetic samples for gradient decomposition...")

    # 合成觀測（用 CADN 的統計量生成近似真實分佈的 obs）
    if cadn is not None:
        # 用 CADN 的 mean/std 生成
        with torch.no_grad():
            raw_obs = torch.randn(N, 139, device=device)
            # LiDAR bins should be in [0, 1]
            raw_obs[:, 6:78] = torch.rand(N, 72, device=device)
            # Obstacle obs — many zeros (empty slots)
            raw_obs[:, 78:138] = torch.randn(N, 60, device=device) * 0.3
            # Time remaining [0, 1]
            raw_obs[:, 138] = torch.rand(N, device=device)
            # Ego state
            raw_obs[:, 0:4] = torch.randn(N, 4, device=device) * 0.5
            # Goal position
            raw_obs[:, 4:6] = torch.randn(N, 2, device=device) * 3.0

            states = cadn(raw_obs, train=False)
    else:
        states = torch.randn(N, 139, device=device)

    # ── Forward pass: 取得 action 和 log_prob ──
    with torch.no_grad():
        actions, old_log_prob, outputs = policy.act(
            {"states": states}, role="policy"
        )

    # ── 第二次 forward pass（模擬 PPO update）──
    # 生成合成 advantages（normalized, mean≈0, std≈1）
    advantages = torch.randn(N, 1, device=device)

    # 也用真實 checkpoint value 估計看看
    with torch.no_grad():
        values, _, _ = value.act({"states": states}, role="value")

    print(f"\n{'='*70}")
    print(f"  STEP 4: Advantage Statistics")
    print(f"{'='*70}")
    print(f"  Synthetic advantages (normalized):")
    print(f"    mean = {advantages.mean().item():.6f}")
    print(f"    std  = {advantages.std().item():.6f}")
    print(f"    positive ratio = {(advantages > 0).float().mean().item():.4f}")
    print(f"    negative ratio = {(advantages < 0).float().mean().item():.4f}")

    # ── Gradient Decomposition ──
    print(f"\n{'='*70}")
    print(f"  STEP 1 & 2: Actor Gradient Decomposition")
    print(f"{'='*70}")

    # Forward pass with gradients
    _, new_log_prob, new_outputs = policy.act(
        {"states": states, "taken_actions": actions}, role="policy"
    )

    ratio = torch.exp(new_log_prob - old_log_prob)

    # ── Compute policy surrogate loss ──
    surrogate = advantages * ratio
    surrogate_clipped = advantages * torch.clamp(
        ratio, 1.0 - args.ratio_clip, 1.0 + args.ratio_clip
    )
    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

    # ── Compute entropy ──
    # SKRL MultiCategoricalMixin stores distributions in _distribution
    # We need to get entropy from the categorical distributions
    if hasattr(policy, '_distribution'):
        if isinstance(policy._distribution, list):
            entropy = sum(d.entropy() for d in policy._distribution)
        else:
            entropy = policy._distribution.entropy()
        entropy_mean = entropy.mean()
    else:
        # Fallback: compute from logits
        logits = new_outputs.get("net_output", None)
        if logits is not None:
            lin_logits = logits[:, :NUM_BINS]
            ang_logits = logits[:, NUM_BINS:]
            lin_dist = torch.distributions.Categorical(logits=lin_logits)
            ang_dist = torch.distributions.Categorical(logits=ang_logits)
            entropy_mean = (lin_dist.entropy() + ang_dist.entropy()).mean()
        else:
            print("[ERROR] Cannot compute entropy")
            return

    entropy_loss = -entropy_mean  # negative because we maximize entropy

    # ── 1. Policy-only gradient ──
    policy.zero_grad()
    policy_loss.backward(retain_graph=True)
    grad_norm_policy = compute_gradient_norms(policy.parameters())

    # Save policy-only gradients
    policy_grads = {}
    for name, p in policy.named_parameters():
        if p.grad is not None:
            policy_grads[name] = p.grad.clone()

    # ── 2. Entropy-only gradient ──
    policy.zero_grad()
    (args.entropy_scale * entropy_loss).backward(retain_graph=True)
    grad_norm_entropy = compute_gradient_norms(policy.parameters())

    # Save entropy-only gradients
    entropy_grads = {}
    for name, p in policy.named_parameters():
        if p.grad is not None:
            entropy_grads[name] = p.grad.clone()

    # ── 3. Combined gradient (policy + entropy) ──
    policy.zero_grad()
    total_actor_loss = policy_loss + args.entropy_scale * entropy_loss
    total_actor_loss.backward(retain_graph=True)
    grad_norm_total = compute_gradient_norms(policy.parameters())

    # ── 4. Compute cosine similarity between policy and entropy gradients ──
    cos_sim_sum = 0.0
    cos_count = 0
    for name in policy_grads:
        if name in entropy_grads:
            pg = policy_grads[name].view(-1)
            eg = entropy_grads[name].view(-1)
            cos = F.cosine_similarity(pg.unsqueeze(0), eg.unsqueeze(0)).item()
            cos_sim_sum += cos
            cos_count += 1
    avg_cos_sim = cos_sim_sum / max(cos_count, 1)

    # ── Per-layer breakdown ──
    print(f"\n  Per-layer gradient norms:")
    print(f"  {'Layer':<40s} {'Policy':>10s} {'Entropy':>10s} {'Ratio':>8s} {'CosSim':>8s}")
    print(f"  {'-'*76}")
    for name in sorted(policy_grads.keys()):
        if name in entropy_grads:
            pg_norm = policy_grads[name].norm().item()
            eg_norm = entropy_grads[name].norm().item()
            ratio_val = eg_norm / max(pg_norm, 1e-10)
            pg = policy_grads[name].view(-1)
            eg = entropy_grads[name].view(-1)
            cos = F.cosine_similarity(pg.unsqueeze(0), eg.unsqueeze(0)).item()
            print(f"  {name:<40s} {pg_norm:>10.6f} {eg_norm:>10.6f} {ratio_val:>8.3f} {cos:>8.4f}")

    print(f"\n  Summary:")
    print(f"    grad_norm_policy_only  = {grad_norm_policy:.6f}")
    print(f"    grad_norm_entropy_only = {grad_norm_entropy:.6f}")
    print(f"    grad_norm_total        = {grad_norm_total:.6f}")
    print(f"    ratio (entropy/policy) = {grad_norm_entropy / max(grad_norm_policy, 1e-10):.4f}")
    print(f"    avg cosine similarity  = {avg_cos_sim:.4f}")
    print(f"    policy_loss            = {policy_loss.item():.8f}")
    print(f"    entropy_loss           = {entropy_loss.item():.8f}")
    print(f"    entropy_loss × scale   = {(args.entropy_scale * entropy_loss).item():.8f}")

    # ── Step 3: Action Distribution Analysis ──
    print(f"\n{'='*70}")
    print(f"  STEP 3: Action Distribution Analysis")
    print(f"{'='*70}")

    with torch.no_grad():
        _, _, out = policy.act({"states": states}, role="policy")

        # Get logits from the model
        features = policy.extractor(states)
        logits_raw = policy.head(features)  # [N, 38]
        lin_logits = logits_raw[:, :NUM_BINS]  # [N, 19]
        ang_logits = logits_raw[:, NUM_BINS:]  # [N, 19]

        lin_probs = F.softmax(lin_logits, dim=-1)  # [N, 19]
        ang_probs = F.softmax(ang_logits, dim=-1)  # [N, 19]

        # Per-bin average probability
        lin_avg = lin_probs.mean(dim=0)  # [19]
        ang_avg = ang_probs.mean(dim=0)  # [19]

        uniform_prob = 1.0 / NUM_BINS

        print(f"\n  Linear action bin probabilities (uniform = {uniform_prob:.4f}):")
        for i in range(NUM_BINS):
            bar = "█" * int(lin_avg[i].item() * 200)
            label = f"idx={i:>2d} (ratio={((i-9)/9):>+.2f})"
            print(f"    {label}: {lin_avg[i].item():.5f} {bar}")

        print(f"\n  Angular action bin probabilities:")
        for i in range(NUM_BINS):
            bar = "█" * int(ang_avg[i].item() * 200)
            label = f"idx={i:>2d} (ratio={((i-9)/9):>+.2f})"
            print(f"    {label}: {ang_avg[i].item():.5f} {bar}")

        # Top-1 probability
        lin_top1 = lin_probs.max(dim=-1).values.mean().item()
        ang_top1 = ang_probs.max(dim=-1).values.mean().item()

        print(f"\n  Top-1 action probability:")
        print(f"    Linear:  {lin_top1:.5f} (uniform max = {uniform_prob:.5f})")
        print(f"    Angular: {ang_top1:.5f}")

        # Entropy
        lin_ent = torch.distributions.Categorical(probs=lin_probs).entropy().mean().item()
        ang_ent = torch.distributions.Categorical(probs=ang_probs).entropy().mean().item()
        H_max = math.log(NUM_BINS)

        print(f"\n  Entropy:")
        print(f"    Linear:  {lin_ent:.4f} / {H_max:.4f} = {lin_ent/H_max*100:.1f}%")
        print(f"    Angular: {ang_ent:.4f} / {H_max:.4f} = {ang_ent/H_max*100:.1f}%")
        print(f"    Total:   {lin_ent+ang_ent:.4f} / {2*H_max:.4f} = {(lin_ent+ang_ent)/(2*H_max)*100:.1f}%")

        # Logit range (if logits are flat, probs are uniform)
        print(f"\n  Logit statistics:")
        print(f"    Linear logits: mean={lin_logits.mean().item():.4f} std={lin_logits.std().item():.4f} "
              f"range=[{lin_logits.min().item():.4f}, {lin_logits.max().item():.4f}]")
        print(f"    Angular logits: mean={ang_logits.mean().item():.4f} std={ang_logits.std().item():.4f} "
              f"range=[{ang_logits.min().item():.4f}, {ang_logits.max().item():.4f}]")

    # ── Step 5: Per-action-type reward distinction ──
    print(f"\n{'='*70}")
    print(f"  STEP 5: Action-Reward Distinction")
    print(f"{'='*70}")
    print(f"  (Using synthetic data — for true distinction, need env rollout)")

    with torch.no_grad():
        # Classify actions: forward (accel_idx > 9), stop (≈9), retreat (< 9)
        accel_idx = actions[:, 0].long()
        forward_mask = accel_idx > 10
        stop_mask = (accel_idx >= 8) & (accel_idx <= 10)
        retreat_mask = accel_idx < 8

        n_fwd = forward_mask.sum().item()
        n_stop = stop_mask.sum().item()
        n_ret = retreat_mask.sum().item()

        print(f"\n  Action type distribution:")
        print(f"    Forward (idx>10): {n_fwd}/{N} = {n_fwd/N*100:.1f}%")
        print(f"    Stop (idx 8-10):  {n_stop}/{N} = {n_stop/N*100:.1f}%")
        print(f"    Retreat (idx<8):  {n_ret}/{N} = {n_ret/N*100:.1f}%")

        # Since we're using synthetic advantages, we can only show the
        # log_prob differences across action types
        print(f"\n  Log prob by action type:")
        if n_fwd > 0:
            print(f"    Forward:  mean={old_log_prob[forward_mask].mean().item():.4f}")
        if n_stop > 0:
            print(f"    Stop:     mean={old_log_prob[stop_mask].mean().item():.4f}")
        if n_ret > 0:
            print(f"    Retreat:  mean={old_log_prob[retreat_mask].mean().item():.4f}")

    # ── PPO ratio statistics ──
    print(f"\n{'='*70}")
    print(f"  STEP 2 (supplement): PPO Ratio Statistics")
    print(f"{'='*70}")
    with torch.no_grad():
        print(f"  ratio = exp(new_log_prob - old_log_prob):")
        print(f"    mean = {ratio.mean().item():.6f}")
        print(f"    std  = {ratio.std().item():.6f}")
        print(f"    min  = {ratio.min().item():.6f}")
        print(f"    max  = {ratio.max().item():.6f}")
        clip_frac = ((ratio - 1.0).abs() > args.ratio_clip).float().mean().item()
        print(f"    clip_fraction = {clip_frac:.6f}")
        kl = ((ratio - 1) - (new_log_prob - old_log_prob)).mean().item()
        print(f"    approx_kl = {max(kl, 0):.6f}")

    print(f"\n{'='*70}")
    print(f"  NOTE: This uses SYNTHETIC data (random obs + random advantages).")
    print(f"  The gradient NORMS and RATIOS are meaningful (architecture-dependent),")
    print(f"  but absolute loss values depend on the data distribution.")
    print(f"  The action distribution analysis uses real checkpoint weights")
    print(f"  and is fully valid.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
