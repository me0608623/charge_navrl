"""Verify navrl_dense_v8 module output matches env reward_manager output.

Launches NavRL-Ground-V8 env, runs N steps, compares:
  - env.reward_buf (from reward_manager)
  - navrl_dense_v8.compute() (our external module)

Usage:
  ./isaaclab.sh -p scripts/reinforcement_learning/skrl/tests/test_navrl_v8_identity.py \
    --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL-Ground-V8 \
    --num_envs 4 --headless --steps 20
"""

from __future__ import annotations

import argparse
import sys
import torch

# --- AppLauncher (must come before any isaaclab imports) ---
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Navigation-Charge-VLP16-Curriculum-NavRL-Ground-V8")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Post-launcher imports ---
import gymnasium as gym
import isaaclab_tasks  # noqa: F401 — registers gym tasks

# Add skrl to path
from pathlib import Path
_skrl_root = Path(__file__).resolve().parent.parent
if str(_skrl_root) not in sys.path:
    sys.path.insert(0, str(_skrl_root))

from rnn_car_modular.rewards.navrl_dense_v8 import NavRLDenseV8Reward, REWARD_TERMS


def main():
    print("=" * 70)
    print("NavRL-Ground V8 Identity Test")
    print("=" * 70)

    # --- 1. Create env ---
    env = gym.make(args_cli.task, cfg={"scene": {"num_envs": args_cli.num_envs}})
    env_unwrapped = env.unwrapped
    print(f"[ENV] task={args_cli.task}, num_envs={args_cli.num_envs}")

    # --- 2. Print reward_manager terms for comparison ---
    rm = env_unwrapped.reward_manager
    print(f"\n[REWARD_MANAGER] {len(rm._term_names)} terms:")
    for i, (name, cfg) in enumerate(zip(rm._term_names, rm._term_cfgs)):
        print(f"  [{i}] {name:30s} weight={cfg.weight:>8.3f}")

    print(f"\n[MODULE] {len(REWARD_TERMS)} terms:")
    for name, cfg in REWARD_TERMS.items():
        print(f"       {name:30s} weight={cfg.weight:>8.3f}")

    # --- 3. Check weight consistency ---
    print("\n[WEIGHT CHECK]")
    weight_match = True
    for name, mod_cfg in REWARD_TERMS.items():
        if name in rm._term_names:
            idx = rm._term_names.index(name)
            rm_weight = rm._term_cfgs[idx].weight
            if abs(mod_cfg.weight - rm_weight) > 1e-6:
                print(f"  MISMATCH: {name} module={mod_cfg.weight} vs rm={rm_weight}")
                weight_match = False
            else:
                print(f"  OK: {name} weight={mod_cfg.weight}")
        else:
            print(f"  MISSING in reward_manager: {name}")
            weight_match = False

    # Check for non-zero rm terms not in our module
    for i, name in enumerate(rm._term_names):
        rm_weight = rm._term_cfgs[i].weight
        if name not in REWARD_TERMS and abs(rm_weight) > 1e-6:
            print(f"  EXTRA non-zero in rm: {name} weight={rm_weight}")
            weight_match = False

    if weight_match:
        print("  >>> All weights match!")
    else:
        print("  >>> WARNING: weight mismatch detected")

    # --- 4. Create module ---
    module = NavRLDenseV8Reward()

    # --- 5. Run steps and compare ---
    print(f"\n[RUNNING] {args_cli.steps} steps...")
    obs, _ = env.reset()
    N = env_unwrapped.num_envs
    device = env_unwrapped.device

    max_abs_diff = 0.0
    max_rel_diff = 0.0
    all_close_count = 0
    total_count = 0

    for step in range(args_cli.steps):
        # Random actions
        action_space = env.action_space
        actions = torch.tensor(
            [action_space.sample() for _ in range(N)],
            device=device, dtype=torch.float32,
        )

        # env.step() → reward_manager computes reward internally
        obs, rm_reward, terminated, truncated, info = env.step(actions)
        rm_reward_flat = rm_reward.squeeze(-1)  # [N]

        # Our module computes externally
        mod_reward, mod_breakdown = module.compute(
            env_unwrapped, actions, terminated, truncated,
        )

        # Compare
        abs_diff = (rm_reward_flat - mod_reward).abs()
        max_diff_this = abs_diff.max().item()
        max_abs_diff = max(max_abs_diff, max_diff_this)

        denom = rm_reward_flat.abs().clamp(min=1e-6)
        rel_diff = (abs_diff / denom).max().item()
        max_rel_diff = max(max_rel_diff, rel_diff)

        is_close = torch.allclose(rm_reward_flat, mod_reward, atol=1e-4, rtol=1e-3)
        if is_close:
            all_close_count += 1
        total_count += 1

        status = "OK" if is_close else "DIFF"
        print(
            f"  step {step:3d}: {status} | "
            f"rm_reward=[{rm_reward_flat.mean():.4f}] "
            f"mod_reward=[{mod_reward.mean():.4f}] "
            f"max_abs_diff={max_diff_this:.6f}"
        )

        # Per-term comparison on first step
        if step == 0:
            print(f"\n  [TERM BREAKDOWN step 0]")
            dt = env_unwrapped.step_dt
            for name, raw in mod_breakdown.items():
                weighted = raw * REWARD_TERMS[name].weight * dt
                if name in rm._term_names:
                    idx = rm._term_names.index(name)
                    rm_val = rm._step_reward[:, idx] * dt  # rm stores value/dt
                    term_diff = (weighted - rm_val).abs().max().item()
                    print(
                        f"    {name:25s}: mod={weighted.mean():.6f}  "
                        f"rm={rm_val.mean():.6f}  diff={term_diff:.6f}"
                    )
                else:
                    print(f"    {name:25s}: mod={weighted.mean():.6f}  rm=N/A")
            print()

    # --- 6. Summary ---
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Steps tested:    {total_count}")
    print(f"  Steps allclose:  {all_close_count}/{total_count}")
    print(f"  Max abs diff:    {max_abs_diff:.8f}")
    print(f"  Max rel diff:    {max_rel_diff:.8f}")

    if all_close_count == total_count:
        print(f"\n  >>> PASS: navrl_dense_v8 output is IDENTICAL to reward_manager")
    else:
        print(f"\n  >>> FAIL: {total_count - all_close_count} steps had differences")
        print(f"      Check weight/param consistency above")

    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
