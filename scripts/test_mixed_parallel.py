#!/usr/bin/env python3
"""Quick test of mixed parallel environment with 10 environments"""

import gymnasium as gym
from isaaclab.app import AppLauncher

# Setup app launcher - headless mode
app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import isaaclab_tasks
from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.cfg.charge_env_cfg_phase0 import (
    ChargeNavigationEnvCfgPhase0_PLAY,
)

def main():
    # Create environment using PLAY config (10 environments)
    env = gym.make("Isaac-Navigation-Charge-Phase0-Play", cfg=ChargeNavigationEnvCfgPhase0_PLAY())
    env_unwrapped = env.unwrapped

    print("=" * 80)
    print("Quick Mixed Parallel Environment Test (10 environments)")
    print("=" * 80)

    # Reset environment
    obs, info = env.reset()

    print("\n[1] Environment Info:")
    print(f"  Number of environments: {env_unwrapped.num_envs}")

    # Check difficulty distribution
    print("\n[2] Difficulty Distribution:")
    if hasattr(env_unwrapped, "_env_difficulty"):
        difficulties = env_unwrapped._env_difficulty.cpu().numpy()
        empty_count = (difficulties == 0).sum()
        static_count = (difficulties == 1).sum()
        dynamic_count = (difficulties == 2).sum()

        print(f"  Empty (difficulty=0): {empty_count}/10")
        print(f"  Static (difficulty=1): {static_count}/10")
        print(f"  Dynamic (difficulty=2): {dynamic_count}/10")

        # Show per-environment
        for i, diff in enumerate(difficulties):
            if diff == 0:
                print(f"    Env {i}: Empty")
            elif diff == 1:
                num_vis = env_unwrapped._env_num_visible_obstacles[i].item()
                print(f"    Env {i}: Static ({num_vis} obstacles)")
            else:
                num_vis = env_unwrapped._env_num_visible_obstacles[i].item()
                print(f"    Env {i}: Dynamic ({num_vis} obstacles)")

    # Check obstacle positions
    print("\n[3] Obstacle Visibility:")
    if hasattr(env_unwrapped.scene, "obstacle_0"):
        obstacle_0_z = env_unwrapped.scene.obstacle_0.data.root_pos_w[:, 2]
        hidden_count = (obstacle_0_z < 0).sum().item()
        visible_count = (obstacle_0_z >= 0).sum().item()
        print(f"  Obstacle_0: {visible_count} visible, {hidden_count} hidden")

    # Check observations
    print("\n[4] Observation Shape:")
    if isinstance(obs, dict):
        print(f"  Policy obs shape: {obs['policy'].shape}")
        print(f"  Critic obs shape: {obs['critic'].shape}")

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
