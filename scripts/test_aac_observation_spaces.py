#!/usr/bin/env python3
"""Test script to verify AAC observation spaces"""

import sys
import gymnasium as gym
from isaaclab.app import AppLauncher

# Create minimal argparse
class Args:
    headless = True
    enable_cameras = False

# Setup app launcher
app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

# Import after app launcher
import isaaclab_tasks
import torch

def test_observation_spaces():
    """Test that the observation spaces are correctly configured for AAC"""
    # Create environment
    env = gym.make("Isaac-Navigation-Charge-Phase0", num_envs=2)

    print("=" * 80)
    print("Testing AAC Observation Spaces")
    print("=" * 80)

    # Check single_observation_space
    print("\nsingle_observation_space:")
    if hasattr(env.unwrapped, 'single_observation_space'):
        for key, space in env.unwrapped.single_observation_space.items():
            print(f"  {key}: {space}")

    # Check observation space
    print("\nobservation_space:")
    if hasattr(env.unwrapped, 'observation_space'):
        print(f"  {env.unwrapped.observation_space}")

    # Reset environment and check observation structure
    print("\n" + "=" * 80)
    print("Testing step() to verify observation dict structure")
    print("=" * 80)

    obs, info = env.reset()

    print("\nReturned observations from reset():")
    if isinstance(obs, dict):
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: type={type(value)}")
    else:
        print(f"  observations type: {type(obs)}")
        if hasattr(obs, 'shape'):
            print(f"  observations shape: {obs.shape}")

    # Test step
    actions = torch.zeros(env.unwrapped.num_envs, env.unwrapped.action_space.shape[0])
    obs, rewards, terminated, truncated, info = env.step(actions)

    print("\nReturned observations from step():")
    if isinstance(obs, dict):
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: type={type(value)}")
    else:
        print(f"  observations type: {type(obs)}")
        if hasattr(obs, 'shape'):
            print(f"  observations shape: {obs.shape}")

    # Verify dimensions
    print("\n" + "=" * 80)
    print("Verification")
    print("=" * 80)

    expected_dims = {
        "policy": 81,  # 72 + 2 + 2 + 1 + 1 + 1 + 2
        "critic": 131,  # 81 + 50
    }

    if isinstance(obs, dict):
        all_ok = True
        for key, expected_dim in expected_dims.items():
            if key in obs:
                actual_dim = obs[key].shape[-1] if len(obs[key].shape) > 0 else 0
                if actual_dim == expected_dim:
                    print(f"  ✓ {key}: {actual_dim} dim (expected {expected_dim})")
                else:
                    print(f"  ✗ {key}: {actual_dim} dim (expected {expected_dim}) - MISMATCH!")
                    all_ok = False
            else:
                print(f"  ✗ {key}: NOT FOUND in observations!")
                all_ok = False

        if all_ok:
            print("\n✓ All observation dimensions match!")
        else:
            print("\n✗ Some observation dimensions DO NOT match!")

    env.close()

if __name__ == "__main__":
    try:
        test_observation_spaces()
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
