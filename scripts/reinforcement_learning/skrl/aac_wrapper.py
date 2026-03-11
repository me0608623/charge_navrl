# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Custom IsaacLab wrapper for Asymmetric Actor-Critic (AAC) architecture.

This wrapper extends the standard IsaacLab wrapper to properly support:
- shared_states (critic observations) for single-agent training
- Optional frame stacking (disabled in v2 baseline: num_stack=1)

v2 baseline (79D):
- Policy: 79D single-frame observation
- Critic: 79D (same as policy, no privileged info)
- Reset Handling: Properly clears history ghost for reset environments
"""

from typing import Any, Tuple

import gymnasium
import torch
from gymnasium.spaces import Box

from skrl.envs.wrappers.torch import Wrapper
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space, unflatten_tensorized_space


class AACIsaacLabWrapper(Wrapper):
    """Isaac Lab environment wrapper for AAC (Asymmetric Actor-Critic) with Frame Stacking.

    Frame Stacking Architecture:
    - Original Actor observation: 81 dim
    - Stacked Actor observation: 81 * 3 = 243 dim (3 frames)
    - Critic observation: 243 + 50 (privileged obstacles) = 293 dim

    The wrapper:
    - Maintains a rolling buffer of past observations
    - Uses observations["policy"] for the actor (stacked, 243-dim)
    - Provides observations["critic"] as shared_states for the critic (293-dim)

    Important: Reset handling clears history for environments that have been reset.
    """

    def __init__(self, env: Any, num_stack: int = 1) -> None:
        """Initialize the AAC Isaac Lab wrapper.

        Args:
            env: The Isaac Lab environment to wrap
            num_stack: Number of frames to stack (default: 1 = no stacking)
        """
        super().__init__(env)

        self.num_stack = num_stack
        self._device = env.device if hasattr(env, 'device') else torch.device('cuda')

        # Track privileged info dimension (0 in v2 baseline)
        self.privileged_dim = 0

        # Cache the original observation and state spaces from unwrapped environment
        self._original_observation_space = self._get_original_observation_space()
        self._original_state_space = self._get_original_state_space()

        # Get original observation dimensions
        self._reset_once = True
        self._observations = None
        self._states = None  # Shared states (critic observations)
        self._info = {}

        # Frame stacking buffer: [num_envs, stacked_obs_dim]
        self.stacked_obs = None

        # Create modified observation and state spaces for SKRL
        self._modified_observation_space = None
        self._modified_state_space = None
        self._create_modified_spaces()

        # Debug: Print observation and state spaces (only once at startup)
        if not hasattr(AACIsaacLabWrapper, '_debug_printed'):
            print(f"[AAC_WRAPPER] Original observation_space: {self._original_observation_space}")
            print(f"[AAC_WRAPPER] Modified observation_space (stacked): {self._modified_observation_space}")
            print(f"[AAC_WRAPPER] Modified state_space (stacked+privileged): {self._modified_state_space}")
            if self._original_observation_space is not None and hasattr(self._original_observation_space, 'shape'):
                self._original_obs_dim = self._original_observation_space.shape[0]
                print(f"[AAC_WRAPPER] Original obs dim: {self._original_obs_dim}")
                print(f"[AAC_WRAPPER] Stacked obs dim: {self._original_obs_dim * self.num_stack}")
                print(f"[AAC_WRAPPER] Critic state dim: {self._original_obs_dim * self.num_stack + self.privileged_dim}")
            AACIsaacLabWrapper._debug_printed = True

    def _get_original_observation_space(self) -> gymnasium.Space:
        """Get the original observation space from the unwrapped environment.

        Returns:
            The original policy observation space (81-dim Box)
        """
        # First try: single_observation_space
        if hasattr(self._unwrapped, 'single_observation_space'):
            single_obs = self._unwrapped.single_observation_space
            if hasattr(single_obs, '__contains__') and 'policy' in single_obs:
                return single_obs['policy']
            if hasattr(single_obs, 'spaces') and 'policy' in single_obs.spaces:
                return single_obs.spaces['policy']

        # Second try: observation_space
        if hasattr(self._unwrapped, 'observation_space'):
            obs = self._unwrapped.observation_space
            if hasattr(obs, '__contains__') and 'policy' in obs:
                return obs['policy']
            if hasattr(obs, 'spaces') and 'policy' in obs.spaces:
                return obs.spaces['policy']

        # Fallback: return observation_space directly
        return self._unwrapped.observation_space

    def _get_original_state_space(self) -> gymnasium.Space:
        """Get the original state space from the unwrapped environment.

        Returns:
            The original critic observation space (131-dim Box)
        """
        # First try: single_observation_space (for vectorized environments)
        if hasattr(self._unwrapped, 'single_observation_space'):
            single_obs = self._unwrapped.single_observation_space
            # Try .spaces attribute (for gym.spaces.Dict)
            if hasattr(single_obs, 'spaces') and 'critic' in single_obs.spaces:
                return single_obs.spaces['critic']

        # Second try: observation_space (for non-vectorized environments)
        if hasattr(self._unwrapped, 'observation_space'):
            obs = self._unwrapped.observation_space
            # Try .spaces attribute
            if hasattr(obs, 'spaces') and 'critic' in obs.spaces:
                return obs.spaces['critic']

        # Fallback: return policy observation space (this makes it symmetric, not AAC)
        return self._get_original_observation_space()

    def _create_modified_spaces(self) -> None:
        """Create modified observation and state spaces with correct dimensions.

        This is crucial for SKRL to initialize preprocessors with correct sizes:
        - observation_space: 81 -> 243 (frame stacking)
        - state_space: 131 -> 293 (frame stacking + privileged)
        """
        import numpy as np

        # Get original observation space (should be Box with shape (81,))
        orig_obs = self._original_observation_space

        if orig_obs is not None and hasattr(orig_obs, 'shape') and hasattr(orig_obs, 'low'):
            # Create modified observation space: 81 -> 81 * num_stack = 243
            orig_dim = orig_obs.shape[0]
            stacked_dim = orig_dim * self.num_stack

            # Stack the low and high bounds
            stacked_low = np.repeat(orig_obs.low, self.num_stack)
            stacked_high = np.repeat(orig_obs.high, self.num_stack)

            self._modified_observation_space = Box(
                low=stacked_low,
                high=stacked_high,
                shape=(stacked_dim,),
                dtype=orig_obs.dtype
            )
        else:
            # Fallback: use original space
            self._modified_observation_space = orig_obs

        # Create modified state space for critic (stacked obs + privileged)
        # For AAC, the state space should be: 243 (stacked) + 50 (privileged) = 293
        if self._modified_observation_space is not None:
            stacked_dim = self._modified_observation_space.shape[0]
            state_dim = stacked_dim + self.privileged_dim

            # Use reasonable bounds for privileged info
            state_low = np.concatenate([
                self._modified_observation_space.low,
                np.full(self.privileged_dim, -np.inf, dtype=self._modified_observation_space.dtype)
            ])
            state_high = np.concatenate([
                self._modified_observation_space.high,
                np.full(self.privileged_dim, np.inf, dtype=self._modified_observation_space.dtype)
            ])

            self._modified_state_space = Box(
                low=state_low,
                high=state_high,
                shape=(state_dim,),
                dtype=self._modified_observation_space.dtype
            )
        else:
            self._modified_state_space = None

    @property
    def state_space(self) -> gymnasium.Space:
        """Get the state space (critic observations) with correct dimensions.

        Returns:
            The state space for privileged information (293-dim Box)
        """
        if self._modified_state_space is not None:
            return self._modified_state_space

        # Fallback: try to get from unwrapped environment
        if hasattr(self._unwrapped, 'single_observation_space'):
            single_obs = self._unwrapped.single_observation_space
            if hasattr(single_obs, 'spaces') and 'critic' in single_obs.spaces:
                return single_obs.spaces['critic']

        if hasattr(self._unwrapped, 'observation_space'):
            obs = self._unwrapped.observation_space
            if hasattr(obs, 'spaces') and 'critic' in obs.spaces:
                return obs.spaces['critic']

        return self.observation_space

    @property
    def observation_space(self) -> gymnasium.Space:
        """Get the observation space (policy observations) with correct dimensions.

        Returns:
            The observation space for the actor (243-dim Box)
        """
        if self._modified_observation_space is not None:
            return self._modified_observation_space

        # Fallback: try to get from unwrapped environment
        if hasattr(self._unwrapped, 'single_observation_space'):
            single_obs = self._unwrapped.single_observation_space
            if hasattr(single_obs, '__contains__') and 'policy' in single_obs:
                return single_obs['policy']
            if hasattr(single_obs, 'spaces') and 'policy' in single_obs.spaces:
                return single_obs.spaces['policy']

        if hasattr(self._unwrapped, 'observation_space'):
            obs = self._unwrapped.observation_space
            if hasattr(obs, '__contains__') and 'policy' in obs:
                return obs['policy']
            if hasattr(obs, 'spaces') and 'policy' in obs.spaces:
                return obs.spaces['policy']

        return self._unwrapped.observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        """Get the action space.

        Returns:
            The action space
        """
        try:
            return self._unwrapped.single_action_space
        except AttributeError:
            return self._unwrapped.action_space

    def state(self) -> torch.Tensor:
        """Get the privileged observations (states) for the critic.

        Returns:
            Flattened state tensor for critic observations
        """
        return self._states

    def _init_stacked_obs(self, initial_obs: torch.Tensor) -> None:
        """Initialize the stacked observation buffer.

        Args:
            initial_obs: Initial observation [num_envs, obs_dim]
        """
        num_envs, obs_dim = initial_obs.shape
        stacked_dim = obs_dim * self.num_stack

        # Create buffer filled with initial observations (repeat for all stacks)
        self.stacked_obs = initial_obs.repeat(1, self.num_stack)  # [num_envs, obs_dim * num_stack]

        print(f"[AAC_WRAPPER] Initialized stacked_obs: {self.stacked_obs.shape}")
        print(f"[AAC_WRAPPER] Stack {self.num_stack} frames, original dim: {obs_dim}, stacked dim: {stacked_dim}")

    def _update_stacked_obs(self, new_obs: torch.Tensor, reset_mask: torch.Tensor = None) -> None:
        """Update the stacked observation buffer with new observations.

        Rolling buffer operation:
        1. Shift left: stacked_obs[:, :-obs_dim] = stacked_obs[:, obs_dim:]
        2. Write new: stacked_obs[:, -obs_dim:] = new_obs

        For reset environments, clear history ghost by repeating new obs.

        Args:
            new_obs: New observation [num_envs, obs_dim]
            reset_mask: Boolean mask [num_envs] indicating which envs reset
        """
        obs_dim = new_obs.shape[1]

        # Normal rolling update: shift left and write new at the end
        self.stacked_obs[:, :-obs_dim] = self.stacked_obs[:, obs_dim:].clone()
        self.stacked_obs[:, -obs_dim:] = new_obs

        # Handle reset: clear history ghost for reset environments
        if reset_mask is not None and reset_mask.any():
            reset_indices = torch.where(reset_mask)[0]
            if len(reset_indices) > 0:
                # Repeat new obs for all stack positions (clear history)
                self.stacked_obs[reset_indices] = new_obs[reset_indices].repeat(1, self.num_stack)

    # ========================================================================
    # Fix 1: Observation Clamping — 防止物理爆炸毒化 RunningStandardScaler
    # ========================================================================
    # 物理引擎偶爾會產生極端值（如 lin_vel=107,546），一次就能永久毒化
    # RunningStandardScaler 的 running_mean/running_var，導致後續所有觀測
    # 被錯誤歸一化。在 wrapper 層面 clamp 觀測值是最安全的防線。
    OBS_CLAMP_RANGE = 100.0  # 合理觀測範圍上限

    def _clamp_observations(self, obs: torch.Tensor) -> torch.Tensor:
        """Clamp observations to prevent physics explosion from poisoning scaler.

        Args:
            obs: Raw observation tensor

        Returns:
            Clamped observation tensor with NaN replaced by 0
        """
        obs = torch.nan_to_num(obs, nan=0.0, posinf=self.OBS_CLAMP_RANGE, neginf=-self.OBS_CLAMP_RANGE)
        obs = obs.clamp(-self.OBS_CLAMP_RANGE, self.OBS_CLAMP_RANGE)
        return obs

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment with frame stacking.

        Args:
            actions: Actions to execute

        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
            - observations: Stacked observations [num_envs, 243]
            - info["shared_states"]: Stacked obs + privileged info [num_envs, 293]
        """
        actions = unflatten_tensorized_space(self.action_space, actions)
        observations, reward, terminated, truncated, self._info = self._env.step(actions)

        # Extract policy observations for actor (81-dim)
        # IsaacLab environment returns torch tensors directly, just flatten them
        new_actor_obs = flatten_tensorized_space(observations["policy"])
        new_actor_obs = self._clamp_observations(new_actor_obs)  # Fix 1: clamp

        # Extract critic observations for value function (privileged, 131-dim)
        if "critic" in observations:
            privileged_info = flatten_tensorized_space(observations["critic"])
            privileged_info = self._clamp_observations(privileged_info)  # Fix 1: clamp
        else:
            # Fall back to policy observations if critic not available
            privileged_info = new_actor_obs

        # Initialize stacked buffer on first step
        if self.stacked_obs is None:
            self._init_stacked_obs(new_actor_obs)

        # Update stacked buffer with new observations
        # Create reset mask from terminated and truncated
        reset_mask = (terminated.view(-1) | truncated.view(-1))
        self._update_stacked_obs(new_actor_obs, reset_mask)

        # Return stacked observations for Actor
        self._observations = self.stacked_obs.clone()

        # Prepare shared_states for Critic: stacked obs + privileged info
        # Extract only the obstacle state (last 50 dims) from privileged_info
        # Assuming privileged_info = [base_obs(81), obstacle_state(50)]
        if privileged_info.shape[-1] > new_actor_obs.shape[-1]:
            obstacle_state = privileged_info[:, -self.privileged_dim:]
        else:
            # No separate privileged info, use zeros
            obstacle_state = torch.zeros(new_actor_obs.shape[0], self.privileged_dim,
                                         device=new_actor_obs.device)

        # Concatenate: [stacked_obs(243), obstacle_state(50)] = [293]
        self._states = torch.cat([self.stacked_obs, obstacle_state], dim=-1)

        # IMPORTANT: Add shared_states to info for AAC agent to use
        self._info["shared_states"] = self._states

        # Debug: Print shapes (only once at startup)
        if not hasattr(AACIsaacLabWrapper, '_shape_debug_printed'):
            print(f"[AAC_WRAPPER] Actor observations shape: {self._observations.shape}")
            print(f"[AAC_WRAPPER] Critic shared_states shape: {self._states.shape}")
            print(f"[AAC_WRAPPER] Expected: Actor=243 (81*3), Critic=293 (243+50)")
            AACIsaacLabWrapper._shape_debug_printed = True

        return (
            self._observations,
            reward.view(-1, 1),
            terminated.view(-1, 1),
            truncated.view(-1, 1),
            self._info,
        )

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment.

        Returns:
            Tuple of (observations, info)
            - observations: Stacked observations [num_envs, 243]
            - info["shared_states"]: Stacked obs + privileged info [num_envs, 293]
        """
        if self._reset_once:
            observations, self._info = self._env.reset()

            # Extract policy observations (81-dim)
            # IsaacLab environment returns torch tensors directly, just flatten them
            new_actor_obs = flatten_tensorized_space(observations["policy"])
            new_actor_obs = self._clamp_observations(new_actor_obs)  # Fix 1: clamp

            # Extract critic observations (privileged, 131-dim)
            if "critic" in observations:
                privileged_info = flatten_tensorized_space(observations["critic"])
                privileged_info = self._clamp_observations(privileged_info)  # Fix 1: clamp
            else:
                privileged_info = new_actor_obs

            # Initialize stacked buffer (all frames get same initial obs)
            self._init_stacked_obs(new_actor_obs)

            # Return stacked observations for Actor
            self._observations = self.stacked_obs.clone()

            # Prepare shared_states for Critic
            if privileged_info.shape[-1] > new_actor_obs.shape[-1]:
                obstacle_state = privileged_info[:, -self.privileged_dim:]
            else:
                obstacle_state = torch.zeros(new_actor_obs.shape[0], self.privileged_dim,
                                             device=new_actor_obs.device)

            self._states = torch.cat([self.stacked_obs, obstacle_state], dim=-1)

            # IMPORTANT: Add shared_states to info for AAC agent to use
            self._info["shared_states"] = self._states

            self._reset_once = False

        return self._observations, self._info

    def render(self, *args, **kwargs) -> None:
        """Render the environment (no-op)."""
        return None

    def close(self) -> None:
        """Close the environment."""
        self._env.close()


def wrap_env_for_aac(env: Any, ml_framework: str = "torch", num_stack: int = 3):
    """Wrap Isaac Lab environment for AAC training with frame stacking.

    Args:
        env: The Isaac Lab environment
        ml_framework: ML framework (default: "torch")
        num_stack: Number of frames to stack (default: 3)

    Returns:
        Wrapped environment with AAC support and frame stacking
    """
    if ml_framework.startswith("torch"):
        return AACIsaacLabWrapper(env, num_stack=num_stack)
    else:
        raise ValueError(f"AAC wrapper only supports torch, got {ml_framework}")
