# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Custom IsaacLab wrapper for Asymmetric Actor-Critic (AAC) architecture.

This wrapper extends the standard IsaacLab wrapper to properly support:
- shared_states (critic observations) for single-agent training
- Optional frame stacking (current VLP16 config uses num_stack=1)

Current VLP16 AC config (139D, symmetric):
- Policy obs:  139D = ego(4) + goal(2) + LiDAR(72) + obstacles(60) + time(1)
- Critic obs:  139D (same as policy, no privileged info)
- privileged_dim = 0 (symmetric actor-critic)
- num_stack = 1 (no frame stacking)
"""

from typing import Any, Tuple

import gymnasium
import numpy as np
import torch
from gymnasium.spaces import Box, MultiDiscrete

from skrl.envs.wrappers.torch import Wrapper
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space, unflatten_tensorized_space


class AACIsaacLabWrapper(Wrapper):
    """Isaac Lab environment wrapper for AAC (Asymmetric Actor-Critic).

    Supports optional frame stacking and privileged critic observations.
    With num_stack=1 and privileged_dim=0 (current VLP16 config), both
    actor and critic receive the same 139D observation.

    The wrapper:
    - Maintains a rolling buffer of past observations (when num_stack > 1)
    - Uses observations["policy"] for the actor
    - Provides observations["critic"] as shared_states for the critic
    - Clamps observations and replaces NaN to protect RunningStandardScaler

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

        # Privileged info dimension for critic (0 = symmetric actor-critic)
        self.privileged_dim = 0

        # Cache the original observation and state spaces from unwrapped environment
        self._original_observation_space = self._get_original_observation_space()
        self._original_state_space = self._get_original_state_space()

        # Get original observation dimensions
        self._reset_once = True
        self._observations = None
        self._states = None  # Shared states (critic observations)
        self._info = {}

        # Frame stacking buffer: [num_envs, obs_dim * num_stack]
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
        """Get the original observation space from the unwrapped environment."""
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
        """Get the original state space from the unwrapped environment."""
        # First try: single_observation_space (for vectorized environments)
        if hasattr(self._unwrapped, 'single_observation_space'):
            single_obs = self._unwrapped.single_observation_space
            if hasattr(single_obs, 'spaces') and 'critic' in single_obs.spaces:
                return single_obs.spaces['critic']

        # Second try: observation_space (for non-vectorized environments)
        if hasattr(self._unwrapped, 'observation_space'):
            obs = self._unwrapped.observation_space
            if hasattr(obs, 'spaces') and 'critic' in obs.spaces:
                return obs.spaces['critic']

        # Fallback: return policy observation space (symmetric actor-critic)
        return self._get_original_observation_space()

    def _create_modified_spaces(self) -> None:
        """Create modified observation and state spaces with correct dimensions.

        This is crucial for SKRL to initialize preprocessors with correct sizes:
        - observation_space: obs_dim * num_stack
        - state_space: obs_dim * num_stack + privileged_dim
        """
        import numpy as np

        orig_obs = self._original_observation_space

        if orig_obs is not None and hasattr(orig_obs, 'shape') and hasattr(orig_obs, 'low'):
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
        if self._modified_observation_space is not None:
            stacked_dim = self._modified_observation_space.shape[0]
            state_dim = stacked_dim + self.privileged_dim

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
        """Get the state space (critic observations) with correct dimensions."""
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
        """Get the observation space (policy observations) with correct dimensions."""
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

        Override to return MultiDiscrete([19, 19]) for the discrete differential
        drive action term. Isaac Lab's action manager creates Box(2) from action_dim=2,
        but SKRL needs the correct discrete space type for MultiCategoricalMixin.
        """
        if not hasattr(self, '_cached_action_space'):
            try:
                raw_space = self._unwrapped.single_action_space
            except AttributeError:
                raw_space = self._unwrapped.action_space

            # Detect discrete action term and override to MultiDiscrete
            if hasattr(raw_space, 'shape') and raw_space.shape == (2,):
                # Check if it's our discrete differential drive (action_dim=2)
                try:
                    env = self._unwrapped
                    action_term = list(env.action_manager._terms.values())[0]
                    if hasattr(action_term, 'cfg') and hasattr(action_term.cfg, 'num_bins'):
                        n = action_term.cfg.num_bins
                        self._cached_action_space = MultiDiscrete(np.array([n, n]))
                        print(f"[AAC_WRAPPER] Action space: MultiDiscrete([{n}, {n}])")
                        return self._cached_action_space
                except Exception:
                    pass
            self._cached_action_space = raw_space
        return self._cached_action_space

    def state(self) -> torch.Tensor:
        """Get the privileged observations (states) for the critic."""
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
    # Observation Clamping — prevent physics explosion from poisoning scaler
    # ========================================================================
    # Physics engine occasionally produces extreme values (e.g. lin_vel=107,546).
    # A single extreme value can permanently corrupt RunningStandardScaler's
    # running_mean/running_var. Clamping at the wrapper level is the safest defense.
    OBS_CLAMP_RANGE = 100.0

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
            - observations: [num_envs, obs_dim * num_stack]
            - info["shared_states"]: [num_envs, obs_dim * num_stack + privileged_dim]
        """
        actions = unflatten_tensorized_space(self.action_space, actions)
        observations, reward, terminated, truncated, self._info = self._env.step(actions)

        # Extract policy observations for actor
        new_actor_obs = flatten_tensorized_space(observations["policy"])
        new_actor_obs = self._clamp_observations(new_actor_obs)

        # Extract critic observations (may contain privileged info)
        if "critic" in observations:
            privileged_info = flatten_tensorized_space(observations["critic"])
            privileged_info = self._clamp_observations(privileged_info)
        else:
            privileged_info = new_actor_obs

        # Initialize stacked buffer on first step
        if self.stacked_obs is None:
            self._init_stacked_obs(new_actor_obs)

        # Update stacked buffer with new observations
        reset_mask = (terminated.view(-1) | truncated.view(-1))
        self._update_stacked_obs(new_actor_obs, reset_mask)

        # Return stacked observations for Actor
        self._observations = self.stacked_obs.clone()

        # Prepare shared_states for Critic: stacked obs + privileged info
        if privileged_info.shape[-1] > new_actor_obs.shape[-1]:
            obstacle_state = privileged_info[:, -self.privileged_dim:]
        else:
            obstacle_state = torch.zeros(new_actor_obs.shape[0], self.privileged_dim,
                                         device=new_actor_obs.device)

        self._states = torch.cat([self.stacked_obs, obstacle_state], dim=-1)

        # Add shared_states to info for AAC agent to use
        self._info["shared_states"] = self._states

        # Debug: Print shapes (only once at startup)
        if not hasattr(AACIsaacLabWrapper, '_shape_debug_printed'):
            print(f"[AAC_WRAPPER] Actor observations shape: {self._observations.shape}")
            print(f"[AAC_WRAPPER] Critic shared_states shape: {self._states.shape}")
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
            - observations: [num_envs, obs_dim * num_stack]
            - info["shared_states"]: [num_envs, obs_dim * num_stack + privileged_dim]
        """
        if self._reset_once:
            observations, self._info = self._env.reset()

            # Extract policy observations
            new_actor_obs = flatten_tensorized_space(observations["policy"])
            new_actor_obs = self._clamp_observations(new_actor_obs)

            # Extract critic observations
            if "critic" in observations:
                privileged_info = flatten_tensorized_space(observations["critic"])
                privileged_info = self._clamp_observations(privileged_info)
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

            # Add shared_states to info for AAC agent to use
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
