# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CBF Action Wrapper for SKRL PPO

This module implements a Gym ActionWrapper that integrates Control Barrier Functions (CBF)
as a safety filter between the PPO policy output and the environment.

The key innovation is the penalty mechanism that prevents PPO from becoming dependent
on the CBF safety filter.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, DirectRLEnv

from ..hierarchical.safety_shield import (
    SafetyShield,
    RuleBasedShield,
    CBFShield,
    CompositeShield,
)
from ..hierarchical.types import SafetyAction, Obstacle


@dataclass
class CBFWrapperConfig:
    """Configuration for CBF Action Wrapper

    Attributes:
        safety_distance: Safety distance in meters
        danger_distance: Danger distance in meters
        emergency_distance: Emergency distance in meters
        penalty_coeff: Penalty coefficient for CBF correction (prevents dependency)
        alpha: CBF alpha parameter (controls convergence speed)
        max_linear_speed: Maximum linear speed
        max_angular_speed: Maximum angular speed
        shield_type: Type of safety shield ('rule', 'cbf', 'composite')
        track_stats: Whether to track CBF statistics
    """

    safety_distance: float = 0.8
    danger_distance: float = 0.5
    emergency_distance: float = 0.3
    penalty_coeff: float = 0.5
    alpha: float = 1.0
    max_linear_speed: float = 1.5
    max_angular_speed: float = 1.5
    shield_type: str = "composite"  # 'rule', 'cbf', 'composite'
    track_stats: bool = True

    def __post_init__(self):
        if self.shield_type not in ["rule", "cbf", "composite"]:
            raise ValueError(f"Invalid shield_type: {self.shield_type}")


class CBFActionWrapper(gym.ActionWrapper):
    """CBF Action Wrapper for SKRL PPO

    This wrapper intercepts actions from the PPO policy and filters them through
    a Control Barrier Function (CBF) safety filter.

    Key features:
    1. Intercepts PPO actions before they reach the environment
    2. Applies CBF safety filter to ensure safe actions
    3. Adds penalty to reward when CBF modifies the action
    4. Tracks CBF intervention statistics for monitoring

    The penalty mechanism is crucial for On-Policy PPO:
    - Without penalty, PPO learns to rely on CBF and never learns safe behavior
    - With penalty, PPO learns to avoid actions that would trigger CBF

    Example:
        >>> # from isaaclab_rl.skrl import SkrlVecEnvWrapper
        >>> env = gym.make("Isaac-Navigation-Charge-Phase0")
        >>> safe_env = CBFActionWrapper(env, penalty_coeff=0.5)
        >>> safe_env = Sb3VecEnvWrapper(safe_env)
        >>> model = PPO("MlpPolicy", safe_env)
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[CBFWrapperConfig] = None,
    ):
        """Initialize CBF Action Wrapper

        Args:
            env: The environment to wrap (should be Isaac Lab env)
            config: CBF wrapper configuration
        """
        super().__init__(env)

        self.config = config or CBFWrapperConfig()

        # Initialize safety shield based on configuration
        self._init_safety_shield()

        # Statistics tracking
        self._stats = {
            "cbf_triggered_count": 0,
            "cbf_total_steps": 0,
            "cbf_total_correction": 0.0,
            "cbf_max_correction": 0.0,
        }

        # Last correction info (for reward penalty)
        self._last_corrections = np.zeros(env.unwrapped.num_envs)
        self._last_triggered = np.zeros(env.unwrapped.num_envs, dtype=bool)

    def _init_safety_shield(self):
        """Initialize the appropriate safety shield based on configuration"""
        if self.config.shield_type == "rule":
            self.shield = RuleBasedShield(
                safety_distance=self.config.safety_distance,
                danger_distance=self.config.danger_distance,
                emergency_distance=self.config.emergency_distance,
                max_linear_speed=self.config.max_linear_speed,
                max_angular_speed=self.config.max_angular_speed,
            )
        elif self.config.shield_type == "cbf":
            self.shield = CBFShield(
                safety_distance=self.config.safety_distance,
                alpha=self.config.alpha,
                max_linear_speed=self.config.max_linear_speed,
                max_angular_speed=self.config.max_angular_speed,
            )
        else:  # composite
            self.shield = CompositeShield(
                rule_shield=RuleBasedShield(
                    safety_distance=self.config.safety_distance,
                    danger_distance=self.config.danger_distance,
                    emergency_distance=self.config.emergency_distance,
                    max_linear_speed=self.config.max_linear_speed,
                    max_angular_speed=self.config.max_angular_speed,
                ),
                cbf_shield=CBFShield(
                    safety_distance=self.config.safety_distance,
                    alpha=self.config.alpha,
                    max_linear_speed=self.config.max_linear_speed,
                    max_angular_speed=self.config.max_angular_speed,
                ),
            )

    def action(self, action: np.ndarray) -> np.ndarray:
        """Filter action through CBF safety shield

        This is called by the environment before applying the action.

        Args:
            action: Raw action from PPO policy [num_envs, 2]

        Returns:
            Filtered safe action [num_envs, 2]
        """
        # Get environment state
        unwrapped = self.env.unwrapped

        # Convert action to tensor for processing
        if isinstance(action, np.ndarray):
            action_tensor = torch.from_numpy(action).to(device=unwrapped.device, dtype=torch.float32)
        else:
            action_tensor = action.to(device=unwrapped.device, dtype=torch.float32)

        # Get robot state from environment
        robot_pos = self._get_robot_position(unwrapped)
        robot_vel = self._get_robot_velocity(unwrapped)
        lidar_scan = self._get_lidar_scan(unwrapped)
        obstacles = self._get_obstacles(unwrapped)

        # Process each environment
        num_envs = unwrapped.num_envs
        safe_actions = []
        corrections = np.zeros(num_envs)
        triggered = np.zeros(num_envs, dtype=bool)

        for i in range(num_envs):
            act = action_tensor[i]
            pos = robot_pos[i] if robot_pos is not None else None
            vel = robot_vel[i] if robot_vel is not None else None
            scan = lidar_scan[i] if lidar_scan is not None else None
            obs_list = obstacles[i] if obstacles is not None else []

            # Apply safety filter
            safe_action = self._filter_single_action(act, pos, vel, scan, obs_list)

            # Calculate correction
            original_act = act.cpu().numpy() if isinstance(act, torch.Tensor) else act
            safe_act = np.array([safe_action.linear_speed, safe_action.angular_speed])
            correction = np.linalg.norm(original_act - safe_act)
            corrections[i] = correction
            triggered[i] = not safe_action.safe

            safe_actions.append(safe_act)

            # Update statistics
            if self.config.track_stats:
                self._stats["cbf_total_steps"] += 1
                if not safe_action.safe:
                    self._stats["cbf_triggered_count"] += 1
                    self._stats["cbf_total_correction"] += correction
                    self._stats["cbf_max_correction"] = max(
                        self._stats["cbf_max_correction"], correction
                    )

        # Store for reward penalty
        self._last_corrections = corrections
        self._last_triggered = triggered

        return np.array(safe_actions, dtype=np.float32)

    def _filter_single_action(
        self,
        action: torch.Tensor,
        robot_pos: Optional[torch.Tensor],
        robot_vel: Optional[torch.Tensor],
        lidar_scan: Optional[torch.Tensor],
        obstacles: List[Obstacle],
    ) -> SafetyAction:
        """Filter a single action through the safety shield

        Args:
            action: Action tensor [2]
            robot_pos: Robot position tensor
            robot_vel: Robot velocity tensor
            lidar_scan: Lidar scan tensor
            obstacles: List of obstacles

        Returns:
            SafetyAction with filtered action
        """
        # Handle missing data gracefully
        if robot_pos is None or robot_vel is None:
            # No state available, return original action
            linear, angular = action.tolist()
            return SafetyAction.safe_action(linear, angular)

        if lidar_scan is None:
            # No lidar data, use obstacle-based filtering only
            lidar_scan = torch.zeros(72)  # Default lidar size

        try:
            return self.shield.filter_action(action, robot_pos, robot_vel, lidar_scan, obstacles)
        except Exception as e:
            # On error, return safe action (stop)
            return SafetyAction.unsafe(0.0, 0.0, reason=f"Filter error: {e}")

    def get_cbf_penalty(self) -> np.ndarray:
        """Get CBF correction penalty for reward calculation

        Returns:
            Penalty array [num_envs] based on correction magnitude
        """
        return self.config.penalty_coeff * (self._last_corrections ** 2)

    def get_cbf_info(self) -> dict:
        """Get CBF statistics and last step info

        Returns:
            Dictionary with CBF statistics
        """
        info = {
            "cbf_triggered": self._last_triggered.copy(),
            "cbf_correction": self._last_corrections.copy(),
            "cbf_trigger_rate": self._get_trigger_rate(),
            "cbf_mean_correction": self._get_mean_correction(),
        }

        if self.config.track_stats:
            info.update(self._stats.copy())

        return info

    def reset_stats(self):
        """Reset CBF statistics tracking"""
        self._stats = {
            "cbf_triggered_count": 0,
            "cbf_total_steps": 0,
            "cbf_total_correction": 0.0,
            "cbf_max_correction": 0.0,
        }

    def _get_trigger_rate(self) -> float:
        """Calculate CBF trigger rate"""
        if self._stats["cbf_total_steps"] == 0:
            return 0.0
        return self._stats["cbf_triggered_count"] / self._stats["cbf_total_steps"]

    def _get_mean_correction(self) -> float:
        """Calculate mean correction magnitude"""
        if self._stats["cbf_triggered_count"] == 0:
            return 0.0
        return self._stats["cbf_total_correction"] / self._stats["cbf_triggered_count"]

    # ========================================================================
    # Environment state extraction methods
    # ========================================================================

    def _get_robot_position(self, env) -> Optional[torch.Tensor]:
        """Extract robot position from environment

        This is a placeholder - actual implementation depends on your environment structure.
        """
        try:
            # Try to get position from scene data
            if hasattr(env, "scene"):
                # Common pattern in Isaac Lab
                if hasattr(env.scene, "robot"):
                    return env.scene.robot.data.root_pos_w[:, :2]  # [x, y]
            return None
        except Exception:
            return None

    def _get_robot_velocity(self, env) -> Optional[torch.Tensor]:
        """Extract robot velocity from environment"""
        try:
            if hasattr(env, "scene"):
                if hasattr(env.scene, "robot"):
                    return env.scene.robot.data.root_vel_w[:, :2]  # [vx, vy]
            return None
        except Exception:
            return None

    def _get_lidar_scan(self, env) -> Optional[torch.Tensor]:
        """Extract lidar scan from environment"""
        try:
            # Try to get from sensors
            if hasattr(env, "sensors"):
                # Look for ray caster sensor
                for sensor_name, sensor in env.sensors.items():
                    if "ray" in sensor_name.lower() or "lidar" in sensor_name.lower():
                        data = sensor.data.out_hits
                        if data is not None and data.numel() > 0:
                            # Process ray caster data to distances
                            distances = torch.norm(data, dim=-1)
                            # Normalize to [0, 1] based on max range
                            max_range = 10.0  # Adjust based on your sensor config
                            return 1.0 - (distances / max_range).clamp(0, 1)
            return None
        except Exception:
            return None

    def _get_obstacles(self, env) -> Optional[List[List[Obstacle]]]:
        """Extract obstacles from environment

        Returns:
            List of obstacle lists for each environment
        """
        try:
            num_envs = env.num_envs
            obstacles_per_env = []

            for i in range(num_envs):
                env_obstacles = []

                # Try to get obstacle positions from the scene
                if hasattr(env, "scene"):
                    # Look for obstacles in the scene
                    # This is highly environment-specific
                    if hasattr(env.scene, "obstacles_cfg"):
                        for obs_name, obs_cfg in env.scene.obstacles_cfg.items():
                            # Get position from simulation
                            if hasattr(obs_cfg, "position"):
                                pos = obs_cfg.position
                                size = getattr(obs_cfg, "size", 1.0)
                                shape = getattr(obs_cfg, "shape", "circle")
                                env_obstacles.append(Obstacle(pos, size, shape))

                obstacles_per_env.append(env_obstacles)

            return obstacles_per_env
        except Exception:
            # Return empty obstacle lists if extraction fails
            return [[] for _ in range(env.num_envs)]


class CBFRewardWrapper(gym.RewardWrapper):
    """Reward wrapper that adds CBF correction penalty

    This wrapper modifies the reward to include a penalty when CBF
    modifies the policy's action. This prevents PPO from becoming
    dependent on the CBF safety filter.

    Should be used together with CBFActionWrapper:
        >>> env = gym.make("Isaac-Navigation-Charge-Phase0")
        >>> env = CBFActionWrapper(env)
        >>> env = CBFRewardWrapper(env)
    """

    def __init__(self, env: gym.Env, penalty_coeff: float = 0.5):
        """Initialize CBF Reward Wrapper

        Args:
            env: Environment (should be wrapped with CBFActionWrapper)
            penalty_coeff: Penalty coefficient for CBF correction
        """
        super().__init__(env)
        self.penalty_coeff = penalty_coeff

        # Get reference to CBFActionWrapper if available
        self.cbf_wrapper = None
        if hasattr(env, "unwrapped"):
            # Try to find CBFActionWrapper in the wrapper chain
            current = env
            while current is not None:
                if isinstance(current, CBFActionWrapper):
                    self.cbf_wrapper = current
                    break
                if hasattr(current, "env"):
                    current = current.env
                else:
                    break

    def reward(self, reward: np.ndarray) -> np.ndarray:
        """Apply CBF penalty to reward

        Args:
            reward: Original reward array [num_envs]

        Returns:
            Modified reward with CBF penalty
        """
        if self.cbf_wrapper is not None:
            penalty = self.cbf_wrapper.get_cbf_penalty()
            return reward - penalty

        # Fallback: no penalty if CBF wrapper not found
        return reward


class CBFInfoWrapper(gym.Wrapper):
    """Info wrapper that adds CBF statistics to episode info

    This wrapper adds CBF trigger information to the info dict so it can
    be logged by SKRL and viewed in TensorBoard.
    """

    def __init__(self, env: gym.Env):
        """Initialize CBF Info Wrapper

        Args:
            env: Environment (should be wrapped with CBFActionWrapper)
        """
        super().__init__(env)

        # Get reference to CBFActionWrapper
        self.cbf_wrapper = None
        current = env
        while current is not None:
            if isinstance(current, CBFActionWrapper):
                self.cbf_wrapper = current
                break
            if hasattr(current, "env"):
                current = current.env
            else:
                break

    def step(self, action):
        """Step environment and add CBF info to result"""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add CBF info if available
        if self.cbf_wrapper is not None:
            cbf_info = self.cbf_wrapper.get_cbf_info()
            # Merge into info dict (handle vectorized env)
            if isinstance(info, list):
                for i, env_info in enumerate(info):
                    env_info["cbf_triggered"] = cbf_info["cbf_triggered"][i]
                    env_info["cbf_correction"] = cbf_info["cbf_correction"][i]
            else:
                info.update(cbf_info)

        return obs, reward, terminated, truncated, info


def wrap_env_with_cbf(
    env: gym.Env,
    config: Optional[CBFWrapperConfig] = None,
    include_reward_wrapper: bool = True,
    include_info_wrapper: bool = True,
) -> gym.Env:
    """Convenience function to wrap environment with full CBF protection

    Args:
        env: The Isaac Lab environment
        config: CBF wrapper configuration
        include_reward_wrapper: Whether to include reward penalty wrapper
        include_info_wrapper: Whether to include info tracking wrapper

    Returns:
        Fully wrapped environment with CBF safety filter

    Example:
        >>> env = gym.make("Isaac-Navigation-Charge-Phase0")
        >>> env = wrap_env_with_cbf(env, penalty_coeff=0.5)
        >>> env = Sb3VecEnvWrapper(env)
        >>> model = PPO("MlpPolicy", env)
    """
    wrapped = CBFActionWrapper(env, config)

    if include_reward_wrapper:
        penalty_coeff = config.penalty_coeff if config else 0.5
        wrapped = CBFRewardWrapper(wrapped, penalty_coeff)

    if include_info_wrapper:
        wrapped = CBFInfoWrapper(wrapped)

    return wrapped
