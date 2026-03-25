# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment wrappers for Isaac Lab RL training."""

from typing import Optional

import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnvWrapper


# ============================================================================
# Constants for sanitization
# ============================================================================

REWARD_CLIP_MIN = -100.0
REWARD_CLIP_MAX = 100.0
SANITIZE_NAN_VALUE = 0.0
SANITIZE_POSINF_VALUE = 100.0
SANITIZE_NEGINF_VALUE = -100.0


# ============================================================================
# Observation Sanitization Wrapper
# ============================================================================

class SanitizeObservationsWrapper(VecEnvWrapper):
    """Wrapper to clean NaN and Inf values from observations.

    Problem: In some cases, the environment may produce nan/inf observations, causing:
    - VecNormalize normalization failure
    - PPO policy network to crash with "Expected parameter loc ... to satisfy the constraint Real()" error

    Solution: Replace all nan/inf with safe values before returning observations to SB3.
    """

    def __init__(
        self,
        venv: VecEnvWrapper,
        nan_value: float = SANITIZE_NAN_VALUE,
        posinf_value: float = SANITIZE_POSINF_VALUE,
        neginf_value: float = SANITIZE_NEGINF_VALUE,
    ):
        super().__init__(venv)
        self.nan_value = nan_value
        self.posinf_value = posinf_value
        self.neginf_value = neginf_value
        self.nan_count = 0

    def _sanitize(self, obs):
        """Clean NaN/Inf values from observations."""
        if isinstance(obs, torch.Tensor):
            has_nan = torch.isnan(obs).any()
            has_inf = torch.isinf(obs).any()
            if has_nan or has_inf:
                self.nan_count += 1
                if self.nan_count <= 10:
                    nan_count = torch.isnan(obs).sum().item()
                    inf_count = torch.isinf(obs).sum().item()
                    print(f"[WARNING] SanitizeObservationsWrapper: Found nan={has_nan} ({nan_count}), inf={has_inf} ({inf_count})")
                    print(f"         obs shape={obs.shape}, dtype={obs.dtype}")
                    print(f"         obs min={obs.min().item():.6f}, max={obs.max().item():.6f}")
                elif self.nan_count == 11:
                    print("[WARNING] SanitizeObservationsWrapper: Further nan/inf warnings suppressed")
            return torch.nan_to_num(
                obs,
                nan=self.nan_value,
                posinf=self.posinf_value,
                neginf=self.neginf_value
            )
        elif isinstance(obs, np.ndarray):
            has_nan = np.isnan(obs).any()
            has_inf = np.isinf(obs).any()
            if has_nan or has_inf:
                self.nan_count += 1
                if self.nan_count <= 10:
                    nan_count = np.isnan(obs).sum()
                    inf_count = np.isinf(obs).sum()
                    print(f"[WARNING] SanitizeObservationsWrapper: Found nan={has_nan} ({nan_count}), inf={has_inf} ({inf_count})")
                    print(f"         obs shape={obs.shape}, dtype={obs.dtype}")
                    obs_finite = obs[np.isfinite(obs)]
                    if len(obs_finite) > 0:
                        print(f"         obs min={obs_finite.min():.6f}, max={obs_finite.max():.6f}")
                elif self.nan_count == 11:
                    print("[WARNING] SanitizeObservationsWrapper: Further nan/inf warnings suppressed")
            return np.nan_to_num(
                obs,
                nan=self.nan_value,
                posinf=self.posinf_value,
                neginf=self.neginf_value
            )
        return obs

    def reset(self):
        """Reset environment and sanitize observations."""
        obs = self.venv.reset()
        return self._sanitize(obs)

    def step_async(self, actions):
        """Async step."""
        self.venv.step_async(actions)

    def step_wait(self):
        """Wait for step completion and return sanitized results."""
        obs, reward, done, info = self.venv.step_wait()

        # Clean nan/inf from observations
        obs = self._sanitize(obs)

        # Clean nan/inf from rewards (important!)
        if isinstance(reward, np.ndarray):
            if np.isnan(reward).any() or np.isinf(reward).any():
                nan_count = np.isnan(reward).sum()
                inf_count = np.isinf(reward).sum()
                print(f"[WARNING] SanitizeObservationsWrapper: Found nan/inf in reward! nan={nan_count}, inf={inf_count}")
                print(f"         reward min={reward[np.isfinite(reward)].min():.6f}, max={reward[np.isfinite(reward)].max():.6f}")
                reward = np.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=-10.0)
            # Extra protection: clip rewards to reasonable range (prevent gradient explosion)
            reward = np.clip(reward, REWARD_CLIP_MIN, REWARD_CLIP_MAX)

        return obs, reward, done, info


# ============================================================================
# Isaac Lab Metrics Wrapper
# ============================================================================

class IsaacLabMetricsWrapper(VecEnvWrapper):
    """Wrapper to collect Isaac Lab statistics and log to TensorBoard/WandB.

    Features:
    1. Convert Gymnasium API (returns tuple) to SB3 old API (returns obs)
    2. Extract Isaac Lab statistics from info dictionary
    3. Accumulate statistics from multiple episodes and calculate averages
    """

    def __init__(self, venv: VecEnvWrapper):
        super().__init__(venv)
        self.metrics_accumulator: dict = {}
        self.episode_count = 0

    def reset(self):
        """Reset environment and return obs (SB3 old API)."""
        result = self.venv.reset()

        # Parse Gymnasium return value
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        elif isinstance(result, tuple) and len(result) >= 4:
            obs = result[0]
            info = result[3] if len(result) > 3 else {}
        else:
            obs = result
            info = {}

        # Collect statistics from info
        self._collect_metrics_from_info(info)

        # Convert torch.Tensor to np.ndarray
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()

        return obs

    def step_async(self, actions):
        """Async step."""
        self.venv.step_async(actions)

    def step_wait(self):
        """Wait for step completion and return (obs, reward, done, info)."""
        result = self.venv.step_wait()

        # Handle Gymnasium API
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated | truncated
        elif isinstance(result, tuple) and len(result) == 4:
            obs, reward, done, info = result
            terminated = None
            truncated = None
        else:
            raise ValueError(f"Unexpected step_wait result: {len(result)} values")

        # Collect statistics from info list
        self._collect_metrics_from_info_list(info)

        # Convert torch.Tensor to np.ndarray
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if isinstance(reward, torch.Tensor):
            reward = reward.cpu().numpy()
        if isinstance(done, torch.Tensor):
            done = done.cpu().numpy()

        return obs, reward, done, info

    def _collect_metrics_from_info(self, info: dict) -> None:
        """Collect statistics from a single info dictionary."""
        if not isinstance(info, dict):
            return

        for key, value in info.items():
            if key.startswith("Episode_") or key in ["Episode_Reward", "Episode_Termination"]:
                if isinstance(value, (int, float)):
                    self._accumulate_metric(key, float(value))
                elif isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        self._accumulate_metric(key, float(value.item()))
                    else:
                        self._accumulate_metric(key, float(value.mean().item()))
                elif hasattr(value, "item"):
                    self._accumulate_metric(key, float(value.item()))

    def _collect_metrics_from_info_list(self, info_list: list) -> None:
        """Collect statistics from info list (one info per sub-environment).

        Note: Isaac Lab's extras["log"] contains statistics for all reset environments (averaged),
        Sb3VecEnvWrapper assigns this average to each terminated environment's info["episode"].
        Therefore, only take the first non-empty episode data to avoid duplicate accumulation.
        """
        if not isinstance(info_list, list):
            return

        collected = False

        for info in info_list:
            if collected:
                break

            if not isinstance(info, dict):
                continue

            if "episode" in info:
                if info["episode"] is not None and isinstance(info["episode"], dict):
                    episode_data = info["episode"]
                    for key, value in episode_data.items():
                        if isinstance(value, (int, float)):
                            self._accumulate_metric(key, float(value))
                        elif isinstance(value, torch.Tensor):
                            if value.numel() == 1:
                                self._accumulate_metric(key, float(value.item()))
                            else:
                                self._accumulate_metric(key, float(value.mean().item()))
                        elif hasattr(value, "item"):
                            self._accumulate_metric(key, float(value.item()))
                        else:
                            try:
                                self._accumulate_metric(key, float(value))
                            except (TypeError, ValueError):
                                pass
                    collected = True
                    self.episode_count += 1

    def _accumulate_metric(self, key: str, value: float) -> None:
        """Accumulate metric value."""
        if key not in self.metrics_accumulator:
            self.metrics_accumulator[key] = []
        self.metrics_accumulator[key].append(value)

    def get_metrics(self) -> dict:
        """Get accumulated statistics (averages)."""
        if not self.metrics_accumulator:
            return {}

        metrics = {}
        for key, values in self.metrics_accumulator.items():
            if values:
                metrics[key] = sum(values) / len(values)
        return metrics

    def get_latest_metrics(self) -> dict:
        """Get latest statistics (last episode)."""
        if not self.metrics_accumulator:
            return {}

        latest = {}
        for key, values in self.metrics_accumulator.items():
            if values:
                latest[key] = values[-1]
        return latest

    def clear_metrics(self) -> None:
        """Clear accumulated statistics."""
        self.metrics_accumulator = {}
        self.episode_count = 0
