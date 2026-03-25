# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom callbacks for Stable Baselines3 training."""

import time
from typing import Optional

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


# ============================================================================
# NaN Protection Callback
# ============================================================================

class NanProtectionCallback(BaseCallback):
    """Callback to detect and prevent policy network weights from becoming NaN.

    Problem: During training, network weights may become NaN, causing:
    - Policy outputs NaN actions
    - PPO crashes with "Expected parameter loc ... to satisfy the constraint Real()" error

    Solution: Check network weights after each training, stop training immediately when NaN is detected.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.last_safe_check: Optional[int] = None

    def _on_step(self) -> bool:
        # Check every 1000 steps (after rollout, before train)
        if self.n_calls % 1000 != 0:
            return True

        # Check policy network weights
        policy = self.model.policy
        has_nan = False
        nan_info = []

        for name, param in policy.named_parameters():
            if torch.isnan(param).any():
                has_nan = True
                nan_count = torch.isnan(param).sum().item()
                nan_info.append(f"  {name}: {nan_count} nan values")

        # Check log_std (common source of NaN)
        if hasattr(policy, "log_std"):
            if torch.isnan(policy.log_std).any():
                has_nan = True
                nan_info.append(f"  log_std: {torch.isnan(policy.log_std).sum().item()} nan values")

        if has_nan:
            print(f"\n{'='*70}")
            print(f"[CRITICAL] NaN detected in policy network weights at step {self.num_timesteps}!")
            print(f"Affected parameters:")
            for info in nan_info:
                print(info)
            print(f"{'='*70}\n")
            print(f"[INFO] Stopping training to prevent further corruption.")
            print(f"[INFO] Last safe checkpoint should be available.")
            return False  # Stop training

        return True


# ============================================================================
# WandB Callback
# ============================================================================

class WandBCallback(BaseCallback):
    """Custom WandB callback for logging training metrics.

    This callback logs SB3 training statistics to WandB after each rollout.

    Logged metrics categories:
    ┌─────────────────────────────────────────────────────────────────────┐
    │ 📊 Reward Curve                                                     │
    │ ├─ train/ep_rew_mean           Main reward curve                    │
    │ ├─ rewards/total               Total reward                         │
    │ ├─ rewards/progress            Progress reward                      │
    │ ├─ rewards/goal_reached        Goal reached reward                  │
    │ ├─ rewards/velocity_toward     Velocity toward goal                 │
    │ ├─ rewards/collision_penalty   Collision penalty                    │
    │ ├─ rewards/safety_field        Safety field penalty                 │
    │ ├─ rewards/action_smoothness   Action smoothness penalty            │
    │ └─ rewards/time_out            Time out penalty                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │ 🔥 Entropy                                                         │
    │ ├─ train/policy_entropy        Policy entropy                       │
    │ ├─ train/entropy_loss          Entropy loss                         │
    │ └─ train/ent_coef              Entropy coefficient                  │
    ├─────────────────────────────────────────────────────────────────────┤
    │ 📈 Other Metrics                                                    │
    │ ├─ rollouts/ep_len_mean        Episode length                       │
    │ ├─ success_rate                Success rate                         │
    │ └─ train/learning_rate         Learning rate                        │
    └─────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, verbose: int = 1, metrics_wrapper: Optional["IsaacLabMetricsWrapper"] = None):
        super().__init__(verbose)
        self._last_log_time = time.time()
        self.metrics_wrapper = metrics_wrapper
        self._rollout_count = 0
        self._reward_history: list[float] = []

    def _on_training_start(self) -> None:
        """Called at the start of training."""
        try:
            import wandb
            if wandb.run is not None:
                print(f"[WandB] Monitoring run: {wandb.run.name}")
                print(f"[WandB] Project: {wandb.run.project}")
                print(f"[WandB] URL: {wandb.run.url}")
        except Exception as e:
            print(f"[WARNING] WandB check failed: {e}")

    def _on_step(self) -> bool:
        """Called after each step.

        Must return True to continue training, False to abort.
        """
        # Only log in _on_rollout_end(), avoid duplication
        return True

    def _extract_env_metrics(self) -> dict:
        """Extract accumulated statistics from metrics_wrapper."""
        env_metrics = {}

        try:
            if self.metrics_wrapper is not None:
                env_metrics = self.metrics_wrapper.get_metrics()
        except Exception as e:
            # Silently ignore extraction errors
            pass

        return env_metrics

    def _on_rollout_end(self) -> None:
        """Called after each rollout, log training statistics."""
        self._rollout_count += 1

        try:
            import wandb
            if wandb.run is None:
                return

            # Build metrics dictionary to log
            metrics = {}

            # ================================================================
            # 📊 Extract training metrics from SB3 logger (reward curve + entropy)
            # ================================================================
            if hasattr(self.model, "logger"):
                logger = self.model.logger
                if hasattr(logger, "name_to_value"):
                    # Debug: print available metrics in logger (first 3 rollouts)
                    if self._rollout_count <= 3 and len(logger.name_to_value) > 0:
                        print(f"[DEBUG] Logger keys: {list(logger.name_to_value.keys())}")

                    for name, value in logger.name_to_value.items():
                        # Fix: handle more numeric types (numpy, torch, etc.)
                        try:
                            if hasattr(value, "item"):
                                value = value.item()
                            value_float = float(value)
                        except (TypeError, ValueError):
                            continue  # Skip non-convertible values

                        # Reorganize metric names for better WandB visualization
                        # Map rollouts/ep_rew_mean to train/ep_rew_mean (main reward curve)
                        if name == "rollouts/ep_rew_mean":
                            metrics["train/ep_rew_mean"] = value_float
                            metrics[name] = value_float  # Keep original name for compatibility
                            # Record to reward history
                            self._reward_history.append(value_float)
                            # Calculate moving average (last 100 rollouts)
                            if len(self._reward_history) >= 100:
                                ma_100 = sum(self._reward_history[-100:]) / 100
                                metrics["train/ep_rew_mean_ma100"] = ma_100
                        elif name == "rollouts/ep_len_mean":
                            metrics[name] = value_float
                        # Entropy-related metrics (group to train/)
                        elif "entropy" in name.lower():
                            metrics[f"train/{name}"] = value_float
                            metrics[name] = value_float
                        # Other training metrics
                        elif name.startswith("train/"):
                            metrics[name] = value_float
                        else:
                            metrics[name] = value_float

            # ================================================================
            # 📊 Extract reward component breakdown from environment (reward curve analysis)
            # ================================================================
            env_metrics = self._extract_env_metrics()

            # Reorganize reward metrics to rewards/ namespace
            reward_mapping = {
                "Episode_Reward/progress_to_goal": "rewards/progress",
                "Episode_Reward/reaching_goal": "rewards/goal_reached",
                "Episode_Reward/velocity_toward_goal": "rewards/velocity_toward",
                "Episode_Reward/collision_penalty_reward": "rewards/collision_penalty",
                "Episode_Reward/safety_field_penalty": "rewards/safety_field",
                "Episode_Reward/action_smoothness": "rewards/action_smoothness",
                "Episode_Reward/time_out_penalty": "rewards/time_out",
            }

            for old_name, new_name in reward_mapping.items():
                if old_name in env_metrics:
                    metrics[new_name] = env_metrics[old_name]

            # Calculate total reward (sum of components)
            total_reward = sum(
                env_metrics.get(k, 0)
                for k in reward_mapping.keys()
            )
            if total_reward != 0:
                metrics["rewards/total"] = total_reward

            # Termination condition statistics
            for key, value in env_metrics.items():
                if key.startswith("Episode_Termination/"):
                    metrics[key] = value

            # ================================================================
            # 📈 Calculate success rate
            # ================================================================
            goal_reached_key = "Episode_Termination/goal_reached"

            if goal_reached_key in metrics:
                goal_reached = metrics[goal_reached_key]

                # Success rate: goal_reached / (goal_reached + collision + time_out)
                all_terminations = sum(
                    v for k, v in metrics.items()
                    if k.startswith("Episode_Termination/")
                )

                if all_terminations > 0:
                    success_rate = (goal_reached / all_terminations) * 100
                    metrics["success_rate"] = success_rate
                    metrics["failure_rate"] = 100 - success_rate

            # ================================================================
            # 🔥 Additional entropy analysis metrics
            # ================================================================
            if "train/policy_entropy" in metrics:
                entropy = metrics["train/policy_entropy"]
                # Higher entropy = more exploration, lower entropy = more exploitation
                if entropy < 0.1:
                    metrics["train/exploration_status"] = "exploitation"
                elif entropy > 0.5:
                    metrics["train/exploration_status"] = "exploration"
                else:
                    metrics["train/exploration_status"] = "transition"

            # ================================================================
            # 📊 Log to WandB
            # ================================================================
            if metrics:
                wandb.log(metrics, step=self.num_timesteps)
                current_time = time.time()
                time_since_last_log = current_time - self._last_log_time
                self._last_log_time = current_time

                if self.verbose > 0:
                    # Print key metrics summary
                    reward_msg = f"Reward: {metrics.get('train/ep_rew_mean', 0):.2f}"
                    entropy_msg = f"Entropy: {metrics.get('train/policy_entropy', metrics.get('train/entropy_loss', 0)):.4f}"
                    success_msg = f"Success: {metrics.get('success_rate', 0):.1f}%"

                    # Debug: show available metrics (first few rollouts)
                    if self._rollout_count <= 3:
                        available_keys = [k for k in metrics.keys() if not k.startswith("Episode_")]
                        print(f"[DEBUG] Available metrics: {available_keys}")

                    print(f"[WandB] Rollout {self._rollout_count} | Step {self.num_timesteps} | "
                          f"{reward_msg} | {entropy_msg} | {success_msg}")

                # Clear accumulator for next round
                if self.metrics_wrapper is not None:
                    self.metrics_wrapper.clear_metrics()

        except Exception as e:
            if self.verbose > 0:
                print(f"[WARNING] WandB logging failed: {e}")
                import traceback
                traceback.print_exc()

    def _on_training_end(self) -> None:
        """Called at the end of training, log final statistics summary."""
        try:
            import wandb
            if wandb.run is not None:
                # Log training summary
                summary = {
                    "training/total_rollouts": self._rollout_count,
                    "training/total_timesteps": self.num_timesteps,
                }

                # Calculate reward statistics
                if self._reward_history:
                    import numpy as np
                    summary["training/final_reward_mean"] = np.mean(self._reward_history[-100:])
                    summary["training/best_reward_mean"] = np.max(self._reward_history)
                    summary["training/reward_std"] = np.std(self._reward_history[-100:])

                wandb.run.summary.update(summary)

                print(f"\n{'='*60}")
                print(f"[WandB] Training completed summary")
                print(f"[WandB] Total rollouts: {self._rollout_count}")
                print(f"[WandB] Total timesteps: {self.num_timesteps}")
                if self._reward_history:
                    print(f"[WandB] Final reward (100 avg): {summary.get('training/final_reward_mean', 0):.2f}")
                    print(f"[WandB] Best reward: {summary.get('training/best_reward_mean', 0):.2f}")
                print(f"[WandB] View results: {wandb.run.url}")
                print(f"{'='*60}\n")
        except Exception as e:
            print(f"[WARNING] WandB finalization failed: {e}")
