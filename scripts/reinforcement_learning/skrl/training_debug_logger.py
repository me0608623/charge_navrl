"""Comprehensive Training Debug Logger for Isaac Lab RL.

Collects per-step metrics across 5 categories:
  A) Reward: per-term instant + raw vs normalized
  B) Episode / Termination: success/collision/timeout counts, episode length
  C) Action pipeline: raw vs applied, clipping rate, saturation ratio
  D) Agent motion: velocity, progress, obstacle distance
  E) PPO internal: advantage, grad norms, KL, clip fraction, param diff

Design:
  - `step()` is called EVERY env step with (actions, rewards, dones, infos, env)
  - `on_ppo_update()` is called after each PPO _update with metrics dict
  - `get_and_reset()` returns aggregated stats for WandB/TensorBoard and clears buffers
  - Zero modification to Isaac Lab core code; reads env internals via public APIs

Usage in WandBSequentialTrainer:
    debug_logger = TrainingDebugLogger(env)
    ...
    debug_logger.step(actions, rewards, terminated, truncated, infos)
    ...
    metrics = debug_logger.get_and_reset()
    wandb.log(metrics, step=timestep)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import torch
import numpy as np


class TrainingDebugLogger:
    """Collects per-step training diagnostics and returns aggregated metrics."""

    def __init__(
        self,
        env,
        log_interval: int = 128,
    ):
        """
        Args:
            env: The SKRL-wrapped environment (AACIsaacLabWrapper).
                 We unwrap to get the Isaac Lab ManagerBasedRLEnv.
            log_interval: Expected number of steps between get_and_reset() calls
                          (used only for buffer pre-allocation hint, not enforced).
        """
        self._wrapped_env = env
        self._isaac_env = self._unwrap_to_isaac(env)
        self._log_interval = log_interval
        self._step_count = 0
        self._update_count = 0

        # ── One-time diagnostics flags ──
        self._diag_printed = False
        self._motion_error_printed = False
        self._reward_error_printed = False
        self._action_error_printed = False
        self._lidar_error_printed = False

        # ── Accumulators (lists of scalars per step) ──
        # A) Reward
        self._reward_total: list[float] = []
        self._reward_total_min: list[float] = []
        self._reward_total_max: list[float] = []
        # Per-term instant reward (keyed by term name)
        self._reward_terms: dict[str, list[float]] = defaultdict(list)

        # B) Episode / Termination
        self._done_count: list[int] = []       # num done per step
        self._episode_lengths: list[float] = []  # length of finished episodes
        self._term_counts: dict[str, int] = defaultdict(int)  # cumulative
        self._total_episodes: int = 0

        # C) Action pipeline
        self._raw_action_mean: list[float] = []
        self._raw_action_std: list[float] = []
        self._raw_action_min: list[float] = []
        self._raw_action_max: list[float] = []
        self._applied_action_mean: list[float] = []
        self._applied_action_std: list[float] = []
        self._applied_action_min: list[float] = []
        self._applied_action_max: list[float] = []
        self._action_clipping_rate: list[float] = []
        self._action_saturation_rate: list[float] = []
        self._action_safety_override_rate: list[float] = []

        # D) Motion state
        self._lin_vel_mean: list[float] = []
        self._lin_vel_max: list[float] = []
        self._ang_vel_mean: list[float] = []
        self._ang_vel_max: list[float] = []
        self._progress_per_step: list[float] = []  # delta distance to goal
        self._min_obstacle_dist: list[float] = []
        # LiDAR diagnostics
        self._lidar_min_raw: list[float] = []
        self._lidar_max_range_hit_ratio: list[float] = []

        # E) PPO internal (filled by on_ppo_update)
        self._ppo_metrics: dict[str, list[float]] = defaultdict(list)

        # Print unwrap result
        if self._isaac_env is not None:
            print(f"[DebugLogger] ✓ Isaac Lab env unwrap successful: {type(self._isaac_env).__name__}")
            if hasattr(self._isaac_env, 'scene'):
                scene_keys = list(self._isaac_env.scene.keys()) if hasattr(self._isaac_env.scene, 'keys') else dir(self._isaac_env.scene)
                print(f"[DebugLogger]   Scene entities: {scene_keys[:10]}")
        else:
            print(f"[DebugLogger] ✗ WARNING: Isaac Lab env unwrap FAILED! Motion/action/reward-term metrics will be UNAVAILABLE.")
            print(f"[DebugLogger]   Wrapper chain: {self._get_wrapper_chain(env)}")

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def step(
        self,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: dict,
    ) -> None:
        """Called every env step. All tensors are [num_envs, ...].

        This must be FAST — only scalar reductions on GPU tensors.
        """
        self._step_count += 1

        # ── One-time diagnostic print ──
        if not self._diag_printed:
            self._print_diagnostic(actions, rewards, terminated, truncated, infos)
            self._diag_printed = True

        # ── A) Reward ──
        with torch.no_grad():
            r = rewards.view(-1)
            self._reward_total.append(r.mean().item())
            self._reward_total_min.append(r.min().item())
            self._reward_total_max.append(r.max().item())

        # Per-term reward from Isaac Lab extras
        self._collect_reward_terms(infos)

        # ── B) Episode / Termination ──
        dones = (terminated.view(-1) | truncated.view(-1))
        num_done = dones.sum().item()
        self._done_count.append(int(num_done))

        if num_done > 0:
            self._total_episodes += int(num_done)
            self._collect_termination_stats(infos, dones)
            self._collect_episode_lengths(dones)

        # ── C) Action pipeline ──
        self._collect_action_stats(actions)

        # ── D) Motion state ──
        self._collect_motion_stats(dones)

    def on_ppo_update(self, metrics: dict[str, float]) -> None:
        """Called after each PPO _update with internal metrics.

        Args:
            metrics: dict like {"advantage_mean": 0.01, "grad_norm_actor": 0.5, ...}
        """
        self._update_count += 1
        for k, v in metrics.items():
            self._ppo_metrics[k].append(v)

    def get_and_reset(self) -> dict[str, float]:
        """Return aggregated metrics dict (flat keys for WandB/TB) and clear buffers."""
        m: dict[str, float] = {}
        n = max(len(self._reward_total), 1)

        # ── A) Reward ──
        if self._reward_total:
            m["Debug_Reward/instant_mean"] = np.mean(self._reward_total)
            m["Debug_Reward/instant_min"] = np.mean(self._reward_total_min)
            m["Debug_Reward/instant_max"] = np.mean(self._reward_total_max)

        for term_name, vals in self._reward_terms.items():
            if vals:
                m[f"Debug_RewardTerm/{term_name}_mean"] = np.mean(vals)

        # ── B) Episode / Termination ──
        if self._done_count:
            total_done = sum(self._done_count)
            m["Debug_Episode/done_rate_per_step"] = total_done / n
            m["Debug_Episode/total_episodes_window"] = self._total_episodes

            if self._total_episodes > 0:
                for term_name, count in self._term_counts.items():
                    m[f"Debug_Termination/{term_name}_count"] = count
                    m[f"Debug_Termination/{term_name}_rate"] = count / max(self._total_episodes, 1)

        if self._episode_lengths:
            m["Debug_Episode/length_mean"] = np.mean(self._episode_lengths)
            m["Debug_Episode/length_min"] = np.min(self._episode_lengths)
            m["Debug_Episode/length_max"] = np.max(self._episode_lengths)

        # ── C) Action pipeline ──
        if self._raw_action_mean:
            m["Debug_Action/raw_mean"] = np.mean(self._raw_action_mean)
            m["Debug_Action/raw_std"] = np.mean(self._raw_action_std)
            m["Debug_Action/raw_min"] = np.mean(self._raw_action_min)
            m["Debug_Action/raw_max"] = np.mean(self._raw_action_max)
        if self._applied_action_mean:
            m["Debug_Action/applied_mean"] = np.mean(self._applied_action_mean)
            m["Debug_Action/applied_std"] = np.mean(self._applied_action_std)
            m["Debug_Action/applied_min"] = np.mean(self._applied_action_min)
            m["Debug_Action/applied_max"] = np.mean(self._applied_action_max)
        if self._action_clipping_rate:
            m["Debug_Action/clipping_rate"] = np.mean(self._action_clipping_rate)
        if self._action_saturation_rate:
            m["Debug_Action/saturation_rate"] = np.mean(self._action_saturation_rate)
        if self._action_safety_override_rate:
            m["Debug_Action/safety_override_rate"] = np.mean(self._action_safety_override_rate)

        # ── D) Motion state ──
        if self._lin_vel_mean:
            m["Debug_Motion/lin_vel_mean"] = np.mean(self._lin_vel_mean)
            m["Debug_Motion/lin_vel_max"] = np.mean(self._lin_vel_max)
        if self._ang_vel_mean:
            m["Debug_Motion/ang_vel_mean"] = np.mean(self._ang_vel_mean)
            m["Debug_Motion/ang_vel_max"] = np.mean(self._ang_vel_max)
        if self._progress_per_step:
            m["Debug_Motion/progress_per_step_mean"] = np.mean(self._progress_per_step)
            m["Debug_Motion/progress_per_step_std"] = np.std(self._progress_per_step) if len(self._progress_per_step) > 1 else 0.0
        if self._min_obstacle_dist:
            m["Debug_Motion/min_obstacle_dist_mean"] = np.mean(self._min_obstacle_dist)
            m["Debug_Motion/min_obstacle_dist_min"] = np.min(self._min_obstacle_dist)
        # LiDAR diagnostics
        if self._lidar_min_raw:
            m["Debug_Motion/lidar_min_raw"] = np.mean(self._lidar_min_raw)
        if self._lidar_max_range_hit_ratio:
            m["Debug_Motion/lidar_max_range_hit_ratio"] = np.mean(self._lidar_max_range_hit_ratio)

        # ── E) PPO internal ──
        for k, vals in self._ppo_metrics.items():
            if vals:
                m[f"Debug_PPO/{k}"] = np.mean(vals)

        m["Debug_PPO/update_count"] = self._update_count
        m["Debug_Meta/steps_in_window"] = self._step_count

        # Reset all buffers
        self._reset()
        return m

    # ──────────────────────────────────────────────────────────────────────
    # One-time diagnostic print
    # ──────────────────────────────────────────────────────────────────────

    def _print_diagnostic(
        self,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: dict,
    ) -> None:
        """Print one-time diagnostic to verify tensor shapes, dtypes, and env structure."""
        print("\n" + "=" * 70)
        print("[DebugLogger] === ONE-TIME DIAGNOSTIC (step 1) ===")
        print("=" * 70)
        print(f"  actions:    shape={actions.shape}, dtype={actions.dtype}, device={actions.device}")
        print(f"              min={actions.min().item():.4f}, max={actions.max().item():.4f}, mean={actions.float().mean().item():.4f}")
        print(f"  rewards:    shape={rewards.shape}, dtype={rewards.dtype}")
        print(f"              min={rewards.min().item():.4f}, max={rewards.max().item():.4f}, mean={rewards.mean().item():.4f}")
        print(f"  terminated: shape={terminated.shape}, dtype={terminated.dtype}, sum={terminated.sum().item()}")
        print(f"  truncated:  shape={truncated.shape}, dtype={truncated.dtype}, sum={truncated.sum().item()}")
        print(f"  infos keys: {list(infos.keys())[:20]}")
        if "log" in infos:
            log_keys = list(infos["log"].keys())
            print(f"  infos['log'] keys ({len(log_keys)}): {log_keys[:15]}")

        isaac_env = self._isaac_env
        if isaac_env is not None:
            print(f"\n  Isaac env type: {type(isaac_env).__name__}")
            if hasattr(isaac_env, 'num_envs'):
                print(f"  num_envs: {isaac_env.num_envs}")

            # Check robot access
            try:
                robot = isaac_env.scene["robot"]
                lin_vel = robot.data.root_lin_vel_w
                print(f"  robot.data.root_lin_vel_w: shape={lin_vel.shape}, dtype={lin_vel.dtype}")
                speed = torch.norm(lin_vel[:, :2], dim=1)
                print(f"  speed (L2 norm xy):        min={speed.min().item():.4f}, max={speed.max().item():.4f}, mean={speed.mean().item():.4f}")
                ang_vel = robot.data.root_ang_vel_w[:, 2]
                print(f"  angular_vel_z:             min={ang_vel.min().item():.4f}, max={ang_vel.max().item():.4f}")
            except Exception as e:
                print(f"  ✗ robot access FAILED: {type(e).__name__}: {e}")

            # Check reward manager
            try:
                rm = isaac_env.reward_manager
                term_names = list(rm._episode_sums.keys())
                print(f"  reward_manager terms ({len(term_names)}): {term_names}")
                for name, tensor in rm._episode_sums.items():
                    print(f"    {name}: shape={tensor.shape}, mean={tensor.mean().item():.4f}")
            except Exception as e:
                print(f"  ✗ reward_manager access FAILED: {type(e).__name__}: {e}")

            # Check action manager
            try:
                terms = list(isaac_env.action_manager._terms.values())
                action_term = terms[0]
                print(f"  action_term type: {type(action_term).__name__}")
                if hasattr(action_term, 'processed_actions'):
                    pa = action_term.processed_actions
                    print(f"  processed_actions: shape={pa.shape}, min={pa.min().item():.4f}, max={pa.max().item():.4f}")
                if hasattr(action_term, '_current_velocity'):
                    cv = action_term._current_velocity
                    print(f"  _current_velocity: shape={cv.shape}, min={cv.min().item():.4f}, max={cv.max().item():.4f}")
            except Exception as e:
                print(f"  ✗ action_manager access FAILED: {type(e).__name__}: {e}")

            # Check LiDAR sensor
            try:
                sensors = isaac_env.scene.sensors if hasattr(isaac_env.scene, 'sensors') else {}
                print(f"  scene.sensors keys: {list(sensors.keys()) if hasattr(sensors, 'keys') else 'N/A'}")
                sensor = sensors.get("lidar", None)
                if sensor is not None:
                    hits = sensor.data.ray_hits_w
                    print(f"  lidar.ray_hits_w: shape={hits.shape}, dtype={hits.dtype}")
                    # Check for max_range/NaN patterns
                    pos = sensor.data.pos_w[:, :2]
                    dists = torch.norm(hits[:, :, :2] - pos.unsqueeze(1), dim=-1)
                    dists_valid = dists[torch.isfinite(dists)]
                    if dists_valid.numel() > 0:
                        print(f"  lidar distances:   min={dists_valid.min().item():.3f}, max={dists_valid.max().item():.3f}, mean={dists_valid.mean().item():.3f}")
                    nan_ratio = (~torch.isfinite(dists)).float().mean().item()
                    print(f"  lidar NaN/Inf ratio: {nan_ratio:.4f}")
                    if hasattr(sensor.cfg, 'max_distance'):
                        print(f"  lidar max_distance: {sensor.cfg.max_distance}")
                else:
                    print(f"  ✗ lidar sensor not found in scene.sensors")
            except Exception as e:
                print(f"  ✗ lidar access FAILED: {type(e).__name__}: {e}")
        else:
            print(f"\n  ✗ Isaac env is None — all env-dependent metrics will be EMPTY")

        print("=" * 70 + "\n")

    # ──────────────────────────────────────────────────────────────────────
    # Internal collection helpers
    # ──────────────────────────────────────────────────────────────────────

    def _collect_reward_terms(self, infos: dict) -> None:
        """Extract per-step reward breakdown from Isaac Lab RewardManager.

        The RewardManager stores _episode_sums which accumulate per episode.
        We track deltas to get per-step values.
        """
        isaac_env = self._isaac_env
        if isaac_env is None:
            return

        try:
            rm = isaac_env.reward_manager
            # The RewardManager._episode_sums is Dict[str, Tensor[num_envs]]
            # Each step, the manager adds func() * weight * dt to these sums.
            # We track snapshots to compute per-step deltas.
            if not hasattr(self, '_prev_episode_sums'):
                self._prev_episode_sums = {}
                for name, tensor in rm._episode_sums.items():
                    self._prev_episode_sums[name] = tensor.clone()
                return

            for name, tensor in rm._episode_sums.items():
                prev = self._prev_episode_sums.get(name)
                if prev is not None:
                    delta = tensor - prev
                    # Handle episode resets (episode_sums get zeroed)
                    # When reset happens, delta can be very negative; use abs threshold
                    # to detect and skip reset frames
                    mean_delta = delta.mean().item()
                    if abs(mean_delta) < 50.0:  # Skip if looks like a reset spike
                        self._reward_terms[name].append(mean_delta)
                self._prev_episode_sums[name] = tensor.clone()

        except Exception as e:
            if not self._reward_error_printed:
                print(f"[DebugLogger] ✗ _collect_reward_terms FAILED: {type(e).__name__}: {e}")
                self._reward_error_printed = True

    def _collect_termination_stats(self, infos: dict, dones: torch.Tensor) -> None:
        """Count termination reasons from infos['log'] Episode_Termination data."""
        log_data = infos.get("log", {})
        num_done = dones.sum().item()

        for k, v in log_data.items():
            if k.startswith("Episode_Termination/"):
                term_name = k.split("/", 1)[1]
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                # val is mean proportion across reset envs
                count = val * num_done
                self._term_counts[term_name] += count

    def _collect_episode_lengths(self, dones: torch.Tensor) -> None:
        """Record lengths of finished episodes."""
        isaac_env = self._isaac_env
        if isaac_env is None or not hasattr(isaac_env, 'episode_length_buf'):
            return
        try:
            done_lengths = isaac_env.episode_length_buf[dones].float()
            if done_lengths.numel() > 0:
                for length in done_lengths.cpu().tolist():
                    self._episode_lengths.append(length)
        except (IndexError, RuntimeError):
            pass

    def _collect_action_stats(self, actions: torch.Tensor) -> None:
        """Collect raw (policy output) and applied (post-processing) action stats."""
        with torch.no_grad():
            # Raw actions from policy network
            a = actions.float().view(-1)
            self._raw_action_mean.append(a.mean().item())
            self._raw_action_std.append(a.std().item() if a.numel() > 1 else 0.0)
            self._raw_action_min.append(a.min().item())
            self._raw_action_max.append(a.max().item())

        # Applied actions from the action manager
        isaac_env = self._isaac_env
        if isaac_env is None:
            return

        try:
            action_term = list(isaac_env.action_manager._terms.values())[0]

            # Processed actions (what's actually sent to sim)
            pa = action_term.processed_actions.float().view(-1)
            self._applied_action_mean.append(pa.mean().item())
            self._applied_action_std.append(pa.std().item() if pa.numel() > 1 else 0.0)
            self._applied_action_min.append(pa.min().item())
            self._applied_action_max.append(pa.max().item())

            # Discrete action: check saturation (velocity at max_v or 0)
            if hasattr(action_term, '_current_velocity') and hasattr(action_term, 'cfg'):
                vel = action_term._current_velocity
                max_v = action_term.cfg.max_linear_velocity
                num_envs = vel.numel()
                if num_envs > 0:
                    at_max = (vel >= max_v - 1e-4).sum().item()
                    at_zero = (vel <= 1e-4).sum().item()
                    self._action_saturation_rate.append((at_max + at_zero) / num_envs)

            # Check how many accelerations were clipped by saturation logic
            if hasattr(action_term, '_applied_accelerations') and hasattr(action_term, '_raw_actions'):
                raw = action_term._raw_actions
                applied_a = action_term._applied_accelerations[:, 0]
                # Saturation override: raw requested non-zero accel but applied is zero
                if raw.numel() > 0:
                    # Use the accel_table to get the raw acceleration
                    idx_a = raw[:, 0].round().long().clamp(0, action_term.cfg.num_accel_bins - 1)
                    a_raw = action_term.accel_table[idx_a]
                    overridden = ((a_raw.abs() > 1e-6) & (applied_a.abs() < 1e-6)).float()
                    self._action_safety_override_rate.append(overridden.mean().item())

        except Exception as e:
            if not self._action_error_printed:
                print(f"[DebugLogger] ✗ _collect_action_stats FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                self._action_error_printed = True

    def _collect_motion_stats(self, dones: torch.Tensor) -> None:
        """Collect robot velocity, progress-to-goal, obstacle distance.

        Args:
            dones: [N] boolean tensor indicating which envs just finished.
        """
        isaac_env = self._isaac_env
        if isaac_env is None:
            return

        try:
            # Robot velocity (nan_to_num: physics engine may produce NaN after resets)
            robot = isaac_env.scene["robot"]
            lin_vel = torch.nan_to_num(robot.data.root_lin_vel_w[:, :2], nan=0.0)
            ang_vel = torch.nan_to_num(robot.data.root_ang_vel_w[:, 2], nan=0.0)
            speed = torch.norm(lin_vel, dim=1)

            speed_mean = speed.mean().item()
            speed_max = speed.max().item()

            self._lin_vel_mean.append(speed_mean)
            self._lin_vel_max.append(speed_max)
            self._ang_vel_mean.append(ang_vel.abs().mean().item())
            self._ang_vel_max.append(ang_vel.abs().max().item())

            # Progress per step (distance reduction to goal)
            robot_pos = torch.nan_to_num(robot.data.root_pos_w[:, :2], nan=0.0)
            goal_pos = None
            if hasattr(isaac_env, '_local_goal_world') and isaac_env._local_goal_world is not None:
                goal_pos = isaac_env._local_goal_world
            else:
                try:
                    goal_pos = isaac_env.command_manager.get_command("goal_command")[:, :2]
                except (KeyError, AttributeError):
                    pass

            if goal_pos is not None:
                d_curr = torch.norm(goal_pos - robot_pos, dim=1)

                if hasattr(self, '_prev_goal_dist_debug'):
                    # ── FIX: Clear prev distance for envs that just reset ──
                    # When dones=True, the env has reset and d_prev is from old episode.
                    # Mask out these envs to avoid cross-episode jumps.
                    valid_mask = ~dones  # Only use envs that did NOT just reset
                    progress = self._prev_goal_dist_debug - d_curr
                    # Also skip unusually large jumps (safety net)
                    valid_mask = valid_mask & (progress.abs() < 2.0)
                    if valid_mask.any():
                        self._progress_per_step.append(progress[valid_mask].mean().item())

                self._prev_goal_dist_debug = d_curr.clone()

        except Exception as e:
            if not self._motion_error_printed:
                print(f"[DebugLogger] ✗ _collect_motion_stats FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                self._motion_error_printed = True
            return

        # LiDAR obstacle distance (separate try block — don't skip motion on lidar fail)
        try:
            self._collect_lidar_stats(isaac_env)
        except Exception as e:
            if not self._lidar_error_printed:
                print(f"[DebugLogger] ✗ _collect_lidar_stats FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                self._lidar_error_printed = True

    def _collect_lidar_stats(self, isaac_env) -> None:
        """Collect LiDAR-based obstacle distance metrics."""
        sensors = isaac_env.scene.sensors if hasattr(isaac_env.scene, 'sensors') else {}
        sensor = sensors.get("lidar", None)
        if sensor is None:
            return

        hits = sensor.data.ray_hits_w  # [N, num_rays, 3]
        pos = sensor.data.pos_w        # [N, 3]

        # Compute distances from sensor position to hit points (2D XY)
        dists = torch.norm(hits[:, :, :2] - pos[:, :2].unsqueeze(1), dim=-1)  # [N, num_rays]

        # Identify max-range hits (NaN, Inf, or very large = no obstacle)
        max_range = sensor.cfg.max_distance if hasattr(sensor.cfg, 'max_distance') else 20.0
        invalid = ~torch.isfinite(dists) | (dists > max_range - 0.01)

        # Replace invalid with max_range for min computation
        dists_clamped = torch.where(invalid, torch.tensor(max_range, device=dists.device), dists)

        # Per-env min obstacle distance
        d_min_per_env = dists_clamped.min(dim=1)[0]  # [N]
        self._min_obstacle_dist.append(d_min_per_env.mean().item())

        # Raw min (without clamping — for diagnostics)
        dists_finite = torch.where(torch.isfinite(dists), dists, torch.tensor(max_range, device=dists.device))
        self._lidar_min_raw.append(dists_finite.min().item())

        # Max-range hit ratio (fraction of rays that didn't hit anything)
        self._lidar_max_range_hit_ratio.append(invalid.float().mean().item())

    # ──────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _unwrap_to_isaac(env):
        """Walk the wrapper chain to find the Isaac Lab ManagerBasedRLEnv."""
        current = env
        for _ in range(20):  # safety limit
            if hasattr(current, 'reward_manager'):
                return current
            # SKRL IsaacLabWrapper uses _unwrapped property
            if hasattr(current, '_unwrapped'):
                unwrapped = current._unwrapped
                if hasattr(unwrapped, 'reward_manager'):
                    return unwrapped
            if hasattr(current, 'unwrapped'):
                unwrapped = current.unwrapped
                if hasattr(unwrapped, 'reward_manager'):
                    return unwrapped
            if hasattr(current, '_env'):
                current = current._env
            elif hasattr(current, 'env'):
                current = current.env
            else:
                break
        return None

    @staticmethod
    def _get_wrapper_chain(env) -> str:
        """Return a string showing the wrapper chain for debugging."""
        chain = []
        current = env
        for _ in range(20):
            chain.append(type(current).__name__)
            if hasattr(current, '_env'):
                current = current._env
            elif hasattr(current, 'env'):
                current = current.env
            else:
                break
        return " → ".join(chain)

    def _reset(self) -> None:
        """Clear all per-window accumulators (keep cumulative counters)."""
        self._step_count = 0
        # Keep _update_count and _total_episodes as they are cumulative

        self._reward_total.clear()
        self._reward_total_min.clear()
        self._reward_total_max.clear()
        for v in self._reward_terms.values():
            v.clear()

        self._done_count.clear()
        self._episode_lengths.clear()
        self._term_counts.clear()
        self._total_episodes = 0

        self._raw_action_mean.clear()
        self._raw_action_std.clear()
        self._raw_action_min.clear()
        self._raw_action_max.clear()
        self._applied_action_mean.clear()
        self._applied_action_std.clear()
        self._applied_action_min.clear()
        self._applied_action_max.clear()
        self._action_clipping_rate.clear()
        self._action_saturation_rate.clear()
        self._action_safety_override_rate.clear()

        self._lin_vel_mean.clear()
        self._lin_vel_max.clear()
        self._ang_vel_mean.clear()
        self._ang_vel_max.clear()
        self._progress_per_step.clear()
        self._min_obstacle_dist.clear()
        self._lidar_min_raw.clear()
        self._lidar_max_range_hit_ratio.clear()

        for v in self._ppo_metrics.values():
            v.clear()
