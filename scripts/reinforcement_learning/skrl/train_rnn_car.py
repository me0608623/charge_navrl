#!/usr/bin/env python3
"""
train_rnn_car.py — Multi-Agent Modular RNN 訓練 (Isaac Lab 版) v5

完全對齊 Warp Drive train_rnn_car.py 設計:
  - Charge (=Spot): ModularRNN-A2CK (vanilla RNN, per-head entropy, loss clamping)
  - Obstacle: FC-PPO (learnable, parameter shared, 取代腳本移動)
  - Alternating training: 2:1 charge:obstacle (Warp Drive train_goal_rate=3)

v5 WD-aligned changes:
  - γ = 0.984 constant (WD: 1-(1-0.92)/fps, fps=5)
  - PolicyHead [256,256,256,512], ValueHead [256,256,256,512,512]
  - Vanilla RNN (not GRU), memory_dim=30, module_connect_dim=12
  - aux_lr=0 (preprocess trained by RL gradient only)
  - vf_coeff=0.025 (WD: spot_vf_loss_coeff)
  - A2C mode (--use_a2c): no PPO clipping, single epoch
  - Per-phase obstacle_speed_rate: 0.8→0.85→1.15
  - LR: rl_head=0.0002, rnn=0.0005

Usage:
  PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_rnn_car.py \\
    --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \\
    --num_envs 4096 --headless --seed 1 --use_a2c \\
    --run_name wd_aligned_v5_s1
"""

# ============================================================================
# 1. CLI + AppLauncher (must be before any Isaac imports)
# ============================================================================

import argparse
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Charge + Obstacle with Modular RNN (Warp Drive port)")

# --- Isaac Lab 標準 ---
parser.add_argument("--task", type=str, default="Isaac-Navigation-Charge-VLP16-Curriculum-NavRL")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=1)

# --- Charge 訓練超參 ---
parser.add_argument("--timesteps", type=int, default=262_144,
                    help="Total timesteps. 262144 = 1024 updates x 256 rollout")
parser.add_argument("--rollout_length", type=int, default=256, help="Steps per rollout")
parser.add_argument("--ppo_epochs", type=int, default=4, help="PPO learning epochs")
parser.add_argument("--mini_batches", type=int, default=32, help="Mini-batches per epoch")
parser.add_argument("--lr", type=float, default=2e-4,
                    help="Charge RL head LR. WD: spot_lr=0.0002")
parser.add_argument("--rnn_lr", type=float, default=5e-4,
                    help="Charge RNN module LR. WD: spot_rnn_model_lr=0.0005")
parser.add_argument("--aux_lr", type=float, default=0.0,
                    help="Charge preprocess aux loss LR. WD: spot_preprocess_model_lr=0 (disabled)")
parser.add_argument("--gamma", type=float, default=0.984,
                    help="Discount factor. WD: 1-(1-0.92)/fps = 0.984 (fps=5)")
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--clip_eps", type=float, default=0.2)
parser.add_argument("--vf_coeff", type=float, default=0.025,
                    help="Value loss coefficient. WD: spot_vf_loss_coeff=0.025")
parser.add_argument("--ent_coeff", type=float, default=0.0,
                    help="Legacy single entropy coeff. 0=use per-head WD coeffs from curriculum")
parser.add_argument("--ent_coeff_linear", type=float, default=0.0,
                    help="Head1 (linear accel) entropy coeff. 0=auto from curriculum. "
                         "WD: 0.10 (Phase 1), 0.30 (Phase 2+)")
parser.add_argument("--ent_coeff_angular", type=float, default=0.0,
                    help="Head2 (angular vel) entropy coeff. 0=auto from curriculum. "
                         "WD: 0.375 (all phases)")
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--use_a2c", action="store_true", default=False,
                    help="Use A2C (no PPO clipping). WD: A2CK mode")
parser.add_argument("--tbptt_len", type=int, default=0,
                    help="TBPTT sequence length. 0=disabled (step-by-step). WD: 15")
parser.add_argument("--lr_decay", type=float, default=0.0,
                    help="Linear LR decay factor per iteration. 0=no decay. "
                         "WD: uses ParamScheduler")

# --- Modular RNN ---
parser.add_argument("--hidden_dim", type=int, default=30,
                    help="RNN hidden state dim. WD: memory_dim=30")
parser.add_argument("--preprocess_dim", type=int, default=12,
                    help="Preprocess feature dim. WD: module_connect_dim=12")
parser.add_argument("--fc_dim", type=int, default=48, help="FC front dim before RNN")
parser.add_argument("--rnn_type", type=str, default="RNN", choices=["RNN", "GRU"],
                    help="RNN type. WD: vanilla RNN")

# --- Obstacle Policy ---
parser.add_argument("--obs_lr", type=float, default=3e-4, help="Obstacle policy LR")
parser.add_argument("--obs_ent_coeff", type=float, default=0.1,
                    help="Obstacle entropy coeff (high = explore more)")
parser.add_argument("--obs_speed_limit", type=float, default=0.8,
                    help="Obstacle max speed (m/s), relative to charge max 1.0")
parser.add_argument("--train_goal_rate", type=int, default=3,
                    help="Warp Drive: every N iters, 1 trains obstacle (rest train charge)")
parser.add_argument("--obs_reward_mode", type=str, default="zero",
                    choices=["zero", "approach"],
                    help="zero=Warp Drive style, approach=weak adversarial")
parser.add_argument("--max_active_obstacles", type=int, default=10)

# --- Randomization (Warp Drive: obs_size_rand + floor_width_bias) ---
parser.add_argument("--obs_size_rand", type=float, default=0.0,
                    help="Obstacle collision radius randomization range (m). "
                         "0=auto from curriculum. WD: 0.1→0.4 across phases")
parser.add_argument("--obs_collision_base", type=float, default=0.9,
                    help="Base collision distance (m). Randomized ± obs_size_rand/2")
parser.add_argument("--scene_bound_rand", type=float, default=0.0,
                    help="Scene bound randomization range (m). "
                         "0=auto from curriculum. Effective bound = base ± rand/2")
parser.add_argument("--scene_bound_base", type=float, default=7.0,
                    help="Base obstacle movement boundary (m)")

# --- Logging ---
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--log_interval", type=int, default=10, help="Log every N iterations")
parser.add_argument("--save_interval", type=int, default=100, help="Save checkpoint every N iterations")
parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint path")

# --- Env config overrides ---
parser.add_argument("--reward_mode", type=str, default="current")
parser.add_argument("--curriculum_version", type=str, default=None,
                    help="Curriculum version (default: warp_drive_v1 for multi-agent)")
parser.add_argument("--lidar_no_noise", action="store_true", default=False)
parser.add_argument("--no_domain_randomization", action="store_true", default=False)
parser.add_argument("--reward_speed_v05", action="store_true", default=False)
parser.add_argument("--play", action="store_true", default=False,
                    help="Inference only — no PPO training, just run rollout with loaded checkpoint")
parser.add_argument("--initial_stage", type=int, default=1,
                    help="Force curriculum to start at this stage (1-8)")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

headless_mode = getattr(args_cli, "headless", False) or "--headless" in sys.argv
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================
# 2. Post-launcher imports
# ============================================================================

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: register tasks
from isaaclab_tasks.utils.hydra import hydra_task_config

sys.path.insert(0, str(Path(__file__).parent))
from modular_rnn_models import (
    LidarStateExtractor,
    PreprocessRNN,
    PolicyHead,
    ValueHead,
    RNNStateManager,
    ObstaclePolicyFC,
    ObstacleValueFC,
    OBS_POLICY_OBS_DIM,
    LIDAR_START,
    LIDAR_END,
    NUM_BINS,
)

print("[INFO] Multi-Agent Modular RNN Training v5 — WD-Aligned (A2CK + vanilla RNN)")


# ============================================================================
# Running observation normalizer
# ============================================================================

class RunningNormalizer:
    """Welford's online algorithm for running mean/var normalization."""

    def __init__(self, shape, device, clip=5.0):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = 1e-4
        self.clip = clip

    @torch.no_grad()
    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x):
        return torch.clamp((x - self.mean) / (self.var.sqrt() + 1e-8), -self.clip, self.clip)


# ============================================================================
# Charge Rollout Buffer (PPO + aux loss)
# ============================================================================

class ChargeRolloutBuffer:
    def __init__(self, num_steps, num_envs, rl_input_dim, obs_dim, hidden_dim, device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.rl_inputs = torch.zeros(num_steps, num_envs, rl_input_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, 2, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.raw_obs = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.hiddens = torch.zeros(num_steps, num_envs, hidden_dim, device=device)
        self.ptr = 0

    def add(self, rl_input, action, log_prob, reward, value, done, raw_ob, hidden):
        i = self.ptr
        self.rl_inputs[i] = rl_input
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = done
        self.raw_obs[i] = raw_ob
        self.hiddens[i] = hidden.squeeze(0)
        self.ptr += 1

    def reset(self):
        self.ptr = 0


# ============================================================================
# Obstacle Rollout Buffer
# ============================================================================

class ObstacleRolloutBuffer:
    """Flat buffer: each obstacle is an independent sample."""

    def __init__(self, num_steps, num_envs, max_obstacles, obs_dim, act_dim, device):
        self.num_steps = num_steps
        self.B = num_envs * max_obstacles  # flat batch dimension
        self.device = device
        self.obs = torch.zeros(num_steps, self.B, obs_dim, device=device)
        self.actions = torch.zeros(num_steps, self.B, act_dim, device=device)
        self.log_probs = torch.zeros(num_steps, self.B, device=device)
        self.rewards = torch.zeros(num_steps, self.B, device=device)
        self.values = torch.zeros(num_steps, self.B, device=device)
        self.dones = torch.zeros(num_steps, self.B, device=device)
        self.ptr = 0

    def add(self, obs, action, log_prob, reward, value, done):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = done
        self.ptr += 1

    def reset(self):
        self.ptr = 0


# ============================================================================
# Obstacle environment interface
# ============================================================================

def build_obstacle_obs(env_unwrapped, max_obstacles: int, device) -> torch.Tensor:
    """Read obstacle + robot state, build [num_envs, N, 9] obs tensor.

    Per-obstacle obs (body-frame of env):
      own_local_xy(2) + own_vel(2) + robot_rel_xy(2) + robot_vel(2) + d_wall(1) = 9D
    """
    num_envs = env_unwrapped.num_envs
    env_origins = env_unwrapped.scene.env_origins  # [num_envs, 3]

    # Robot state
    robot_pos_w = env_unwrapped.scene["robot"].data.root_pos_w[:, :3]  # [E, 3]
    robot_local = robot_pos_w - env_origins  # [E, 3]
    robot_vel_w = env_unwrapped.scene["robot"].data.root_lin_vel_w[:, :2]  # [E, 2]

    # Ensure velocity cache exists
    if not hasattr(env_unwrapped, "_obstacle_velocities"):
        env_unwrapped._obstacle_velocities = torch.zeros(num_envs, max_obstacles, 2, device=device)

    obs_all = torch.zeros(num_envs, max_obstacles, OBS_POLICY_OBS_DIM, device=device)

    # Build obstacle cache if needed
    if not hasattr(env_unwrapped, "_obs_policy_cache"):
        env_unwrapped._obs_policy_cache = []
        for i in range(max_obstacles):
            name = f"obstacle_{i}"
            if name in env_unwrapped.scene.keys():
                env_unwrapped._obs_policy_cache.append(env_unwrapped.scene[name])
            else:
                env_unwrapped._obs_policy_cache.append(None)

    # Per-env scene bounds (randomized per episode reset)
    if hasattr(env_unwrapped, "_scene_bounds"):
        bound_limits = env_unwrapped._scene_bounds  # [E]
    else:
        bound_limits = torch.full((num_envs,), 7.0, device=device)

    for i, obstacle in enumerate(env_unwrapped._obs_policy_cache):
        if obstacle is None:
            continue
        obs_pos_w = obstacle.data.root_pos_w[:, :3]  # [E, 3]
        obs_local = obs_pos_w - env_origins  # [E, 3]

        # Check if active (Z > 0)
        active = obs_pos_w[:, 2] > 0.0  # [E]

        obs_xy = obs_local[:, :2]  # [E, 2]
        obs_vel = env_unwrapped._obstacle_velocities[:, i, :]  # [E, 2]
        robot_rel = robot_local[:, :2] - obs_xy  # [E, 2] relative robot pos

        # d_wall: min distance to any boundary (per-env bound)
        d_left = obs_xy[:, 0] + bound_limits
        d_right = bound_limits - obs_xy[:, 0]
        d_bottom = obs_xy[:, 1] + bound_limits
        d_top = bound_limits - obs_xy[:, 1]
        d_wall = torch.stack([d_left, d_right, d_bottom, d_top], dim=-1).min(dim=-1).values  # [E]
        d_wall = torch.max(torch.min(d_wall, bound_limits), torch.zeros_like(d_wall))

        obs_all[:, i, 0:2] = obs_xy
        obs_all[:, i, 2:4] = obs_vel
        obs_all[:, i, 4:6] = robot_rel
        obs_all[:, i, 6:8] = robot_vel_w
        obs_all[:, i, 8] = d_wall

        # Zero out inactive obstacles
        inactive = ~active
        if inactive.any():
            obs_all[inactive, i, :] = 0.0

    return obs_all  # [num_envs, N, 9]


def apply_obstacle_actions(env_unwrapped, actions: torch.Tensor, max_obstacles: int,
                           dt: float = 0.2, speed_limit: float = 0.8, bound_limit: float = 7.0):
    """Apply obstacle policy actions: velocity → position update → write_root_pose_to_sim.

    Only moves ACTIVE obstacles (Z > 0). Hidden obstacles (Z=-10) are skipped.
    Uses per-env _scene_bounds if available (scene size randomization).

    Args:
        actions: [num_envs, N, 2] velocity commands in [-1, 1], scaled by speed_limit
    """
    env_origins = env_unwrapped.scene.env_origins  # [E, 3]

    velocity = actions * speed_limit  # [E, N, 2]

    # Per-env scene bounds (randomized per episode reset)
    if hasattr(env_unwrapped, "_scene_bounds"):
        bound_limits = env_unwrapped._scene_bounds  # [E]
    else:
        bound_limits = torch.full((env_unwrapped.num_envs,), bound_limit, device=actions.device)

    # Update velocity cache for obs_functions.py
    if not hasattr(env_unwrapped, "_obstacle_velocities"):
        env_unwrapped._obstacle_velocities = torch.zeros(
            env_unwrapped.num_envs, max_obstacles, 2, device=actions.device)

    for i, obstacle in enumerate(env_unwrapped._obs_policy_cache):
        if obstacle is None:
            continue

        pos_w = obstacle.data.root_pos_w.clone()  # [E, 3]
        active = pos_w[:, 2] > 0.0  # only move visible obstacles

        if not active.any():
            # Zero velocity for inactive obstacles
            env_unwrapped._obstacle_velocities[:, i, :] = 0.0
            continue

        # Local frame
        local_xy = pos_w[:, :2] - env_origins[:, :2]

        # Apply velocity only to active envs
        vel_i = velocity[:, i, :] * active.unsqueeze(-1).float()  # zero for inactive
        local_xy = local_xy + vel_i * dt

        # Geofence with soft bounce: reflect when hitting per-env boundary
        for dim in range(2):
            over_max = local_xy[:, dim] > bound_limits
            under_min = local_xy[:, dim] < -bound_limits
            local_xy[over_max, dim] = 2 * bound_limits[over_max] - local_xy[over_max, dim]
            local_xy[under_min, dim] = -2 * bound_limits[under_min] - local_xy[under_min, dim]
            # Final safety clamp (element-wise with per-env bounds)
            local_xy[:, dim] = torch.max(torch.min(local_xy[:, dim], bound_limits), -bound_limits)

        # Write back to world frame (only active)
        new_pos = pos_w.clone()
        new_pos[:, :2] = local_xy + env_origins[:, :2]
        # Preserve Z for inactive (keep at -10)
        new_pos[:, :2] = torch.where(active.unsqueeze(-1), new_pos[:, :2], pos_w[:, :2])

        quat = torch.zeros(env_unwrapped.num_envs, 4, device=actions.device)
        quat[:, 0] = 1.0  # identity quaternion
        pose = torch.cat([new_pos, quat], dim=-1)  # [E, 7]

        obstacle.write_root_pose_to_sim(pose)

        # Update velocity cache (only for active)
        env_unwrapped._obstacle_velocities[:, i, :] = vel_i


def compute_obstacle_reward(env_unwrapped, obs_obs: torch.Tensor, max_obstacles: int,
                            mode: str = "zero") -> torch.Tensor:
    """Compute per-obstacle reward.

    Args:
        obs_obs: [num_envs, N, 9] obstacle observations
        mode: "zero" (Warp Drive) or "approach" (weak adversarial)
    Returns:
        [num_envs * N] flat reward
    """
    num_envs = obs_obs.shape[0]
    N = max_obstacles

    if mode == "zero":
        return torch.zeros(num_envs * N, device=obs_obs.device)

    # "approach" mode: small reward for being close to robot
    robot_rel = obs_obs[:, :, 4:6]  # [E, N, 2]
    dist_to_robot = robot_rel.norm(dim=-1)  # [E, N]

    # Reward: closer → higher, max 0.1 at distance 0
    reward = (0.1 * (2.0 - dist_to_robot).clamp(0.0, 2.0) / 2.0)  # [E, N]

    # Speed penalty
    obs_vel = obs_obs[:, :, 2:4]
    speed = obs_vel.norm(dim=-1)
    reward = reward - 0.01 * speed

    return reward.reshape(-1)  # [E*N]


# ============================================================================
# Warp Drive Charge Reward — 完全對齊原作 per-phase sparse reward
#
# WD spot reward 結構 (flat terrain, car_mode):
#   1. spot_reward_get_goal: +40 on goal reached
#   2. spot_penalty_hit: -5 ~ -200 on collision (wall/obstacle/any)
#   3. spot_cost_operate: per-step action cost (Phase 1 only, ~0.03/fps)
#   4. spot_reward_on_floor: 0 (all phases, flat terrain)
#
# 相比 Isaac Lab NavRL dense rewards，WD 更 sparse：
#   - 沒有 velocity_to_goal (方向速度獎勵)
#   - 沒有 safe_progress (安全進度 PBRS)
#   - 沒有 safety_log_distance (LiDAR 距離獎勵)
# ============================================================================

def compute_wd_charge_reward(
    env_unwrapped,
    actions: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    penalty_hit: float,
    reward_get_goal: float,
    cost_operate: float,
    rl_fps: float = 5.0,
    cost_turn_rate: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute Warp Drive style sparse reward for charge agent.

    Uses termination_manager per-term buffers to detect goal_reached vs collision.

    Returns:
        (reward, breakdown) where breakdown contains per-component tensors:
          - goal_reward: [N] goal reaching reward (WD: car goal reward)
          - wall_hit_reward: [N] wall/static collision penalty (WD: car static obstacle reward)
          - obs_hit_reward: [N] dynamic obstacle collision penalty (WD: car dynamic obstacle reward)
          - floor_reward: [N] always 0 for flat terrain (WD: car floor reward)
          - action_reward: [N] action cost (WD: car dynamic reward / cost_operate)
          - goal_reached: [N] bool
          - wall_collision: [N] bool
          - obs_collision: [N] bool
          - other_death: [N] bool (tipped/explosion)
    """
    N = terminated.shape[0]
    device = terminated.device
    reward = torch.zeros(N, device=device)

    terminated_flat = terminated.squeeze(-1).bool() if terminated.dim() > 1 else terminated.bool()
    truncated_flat = truncated.squeeze(-1).bool() if truncated.dim() > 1 else truncated.bool()

    # --- Detect goal_reached vs collision type from termination_manager ---
    goal_reached = torch.zeros(N, dtype=torch.bool, device=device)
    wall_collision = torch.zeros(N, dtype=torch.bool, device=device)
    obs_collision = torch.zeros(N, dtype=torch.bool, device=device)
    other_death = torch.zeros(N, dtype=torch.bool, device=device)

    try:
        tm = env_unwrapped.termination_manager
        for name in tm._term_names:
            buf = tm.get_term(name)
            if buf is None:
                continue
            if "goal_reached" in name or "reaching_goal" in name:
                goal_reached = goal_reached | buf.bool()
            elif "wall_collision" in name:
                wall_collision = wall_collision | buf.bool()
            elif "obstacle_collision" in name:
                obs_collision = obs_collision | buf.bool()
            elif "collision" in name:
                # Generic collision — attribute to obstacle if not wall
                obs_collision = obs_collision | buf.bool()
            elif "tipped" in name or "explosion" in name or "flying" in name:
                other_death = other_death | buf.bool()
    except (AttributeError, RuntimeError):
        # Fallback
        obs_collision = terminated_flat & ~truncated_flat

    any_collision = wall_collision | obs_collision | other_death

    # --- Per-component reward ---
    goal_reward = torch.zeros(N, device=device)
    wall_hit_reward = torch.zeros(N, device=device)
    obs_hit_reward = torch.zeros(N, device=device)
    action_reward = torch.zeros(N, device=device)

    # Goal reached: +reward_get_goal
    goal_reward[goal_reached] = reward_get_goal
    reward += goal_reward

    # Wall/static collision: penalty (WD: car static obstacle reward)
    wall_hit_reward[wall_collision] = penalty_hit
    reward += wall_hit_reward

    # Obstacle/dynamic collision: penalty (WD: car dynamic obstacle reward)
    obs_hit_reward[obs_collision & ~wall_collision] = penalty_hit
    reward += obs_hit_reward

    # Other death (tipped/explosion): penalty
    other_hit = other_death & ~wall_collision & ~obs_collision
    reward[other_hit] += penalty_hit

    # --- Action cost (WD: only meaningful when cost_operate > 0, typically Phase 1) ---
    if cost_operate > 0 and actions is not None:
        cost_per_step = cost_operate / rl_fps
        nomal_acc = (actions[:, 0].float() - 9.0).abs() / 9.0
        nomal_turn = (actions[:, 1].float() - 9.0).abs() / 9.0
        action_reward = cost_per_step * (1.0 - nomal_acc) ** 2
        action_reward = action_reward + cost_per_step * cost_turn_rate * (1.0 - nomal_turn) ** 2
        alive = ~terminated_flat
        action_reward = action_reward * alive.float()
        reward += action_reward

    breakdown = {
        "goal_reward": goal_reward,              # WD: car goal reward
        "wall_hit_reward": wall_hit_reward,      # WD: car static obstacle reward
        "obs_hit_reward": obs_hit_reward,        # WD: car dynamic obstacle reward
        "floor_reward": torch.zeros(N, device=device),  # WD: car floor reward (0 for flat)
        "action_reward": action_reward,          # WD: car dynamic reward (action cost)
        "goal_reached": goal_reached,
        "wall_collision": wall_collision,
        "obs_collision": obs_collision,
        "other_death": other_death,
    }

    return reward, breakdown


# ============================================================================
# PPO Utilities
# ============================================================================

def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


def sample_action(logits):
    logits_a = logits[:, :NUM_BINS]
    logits_w = logits[:, NUM_BINS:]
    dist_a = Categorical(logits=logits_a)
    dist_w = Categorical(logits=logits_w)
    action_a = dist_a.sample()
    action_w = dist_w.sample()
    log_prob = dist_a.log_prob(action_a) + dist_w.log_prob(action_w)
    entropy = dist_a.entropy() + dist_w.entropy()
    return torch.stack([action_a, action_w], dim=-1), log_prob, entropy


def evaluate_actions(logits, actions):
    """Returns (log_prob, entropy_linear, entropy_angular)."""
    logits_a = logits[:, :NUM_BINS]
    logits_w = logits[:, NUM_BINS:]
    dist_a = Categorical(logits=logits_a)
    dist_w = Categorical(logits=logits_w)
    log_prob = dist_a.log_prob(actions[:, 0]) + dist_w.log_prob(actions[:, 1])
    return log_prob, dist_a.entropy(), dist_w.entropy()


def ppo_update_continuous(policy, value_fn, buffer, optimizer, epochs, mini_batches,
                          clip_eps, vf_coeff, ent_coeff, max_grad_norm, gamma, gae_lambda):
    """Generic PPO update for continuous action policies (obstacle)."""
    T = buffer.ptr
    B = buffer.B

    # Bootstrap value for last step
    with torch.no_grad():
        last_obs = buffer.obs[T - 1]
        last_value = value_fn(last_obs).squeeze(-1)

    advantages, returns = compute_gae(
        buffer.rewards[:T], buffer.values[:T], buffer.dones[:T],
        last_value, gamma, gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    flat_obs = buffer.obs[:T].reshape(-1, buffer.obs.shape[-1])
    flat_actions = buffer.actions[:T].reshape(-1, buffer.actions.shape[-1])
    flat_log_probs = buffer.log_probs[:T].reshape(-1)
    flat_adv = advantages.reshape(-1)
    flat_ret = returns.reshape(-1)

    batch_size = T * B
    mini_batch_size = batch_size // mini_batches

    losses = []
    for _ in range(epochs):
        indices = torch.randperm(batch_size, device=flat_obs.device)
        for start in range(0, batch_size, mini_batch_size):
            end = min(start + mini_batch_size, batch_size)
            idx = indices[start:end]

            mb_obs = flat_obs[idx]
            mb_act = flat_actions[idx]
            mb_old_lp = flat_log_probs[idx]
            mb_adv = flat_adv[idx]
            mb_ret = flat_ret[idx]

            new_lp, entropy = policy.evaluate(mb_obs, mb_act)
            new_val = value_fn(mb_obs).squeeze(-1)

            ratio = (new_lp - mb_old_lp).exp()
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
            p_loss = -torch.min(surr1, surr2).mean()
            v_loss = F.mse_loss(new_val, mb_ret)
            loss = p_loss + vf_coeff * v_loss - ent_coeff * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(value_fn.parameters()), max_grad_norm)
            optimizer.step()
            losses.append(p_loss.item())

    return np.mean(losses) if losses else 0.0


# ============================================================================
# WandB Metrics — Warp Drive 完整指標移植 (spot→charge)
#
# Warp Drive 原始指標結構:
#   A) {policy} loss, {policy} loss coefficient entropy
#   A2) {policy}_module_feture_{N}_loss, {policy}_preprcess_loss
#   B1) Per-agent reward decomposition + event stats
#   C) Phase table (artifact)
#
# Isaac Lab 對應:
#   charge = spot, obstacle = obstacle, goal = goal (virtual)
#   Isaac Lab reward terms → Warp Drive display_reward mapping
# ============================================================================

# Isaac Lab reward term → Warp Drive metric name 映射
_REWARD_TERM_MAP = {
    # charge (spot) 相關
    "reaching_goal":            "charge goal reward",
    "velocity_to_goal":         "charge goal reward",       # 累加到 goal reward
    "safe_progress":            "charge dynamic reward",    # WD: spot dynamic reward
    "safety_log_distance":      "charge dynamic obstacle reward",
    "near_obstacle_penalty":    "charge static obstacle reward",
    "collision_terminal":       "charge hit penalty",
    "time_penalty":             "charge floor reward",      # WD: spot floor reward (存活懲罰)
    "velocity_too_low":         "charge floor reward",
    "acceleration_penalty":     "charge action penalty",
    "angular_velocity_penalty": "charge action penalty",
    "potential_progress":       "charge dynamic reward",
}


class MetricsCollector:
    """Warp Drive 風格完整指標收集器。

    指標命名對照 Warp Drive custom_trainer.py:
      spot → charge, car → charge
      goal → goal (unchanged)
      obstacle → obstacle
    """

    def __init__(self, num_envs: int, max_obstacles: int, device):
        self.num_envs = num_envs
        self.max_obstacles = max_obstacles
        self.device = device

        # --- Per-env episode accumulators ---
        self._ep_reward = torch.zeros(num_envs, device=device)
        self._ep_length = torch.zeros(num_envs, device=device)
        self._ep_steps_alive = torch.zeros(num_envs, device=device)  # 存活步數

        # --- Per-env WD reward decomposition accumulators ---
        # 每個 env 的 episode 內累加各 reward 分量，episode 結束時記錄
        self._ep_goal_reward = torch.zeros(num_envs, device=device)
        self._ep_wall_hit_reward = torch.zeros(num_envs, device=device)
        self._ep_obs_hit_reward = torch.zeros(num_envs, device=device)
        self._ep_floor_reward = torch.zeros(num_envs, device=device)
        self._ep_action_reward = torch.zeros(num_envs, device=device)

        # --- Completed episode reward decomposition ---
        self._completed_goal_reward: list[float] = []
        self._completed_wall_hit_reward: list[float] = []
        self._completed_obs_hit_reward: list[float] = []
        self._completed_floor_reward: list[float] = []
        self._completed_action_reward: list[float] = []

        # --- Completed episode stats ---
        self._completed_rewards: list[float] = []
        self._completed_lengths: list[float] = []
        self._completed_alive: list[float] = []

        # --- Termination counters ---
        self._goal_reached = 0
        self._collision = 0           # total: wall + obstacle + geometric
        self._wall_collision = 0
        self._obstacle_collision = 0
        self._timeout = 0
        self._tipped_over = 0
        self._total_eps = 0
        self._first_step_deaths = 0   # dies_at_birth: done at step <= 2

        # --- Reward term accumulators (Isaac Lab Episode_Reward/) ---
        self._reward_terms: dict[str, list[float]] = {}

        # --- Warp Drive display_reward accumulators ---
        self._wd_metrics: dict[str, list[float]] = {}

        # --- Curriculum info ---
        self._curriculum_info: dict[str, float] = {}

        # --- Obstacle policy per-step stats ---
        self._obs_speeds: list[float] = []
        self._obs_distances: list[float] = []  # distance to robot

    def step(self, obs, reward, done, info,
             obs_obs: torch.Tensor | None = None,
             obs_actions: torch.Tensor | None = None,
             reward_breakdown: dict[str, torch.Tensor] | None = None):
        """Record one env step.

        Args:
            obs_obs: [E, N, 9] obstacle observations (for obstacle metrics)
            obs_actions: [E, N, 2] obstacle actions (for speed metrics)
            reward_breakdown: dict from compute_wd_charge_reward (per-component tensors)
        """
        self._ep_reward += reward.reshape(-1)
        self._ep_length += 1.0
        self._ep_steps_alive += (1.0 - done.reshape(-1).float())

        # --- Accumulate WD reward decomposition ---
        if reward_breakdown is not None:
            self._ep_goal_reward += reward_breakdown["goal_reward"]
            self._ep_wall_hit_reward += reward_breakdown["wall_hit_reward"]
            self._ep_obs_hit_reward += reward_breakdown["obs_hit_reward"]
            self._ep_floor_reward += reward_breakdown["floor_reward"]
            self._ep_action_reward += reward_breakdown["action_reward"]

        # --- Parse info["log"] ---
        if "log" in info:
            for key, val in info["log"].items():
                if key.startswith("Curriculum/"):
                    k = key.replace("Curriculum/", "")
                    try:
                        self._curriculum_info[k] = val.item() if isinstance(val, torch.Tensor) else float(val)
                    except (ValueError, TypeError):
                        pass

                if key.startswith("Episode_Reward/"):
                    k = key.replace("Episode_Reward/", "")
                    v = val.item() if isinstance(val, torch.Tensor) else float(val)
                    self._reward_terms.setdefault(k, []).append(v)

                    # Map to Warp Drive metric name
                    wd_name = _REWARD_TERM_MAP.get(k)
                    if wd_name:
                        self._wd_metrics.setdefault(wd_name, []).append(v)

                if key.startswith("Episode_Termination/"):
                    k = key.replace("Episode_Termination/", "")
                    v = val.item() if isinstance(val, torch.Tensor) else float(val)
                    num_done = done.reshape(-1).bool().sum().item()
                    count = max(0, round(v * num_done))
                    if "goal_reached" in k:
                        self._goal_reached += count
                    elif "obstacle_collision" in k:
                        self._obstacle_collision += count
                        self._collision += count
                    elif "wall_collision" in k:
                        self._wall_collision += count
                        self._collision += count
                    elif "collision" in k:  # generic collision
                        self._collision += count
                    elif "time_out" in k:
                        self._timeout += count
                    elif "tipped_over" in k or "physics_explosion" in k:
                        self._tipped_over += count

        # --- Obstacle policy metrics ---
        if obs_actions is not None:
            speed = obs_actions.reshape(-1, 2).norm(dim=-1).mean().item()
            self._obs_speeds.append(speed)
        if obs_obs is not None:
            robot_rel = obs_obs[:, :, 4:6]  # [E, N, 2]
            dist = robot_rel.norm(dim=-1).mean().item()
            self._obs_distances.append(dist)

        # --- Episode completion ---
        done_mask = done.reshape(-1).bool()
        if done_mask.any():
            ids = done_mask.nonzero(as_tuple=False).reshape(-1)
            for idx in ids:
                self._completed_rewards.append(self._ep_reward[idx].item())
                self._completed_lengths.append(self._ep_length[idx].item())
                self._completed_alive.append(self._ep_steps_alive[idx].item())
                # WD reward decomposition per completed episode
                self._completed_goal_reward.append(self._ep_goal_reward[idx].item())
                self._completed_wall_hit_reward.append(self._ep_wall_hit_reward[idx].item())
                self._completed_obs_hit_reward.append(self._ep_obs_hit_reward[idx].item())
                self._completed_floor_reward.append(self._ep_floor_reward[idx].item())
                self._completed_action_reward.append(self._ep_action_reward[idx].item())
                self._total_eps += 1
                # dies_at_birth: episode ended within 2 steps
                if self._ep_length[idx].item() <= 2:
                    self._first_step_deaths += 1
            self._ep_reward[ids] = 0.0
            self._ep_length[ids] = 0.0
            self._ep_steps_alive[ids] = 0.0
            self._ep_goal_reward[ids] = 0.0
            self._ep_wall_hit_reward[ids] = 0.0
            self._ep_obs_hit_reward[ids] = 0.0
            self._ep_floor_reward[ids] = 0.0
            self._ep_action_reward[ids] = 0.0

    def get_gamma(self):
        return self._curriculum_info.get("gamma", None)

    def collect(self, penalty_hit: float = -5.0, reward_get_goal: float = 40.0,
                cost_operate: float = 0.0, episode_length: float = 300.0,
                num_goals: int = 1) -> dict[str, float]:
        """Collect all metrics in Warp Drive naming convention.

        WD 論文指標完全對齊:
          - car goal reward: 到達目標的 reward 期望值
          - car static obstacle reward: 撞牆 penalty 期望值 (VR_Spot-Map)
          - car dynamic obstacle reward: 撞障礙物 penalty 期望值 (VR_Spot-Obs)
          - car floor reward: 地板 reward (flat terrain = 0)
          - car dynamic reward: action cost 期望值
          - car hit probability: 碰撞率 (p_Spot-Obs + p_Spot-Map)
          - car time survive expected value: 存活 reward 期望值 (WD 公式)
          - car time action expected value: action cost 期望值 (WD 公式)

        Args:
            penalty_hit: current phase collision penalty (for WD expected value formula)
            reward_get_goal: current phase goal reward
            cost_operate: current phase action cost
            episode_length: current phase episode length in steps
            num_goals: current phase number of goals
        """
        m: dict[str, float] = {}
        eps = 1e-8
        te = self._total_eps + eps

        # ==============================================================
        # B1: WD reward decomposition (from compute_wd_charge_reward breakdown)
        # ==============================================================

        # Per-episode mean reward by component (WD: divided by num_spot=1)
        if self._completed_goal_reward:
            m["charge goal reward"] = np.mean(self._completed_goal_reward)
        if self._completed_wall_hit_reward:
            m["charge static obstacle reward"] = np.mean(self._completed_wall_hit_reward)
        if self._completed_obs_hit_reward:
            m["charge dynamic obstacle reward"] = np.mean(self._completed_obs_hit_reward)
        if self._completed_floor_reward:
            m["charge floor reward"] = np.mean(self._completed_floor_reward)
        if self._completed_action_reward:
            m["charge dynamic reward"] = np.mean(self._completed_action_reward)

        # Also log Isaac Lab env reward terms (if env provides them)
        for wd_name, vals in self._wd_metrics.items():
            if vals:
                m.setdefault(wd_name, np.mean(vals))

        # ==============================================================
        # B1: Event stats (碰撞率分解)
        # ==============================================================

        # WD: car hit probability = total collision / total episodes
        m["charge hit probability"] = self._collision / te
        # WD: car dies at birth probability
        m["charge dies at birth probability"] = self._first_step_deaths / te

        # 碰撞分解: 靜態 (wall) vs 動態 (obstacle)
        # WD 論文: p_Spot-Map = wall collision rate, p_Spot-Obs = obstacle collision rate
        m["charge wall collision rate"] = self._wall_collision / te
        m["charge obstacle collision rate"] = self._obstacle_collision / te

        # VR (vulnerability ratio): 各碰撞類型佔總碰撞的比例
        total_col = self._collision + eps
        m["charge VR wall"] = self._wall_collision / total_col        # VR_Spot-Map
        m["charge VR obstacle"] = self._obstacle_collision / total_col  # VR_Spot-Obs

        # WD: goal remain probability (1 - goals_reached/total)
        if self._total_eps > 0:
            m["goal remain probability"] = 1.0 - self._goal_reached / te

        # ==============================================================
        # B1: WD time-based expected values (原始公式)
        #
        # WD spot time survive expected value =
        #   (num_goals * (1 - goal_remain) * reward_get_goal
        #    + (hit_prob - dies_at_birth + 0.0001) * penalty_hit) / num_spot
        #
        # WD spot time action expected value =
        #   (1 - dies_at_birth) * episode_length * cost_operate
        # ==============================================================

        hit_prob = self._collision / te
        dies_at_birth = self._first_step_deaths / te
        goal_remain = 1.0 - self._goal_reached / te if self._total_eps > 0 else 1.0

        # Survive expected value (WD 公式)
        m["charge time survive expected value"] = (
            num_goals * (1 - goal_remain) * reward_get_goal
            + (hit_prob - dies_at_birth + 0.0001) * penalty_hit
        )  # num_spot=1, so no division

        # Action expected value (WD 公式)
        m["charge time action expected value"] = (
            (1 - dies_at_birth) * episode_length * cost_operate
        )

        # Goal remain probability (time-based, same as non-time for single-policy)
        m["charge time goal remain probability"] = goal_remain

        # ==============================================================
        # Standard episode stats
        # ==============================================================

        if self._completed_rewards:
            m["charge/reward_mean"] = np.mean(self._completed_rewards)
            m["charge/reward_max"] = np.max(self._completed_rewards)
            m["charge/reward_min"] = np.min(self._completed_rewards)
        if self._completed_lengths:
            m["charge/episode_length_mean"] = np.mean(self._completed_lengths)
        m["charge/total_episodes"] = self._total_eps
        m["charge/goal_reach_rate"] = self._goal_reached / te
        m["charge/hit_probability"] = self._collision / te
        m["charge/timeout_rate"] = self._timeout / te
        m["charge/survival_probability"] = 1.0 - self._collision / te

        # --- Goal agent metrics ---
        m["goal/reached_count"] = self._goal_reached

        # --- Obstacle agent metrics ---
        m["obstacle/collision_count"] = self._collision
        m["obstacle/collision_rate"] = self._collision / te
        if self._obs_speeds:
            m["obstacle/mean_speed"] = np.mean(self._obs_speeds)
        if self._obs_distances:
            m["obstacle/mean_distance_to_robot"] = np.mean(self._obs_distances)

        # --- All Isaac Lab reward terms (raw) ---
        for k, vals in self._reward_terms.items():
            if vals:
                m[f"charge/reward_term/{k}"] = np.mean(vals)

        # --- Curriculum info ---
        for k, v in self._curriculum_info.items():
            m[f"curriculum/{k}"] = v

        return m

    def reset(self):
        self._completed_rewards.clear()
        self._completed_lengths.clear()
        self._completed_alive.clear()
        self._completed_goal_reward.clear()
        self._completed_wall_hit_reward.clear()
        self._completed_obs_hit_reward.clear()
        self._completed_floor_reward.clear()
        self._completed_action_reward.clear()
        self._goal_reached = 0
        self._collision = 0
        self._wall_collision = 0
        self._obstacle_collision = 0
        self._timeout = 0
        self._tipped_over = 0
        self._total_eps = 0
        self._first_step_deaths = 0
        self._reward_terms.clear()
        self._wd_metrics.clear()
        self._obs_speeds.clear()
        self._obs_distances.clear()


# ============================================================================
# Main Training
# ============================================================================

@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):

    # --- Seed ---
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)

    # --- Env config overrides ---
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    if args_cli.lidar_no_noise:
        try:
            lidar_cfg = env_cfg.observations.policy.lidar_ranges
            lidar_cfg.params["displacement_std"] = 0.0
            lidar_cfg.params["hole_rate"] = 0.0
            lidar_cfg.params["distractor_rate"] = 0.0
            lidar_cfg.params["use_Unoise"] = False
            print("[INFO] LiDAR noise disabled")
        except Exception as e:
            print(f"[WARN] Failed to disable LiDAR noise: {e}")

    # --- Curriculum version ---
    cv = args_cli.curriculum_version
    if cv is None:
        cv = "warp_drive_v1"  # default for multi-agent training
    cur = getattr(env_cfg, 'curriculum', None)
    if cur is not None:
        term = getattr(cur, 'goal_obstacle_curriculum', None)
        if term is not None:
            term.params["curriculum_version"] = cv
            term.params["initial_stage"] = args_cli.initial_stage
            print(f"[INFO] Curriculum version: {cv}, initial_stage: {args_cli.initial_stage}")

    # --- Create env ---
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    num_envs = env.num_envs
    device = env.device

    # Activate obstacle policy mode (disables scripted motion in obstacles.py)
    env.unwrapped._obstacle_policy_active = True
    print(f"[INFO] Obstacle policy ACTIVE — scripted motion DISABLED")

    obs_dim = env.observation_space.shape[-1]  # 139
    N_obs = args_cli.max_active_obstacles

    # --- Obstacle size + scene bound randomization (Warp Drive: obs_size_rand, floor_width_bias) ---
    # _obstacle_radii: [E, N] per-env per-obstacle collision radius
    # _scene_bounds: [E] per-env obstacle movement boundary
    # Updated from curriculum stage config or CLI override
    _obs_size_rand = args_cli.obs_size_rand  # 0 = auto from curriculum
    _scene_bound_rand = args_cli.scene_bound_rand  # 0 = auto from curriculum
    _obs_collision_base = args_cli.obs_collision_base
    _scene_bound_base = args_cli.scene_bound_base

    env.unwrapped._obstacle_radii = torch.full(
        (num_envs, N_obs), _obs_collision_base, device=device)
    env.unwrapped._scene_bounds = torch.full(
        (num_envs,), _scene_bound_base, device=device)

    def _randomize_obstacle_sizes(env_ids: torch.Tensor, rand_range: float):
        """Re-randomize obstacle collision radii for given env_ids."""
        if rand_range <= 0.0:
            return
        n = env_ids.shape[0]
        # radius = base ± rand/2 (matching WD: agent_size = rand * obs_size_rand + bias - obs_size_rand/2)
        noise = (torch.rand(n, N_obs, device=device) - 0.5) * rand_range
        env.unwrapped._obstacle_radii[env_ids] = (_obs_collision_base + noise).clamp(min=0.3)

    def _randomize_scene_bounds(env_ids: torch.Tensor, rand_range: float):
        """Re-randomize scene boundary for given env_ids."""
        if rand_range <= 0.0:
            return
        n = env_ids.shape[0]
        noise = (torch.rand(n, device=device) - 0.5) * rand_range
        env.unwrapped._scene_bounds[env_ids] = (_scene_bound_base + noise).clamp(min=4.0)

    # Initial randomization
    all_ids = torch.arange(num_envs, device=device)
    _randomize_obstacle_sizes(all_ids, _obs_size_rand)
    _randomize_scene_bounds(all_ids, _scene_bound_rand)

    print(f"[INFO] Randomization: obs_size_rand={_obs_size_rand} (base={_obs_collision_base}), "
          f"scene_bound_rand={_scene_bound_rand} (base={_scene_bound_base})")
    print(f"[INFO] Env: {args_cli.task}, {num_envs} envs, device={device}")
    print(f"[INFO] obs_dim={obs_dim}, max_active_obstacles={N_obs}")

    # --- Charge policy obs ---
    POLICY_OBS_INDICES = list(range(0, 78)) + [138]
    policy_obs_dim = len(POLICY_OBS_INDICES)  # 79

    # --- Build Charge models ---
    extractor = LidarStateExtractor().to(device)
    preprocess_rnn = PreprocessRNN(
        input_dim=extractor.output_dim, fc_dim=args_cli.fc_dim,
        hidden_dim=args_cli.hidden_dim, preprocess_dim=args_cli.preprocess_dim,
        rnn_type=args_cli.rnn_type,
    ).to(device)
    rl_input_dim = policy_obs_dim + args_cli.preprocess_dim  # 79+12=91 (WD-aligned)
    policy_head = PolicyHead(input_dim=rl_input_dim).to(device)
    value_head = ValueHead(input_dim=rl_input_dim).to(device)
    rnn_state = RNNStateManager(num_envs, args_cli.hidden_dim, device)
    obs_normalizer = RunningNormalizer(obs_dim, device)

    # WD optimizer structure:
    #   - RL head (policy+value): spot_lr=0.0002
    #   - RNN module: spot_rnn_model_lr=0.0005 (via RL gradient, since aux_lr=0)
    #   - Preprocess (extractor+rnn): spot_preprocess_model_lr=0 (disabled by default)
    charge_params_rl = list(policy_head.parameters()) + list(value_head.parameters())
    charge_opt_rl = torch.optim.Adam(charge_params_rl, lr=args_cli.lr, eps=1e-5)

    # RNN params get rnn_lr (WD: trained via RL gradient when aux_lr=0)
    charge_params_rnn = list(preprocess_rnn.parameters())
    charge_params_extractor = list(extractor.parameters())
    charge_params_aux = charge_params_extractor + charge_params_rnn

    if args_cli.aux_lr > 0:
        # Aux mode: extractor+rnn trained by prediction loss (original approach)
        charge_opt_aux = torch.optim.Adam(charge_params_aux, lr=args_cli.aux_lr, eps=1e-5)
    else:
        # WD mode: aux_lr=0 → RNN trained by RL gradient (rnn_lr), extractor by RL too
        charge_opt_aux = None  # No aux training

    # When aux_lr=0 (WD default), RNN is trained by RL gradient → add to RL optimizer
    if args_cli.aux_lr == 0:
        charge_opt_rl = torch.optim.Adam([
            {"params": charge_params_rl, "lr": args_cli.lr},
            {"params": charge_params_rnn, "lr": args_cli.rnn_lr},
            {"params": charge_params_extractor, "lr": args_cli.rnn_lr},
        ], eps=1e-5)

    # Store initial LR for decay
    for pg in charge_opt_rl.param_groups:
        pg["initial_lr"] = pg["lr"]

    total_charge_params = sum(p.numel() for p in charge_params_rl + charge_params_aux)
    print(f"[INFO] Charge: {policy_obs_dim}D + {args_cli.rnn_type} {args_cli.preprocess_dim}D = {rl_input_dim}D, {total_charge_params:,} params")
    print(f"[INFO] Charge LR: rl_head={args_cli.lr}, rnn={args_cli.rnn_lr}, aux={'disabled' if args_cli.aux_lr == 0 else args_cli.aux_lr}")

    # --- Build Obstacle models ---
    obs_policy = ObstaclePolicyFC().to(device)
    obs_value = ObstacleValueFC().to(device)
    obs_params = list(obs_policy.parameters()) + list(obs_value.parameters())
    obs_optimizer = torch.optim.Adam(obs_params, lr=args_cli.obs_lr, eps=1e-5)

    total_obs_params = sum(p.numel() for p in obs_params)
    print(f"[INFO] Obstacle: {OBS_POLICY_OBS_DIM}D obs, 2D act, {total_obs_params:,} params")
    print(f"[INFO] Alternating: train_goal_rate={args_cli.train_goal_rate} "
          f"(charge:{args_cli.train_goal_rate-1}, obstacle:1)")

    # --- Training config ---
    RL = args_cli.rollout_length
    batch_size = num_envs * RL
    mini_batch_size = batch_size // args_cli.mini_batches
    total_timesteps = args_cli.timesteps
    num_iterations = total_timesteps // RL

    # WD: γ = 1 - (1 - 0.92) / rl_fps = 0.984 (fps=5) — constant across all phases
    current_gamma = args_cli.gamma

    print(f"[INFO] {num_iterations} iterations, {RL} rollout, {batch_size:,} batch, "
          f"gamma={current_gamma} (WD constant)")
    print(f"[INFO] A2C mode: {'ON (no clipping)' if args_cli.use_a2c else 'OFF (PPO clip={})'.format(args_cli.clip_eps)}")

    # --- Buffers ---
    charge_buf = ChargeRolloutBuffer(RL, num_envs, rl_input_dim, obs_dim, args_cli.hidden_dim, device)
    obs_buf = ObstacleRolloutBuffer(RL, num_envs, N_obs, OBS_POLICY_OBS_DIM, 2, device)

    # --- Metrics ---
    metrics = MetricsCollector(num_envs, N_obs, device)

    # --- Log dir + WandB ---
    run_name = args_cli.run_name or f"marl_{datetime.now().strftime('%m%d_%H%M')}"
    log_dir = os.path.join("logs", "rnn_car", run_name)
    os.makedirs(log_dir, exist_ok=True)

    wandb_run = None
    if headless_mode:
        try:
            import wandb
            wandb.init(
                project="charge_skrl", name=run_name,
                config={
                    "agent": "MARL-ModularRNN-A2CK-v5" if args_cli.use_a2c else "MARL-ModularRNN-PPO-v5",
                    "num_envs": num_envs, "seed": args_cli.seed,
                    "charge_lr": args_cli.lr, "rnn_lr": args_cli.rnn_lr,
                    "aux_lr": args_cli.aux_lr, "vf_coeff": args_cli.vf_coeff,
                    "gamma": args_cli.gamma, "use_a2c": args_cli.use_a2c,
                    "rnn_type": args_cli.rnn_type,
                    "hidden_dim": args_cli.hidden_dim,
                    "preprocess_dim": args_cli.preprocess_dim,
                    "obs_lr": args_cli.obs_lr, "obs_ent_coeff": args_cli.obs_ent_coeff,
                    "obs_speed_limit": args_cli.obs_speed_limit,
                    "train_goal_rate": args_cli.train_goal_rate,
                    "obs_reward_mode": args_cli.obs_reward_mode,
                    "rollout_length": RL, "ppo_epochs": args_cli.ppo_epochs,
                    "tbptt_len": args_cli.tbptt_len,
                    "lr_decay": args_cli.lr_decay,
                    "total_charge_params": total_charge_params,
                    "total_obs_params": total_obs_params,
                    "obs_size_rand": args_cli.obs_size_rand,
                    "obs_collision_base": args_cli.obs_collision_base,
                    "scene_bound_rand": args_cli.scene_bound_rand,
                    "scene_bound_base": args_cli.scene_bound_base,
                },
                tags=["marl", "obstacle-policy", "v5", "wd-aligned"],
            )
            wandb_run = wandb.run
            print(f"[INFO] WandB: {wandb.run.name}")
        except Exception as e:
            print(f"[WARN] WandB init failed: {e}")

    # --- Load checkpoint ---
    if args_cli.checkpoint:
        ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
        extractor.load_state_dict(ckpt["extractor"])
        preprocess_rnn.load_state_dict(ckpt["preprocess_rnn"])
        policy_head.load_state_dict(ckpt["policy_head"])
        value_head.load_state_dict(ckpt["value_head"])
        if "obs_policy" in ckpt:
            obs_policy.load_state_dict(ckpt["obs_policy"])
            obs_value.load_state_dict(ckpt["obs_value"])
        if "obs_normalizer" in ckpt:
            obs_normalizer.mean = ckpt["obs_normalizer"]["mean"]
            obs_normalizer.var = ckpt["obs_normalizer"]["var"]
            obs_normalizer.count = ckpt["obs_normalizer"]["count"]
        print(f"[INFO] Loaded checkpoint: {args_cli.checkpoint}")

    # --- Initial reset ---
    obs, info = env.reset()
    start_time = time.time()

    # --- WD reward params (initial, Phase 1 defaults) ---
    _spot_penalty_hit = -5.0
    _spot_reward_get_goal = 40.0
    _spot_cost_operate = 0.03  # Phase 1 default (will update from curriculum)

    # --- WD entropy params (initial, Phase 1 defaults) ---
    # A2CK per-head: spot_entropy_coeff=0.04×2.5=0.10, action2=0.15×2.5=0.375
    _ent_coeff_linear = args_cli.ent_coeff_linear if args_cli.ent_coeff_linear > 0 else 0.10
    _ent_coeff_angular = args_cli.ent_coeff_angular if args_cli.ent_coeff_angular > 0 else 0.375
    if args_cli.ent_coeff > 0:  # Legacy single coeff override
        _ent_coeff_linear = args_cli.ent_coeff
        _ent_coeff_angular = args_cli.ent_coeff

    print(
        f"[INFO] Warp Drive reward: "
        f"penalty_hit={_spot_penalty_hit}, get_goal={_spot_reward_get_goal}, "
        f"cost_operate={_spot_cost_operate}"
    )
    print(
        f"[INFO] Warp Drive entropy (A2CK per-head): "
        f"linear={_ent_coeff_linear:.3f}, angular={_ent_coeff_angular:.3f} "
        f"(Phase 1 defaults, auto-sync from curriculum)"
    )

    # ========================================================================
    # Training Loop
    # ========================================================================

    for iteration in range(num_iterations):
        iter_start = time.time()
        charge_buf.reset()
        obs_buf.reset()
        metrics.reset()

        # === Determine who trains this iteration (Warp Drive alternation) ===
        train_charge = (iteration % args_cli.train_goal_rate != 1)
        train_obstacle = (iteration % args_cli.train_goal_rate == 1)

        # === Sync params from curriculum (gamma is constant per WD) ===

        # Update obs_size_rand / scene_bound_rand from curriculum (if not CLI-overridden)
        if args_cli.obs_size_rand == 0.0:
            _obs_size_rand = metrics._curriculum_info.get("obs_size_rand", 0.0)
        if args_cli.scene_bound_rand == 0.0:
            _scene_bound_rand = metrics._curriculum_info.get("scene_bound_rand", 0.0)

        # Sync WD reward params from curriculum (per-phase)
        _spot_penalty_hit = metrics._curriculum_info.get("spot_penalty_hit", -5.0)
        _spot_reward_get_goal = metrics._curriculum_info.get("spot_reward_get_goal", 40.0)
        _spot_cost_operate = metrics._curriculum_info.get("spot_cost_operate", 0.0)

        # Sync WD entropy params from curriculum (per-phase, A2CK per-head)
        if args_cli.ent_coeff_linear == 0.0:
            _ent_coeff_linear = metrics._curriculum_info.get("ent_coeff_linear", 0.30)
        if args_cli.ent_coeff_angular == 0.0:
            _ent_coeff_angular = metrics._curriculum_info.get("ent_coeff_angular", 0.375)
        # Legacy fallback: if --ent_coeff is set, use it for both heads
        if args_cli.ent_coeff > 0:
            _ent_coeff_linear = args_cli.ent_coeff
            _ent_coeff_angular = args_cli.ent_coeff

        # Sync obstacle speed rate from curriculum (per-phase)
        # WD: obs_speed_rate 0.8 → 0.85 → 1.15 across phases
        _obs_speed_limit = metrics._curriculum_info.get(
            "obstacle_speed_rate", args_cli.obs_speed_limit)

        # === LR decay (WD: ParamScheduler) ===
        if args_cli.lr_decay > 0 and iteration > 0:
            decay = max(0.01, 1.0 - args_cli.lr_decay * iteration)
            for pg in charge_opt_rl.param_groups:
                pg["lr"] = pg.get("initial_lr", pg["lr"]) * decay

        # === Rollout: BOTH policies act, alternating trains ===
        extractor.eval(); preprocess_rnn.eval(); policy_head.eval(); value_head.eval()
        obs_policy.eval(); obs_value.eval()

        for step in range(RL):
            # --- 1. Charge forward ---
            with torch.no_grad():
                obs_normalizer.update(obs)
                obs_normed = obs_normalizer.normalize(obs)
                features = extractor(obs_normed)
                hidden = rnn_state.get()
                rnn_feat, _, new_hidden = preprocess_rnn(features, hidden)
                p_obs = obs_normed[:, POLICY_OBS_INDICES]
                rl_in = torch.cat([p_obs, rnn_feat], dim=-1)
                logits = policy_head(rl_in)
                value = value_head(rl_in).squeeze(-1)
                actions, log_prob, _ = sample_action(logits)

            # --- 2. Env step ---
            next_obs, reward, terminated, truncated, info = env.step(actions.float())
            done = (terminated.squeeze(-1) | truncated.squeeze(-1)).float()
            env_reward_flat = reward.squeeze(-1)  # Isaac Lab env reward (for logging)

            # Warp Drive style sparse reward (for PPO training)
            reward_flat, reward_breakdown = compute_wd_charge_reward(
                env.unwrapped, actions, terminated, truncated,
                _spot_penalty_hit, _spot_reward_get_goal, _spot_cost_operate,
            )

            # --- 3. Obstacle forward + apply ---
            with torch.no_grad():
                obs_obs = build_obstacle_obs(env.unwrapped, N_obs, device)  # [E, N, 9]
                obs_flat = obs_obs.reshape(-1, OBS_POLICY_OBS_DIM)  # [E*N, 9]
                obs_act, obs_lp, obs_ent = obs_policy.sample(obs_flat)  # [E*N, 2]
                obs_val = obs_value(obs_flat).squeeze(-1)  # [E*N]

            apply_obstacle_actions(env.unwrapped, obs_act.reshape(num_envs, N_obs, 2),
                                   N_obs, dt=0.2, speed_limit=_obs_speed_limit)

            # Obstacle reward
            obs_rew = compute_obstacle_reward(env.unwrapped, obs_obs, N_obs, args_cli.obs_reward_mode)

            # Obstacle done = charge done (broadcast to all obstacles)
            obs_done = done.unsqueeze(-1).expand(-1, N_obs).reshape(-1)

            # --- 4. Store transitions ---
            charge_buf.add(rl_in, actions, log_prob, reward_flat, value, done, obs, hidden)
            obs_buf.add(obs_flat, obs_act, obs_lp, obs_rew, obs_val, obs_done)

            # --- 5. Metrics ---
            metrics.step(obs, reward, done, info,
                         obs_obs=obs_obs, obs_actions=obs_act.reshape(num_envs, N_obs, 2),
                         reward_breakdown=reward_breakdown)

            # --- 6. Update states ---
            rnn_state.update(new_hidden)
            done_mask = done.bool()
            if done_mask.any():
                done_ids = done_mask.nonzero(as_tuple=False).reshape(-1)
                rnn_state.reset(done_ids)
                # Reset obstacle velocity cache for done envs
                if hasattr(env.unwrapped, "_obstacle_velocities"):
                    env.unwrapped._obstacle_velocities[done_ids] = 0.0
                # Re-randomize obstacle sizes and scene bounds on episode reset
                _randomize_obstacle_sizes(done_ids, _obs_size_rand)
                _randomize_scene_bounds(done_ids, _scene_bound_rand)
            obs = next_obs

        # === Charge PPO Update (skip in play mode) ===
        charge_ppo_loss = 0.0
        charge_vf_loss = 0.0
        charge_entropy = 0.0
        aux_loss_val = 0.0
        if args_cli.play:
            train_charge = False
            train_obstacle = False
        aux_per_feature = {}

        if train_charge:
            # Bootstrap
            with torch.no_grad():
                obs_normed = obs_normalizer.normalize(obs)
                features = extractor(obs_normed)
                hidden = rnn_state.get()
                rnn_feat, _, _ = preprocess_rnn(features, hidden)
                p_obs = obs_normed[:, POLICY_OBS_INDICES]
                rl_in = torch.cat([p_obs, rnn_feat], dim=-1)
                last_value = value_head(rl_in).squeeze(-1)

            advantages, returns = compute_gae(
                charge_buf.rewards, charge_buf.values, charge_buf.dones,
                last_value, current_gamma, args_cli.gae_lambda)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            policy_head.train(); value_head.train()
            # When aux_lr=0 (WD default), extractor+rnn trained by RL gradient
            if args_cli.aux_lr == 0:
                extractor.train(); preprocess_rnn.train()

            flat_ri = charge_buf.rl_inputs.reshape(-1, rl_input_dim)
            flat_act = charge_buf.actions.reshape(-1, 2)
            flat_lp = charge_buf.log_probs.reshape(-1)
            flat_adv = advantages.reshape(-1)
            flat_ret = returns.reshape(-1)

            # A2C: single epoch, no mini-batching; PPO: multiple epochs + mini-batches
            n_epochs = 1 if args_cli.use_a2c else args_cli.ppo_epochs

            ppo_l, vf_l, ent_l = [], [], []
            for _ in range(n_epochs):
                if args_cli.use_a2c:
                    # A2C: full batch, single pass (WD: A2CK mode)
                    batches = [(torch.arange(batch_size, device=device),)]
                else:
                    idx = torch.randperm(batch_size, device=device)
                    batches = [(idx[s:min(s + mini_batch_size, batch_size)],)
                               for s in range(0, batch_size, mini_batch_size)]

                for (mb,) in batches:
                    nl = policy_head(flat_ri[mb])
                    nlp, ent_lin, ent_ang = evaluate_actions(nl, flat_act[mb])
                    nv = value_head(flat_ri[mb]).squeeze(-1)

                    if args_cli.use_a2c:
                        # A2CK: no clipping, pure policy gradient
                        pl = -(nlp * flat_adv[mb].detach()).mean()
                    else:
                        # PPO: clipped surrogate
                        ratio = (nlp - flat_lp[mb]).exp()
                        s1 = ratio * flat_adv[mb]
                        s2 = torch.clamp(ratio, 1 - args_cli.clip_eps, 1 + args_cli.clip_eps) * flat_adv[mb]
                        pl = -torch.min(s1, s2).mean()

                    vl = F.mse_loss(nv, flat_ret[mb])

                    # WD A2CK: per-head entropy with separate coefficients
                    entropy_loss = (_ent_coeff_linear * ent_lin.mean()
                                    + _ent_coeff_angular * ent_ang.mean())

                    # WD loss clamping: prevent catastrophic gradient spikes
                    pl_clamped = pl
                    if abs(pl.item()) > 20.0:
                        pl_clamped = pl * (20.0 / abs(pl.item()))
                    vf_term = args_cli.vf_coeff * vl
                    if abs(vf_term.item()) > 30.0:
                        max_30 = abs(max(min(30.0, abs(pl.item())), 10.0) / vl.item())
                        vl = vl * max_30

                    loss = pl_clamped + args_cli.vf_coeff * vl - entropy_loss
                    charge_opt_rl.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(charge_opt_rl.param_groups[0]["params"]) +
                        (list(charge_opt_rl.param_groups[1]["params"]) if len(charge_opt_rl.param_groups) > 1 else []) +
                        (list(charge_opt_rl.param_groups[2]["params"]) if len(charge_opt_rl.param_groups) > 2 else []),
                        args_cli.max_grad_norm)
                    charge_opt_rl.step()
                    ppo_l.append(pl.item()); vf_l.append(vl.item())
                    ent_l.append((ent_lin.mean() + ent_ang.mean()).item())

            charge_ppo_loss = np.mean(ppo_l)
            charge_vf_loss = np.mean(vf_l)
            charge_entropy = np.mean(ent_l)

            # Auxiliary loss (only when aux_lr > 0; WD default=0 → skip)
            aux_per_feature = {}
            if charge_opt_aux is not None:
                extractor.train(); preprocess_rnn.train()
                valid_t, valid_e = [], []
                for t in range(RL - 1):
                    nd = (charge_buf.dones[t] < 0.5).nonzero(as_tuple=False).reshape(-1)
                    if len(nd) > 0:
                        valid_t.append(torch.full_like(nd, t)); valid_e.append(nd)

                if valid_t:
                    at = torch.cat(valid_t); ae = torch.cat(valid_e)
                    a_obs = charge_buf.raw_obs[at, ae]
                    a_hid = charge_buf.hiddens[at, ae].unsqueeze(0)
                    a_tgt = charge_buf.raw_obs[at + 1, ae, LIDAR_START:LIDAR_END]
                    abs_ = min(4096, len(at))
                    alv = 0.0; nab = 0
                    perm = torch.randperm(len(at), device=device)
                    for ab_s in range(0, len(at), abs_):
                        ab_e = min(ab_s + abs_, len(at))
                        ab_i = perm[ab_s:ab_e]
                        mo = obs_normalizer.normalize(a_obs[ab_i])
                        ft = extractor(mo)
                        _, pred, _ = preprocess_rnn(ft, a_hid[:, ab_i, :], training=True)
                        tgt_n = (a_tgt[ab_i] - obs_normalizer.mean[LIDAR_START:LIDAR_END]) / (obs_normalizer.var[LIDAR_START:LIDAR_END].sqrt() + 1e-8)
                        l = F.smooth_l1_loss(pred, tgt_n)
                        charge_opt_aux.zero_grad(); l.backward()
                        nn.utils.clip_grad_norm_(charge_params_aux, args_cli.max_grad_norm)
                        charge_opt_aux.step()
                        alv += l.item(); nab += 1
                        # WD A2: per-feature loss (last batch only)
                        if ab_s + abs_ >= len(at):
                            with torch.no_grad():
                                per_feat = F.smooth_l1_loss(pred, tgt_n, reduction='none').mean(dim=0)
                                for fi in range(min(per_feat.shape[0], 72)):
                                    aux_per_feature[f"charge_module_feture_{fi}_loss"] = per_feat[fi].item()
                    aux_loss_val = alv / max(nab, 1)

        # === Obstacle PPO Update ===
        obs_loss = 0.0
        if train_obstacle:
            obs_policy.train(); obs_value.train()
            obs_loss = ppo_update_continuous(
                obs_policy, obs_value, obs_buf, obs_optimizer,
                epochs=args_cli.ppo_epochs, mini_batches=args_cli.mini_batches,
                clip_eps=args_cli.clip_eps, vf_coeff=args_cli.vf_coeff,
                ent_coeff=args_cli.obs_ent_coeff, max_grad_norm=args_cli.max_grad_norm,
                gamma=current_gamma, gae_lambda=args_cli.gae_lambda)

        # === Logging ===
        elapsed = time.time() - iter_start
        total_steps = (iteration + 1) * RL
        fps = num_envs * RL / elapsed
        # Episode length in steps (from curriculum episode_length_s / dt)
        _ep_len_s = metrics._curriculum_info.get("episode_length_s", 60.0)
        _ep_len_steps = _ep_len_s / 0.2  # dt=0.2s
        _num_goals = int(metrics._curriculum_info.get("num_goals", 1))
        wd = metrics.collect(
            penalty_hit=_spot_penalty_hit, reward_get_goal=_spot_reward_get_goal,
            cost_operate=_spot_cost_operate, episode_length=_ep_len_steps,
            num_goals=_num_goals,
        )

        if (iteration + 1) % args_cli.log_interval == 0 or iteration == 0:
            stage = wd.get("curriculum/stage", 0)
            sr = wd.get("charge/goal_reach_rate", 0)
            cr = wd.get("charge/hit_probability", 0)
            rwd = wd.get("charge/reward_mean", 0)
            who = "CHARGE" if train_charge else "OBS"
            print(
                f"[{iteration+1}/{num_iterations}] {who} "
                f"S{int(stage)} | g={current_gamma:.3f} | fps={fps:.0f} | "
                f"R={rwd:.1f} SR={sr:.1%} CR={cr:.1%} | "
                f"penalty={_spot_penalty_hit:.0f} | "
                f"ppo={charge_ppo_loss:.4f} vf={charge_vf_loss:.4f} "
                f"ent={charge_entropy:.3f} aux={aux_loss_val:.4f} "
                f"obs_loss={obs_loss:.4f}")

        if wandb_run is not None:
            log_data = {
                # --- A) Warp Drive: {policy} loss ---
                "charge loss": charge_ppo_loss,
                "charge loss coefficient entropy": charge_entropy,
                "charge vf_loss": charge_vf_loss,
                "obstacle loss": obs_loss,
                "obstacle loss coefficient entropy": args_cli.obs_ent_coeff,
                # --- A2) Warp Drive: module preprocess loss ---
                "charge_preprcess_loss": aux_loss_val,  # WD original typo preserved
            }
            # --- A2) Per-feature module loss (WD: charge_module_feture_{N}_loss) ---
            if train_charge and aux_per_feature:
                log_data.update(aux_per_feature)
            log_data.update({
                # --- Train info ---
                "train/fps": fps,
                "train/gamma": current_gamma,
                "train/active_agent": "charge" if train_charge else "obstacle",
                # --- WD reward params (per-phase) ---
                "wd_reward/spot_penalty_hit": _spot_penalty_hit,
                "wd_reward/spot_reward_get_goal": _spot_reward_get_goal,
                "wd_reward/spot_cost_operate": _spot_cost_operate,
                # --- WD entropy params (per-phase, A2CK per-head) ---
                "wd_entropy/ent_coeff_linear": _ent_coeff_linear,
                "wd_entropy/ent_coeff_angular": _ent_coeff_angular,
                # --- WD obstacle speed (per-phase) ---
                "wd_obs/speed_limit": _obs_speed_limit,
            })
            log_data.update(wd)
            wandb_run.log(log_data, step=total_steps)

        # === Save checkpoint ===
        if (iteration + 1) % args_cli.save_interval == 0 or iteration == num_iterations - 1:
            ckpt_path = os.path.join(log_dir, f"checkpoint_{total_steps}.pt")
            torch.save({
                "extractor": extractor.state_dict(),
                "preprocess_rnn": preprocess_rnn.state_dict(),
                "policy_head": policy_head.state_dict(),
                "value_head": value_head.state_dict(),
                "obs_policy": obs_policy.state_dict(),
                "obs_value": obs_value.state_dict(),
                "charge_opt_rl": charge_opt_rl.state_dict(),
                "charge_opt_aux": charge_opt_aux.state_dict() if charge_opt_aux else {},
                "obs_optimizer": obs_optimizer.state_dict(),
                "obs_normalizer": {
                    "mean": obs_normalizer.mean, "var": obs_normalizer.var,
                    "count": obs_normalizer.count},
                "iteration": iteration, "total_steps": total_steps,
                "args": vars(args_cli),
            }, ckpt_path)
            print(f"[SAVE] {ckpt_path}")

    # === Finish ===
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete: {total_timesteps:,} steps in {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"{'='*60}")
    if wandb_run is not None:
        import wandb
        wandb.finish()
    env.close()


if __name__ == "__main__":
    main()
