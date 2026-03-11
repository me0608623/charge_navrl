"""ORCA-Inspired Feature Encoder for RL Navigation

Computes 8-dimensional ORCA (Optimal Reciprocal Collision Avoidance) features
from obstacle observations. Two entry points:

1. compute_orca_features_actor(topk_obs):
   Uses topk_obstacles_goal_centric output (already in goal-centric frame).
   topk layout per obstacle (6 dims):
     [0] rel_x_goal      — relative position x (goal frame)
     [1] rel_y_goal      — relative position y (goal frame)
     [2] distance         — robot-obstacle distance
     [3] vel_x_goal       — relative velocity x (goal frame)
     [4] vel_y_goal       — relative velocity y (goal frame)
     [5] size             — obstacle radius

2. compute_orca_features_critic(obstacles_state):
   Uses dynamic_obstacles_state output (privileged, robot frame).
   Layout per obstacle (5 dims):
     [0] x     — relative position x (robot frame)
     [1] y     — relative position y (robot frame)
     [2] dir   — obstacle heading (relative to robot, radians)
     [3] v     — obstacle speed (m/s, scalar)
     [4] size  — obstacle radius

Output S_orca (8 dims):
  [0] min_TTC                     — minimum time-to-collision
  [1] mean_TTC                    — mean TTC (clamped)
  [2] avoidance_vector_x          — weighted repulsive vector x
  [3] avoidance_vector_y          — weighted repulsive vector y
  [4] distance_to_VO_boundary     — min VO margin, normalized [-1, 1]
  [5] collision_risk_score        — max sigmoid risk
  [6] nearest_dynamic_distance    — min safe distance, normalized
  [7] relative_velocity_alignment — max approach alignment [0, 1]

References:
  - ORCA: van den Berg et al., "Reciprocal n-body Collision Avoidance" (2011)
  - NavRL: Loquercio et al., IEEE RA-L 2025
  - CADRL: Chen et al., ICRA 2017
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import SceneEntityCfg


# ============================================================================
# Constants
# ============================================================================

TTC_MAX = 10.0          # Maximum TTC cap (seconds)
EPS = 1e-6              # Numerical stability
SIGMA_D = 2.0           # Distance decay for avoidance weighting
SIGMA_T = 3.0           # TTC decay for avoidance weighting
TAU = 3.0               # VO time horizon (seconds)
RISK_K = 1.0            # Sigmoid steepness for collision risk
NORM_DIST = 8.0         # Distance normalization factor


# ============================================================================
# Core ORCA computation (shared logic)
# ============================================================================

def _compute_orca_8d(
    p_rel: torch.Tensor,       # [num_envs, K, 2] relative position xy
    v_rel: torch.Tensor,       # [num_envs, K, 2] relative velocity xy
    distance: torch.Tensor,    # [num_envs, K] robot-obstacle distance
    radius: torch.Tensor,      # [num_envs, K] obstacle radius
    valid_mask: torch.Tensor,  # [num_envs, K] bool — True if obstacle exists
    robot_radius: float = 0.5,
    inflation_margin: float = 0.1,
) -> torch.Tensor:
    """Compute 8-dim ORCA features from parsed obstacle data.

    All inputs are already in a consistent coordinate frame (goal-centric or robot).
    The computation is purely vectorized torch — no Python loops over envs.

    Args:
        p_rel: Relative positions [num_envs, K, 2]
        v_rel: Relative velocities [num_envs, K, 2]
        distance: Scalar distances [num_envs, K]
        radius: Obstacle radii [num_envs, K]
        valid_mask: Boolean mask for real obstacles [num_envs, K]
        robot_radius: Robot body radius
        inflation_margin: Safety inflation

    Returns:
        [num_envs, 8] ORCA feature vector
    """
    device = p_rel.device
    num_envs, K = distance.shape

    # Combined radius: R = robot_radius + obstacle_radius + inflation
    R = robot_radius + radius + inflation_margin  # [num_envs, K]

    # Safe distance: d_safe = clamp(distance - R, min=eps)
    d_safe = (distance - R).clamp(min=EPS)  # [num_envs, K]

    # Unit direction: e = normalize(p_rel)
    p_norm = torch.norm(p_rel, dim=2, keepdim=True).clamp(min=EPS)  # [num_envs, K, 1]
    e = p_rel / p_norm  # [num_envs, K, 2]

    # Closing speed: closing = -dot(v_rel, e) — positive means approaching
    closing = -(v_rel * e).sum(dim=2)  # [num_envs, K]

    # Relative velocity magnitude
    v_rel_norm = torch.norm(v_rel, dim=2).clamp(min=EPS)  # [num_envs, K]

    # ========== [0] min_TTC, [1] mean_TTC ==========
    # TTC_i = d_safe / closing  (when closing > 0, else TTC_MAX)
    ttc_raw = d_safe / (closing.clamp(min=EPS))  # [num_envs, K]
    # Not approaching → TTC_MAX
    ttc = torch.where(closing > 0, ttc_raw, torch.full_like(ttc_raw, TTC_MAX))
    ttc = ttc.clamp(max=TTC_MAX)
    # Mask out invalid obstacles
    ttc_masked = torch.where(valid_mask, ttc, torch.full_like(ttc, TTC_MAX))

    min_ttc = ttc_masked.min(dim=1).values  # [num_envs]
    # Mean TTC over valid obstacles only
    valid_count = valid_mask.float().sum(dim=1).clamp(min=1.0)  # [num_envs]
    mean_ttc = (ttc_masked * valid_mask.float()).sum(dim=1) / valid_count  # [num_envs]
    # Normalize to [0, 1] range
    min_ttc_norm = (min_ttc / TTC_MAX).clamp(0, 1)  # [num_envs]
    mean_ttc_norm = (mean_ttc / TTC_MAX).clamp(0, 1)  # [num_envs]

    # ========== [2,3] avoidance_vector_x, avoidance_vector_y ==========
    # w_i = exp(-d_safe/sigma_d) * exp(-TTC/sigma_t)  (urgency weight)
    # a_i = -normalize(p_rel)  (repulsive direction)
    # u = sum(w_i * a_i), avoidance_vector = normalize(u)
    w = torch.exp(-d_safe / SIGMA_D) * torch.exp(-ttc_masked / SIGMA_T)  # [num_envs, K]
    w = w * valid_mask.float()  # zero out invalid
    a = -e  # [num_envs, K, 2] repulsive direction
    weighted_a = w.unsqueeze(2) * a  # [num_envs, K, 2]
    u = weighted_a.sum(dim=1)  # [num_envs, 2]
    u_norm = torch.norm(u, dim=1, keepdim=True).clamp(min=EPS)  # [num_envs, 1]
    avoidance_vec = u / u_norm  # [num_envs, 2]
    # If no obstacles, avoidance_vec = [0, 0]
    has_any_valid = valid_mask.any(dim=1, keepdim=True)  # [num_envs, 1]
    avoidance_vec = avoidance_vec * has_any_valid.float()

    # ========== [4] distance_to_VO_boundary ==========
    # vo_margin_i = d_safe_i - closing_i * tau
    # Positive = outside VO (safe), Negative = inside VO (unsafe)
    vo_margin = d_safe - closing * TAU  # [num_envs, K]
    vo_margin_masked = torch.where(valid_mask, vo_margin, torch.full_like(vo_margin, 1e6))
    min_vo_margin = vo_margin_masked.min(dim=1).values  # [num_envs]
    # Normalize to [-1, 1]: tanh is a natural bounded normalizer
    vo_boundary_norm = torch.tanh(min_vo_margin / NORM_DIST)  # [num_envs]

    # ========== [5] collision_risk_score ==========
    # risk_i = sigmoid((tau - TTC_i) / k)
    # High risk when TTC < tau, low risk when TTC > tau
    risk = torch.sigmoid((TAU - ttc_masked) / RISK_K)  # [num_envs, K]
    risk = risk * valid_mask.float()
    max_risk = risk.max(dim=1).values  # [num_envs]

    # ========== [6] nearest_dynamic_distance ==========
    # min(d_safe) normalized
    d_safe_masked = torch.where(valid_mask, d_safe, torch.full_like(d_safe, NORM_DIST))
    min_d_safe = d_safe_masked.min(dim=1).values  # [num_envs]
    nearest_dist_norm = (min_d_safe / NORM_DIST).clamp(0, 1)  # [num_envs]

    # ========== [7] relative_velocity_alignment ==========
    # approach_i = max(0, closing_i / (||v_rel_i|| + eps))
    # Measures how directly obstacle is approaching (1 = head-on, 0 = tangential)
    approach = (closing / v_rel_norm).clamp(0, 1)  # [num_envs, K]
    approach = approach * valid_mask.float()
    max_approach = approach.max(dim=1).values  # [num_envs]

    # ========== Assemble output ==========
    output = torch.stack([
        min_ttc_norm,               # [0] min_TTC (normalized)
        mean_ttc_norm,              # [1] mean_TTC (normalized)
        avoidance_vec[:, 0],        # [2] avoidance_vector_x
        avoidance_vec[:, 1],        # [3] avoidance_vector_y
        vo_boundary_norm,           # [4] distance_to_VO_boundary (normalized)
        max_risk,                   # [5] collision_risk_score
        nearest_dist_norm,          # [6] nearest_dynamic_distance (normalized)
        max_approach,               # [7] relative_velocity_alignment
    ], dim=1)  # [num_envs, 8]

    # Safety: replace any nan/inf
    output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)

    return output


# ============================================================================
# Actor entry point: uses topk_obstacles_goal_centric output
# ============================================================================

def compute_orca_features_actor(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    top_k: int = 5,
    robot_radius: float = 0.5,
    inflation_margin: float = 0.1,
    debug: bool = False,
) -> torch.Tensor:
    """Compute ORCA features from topk_obstacles observation (Actor).

    Parses the flat topk_obstacles tensor (from the preceding ObsTerm)
    and computes 8-dim ORCA features.

    topk layout per obstacle (6 dims):
      [0] rel_x_goal, [1] rel_y_goal, [2] distance,
      [3] vel_x_goal, [4] vel_y_goal, [5] size

    Args:
        env: Environment instance
        robot_cfg: Robot scene entity config
        top_k: Number of top-k obstacles (must match topk_obstacles config)
        robot_radius: Robot body radius (meters)
        inflation_margin: Safety inflation margin (meters)
        debug: If True, print S_orca values for env 0

    Returns:
        [num_envs, 8] ORCA feature vector
    """
    from .obstacle_observations import topk_obstacles_goal_centric

    # Get raw topk observation: [num_envs, top_k * 6]
    topk_flat = topk_obstacles_goal_centric(
        env, robot_cfg=robot_cfg, top_k=top_k,
    )
    num_envs = topk_flat.shape[0]
    device = topk_flat.device

    # Reshape to [num_envs, K, 6]
    topk = topk_flat.reshape(num_envs, top_k, 6)

    # Parse fields
    p_rel = topk[:, :, 0:2]      # [num_envs, K, 2] (rel_x_goal, rel_y_goal)
    distance = topk[:, :, 2]     # [num_envs, K]
    v_rel = topk[:, :, 3:5]      # [num_envs, K, 2] (vel_x_goal, vel_y_goal)
    radius = topk[:, :, 5]       # [num_envs, K]

    # Valid mask: obstacle exists if distance > 0 (zero-padded = no obstacle)
    valid_mask = distance > EPS   # [num_envs, K]

    output = _compute_orca_8d(
        p_rel, v_rel, distance, radius, valid_mask,
        robot_radius=robot_radius,
        inflation_margin=inflation_margin,
    )

    if debug:
        _debug_print("Actor", output)

    return output


# ============================================================================
# Critic entry point: uses dynamic_obstacles_state output (privileged)
# ============================================================================

def compute_orca_features_critic(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_obstacles: int = 10,
    robot_radius: float = 0.5,
    inflation_margin: float = 0.1,
    debug: bool = False,
) -> torch.Tensor:
    """Compute ORCA features from privileged obstacles_state (Critic).

    Uses true obstacle positions, velocities, and sizes from the scene
    (not filtered through topk). This gives the Critic a more accurate
    collision risk assessment.

    obstacles_state layout per obstacle (5 dims):
      [0] x, [1] y — relative position (robot frame)
      [2] dir      — heading (radians, relative to robot)
      [3] v        — speed scalar (m/s)
      [4] size     — obstacle radius

    Args:
        env: Environment instance
        asset_cfg: Robot scene entity config
        max_obstacles: Maximum obstacles in scene
        robot_radius: Robot body radius (meters)
        inflation_margin: Safety inflation margin (meters)
        debug: If True, print S_orca values for env 0

    Returns:
        [num_envs, 8] ORCA feature vector
    """
    from .functions import dynamic_obstacles_state

    # Get privileged observation: [num_envs, max_obstacles * 5]
    obs_flat = dynamic_obstacles_state(
        env, asset_cfg=asset_cfg,
        num_obstacles=0,  # Will auto-detect from env
        max_obstacles=max_obstacles,
        max_distance=15.0,
    )
    num_envs = obs_flat.shape[0]
    device = obs_flat.device

    # Reshape to [num_envs, max_obstacles, 5]
    obs = obs_flat.reshape(num_envs, max_obstacles, 5)

    # Parse fields
    pos_xy = obs[:, :, 0:2]      # [num_envs, K, 2] (x, y in robot frame)
    direction = obs[:, :, 2]     # [num_envs, K] heading (radians)
    speed = obs[:, :, 3]         # [num_envs, K] speed scalar
    size = obs[:, :, 4]          # [num_envs, K] radius

    # Compute distance from position
    distance = torch.norm(pos_xy, dim=2).clamp(min=EPS)  # [num_envs, K]

    # Reconstruct velocity vector from direction and speed
    # v_rel = speed * [cos(dir), sin(dir)]
    # Note: dir is already relative to robot, so this gives velocity in robot frame
    v_rel = torch.stack([
        speed * torch.cos(direction),
        speed * torch.sin(direction),
    ], dim=2)  # [num_envs, K, 2]

    # Valid mask: obstacle exists if any of (x, y, size) is nonzero
    # (hidden obstacles are zero-padded by dynamic_obstacles_state)
    valid_mask = (pos_xy.abs().sum(dim=2) > EPS) | (size > EPS)  # [num_envs, K]

    output = _compute_orca_8d(
        pos_xy, v_rel, distance, size, valid_mask,
        robot_radius=robot_radius,
        inflation_margin=inflation_margin,
    )

    if debug:
        _debug_print("Critic", output)

    return output


# ============================================================================
# Debug helper
# ============================================================================

def _debug_print(source: str, output: torch.Tensor) -> None:
    """Print ORCA features for env 0."""
    names = [
        "min_TTC", "mean_TTC",
        "avoid_x", "avoid_y",
        "VO_boundary", "risk_score",
        "nearest_dist", "vel_align",
    ]
    vals = output[0].detach().cpu().tolist()
    parts = [f"{n}={v:.3f}" for n, v in zip(names, vals)]
    print(f"[ORCA {source}] {' | '.join(parts)}")


# ============================================================================
# Observation functions for IsaacLab ObsTerm
# ============================================================================

def orca_features_actor(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    top_k: int = 5,
    robot_radius: float = 0.5,
    inflation_margin: float = 0.1,
    debug: bool = False,
) -> torch.Tensor:
    """ObsTerm-compatible wrapper for Actor ORCA features.

    Returns [num_envs, 8] appended to policy observation.
    """
    return compute_orca_features_actor(
        env, robot_cfg=robot_cfg, top_k=top_k,
        robot_radius=robot_radius, inflation_margin=inflation_margin,
        debug=debug,
    )


def orca_features_critic(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_obstacles: int = 10,
    robot_radius: float = 0.5,
    inflation_margin: float = 0.1,
    debug: bool = False,
) -> torch.Tensor:
    """ObsTerm-compatible wrapper for Critic ORCA features.

    Returns [num_envs, 8] appended to critic observation.
    """
    return compute_orca_features_critic(
        env, asset_cfg=asset_cfg, max_obstacles=max_obstacles,
        robot_radius=robot_radius, inflation_margin=inflation_margin,
        debug=debug,
    )


__all__ = [
    "compute_orca_features_actor",
    "compute_orca_features_critic",
    "orca_features_actor",
    "orca_features_critic",
]
