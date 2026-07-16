"""Pure, sim-free geometry helpers for the corridor-crossing injector.

Corridor is oriented along world-Y: robot travels +y, walls run along y at
local x = +/- half_width, pedestrian crosses along world-X (native
horizontal_crossing axis). No Isaac Sim import here so these stay unit-testable.
"""

from __future__ import annotations

import math

import torch

# Quaternion (qw, qx, qy, qz) for a +90 degree rotation about Z.
YAW90_QUAT: tuple[float, float, float, float] = (
    math.cos(math.pi / 4),
    0.0,
    0.0,
    math.sin(math.pi / 4),
)


def select_corridor_envs(env_ids: torch.Tensor, fraction: float, rand: torch.Tensor) -> torch.Tensor:
    """Return the subset of env_ids selected for corridor injection.

    rand is a [len(env_ids)] float tensor in [0, 1) supplied by the caller so
    that selection is deterministic under test.
    """
    if fraction <= 0.0:
        return env_ids[:0]
    return env_ids[rand < fraction]


def robot_corridor_pose(origins: torch.Tensor, spawn_y: float) -> torch.Tensor:
    """[K,7] world pose at origin + (0, spawn_y, 0), facing +y (yaw +90 deg)."""
    k = origins.shape[0]
    pose = torch.zeros(k, 7, device=origins.device, dtype=origins.dtype)
    pose[:, 0] = origins[:, 0]
    pose[:, 1] = origins[:, 1] + spawn_y
    pose[:, 2] = origins[:, 2]
    qw, qx, qy, qz = YAW90_QUAT
    pose[:, 3] = qw
    pose[:, 4] = qx
    pose[:, 5] = qy
    pose[:, 6] = qz
    return pose


def goal_corridor_pos(origins: torch.Tensor, goal_y: float) -> torch.Tensor:
    """[K,3] world goal at origin + (0, goal_y, 0)."""
    goal = origins.clone()
    goal[:, 1] = origins[:, 1] + goal_y
    goal[:, 2] = 0.0
    return goal


def wall_corridor_pose(origins: torch.Tensor, wall_x: float, wall_z: float) -> torch.Tensor:
    """[K,7] world pose at origin + (wall_x, 0, wall_z), rotated 90 deg about Z."""
    k = origins.shape[0]
    pose = torch.zeros(k, 7, device=origins.device, dtype=origins.dtype)
    pose[:, 0] = origins[:, 0] + wall_x
    pose[:, 1] = origins[:, 1]
    pose[:, 2] = wall_z
    qw, qx, qy, qz = YAW90_QUAT
    pose[:, 3] = qw
    pose[:, 4] = qx
    pose[:, 5] = qy
    pose[:, 6] = qz
    return pose


def wall_corridor_aabb(mesh_length: float) -> tuple[float, float]:
    """Local AABB (size_x, size_y) of a 90-deg-rotated wall.

    A native wall mesh is (length, 1.0) along (x, y). Rotating 90 deg about Z
    swaps them: the long axis now runs along y.
    """
    return (1.0, mesh_length)
