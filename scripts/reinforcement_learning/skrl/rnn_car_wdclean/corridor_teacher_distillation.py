"""Post-PPO action projection for privileged corridor supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class CorridorProjectionStats:
    active_count: int
    optimizer_steps: int
    loss_before: float
    loss_after: float
    agreement_before_linear: float
    agreement_before_angular: float
    agreement_before_joint: float
    agreement_after_linear: float
    agreement_after_angular: float
    agreement_after_joint: float
    within_one_before_linear: float
    within_one_before_angular: float
    within_one_before_joint: float
    within_one_after_linear: float
    within_one_after_angular: float
    within_one_after_joint: float


def select_corridor_interventions(
    policy_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    any_feasible: torch.Tensor,
    obstacle_collision_grid: torch.Tensor,
    wall_collision_grid: torch.Tensor,
    min_obstacle_clearance_grid: torch.Tensor,
    *,
    clearance_threshold_m: float,
) -> dict[str, torch.Tensor]:
    """Select states where the deployed policy needs a safe intervention."""

    if policy_actions.ndim != 2 or policy_actions.shape[-1] != 2:
        raise ValueError("policy_actions must have shape [B, 2]")
    if teacher_actions.shape != policy_actions.shape:
        raise ValueError(
            "teacher_actions must match policy_actions, got "
            f"{tuple(teacher_actions.shape)} and {tuple(policy_actions.shape)}"
        )
    batch_size = policy_actions.shape[0]
    expected_grid_shape = (
        batch_size,
        obstacle_collision_grid.shape[-2],
        obstacle_collision_grid.shape[-1],
    )
    for name, grid in (
        ("obstacle_collision_grid", obstacle_collision_grid),
        ("wall_collision_grid", wall_collision_grid),
        ("min_obstacle_clearance_grid", min_obstacle_clearance_grid),
    ):
        if grid.ndim != 3 or grid.shape != expected_grid_shape:
            raise ValueError(
                f"{name} must have shape {expected_grid_shape}, got "
                f"{tuple(grid.shape)}"
            )
    if any_feasible.reshape(-1).numel() != batch_size:
        raise ValueError("any_feasible must have one value per sample")
    if clearance_threshold_m < 0.0:
        raise ValueError("clearance_threshold_m must be non-negative")

    policy_actions = policy_actions.long()
    num_linear = obstacle_collision_grid.shape[1]
    num_angular = obstacle_collision_grid.shape[2]
    if policy_actions.numel() > 0:
        if (
            policy_actions[:, 0].min().item() < 0
            or policy_actions[:, 0].max().item() >= num_linear
            or policy_actions[:, 1].min().item() < 0
            or policy_actions[:, 1].max().item() >= num_angular
        ):
            raise ValueError("policy action is outside the teacher grid")

    row = torch.arange(batch_size, device=policy_actions.device)
    linear = policy_actions[:, 0]
    angular = policy_actions[:, 1]
    obstacle_collision = obstacle_collision_grid[row, linear, angular].bool()
    wall_collision = wall_collision_grid[row, linear, angular].bool()
    clearance = min_obstacle_clearance_grid[row, linear, angular]
    low_clearance = clearance < float(clearance_threshold_m)
    teacher_differs = (teacher_actions.long() != policy_actions).any(dim=-1)
    feasible = any_feasible.reshape(-1).bool()
    intervention = (
        feasible
        & teacher_differs
        & (obstacle_collision | wall_collision | low_clearance)
    )
    return {
        "intervention": intervention,
        "teacher_differs": teacher_differs,
        "policy_obstacle_collision": obstacle_collision,
        "policy_wall_collision": wall_collision,
        "policy_low_clearance": low_clearance,
        "policy_min_obstacle_clearance_m": clearance,
    }


def _soft_neighbor_targets(
    action_indices: torch.Tensor,
    *,
    num_bins: int,
    neighbor_mass: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a center target with optional probability on adjacent bins."""

    if action_indices.ndim != 1:
        raise ValueError("action_indices must be one-dimensional")
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    if not 0.0 <= neighbor_mass < 1.0:
        raise ValueError("neighbor_mass must be in [0, 1)")
    if action_indices.numel() > 0:
        if action_indices.min().item() < 0 or action_indices.max().item() >= num_bins:
            raise ValueError("action index is outside the action grid")

    targets = torch.zeros(
        action_indices.shape[0],
        num_bins,
        dtype=dtype,
        device=action_indices.device,
    )
    if action_indices.numel() == 0:
        return targets

    rows = torch.arange(action_indices.shape[0], device=action_indices.device)
    targets[rows, action_indices] = 1.0 - float(neighbor_mass)
    left_valid = action_indices > 0
    right_valid = action_indices < num_bins - 1
    neighbor_count = left_valid.long() + right_valid.long()
    share = torch.where(
        neighbor_count > 0,
        torch.full_like(neighbor_count, float(neighbor_mass), dtype=dtype)
        / neighbor_count.clamp_min(1).to(dtype),
        torch.zeros_like(neighbor_count, dtype=dtype),
    )
    if left_valid.any():
        targets[
            rows[left_valid], action_indices[left_valid] - 1
        ] += share[left_valid]
    if right_valid.any():
        targets[
            rows[right_valid], action_indices[right_valid] + 1
        ] += share[right_valid]
    return targets


def _two_head_soft_action_loss(
    logits: torch.Tensor,
    teacher_actions: torch.Tensor,
    *,
    num_bins: int,
    neighbor_mass: float,
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != 2 * num_bins:
        raise ValueError(
            f"logits must have shape [B, {2 * num_bins}], got "
            f"{tuple(logits.shape)}"
        )
    if teacher_actions.shape != (logits.shape[0], 2):
        raise ValueError(
            "teacher_actions must have shape [B, 2], got "
            f"{tuple(teacher_actions.shape)}"
        )
    teacher_actions = teacher_actions.long()
    linear_target = _soft_neighbor_targets(
        teacher_actions[:, 0],
        num_bins=num_bins,
        neighbor_mass=neighbor_mass,
        dtype=logits.dtype,
    )
    angular_target = _soft_neighbor_targets(
        teacher_actions[:, 1],
        num_bins=num_bins,
        neighbor_mass=neighbor_mass,
        dtype=logits.dtype,
    )
    linear_log_prob = logits[:, :num_bins].log_softmax(dim=-1)
    angular_log_prob = logits[:, num_bins:].log_softmax(dim=-1)
    linear_loss = -(linear_target * linear_log_prob).sum(dim=-1).mean()
    angular_loss = -(angular_target * angular_log_prob).sum(dim=-1).mean()
    return linear_loss + angular_loss


def post_update_corridor_action_projection(
    student_forward: Callable[[torch.Tensor], torch.Tensor],
    student_parameters: Iterable[nn.Parameter],
    sample_indices: torch.Tensor,
    teacher_actions: torch.Tensor,
    *,
    num_bins: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    max_grad_norm: float,
    neighbor_mass: float,
) -> CorridorProjectionStats:
    """Fit current policy logits to privileged actions on corridor frames only."""

    if epochs < 0:
        raise ValueError("epochs must be non-negative")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive")
    if not 0.0 <= neighbor_mass < 1.0:
        raise ValueError("neighbor_mass must be in [0, 1)")
    sample_indices = sample_indices.reshape(-1).long()
    if teacher_actions.shape != (sample_indices.numel(), 2):
        raise ValueError(
            "teacher_actions must align with sample_indices, got "
            f"{tuple(teacher_actions.shape)} and {sample_indices.numel()}"
        )

    parameters = [
        parameter for parameter in student_parameters
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("student_parameters has no trainable parameters")

    def measure() -> tuple[float, ...]:
        if sample_indices.numel() == 0:
            nan = float("nan")
            return (nan,) * 7
        totals = torch.zeros(7, dtype=torch.float64)
        count = 0
        with torch.no_grad():
            for start in range(0, sample_indices.numel(), batch_size):
                stop = min(start + batch_size, sample_indices.numel())
                indices = sample_indices[start:stop]
                target = teacher_actions[start:stop].long()
                logits = student_forward(indices)
                loss = _two_head_soft_action_loss(
                    logits,
                    target,
                    num_bins=num_bins,
                    neighbor_mass=neighbor_mass,
                )
                prediction = torch.stack(
                    [
                        logits[:, :num_bins].argmax(dim=-1),
                        logits[:, num_bins:].argmax(dim=-1),
                    ],
                    dim=-1,
                )
                exact = prediction == target
                within_one = (prediction - target).abs() <= 1
                weight = stop - start
                totals += torch.tensor(
                    [
                        loss.item(),
                        exact[:, 0].float().mean().item(),
                        exact[:, 1].float().mean().item(),
                        exact.all(dim=-1).float().mean().item(),
                        within_one[:, 0].float().mean().item(),
                        within_one[:, 1].float().mean().item(),
                        within_one.all(dim=-1).float().mean().item(),
                    ],
                    dtype=torch.float64,
                ) * weight
                count += weight
        return tuple((totals / count).tolist())

    before = measure()
    optimizer_steps = 0
    if epochs > 0 and sample_indices.numel() > 0:
        optimizer = torch.optim.SGD(parameters, lr=float(learning_rate))
        for _ in range(epochs):
            order = torch.randperm(
                sample_indices.numel(), device=sample_indices.device
            )
            for start in range(0, order.numel(), batch_size):
                positions = order[start:start + batch_size]
                indices = sample_indices[positions]
                target = teacher_actions[positions].long()
                logits = student_forward(indices)
                loss = _two_head_soft_action_loss(
                    logits,
                    target,
                    num_bins=num_bins,
                    neighbor_mass=neighbor_mass,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, float(max_grad_norm))
                optimizer.step()
                optimizer_steps += 1
        optimizer.zero_grad(set_to_none=True)
    after = measure()
    return CorridorProjectionStats(
        active_count=int(sample_indices.numel()),
        optimizer_steps=optimizer_steps,
        loss_before=before[0],
        loss_after=after[0],
        agreement_before_linear=before[1],
        agreement_before_angular=before[2],
        agreement_before_joint=before[3],
        agreement_after_linear=after[1],
        agreement_after_angular=after[2],
        agreement_after_joint=after[3],
        within_one_before_linear=before[4],
        within_one_before_angular=before[5],
        within_one_before_joint=before[6],
        within_one_after_linear=after[4],
        within_one_after_angular=after[5],
        within_one_after_joint=after[6],
    )
