"""Policy-retention losses for narrow-passage replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch import nn
import torch.nn.functional as F


NUM_ACTION_BINS = 19
TOTAL_LOGITS = 2 * NUM_ACTION_BINS


def apply_masked_deterministic_teacher_actions(
    student_actions: torch.Tensor,
    teacher_logits: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    num_bins: int = NUM_ACTION_BINS,
) -> torch.Tensor:
    """Use the teacher's deterministic action only on selected environments."""
    if student_actions.ndim != 2 or student_actions.shape[-1] != 2:
        raise ValueError(
            "student_actions must have shape [B, 2], got "
            f"{tuple(student_actions.shape)}"
        )
    expected_logits = 2 * num_bins
    if (
        teacher_logits.ndim != 2
        or teacher_logits.shape[0] != student_actions.shape[0]
        or teacher_logits.shape[-1] != expected_logits
    ):
        raise ValueError(
            f"teacher_logits must have shape [B, {expected_logits}] for the "
            f"same batch, got {tuple(teacher_logits.shape)}"
        )
    mask = active_mask.reshape(-1).to(
        device=student_actions.device,
        dtype=torch.bool,
    )
    if mask.numel() != student_actions.shape[0]:
        raise ValueError(
            f"mask has {mask.numel()} entries for batch "
            f"{student_actions.shape[0]}"
        )

    teacher_logits = teacher_logits.to(device=student_actions.device)
    teacher_actions = torch.stack(
        (
            teacher_logits[:, :num_bins].argmax(dim=-1),
            teacher_logits[:, num_bins:].argmax(dim=-1),
        ),
        dim=-1,
    ).to(dtype=student_actions.dtype)
    return torch.where(mask.unsqueeze(-1), teacher_actions, student_actions)


@dataclass(frozen=True)
class RetentionLoss:
    loss: torch.Tensor
    margin_loss: torch.Tensor
    action_ce_loss: torch.Tensor
    kl_linear: torch.Tensor
    kl_angular: torch.Tensor
    margin_linear: torch.Tensor
    margin_angular: torch.Tensor
    action_ce_linear: torch.Tensor
    action_ce_angular: torch.Tensor
    agreement_linear: torch.Tensor
    agreement_angular: torch.Tensor
    active_count: int


@dataclass(frozen=True)
class PostUpdateProjectionStats:
    active_count: int
    optimizer_steps: int
    kl_before_linear: float
    kl_before_angular: float
    kl_after_linear: float
    kl_after_angular: float
    agreement_before_linear: float
    agreement_before_angular: float
    agreement_after_linear: float
    agreement_after_angular: float
    margin_before_linear: float
    margin_before_angular: float
    margin_after_linear: float
    margin_after_angular: float


@dataclass(frozen=True)
class DualPostUpdateProjectionStats:
    primary: PostUpdateProjectionStats
    anchor: PostUpdateProjectionStats
    optimizer_steps: int


def categorical_forward_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
) -> torch.Tensor:
    """Return KL(teacher || student) for every sample."""
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            "teacher/student logits must have identical shapes, got "
            f"{tuple(teacher_logits.shape)} and {tuple(student_logits.shape)}"
        )
    teacher_log_prob = F.log_softmax(teacher_logits, dim=-1)
    student_log_prob = F.log_softmax(student_logits, dim=-1)
    teacher_prob = teacher_log_prob.exp()
    return (teacher_prob * (teacher_log_prob - student_log_prob)).sum(dim=-1)


def teacher_argmax_margin_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    """Keep the teacher's deterministic action ahead by ``margin`` logits."""
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            "teacher/student logits must have identical shapes, got "
            f"{tuple(teacher_logits.shape)} and {tuple(student_logits.shape)}"
        )
    if margin < 0.0:
        raise ValueError(f"margin must be non-negative, got {margin}")
    teacher_action = teacher_logits.argmax(dim=-1, keepdim=True)
    student_teacher_logit = student_logits.gather(
        dim=-1, index=teacher_action
    ).squeeze(-1)
    student_other_max = student_logits.scatter(
        dim=-1,
        index=teacher_action,
        value=torch.finfo(student_logits.dtype).min,
    ).max(dim=-1).values
    student_margin = student_teacher_logit - student_other_max
    return F.relu(float(margin) - student_margin)


def teacher_argmax_cross_entropy(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
) -> torch.Tensor:
    """Distill the teacher's deterministic action with per-sample CE."""
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            "teacher/student logits must have identical shapes, got "
            f"{tuple(teacher_logits.shape)} and {tuple(student_logits.shape)}"
        )
    teacher_action = teacher_logits.argmax(dim=-1)
    return F.cross_entropy(
        student_logits,
        teacher_action,
        reduction="none",
    )


def masked_two_head_retention_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    num_bins: int = NUM_ACTION_BINS,
    argmax_margin: float = 0.0,
) -> RetentionLoss:
    """Compute forward KL for two categorical action heads on selected frames."""
    expected_dim = 2 * num_bins
    if teacher_logits.ndim != 2 or teacher_logits.shape[-1] != expected_dim:
        raise ValueError(
            f"expected [B, {expected_dim}] teacher logits, got "
            f"{tuple(teacher_logits.shape)}"
        )
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "teacher/student logits must have identical shapes, got "
            f"{tuple(teacher_logits.shape)} and {tuple(student_logits.shape)}"
        )
    mask = active_mask.reshape(-1).to(device=student_logits.device, dtype=torch.bool)
    if mask.numel() != student_logits.shape[0]:
        raise ValueError(
            f"mask has {mask.numel()} entries for batch {student_logits.shape[0]}"
        )

    teacher_linear = teacher_logits[:, :num_bins]
    teacher_angular = teacher_logits[:, num_bins:]
    student_linear = student_logits[:, :num_bins]
    student_angular = student_logits[:, num_bins:]

    if not bool(mask.any()):
        zero = student_logits.sum() * 0.0
        nan = torch.full((), float("nan"), device=student_logits.device)
        return RetentionLoss(
            loss=zero,
            margin_loss=zero,
            action_ce_loss=zero,
            kl_linear=nan,
            kl_angular=nan,
            margin_linear=nan,
            margin_angular=nan,
            action_ce_linear=nan,
            action_ce_angular=nan,
            agreement_linear=nan,
            agreement_angular=nan,
            active_count=0,
        )

    kl_linear = categorical_forward_kl(
        teacher_linear[mask], student_linear[mask]
    ).mean()
    kl_angular = categorical_forward_kl(
        teacher_angular[mask], student_angular[mask]
    ).mean()
    margin_linear = teacher_argmax_margin_loss(
        teacher_linear[mask],
        student_linear[mask],
        margin=argmax_margin,
    ).mean()
    margin_angular = teacher_argmax_margin_loss(
        teacher_angular[mask],
        student_angular[mask],
        margin=argmax_margin,
    ).mean()
    action_ce_linear = teacher_argmax_cross_entropy(
        teacher_linear[mask],
        student_linear[mask],
    ).mean()
    action_ce_angular = teacher_argmax_cross_entropy(
        teacher_angular[mask],
        student_angular[mask],
    ).mean()
    agreement_linear = (
        teacher_linear[mask].argmax(dim=-1)
        == student_linear[mask].argmax(dim=-1)
    ).float().mean()
    agreement_angular = (
        teacher_angular[mask].argmax(dim=-1)
        == student_angular[mask].argmax(dim=-1)
    ).float().mean()
    return RetentionLoss(
        loss=kl_linear + kl_angular,
        margin_loss=margin_linear + margin_angular,
        action_ce_loss=action_ce_linear + action_ce_angular,
        kl_linear=kl_linear,
        kl_angular=kl_angular,
        margin_linear=margin_linear,
        margin_angular=margin_angular,
        action_ce_linear=action_ce_linear,
        action_ce_angular=action_ce_angular,
        agreement_linear=agreement_linear,
        agreement_angular=agreement_angular,
        active_count=int(mask.sum().item()),
    )


def post_update_kl_projection(
    student_forward: Callable[[torch.Tensor], torch.Tensor],
    student_parameters: Iterable[nn.Parameter],
    inputs: torch.Tensor,
    teacher_logits: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    num_bins: int = NUM_ACTION_BINS,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    max_grad_norm: float,
    argmax_margin: float = 0.0,
    margin_weight: float = 0.0,
    action_ce_weight: float = 0.0,
) -> PostUpdateProjectionStats:
    """Project an already-updated student back toward the teacher on replay.

    Unlike adding KL to the joint PPO loss, this runs after the PPO optimizer
    step, when teacher/student KL is non-zero and therefore supplies a real
    corrective gradient. Optional argmax losses preserve deterministic actions,
    which a small distribution KL alone does not guarantee. A stateless SGD
    optimizer avoids contaminating the main PPO Adam moments.
    """
    if epochs < 0:
        raise ValueError(f"epochs must be non-negative, got {epochs}")
    if learning_rate <= 0.0:
        raise ValueError(
            f"learning_rate must be positive, got {learning_rate}"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if max_grad_norm <= 0.0:
        raise ValueError(
            f"max_grad_norm must be positive, got {max_grad_norm}"
        )
    if argmax_margin < 0.0:
        raise ValueError(
            f"argmax_margin must be non-negative, got {argmax_margin}"
        )
    if margin_weight < 0.0:
        raise ValueError(
            f"margin_weight must be non-negative, got {margin_weight}"
        )
    if action_ce_weight < 0.0:
        raise ValueError(
            f"action_ce_weight must be non-negative, got {action_ce_weight}"
        )
    if inputs.shape[0] != teacher_logits.shape[0]:
        raise ValueError(
            "inputs and teacher logits must share batch dimension, got "
            f"{inputs.shape[0]} and {teacher_logits.shape[0]}"
        )
    mask = active_mask.reshape(-1).to(
        device=teacher_logits.device, dtype=torch.bool
    )
    if mask.numel() != teacher_logits.shape[0]:
        raise ValueError(
            f"mask has {mask.numel()} entries for batch "
            f"{teacher_logits.shape[0]}"
        )

    parameters = [
        parameter for parameter in student_parameters
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("student_parameters has no trainable parameters")
    active_indices = mask.nonzero(as_tuple=False).flatten()

    def measure() -> tuple[float, ...]:
        if active_indices.numel() == 0:
            nan = float("nan")
            return (nan,) * 6
        totals = torch.zeros(6, dtype=torch.float64)
        count = 0
        with torch.no_grad():
            for start in range(0, active_indices.numel(), batch_size):
                indices = active_indices[start:start + batch_size]
                student_logits = student_forward(inputs[indices])
                result = masked_two_head_retention_loss(
                    teacher_logits[indices],
                    student_logits,
                    torch.ones(
                        indices.numel(),
                        dtype=torch.bool,
                        device=student_logits.device,
                    ),
                    num_bins=num_bins,
                    argmax_margin=argmax_margin,
                )
                weight = result.active_count
                totals += torch.tensor(
                    [
                        result.kl_linear.item(),
                        result.kl_angular.item(),
                        result.agreement_linear.item(),
                        result.agreement_angular.item(),
                        result.margin_linear.item(),
                        result.margin_angular.item(),
                    ],
                    dtype=torch.float64,
                ) * weight
                count += weight
        return tuple((totals / count).tolist())

    before = measure()
    optimizer_steps = 0
    if epochs > 0 and active_indices.numel() > 0:
        optimizer = torch.optim.SGD(parameters, lr=float(learning_rate))
        for _ in range(epochs):
            permutation = active_indices[
                torch.randperm(
                    active_indices.numel(), device=active_indices.device
                )
            ]
            for start in range(0, permutation.numel(), batch_size):
                indices = permutation[start:start + batch_size]
                student_logits = student_forward(inputs[indices])
                result = masked_two_head_retention_loss(
                    teacher_logits[indices],
                    student_logits,
                    torch.ones(
                        indices.numel(),
                        dtype=torch.bool,
                        device=student_logits.device,
                    ),
                    num_bins=num_bins,
                    argmax_margin=argmax_margin,
                )
                optimizer.zero_grad(set_to_none=True)
                projection_loss = (
                    result.loss
                    + float(margin_weight) * result.margin_loss
                    + float(action_ce_weight) * result.action_ce_loss
                )
                projection_loss.backward()
                nn.utils.clip_grad_norm_(parameters, float(max_grad_norm))
                optimizer.step()
                optimizer_steps += 1
        optimizer.zero_grad(set_to_none=True)
    after = measure()
    return PostUpdateProjectionStats(
        active_count=int(active_indices.numel()),
        optimizer_steps=optimizer_steps,
        kl_before_linear=before[0],
        kl_before_angular=before[1],
        kl_after_linear=after[0],
        kl_after_angular=after[1],
        agreement_before_linear=before[2],
        agreement_before_angular=before[3],
        agreement_after_linear=after[2],
        agreement_after_angular=after[3],
        margin_before_linear=before[4],
        margin_before_angular=before[5],
        margin_after_linear=after[4],
        margin_after_angular=after[5],
    )


def post_update_dual_teacher_projection(
    student_forward: Callable[[torch.Tensor], torch.Tensor],
    student_parameters: Iterable[nn.Parameter],
    inputs: torch.Tensor,
    primary_teacher_logits: torch.Tensor,
    primary_mask: torch.Tensor,
    anchor_teacher_logits: torch.Tensor,
    anchor_mask: torch.Tensor,
    *,
    anchor_weight: float,
    num_bins: int = NUM_ACTION_BINS,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    max_grad_norm: float,
    argmax_margin: float = 0.0,
    margin_weight: float = 0.0,
    action_ce_weight: float = 0.0,
) -> DualPostUpdateProjectionStats:
    """Project toward one teacher while functionally anchoring another domain.

    Each group contributes its own mean loss, so a large anchor group cannot
    dominate merely because it contains more frames. The primary group may use
    deterministic-action losses; the anchor is distribution KL only.
    """
    if anchor_weight <= 0.0:
        raise ValueError(f"anchor_weight must be positive, got {anchor_weight}")
    if epochs < 0:
        raise ValueError(f"epochs must be non-negative, got {epochs}")
    if learning_rate <= 0.0:
        raise ValueError(
            f"learning_rate must be positive, got {learning_rate}"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if max_grad_norm <= 0.0:
        raise ValueError(
            f"max_grad_norm must be positive, got {max_grad_norm}"
        )
    if argmax_margin < 0.0:
        raise ValueError(
            f"argmax_margin must be non-negative, got {argmax_margin}"
        )
    if margin_weight < 0.0:
        raise ValueError(
            f"margin_weight must be non-negative, got {margin_weight}"
        )
    if action_ce_weight < 0.0:
        raise ValueError(
            f"action_ce_weight must be non-negative, got {action_ce_weight}"
        )
    expected_shape = primary_teacher_logits.shape
    if anchor_teacher_logits.shape != expected_shape:
        raise ValueError(
            "primary and anchor teacher logits must have identical shapes, "
            f"got {expected_shape} and {anchor_teacher_logits.shape}"
        )
    if inputs.shape[0] != expected_shape[0]:
        raise ValueError(
            "inputs and teacher logits must share batch dimension, got "
            f"{inputs.shape[0]} and {expected_shape[0]}"
        )

    primary_mask = primary_mask.reshape(-1).to(
        device=primary_teacher_logits.device, dtype=torch.bool
    )
    anchor_mask = anchor_mask.reshape(-1).to(
        device=primary_teacher_logits.device, dtype=torch.bool
    )
    if primary_mask.numel() != expected_shape[0]:
        raise ValueError(
            f"primary mask has {primary_mask.numel()} entries for batch "
            f"{expected_shape[0]}"
        )
    if anchor_mask.numel() != expected_shape[0]:
        raise ValueError(
            f"anchor mask has {anchor_mask.numel()} entries for batch "
            f"{expected_shape[0]}"
        )
    if bool((primary_mask & anchor_mask).any()):
        raise ValueError("primary and anchor masks must be disjoint")

    primary_indices = primary_mask.nonzero(as_tuple=False).flatten()
    anchor_indices = anchor_mask.nonzero(as_tuple=False).flatten()
    if primary_indices.numel() == 0:
        raise ValueError("primary mask selects no frames")
    if anchor_indices.numel() == 0:
        raise ValueError("anchor mask selects no frames")

    parameters = [
        parameter for parameter in student_parameters
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("student_parameters has no trainable parameters")

    def measure(
        teacher_logits: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[float, ...]:
        totals = torch.zeros(6, dtype=torch.float64)
        count = 0
        with torch.no_grad():
            for start in range(0, indices.numel(), batch_size):
                batch_indices = indices[start:start + batch_size]
                student_logits = student_forward(inputs[batch_indices])
                result = masked_two_head_retention_loss(
                    teacher_logits[batch_indices],
                    student_logits,
                    torch.ones(
                        batch_indices.numel(),
                        dtype=torch.bool,
                        device=student_logits.device,
                    ),
                    num_bins=num_bins,
                    argmax_margin=argmax_margin,
                )
                weight = result.active_count
                totals += torch.tensor(
                    [
                        result.kl_linear.item(),
                        result.kl_angular.item(),
                        result.agreement_linear.item(),
                        result.agreement_angular.item(),
                        result.margin_linear.item(),
                        result.margin_angular.item(),
                    ],
                    dtype=torch.float64,
                ) * weight
                count += weight
        return tuple((totals / count).tolist())

    primary_before = measure(primary_teacher_logits, primary_indices)
    anchor_before = measure(anchor_teacher_logits, anchor_indices)
    optimizer_steps = 0
    if epochs > 0:
        optimizer = torch.optim.SGD(parameters, lr=float(learning_rate))
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            groups = (
                (
                    primary_teacher_logits,
                    primary_indices,
                    1.0,
                    True,
                ),
                (
                    anchor_teacher_logits,
                    anchor_indices,
                    float(anchor_weight),
                    False,
                ),
            )
            for teacher_logits, indices, group_weight, use_action_terms in groups:
                permutation = indices[
                    torch.randperm(indices.numel(), device=indices.device)
                ]
                for start in range(0, permutation.numel(), batch_size):
                    batch_indices = permutation[start:start + batch_size]
                    student_logits = student_forward(inputs[batch_indices])
                    result = masked_two_head_retention_loss(
                        teacher_logits[batch_indices],
                        student_logits,
                        torch.ones(
                            batch_indices.numel(),
                            dtype=torch.bool,
                            device=student_logits.device,
                        ),
                        num_bins=num_bins,
                        argmax_margin=argmax_margin,
                    )
                    loss = result.loss
                    if use_action_terms:
                        loss = (
                            loss
                            + float(margin_weight) * result.margin_loss
                            + float(action_ce_weight)
                            * result.action_ce_loss
                        )
                    normalized_weight = (
                        group_weight
                        * batch_indices.numel()
                        / indices.numel()
                    )
                    (normalized_weight * loss).backward()
            nn.utils.clip_grad_norm_(parameters, float(max_grad_norm))
            optimizer.step()
            optimizer_steps += 1
        optimizer.zero_grad(set_to_none=True)

    primary_after = measure(primary_teacher_logits, primary_indices)
    anchor_after = measure(anchor_teacher_logits, anchor_indices)

    def group_stats(
        indices: torch.Tensor,
        before: tuple[float, ...],
        after: tuple[float, ...],
    ) -> PostUpdateProjectionStats:
        return PostUpdateProjectionStats(
            active_count=int(indices.numel()),
            optimizer_steps=optimizer_steps,
            kl_before_linear=before[0],
            kl_before_angular=before[1],
            kl_after_linear=after[0],
            kl_after_angular=after[1],
            agreement_before_linear=before[2],
            agreement_before_angular=before[3],
            agreement_after_linear=after[2],
            agreement_after_angular=after[3],
            margin_before_linear=before[4],
            margin_before_angular=before[5],
            margin_after_linear=after[4],
            margin_after_angular=after[5],
        )

    return DualPostUpdateProjectionStats(
        primary=group_stats(
            primary_indices, primary_before, primary_after
        ),
        anchor=group_stats(anchor_indices, anchor_before, anchor_after),
        optimizer_steps=optimizer_steps,
    )
