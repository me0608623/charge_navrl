"""Policy-retention losses for narrow-passage replay."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


NUM_ACTION_BINS = 19
TOTAL_LOGITS = 2 * NUM_ACTION_BINS


@dataclass(frozen=True)
class RetentionLoss:
    loss: torch.Tensor
    kl_linear: torch.Tensor
    kl_angular: torch.Tensor
    agreement_linear: torch.Tensor
    agreement_angular: torch.Tensor
    active_count: int


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


def masked_two_head_retention_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    num_bins: int = NUM_ACTION_BINS,
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
            kl_linear=nan,
            kl_angular=nan,
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
        kl_linear=kl_linear,
        kl_angular=kl_angular,
        agreement_linear=agreement_linear,
        agreement_angular=agreement_angular,
        active_count=int(mask.sum().item()),
    )
