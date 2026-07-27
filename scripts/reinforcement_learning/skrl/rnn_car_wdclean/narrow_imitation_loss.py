"""Hard-label imitation loss against the scripted direct-crossing teacher.

07-27 verdict (N1). The SA5 narrow teacher was itself a detourer
(3-seed n=2094: crossing 95.9%, direct 0.000, pre-cross |y| p95 3.86 m), so
its KL term had been distilling the detour into every student. It is
removed and replaced by supervision from the scripted teacher, which
provably crosses straight (3-seed n=7104, direct 1.000):

    total = PPO loss + lambda * [CE(linear head, teacher linear bin)
                                 + CE(angular head, teacher angular bin)]

Two deliberate constraints from the verdict live here:

* the term applies **only** to narrow-replay frames — the caller passes the
  per-frame mask and every other frame contributes exactly zero;
* the teacher never drives. It only labels; PPO keeps collecting its own
  on-policy data, so rollouts stay clean.

`teacher_retention.masked_two_head_retention_loss` stays for soft
teacher-logit targets. This module is the hard-label path: the scripted
teacher emits one chosen bin per head, not a distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


NUM_ACTION_BINS = 19


@dataclass(frozen=True)
class NarrowImitationLoss:
    loss: torch.Tensor
    ce_linear: torch.Tensor
    ce_angular: torch.Tensor
    agreement_linear: float
    agreement_angular: float
    active_count: int


def scripted_action_ce_loss(
    student_logits: torch.Tensor,
    teacher_action_indices: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    num_bins: int = NUM_ACTION_BINS,
) -> NarrowImitationLoss:
    """Cross-entropy of both action heads against the scripted teacher.

    Args:
        student_logits: [B, 2*num_bins] — linear head then angular head.
        teacher_action_indices: [B, 2] long — chosen (linear, angular) bin.
        active_mask: [B] bool — True only on narrow-replay frames.

    Returns the summed CE over the two heads, averaged across active frames.
    An all-false mask returns a zero that still carries a gradient path, so
    a batch with no narrow env cannot break `backward()`.
    """

    expected_dim = 2 * num_bins
    if student_logits.ndim != 2 or student_logits.shape[-1] != expected_dim:
        raise ValueError(
            f"expected [B, {expected_dim}] student logits, got "
            f"{tuple(student_logits.shape)}"
        )
    batch = student_logits.shape[0]
    idx = teacher_action_indices
    if idx.ndim != 2 or idx.shape != (batch, 2):
        raise ValueError(
            f"expected [{batch}, 2] teacher action indices, got "
            f"{tuple(idx.shape)}"
        )
    mask = active_mask.reshape(-1).to(
        device=student_logits.device, dtype=torch.bool
    )
    if mask.numel() != batch:
        raise ValueError(
            f"mask has {mask.numel()} entries for batch {batch}"
        )

    zero = student_logits.sum() * 0.0
    if not bool(mask.any()):
        return NarrowImitationLoss(
            loss=zero,
            ce_linear=zero,
            ce_angular=zero,
            agreement_linear=float("nan"),
            agreement_angular=float("nan"),
            active_count=0,
        )

    idx = idx.to(device=student_logits.device, dtype=torch.long)
    selected_idx = idx[mask]
    if bool(
        ((selected_idx < 0) | (selected_idx >= num_bins)).any()
    ):
        raise ValueError(
            f"teacher action indices must lie in [0, {num_bins})"
        )

    selected = student_logits[mask]
    linear = selected[:, :num_bins]
    angular = selected[:, num_bins:]
    ce_linear = F.cross_entropy(linear, selected_idx[:, 0])
    ce_angular = F.cross_entropy(angular, selected_idx[:, 1])

    with torch.no_grad():
        agreement_linear = float(
            (linear.argmax(dim=-1) == selected_idx[:, 0]).float().mean()
        )
        agreement_angular = float(
            (angular.argmax(dim=-1) == selected_idx[:, 1]).float().mean()
        )

    return NarrowImitationLoss(
        loss=ce_linear + ce_angular,
        ce_linear=ce_linear,
        ce_angular=ce_angular,
        agreement_linear=agreement_linear,
        agreement_angular=agreement_angular,
        active_count=int(mask.sum().item()),
    )
