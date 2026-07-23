from __future__ import annotations

import sys
from pathlib import Path

import torch


SKRL_ROOT = Path(__file__).resolve().parents[1]
if str(SKRL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKRL_ROOT))

from rnn_car_wdclean.teacher_retention import (  # noqa: E402
    masked_two_head_retention_loss,
)


def test_identical_policy_has_zero_loss_and_full_agreement() -> None:
    torch.manual_seed(7)
    logits = torch.randn(8, 38)
    result = masked_two_head_retention_loss(
        logits, logits.clone().requires_grad_(True), torch.ones(8, dtype=torch.bool)
    )

    assert torch.allclose(result.loss, torch.zeros_like(result.loss), atol=1e-6)
    assert result.agreement_linear.item() == 1.0
    assert result.agreement_angular.item() == 1.0
    assert result.active_count == 8


def test_only_masked_frames_receive_gradient() -> None:
    torch.manual_seed(11)
    teacher = torch.randn(6, 38)
    student = torch.randn(6, 38, requires_grad=True)
    mask = torch.tensor([True, False, True, False, False, True])

    result = masked_two_head_retention_loss(teacher, student, mask)
    result.loss.backward()

    assert student.grad is not None
    assert student.grad[mask].abs().sum().item() > 0.0
    assert student.grad[~mask].abs().sum().item() == 0.0
    assert result.active_count == 3


def test_empty_mask_is_differentiable_zero() -> None:
    student = torch.randn(4, 38, requires_grad=True)
    teacher = torch.randn(4, 38)

    result = masked_two_head_retention_loss(
        teacher, student, torch.zeros(4, dtype=torch.bool)
    )
    result.loss.backward()

    assert result.loss.item() == 0.0
    assert result.active_count == 0
    assert student.grad is not None
    assert student.grad.abs().sum().item() == 0.0
