from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


SKRL_ROOT = Path(__file__).resolve().parents[1]
if str(SKRL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKRL_ROOT))

from rnn_car_wdclean.teacher_retention import (  # noqa: E402
    apply_masked_deterministic_teacher_actions,
    masked_two_head_retention_loss,
    post_update_dual_teacher_projection,
    post_update_kl_projection,
    teacher_argmax_cross_entropy,
    teacher_argmax_margin_loss,
)


def test_teacher_action_override_changes_only_masked_rows() -> None:
    student_actions = torch.tensor(
        [[1, 2], [3, 4], [5, 6], [7, 8]],
        dtype=torch.long,
    )
    teacher_logits = torch.zeros(4, 38)
    teacher_logits[0, 10] = 2.0
    teacher_logits[0, 19 + 11] = 3.0
    teacher_logits[1, 12] = 2.0
    teacher_logits[1, 19 + 13] = 3.0
    teacher_logits[2, 14] = 2.0
    teacher_logits[2, 19 + 15] = 3.0
    teacher_logits[3, 16] = 2.0
    teacher_logits[3, 19 + 17] = 3.0
    mask = torch.tensor([True, False, True, False])

    actions = apply_masked_deterministic_teacher_actions(
        student_actions,
        teacher_logits,
        mask,
    )

    assert torch.equal(actions[0], torch.tensor([10, 11]))
    assert torch.equal(actions[1], student_actions[1])
    assert torch.equal(actions[2], torch.tensor([14, 15]))
    assert torch.equal(actions[3], student_actions[3])
    assert actions.dtype == student_actions.dtype


def test_teacher_action_override_rejects_mismatched_batch() -> None:
    with pytest.raises(ValueError, match="same batch"):
        apply_masked_deterministic_teacher_actions(
            torch.zeros(4, 2, dtype=torch.long),
            torch.zeros(3, 38),
            torch.ones(4, dtype=torch.bool),
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


def test_argmax_margin_is_zero_when_teacher_action_has_required_lead() -> None:
    teacher = torch.tensor([[2.0, 0.0, -1.0]])
    student = torch.tensor([[1.0, 0.5, -1.0]], requires_grad=True)

    loss = teacher_argmax_margin_loss(teacher, student, margin=0.2)

    assert loss.item() == 0.0


def test_argmax_margin_directly_repairs_deterministic_action_order() -> None:
    teacher = torch.tensor([[2.0, 1.0, 0.0]])
    student = torch.tensor([[0.9, 1.0, 0.0]], requires_grad=True)

    loss = teacher_argmax_margin_loss(teacher, student, margin=0.2).mean()
    loss.backward()

    assert loss.item() == pytest.approx(0.3)
    assert student.grad is not None
    assert student.grad[0, 0].item() < 0.0
    assert student.grad[0, 1].item() > 0.0


def test_two_head_margin_uses_only_masked_frames() -> None:
    teacher = torch.zeros(2, 38)
    teacher[:, 0] = 1.0
    teacher[:, 19] = 1.0
    student = teacher.clone().requires_grad_(True)
    student.data[0, 0] = 0.0
    student.data[0, 1] = 0.1
    student.data[0, 19] = 0.0
    student.data[0, 20] = 0.1

    result = masked_two_head_retention_loss(
        teacher,
        student,
        torch.tensor([True, False]),
        argmax_margin=0.2,
    )
    result.margin_loss.backward()

    assert result.margin_loss.item() == pytest.approx(0.6)
    assert student.grad is not None
    assert student.grad[0].abs().sum().item() > 0.0
    assert student.grad[1].abs().sum().item() == 0.0


def test_teacher_action_ce_has_gradient_at_identical_policy() -> None:
    teacher = torch.tensor([[2.0, 1.0, 0.0]])
    student = teacher.clone().requires_grad_(True)

    loss = teacher_argmax_cross_entropy(teacher, student).mean()
    loss.backward()

    assert loss.item() > 0.0
    assert student.grad is not None
    assert student.grad[0, 0].item() < 0.0
    assert student.grad[0, 1:].sum().item() > 0.0


def test_post_update_projection_reduces_forward_kl() -> None:
    torch.manual_seed(19)
    inputs = torch.randn(96, 5)
    teacher = torch.nn.Linear(5, 38)
    student = torch.nn.Linear(5, 38)
    student.load_state_dict(teacher.state_dict())
    with torch.no_grad():
        student.weight.add_(0.25 * torch.randn_like(student.weight))
        student.bias.add_(0.15 * torch.randn_like(student.bias))
        teacher_logits = teacher(inputs)

    stats = post_update_kl_projection(
        student,
        student.parameters(),
        inputs,
        teacher_logits,
        torch.ones(96, dtype=torch.bool),
        epochs=20,
        learning_rate=0.2,
        batch_size=24,
        max_grad_norm=10.0,
    )

    assert stats.active_count == 96
    assert stats.optimizer_steps == 80
    assert stats.kl_after_linear < stats.kl_before_linear
    assert stats.kl_after_angular < stats.kl_before_angular


def test_post_update_projection_is_noop_for_identical_policy() -> None:
    torch.manual_seed(23)
    inputs = torch.randn(32, 3)
    student = torch.nn.Linear(3, 38)
    with torch.no_grad():
        teacher_logits = student(inputs).clone()
    before = {
        key: value.detach().clone()
        for key, value in student.state_dict().items()
    }

    stats = post_update_kl_projection(
        student,
        student.parameters(),
        inputs,
        teacher_logits,
        torch.ones(32, dtype=torch.bool),
        epochs=4,
        learning_rate=0.1,
        batch_size=16,
        max_grad_norm=1.0,
    )

    for key, value in student.state_dict().items():
        assert torch.allclose(value, before[key], atol=1e-7)
    assert stats.kl_before_linear == pytest.approx(0.0, abs=1e-7)
    assert stats.kl_after_linear == pytest.approx(0.0, abs=1e-7)
    assert stats.agreement_after_linear == 1.0
    assert stats.agreement_after_angular == 1.0


def test_post_update_margin_repairs_deterministic_action_order() -> None:
    inputs = torch.zeros(32, 1)
    student = torch.nn.Linear(1, 38)
    with torch.no_grad():
        student.weight.zero_()
        student.bias.zero_()
        student.bias[1] = 0.3
        student.bias[20] = 0.3
        teacher_logits = torch.zeros(32, 38)
        teacher_logits[:, 0] = 1.0
        teacher_logits[:, 19] = 1.0

    stats = post_update_kl_projection(
        student,
        student.parameters(),
        inputs,
        teacher_logits,
        torch.ones(32, dtype=torch.bool),
        epochs=20,
        learning_rate=0.2,
        batch_size=32,
        max_grad_norm=10.0,
        argmax_margin=0.2,
        margin_weight=1.0,
    )

    assert stats.agreement_before_linear == 0.0
    assert stats.agreement_before_angular == 0.0
    assert stats.agreement_after_linear == 1.0
    assert stats.agreement_after_angular == 1.0
    assert stats.margin_after_linear < stats.margin_before_linear
    assert stats.margin_after_angular < stats.margin_before_angular


def test_dual_projection_reduces_primary_and_anchor_kl() -> None:
    torch.manual_seed(31)
    inputs = torch.randn(128, 6)
    primary_teacher = torch.nn.Linear(6, 38)
    anchor_teacher = torch.nn.Linear(6, 38)
    student = torch.nn.Linear(6, 38)
    with torch.no_grad():
        primary_logits = primary_teacher(inputs)
        anchor_logits = anchor_teacher(inputs)
    primary_mask = torch.zeros(128, dtype=torch.bool)
    primary_mask[:48] = True
    anchor_mask = ~primary_mask

    stats = post_update_dual_teacher_projection(
        student,
        student.parameters(),
        inputs,
        primary_logits,
        primary_mask,
        anchor_logits,
        anchor_mask,
        anchor_weight=0.5,
        epochs=40,
        learning_rate=0.1,
        batch_size=16,
        max_grad_norm=10.0,
        argmax_margin=0.2,
        margin_weight=0.25,
    )

    assert stats.optimizer_steps == 40
    assert stats.primary.active_count == 48
    assert stats.anchor.active_count == 80
    assert (
        stats.primary.kl_after_linear
        < stats.primary.kl_before_linear
    )
    assert (
        stats.primary.kl_after_angular
        < stats.primary.kl_before_angular
    )
    assert stats.anchor.kl_after_linear < stats.anchor.kl_before_linear
    assert stats.anchor.kl_after_angular < stats.anchor.kl_before_angular


def test_dual_projection_rejects_overlapping_masks() -> None:
    inputs = torch.randn(8, 2)
    student = torch.nn.Linear(2, 38)
    teacher_logits = torch.randn(8, 38)
    primary_mask = torch.tensor(
        [True, True, False, False, False, False, False, False]
    )
    anchor_mask = torch.tensor(
        [False, True, True, False, False, False, False, False]
    )

    with pytest.raises(ValueError, match="disjoint"):
        post_update_dual_teacher_projection(
            student,
            student.parameters(),
            inputs,
            teacher_logits,
            primary_mask,
            teacher_logits,
            anchor_mask,
            anchor_weight=0.5,
            epochs=1,
            learning_rate=0.1,
            batch_size=4,
            max_grad_norm=1.0,
        )
