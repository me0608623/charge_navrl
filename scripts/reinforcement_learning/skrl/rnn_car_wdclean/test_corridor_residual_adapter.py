"""Unit tests for the deployable corridor residual adapter."""

import torch

from modular_rnn_models import (
    CorridorResidualAdapter,
    TOTAL_LOGITS,
    balanced_binary_gate_loss,
)


def test_zero_initialized_adapter_preserves_base_logits_exactly() -> None:
    torch.manual_seed(7)
    adapter = CorridorResidualAdapter(
        input_dim=83,
        hidden_dim=32,
        gate_init_probability=0.01,
        max_logit_delta=2.0,
    )
    obs = torch.randn(16, 83)
    base_logits = torch.randn(16, TOTAL_LOGITS)

    residual, gate_logits, gate_probability = adapter(obs)

    assert torch.count_nonzero(residual).item() == 0
    assert torch.equal(base_logits + residual, base_logits)
    assert gate_logits.shape == (16,)
    assert gate_probability.shape == (16,)


def test_gate_checkpoint_shape_matches_probe_mlp() -> None:
    adapter = CorridorResidualAdapter(input_dim=83, hidden_dim=64)
    probe_state = {
        "0.weight": torch.randn(64, 83),
        "0.bias": torch.randn(64),
        "2.weight": torch.randn(1, 64),
        "2.bias": torch.randn(1),
    }

    adapter.gate_net.load_state_dict(probe_state)

    for key, expected in probe_state.items():
        assert torch.equal(adapter.gate_net.state_dict()[key], expected)


def test_policy_feature_residual_preserves_initial_policy_exactly() -> None:
    adapter = CorridorResidualAdapter(
        input_dim=83,
        residual_input_dim=179,
        hidden_dim=32,
    )
    obs = torch.randn(8, 83)
    policy_features = torch.randn(8, 179)
    base_logits = torch.randn(8, TOTAL_LOGITS)

    residual, _, _ = adapter(obs, policy_features)

    assert torch.count_nonzero(residual).item() == 0
    assert torch.equal(base_logits + residual, base_logits)


def test_balanced_gate_loss_weights_both_classes_equally() -> None:
    logits = torch.tensor([-2.0, -2.0, -2.0, 2.0])
    targets = torch.tensor([False, False, False, True])
    loss = balanced_binary_gate_loss(logits, targets)

    positive_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[targets], torch.ones_like(logits[targets])
    )
    negative_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[~targets], torch.zeros_like(logits[~targets])
    )
    assert torch.allclose(loss, 0.5 * (positive_loss + negative_loss))


def test_adapter_rejects_wrong_observation_width() -> None:
    adapter = CorridorResidualAdapter(input_dim=83)

    try:
        adapter(torch.zeros(2, 82))
    except ValueError as error:
        assert "83D input" in str(error)
    else:
        raise AssertionError("wrong observation width must fail")


def test_adapter_rejects_wrong_residual_feature_width() -> None:
    adapter = CorridorResidualAdapter(
        input_dim=83,
        residual_input_dim=179,
    )

    try:
        adapter(torch.zeros(2, 83), torch.zeros(2, 178))
    except ValueError as error:
        assert "179D residual input" in str(error)
    else:
        raise AssertionError("wrong residual feature width must fail")
