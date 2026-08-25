from types import SimpleNamespace

import pytest
import torch

from rnn_car_wdclean.vehicle_speed_rate import (
    ACTION_LIMIT_FIELDS,
    apply_vehicle_speed_rate_action_limits,
    apply_vehicle_speed_rate_observation,
    validate_vehicle_speed_rate,
)


def _action_cfg(**overrides):
    values = {
        "max_linear_velocity": 1.0,
        "max_linear_accel": 0.5,
        "max_angular_vel": 1.2,
        "max_angular_accel": 3.0,
        "deployment_speed_scale": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("rate", [0.0, -0.1, 1.01, float("nan"), float("inf")])
def test_invalid_speed_rate_fails_closed(rate):
    with pytest.raises(ValueError):
        validate_vehicle_speed_rate(rate, "ego")


def test_invalid_observation_mode_fails_closed():
    with pytest.raises(ValueError):
        validate_vehicle_speed_rate(0.7, "vehicle-ish")


def test_rate_one_is_bitwise_noop_for_observation():
    raw = torch.randn(3, 83)
    transformed = apply_vehicle_speed_rate_observation(raw, 1.0, "ego")
    assert transformed is raw
    assert torch.equal(transformed, raw)


def test_ego_mode_matches_deployed_vehicle_before_normalization():
    raw = torch.zeros(2, 83)
    raw[:, 0:3] = torch.tensor([[0.5, -0.9, 0.8], [0.1, 0.2, 0.3]])
    raw[:, 4:6] = torch.tensor([[3.0, 4.0], [18.0, 0.0]])
    raw[:, 6:78] = 0.25
    raw[:, 78] = 0.4
    raw[:, 79:83] = torch.tensor([1.0, -2.0, 3.0, -4.0])
    original = raw.clone()

    transformed = apply_vehicle_speed_rate_observation(raw, 0.7, "ego")

    assert torch.equal(raw[:, 6:79], transformed[:, 6:79])
    assert torch.allclose(
        transformed[:, 0:3],
        torch.clamp(raw[:, 0:3] / 0.7, -1.0, 1.0),
    )
    assert torch.allclose(transformed[0, 4:6], raw[0, 4:6] / 0.7)
    assert transformed[1, 4:6].norm().item() == pytest.approx(18.0)
    assert torch.allclose(transformed[:, 79:83], raw[:, 79:83] / 0.7)
    assert torch.equal(raw, original)


def test_none_mode_changes_actions_but_not_observations():
    raw = torch.randn(4, 83)
    assert apply_vehicle_speed_rate_observation(raw, 0.7, "none") is raw


def test_action_limits_all_scale_once_and_keep_downstream_scale_one():
    cfg = _action_cfg()
    contract = apply_vehicle_speed_rate_action_limits(cfg, 0.7)
    expected = {
        "max_linear_velocity": 0.7,
        "max_linear_accel": 0.35,
        "max_angular_vel": 0.84,
        "max_angular_accel": 2.1,
    }
    assert set(contract["after"]) == set(ACTION_LIMIT_FIELDS)
    for field, value in expected.items():
        assert getattr(cfg, field) == pytest.approx(value)
        assert contract["after"][field] == pytest.approx(value)
    assert contract["deployment_speed_scale"] == 1.0


def test_action_scaling_rejects_double_deployment_scaling():
    with pytest.raises(ValueError, match="double scaling"):
        apply_vehicle_speed_rate_action_limits(
            _action_cfg(deployment_speed_scale=0.7),
            0.7,
        )
