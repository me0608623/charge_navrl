"""Contract tests for the rebuilt SA3-SA4 speed-density lineage."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys

import pytest
import torch


_REPO = pathlib.Path(__file__).resolve().parents[4]
_SKRL = _REPO / "scripts" / "reinforcement_learning" / "skrl"
if str(_SKRL) not in sys.path:
    sys.path.insert(0, str(_SKRL))

from rnn_car_modular.configs.e2e_sa3_k8_obb_speed_density_v3_from_sa2_r1_c100 import (  # noqa: E402
    CONFIG as SA3_CONFIG,
    PARENT_CHECKPOINT_SHA256,
)
from rnn_car_modular.configs.e2e_sa4_k8_obb_speed_density_v3_from_sa3 import (  # noqa: E402
    CONFIG as SA4_CONFIG,
    PARENT_CHECKPOINT as SA4_PARENT_CHECKPOINT,
    PARENT_CHECKPOINT_SHA256 as SA4_PARENT_CHECKPOINT_SHA256,
)
from rnn_car_modular.configs.sim2real_speed_density_curriculum_v3 import (  # noqa: E402
    P035,
    P060,
    P080,
    P100,
    SA3_SPEED_DENSITY_MIX,
    SA4_SPEED_DENSITY_MIX,
)
from rnn_car_modular.configs.sim2real_stage_curriculum_v2 import (  # noqa: E402
    make_sim2real_curriculum_config,
)


def _load_corridor_density_module():
    path = (
        _REPO
        / "source"
        / "isaaclab_tasks"
        / "isaaclab_tasks"
        / "manager_based"
        / "locomotion"
        / "velocity"
        / "config"
        / "charge_skrl"
        / "mdp"
        / "events"
        / "corridor_density.py"
    )
    spec = importlib.util.spec_from_file_location("_corridor_density_v3_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_profiles(mix):
    return [
        [list(counts), list(speed_range), weight]
        for counts, speed_range, weight in mix
    ]


def test_frozen_json_and_python_profiles_are_identical():
    frozen = json.loads(
        (_REPO / "docs/freeze/sim2real_speed_density_curriculum_v3_20260821.json")
        .read_text(encoding="utf-8")
    )
    assert frozen["stages"]["SA3"]["profiles"] == _json_profiles(
        SA3_SPEED_DENSITY_MIX
    )
    assert frozen["stages"]["SA4"]["profiles"] == _json_profiles(
        SA4_SPEED_DENSITY_MIX
    )
    assert frozen["invariants"]["vehicle_speed_rate_from_sa3"] == 0.7
    assert frozen["invariants"]["p100_allowed_in_sa3_or_sa4"] is False


def test_sa3_contract_changes_at_the_stage_boundary():
    assert SA3_CONFIG.initial_stage == 3
    assert SA3_CONFIG.speed_rate == 0.7
    assert SA3_CONFIG.speed_rate_obs == "ego"
    assert SA3_CONFIG.no_resume_optimizer is True
    assert SA3_CONFIG.timesteps == 38_400
    assert SA3_CONFIG.save_interval == 50
    assert SA3_CONFIG.long_corridor_speed_density_mix == SA3_SPEED_DENSITY_MIX
    assert SA3_CONFIG.long_corridor_obstacle_count_mix is None
    assert SA3_CONFIG.lidar_distractor_eligibility == "valid_return_only"
    assert SA3_CONFIG.actuator_delay_range == (1, 2)
    assert SA3_CONFIG.actuator_velocity_scale == (1.0, 1.0)
    assert SA3_CONFIG.actuator_motor_lag == 1.0
    assert len(PARENT_CHECKPOINT_SHA256) == 64


def test_sa4_uses_the_gate_accepted_sha_locked_c300_parent():
    assert SA4_CONFIG.initial_stage == 4
    assert SA4_CONFIG.checkpoint == SA4_PARENT_CHECKPOINT
    assert SA4_PARENT_CHECKPOINT.endswith("checkpoint_38400.pt")
    assert len(SA4_PARENT_CHECKPOINT_SHA256) == 64
    assert SA4_CONFIG.long_corridor_speed_density_mix == SA4_SPEED_DENSITY_MIX
    assert "from_formally_accepted_sa3_v3_c300" in SA4_CONFIG.tags
    assert "parent_sha256_locked" in SA4_CONFIG.tags
    assert "not_ready_to_run" not in SA4_CONFIG.tags


def test_fast_profiles_are_low_density_and_p100_is_excluded():
    for mix in (SA3_SPEED_DENSITY_MIX, SA4_SPEED_DENSITY_MIX):
        assert abs(sum(weight for _, _, weight in mix) - 1.0) < 1e-12
        for (static, dynamic), speed_range, _ in mix:
            assert dynamic == 1
            assert speed_range != P100
            if speed_range in (P080, P100):
                assert static <= 1
    assert P080 not in {speed_range for _, speed_range, _ in SA3_SPEED_DENSITY_MIX}
    assert {P035, P060} == {
        speed_range for _, speed_range, _ in SA3_SPEED_DENSITY_MIX
    }


def test_frozen_v2_configs_keep_joint_mix_disabled():
    for stage in range(1, 9):
        checkpoint = None if stage == 1 else f"pending-sa{stage}.pt"
        config = make_sim2real_curriculum_config(stage, checkpoint=checkpoint)
        assert config.long_corridor_speed_density_mix is None


def test_joint_sampler_preserves_profile_pairing_and_carry_quota():
    module = _load_corridor_density_module()
    carry = {}
    all_counts = []
    all_ranges = []
    all_ids = []
    for _ in range(20):
        counts, ranges, profile_ids = (
            module.sample_speed_density_profiles_systematic(
                17,
                mix=SA3_SPEED_DENSITY_MIX,
                device="cpu",
                carry=carry,
            )
        )
        all_counts.append(counts)
        all_ranges.append(ranges)
        all_ids.append(profile_ids)
        for row, speed_range, profile_id in zip(
            counts.tolist(), ranges.tolist(), profile_ids.tolist()
        ):
            expected_counts, expected_speed, _ = SA3_SPEED_DENSITY_MIX[profile_id]
            assert tuple(row) == expected_counts
            assert tuple(speed_range) == pytest.approx(expected_speed)

    ids = torch.cat(all_ids)
    realized = torch.bincount(ids, minlength=len(SA3_SPEED_DENSITY_MIX)).double()
    realized /= realized.sum()
    expected = torch.tensor(
        [weight for _, _, weight in SA3_SPEED_DENSITY_MIX], dtype=torch.double
    )
    assert torch.max(torch.abs(realized - expected)).item() <= 1.0 / ids.numel()
    assert torch.cat(all_counts).shape == (340, 2)
    assert torch.cat(all_ranges).shape == (340, 2)


def test_joint_curriculum_motion_weights_exclude_random_2d():
    module = _load_corridor_density_module()
    counts = torch.ones(600, dtype=torch.long)
    interactions = torch.full(
        (600,), module.INTERACTION_INDEPENDENT, dtype=torch.long
    )
    families, pairs = module.assign_families_and_pairs(
        counts,
        interactions,
        device="cpu",
        family_debt={},
        family_weights=(0.5, 0.5, 0.0),
    )
    active = families[:, 0]
    assert not bool((active == module.MOTION_RANDOM_2D).any())
    assert abs(float((active == module.MOTION_LATERAL).float().mean()) - 0.5) < 0.01
    assert abs(float((active == module.MOTION_LONGITUDINAL).float().mean()) - 0.5) < 0.01
    assert bool((pairs == module.NO_PAIR).all())


@pytest.mark.parametrize(
    "bad_mix",
    [
        (((0, 1), (0.5, 0.7), 0.9),),
        (((0, 0), (0.5, 0.7), 1.0),),
        (((0, 1), (0.7, 0.5), 1.0),),
        (
            ((0, 1), (0.5, 0.7), 0.5),
            ((0, 1), (0.5, 0.7), 0.5),
        ),
    ],
)
def test_joint_sampler_fails_closed_on_invalid_profiles(bad_mix):
    module = _load_corridor_density_module()
    with pytest.raises(ValueError):
        module.validate_speed_density_mix(bad_mix)


def test_training_and_reset_event_wire_the_joint_field():
    trainer = (
        _REPO
        / "scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py"
    ).read_text(encoding="utf-8")
    event = (
        _REPO
        / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
        / "config/charge_skrl/mdp/events/long_corridor_replay.py"
    ).read_text(encoding="utf-8")
    assert '"long_corridor_speed_density_mix"' in trainer
    assert '"speed_density_mix": speed_density_mix' in event
    assert "speed_density_mix=speed_density_mix" in event
    assert "obstacle_count_mix and speed_density_mix are mutually exclusive" in event


def test_runtime_audit_reports_joint_profiles_and_fails_on_pairing_drift():
    event = (
        _REPO
        / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
        / "config/charge_skrl/mdp/events/long_corridor_replay.py"
    ).read_text(encoding="utf-8")
    assert "_long_corridor_speed_density_profile_total += torch.bincount" in event
    assert 'f" speed_density_profiles={profile_realized}"' in event
    assert 'f" profile_range_mismatch={range_mismatch}"' in event
    assert "joint speed-density profile/range pairing drifted" in event
    ensure_state = re.search(
        r"def _ensure_state\(env\).*?\ndef ", event, re.S
    )
    assert ensure_state is not None
    assert "speed_density_mix" not in ensure_state.group(0)
    assert "expected_profiles = len(speed_density_mix)" in event
