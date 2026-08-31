"""Fail-closed contract tests for the c600-based SA6-v3 pilot."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from rnn_car_modular.configs import (  # noqa: E402
    e2e_sa1_k8_obb_sim2real_v1 as sa1,
)
from rnn_car_modular.configs import (  # noqa: E402
    e2e_sa2_k8_obb_sim2real_from_sa1_r1_c1700 as sa2,
)
from rnn_car_modular.configs import (  # noqa: E402
    e2e_sa3_k8_obb_speed_density_v3_from_sa2_r1_c100 as sa3,
)
from rnn_car_modular.configs import e2e_sa4_v3_cont300_from_c300 as sa4  # noqa: E402
from rnn_car_modular.configs import (  # noqa: E402
    e2e_sa5_v3_c50_stage3_b3_cont500_from_it100 as sa5,
)
from rnn_car_modular.configs import (  # noqa: E402
    e2e_sa6_v3_from_sa5_c600_p50 as pilot,
)
from rnn_car_modular.configs import (  # noqa: E402
    e2e_sa6_v3_from_sa5_c600_p50_smoke as smoke,
)
from rnn_car_modular.configs.sim2real_stage_curriculum_v2 import (  # noqa: E402
    STAGE_SCENE_CURRICULUM,
)
from rnn_car_modular.configs.registry import get_experiment_config  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_authorization_builds_config_but_does_not_launch() -> None:
    record = json.loads(pilot.AUTHORIZATION_RECORD.read_text(encoding="utf-8"))
    assert _sha256(pilot.AUTHORIZATION_RECORD) == pilot.AUTHORIZATION_RECORD_SHA256
    assert record["schema"] == "sa6_v3_c600_pilot_authorization/v1"
    assert record["decision"] == "AUTHORIZE_BUILD_SA6_V3_C600_P50_PILOT_CONFIG"
    assert record["automatic_actions"] == {
        "config_build_authorized": True,
        "gpu_smoke_auto_start": False,
        "training_launch_authorized": False,
        "training_auto_start": False,
    }
    assert pilot.TRAINING_STARTED is False


def test_parent_and_behavior_sources_are_sha_locked() -> None:
    for path, expected in (
        (pilot.PARENT_CHECKPOINT, pilot.PARENT_CHECKPOINT_SHA256),
        (pilot.SA5_SOURCE_CONFIG, pilot.SA5_SOURCE_CONFIG_SHA256),
        (pilot.STAGE_CURRICULUM, pilot.STAGE_CURRICULUM_SHA256),
        (pilot.TRAINER, pilot.TRAINER_SHA256),
        (pilot.CORRIDOR_REPLAY, pilot.CORRIDOR_REPLAY_SHA256),
        (pilot.CORRIDOR_GEOMETRY, pilot.CORRIDOR_GEOMETRY_SHA256),
        (pilot.PARENT_ACCEPTANCE_RECORD, pilot.PARENT_ACCEPTANCE_RECORD_SHA256),
    ):
        assert _sha256(path) == expected


def test_c600_contains_the_ppo_optimizer_that_will_be_resumed() -> None:
    payload = torch.load(
        pilot.PARENT_CHECKPOINT, map_location="cpu", weights_only=False
    )
    assert payload["total_steps"] == 64_000
    assert payload["iteration"] == 499
    assert len(payload["charge_opt_rl"]["state"]) == 38
    assert pilot.CONFIG.no_resume_optimizer is False


def test_sa6_pilot_budget_and_checkpoints_are_exact() -> None:
    assert pilot.CONFIG.initial_stage == 6
    assert pilot.CONFIG.fixed_stage is True
    assert pilot.CONFIG.num_envs == 1024
    assert pilot.CONFIG.seed == 42
    assert pilot.CONFIG.rollout_length == 128
    assert pilot.TRAINING_ITERATIONS == 50
    assert pilot.CONFIG.timesteps == 6_400
    assert pilot.CONFIG.save_interval == 25
    assert pilot.CHECKPOINTS == (
        (25, "checkpoint_3200.pt"),
        (50, "checkpoint_6400.pt"),
    )


def test_sa6_metadata_does_not_carry_stale_hold_or_continuation_tags() -> None:
    assert pilot.CONFIG.tags == pilot.SA6_TAGS
    assert "sa6_hold" not in pilot.CONFIG.tags
    assert "sa4" not in pilot.CONFIG.tags
    assert not any(
        tag.startswith("exact_optimizer_continuation") for tag in pilot.CONFIG.tags
    )
    assert "pilot50_ready_not_run" in pilot.CONFIG.tags


def test_sa6_geometry_comes_from_the_stage_six_spec() -> None:
    spec = STAGE_SCENE_CURRICULUM[6]
    assert pilot.CONFIG.room_size == spec.room_half_extent == 7.0
    assert pilot.CONFIG.narrow_passage_fraction == spec.narrow_fraction == 0.12
    assert pilot.CONFIG.narrow_passage_fixed_width_range == spec.narrow_width_range
    assert pilot.CONFIG.long_corridor_fraction == spec.corridor_fraction == 0.10
    assert pilot.CONFIG.long_corridor_free_width == spec.corridor_free_width == 4.0
    assert pilot.SA6_TARGET_SPEED_RANGE == spec.corridor_speed_range


def test_low_density_retention_is_preserved_while_sa6_target_is_added() -> None:
    mix = pilot.SA6_SPEED_DENSITY_MIX
    assert mix[:3] == sa5.CONFIG.long_corridor_speed_density_mix[:3]
    assert [row[0] for row in mix] == [
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 2),
        (4, 2),
        (4, 3),
    ]
    assert sum(row[2] for row in mix[:3]) == pytest.approx(0.35)
    assert sum(row[2] for row in mix[3:]) == pytest.approx(0.65)
    assert sum(row[2] for row in mix) == pytest.approx(1.0)
    assert all(row[1] == pilot.SA6_TARGET_SPEED_RANGE for row in mix[3:])
    assert pilot.CONFIG.long_corridor_speed_density_mix == mix


def test_k8_sensor_actuator_reward_and_network_contract_are_unchanged() -> None:
    for name in pilot.RETAINED_BEHAVIOR_FIELDS:
        assert getattr(pilot.CONFIG, name) == getattr(sa5.CONFIG, name)
    assert pilot.CONFIG.lidar_frame_stack == 8
    assert pilot.CONFIG.lidar_no_noise is False
    assert pilot.CONFIG.vlp16_noise_mode == "full"
    assert pilot.CONFIG.lidar_distractor_eligibility == "valid_return_only"
    assert pilot.CONFIG.enable_actuator_dr is True
    assert pilot.CONFIG.actuator_delay_range == (1, 2)
    assert pilot.CONFIG.actuator_velocity_scale == (1.0, 1.0)
    assert pilot.CONFIG.actuator_motor_lag == 1.0
    assert pilot.CONFIG.speed_rate == 0.7


@pytest.mark.parametrize(
    ("label", "config", "eligibility", "delay", "speed_rate"),
    (
        ("SA1", sa1.CONFIG, "all_rays", (0, 2), 1.0),
        ("SA2", sa2.CONFIG, "all_rays", (0, 2), 1.0),
        ("SA3", sa3.CONFIG, "valid_return_only", (1, 2), 0.7),
        ("SA4", sa4.CONFIG, "valid_return_only", (1, 2), 0.7),
        ("SA5", sa5.CONFIG, "valid_return_only", (1, 2), 0.7),
    ),
)
def test_current_sa1_sa5_mainline_enables_noise_and_actuator_delay_dr(
    label: str,
    config: object,
    eligibility: str,
    delay: tuple[int, int],
    speed_rate: float,
) -> None:
    del label
    assert config.lidar_frame_stack == 8
    assert config.lidar_no_noise is False
    assert config.vlp16_noise_mode == "full"
    assert config.lidar_distractor_eligibility == eligibility
    assert config.enable_actuator_dr is True
    assert config.actuator_delay_range == delay
    assert config.actuator_velocity_scale == (1.0, 1.0)
    assert config.actuator_motor_lag == 1.0
    assert config.speed_rate == speed_rate


def test_config_diff_from_c600_source_is_exactly_allowlisted() -> None:
    changed = {
        field.name
        for field in fields(pilot.CONFIG)
        if field.name not in pilot.METADATA_FIELDS
        and getattr(pilot.CONFIG, field.name) != getattr(sa5.CONFIG, field.name)
    }
    assert changed == pilot.EXPECTED_BEHAVIORAL_DIFFS


def test_smoke_changes_only_runtime_scale() -> None:
    changed = {
        field.name
        for field in fields(smoke.CONFIG)
        if field.name not in smoke.METADATA_FIELDS
        and getattr(smoke.CONFIG, field.name) != getattr(pilot.CONFIG, field.name)
    }
    assert changed == {"num_envs", "timesteps", "save_interval"}
    assert smoke.CONFIG.num_envs == 64
    assert smoke.CONFIG.timesteps == 128
    assert smoke.CONFIG.save_interval == 1


def test_config_import_has_no_launch_side_effect() -> None:
    source = Path(pilot.__file__).read_text(encoding="utf-8")
    for forbidden in ("systemctl", "systemd-run", "subprocess", "Popen("):
        assert forbidden not in source
    assert pilot.TRAINING_STARTED is False


def test_registry_resolves_both_pilot_configs() -> None:
    resolved = get_experiment_config("e2e_sa6_v3_from_sa5_c600_p50")
    resolved_smoke = get_experiment_config("e2e_sa6_v3_from_sa5_c600_p50_smoke")
    assert resolved.name == pilot.CONFIG.name
    assert resolved.checkpoint == str(pilot.PARENT_CHECKPOINT)
    assert resolved_smoke.name == smoke.CONFIG.name
    assert resolved_smoke.num_envs == 64
