"""Locked 50-iteration SA6-v3 pilot from the human-accepted SA5 c600."""

from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path

from rnn_car_modular.configs.e2e_sa5_v3_c50_stage3_b3_cont500_from_it100 import (
    CONFIG as _SA5_C600_SOURCE,
)
from rnn_car_modular.configs.sim2real_speed_density_curriculum_v3 import P060
from rnn_car_modular.configs.sim2real_stage_curriculum_v2 import (
    STAGE_SCENE_CURRICULUM,
)


REPO = Path(__file__).resolve().parents[5]
RUN_NAME = "sa6_v3_from_sa5_c600_ne1024_s42_p50_r1"

PARENT_CHECKPOINT = (
    REPO
    / "logs/rnn_car/"
    "sa5_v3_c50_stage3_b3_cont500_from_it100_ne1024_s42_p500_r1/"
    "checkpoint_64000.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "c1a24684eb787a0865b007a2a712b744f0f74c39f812c62b9b18d4ed893c271a"
)
PARENT_ACCEPTANCE_RECORD = (
    REPO / "docs/freeze/sa5_c600_human_acceptance_20260826.json"
)
PARENT_ACCEPTANCE_RECORD_SHA256 = (
    "769e81726c0ac98ec3c3030ac98a6910658384aa7b217178098d221ac44dfd19"
)
AUTHORIZATION_RECORD = (
    REPO / "docs/freeze/sa6_v3_from_sa5_c600_p50_authorization_20260826.json"
)
AUTHORIZATION_RECORD_SHA256 = (
    "f5fa40b59a884db02c47209ba97e483af69ad4c542b7739cc6db9d3fa5cc088e"
)
SA5_SOURCE_CONFIG = (
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "e2e_sa5_v3_c50_stage3_b3_cont500_from_it100.py"
)
SA5_SOURCE_CONFIG_SHA256 = (
    "cfb5c2cd7827696def1ff93f463a7f8defa6cff2644e918b24c2a90c86cee022"
)
STAGE_CURRICULUM = (
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "sim2real_stage_curriculum_v2.py"
)
STAGE_CURRICULUM_SHA256 = (
    "f594920f61361db99b943cb82d51d2763bc4670c244d26a4cc82041f03a3f7da"
)
TRAINER = REPO / "scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py"
TRAINER_SHA256 = (
    "4209725a07954631a2490cedfd925633e5adb95aec8373c4a2160516b3349090"
)
CORRIDOR_REPLAY = (
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/"
    "velocity/config/charge_skrl/mdp/events/long_corridor_replay.py"
)
CORRIDOR_REPLAY_SHA256 = (
    "9e6e1f6f922daede1c58343f93e12b279ff3798f55449287ca13eb99094c0067"
)
CORRIDOR_GEOMETRY = (
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/"
    "velocity/config/charge_skrl/mdp/events/long_corridor_replay_geometry.py"
)
CORRIDOR_GEOMETRY_SHA256 = (
    "f57f4351573af93cd3e42d60554ba9d77956a49ed5614a9df5d2b40b646466a4"
)

TRAINING_ITERATIONS = 50
ROLLOUT_LENGTH = 128
SAVE_INTERVAL = 25
CHECKPOINTS = (
    (25, "checkpoint_3200.pt"),
    (50, "checkpoint_6400.pt"),
)
TRAINING_STARTED = False

SA6_SPEC = STAGE_SCENE_CURRICULUM[6]
SA6_TARGET_SPEED_RANGE = SA6_SPEC.corridor_speed_range
SA6_SPEED_DENSITY_MIX = (
    ((0, 1), P060, 0.10),
    ((1, 1), P060, 0.10),
    ((2, 1), P060, 0.15),
    ((3, 2), SA6_TARGET_SPEED_RANGE, 0.20),
    ((4, 2), SA6_TARGET_SPEED_RANGE, 0.30),
    ((4, 3), SA6_TARGET_SPEED_RANGE, 0.15),
)
SA6_TAGS = (
    "e2e",
    "frame_stack8",
    "ppo",
    "clean_progress",
    "anti_spin_a",
    "future_occupancy",
    "obb_collision",
    "physical_footprint",
    "action_history_2step",
    "sim2real_new_lineage",
    "vlp16_measured_full",
    "sim2real_curriculum_v2",
    "sim2real_speed_density_v3",
    "vehicle_speed_rate_0p7_from_sa3",
    "joint_speed_density_profiles",
    "valid_return_only",
    "actdelay_u1_2",
    "sa6_v3",
    "from_human_accepted_sa5_c600",
    "human_waiver_preserved",
    "resume_c600_ppo_optimizer",
    "scene_reset_78_12_10",
    "low_density_retention_35pct",
    "sa6_target_65pct",
    "p100_excluded",
    "pilot50_ready_not_run",
)

METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
EXPECTED_BEHAVIORAL_DIFFS = frozenset(
    {
        "checkpoint",
        "initial_stage",
        "room_size",
        "narrow_passage_segment_length",
        "narrow_passage_fixed_width_range",
        "long_corridor_free_width",
        "long_corridor_dynamic_speed_range",
        "long_corridor_speed_density_mix",
        "timesteps",
        "save_interval",
    }
)
RETAINED_BEHAVIOR_FIELDS = frozenset(
    {
        "algorithm",
        "reward_profile",
        "reward_mode",
        "encoder_profile",
        "critic_profile",
        "charge_encoder_mode",
        "lidar_frame_stack",
        "end_to_end_frame_stack",
        "use_action_history",
        "speed_rate",
        "speed_rate_obs",
        "lidar_no_noise",
        "vlp16_noise_mode",
        "lidar_distractor_eligibility",
        "enable_actuator_dr",
        "actuator_delay_range",
        "actuator_velocity_scale",
        "actuator_motor_lag",
        "actuator_motor_lag_by_channel",
        "obs_delay_steps",
        "long_corridor_dynamic_motion_mode",
        "long_corridor_dynamic_motion_weights",
        "long_corridor_random_2d_kinematics",
        "hidden_dim",
        "preprocess_dim",
        "fc_dim",
        "wd_middle_dim",
        "rnn_type",
    }
)


def _verify_sha256(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"{label} hash mismatch: expected {expected}, got {actual}"
        )


for _path, _expected, _label in (
    (PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256, "SA5 c600 parent"),
    (
        PARENT_ACCEPTANCE_RECORD,
        PARENT_ACCEPTANCE_RECORD_SHA256,
        "SA5 c600 human acceptance",
    ),
    (AUTHORIZATION_RECORD, AUTHORIZATION_RECORD_SHA256, "SA6 pilot authorization"),
    (SA5_SOURCE_CONFIG, SA5_SOURCE_CONFIG_SHA256, "SA5 c600 source config"),
    (STAGE_CURRICULUM, STAGE_CURRICULUM_SHA256, "stage curriculum"),
    (TRAINER, TRAINER_SHA256, "trainer"),
    (CORRIDOR_REPLAY, CORRIDOR_REPLAY_SHA256, "corridor replay"),
    (CORRIDOR_GEOMETRY, CORRIDOR_GEOMETRY_SHA256, "corridor geometry"),
):
    _verify_sha256(_path, _expected, _label)

_acceptance = json.loads(PARENT_ACCEPTANCE_RECORD.read_text(encoding="utf-8"))
_accepted_checkpoint = _acceptance.get("checkpoint") or {}
_machine_screen = _acceptance.get("machine_screen") or {}
if (
    _acceptance.get("schema") != "sa5_checkpoint_human_acceptance/v1"
    or _acceptance.get("status") != "HUMAN_ACCEPTED_SA5_PARENT_C600_WITH_WAIVER"
    or _acceptance.get("accepted_role")
    != "SA5_PARENT_FOR_SUBSEQUENT_DEVELOPMENT"
    or (REPO / str(_accepted_checkpoint.get("path"))).resolve()
    != PARENT_CHECKPOINT.resolve()
    or _accepted_checkpoint.get("sha256") != PARENT_CHECKPOINT_SHA256
    or _machine_screen.get("candidate_pass") is not False
    or _machine_screen.get("machine_verdict_preserved") is not True
):
    raise RuntimeError("SA5 acceptance record does not identify the waived c600 parent")

_authorization = json.loads(AUTHORIZATION_RECORD.read_text(encoding="utf-8"))
_authorized_parent = _authorization.get("parent") or {}
_authorized_source = _authorization.get("source_contract") or {}
_authorized_retained = _authorization.get("retained_behavior_contract") or {}
_authorized_scene = _authorization.get("sa6_scene_contract") or {}
_authorized_pilot = _authorization.get("pilot") or {}
_automatic = _authorization.get("automatic_actions") or {}
if (
    _authorization.get("schema") != "sa6_v3_c600_pilot_authorization/v1"
    or _authorization.get("decision")
    != "AUTHORIZE_BUILD_SA6_V3_C600_P50_PILOT_CONFIG"
    or (REPO / str(_authorized_parent.get("checkpoint"))).resolve()
    != PARENT_CHECKPOINT.resolve()
    or _authorized_parent.get("sha256") != PARENT_CHECKPOINT_SHA256
    or _authorized_parent.get("resume_ppo_optimizer") is not True
    or _authorized_parent.get("rl_optimizer_state_entries") != 38
    or _authorized_source.get("sa5_source_config_sha256")
    != SA5_SOURCE_CONFIG_SHA256
    or _authorized_source.get("stage_curriculum_sha256")
    != STAGE_CURRICULUM_SHA256
    or _authorized_source.get("trainer_sha256") != TRAINER_SHA256
    or _authorized_source.get("corridor_replay_sha256")
    != CORRIDOR_REPLAY_SHA256
    or _authorized_source.get("corridor_geometry_sha256")
    != CORRIDOR_GEOMETRY_SHA256
    or _authorized_retained.get("vlp16_noise_mode") != "full"
    or _authorized_retained.get("lidar_distractor_eligibility")
    != "valid_return_only"
    or _authorized_retained.get("enable_actuator_dr") is not True
    or _authorized_retained.get("actuator_delay_range") != [1, 2]
    or _authorized_retained.get("speed_rate") != 0.7
    or _authorized_scene.get("speed_density_mix")
    != [[list(counts), list(speed), weight] for counts, speed, weight in SA6_SPEED_DENSITY_MIX]
    or _authorized_pilot.get("iterations") != TRAINING_ITERATIONS
    or _authorized_pilot.get("timesteps") != TRAINING_ITERATIONS * ROLLOUT_LENGTH
    or _authorized_pilot.get("checkpoints") != [list(row) for row in CHECKPOINTS]
    or _automatic.get("config_build_authorized") is not True
    or _automatic.get("training_launch_authorized") is not False
    or _automatic.get("training_auto_start") is not False
):
    raise RuntimeError("authorization does not permit this exact SA6 pilot config")

CONFIG = replace(
    _SA5_C600_SOURCE,
    name="e2e_sa6_v3_from_sa5_c600_p50",
    description=(
        "Fifty-iteration SA6-v3 pilot from human-accepted SA5 c600 with exact "
        "K8 sim-to-real retention and a low-density-preserving SA6 scene mix."
    ),
    checkpoint=str(PARENT_CHECKPOINT),
    no_resume_optimizer=False,
    initial_stage=6,
    room_size=SA6_SPEC.room_half_extent,
    narrow_passage_segment_length=SA6_SPEC.room_half_extent,
    narrow_passage_fixed_width_range=SA6_SPEC.narrow_width_range,
    long_corridor_free_width=SA6_SPEC.corridor_free_width,
    long_corridor_dynamic_speed_range=(
        min(speed[0] for _, speed, _ in SA6_SPEED_DENSITY_MIX),
        max(speed[1] for _, speed, _ in SA6_SPEED_DENSITY_MIX),
    ),
    long_corridor_speed_density_mix=SA6_SPEED_DENSITY_MIX,
    timesteps=TRAINING_ITERATIONS * ROLLOUT_LENGTH,
    save_interval=SAVE_INTERVAL,
    tags=SA6_TAGS,
    notes=(
        f"SA6 pilot authorization {AUTHORIZATION_RECORD_SHA256} preserves the "
        "c600 K8, reward, network, "
        "speed_rate=0.7, full VLP-16 valid-return-only noise and U{1,2} "
        "decoded-command delay contracts. The c600 PPO optimizer is resumed "
        "because action, observation, reward and network are unchanged. Scene "
        "difficulty moves to SA6 while retaining 35% 0S1D/1S1D/2S1D replay. "
        "The parent remains a human-waived development checkpoint, not a machine "
        "SA5 Gate pass. Importing this config never launches training."
    ),
)

_actual_behavioral_diffs = frozenset(
    field.name
    for field in fields(CONFIG)
    if field.name not in METADATA_FIELDS
    and getattr(CONFIG, field.name) != getattr(_SA5_C600_SOURCE, field.name)
)
if _actual_behavioral_diffs != EXPECTED_BEHAVIORAL_DIFFS:
    raise RuntimeError(
        "SA6 pilot drifted from c600 source: expected "
        f"{sorted(EXPECTED_BEHAVIORAL_DIFFS)}, got "
        f"{sorted(_actual_behavioral_diffs)}"
    )

for _field in RETAINED_BEHAVIOR_FIELDS:
    if getattr(CONFIG, _field) != getattr(_SA5_C600_SOURCE, _field):
        raise RuntimeError(f"SA6 pilot drifted retained behavior field: {_field}")

assert CONFIG.initial_stage == 6 and CONFIG.fixed_stage is True
assert CONFIG.checkpoint == str(PARENT_CHECKPOINT)
assert CONFIG.no_resume_optimizer is False
assert CONFIG.num_envs == 1024 and CONFIG.seed == 42
assert CONFIG.rollout_length == ROLLOUT_LENGTH
assert CONFIG.timesteps == 6_400 and CONFIG.save_interval == SAVE_INTERVAL
assert CONFIG.lidar_frame_stack == 8 and CONFIG.end_to_end_frame_stack is True
assert CONFIG.lidar_no_noise is False and CONFIG.vlp16_noise_mode == "full"
assert CONFIG.lidar_distractor_eligibility == "valid_return_only"
assert CONFIG.enable_actuator_dr is True
assert CONFIG.actuator_delay_range == (1, 2)
assert CONFIG.actuator_velocity_scale == (1.0, 1.0)
assert CONFIG.actuator_motor_lag == 1.0
assert CONFIG.speed_rate == 0.7 and CONFIG.speed_rate_obs == "ego"
assert abs(sum(weight for _, _, weight in SA6_SPEED_DENSITY_MIX) - 1.0) < 1e-12
assert CONFIG.tags == SA6_TAGS
assert "sa6_hold" not in CONFIG.tags and "sa4" not in CONFIG.tags
