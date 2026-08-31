"""Fresh matched control for the c250 SA6 profile-curriculum pilot."""

from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path

from rnn_car_modular.configs.e2e_sa6_v3_cont300_from_it50 import (
    CONFIG as _C250_SOURCE,
)


REPO = Path(__file__).resolve().parents[5]
RUN_NAME = "sa6_v4_c250_profile_curriculum_control_ne1024_s42_p50_r1"
PARENT_CHECKPOINT = (
    REPO
    / "logs/rnn_car/sa6_v3_cont300_from_it50_ne1024_s42_p300_r1/"
    "checkpoint_25600.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "4044d6dff443b3d4a2a36dfa067a99acd0e60b167919caa8f9b89cf213de0803"
)
SOURCE_CONFIG = (
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "e2e_sa6_v3_cont300_from_it50.py"
)
SOURCE_CONFIG_SHA256 = (
    "5b5a8c03bc66a4968fbfbf54e464f3397d7255e81fdad6b7e6886cc52233dc4f"
)
DESIGN_RECORD = (
    REPO
    / "docs/freeze/"
    "sa6_v4_c250_profile_curriculum_pilot_design_20260829.json"
)
DESIGN_RECORD_SHA256 = (
    "c72061f1efa865838eb0582ff87fe1de0ae74f2b8f5a31546df4d89337aeb548"
)

TRAINING_ITERATIONS = 50
ROLLOUT_LENGTH = 128
SAVE_INTERVAL = 25
CHECKPOINTS = (
    (25, "checkpoint_3200.pt"),
    (50, "checkpoint_6400.pt"),
)
TRAINING_STARTED = False
METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
EXPECTED_SOURCE_DIFFS = frozenset({"checkpoint", "timesteps", "save_interval"})
CONTROL_MIX = _C250_SOURCE.long_corridor_speed_density_mix


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
    (PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256, "SA6 c250 diagnostic anchor"),
    (SOURCE_CONFIG, SOURCE_CONFIG_SHA256, "SA6 c250 source config"),
    (DESIGN_RECORD, DESIGN_RECORD_SHA256, "SA6 profile-curriculum design"),
):
    _verify_sha256(_path, _expected, _label)

_design = json.loads(DESIGN_RECORD.read_text(encoding="utf-8"))
_parent = _design.get("parent") or {}
_shared = _design.get("shared_contract") or {}
_automatic = _design.get("automatic_actions") or {}
if (
    _design.get("schema")
    != "sa6_v4_c250_profile_curriculum_pilot_design/v1"
    or _design.get("decision") != "ESTABLISH_READY_NOT_RUN"
    or (REPO / str(_parent.get("checkpoint"))).resolve()
    != PARENT_CHECKPOINT.resolve()
    or _parent.get("sha256") != PARENT_CHECKPOINT_SHA256
    or _parent.get("conceptual_iteration") != 250
    or _parent.get("checkpoint_total_steps") != 25_600
    or _parent.get("checkpoint_iteration_zero_based") != 199
    or _parent.get("rl_optimizer_state_entries") != 38
    or _parent.get("resume_rl_optimizer") is not True
    or _shared.get("training_iterations") != TRAINING_ITERATIONS
    or _shared.get("save_interval_iterations") != SAVE_INTERVAL
    or _shared.get("checkpoints") != [list(row) for row in CHECKPOINTS]
    or _shared.get("only_allowed_behavioral_difference_between_arms")
    != "long_corridor_speed_density_mix"
    or _automatic.get("config_build_authorized") is not True
    or _automatic.get("cpu_tests_authorized") is not True
    or _automatic.get("gpu_smoke_authorized") is not False
    or _automatic.get("training_launch_authorized") is not False
    or _automatic.get("auto_extend_authorized") is not False
    or _automatic.get("start_sa7_authorized") is not False
    or _automatic.get("enable_teacher_distillation_or_override") is not False
):
    raise RuntimeError("design record does not permit this READY_NOT_RUN control")

CONFIG = replace(
    _C250_SOURCE,
    name="e2e_sa6_v4_c250_profile_curriculum_control_p50",
    description=(
        "Fresh 50-iteration c250 continuation used as the matched control for "
        "the SA6 profile-curriculum pilot; READY_NOT_RUN."
    ),
    checkpoint=str(PARENT_CHECKPOINT),
    timesteps=TRAINING_ITERATIONS * ROLLOUT_LENGTH,
    save_interval=SAVE_INTERVAL,
    tags=tuple(
        tag
        for tag in _C250_SOURCE.tags
        if tag
        not in {
            "exact_optimizer_continuation_it50_to_it350",
            "user_authorized_300_additional_iterations",
            "fixed_comparison_required",
        }
    )
    + (
        "sa6_not_graduated",
        "diagnostic_anchor_c250",
        "profile_curriculum_matched_control",
        "ready_not_run",
    ),
    notes=(
        "Fresh process-matched control from the non-graduated c250 diagnostic "
        f"anchor. Design {DESIGN_RECORD_SHA256} resumes the exact PPO optimizer "
        "and forbids GPU launch, automatic extension, teacher distillation, "
        "action override, and SA7."
    ),
)

_actual_source_diffs = frozenset(
    field.name
    for field in fields(CONFIG)
    if field.name not in METADATA_FIELDS
    and getattr(CONFIG, field.name) != getattr(_C250_SOURCE, field.name)
)
if _actual_source_diffs != EXPECTED_SOURCE_DIFFS:
    raise RuntimeError(
        "SA6 c250 control drifted: expected "
        f"{sorted(EXPECTED_SOURCE_DIFFS)}, got {sorted(_actual_source_diffs)}"
    )

assert CONFIG.checkpoint == str(PARENT_CHECKPOINT)
assert CONFIG.no_resume_optimizer is False
assert CONFIG.initial_stage == 6 and CONFIG.fixed_stage is True
assert CONFIG.num_envs == 1024 and CONFIG.seed == 42
assert CONFIG.rollout_length == ROLLOUT_LENGTH
assert CONFIG.timesteps == 6_400 and CONFIG.save_interval == SAVE_INTERVAL
assert CONFIG.long_corridor_speed_density_mix == CONTROL_MIX
assert CONFIG.lidar_frame_stack == 8 and CONFIG.end_to_end_frame_stack is True
assert CONFIG.lidar_no_noise is False and CONFIG.vlp16_noise_mode == "full"
assert CONFIG.lidar_distractor_eligibility == "valid_return_only"
assert CONFIG.enable_actuator_dr is True
assert CONFIG.actuator_delay_range == (1, 2)
assert CONFIG.speed_rate == 0.7 and CONFIG.speed_rate_obs == "ego"
assert CONFIG.long_corridor_dynamic_motion_mode == "env_stratified"
assert CONFIG.long_corridor_dynamic_motion_weights == (0.3, 0.3, 0.4)
assert CONFIG.long_corridor_random_2d_kinematics == "patrol"
assert CONFIG.teacher_retention_checkpoint is None
assert CONFIG.teacher_retention_rollout_override is False
assert CONFIG.corridor_teacher_distill_epochs == 0

