"""SA5 sim-to-real v2 run from the conditionally accepted SA4-R3 it125.

The frozen SA4 screen remained a machine FAIL because the lateral cell missed
SR/CR by 1.0465 percentage points. The user explicitly accepted that bounded
miss as a human waiver. This module records that provenance without rewriting
the screen verdict.

SA4 to SA5 is a stage transition: model weights are loaded, optimizer state is
reset, and only the canonical SA5 scene fields plus run budget/metadata change.
The accepted SA4-R3 sensor, actuator, reward, model and PPO recipe is retained.
"""

from dataclasses import fields, replace
import hashlib
from pathlib import Path

from rnn_car_modular.configs.e2e_sa4_r3_cont25_from_it100 import (
    CONFIG as _SA4_IT125_RECIPE,
)
from rnn_car_modular.configs.sim2real_stage_curriculum_v2 import (
    STAGE_SCENE_CURRICULUM,
    make_sim2real_curriculum_config,
)


PARENT_CHECKPOINT = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa4_r3_cont25_from_it100_ne1024_s42_p25_r1/checkpoint_3200.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "57f43d07255971ad607e0bf3234dd7d46984c9c71cc9550e2e9461fbc95ab71d"
)
PARENT_CONCEPTUAL_ITERATION = 125
TRAINING_ITERATIONS = 300
ROLLOUT_LENGTH = 128
SAVE_INTERVAL = 50
CHECKPOINTS = tuple(
    (iteration, f"checkpoint_{iteration * ROLLOUT_LENGTH}.pt")
    for iteration in range(SAVE_INTERVAL, TRAINING_ITERATIONS + 1, SAVE_INTERVAL)
)

METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
STAGE_SCENE_FIELDS = frozenset(
    {
        "initial_stage",
        "room_size",
        "narrow_passage_fraction",
        "narrow_passage_segment_length",
        "narrow_passage_fixed_width_range",
        "narrow_passage_fixed_yaw_limit_deg",
        "narrow_passage_exact_width",
        "narrow_passage_exact_width_ratio",
        "long_corridor_fraction",
        "long_corridor_free_width",
        "long_corridor_length",
        "long_corridor_static_obstacles",
        "long_corridor_dynamic_obstacles",
        "long_corridor_obstacle_count_mix",
        "long_corridor_dynamic_speed_range",
        "long_corridor_dynamic_motion_mode",
        "long_corridor_dynamic_motion_weights",
        "long_corridor_random_2d_kinematics",
    }
)
RUN_FIELDS = frozenset(
    {"checkpoint", "no_resume_optimizer", "timesteps", "save_interval"}
)
ALLOWED_DIFFS = METADATA_FIELDS | STAGE_SCENE_FIELDS | RUN_FIELDS

# These are the fields whose values actually differ from the SA4 it125 recipe.
# Stage fields that happen to retain the same value are still checked against
# the canonical SA5 builder below.
EXPECTED_CHANGED_FIELDS = frozenset(
    {
        "checkpoint",
        "initial_stage",
        "long_corridor_dynamic_motion_weights",
        "long_corridor_dynamic_speed_range",
        "long_corridor_fraction",
        "long_corridor_free_width",
        "long_corridor_obstacle_count_mix",
        "long_corridor_static_obstacles",
        "narrow_passage_fixed_width_range",
        "narrow_passage_fraction",
        "narrow_passage_segment_length",
        "no_resume_optimizer",
        "room_size",
        "save_interval",
        "timesteps",
    }
)

_parent_path = Path(PARENT_CHECKPOINT)
if not _parent_path.is_file():
    raise FileNotFoundError(f"SA4-R3 it125 parent is missing: {PARENT_CHECKPOINT}")
with _parent_path.open("rb") as _checkpoint_file:
    _parent_sha256 = hashlib.file_digest(_checkpoint_file, "sha256").hexdigest()
if _parent_sha256 != PARENT_CHECKPOINT_SHA256:
    raise RuntimeError(
        "SA4-R3 it125 parent hash mismatch: "
        f"expected {PARENT_CHECKPOINT_SHA256}, got {_parent_sha256}"
    )

_SA5_CANONICAL = make_sim2real_curriculum_config(
    5, checkpoint=PARENT_CHECKPOINT
)

CONFIG = replace(
    _SA4_IT125_RECIPE,
    name="e2e_sa5_k8_obb_sim2real_from_sa4r3_it125_hwaiver",
    description=(
        "SA5 300-iteration sim-to-real v2 run from the conditionally accepted "
        "SA4-R3 conceptual it125 checkpoint."
    ),
    initial_stage=_SA5_CANONICAL.initial_stage,
    room_size=_SA5_CANONICAL.room_size,
    narrow_passage_fraction=_SA5_CANONICAL.narrow_passage_fraction,
    narrow_passage_segment_length=_SA5_CANONICAL.narrow_passage_segment_length,
    narrow_passage_fixed_width_range=(
        _SA5_CANONICAL.narrow_passage_fixed_width_range
    ),
    narrow_passage_fixed_yaw_limit_deg=(
        _SA5_CANONICAL.narrow_passage_fixed_yaw_limit_deg
    ),
    narrow_passage_exact_width=_SA5_CANONICAL.narrow_passage_exact_width,
    narrow_passage_exact_width_ratio=(
        _SA5_CANONICAL.narrow_passage_exact_width_ratio
    ),
    long_corridor_fraction=_SA5_CANONICAL.long_corridor_fraction,
    long_corridor_free_width=_SA5_CANONICAL.long_corridor_free_width,
    long_corridor_length=_SA5_CANONICAL.long_corridor_length,
    long_corridor_static_obstacles=(
        _SA5_CANONICAL.long_corridor_static_obstacles
    ),
    long_corridor_dynamic_obstacles=(
        _SA5_CANONICAL.long_corridor_dynamic_obstacles
    ),
    long_corridor_obstacle_count_mix=(
        _SA5_CANONICAL.long_corridor_obstacle_count_mix
    ),
    long_corridor_dynamic_speed_range=(
        _SA5_CANONICAL.long_corridor_dynamic_speed_range
    ),
    long_corridor_dynamic_motion_mode=(
        _SA5_CANONICAL.long_corridor_dynamic_motion_mode
    ),
    long_corridor_dynamic_motion_weights=(
        _SA5_CANONICAL.long_corridor_dynamic_motion_weights
    ),
    long_corridor_random_2d_kinematics=(
        _SA5_CANONICAL.long_corridor_random_2d_kinematics
    ),
    checkpoint=PARENT_CHECKPOINT,
    no_resume_optimizer=True,
    timesteps=TRAINING_ITERATIONS * ROLLOUT_LENGTH,
    save_interval=SAVE_INTERVAL,
    tags=_SA5_CANONICAL.tags
    + (
        "from_sa4r3_it125",
        "human_waiver_parent",
        "valid_return_only",
        "future_occ_0p15_retained",
        "optimizer_reset_stage_transition",
        "p300",
    ),
    notes=(
        f"{_SA5_CANONICAL.notes} Parent is SA4-R3 conceptual it125, SHA-256 "
        f"{PARENT_CHECKPOINT_SHA256}. Its frozen screen remained an overall "
        "machine FAIL: longitudinal and native passed, while lateral missed "
        "SR/CR by 1.0465 percentage points and was accepted by explicit human "
        "waiver. Preserve the "
        "parent's valid-return-only full VLP-16 noise, future occupancy weight "
        "0.15, decoded-command delay U{0,1,2}, K8/83D observation, reward, "
        "model and PPO settings. Reset optimizer for the SA4-to-SA5 stage "
        "transition. Train exactly 300 iterations, save every 50, and do not "
        "auto-launch SA6."
    ),
)


assert CONFIG.initial_stage == 5
assert CONFIG.fixed_stage is True
assert CONFIG.num_envs == 1024
assert CONFIG.seed == 42
assert CONFIG.rollout_length == ROLLOUT_LENGTH
assert CONFIG.timesteps == TRAINING_ITERATIONS * ROLLOUT_LENGTH == 38_400
assert CONFIG.save_interval == SAVE_INTERVAL
assert CONFIG.checkpoint == PARENT_CHECKPOINT
assert CONFIG.no_resume_optimizer is True
assert CONFIG.lidar_frame_stack == 8
assert CONFIG.use_action_history is True
assert CONFIG.lidar_no_noise is False
assert CONFIG.vlp16_noise_mode == "full"
assert CONFIG.lidar_distractor_eligibility == "valid_return_only"
assert CONFIG.actuator_delay_range == (0, 2)
assert CONFIG.future_occupancy_weight == 0.15
assert CONFIG.corridor_teacher_distill_epochs == 0
assert CONFIG.corridor_teacher_goal_denominator_floor_m == 1.0

for _field_name in STAGE_SCENE_FIELDS:
    if getattr(CONFIG, _field_name) != getattr(_SA5_CANONICAL, _field_name):
        raise RuntimeError(
            f"SA5 scene field {_field_name} does not match the canonical builder"
        )

_unexpected = sorted(
    field.name
    for field in fields(CONFIG)
    if field.name not in ALLOWED_DIFFS
    and getattr(CONFIG, field.name) != getattr(_SA4_IT125_RECIPE, field.name)
)
if _unexpected:
    raise RuntimeError(
        "SA5 may change only canonical stage-scene, run and metadata fields; "
        f"unexpected drift: {_unexpected}"
    )

_actual_changed = frozenset(
    field.name
    for field in fields(CONFIG)
    if field.name not in METADATA_FIELDS
    and getattr(CONFIG, field.name) != getattr(_SA4_IT125_RECIPE, field.name)
)
if _actual_changed != EXPECTED_CHANGED_FIELDS:
    raise RuntimeError(
        "SA5 transition field set changed: "
        f"expected {sorted(EXPECTED_CHANGED_FIELDS)}, "
        f"got {sorted(_actual_changed)}"
    )
