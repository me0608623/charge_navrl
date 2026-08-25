"""Exact 25-iteration continuation of SA4-R3 from conceptual iteration 100.

This is the direct continuation of the 83D/K8 SA4-R3 lineage. It restores the
RL optimizer from the conceptual iteration-100 checkpoint and changes only the
additional training budget and lineage metadata. It does not enable corridor
teacher distillation or adopt the rejected R=10 teacher denominator candidate.
"""

from dataclasses import fields, replace
import hashlib
from pathlib import Path

from rnn_car_modular.configs.e2e_sa4_r3_cont50_from_r3_c6400 import (
    CONFIG as _IT100_RECIPE,
)


PARENT_CHECKPOINT = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa4_r3_cont50_from_r3c6400_ne1024_s42_p50_r1/"
    "checkpoint_6400.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "aeddf32b3e41d1920bfc839651f18e09d7384a8f6ab3a5ac6de3c11628520d7d"
)
PARENT_CONCEPTUAL_ITERATION = 100
CONTINUATION_ITERATIONS = 25
ROLLOUT_LENGTH = 128
SAVE_INTERVAL = 25
CONCEPTUAL_CHECKPOINTS = ((125, "checkpoint_3200.pt"),)

METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
ALLOWED_DIFFS = METADATA_FIELDS | frozenset({"checkpoint", "timesteps"})

_parent_path = Path(PARENT_CHECKPOINT)
if not _parent_path.is_file():
    raise FileNotFoundError(f"SA4-R3 it100 parent is missing: {PARENT_CHECKPOINT}")
with _parent_path.open("rb") as _checkpoint_file:
    _parent_sha256 = hashlib.file_digest(_checkpoint_file, "sha256").hexdigest()
if _parent_sha256 != PARENT_CHECKPOINT_SHA256:
    raise RuntimeError(
        "SA4-R3 it100 parent hash mismatch: "
        f"expected {PARENT_CHECKPOINT_SHA256}, got {_parent_sha256}"
    )


CONFIG = replace(
    _IT100_RECIPE,
    name="e2e_sa4_r3_cont25_from_it100",
    description=(
        "Exact SA4-R3 continuation from conceptual it100 for 25 more "
        "iterations, restoring the optimizer and saving conceptual it125."
    ),
    checkpoint=PARENT_CHECKPOINT,
    timesteps=CONTINUATION_ITERATIONS * ROLLOUT_LENGTH,
    tags=_IT100_RECIPE.tags
    + ("it100_to_it125", "same_recipe_continuation", "resume_r3_optimizer"),
    notes=(
        f"{_IT100_RECIPE.notes} Continue the same 83D/K8 SA4-R3 lineage from "
        "conceptual it100 for exactly 25 iterations. Restore optimizer state; "
        "preserve stage, valid-return-only full VLP-16 noise, actuator delay, "
        "reward, scene mix, network, and PPO settings. Corridor teacher "
        "distillation remains disabled and the R=10 goal denominator candidate "
        "is not adopted. Save conceptual it125 and stop; do not launch SA5."
    ),
)


assert CONFIG.initial_stage == 4
assert CONFIG.fixed_stage is True
assert CONFIG.num_envs == 1024
assert CONFIG.seed == 42
assert CONFIG.rollout_length == ROLLOUT_LENGTH
assert CONFIG.timesteps == CONTINUATION_ITERATIONS * ROLLOUT_LENGTH == 3_200
assert CONFIG.save_interval == SAVE_INTERVAL
assert CONFIG.checkpoint == PARENT_CHECKPOINT
assert CONFIG.no_resume_optimizer is False
assert CONFIG.lidar_frame_stack == 8
assert CONFIG.lidar_no_noise is False
assert CONFIG.vlp16_noise_mode == "full"
assert CONFIG.lidar_distractor_eligibility == "valid_return_only"
assert CONFIG.actuator_delay_range == (0, 2)
assert CONFIG.corridor_teacher_distill_epochs == 0
assert CONFIG.corridor_teacher_goal_denominator_floor_m == 1.0

_drifted = sorted(
    field.name
    for field in fields(CONFIG)
    if field.name not in ALLOWED_DIFFS
    and getattr(CONFIG, field.name) != getattr(_IT100_RECIPE, field.name)
)
if _drifted:
    raise RuntimeError(
        "SA4-R3 it100 continuation must preserve the prior recipe; unexpected "
        f"field changes: {_drifted}"
    )

