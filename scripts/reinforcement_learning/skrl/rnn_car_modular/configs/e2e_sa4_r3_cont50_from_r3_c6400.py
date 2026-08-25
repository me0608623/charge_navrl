"""Exact 50-iteration continuation of SA4-R3 from its iteration-50 checkpoint.

This run answers one narrow question: does more optimization under the same
SA4-R3 recipe consolidate the partially learned corridor behavior?  It changes
no sensor noise, actuator delay, reward, scene mix, network, or PPO setting.
The R3 optimizer state is restored so this is a continuation, not a new
warm-start experiment.
"""

from dataclasses import fields, replace
import hashlib
from pathlib import Path

from rnn_car_modular.configs.e2e_sa4_r3_valid_return_noise_from_sa3_r1_c100 import (
    CONFIG as _R3,
)


PARENT_CHECKPOINT = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa4_r3_validreturn_from_sa3r1_c100_ne1024_s42_p50_r1/"
    "checkpoint_6400.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "fa51b5a312d38b7aa1ee93c3464b087b9b9cb985a7137eefdc8fecf40fed166d"
)
PARENT_TOTAL_ITERATION = 50
CONTINUATION_ITERATIONS = 50
ROLLOUT_LENGTH = 128
SAVE_INTERVAL = 25
CONCEPTUAL_CHECKPOINTS = (
    (75, "checkpoint_3200.pt"),
    (100, "checkpoint_6400.pt"),
)

METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
CONTINUATION_FIELDS = frozenset({"checkpoint", "no_resume_optimizer"})
ALLOWED_DIFFS = METADATA_FIELDS | CONTINUATION_FIELDS

_parent_path = Path(PARENT_CHECKPOINT)
if not _parent_path.is_file():
    raise FileNotFoundError(f"SA4-R3 c6400 parent is missing: {PARENT_CHECKPOINT}")
with _parent_path.open("rb") as _checkpoint_file:
    _parent_sha256 = hashlib.file_digest(_checkpoint_file, "sha256").hexdigest()
if _parent_sha256 != PARENT_CHECKPOINT_SHA256:
    raise RuntimeError(
        "SA4-R3 c6400 parent hash mismatch: "
        f"expected {PARENT_CHECKPOINT_SHA256}, got {_parent_sha256}"
    )


CONFIG = replace(
    _R3,
    name="e2e_sa4_r3_cont50_from_r3_c6400",
    description=(
        "Exact SA4-R3 continuation from c6400 for 50 more iterations, "
        "restoring the R3 optimizer and saving conceptual it75/it100."
    ),
    checkpoint=PARENT_CHECKPOINT,
    no_resume_optimizer=False,
    tags=_R3.tags + ("same_recipe_continuation", "resume_r3_optimizer", "it50_to_it100"),
    notes=(
        f"{_R3.notes} Human-authorized bounded continuation from R3 c6400. "
        "Restore the checkpoint optimizer state; preserve valid-return-only "
        "VLP-16 full noise, actuator delay U{0,1,2}, reward, curriculum, "
        "network, PPO, 50-iteration budget, and 25-iteration save interval. "
        "The new run's checkpoint_3200/checkpoint_6400 are conceptual total "
        "iterations 75/100. Do not auto-launch SA5."
    ),
)


assert CONFIG.initial_stage == 4
assert CONFIG.fixed_stage is True
assert CONFIG.num_envs == 1024
assert CONFIG.seed == 42
assert CONFIG.rollout_length == ROLLOUT_LENGTH
assert CONFIG.timesteps == CONTINUATION_ITERATIONS * ROLLOUT_LENGTH == 6_400
assert CONFIG.save_interval == SAVE_INTERVAL == 25
assert CONFIG.checkpoint == PARENT_CHECKPOINT
assert CONFIG.no_resume_optimizer is False
assert CONFIG.lidar_frame_stack == 8
assert CONFIG.lidar_no_noise is False
assert CONFIG.vlp16_noise_mode == "full"
assert CONFIG.lidar_distractor_eligibility == "valid_return_only"
assert CONFIG.actuator_delay_range == (0, 2)
assert CONFIG.future_occupancy_weight == 0.15
assert CONFIG.long_corridor_dynamic_motion_weights == (0.5, 0.5, 0.0)

_drifted = sorted(
    field.name
    for field in fields(CONFIG)
    if field.name not in ALLOWED_DIFFS
    and getattr(CONFIG, field.name) != getattr(_R3, field.name)
)
if _drifted:
    raise RuntimeError(
        "SA4-R3 continuation must preserve the R3 recipe; unexpected field "
        f"changes: {_drifted}"
    )
