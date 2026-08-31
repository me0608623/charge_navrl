"""Rung-1 profile-mix intervention for the c250 SA6 matched pilot."""

from __future__ import annotations

from dataclasses import fields, replace

from rnn_car_modular.configs.e2e_sa6_v4_c250_profile_curriculum_control_p50 import (
    CHECKPOINTS,
    CONFIG as _CONTROL,
    CONTROL_MIX,
    DESIGN_RECORD_SHA256,
    METADATA_FIELDS,
    PARENT_CHECKPOINT,
    PARENT_CHECKPOINT_SHA256,
    ROLLOUT_LENGTH,
    SAVE_INTERVAL,
    TRAINING_ITERATIONS,
)


RUN_NAME = "sa6_v4_c250_profile_curriculum_rung1_ne1024_s42_p50_r1"
CURRICULUM_RUNG1_MIX = (
    CONTROL_MIX[0],
    CONTROL_MIX[1],
    CONTROL_MIX[2],
    (CONTROL_MIX[3][0], CONTROL_MIX[3][1], 0.35),
    (CONTROL_MIX[4][0], CONTROL_MIX[4][1], 0.25),
    (CONTROL_MIX[5][0], CONTROL_MIX[5][1], 0.05),
)
EXPECTED_CONTROL_DIFFS = frozenset({"long_corridor_speed_density_mix"})
TRAINING_STARTED = False

CONFIG = replace(
    _CONTROL,
    name="e2e_sa6_v4_c250_profile_curriculum_rung1_p50",
    description=(
        "SA6 c250 matched intervention moving high-density exposure toward "
        "3S2D while preserving all low-density replay; READY_NOT_RUN."
    ),
    long_corridor_speed_density_mix=CURRICULUM_RUNG1_MIX,
    tags=tuple(
        tag for tag in _CONTROL.tags if tag != "profile_curriculum_matched_control"
    )
    + (
        "profile_curriculum_rung1",
        "profile_mix_only_intervention",
        "low_density_retention_35pct",
        "high_density_total_65pct",
    ),
    notes=(
        f"Matched intervention under design {DESIGN_RECORD_SHA256}. P060 "
        "0S1D/1S1D/2S1D remain 10/10/15 percent. SA6 3S2D/4S2D/4S3D "
        "change from 20/30/15 to 35/25/5 percent. No other behavioral field "
        "changes."
    ),
)

_actual_control_diffs = frozenset(
    field.name
    for field in fields(CONFIG)
    if field.name not in METADATA_FIELDS
    and getattr(CONFIG, field.name) != getattr(_CONTROL, field.name)
)
if _actual_control_diffs != EXPECTED_CONTROL_DIFFS:
    raise RuntimeError(
        "SA6 rung-1 intervention drifted: expected "
        f"{sorted(EXPECTED_CONTROL_DIFFS)}, got {sorted(_actual_control_diffs)}"
    )

for _index in range(3):
    if CURRICULUM_RUNG1_MIX[_index] != CONTROL_MIX[_index]:
        raise RuntimeError("rung-1 changed a P060 retention profile")
for _control_row, _rung_row in zip(CONTROL_MIX, CURRICULUM_RUNG1_MIX):
    if _control_row[:2] != _rung_row[:2]:
        raise RuntimeError("rung-1 changed profile identity or speed range")

assert abs(sum(row[2] for row in CURRICULUM_RUNG1_MIX) - 1.0) < 1e-12
assert sum(row[2] for row in CURRICULUM_RUNG1_MIX[:3]) == 0.35
assert sum(row[2] for row in CURRICULUM_RUNG1_MIX[3:]) == 0.65
assert tuple(row[2] for row in CURRICULUM_RUNG1_MIX[3:]) == (0.35, 0.25, 0.05)
assert CONFIG.checkpoint == str(PARENT_CHECKPOINT)
assert PARENT_CHECKPOINT_SHA256 == (
    "4044d6dff443b3d4a2a36dfa067a99acd0e60b167919caa8f9b89cf213de0803"
)
assert CONFIG.no_resume_optimizer is False
assert CONFIG.timesteps == TRAINING_ITERATIONS * ROLLOUT_LENGTH == 6_400
assert CONFIG.save_interval == SAVE_INTERVAL == 25
assert CHECKPOINTS == ((25, "checkpoint_3200.pt"), (50, "checkpoint_6400.pt"))

