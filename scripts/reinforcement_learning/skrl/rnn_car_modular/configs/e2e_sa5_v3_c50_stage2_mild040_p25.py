"""Stage-2 paired B2 arm: mild 30/30/40 motion-family allocation."""

from __future__ import annotations

from dataclasses import fields, replace

from rnn_car_modular.configs.e2e_sa5_v3_c50_stage2_equal_p25 import (
    AUTHORIZATION_RECORD_SHA256,
    CONFIG as _A2,
    PARENT_CHECKPOINT,
    TRAINING_ITERATIONS,
)


RUN_NAME = "sa5_v3_c50_stage2_mild040_ne1024_s42_p25_r1"
MOTION_WEIGHTS = (0.30, 0.30, 0.40)
METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
EXPECTED_A2_DIFFS = frozenset({"long_corridor_dynamic_motion_weights"})

CONFIG = replace(
    _A2,
    name="e2e_sa5_v3_c50_stage2_mild040_p25",
    description=(
        "Paired stage-2 mild 30/30/40 motion-allocation continuation from "
        "the matched-control it25 checkpoint."
    ),
    long_corridor_dynamic_motion_weights=MOTION_WEIGHTS,
    tags=tuple(tag for tag in _A2.tags if tag != "paired_stage2_a2_equal")
    + (
        "paired_stage2_b2_mild_030_030_040",
        "matched_development_candidate_not_parent",
    ),
    notes=(
        f"{_A2.notes} Paired B2 under authorization "
        f"{AUTHORIZATION_RECORD_SHA256}; the only behavioral difference from "
        "A2 is long_corridor_dynamic_motion_weights=(0.30, 0.30, 0.40)."
    ),
)

_actual_a2_diffs = frozenset(
    field.name
    for field in fields(CONFIG)
    if field.name not in METADATA_FIELDS
    and getattr(CONFIG, field.name) != getattr(_A2, field.name)
)
if _actual_a2_diffs != EXPECTED_A2_DIFFS:
    raise RuntimeError(
        "paired B2 drifted from A2: expected "
        f"{sorted(EXPECTED_A2_DIFFS)}, got {sorted(_actual_a2_diffs)}"
    )

assert CONFIG.checkpoint == PARENT_CHECKPOINT
assert CONFIG.no_resume_optimizer is False
assert CONFIG.timesteps == TRAINING_ITERATIONS * CONFIG.rollout_length
assert CONFIG.long_corridor_dynamic_motion_weights == MOTION_WEIGHTS
assert abs(sum(MOTION_WEIGHTS) - 1.0) < 1e-12

