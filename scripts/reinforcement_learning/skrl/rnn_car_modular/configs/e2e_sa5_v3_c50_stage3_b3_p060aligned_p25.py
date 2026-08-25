"""Stage-3 B3 arm: align 0S1D and 1S1D replay with P060."""

from __future__ import annotations

from dataclasses import fields, replace

from rnn_car_modular.configs.e2e_sa5_v3_c50_stage3_a3_control_p25 import (
    AUTHORIZATION_RECORD_SHA256,
    CONFIG as _A3,
    PARENT_CHECKPOINT,
    TRAINING_ITERATIONS,
)
from rnn_car_modular.configs.sim2real_speed_density_curriculum_v3 import (
    P035,
    P060,
)


RUN_NAME = "sa5_v3_c50_stage3_b3_p060aligned_ne1024_s42_p25_r1"
P060_ALIGNED_MIX = (
    ((0, 1), P060, 0.10),
    ((1, 1), P060, 0.10),
    ((2, 1), P060, 0.15),
    ((3, 2), P035, 0.30),
    ((4, 2), P035, 0.35),
)
METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
EXPECTED_A3_DIFFS = frozenset({"long_corridor_speed_density_mix"})

CONFIG = replace(
    _A3,
    name="e2e_sa5_v3_c50_stage3_b3_p060aligned_p25",
    description=(
        "Stage-3 B3 matched P060-aligned replay diagnostic from the rejected "
        "B2 anchor."
    ),
    long_corridor_speed_density_mix=P060_ALIGNED_MIX,
    tags=tuple(tag for tag in _A3.tags if tag != "paired_stage3_a3_control")
    + (
        "paired_stage3_b3_p060_aligned",
        "only_0s1d_1s1d_speed_range_changed",
    ),
    notes=(
        f"{_A3.notes} Paired B3 under authorization "
        f"{AUTHORIZATION_RECORD_SHA256}; the only behavioral difference from "
        "A3 replaces 0S1D and 1S1D P080 speed ranges with P060 while retaining "
        "all profile probabilities and every other field."
    ),
)

_actual_a3_diffs = frozenset(
    field.name
    for field in fields(CONFIG)
    if field.name not in METADATA_FIELDS
    and getattr(CONFIG, field.name) != getattr(_A3, field.name)
)
if _actual_a3_diffs != EXPECTED_A3_DIFFS:
    raise RuntimeError(
        "paired B3 drifted from A3: expected "
        f"{sorted(EXPECTED_A3_DIFFS)}, got {sorted(_actual_a3_diffs)}"
    )

_control_mix = _A3.long_corridor_speed_density_mix
assert _control_mix[0][0] == (0, 1) and _control_mix[0][1] == (0.7, 0.9)
assert _control_mix[1][0] == (1, 1) and _control_mix[1][1] == (0.7, 0.9)
assert P060_ALIGNED_MIX[0][0] == (0, 1) and P060_ALIGNED_MIX[0][1] == P060
assert P060_ALIGNED_MIX[1][0] == (1, 1) and P060_ALIGNED_MIX[1][1] == P060
assert tuple(row[2] for row in P060_ALIGNED_MIX) == tuple(
    row[2] for row in _control_mix
)
assert P060_ALIGNED_MIX[2:] == _control_mix[2:]
assert abs(sum(row[2] for row in P060_ALIGNED_MIX) - 1.0) < 1e-12
assert CONFIG.checkpoint == PARENT_CHECKPOINT
assert CONFIG.no_resume_optimizer is False
assert CONFIG.timesteps == TRAINING_ITERATIONS * CONFIG.rollout_length
assert CONFIG.long_corridor_dynamic_motion_weights == (0.30, 0.30, 0.40)
