"""P080 replay intervention for the READY_NOT_RUN c550 matched pilot."""

from __future__ import annotations

from dataclasses import fields, replace

from rnn_car_modular.configs.e2e_sa5_v3_c50_stage3_b3_c550_profilemix_control_p50 import (
    CONFIG as _CONTROL,
    DESIGN_RECORD_SHA256,
    METADATA_FIELDS,
    PARENT_CHECKPOINT,
    PARENT_CHECKPOINT_SHA256,
    ROLLOUT_LENGTH,
    SAVE_INTERVAL,
    TRAINING_ITERATIONS,
)
from rnn_car_modular.configs.sim2real_speed_density_curriculum_v3 import (
    P035,
    P060,
    P080,
)


RUN_NAME = "sa5_v3_c50_stage3_b3_c550_p080replay10_ne1024_s42_p50_r1"
P080_REPLAY_MIX = (
    ((0, 1), P060, 0.10),
    ((1, 1), P060, 0.10),
    ((0, 1), P080, 0.05),
    ((1, 1), P080, 0.05),
    ((2, 1), P060, 0.15),
    ((3, 2), P035, 0.25),
    ((4, 2), P035, 0.30),
)
EXPECTED_CONTROL_DIFFS = frozenset({"long_corridor_speed_density_mix"})

CONFIG = replace(
    _CONTROL,
    name="e2e_sa5_v3_c50_stage3_b3_c550_p080replay10_p50",
    description=(
        "Fresh c550 intervention adding 5 percent each of explicit P080 0S1D "
        "and 1S1D replay; READY_NOT_RUN."
    ),
    long_corridor_speed_density_mix=P080_REPLAY_MIX,
    tags=tuple(tag for tag in _CONTROL.tags if tag != "fresh_c550_profilemix_control")
    + (
        "fresh_c550_p080_replay10_intervention",
        "profile_mix_only_intervention",
    ),
    notes=(
        f"{_CONTROL.notes} Intervention under design {DESIGN_RECORD_SHA256}; "
        "P060 profile weights are unchanged, P080 0S1D/1S1D receive 5 percent "
        "each, and P035 3S2D/4S2D each donate 5 percent."
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
        "P080 replay intervention drifted: expected "
        f"{sorted(EXPECTED_CONTROL_DIFFS)}, got {sorted(_actual_control_diffs)}"
    )

_control_p060 = tuple(
    row for row in _CONTROL.long_corridor_speed_density_mix if row[1] == P060
)
_intervention_p060 = tuple(row for row in P080_REPLAY_MIX if row[1] == P060)
if _intervention_p060 != _control_p060:
    raise RuntimeError("intervention changed a P060 profile weight")

assert abs(sum(row[2] for row in P080_REPLAY_MIX) - 1.0) < 1e-12
assert tuple(row for row in P080_REPLAY_MIX if row[1] == P080) == (
    ((0, 1), P080, 0.05),
    ((1, 1), P080, 0.05),
)
assert CONFIG.checkpoint == str(PARENT_CHECKPOINT)
assert PARENT_CHECKPOINT_SHA256 == (
    "501c79199b9c0a73d556d5ac1c1345cde347675a213197414af1183c5a2be4ed"
)
assert CONFIG.no_resume_optimizer is False
assert CONFIG.timesteps == TRAINING_ITERATIONS * ROLLOUT_LENGTH
assert CONFIG.save_interval == SAVE_INTERVAL == 25
