"""One-iteration GPU smoke for the stage-3 B3 P060-aligned arm."""

from __future__ import annotations

from dataclasses import fields, replace

from rnn_car_modular.configs.e2e_sa5_v3_c50_stage3_b3_p060aligned_p25 import (
    CONFIG as _B3,
)


METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
EXPECTED_B3_DIFFS = frozenset({"num_envs", "timesteps", "save_interval"})

CONFIG = replace(
    _B3,
    name="e2e_sa5_v3_c50_stage3_b3_p060aligned_smoke",
    description="64-env one-iteration GPU smoke for paired stage-3 B3.",
    num_envs=64,
    timesteps=128,
    save_interval=1,
    tags=_B3.tags + ("gpu_smoke_only",),
    notes=f"{_B3.notes} Smoke only: 64 env, one iteration.",
)

_actual_diffs = frozenset(
    field.name
    for field in fields(CONFIG)
    if field.name not in METADATA_FIELDS
    and getattr(CONFIG, field.name) != getattr(_B3, field.name)
)
if _actual_diffs != EXPECTED_B3_DIFFS:
    raise RuntimeError(
        "B3 smoke drifted: expected "
        f"{sorted(EXPECTED_B3_DIFFS)}, got {sorted(_actual_diffs)}"
    )

assert CONFIG.num_envs == 64
assert CONFIG.timesteps == 128
assert CONFIG.save_interval == 1
assert CONFIG.checkpoint == _B3.checkpoint
assert CONFIG.no_resume_optimizer is False
assert CONFIG.long_corridor_speed_density_mix == _B3.long_corridor_speed_density_mix
