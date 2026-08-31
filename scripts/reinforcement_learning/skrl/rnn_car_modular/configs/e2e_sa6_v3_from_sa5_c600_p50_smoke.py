"""One-iteration GPU smoke config for the locked c600-based SA6-v3 pilot."""

from __future__ import annotations

from dataclasses import fields, replace

from rnn_car_modular.configs.e2e_sa6_v3_from_sa5_c600_p50 import (
    CONFIG as _PILOT,
)


METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
EXPECTED_DIFFS = frozenset({"num_envs", "timesteps", "save_interval"})

CONFIG = replace(
    _PILOT,
    name="e2e_sa6_v3_from_sa5_c600_p50_smoke",
    description="64-env one-iteration smoke for the c600-based SA6-v3 pilot.",
    num_envs=64,
    timesteps=128,
    save_interval=1,
    tags=_PILOT.tags + ("gpu_smoke_only",),
    notes=f"{_PILOT.notes} Smoke only: 64 envs, one iteration.",
)

_actual_diffs = frozenset(
    field.name
    for field in fields(CONFIG)
    if field.name not in METADATA_FIELDS
    and getattr(CONFIG, field.name) != getattr(_PILOT, field.name)
)
if _actual_diffs != EXPECTED_DIFFS:
    raise RuntimeError(f"SA6 pilot smoke drifted: {sorted(_actual_diffs)}")

assert CONFIG.checkpoint == _PILOT.checkpoint
assert CONFIG.no_resume_optimizer is False
assert CONFIG.initial_stage == 6
assert CONFIG.num_envs == 64
assert CONFIG.timesteps == 128
assert CONFIG.save_interval == 1
