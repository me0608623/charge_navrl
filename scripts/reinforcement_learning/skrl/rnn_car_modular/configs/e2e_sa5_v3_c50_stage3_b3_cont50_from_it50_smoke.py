"""One-iteration GPU smoke for the B3 it50 continuation."""

from __future__ import annotations

from dataclasses import fields, replace

from rnn_car_modular.configs.e2e_sa5_v3_c50_stage3_b3_cont50_from_it50 import (
    CONFIG as _CONT,
)


METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
EXPECTED_DIFFS = frozenset({"num_envs", "timesteps", "save_interval"})

CONFIG = replace(
    _CONT,
    name="e2e_sa5_v3_c50_stage3_b3_cont50_from_it50_smoke",
    description="64-env one-iteration smoke for the B3 it50 continuation.",
    num_envs=64,
    timesteps=128,
    save_interval=1,
    tags=_CONT.tags + ("gpu_smoke_only",),
    notes=f"{_CONT.notes} Smoke only: 64 envs, one iteration.",
)

_actual_diffs = frozenset(
    field.name
    for field in fields(CONFIG)
    if field.name not in METADATA_FIELDS
    and getattr(CONFIG, field.name) != getattr(_CONT, field.name)
)
if _actual_diffs != EXPECTED_DIFFS:
    raise RuntimeError(f"continuation smoke drifted: {sorted(_actual_diffs)}")

assert CONFIG.checkpoint == _CONT.checkpoint
assert CONFIG.no_resume_optimizer is False
assert CONFIG.num_envs == 64
assert CONFIG.timesteps == 128
assert CONFIG.save_interval == 1

