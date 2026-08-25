"""One-iteration GPU smoke for the exact B3 continuation."""

from __future__ import annotations

from dataclasses import fields, replace

from rnn_car_modular.configs.e2e_sa5_v3_c50_stage3_b3_cont25_from_it25 import (
    CONFIG as _CONT,
)


METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
EXPECTED_DIFFS = frozenset({"num_envs", "timesteps", "save_interval"})

CONFIG = replace(
    _CONT,
    name="e2e_sa5_v3_c50_stage3_b3_cont25_smoke",
    description="64-env one-iteration smoke for B3 it25 continuation.",
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
    raise RuntimeError(
        f"continuation smoke drifted: {sorted(_actual_diffs)}"
    )

assert CONFIG.checkpoint == _CONT.checkpoint
assert CONFIG.no_resume_optimizer is False
assert CONFIG.num_envs == 64
assert CONFIG.timesteps == 128
assert CONFIG.save_interval == 1
