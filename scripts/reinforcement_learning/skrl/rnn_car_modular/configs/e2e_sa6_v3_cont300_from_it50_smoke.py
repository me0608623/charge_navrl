"""One-iteration GPU smoke for the exact SA6 it50 continuation."""

from __future__ import annotations

from dataclasses import fields, replace

from rnn_car_modular.configs.e2e_sa6_v3_cont300_from_it50 import CONFIG as _FORMAL


METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
EXPECTED_DIFFS = frozenset({"num_envs", "timesteps", "save_interval"})

CONFIG = replace(
    _FORMAL,
    name="e2e_sa6_v3_cont300_from_it50_smoke",
    description="64-env one-iteration GPU smoke for the SA6 it50 continuation.",
    num_envs=64,
    timesteps=128,
    save_interval=1,
    tags=_FORMAL.tags + ("gpu_smoke_only",),
    notes=f"{_FORMAL.notes} Smoke only: 64 envs and one iteration.",
)

_actual_diffs = frozenset(
    field.name
    for field in fields(CONFIG)
    if field.name not in METADATA_FIELDS
    and getattr(CONFIG, field.name) != getattr(_FORMAL, field.name)
)
if _actual_diffs != EXPECTED_DIFFS:
    raise RuntimeError(f"SA6 continuation smoke drifted: {sorted(_actual_diffs)}")

assert CONFIG.checkpoint == _FORMAL.checkpoint
assert CONFIG.no_resume_optimizer is False
assert CONFIG.initial_stage == 6
assert CONFIG.num_envs == 64
assert CONFIG.timesteps == 128
assert CONFIG.save_interval == 1

