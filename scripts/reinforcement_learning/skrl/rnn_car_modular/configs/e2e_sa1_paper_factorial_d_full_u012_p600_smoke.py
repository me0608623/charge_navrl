"""One-iteration GPU smoke for paper factorial cell D."""

from __future__ import annotations

from dataclasses import fields, replace

from rnn_car_modular.configs.e2e_sa1_paper_factorial_d_full_u012_p600 import (
    CONFIG as _FORMAL,
)


METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
EXPECTED_DIFFS = frozenset({"num_envs", "timesteps", "save_interval"})

CONFIG = replace(
    _FORMAL,
    name="e2e_sa1_paper_factorial_d_full_u012_p600_smoke",
    description="64-env one-iteration GPU smoke for paper factorial cell D.",
    num_envs=64,
    timesteps=128,
    save_interval=1,
    tags=_FORMAL.tags + ("gpu_smoke_only",),
    notes=f"{_FORMAL.notes} Runtime wiring smoke only; no capability claim.",
)

_actual_diffs = frozenset(
    field.name
    for field in fields(CONFIG)
    if field.name not in METADATA_FIELDS
    and getattr(CONFIG, field.name) != getattr(_FORMAL, field.name)
)
if _actual_diffs != EXPECTED_DIFFS:
    raise RuntimeError(f"factorial D smoke drifted: {sorted(_actual_diffs)}")

assert CONFIG.checkpoint is None
assert CONFIG.no_resume_optimizer is True
assert CONFIG.num_envs == 64
assert CONFIG.timesteps == 128
assert CONFIG.save_interval == 1
