"""SA4-D2 Control: the old SA4 recipe, 50 iterations, with per-family accounting.

Why a control run at all
------------------------
The completed SA4 pilot has no per-direction corridor accounting, so its
lateral/longitudinal split cannot be recovered after the fact. Any intervention
measured against it would be compared to a baseline that does not exist. This
run rebuilds that baseline under the metrics that were missing.

What is held fixed
------------------
Everything. This config is derived from
``e2e_sa4_k8_obb_sim2real_from_sa3_r1_c100`` with ``dataclasses.replace``, so
reward, scene mix, optimizer, sim2real mechanisms and the parent checkpoint are
the same objects, not a restatement that could drift. Only the budget changes:

* ``timesteps``      12,800 -> 6,400   (100 -> 50 iterations)
* ``save_interval``  50 -> 25          (checkpoints at 3,200 and 6,400 steps)

``test_sa4_d2_control_config.py`` asserts field-by-field that nothing else
differs, so "only the budget changed" is checked rather than claimed.

Restarting from SA3 c100 -- not from the old SA4 c100 -- is deliberate: the
intervention arm must start from the same point as this control, or the A/B
difference would confound the intervention with 100 extra iterations of drift.
"""

from dataclasses import fields, replace
import hashlib
from pathlib import Path

from rnn_car_modular.configs.e2e_sa4_k8_obb_sim2real_from_sa3_r1_c100 import (
    CONFIG as _SA4_PILOT,
    PARENT_CHECKPOINT,
    PARENT_CHECKPOINT_SHA256,
)


CONTROL_ITERATIONS = 50
ROLLOUT_LENGTH = 128
SAVE_INTERVAL = 25

#: Steps the two checkpoints land on, i.e. what the eval screen will name.
CHECKPOINT_STEPS = (
    SAVE_INTERVAL * ROLLOUT_LENGTH,          # it25 -> checkpoint_3200.pt
    CONTROL_ITERATIONS * ROLLOUT_LENGTH,     # it50 -> checkpoint_6400.pt
)

#: Fields this config is allowed to change relative to the SA4 pilot.
BUDGET_FIELDS = frozenset(
    {"name", "description", "timesteps", "save_interval", "tags", "notes"}
)

_parent_path = Path(PARENT_CHECKPOINT)
if not _parent_path.is_file():
    raise FileNotFoundError(f"SA3 R1 c100 parent checkpoint is missing: {PARENT_CHECKPOINT}")
with _parent_path.open("rb") as _checkpoint_file:
    _parent_sha256 = hashlib.file_digest(_checkpoint_file, "sha256").hexdigest()
if _parent_sha256 != PARENT_CHECKPOINT_SHA256:
    raise RuntimeError(
        "SA3 R1 c100 parent checkpoint hash mismatch: "
        f"expected {PARENT_CHECKPOINT_SHA256}, got {_parent_sha256}"
    )


CONFIG = replace(
    _SA4_PILOT,
    name="e2e_sa4_d2_control_from_sa3_r1_c100",
    description=(
        "SA4-D2 control: the SA4 pilot recipe unchanged, 50 iterations, run to "
        "produce the lateral/longitudinal baseline the original pilot could not "
        "record."
    ),
    timesteps=CONTROL_ITERATIONS * ROLLOUT_LENGTH,
    save_interval=SAVE_INTERVAL,
    tags=_SA4_PILOT.tags + ("d2_control", "ab_baseline", "pilot50"),
    notes=(
        f"{_SA4_PILOT.notes} This is the A arm of a bounded A/B. It changes no "
        "reward, no scene mix and no optimizer setting relative to the SA4 "
        "pilot; only the budget is shorter. The B arm must start from the same "
        "SA3 c100 parent with the same seed and budget so that a single "
        "intervention is the only difference. No intervention is chosen until "
        "this run's reward magnitudes have been read."
    ),
)


# Config lock.
assert CONFIG.initial_stage == 4
assert CONFIG.fixed_stage is True
assert CONFIG.num_envs == 1024
assert CONFIG.seed == 42
assert CONFIG.rollout_length == ROLLOUT_LENGTH
assert CONFIG.timesteps == CONTROL_ITERATIONS * ROLLOUT_LENGTH == 6_400
assert CONFIG.save_interval == SAVE_INTERVAL == 25
assert CONFIG.checkpoint == PARENT_CHECKPOINT
assert CONFIG.no_resume_optimizer is True

# Sim2real mechanisms carried over untouched.
assert CONFIG.lidar_frame_stack == 8
assert CONFIG.actuator_delay_range == (0, 2)
assert CONFIG.lidar_no_noise is False

# Only the budget may differ from the pilot. Checked here as well as in tests so
# an edit to either config fails at import rather than at analysis time.
_drifted = sorted(
    field.name
    for field in fields(CONFIG)
    if field.name not in BUDGET_FIELDS
    and getattr(CONFIG, field.name) != getattr(_SA4_PILOT, field.name)
)
if _drifted:
    raise RuntimeError(
        "SA4-D2 control must differ from the SA4 pilot only in its budget; "
        f"these fields also changed: {_drifted}"
    )
