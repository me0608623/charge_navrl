"""SA4-R2: the SA4 pilot recipe with one changed factor, 50 iterations.

The single behavioural difference
---------------------------------
``future_occupancy_weight`` 0.10 -> 0.15. Nothing else about the reward, the
scene, the optimizer or the sim2real mechanisms moves. The rest of the
``future_occupancy_*`` family (horizon 1.5 s, 8 samples, 1.0 m safe distance,
3.0 m near distance, 0.1 m/s move threshold) is untouched, so the shaping term's
*shape* is fixed and only its magnitude changes.

0.15 is a bounded probe, not a claim about the optimum. It is the smallest step
that is still large enough to move a 0.10-weighted term, chosen before seeing
any result so the value cannot be tuned to the answer.

Why it restarts from SA3 c100
-----------------------------
Both A/B arms begin at the same parent. Warm-starting the intervention from the
old SA4 c100 would confound the weight change with 100 extra iterations of
drift, and the difference could no longer be attributed to the intervention.

Drift lock
----------
``ALLOWED_DIFFS`` names every field permitted to differ from the SA4 pilot:
budget, metadata, and the one intervention. Anything else raises at import.
``test_sa4_r2_intervention_config.py`` additionally asserts that this config and
the D2 control differ **only** in metadata and ``future_occupancy_weight`` --
that is the actual A/B contract, and it is checked rather than asserted in prose.
"""

from dataclasses import fields, replace
import hashlib
from pathlib import Path

from rnn_car_modular.configs.e2e_sa4_k8_obb_sim2real_from_sa3_r1_c100 import (
    CONFIG as _SA4_PILOT,
    PARENT_CHECKPOINT,
    PARENT_CHECKPOINT_SHA256,
)


INTERVENTION_ITERATIONS = 50
ROLLOUT_LENGTH = 128
SAVE_INTERVAL = 25
CHECKPOINT_STEPS = (
    SAVE_INTERVAL * ROLLOUT_LENGTH,              # it25 -> checkpoint_3200.pt
    INTERVENTION_ITERATIONS * ROLLOUT_LENGTH,    # it50 -> checkpoint_6400.pt
)

#: The one behavioural change.
INTERVENTION_FIELD = "future_occupancy_weight"
BASELINE_FUTURE_OCCUPANCY_WEIGHT = 0.10
INTERVENTION_FUTURE_OCCUPANCY_WEIGHT = 0.15

#: Budget and metadata may differ; they carry no behaviour.
BUDGET_FIELDS = frozenset(
    {"name", "description", "timesteps", "save_interval", "tags", "notes"}
)
#: Everything this config is allowed to change relative to the SA4 pilot.
ALLOWED_DIFFS = BUDGET_FIELDS | {INTERVENTION_FIELD}

if _SA4_PILOT.future_occupancy_weight != BASELINE_FUTURE_OCCUPANCY_WEIGHT:
    raise RuntimeError(
        "the SA4 pilot no longer carries the baseline future-occupancy weight "
        f"({_SA4_PILOT.future_occupancy_weight} != "
        f"{BASELINE_FUTURE_OCCUPANCY_WEIGHT}); this probe's step size was chosen "
        "against the old value and must be re-derived"
    )

_parent_path = Path(PARENT_CHECKPOINT)
if not _parent_path.is_file():
    raise FileNotFoundError(
        f"SA3 R1 c100 parent checkpoint is missing: {PARENT_CHECKPOINT}"
    )
with _parent_path.open("rb") as _checkpoint_file:
    _parent_sha256 = hashlib.file_digest(_checkpoint_file, "sha256").hexdigest()
if _parent_sha256 != PARENT_CHECKPOINT_SHA256:
    raise RuntimeError(
        "SA3 R1 c100 parent checkpoint hash mismatch: "
        f"expected {PARENT_CHECKPOINT_SHA256}, got {_parent_sha256}"
    )


CONFIG = replace(
    _SA4_PILOT,
    name="e2e_sa4_r2_futureocc015_from_sa3_r1_c100",
    description=(
        "SA4-R2 intervention: SA4 pilot recipe with future_occupancy_weight "
        "0.10 -> 0.15, 50 iterations, per-family corridor accounting enabled."
    ),
    future_occupancy_weight=INTERVENTION_FUTURE_OCCUPANCY_WEIGHT,
    timesteps=INTERVENTION_ITERATIONS * ROLLOUT_LENGTH,
    save_interval=SAVE_INTERVAL,
    tags=_SA4_PILOT.tags
    + ("d2_intervention", "ab_treatment", "pilot50", "future_occ_0p15"),
    notes=(
        f"{_SA4_PILOT.notes} This is the B arm of a bounded A/B against "
        "e2e_sa4_d2_control_from_sa3_r1_c100. The only behavioural difference is "
        "future_occupancy_weight 0.10 -> 0.15; corridor family weights, "
        "optimizer and learning rate are deliberately untouched this round. "
        "0.15 is a conservative bounded probe and is not claimed to be optimal. "
        "The budget is exactly 50 iterations and does not auto-extend; the next "
        "stage is not started by this run."
    ),
)


# --- Config lock -----------------------------------------------------------
assert CONFIG.initial_stage == 4
assert CONFIG.fixed_stage is True
assert CONFIG.num_envs == 1024
assert CONFIG.seed == 42
assert CONFIG.rollout_length == ROLLOUT_LENGTH
assert CONFIG.timesteps == INTERVENTION_ITERATIONS * ROLLOUT_LENGTH == 6_400
assert CONFIG.save_interval == SAVE_INTERVAL == 25
assert CONFIG.checkpoint == PARENT_CHECKPOINT
assert CONFIG.no_resume_optimizer is True

# The intervention itself.
assert CONFIG.future_occupancy_weight == INTERVENTION_FUTURE_OCCUPANCY_WEIGHT

# The rest of the future-occupancy family defines the term's shape and must not
# move, or "one factor changed" would be false.
assert CONFIG.future_occupancy_horizon_s == _SA4_PILOT.future_occupancy_horizon_s
assert CONFIG.future_occupancy_samples == _SA4_PILOT.future_occupancy_samples
assert (
    CONFIG.future_occupancy_safe_distance_m
    == _SA4_PILOT.future_occupancy_safe_distance_m
)
assert (
    CONFIG.future_occupancy_near_distance_m
    == _SA4_PILOT.future_occupancy_near_distance_m
)
assert (
    CONFIG.future_occupancy_move_threshold_mps
    == _SA4_PILOT.future_occupancy_move_threshold_mps
)

# Sim2real mechanisms and the scene carried over untouched.
assert CONFIG.lidar_frame_stack == 8
assert CONFIG.actuator_delay_range == (0, 2)
assert CONFIG.lidar_no_noise is False
assert CONFIG.long_corridor_dynamic_motion_weights == (0.5, 0.5, 0.0)

# Field-level drift lock: budget, metadata and the intervention only.
_drifted = sorted(
    field.name
    for field in fields(CONFIG)
    if field.name not in ALLOWED_DIFFS
    and getattr(CONFIG, field.name) != getattr(_SA4_PILOT, field.name)
)
if _drifted:
    raise RuntimeError(
        "SA4-R2 must differ from the SA4 pilot only in budget, metadata and "
        f"{INTERVENTION_FIELD}; these fields also changed: {_drifted}"
    )
