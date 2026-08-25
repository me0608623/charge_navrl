"""SA4-R3 short pilot with valid-return-only mixed-pixel eligibility.

The D9 closed-loop evaluation showed lower lateral collision rate on all three
pre-fixed evaluator seeds (515, 616, 818), while each wall-collision increase
remained within the pre-registered +0.5 percentage-point tolerance. That result
authorizes only this 50-iteration pilot; it does not authorize SA5.

R3 starts from the same clean SA3 c100 parent as R2. Relative to R2, the only
behavioral difference is ``lidar_distractor_eligibility`` changing from the
historical ``all_rays`` rule to ``valid_return_only``. Reward, optimizer,
curriculum, scene mix, actuator DR, budget, and checkpoint schedule are fixed.
"""

from dataclasses import fields, replace

from rnn_car_modular.configs.e2e_sa4_r2_futureocc015_from_sa3_r1_c100 import (
    CHECKPOINT_STEPS,
    CONFIG as _SA4_R2,
    INTERVENTION_ITERATIONS,
    PARENT_CHECKPOINT,
    PARENT_CHECKPOINT_SHA256,
    ROLLOUT_LENGTH,
    SAVE_INTERVAL,
)


ELIGIBILITY_FIELD = "lidar_distractor_eligibility"
HISTORICAL_ELIGIBILITY = "all_rays"
R3_ELIGIBILITY = "valid_return_only"
METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
ALLOWED_DIFFS = METADATA_FIELDS | {ELIGIBILITY_FIELD}

if _SA4_R2.lidar_distractor_eligibility != HISTORICAL_ELIGIBILITY:
    raise RuntimeError(
        "SA4-R2 no longer carries historical all_rays eligibility; R3's sole "
        "behavioral difference must be re-established"
    )


CONFIG = replace(
    _SA4_R2,
    name="e2e_sa4_r3_valid_return_noise_from_sa3_r1_c100",
    description=(
        "SA4-R3 50-iteration pilot from clean SA3 c100 with valid-return-only "
        "mixed-pixel eligibility; otherwise identical to SA4-R2."
    ),
    lidar_distractor_eligibility=R3_ELIGIBILITY,
    tags=_SA4_R2.tags
    + ("sa4_r3", "valid_return_only", "d9_3of3_authorized", "pilot50"),
    notes=(
        f"{_SA4_R2.notes} D9 evaluator-seed replication qualified 3/3 fixed "
        "seeds under the pre-registered CR and wall-CR rule. This run changes "
        "only mixed-pixel eligibility from all_rays to valid_return_only. It "
        "starts directly from the clean SA3 c100 parent, saves it25/it50, and "
        "cannot auto-extend or launch SA5."
    ),
)


assert CONFIG.initial_stage == 4
assert CONFIG.fixed_stage is True
assert CONFIG.num_envs == 1024
assert CONFIG.seed == 42
assert CONFIG.rollout_length == ROLLOUT_LENGTH == 128
assert CONFIG.timesteps == INTERVENTION_ITERATIONS * ROLLOUT_LENGTH == 6_400
assert CONFIG.save_interval == SAVE_INTERVAL == 25
assert CHECKPOINT_STEPS == (3200, 6400)
assert CONFIG.checkpoint == PARENT_CHECKPOINT
assert CONFIG.no_resume_optimizer is True
assert CONFIG.lidar_frame_stack == 8
assert CONFIG.actuator_delay_range == (0, 2)
assert CONFIG.lidar_no_noise is False
assert CONFIG.vlp16_noise_mode == "full"
assert CONFIG.lidar_distractor_eligibility == R3_ELIGIBILITY

_drifted = sorted(
    field.name
    for field in fields(CONFIG)
    if field.name not in ALLOWED_DIFFS
    and getattr(CONFIG, field.name) != getattr(_SA4_R2, field.name)
)
if _drifted:
    raise RuntimeError(
        "SA4-R3 must differ from SA4-R2 only in metadata and "
        f"{ELIGIBILITY_FIELD}; these fields also changed: {_drifted}"
    )

