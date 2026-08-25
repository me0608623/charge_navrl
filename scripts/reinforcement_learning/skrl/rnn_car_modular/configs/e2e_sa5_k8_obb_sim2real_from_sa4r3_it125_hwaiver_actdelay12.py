"""SA5 replacement run with the real-robot delay center corrected.

The first SA5 launch used the inherited U{0,1,2} delay contract and was stopped
at iteration 5 when new vehicle measurements arrived. Reanalysis of five clean
moving windows rejected the aggregate 0.28-0.44 angular gain estimate: aligned
steady gains were 0.983-1.046, while the response delay was 350-400 ms.

This config therefore preserves unity velocity scale and changes only actuator
delay from U{0,1,2} (mean 200 ms) to U{1,2} (mean 300 ms). It is a bounded
discrete approximation to measured p10/p50/p90 = 0/300/400 ms, not an exact
probability fit.
"""

from dataclasses import fields, replace

from rnn_car_modular.configs.e2e_sa5_k8_obb_sim2real_from_sa4r3_it125_hwaiver import (
    CHECKPOINTS,
    CONFIG as _SA5_IDENTITY_DELAY,
    PARENT_CHECKPOINT,
    PARENT_CHECKPOINT_SHA256,
    PARENT_CONCEPTUAL_ITERATION,
    ROLLOUT_LENGTH,
    SAVE_INTERVAL,
    TRAINING_ITERATIONS,
)


REAL_ROBOT_DELAY_QUANTILES_MS = (0, 300, 400)
CLEAN_WINDOW_STATIC_GAINS = (1.042, 0.983, 1.024, 1.013, 0.997)
CLEAN_WINDOW_BEST_LAGS_MS = (400, 400, 400, 400, 400)
ACTUATOR_DELAY_RANGE = (1, 2)

METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
ALLOWED_DIFFS = METADATA_FIELDS | frozenset({"actuator_delay_range"})
EXPECTED_CHANGED_FIELDS = frozenset({"actuator_delay_range"})

CONFIG = replace(
    _SA5_IDENTITY_DELAY,
    name="e2e_sa5_k8_obb_sim2real_from_sa4r3_it125_hwaiver_actdelay12",
    description=(
        "SA5 300-iteration sim-to-real v2 replacement run from SA4-R3 it125, "
        "with measured-delay-center correction U{1,2}."
    ),
    actuator_delay_range=ACTUATOR_DELAY_RANGE,
    tags=_SA5_IDENTITY_DELAY.tags
    + (
        "actdelay_u1_2",
        "real_delay_center_300ms",
        "unity_angular_gain_retained",
        "replacement_after_it5_contract_audit",
    ),
    notes=(
        f"{_SA5_IDENTITY_DELAY.notes} The initial identity-delay launch was "
        "stopped at iteration 5 after an aggregate vehicle report suggested "
        "angular tracking of 0.28-0.44. Clean moving-window alignment rejected "
        "that value as a steady-state scale: five static gains were "
        "0.983-1.046 (median about 1.01), with best lag 350-400 ms. Preserve "
        "actuator_velocity_scale=(1.0,1.0) and motor_lag alpha=1.0. Change only "
        "delay U{0,1,2} to U{1,2}, matching a 300 ms mean while retaining a "
        "400 ms upper support. This is a bounded approximation to measured "
        "p10/p50/p90=0/300/400 ms, not an exact empirical distribution."
    ),
)


assert _SA5_IDENTITY_DELAY.actuator_delay_range == (0, 2)
assert CONFIG.actuator_delay_range == ACTUATOR_DELAY_RANGE
assert CONFIG.actuator_velocity_scale == (1.0, 1.0)
assert CONFIG.actuator_motor_lag == 1.0
assert CONFIG.actuator_motor_lag_by_channel is None
assert CONFIG.checkpoint == PARENT_CHECKPOINT
assert CONFIG.no_resume_optimizer is True
assert CONFIG.initial_stage == 5
assert CONFIG.fixed_stage is True
assert CONFIG.num_envs == 1024
assert CONFIG.seed == 42
assert CONFIG.rollout_length == ROLLOUT_LENGTH
assert CONFIG.timesteps == TRAINING_ITERATIONS * ROLLOUT_LENGTH == 38_400
assert CONFIG.save_interval == SAVE_INTERVAL
assert CONFIG.lidar_frame_stack == 8
assert CONFIG.use_action_history is True
assert CONFIG.vlp16_noise_mode == "full"
assert CONFIG.lidar_distractor_eligibility == "valid_return_only"

_unexpected = sorted(
    field.name
    for field in fields(CONFIG)
    if field.name not in ALLOWED_DIFFS
    and getattr(CONFIG, field.name) != getattr(_SA5_IDENTITY_DELAY, field.name)
)
if _unexpected:
    raise RuntimeError(
        "SA5 delay correction may change only actuator_delay_range and metadata; "
        f"unexpected drift: {_unexpected}"
    )

_actual_changed = frozenset(
    field.name
    for field in fields(CONFIG)
    if field.name not in METADATA_FIELDS
    and getattr(CONFIG, field.name) != getattr(_SA5_IDENTITY_DELAY, field.name)
)
if _actual_changed != EXPECTED_CHANGED_FIELDS:
    raise RuntimeError(
        "SA5 delay correction field set changed: "
        f"expected {sorted(EXPECTED_CHANGED_FIELDS)}, "
        f"got {sorted(_actual_changed)}"
    )
