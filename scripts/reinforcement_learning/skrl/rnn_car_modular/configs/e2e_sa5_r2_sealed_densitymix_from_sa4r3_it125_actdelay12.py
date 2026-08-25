"""SA5-R2 sealed correction with explicit low-density corridor coverage.

This launch candidate extends the geometry-only SA5-R2 recipe by one declared
training-distribution change: 35% of corridor resets contain exactly one
dynamic obstacle. The remaining 65% retain the previous 3S2D/4S2D moderate
and hard cases. It remains a warm-start correction from the original SA4-R3
conceptual-it125 parent with a reset optimizer.
"""

from dataclasses import fields, replace
import math

from rnn_car_modular.configs.e2e_sa5_r2_sealed_from_sa4r3_it125_actdelay12 import (
    BOUNDARY_OVERLAP_M,
    CHECKPOINTS,
    CONFIG as _SEALED_GEOMETRY_ONLY,
    INTERACTION_LENGTH_M,
    PARENT_CHECKPOINT,
    PARENT_CHECKPOINT_SHA256,
    PARENT_CONCEPTUAL_ITERATION,
    PHYSICAL_WALL_SPAN_M,
    ROLLOUT_LENGTH,
    SAVE_INTERVAL,
    SEALED_SOURCE_FILES,
    SEALED_SOURCE_SHA256,
    TRAINING_ITERATIONS,
)


RUN_NAME = (
    "sa5_r2_sealed_ldmix_from_sa4r3_it125_"
    "actd12_ne1024_s42_p300_r1"
)

CORRIDOR_DENSITY_MIX = (
    ((0, 1), 0.10),
    ((1, 1), 0.10),
    ((2, 1), 0.15),
    ((3, 2), 0.30),
    ((4, 2), 0.35),
)

METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
INTENDED_BEHAVIOR_FIELDS = frozenset({"long_corridor_obstacle_count_mix"})
ALLOWED_DIFFS = METADATA_FIELDS | INTENDED_BEHAVIOR_FIELDS

CONFIG = replace(
    _SEALED_GEOMETRY_ONLY,
    name="e2e_sa5_r2_sealed_densitymix_from_sa4r3_it125_actdelay12",
    description=(
        "SA5-R2 300-iteration correction from the original SA4-R3 it125 "
        "parent with sealed Stage-5 corridor geometry and explicit low-density "
        "corridor coverage."
    ),
    long_corridor_obstacle_count_mix=CORRIDOR_DENSITY_MIX,
    tags=_SEALED_GEOMETRY_ONLY.tags
    + (
        "low_density_corridor_35pct",
        "single_dynamic_corridor_35pct",
        "density_mix_preregistered",
    ),
    notes=(
        f"{_SEALED_GEOMETRY_ONLY.notes} User-requested deployment coverage: "
        "corridor reset density is frozen to 0S1D=10%, 1S1D=10%, 2S1D=15%, "
        "3S2D=30%, 4S2D=35%. Thus 35% of corridor resets have exactly one "
        "dynamic obstacle and 65% retain moderate/hard 2D cases. This is a "
        "declared second intervention in addition to sealed geometry, so "
        "training improvement cannot be attributed to geometry alone. "
        "Post-training validation must include both fixed 4S2D stress cells "
        "and low-density 0S1D/1S1D behavior checks for unnecessary stopping."
    ),
)


assert CONFIG.checkpoint == PARENT_CHECKPOINT
assert PARENT_CONCEPTUAL_ITERATION == 125
assert CONFIG.no_resume_optimizer is True
assert CONFIG.num_envs == 1024
assert CONFIG.seed == 42
assert CONFIG.initial_stage == 5
assert CONFIG.rollout_length == ROLLOUT_LENGTH == 128
assert CONFIG.timesteps == TRAINING_ITERATIONS * ROLLOUT_LENGTH == 38_400
assert CONFIG.save_interval == SAVE_INTERVAL == 50
assert CONFIG.long_corridor_fraction == 0.10
assert CONFIG.long_corridor_static_obstacles == 4
assert CONFIG.long_corridor_dynamic_obstacles == 2
assert CONFIG.long_corridor_obstacle_count_mix == CORRIDOR_DENSITY_MIX
assert math.isclose(
    sum(weight for _, weight in CORRIDOR_DENSITY_MIX),
    1.0,
    rel_tol=0.0,
    abs_tol=1.0e-12,
)
assert math.isclose(
    sum(
        weight
        for (_, dynamic), weight in CORRIDOR_DENSITY_MIX
        if dynamic == 1
    ),
    0.35,
    rel_tol=0.0,
    abs_tol=1.0e-12,
)
assert math.isclose(
    sum(
        weight
        for (_, dynamic), weight in CORRIDOR_DENSITY_MIX
        if dynamic == 2
    ),
    0.65,
    rel_tol=0.0,
    abs_tol=1.0e-12,
)
assert CONFIG.long_corridor_length == INTERACTION_LENGTH_M == 10.0
assert PHYSICAL_WALL_SPAN_M == 15.0
assert BOUNDARY_OVERLAP_M == 0.5
assert CONFIG.vlp16_noise_mode == "full"
assert CONFIG.lidar_distractor_eligibility == "valid_return_only"
assert CONFIG.actuator_delay_range == (1, 2)
assert CONFIG.actuator_velocity_scale == (1.0, 1.0)
assert CONFIG.actuator_motor_lag == 1.0
assert CHECKPOINTS == (
    (50, "checkpoint_6400.pt"),
    (100, "checkpoint_12800.pt"),
    (150, "checkpoint_19200.pt"),
    (200, "checkpoint_25600.pt"),
    (250, "checkpoint_32000.pt"),
    (300, "checkpoint_38400.pt"),
)
assert set(SEALED_SOURCE_FILES) == set(SEALED_SOURCE_SHA256)
assert len(PARENT_CHECKPOINT_SHA256) == 64

_drifted = {
    field.name
    for field in fields(CONFIG)
    if getattr(CONFIG, field.name) != getattr(_SEALED_GEOMETRY_ONLY, field.name)
}
if _drifted != ALLOWED_DIFFS:
    raise RuntimeError(
        "SA5-R2 sealed+densitymix must differ from geometry-only config only "
        f"in metadata and obstacle-count mix; got {sorted(_drifted)}"
    )
